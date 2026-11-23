"""The OpenRL Tinker HTTP gateway.

Provides `create_app(...)` with explicit dependency injection (`GatewayState`) and
end-to-end binary protobuf passthrough via `server.api.wire` (zero `numpy` imports).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import traceback
import uuid
from collections import defaultdict
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse, Response
from fastapi.utils import is_body_allowed_for_status_code
from opentelemetry import propagate, trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from pydantic import AfterValidator, AliasChoices, BaseModel, BeforeValidator, ConfigDict, Field, ValidationError, ValidationInfo, field_validator
from starlette.exceptions import HTTPException as StarletteHTTPException

from server import openrl_status
from server import telemetry as run_records
from server.api import wire
from server.lora_snapshots import snapshot_name, snapshot_root
from server.model_metadata import (
  TrainingModelMetadata,
  WeightSyncConfig,
  get_model_metadata,
  persist_model_metadata,
  pool_name,
)
from server.observability.router import router as observability_router
from server.sampling_dispatcher import SAMPLE_DEADLINE_SECONDS
from server.sampling_store import SamplingStore
from server.session_registry import SessionRegistry
from server.store import RequestStore, StateStore, get_state_store, get_store
from server.vllm_options import split_stop
from server.worker_manager import LocalWorkerManager, WorkerManager, create_worker_manager, default_metadata, owner_id, owner_of, runtime_of
from training import commands
from training.commands import Command
from training.types import TRAINER_BACKENDS, Datum, FFTConfig, FineTuningType, LoraConfig

DEFAULT_TMP_DIR = os.getenv("OPEN_RL_TMP_DIR", "/tmp/open-rl")
DEFAULT_OPEN_RL_VERSION = os.getenv("OPEN_RL_VERSION") or "dev"
SESSION_REAP_INTERVAL_SEC = 30
RECONCILE_CONCURRENCY = 4
SAVED_WEIGHTS_WAIT_SECONDS = 2.0
RELEASES_URL = "https://github.com/openrl-dev/openrl/releases"
EXAMPLE_MODELS = ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen3-0.6B", "google/gemma-4-e2b", "Qwen/Qwen3-8B"]
SETTINGS_PREFIX = "openrl."

USER_METADATA_DOC = (
  "Free-form client metadata. Keys that start with openrl. are OpenRL settings: "
  "see components.schemas.OpenRLSettings in this document. "
  "Values are strings on the wire."
)


# ***** requests *****


class SessionHeartbeatRequest(BaseModel):
  session_id: str | None = None


class CreateSessionRequest(BaseModel):
  tags: list[str] = Field(default_factory=list)
  user_metadata: dict[str, Any] | None = Field(default=None, description=USER_METADATA_DOC)


class CreateModelRequest(BaseModel):
  base_model: str
  session_id: str | None = None
  user_metadata: dict[str, Any] | None = Field(default=None, description=USER_METADATA_DOC)
  lora_config: LoraConfig = Field(default_factory=LoraConfig)
  # Removed; FFT options are openrl. settings now. Kept to reject it loudly.
  full_config: Any = None

  @field_validator("lora_config", mode="before")
  @classmethod
  def default_config(cls, value):
    return {} if value is None else value


class CreateModelFromStateRequest(BaseModel):
  state_path: str
  restore_optimizer: bool = False
  base_model: str | None = None
  session_id: str | None = None
  user_metadata: dict[str, Any] | None = Field(default=None, description=USER_METADATA_DOC)
  lora_config: LoraConfig = Field(default_factory=LoraConfig)
  full_config: Any = None

  @field_validator("lora_config", mode="before")
  @classmethod
  def default_config(cls, value):
    return {} if value is None else value


def parse_bool(value: Any) -> bool:
  if isinstance(value, str) and value.lower() in ("true", "false"):
    return value.lower() == "true"
  if not isinstance(value, bool):
    raise ValueError(f"must be 'true' or 'false', got {value!r}")
  return value


def parse_trainer_backend(value: str) -> str:
  if value not in TRAINER_BACKENDS and "/" not in value:
    raise ValueError(f"must be one of {', '.join(TRAINER_BACKENDS)} or an image like ghcr.io/org/trainer:tag, got {value!r}")
  return value


class Settings(BaseModel):
  """The descriptions are what get_server_capabilities and the skill show, so
  they are written for the client choosing a value. Each field's wire name is
  its alias, openrl.<field>."""

  model_config = ConfigDict(extra="forbid", alias_generator=lambda field: f"{SETTINGS_PREFIX}{field}", populate_by_name=True)

  sampler_replicas: int = Field(default=1, ge=1, le=8, description="Single-GPU samplers that drain the model's sampling queue together.")
  exclusive: Annotated[bool, BeforeValidator(parse_bool)] = Field(
    default=False,
    description="A workload no other workload shares. The model gets its own trainer and sampler, and nothing time-slices their GPUs.",
  )
  trainer_backend: Annotated[str, AfterValidator(parse_trainer_backend)] = Field(
    default="pytorch",
    description="The trainer the model runs on: pytorch, automodel (LoRA only), or an image like ghcr.io/org/trainer:tag that runs its own trainer.",
  )
  trainer_device_count: int = Field(
    default=1,
    ge=1,
    le=8,
    validation_alias=AliasChoices(f"{SETTINGS_PREFIX}trainer_device_count", f"{SETTINGS_PREFIX}trainer_gpus"),
    description="Accelerator devices the trainer drives as one data-parallel torchrun group. More than one is Automodel only "
    "and makes the model exclusive. openrl.trainer_gpus is a deprecated name for this setting.",
  )
  trainer_cp: int = Field(
    default=1,
    ge=1,
    le=8,
    description="Trainer devices that share one sequence, as context parallelism. The rest of trainer_device_count is data parallel.",
  )
  fine_tuning_type: FineTuningType | None = Field(
    default=None,
    description="lora unless set. A model restored from a checkpoint takes the checkpoint's type.",
  )
  recipe: Literal["rl", "sft"] | None = Field(
    default=None,
    description="What the job runs. rl starts the sampler together with the trainer, so its start-up overlaps the trainer's. "
    "sft or unset starts it at the first sample or save for sampling.",
  )
  client_log_id: str | None = Field(
    default=None,
    description="The id on the client's log lines, as the label openrl_client=<id> in the server's log store. "
    "The inspection API reads the run's client logs by it.",
  )
  weight_sync: Literal["delta", "full"] | None = Field(
    default=None,
    description="Full fine-tuning only. How the trainer hands each step's weights to the sampler: delta (changed weights only, the default) or full.",
  )
  fft_seed: int | None = Field(default=None, description="Full fine-tuning only. The trainer's random seed.")
  fft_cpu_offload: Annotated[bool | None, BeforeValidator(parse_bool)] = Field(
    default=None,
    description="Full fine-tuning only. Park trainer state in host memory between turns. Exclusive models default to off.",
  )


def settings_schema() -> dict[str, Any]:
  return Settings.model_json_schema(ref_template="#/components/schemas/{model}")


def tag_metadata(tags: list[str]) -> dict[str, str]:
  entries = {}
  for tag in tags:
    if tag.startswith(SETTINGS_PREFIX):
      key, sep, value = tag.partition("=")
      if not sep:
        raise ValueError(f"tag {tag!r} needs a value, like {key}=true")
      entries[key] = value
  return entries


def resolve_settings(*user_metadata: dict[str, Any]) -> Settings:
  merged: dict[str, Any] = {}
  for layer in reversed(user_metadata):
    merged.update({key: value for key, value in (layer or {}).items() if key.startswith(SETTINGS_PREFIX)})
  old, new = f"{SETTINGS_PREFIX}trainer_gpus", f"{SETTINGS_PREFIX}trainer_device_count"
  if old in merged and new in merged:
    if str(merged[old]) != str(merged[new]):
      raise ValueError(f"{new}={merged[new]} and {old}={merged[old]} disagree; {old} is the deprecated name of {new}, set only one")
    del merged[old]
  try:
    return Settings.model_validate(merged)
  except ValidationError as exc:
    problems = []
    for error in exc.errors():
      message = "unknown setting" if error["type"] == "extra_forbidden" else error["msg"].removeprefix("Value error, ")
      problems.append(f"{error['loc'][0]}: {message}")
    raise ValueError("; ".join(problems)) from None


class ModelRequest(BaseModel):
  model_id: str


class GetInfoRequest(BaseModel):
  model_id: str | None = None


class ForwardBackwardInput(BaseModel):
  data: list[Datum] = Field(default_factory=list)
  loss_fn: str = "cross_entropy"
  loss_fn_config: dict[str, Any] = Field(default_factory=dict)

  @field_validator("loss_fn_config", mode="before")
  @classmethod
  def default_loss_config(cls, value):
    return {} if value is None else value


class ForwardBackwardRequest(ModelRequest):
  forward_backward_input: ForwardBackwardInput | None = None
  forward_input: ForwardBackwardInput | None = None
  forward_only: bool = False


class RetrieveFutureRequest(BaseModel):
  request_id: str


class OptimStepRequest(ModelRequest):
  adam_params: dict[str, Any] = {}


class SaveWeightsForSamplerRequest(ModelRequest):
  sampling_session_seq_id: int | str | None = None
  name: str | None = None
  alias: str | None = None
  path: str | None = None


class SaveWeightsRequest(ModelRequest):
  seq_id: int | str | None = None
  path: str | None = None


class LoadWeightsRequest(ModelRequest):
  path: str
  optimizer: bool = False


class WeightsInfoRequest(BaseModel):
  tinker_path: str


class CreateSamplingSessionRequest(BaseModel):
  model_path: str | None = None
  base_model: str | None = None
  model_id: str | None = None
  session_id: str | None = None


class SamplingParams(BaseModel):
  max_tokens: int = 20
  temperature: float = 1.0
  stop: Any = None
  top_p: float = 1.0
  top_k: int = -1
  seed: int | None = None

  @field_validator("max_tokens", "temperature", "top_p", "top_k", mode="before")
  @classmethod
  def default_sampling_param(cls, value, info: ValidationInfo):
    return cls.model_fields[info.field_name].default if value is None else value


class AsampleRequest(BaseModel):
  prompt: dict[str, Any] = {}
  sampling_params: SamplingParams = SamplingParams()
  num_samples: int = 1
  prompt_logprobs: bool = Field(default=False, validation_alias=AliasChoices("prompt_logprobs", "include_prompt_logprobs"))
  model_id: str | None = None
  sampling_session_id: str | None = None
  seq_id: int | None = None


@dataclass(frozen=True)
class ParsedForwardBackward:
  """Validated forward_backward/forward payload carrying binary protobuf bytes."""

  model_id: str
  loss_fn: str
  loss_config: dict[str, Any]
  forward_only: bool
  proto_bytes: bytes


# ***** helpers *****


def is_single_process_mode() -> bool:
  return bool(os.getenv("BASE_MODEL")) and not bool(os.getenv("REDIS_URL"))


def get_sampler_backend() -> str:
  if sampling_backend := os.getenv("SAMPLING_BACKEND"):
    return sampling_backend.lower()
  return "torch" if is_single_process_mode() else "vllm"


def get_default_model_name() -> str | None:
  return os.getenv("BASE_MODEL")


def is_fft_enabled() -> bool:
  return os.getenv("OPEN_RL_ENABLE_FFT", "").lower() == "true"


def env_int(name: str) -> int | None:
  value = os.getenv(name)
  return int(value) if value and value.isdigit() else None


def sampling_ref(sampling_session_id: str) -> str:
  return re.sub(r"@[0-9a-f]{32}$", "", sampling_session_id)


def sampler_session_id(model_id: str, seq_id: int | str) -> str:
  return f"tinker://{model_id}/sampler_weights/sampler-{seq_id}"


def sampler_weights_path(model_id: str, name: str) -> str:
  return f"tinker://{model_id}/sampler_weights/{name}"


def resolve_sampler_weights_path(model_id: str, *, tmp_dir: str | None = None) -> str:
  root = DEFAULT_TMP_DIR if tmp_dir is None else tmp_dir
  rel_path = model_id[len("tinker://") :] if model_id.startswith("tinker://") else model_id.lstrip("/")
  local_path = os.path.join(root, "sampler_full", rel_path)
  weights_path = local_path
  if not os.path.basename(weights_path).startswith("sampler-"):
    sampler_weights_dir = os.path.join(weights_path, "sampler_weights")
    if os.path.exists(sampler_weights_dir):
      try:
        steps = [int(d.split("-")[1]) for d in os.listdir(sampler_weights_dir) if d.startswith("sampler-")]
        if steps:
          weights_path = os.path.join(sampler_weights_dir, f"sampler-{max(steps)}")
      except Exception as e:
        print(f"[API_SERVER] Warning: Failed parsing step subdirectories in {sampler_weights_dir}: {e}")
  return weights_path


def tinker_checkpoint_dir(path: str, *, tmp_dir: str | None = None) -> str | None:
  root = DEFAULT_TMP_DIR if tmp_dir is None else tmp_dir
  if not path.startswith("tinker://"):
    return None
  owner, sep, rest = path[len("tinker://") :].partition("/weights/")
  if not (owner and sep):
    return None
  return os.path.join(root, "checkpoints", owner, "weights", rest)


def checkpoint_state_path(model_id: str, name: str, *, tmp_dir: str | None = None) -> str:
  root = DEFAULT_TMP_DIR if tmp_dir is None else tmp_dir
  if (state_dir := tinker_checkpoint_dir(name, tmp_dir=root)) is not None:
    return state_dir
  if os.path.isabs(name):
    return name
  return os.path.join(root, "checkpoints", model_id, "weights", name)


def tinker_state_path(state_path: str, *, tmp_dir: str | None = None) -> str:
  root = os.path.join(DEFAULT_TMP_DIR if tmp_dir is None else tmp_dir, "checkpoints") + os.sep
  if state_path.startswith(root):
    model_id, sep, rest = state_path[len(root) :].partition("/weights/")
    if model_id and sep:
      return f"tinker://{model_id}/weights/{rest}"
  return state_path


def base_model_id_from_sampling_ref(model_id: str | None) -> str | None:
  if not model_id:
    return None
  if model_id.startswith("tinker://"):
    path = model_id[len("tinker://") :]
    parts = path.split("/")
    if len(parts) >= 3 and parts[1] == "sampler_weights":
      return parts[0]
    return path
  return model_id.split("-samp-")[0]


def new_request_id() -> str:
  return str(uuid.uuid4())


def runtime_jobs_key(runtime: str) -> str:
  return f"open_rl:runtime_jobs:{runtime}"


def check_sampler_backend(*, sampler_backend: str | None = None) -> None:
  backend = get_sampler_backend() if sampler_backend is None else sampler_backend
  if backend == "vllm" and not os.getenv("REDIS_URL"):
    raise RuntimeError(
      "Sampling pools need REDIS_URL, because vLLM sampling requests wait in a Redis stream for each pool's dispatcher. "
      "Set REDIS_URL (for example redis://127.0.0.1:6379/0), or set BASE_MODEL and unset SAMPLING_BACKEND to sample with torch in one process."
    )


def check_lora_jobs_per_runtime() -> None:
  value = os.getenv("OPEN_RL_LORA_JOBS_PER_RUNTIME", "")
  if value and not value.isdigit():
    raise RuntimeError(f"OPEN_RL_LORA_JOBS_PER_RUNTIME is {value!r}; it must be 0 (one runtime) or a positive number of jobs")


def translate_future_result(result: dict, *, tmp_dir: str | None = None) -> dict:
  result_type = result.get("type")
  if result_type in {"model_created", "model_loaded_from_state"}:
    response = {
      "model_id": result["model_id"],
      "is_lora": True,
      "type": "create_model" if result_type == "model_created" else "create_model_from_state",
    }
    if "rank" in result:
      response["lora_rank"] = result["rank"]
    elif result.get("fine_tuning_type") == "full":
      response["lora_rank"] = 16
    if result.get("base_model"):
      response["base_model"] = result["base_model"]
    return response

  public_type_by_internal_type = {
    "forward_backward_completed": "forward_backward",
    "optim_step_completed": "optim_step",
    "sample_completed": "sample",
    "state_saved": "save_weights",
    "weights_loaded": "load_weights",
    "sampler_weights_saved": "save_weights_for_sampler",
    "weights_saved": "save_weights",
  }
  if result_type in public_type_by_internal_type:
    response = dict(result)
    response["type"] = public_type_by_internal_type[result_type]
    if result_type == "state_saved" and isinstance(response.get("path"), str):
      response["path"] = tinker_state_path(response["path"], tmp_dir=tmp_dir)
    if result_type in ("state_saved", "sampler_weights_saved", "weights_saved"):
      response.pop("metrics", None)
    return response

  return result


def checkpoint_info(path: str, *, tmp_dir: str | None = None) -> dict[str, Any]:
  state_dir = tinker_checkpoint_dir(path, tmp_dir=tmp_dir) or (path if os.path.isabs(path) else None)
  metadata_path = os.path.join(state_dir, "metadata.json") if state_dir else None
  if not metadata_path or not os.path.exists(metadata_path):
    raise HTTPException(status_code=404, detail=f"No checkpoint at {path}")
  with open(metadata_path) as f:
    saved = json.load(f)
  if not isinstance(saved, dict) or not isinstance(saved.get("base_model"), str) or not saved["base_model"]:
    raise ValueError("Checkpoint metadata must specify base_model")
  saved_model_id = saved.get("model_id", "")
  if not isinstance(saved_model_id, str):
    raise ValueError("Checkpoint model_id must be a string")
  adapter_config_path = os.path.join(state_dir, saved_model_id, "adapter_config.json")
  if not os.path.exists(adapter_config_path):
    adapter_config_path = os.path.join(state_dir, "adapter_config.json")
  is_lora = os.path.exists(adapter_config_path)
  rank = None
  if is_lora:
    with open(adapter_config_path) as f:
      adapter_config = json.load(f)
    if not isinstance(adapter_config, dict):
      raise ValueError("Checkpoint adapter config must be an object")
    rank = adapter_config.get("r")
  return {"base_model": saved["base_model"], "is_lora": is_lora, "lora_rank": rank, "type": "weights_info"}


def sample_sequence_ids(request_id: str, num_samples: int) -> list[str]:
  return [f"{request_id}:{index}" for index in range(max(1, int(num_samples)))]


def visible(path: Path) -> bool:
  if path.is_file():
    return True
  for directory in path.parents:
    try:
      os.listdir(directory)
      break
    except OSError:
      continue
  return path.is_file()


def content_type(request: Request) -> str:
  return request.headers.get("content-type", "").split(";", 1)[0].strip().lower()


async def request_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
  errors = jsonable_encoder(exc.errors(), custom_encoder={bytes: lambda b: f"<{len(b)} bytes>"})
  return JSONResponse(status_code=422, content={"error": "invalid request", "detail": errors})


async def http_error(_: Request, exc: StarletteHTTPException) -> Response:
  if not is_body_allowed_for_status_code(exc.status_code):
    return Response(status_code=exc.status_code, headers=exc.headers)
  return JSONResponse(status_code=exc.status_code, content={"error": exc.detail}, headers=exc.headers)


class FilterNoisyEndpoints(logging.Filter):
  def filter(self, record: logging.LogRecord) -> bool:
    msg = record.getMessage()
    return "retrieve_future" not in msg and "session_heartbeat" not in msg


def configure_telemetry() -> None:
  """Configure OpenTelemetry tracing and uvicorn access-log filtering once."""
  provider = TracerProvider()
  trace.set_tracer_provider(provider)
  if os.getenv("ENABLE_GCP_TRACE", "0") == "1":
    try:
      from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

      exporter = CloudTraceSpanExporter()
      provider.add_span_processor(BatchSpanProcessor(exporter))
      print("OpenTelemetry: Configured GCP CloudTraceSpanExporter")
    except ImportError:
      print("OpenTelemetry: opentelemetry-exporter-gcp-trace is not installed")
  else:
    print("OpenTelemetry: No exporter configured (ENABLE_GCP_TRACE=0)")
  logging.getLogger("uvicorn.access").addFilter(FilterNoisyEndpoints())


async def parse_forward_backward_request(request: Request) -> ParsedForwardBackward:
  """Validate a `forward_backward` or `forward` body and produce binary protobuf bytes.

  On `application/x-protobuf`, validates chunks/tensors via `wire.parse_forward_backward`
  and forwards the raw protobuf bytes untouched (zero `.tolist()` / Pydantic tensor overhead).
  On `application/json`, validates with `ForwardBackwardRequest` and packs protobuf bytes at
  the gateway edge.
  """
  encoding = request.headers.get("content-encoding", "identity").strip().lower()
  if encoding not in ("", "identity"):
    raise HTTPException(status_code=415, detail=f"Content-Encoding {encoding!r} is not supported; this server does not enable request compression")
  body = await request.body()
  media = content_type(request)
  if media == wire.PROTO:
    try:
      parsed = wire.parse_forward_backward(body)
    except wire.ProtoDecodeError as exc:
      raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ParsedForwardBackward(
      model_id=parsed.model_id,
      loss_fn=parsed.loss_fn or "cross_entropy",
      loss_config=wire.loss_config(parsed),
      forward_only=bool(parsed.forward_only),
      proto_bytes=body,
    )
  if media not in (wire.JSON, ""):
    raise HTTPException(status_code=415, detail=f"unsupported Content-Type {media!r}; send application/json or {wire.PROTO}")
  try:
    req = ForwardBackwardRequest.model_validate_json(body or b"{}")
  except ValidationError as exc:
    raise RequestValidationError(exc.errors()) from exc
  fwd_input = req.forward_backward_input or req.forward_input or ForwardBackwardInput()
  raw_json = json.loads(body or b"{}")
  proto_bytes = wire.json_to_forward_backward_proto(raw_json, forward_only=req.forward_only)
  return ParsedForwardBackward(
    model_id=req.model_id,
    loss_fn=fwd_input.loss_fn,
    loss_config=dict(fwd_input.loss_fn_config),
    forward_only=req.forward_only,
    proto_bytes=proto_bytes,
  )


# ***** runtime *****


@dataclass
class GatewayState:
  """Dependency container for a gateway FastAPI instance."""

  store: RequestStore
  state: StateStore
  session_registry: SessionRegistry
  worker_manager: WorkerManager | None = None
  owner_locks: defaultdict[str, asyncio.Lock] = field(default_factory=lambda: defaultdict(asyncio.Lock))
  slot_locks: defaultdict[str, asyncio.Lock] = field(default_factory=lambda: defaultdict(asyncio.Lock))
  tmp_dir: str = field(default_factory=lambda: os.getenv("OPEN_RL_TMP_DIR", "/tmp/open-rl"))
  version: str = field(default_factory=lambda: os.getenv("OPEN_RL_VERSION") or "dev")
  saved_weights_wait_seconds: float = SAVED_WEIGHTS_WAIT_SECONDS
  session_reap_interval_sec: float = SESSION_REAP_INTERVAL_SEC
  reconcile_concurrency: int = RECONCILE_CONCURRENCY

  def sampler_backend(self) -> str:
    return get_sampler_backend()

  def single_process_mode(self) -> bool:
    return is_single_process_mode()

  def read_checkpoint_info(self, path: str) -> dict[str, Any]:
    return checkpoint_info(path, tmp_dir=self.tmp_dir)

  def log_exception(self) -> None:
    traceback.print_exc()


async def bind_session(ctx: GatewayState, session_id: str | None, model_id: str) -> None:
  if not session_id:
    return
  run_records.session_seen(session_id)
  owner = await asyncio.to_thread(owner_of, model_id)
  async with ctx.owner_locks[owner]:
    if ctx.worker_manager is not None:
      await ctx.session_registry.attach(session_id, owner)
    meta = await get_model_metadata(ctx.state, model_id)
    if meta is not None and session_id not in meta.session_ids:
      meta.session_ids.append(session_id)
      await persist_model_metadata(ctx.state, model_id, meta)


async def reap_owner(ctx: GatewayState, owner: str) -> None:
  async with ctx.owner_locks[owner]:
    if await ctx.session_registry.in_use(owner):
      return
    print(f"[API_SERVER] No live session uses {owner}; tearing its workers down")
    for model in await asyncio.to_thread(ctx.worker_manager.release_owner, owner):
      await ctx.state.delete_values(f"open_rl:trainer_ready:{model}")
      if ctx.sampler_backend() == "vllm":
        await SamplingStore(ctx.store.redis).delete_if_empty(pool_name(model))
    await ctx.session_registry.forget(owner)


async def job_is_live(ctx: GatewayState, model_id: str) -> bool:
  meta = await get_model_metadata(ctx.state, model_id)
  if meta is None or meta.status != "active":
    return False
  for session_id in meta.session_ids:
    if await ctx.session_registry.live(session_id):
      return True
  return not meta.session_ids


async def prune_runtime_jobs(ctx: GatewayState, runtime: str) -> int:
  key = runtime_jobs_key(runtime)
  live = 0
  for job in await ctx.state.set_members(key):
    if await job_is_live(ctx, job):
      live += 1
    else:
      await ctx.state.remove_from_set(key, job)
  return live


async def persist_on_runtime_slot(ctx: GatewayState, model_id: str, meta: TrainingModelMetadata) -> None:
  cap = env_int("OPEN_RL_LORA_JOBS_PER_RUNTIME") or 0
  if cap <= 0 or ctx.worker_manager is None or not meta.shares_runtime():
    await persist_model_metadata(ctx.state, model_id, meta)
    return
  async with ctx.slot_locks[meta.shared_runtime(0)]:
    slot = 0
    while await prune_runtime_jobs(ctx, meta.shared_runtime(slot)) >= cap:
      slot += 1
    meta.runtime_slot = slot
    await persist_model_metadata(ctx.state, model_id, meta)
    await ctx.state.add_to_set(runtime_jobs_key(meta.runtime(model_id)), model_id)


async def extract_and_persist_model_metadata(
  ctx: GatewayState,
  req: CreateModelRequest | CreateModelFromStateRequest,
  request: Request | None = None,
  checkpoint_kind: FineTuningType | None = None,
) -> tuple[str, TrainingModelMetadata]:
  base_model = req.base_model
  if not base_model:
    raise ValueError("base_model is required in request payload")
  headers = request.headers if request is not None else {}
  for header, setting in (("x-open-rl-fine-tuning-type", "fine_tuning_type"), ("x-open-rl-weight-sync-strategy", "weight_sync")):
    if header in headers:
      raise ValueError(f"The {header} header was removed; set {SETTINGS_PREFIX}{setting} in TINKER_TAGS or user_metadata")
  if req.full_config is not None:
    raise ValueError(f"full_config was removed; set {SETTINGS_PREFIX}fft_seed and {SETTINGS_PREFIX}fft_cpu_offload in TINKER_TAGS or user_metadata")
  lora_config = req.lora_config.model_dump()

  settings = resolve_settings(req.user_metadata or {}, await ctx.session_registry.user_metadata(req.session_id))
  if checkpoint_kind is not None and settings.fine_tuning_type not in (None, checkpoint_kind):
    raise ValueError(f"{SETTINGS_PREFIX}fine_tuning_type={settings.fine_tuning_type} does not match the {checkpoint_kind} checkpoint")
  fine_tuning_type = settings.fine_tuning_type or checkpoint_kind or "lora"
  if fine_tuning_type == "full" and not is_fft_enabled():
    raise ValueError("Full Fine-Tuning (FFT) is disabled on this Open-RL API server instance")
  fft_settings = {"weight_sync": settings.weight_sync, "fft_seed": settings.fft_seed, "fft_cpu_offload": settings.fft_cpu_offload}
  if fine_tuning_type != "full" and (given := [name for name, value in fft_settings.items() if value is not None]):
    raise ValueError(", ".join(f"{SETTINGS_PREFIX}{name}" for name in given) + " applies to fine_tuning_type=full only")
  weight_sync_cfg = WeightSyncConfig(strategy=settings.weight_sync or "delta")
  cpu_offload = settings.fft_cpu_offload if settings.fft_cpu_offload is not None else not settings.exclusive
  full_config = FFTConfig(seed=settings.fft_seed, cpu_offload=cpu_offload, weight_sync_strategy=weight_sync_cfg.strategy).model_dump()
  wm = ctx.worker_manager
  if settings.exclusive and wm is None:
    raise ValueError("openrl.exclusive needs a server that launches workers per model")
  if settings.sampler_replicas > 1 and isinstance(wm, LocalWorkerManager):
    raise ValueError("openrl.sampler_replicas above 1 needs a server that launches workers as pods")
  if settings.trainer_backend != "pytorch" and wm is None:
    raise ValueError(f"openrl.trainer_backend={settings.trainer_backend} needs a server that launches workers per model")
  if settings.trainer_backend not in TRAINER_BACKENDS and isinstance(wm, LocalWorkerManager):
    raise ValueError("A trainer image needs a server that launches workers as pods")
  if settings.trainer_backend == "automodel" and fine_tuning_type != "lora":
    raise ValueError("The automodel trainer supports LoRA only")
  if settings.trainer_device_count > 1 and settings.trainer_backend != "automodel":
    raise ValueError("openrl.trainer_device_count above 1 needs openrl.trainer_backend=automodel")
  if settings.trainer_device_count > 1 and isinstance(wm, LocalWorkerManager):
    raise ValueError("openrl.trainer_device_count above 1 needs a server that launches workers as pods")
  if settings.trainer_device_count % settings.trainer_cp:
    raise ValueError(f"openrl.trainer_cp={settings.trainer_cp} must divide openrl.trainer_device_count={settings.trainer_device_count}")
  model_id = str(uuid.uuid4())
  meta_obj = TrainingModelMetadata(
    base_model=base_model,
    created_at=time.time(),
    fine_tuning_type=fine_tuning_type,
    weight_sync_config=weight_sync_cfg,
    full_config=full_config,
    lora_config=lora_config,
    sampler_replicas=settings.sampler_replicas,
    exclusive=settings.exclusive,
    trainer_backend=settings.trainer_backend,
    trainer_device_count=settings.trainer_device_count,
    trainer_cp=settings.trainer_cp,
    recipe=settings.recipe,
    client_log_id=settings.client_log_id,
  )
  if meta_obj.sampler_replicas > 1 and meta_obj.shares_gpu():
    raise ValueError("openrl.sampler_replicas above 1 needs samplers on GPUs of their own; set openrl.exclusive=true")
  await persist_on_runtime_slot(ctx, model_id, meta_obj)
  return model_id, meta_obj


async def resolve_active_set_id(ctx: GatewayState, model_id: str | None) -> str | None:
  if not model_id:
    return None
  meta = await get_model_metadata(ctx.state, model_id)
  if meta and meta.fine_tuning_type == "lora" and meta.base_model:
    return f"{meta.runtime(model_id)}-1"
  return None


async def open_future(request_id: str) -> dict[str, str]:
  del request_id
  carrier: dict[str, str] = {}
  propagate.inject(carrier)
  return carrier


async def enqueue(ctx: GatewayState, command: Command) -> str:
  carrier = await open_future(command.request_id)
  active_set_id = await resolve_active_set_id(ctx, command.model_id)
  carrier[commands.ENQUEUED_AT_KEY] = str(time.time())
  await ctx.store.put_request(commands.wire(command.model_copy(update={"trace_context": carrier})), active_set_id=active_set_id)
  run_records.enqueued(command.model_id)
  print(f"[API_SERVER] enqueued op={command.op} request_id={command.request_id} model_id={command.model_id} active_set={active_set_id}")
  return command.request_id


async def ensure_sampler_launched(ctx: GatewayState, model_id: str) -> None:
  if ctx.worker_manager is not None and ctx.sampler_backend() == "vllm":
    try:
      await asyncio.to_thread(ctx.worker_manager.ensure, model_id, "sampler")
    except Exception:
      ctx.log_exception()


async def launch_worker_and_enqueue(ctx: GatewayState, command: Command, start_sampler: bool = False) -> str:
  assert ctx.worker_manager is not None, "Worker manager is initialized by the app lifespan"
  request_id = command.request_id
  try:
    await asyncio.to_thread(ctx.worker_manager.ensure, command.model_id, "trainer")
  except Exception as exc:
    ctx.log_exception()
    await ctx.store.set_result(request_id, {"type": "RequestFailedResponse", "error_message": str(exc)})
    return request_id
  if start_sampler:
    await ensure_sampler_launched(ctx, command.model_id)
  return await enqueue(ctx, command)


async def every_interval(ctx: GatewayState, step: Callable[[], Any]) -> None:
  while True:
    await asyncio.sleep(ctx.session_reap_interval_sec)
    try:
      await step()
    except Exception:
      ctx.log_exception()


async def reap_dead_sessions(ctx: GatewayState) -> None:
  while True:
    await asyncio.sleep(ctx.session_reap_interval_sec)
    try:
      owners = await ctx.session_registry.owners()
    except Exception:
      ctx.log_exception()
      continue
    for owner in owners:
      try:
        await reap_owner(ctx, owner)
      except Exception:
        ctx.log_exception()


async def reconcile_job(ctx: GatewayState, sampling: SamplingStore, pool: str, job: str) -> None:
  meta, runtime, _ = await asyncio.to_thread(runtime_of, job)
  if meta.pool(job) != pool:
    return
  owner = owner_id(runtime)
  async with ctx.owner_locks[owner]:
    if (current := await get_model_metadata(ctx.state, job)) is not None and current.status == "completed":
      if failed := await sampling.fail_job(pool, job, {"type": "RequestFailedResponse", "error_message": f"{job} was deleted"}):
        print(f"[API_SERVER] Failed {failed} sampling requests of {job}, which was deleted")
      return
    if await asyncio.to_thread(ctx.worker_manager.sampler_exists, job):
      return
    if await ctx.session_registry.in_use(owner):
      await asyncio.to_thread(ctx.worker_manager.ensure, job, "sampler")
      return
    message = f"The pool of {job} is gone and its session ended before a sampler read the request"
    if failed := await sampling.fail_job(pool, job, {"type": "RequestFailedResponse", "error_message": message}):
      print(f"[API_SERVER] Failed {failed} sampling requests of {job}, whose pool is gone and whose session ended")


async def reconcile_pools(ctx: GatewayState) -> None:
  if ctx.worker_manager is None:
    return
  sampling = SamplingStore(ctx.store.redis)
  pairs = []
  for pool in await sampling.pools():
    try:
      jobs = await sampling.pending_jobs(pool)
      if not jobs:
        await sampling.delete_if_empty(pool)
      pairs += [(pool, job) for job in jobs]
    except Exception:
      ctx.log_exception()
  slots = asyncio.Semaphore(ctx.reconcile_concurrency)

  async def reconcile(pool: str, job: str) -> None:
    async with slots:
      await reconcile_job(ctx, sampling, pool, job)

  for outcome in await asyncio.gather(*(reconcile(pool, job) for pool, job in pairs), return_exceptions=True):
    if isinstance(outcome, Exception):
      traceback.print_exception(outcome)


async def sweep_sample_deadlines(ctx: GatewayState) -> None:
  result = {"type": "RequestFailedResponse", "error_message": f"No sampler read the request within {SAMPLE_DEADLINE_SECONDS}s"}
  if failed := await SamplingStore(ctx.store.redis).fail_older_than(SAMPLE_DEADLINE_SECONDS, result):
    print(f"[API_SERVER] Failed {failed} sampling requests no sampler read in time")


async def wait_until_visible(ctx: GatewayState, path: Path) -> bool:
  deadline = time.monotonic() + ctx.saved_weights_wait_seconds
  while not await asyncio.to_thread(visible, path):
    if time.monotonic() >= deadline:
      return False
    await asyncio.sleep(0.2)
  return True


async def accept_pool_sample(ctx: GatewayState, req: AsampleRequest, prompt: list[int]) -> str:
  ref = sampling_ref(req.model_id or req.sampling_session_id or "")
  lookup_id = base_model_id_from_sampling_ref(ref) or ref
  meta = await get_model_metadata(ctx.state, lookup_id)
  if meta is None:
    meta, revision, model = default_metadata(lookup_id), "", lookup_id
  elif meta.fine_tuning_type == "full":
    revision, model = resolve_sampler_weights_path(ref, tmp_dir=ctx.tmp_dir), meta.base_model
    if not await wait_until_visible(ctx, Path(revision, "metadata.json")):
      raise HTTPException(status_code=400, detail=f"No saved sampler weights for {ref}")
  else:
    revision, model = "", snapshot_name(ref)
    if not await wait_until_visible(ctx, snapshot_root() / model / "adapter_config.json"):
      raise HTTPException(status_code=400, detail=f"No saved LoRA sampler weights for {ref}")
  pool = meta.pool(lookup_id)

  params = req.sampling_params
  stop, stop_token_ids = split_stop(params.stop)
  return await SamplingStore(ctx.store.redis).accept(
    {
      "request_id": new_request_id(),
      "pool": pool,
      "job": lookup_id,
      "revision": revision,
      "model": model,
      "idempotency_key": f"{req.sampling_session_id}:{req.seq_id}" if req.sampling_session_id and req.seq_id is not None else None,
      "prompt_token_ids": prompt,
      "sampling_params": {"n": req.num_samples, **params.model_dump(exclude={"stop"}), "stop": stop or [], "stop_token_ids": stop_token_ids or []},
      "include_prompt_logprobs": req.prompt_logprobs,
    }
  )


def worker_launcher_mode(ctx: GatewayState) -> str:
  if ctx.single_process_mode():
    return "single_process"
  if ctx.worker_manager is None:
    return "none"
  mode = os.getenv("OPEN_RL_WORKER_MANAGER", "local").lower()
  return "scheduler" if mode == "scheduler" else "process"


def openrl_info(ctx: GatewayState) -> dict[str, Any]:
  release = ctx.version.startswith("v")
  return {
    "version": ctx.version,
    "worker_launcher": worker_launcher_mode(ctx),
    "sampler_backend": ctx.sampler_backend(),
    "fft_enabled": is_fft_enabled(),
    "models": {
      "any_huggingface_model": not ctx.single_process_mode(),
      "default": get_default_model_name(),
      "examples": EXAMPLE_MODELS,
    },
    "settings_schema_url": "/openapi.json#/components/schemas/OpenRLSettings",
    "settings_howto": "Pass as user_metadata on create_session or create_model, or as TINKER_TAGS=openrl.key=value,...",
    "limits": {
      "vllm_max_model_len": env_int("VLLM_MAX_MODEL_LEN"),
      "train_token_budget": env_int("OPEN_RL_TRAIN_TOKEN_BUDGET"),
      "session_ttl_seconds": ctx.session_registry.ttl_seconds,
      "session_reap_interval_seconds": ctx.session_reap_interval_sec,
    },
    "skill_url": f"{RELEASES_URL}/download/{ctx.version}/openrl-skill.tar.gz" if release else None,
    "getting_started_url": "https://github.com/openrl-dev/openrl/blob/main/docs/getting-started.md",
  }


def make_lifespan(ctx: GatewayState):
  @asynccontextmanager
  async def lifespan(_: FastAPI):
    task = None
    check_sampler_backend(sampler_backend=ctx.sampler_backend())
    check_lora_jobs_per_runtime()
    run_records.start("api_server", worker_record=False)
    created_wm = False
    if ctx.worker_manager is None and (is_fft_enabled() or os.getenv("REDIS_URL") or os.getenv("OPEN_RL_WORKER_MANAGER")):
      ctx.worker_manager = create_worker_manager()
      created_wm = True
    if ctx.single_process_mode():
      base_model = os.getenv("BASE_MODEL")
      print("\n" + "=" * 50)
      print(" Open-RL Single-Process Mode")
      print("=" * 50)
      print(f"-> Base model: {base_model or 'unset'}")
      print(f"-> Sampling backend: {ctx.sampler_backend()}")
      print(f"-> FFT enabled     : {is_fft_enabled()}")
      print("-> Server mode     : API server + worker loop in one process\n")
      if not is_fft_enabled():
        from server import training_requests_processor

        worker = training_requests_processor.LoraTrainingWorker()
        if base_model:
          await asyncio.to_thread(worker.load_base_model, base_model)
        task = asyncio.create_task(training_requests_processor.run_training_requests_processor(worker))
    reap_task = asyncio.create_task(reap_dead_sessions(ctx)) if ctx.worker_manager is not None else None
    pool_tasks = []
    if ctx.sampler_backend() == "vllm":
      pool_tasks = [
        asyncio.create_task(every_interval(ctx, lambda: sweep_sample_deadlines(ctx))),
        asyncio.create_task(every_interval(ctx, lambda: reconcile_pools(ctx))),
      ]
    try:
      yield
    finally:
      for background in (reap_task, *pool_tasks):
        if background is not None:
          background.cancel()
      if task is not None:
        task.cancel()
      if ctx.worker_manager is not None and (created_wm or getattr(ctx, "_close_worker_manager_on_exit", True)):
        ctx.worker_manager.close()
        ctx.worker_manager = None
      await run_records.finish()

  return lifespan


# ***** app factory *****


def create_app(
  store: RequestStore | None = None,
  state: StateStore | None = None,
  session_registry: SessionRegistry | None = None,
  worker_manager: WorkerManager | None = None,
  *,
  ctx: GatewayState | None = None,
  tmp_dir: str | None = None,
  version: str | None = None,
  enable_lifespan: bool = True,
  instrument: bool = False,
) -> FastAPI:
  """Build an OpenRL Tinker HTTP gateway with explicit dependencies."""
  if ctx is None:
    resolved_store = get_store() if store is None else store
    resolved_state = get_state_store() if state is None else state
    resolved_registry = SessionRegistry(resolved_state) if session_registry is None else session_registry
    ctx = GatewayState(
      store=resolved_store,
      state=resolved_state,
      session_registry=resolved_registry,
      worker_manager=worker_manager,
      tmp_dir=os.getenv("OPEN_RL_TMP_DIR", "/tmp/open-rl") if tmp_dir is None else tmp_dir,
      version=(os.getenv("OPEN_RL_VERSION") or "dev") if version is None else version,
    )

  lifespan = make_lifespan(ctx) if enable_lifespan else None
  api = FastAPI(
    title="OpenRL API server",
    version=ctx.version,
    description=(
      "A self-hosted, Kubernetes-native post-training API that serves the Tinker API. "
      "Tinker SDK clients use the /api/v1 paths. GET /api/v1/openrl/models/{model_id} is OpenRL's own: "
      "what the server knows about one model, its settings, its workers and whether its session is live. "
      "Repository and docs: https://github.com/openrl-dev/openrl"
    ),
    lifespan=lifespan,
  )
  api.state.gateway = ctx

  def openapi_with_settings() -> dict[str, Any]:
    if api.openapi_schema:
      return api.openapi_schema
    schema = get_openapi(title=api.title, version=api.version, description=api.description, routes=api.routes)
    settings = settings_schema()
    schemas = schema.setdefault("components", {}).setdefault("schemas", {})
    schemas.update(settings.pop("$defs", {}))
    schemas["OpenRLSettings"] = settings
    api.openapi_schema = schema
    return schema

  api.openapi = openapi_with_settings
  if instrument:
    FastAPIInstrumentor.instrument_app(api, excluded_urls="/api/v1/retrieve_future,/api/v1/session_heartbeat")
  api.include_router(observability_router)
  api.add_exception_handler(RequestValidationError, request_validation_error)
  api.add_exception_handler(StarletteHTTPException, http_error)

  # ***** service *****

  @api.get("/api/v1/healthz")
  async def health_check():
    return {"status": "ok"}

  @api.get("/api/v1/get_server_capabilities")
  async def get_server_capabilities():
    model_name = get_default_model_name()
    return {
      "supported_models": [{"model_name": model_name}] if model_name else [],
      "default_model": model_name,
      "single_process": ctx.single_process_mode(),
      "openrl": openrl_info(ctx),
    }

  @api.post("/api/v1/client/config")
  async def client_config(_: dict):
    return {
      "pjwt_auth_enabled": False,
      "credential_default_source": "api_key",
      "sample_dispatch_bytes_semaphore_size": 10 * 1024 * 1024,
      "inflight_response_bytes_semaphore_size": 50 * 1024 * 1024,
    }

  @api.post("/api/v1/create_session")
  async def create_session(req: CreateSessionRequest):
    try:
      user_metadata = {**tag_metadata(req.tags), **(req.user_metadata or {})}
      resolve_settings(user_metadata)
    except ValueError as exc:
      raise HTTPException(status_code=400, detail=str(exc)) from exc
    session_id = f"sess-{uuid.uuid4().hex[:12]}"
    await ctx.session_registry.update_metadata(session_id, user_metadata)
    run_records.session_seen(session_id, user_metadata)
    return {"session_id": session_id, "type": "create_session"}

  @api.post("/api/v1/session_heartbeat")
  async def session_heartbeat(req: SessionHeartbeatRequest):
    if req.session_id:
      await ctx.session_registry.heartbeat(req.session_id)
      run_records.session_seen(req.session_id)
    return {"type": "session_heartbeat"}

  @api.post("/api/v1/create_model")
  async def create_model(req: CreateModelRequest, request: Request) -> dict[str, Any]:
    try:
      model_id, meta = await extract_and_persist_model_metadata(ctx, req, request)
    except ValueError as exc:
      raise HTTPException(status_code=400, detail=str(exc)) from exc
    await bind_session(ctx, req.session_id, model_id)
    command = commands.CreateModel(
      request_id=model_id,
      model_id=model_id,
      base_model=meta.base_model,
      fine_tuning_type=meta.fine_tuning_type,
      lora_config=meta.lora_config or {},
      full_config=meta.full_config or {},
    )
    req_id = (
      await launch_worker_and_enqueue(ctx, command, start_sampler=meta.recipe == "rl")
      if ctx.worker_manager is not None
      else await enqueue(ctx, command)
    )
    return {"request_id": req_id}

  @api.post("/api/v1/delete_model")
  async def delete_model(req: ModelRequest):
    model_id = req.model_id
    meta = await get_model_metadata(ctx.state, model_id)
    if meta is None:
      raise HTTPException(status_code=404, detail=f"Unknown model: {model_id}")
    if meta.shares_runtime():
      delete = commands.DeleteModel(request_id=str(uuid.uuid4()), model_id=model_id)
      await ctx.store.put_request(commands.wire(delete), active_set_id=await resolve_active_set_id(ctx, model_id))
    elif ctx.worker_manager is not None:
      print(f"[API_SERVER] Requesting shutdown of workers for model {model_id}...")
      await ctx.store.put_request(commands.wire(commands.Shutdown(model_id=model_id)), active_set_id=await resolve_active_set_id(ctx, model_id))
    async with ctx.owner_locks[await asyncio.to_thread(owner_of, model_id)]:
      now = time.time()
      meta.status = "completed"
      meta.completed_at = now
      meta.updated_at = now
      await persist_model_metadata(ctx.state, model_id, meta)
      if ctx.sampler_backend() == "vllm":
        result = {"type": "RequestFailedResponse", "error_message": f"{model_id} was deleted"}
        await SamplingStore(ctx.store.redis).fail_job(meta.pool(model_id), model_id, result)
      if ctx.worker_manager is not None and not meta.shares_runtime():
        await asyncio.to_thread(ctx.worker_manager.release, model_id)
        if ctx.sampler_backend() == "vllm":
          await SamplingStore(ctx.store.redis).delete_if_empty(meta.pool(model_id))
    return {"status": "ok"}

  @api.post("/api/v1/create_model_from_state")
  async def create_model_from_state(req: CreateModelFromStateRequest, request: Request) -> dict[str, Any]:
    state_path = req.state_path
    resolved_path = tinker_checkpoint_dir(state_path, tmp_dir=ctx.tmp_dir)
    if resolved_path is None and state_path.startswith("tinker://"):
      raise HTTPException(status_code=400, detail="Invalid checkpoint URI")
    if resolved_path is None:
      resolved_path = state_path if os.path.isabs(state_path) else os.path.join(ctx.tmp_dir, "checkpoints", state_path)
    try:
      checkpoint = await asyncio.to_thread(ctx.read_checkpoint_info, resolved_path)
      kind: FineTuningType = "lora" if checkpoint["is_lora"] else "full"
      if req.base_model is not None and req.base_model != checkpoint["base_model"]:
        raise ValueError("base_model does not match the checkpoint")
      req = req.model_copy(update={"base_model": checkpoint["base_model"]})
      model_id, meta = await extract_and_persist_model_metadata(ctx, req, request, checkpoint_kind=kind)
    except ValueError as exc:
      raise HTTPException(status_code=400, detail=str(exc)) from exc
    await bind_session(ctx, req.session_id, model_id)
    command = commands.CreateModelFromState(
      request_id=model_id,
      model_id=model_id,
      state_path=resolved_path,
      restore_optimizer=req.restore_optimizer,
      fine_tuning_type="full" if meta.fine_tuning_type == "full" else "lora",
    )
    req_id = (
      await launch_worker_and_enqueue(ctx, command, start_sampler=meta.recipe == "rl")
      if ctx.worker_manager is not None
      else await enqueue(ctx, command)
    )
    return {"request_id": req_id}

  @api.post("/api/v1/get_info")
  async def get_info(req: GetInfoRequest):
    model_id = req.model_id
    meta = await get_model_metadata(ctx.state, base_model_id_from_sampling_ref(model_id) or model_id) if model_id else None
    model_name = (meta.base_model if meta is not None else None) or get_default_model_name()
    if not model_name:
      raise HTTPException(status_code=404, detail="No base model is configured")
    lora_rank = meta.lora_config.rank if meta is not None and meta.fine_tuning_type == "lora" else 16
    return {
      "model_data": {"arch": "unknown", "model_name": model_name, "tokenizer_id": model_name},
      "model_id": model_id or "model-live-123",
      "is_lora": True,
      "lora_rank": lora_rank,
      "model_name": model_name,
      "type": "get_info",
    }

  @api.post("/api/v1/retrieve_future")
  async def retrieve_future(req: RetrieveFutureRequest, accept: str = Header(default="")):
    result = await ctx.store.get_result(req.request_id, timeout=60.0)
    if result is None:
      return JSONResponse(status_code=400, content={"type": "RequestFailedResponse", "error_message": "Future not found"})
    if isinstance(result, dict) and result.get("type") == "RequestFailedResponse":
      return JSONResponse(status_code=400, content=result)
    if isinstance(result, dict):
      if wire.PROTO in accept:
        encoded = wire.encode_future_result(result)
        if encoded is not None:
          return Response(content=encoded, media_type=wire.PROTO)
      decoded = wire.decode_future_result_json(result)
      return translate_future_result(decoded, tmp_dir=ctx.tmp_dir)
    return result

  # ***** training *****

  async def enqueue_forward_backward(parsed: ParsedForwardBackward, forward_only: bool) -> dict[str, str]:
    req_id = await enqueue(
      ctx,
      commands.ForwardBackward(
        request_id=new_request_id(),
        model_id=parsed.model_id,
        loss_fn=parsed.loss_fn,
        loss_config=parsed.loss_config,
        forward_only=forward_only,
        proto_bytes=parsed.proto_bytes,
      ),
    )
    return {"request_id": req_id}

  @api.post("/api/v1/forward")
  async def forward(parsed: Annotated[ParsedForwardBackward, Depends(parse_forward_backward_request)]):
    return await enqueue_forward_backward(parsed, forward_only=True)

  @api.post("/api/v1/forward_backward")
  async def forward_backward(parsed: Annotated[ParsedForwardBackward, Depends(parse_forward_backward_request)]):
    return await enqueue_forward_backward(parsed, forward_only=parsed.forward_only)

  @api.post("/api/v1/optim_step")
  async def optim_step(req: OptimStepRequest):
    req_id = await enqueue(ctx, commands.OptimStep(request_id=new_request_id(), model_id=req.model_id, adam_params=req.adam_params))
    return {"request_id": req_id}

  @api.post("/api/v1/save_weights_for_sampler")
  async def save_weights_for_sampler(req: SaveWeightsForSamplerRequest):
    model_id = req.model_id
    await ensure_sampler_launched(ctx, model_id)
    seq_id = req.sampling_session_seq_id
    if seq_id is None:
      seq_id = int(time.time() * 1000)
    alias = req.name or req.alias or req.path
    session_id = sampler_session_id(model_id, seq_id)
    req_id = await enqueue(
      ctx,
      commands.SaveWeightsForSampler(
        request_id=new_request_id(),
        model_id=model_id,
        alias=alias,
        path=sampler_weights_path(model_id, alias) if alias else None,
        sampling_session_id=session_id,
      ),
    )
    return {"request_id": req_id}

  @api.post("/api/v1/save_weights")
  async def save_weights(req: SaveWeightsRequest):
    model_id = req.model_id
    seq_id = req.seq_id
    if seq_id is None:
      seq_id = int(time.time() * 1000)
    alias = req.path or f"{model_id}-samp-{seq_id}"
    state_path = checkpoint_state_path(model_id, alias, tmp_dir=ctx.tmp_dir)
    req_id = await enqueue(
      ctx,
      commands.SaveState(request_id=new_request_id(), model_id=model_id, state_path=state_path, include_optimizer=True, kind="weights"),
    )
    return {"request_id": req_id}

  @api.post("/api/v1/load_weights")
  async def load_weights(req: LoadWeightsRequest):
    model_id = req.model_id
    state_path = req.path
    if state_path.startswith("tinker://") and tinker_checkpoint_dir(state_path, tmp_dir=ctx.tmp_dir) is None:
      raise HTTPException(status_code=400, detail=f"{state_path} is not a tinker://<model>/weights/<name> path")
    resolved_path = checkpoint_state_path(model_id, state_path, tmp_dir=ctx.tmp_dir)
    req_id = await enqueue(
      ctx,
      commands.LoadWeights(request_id=new_request_id(), model_id=model_id, state_path=resolved_path, restore_optimizer=req.optimizer),
    )
    return {"request_id": req_id}

  @api.post("/api/v1/weights_info")
  async def weights_info(req: WeightsInfoRequest):
    return await asyncio.to_thread(ctx.read_checkpoint_info, req.tinker_path)

  # ***** sampling *****

  @api.post("/api/v1/create_sampling_session")
  async def create_sampling_session(req: CreateSamplingSessionRequest):
    if req.model_path and req.model_path.startswith("tinker://"):
      sess_id = req.model_path
      target_model_id = req.model_path[len("tinker://") :].split("/")[0]
    elif req.base_model:
      sess_id = req.base_model
      target_model_id = req.base_model
    else:
      sess_id = req.model_id or "samp-session-live-123"
      target_model_id = sess_id

    await bind_session(ctx, req.session_id, target_model_id)
    if ctx.sampler_backend() == "vllm":
      await ensure_sampler_launched(ctx, target_model_id)
      return {"sampling_session_id": f"{sess_id}@{uuid.uuid4().hex}", "type": "create_sampling_session"}
    return {"sampling_session_id": sess_id, "type": "create_sampling_session"}

  @api.get("/api/v1/samplers/{sampler_id:path}")
  async def get_sampler(sampler_id: str):
    ref = sampling_ref(sampler_id)
    base_model_id = base_model_id_from_sampling_ref(ref)
    model_meta = await get_model_metadata(ctx.state, base_model_id) if base_model_id else None
    base_model = (model_meta.base_model if model_meta is not None else None) or base_model_id or get_default_model_name()
    if not base_model:
      raise HTTPException(status_code=404, detail=f"Unknown sampler {sampler_id}")
    return {
      "sampler_id": sampler_id,
      "base_model": base_model,
      "model_path": ref if ref.startswith("tinker://") else None,
    }

  @api.post("/api/v1/asample")
  async def asample(req: AsampleRequest):
    prompt = [token for chunk in req.prompt.get("chunks", []) for token in chunk.get("tokens", [])]
    params = req.sampling_params
    num_samples = req.num_samples
    model_id = req.model_id or req.sampling_session_id
    base_model_id = base_model_id_from_sampling_ref(model_id)
    lookup_id = base_model_id or model_id

    if ctx.sampler_backend() == "torch":
      req_id = await enqueue(
        ctx,
        commands.Sample(
          request_id=new_request_id(),
          model_id=lookup_id,
          prompt_tokens=prompt,
          max_tokens=params.max_tokens,
          temperature=params.temperature,
          num_samples=num_samples,
          prompt_logprobs=req.prompt_logprobs,
        ),
      )
      return {"request_id": req_id, "sample_sequence_ids": sample_sequence_ids(req_id, num_samples)}

    req_id = await accept_pool_sample(ctx, req, prompt)
    return {"request_id": req_id, "sample_sequence_ids": sample_sequence_ids(req_id, num_samples)}

  # ***** inspection *****

  @api.get("/api/v1/openrl/models/{model_id}")
  async def openrl_model_status(model_id: str):
    status_ctx = openrl_status.StatusContext(
      state=ctx.state,
      registry=ctx.session_registry,
      manager=ctx.worker_manager,
      worker_launcher=worker_launcher_mode(ctx),
    )
    result = await openrl_status.model_status(status_ctx, model_id)
    if result is None:
      raise HTTPException(status_code=404, detail=f"Unknown model: {model_id}")
    return result

  @api.get("/api/v1/list_adapters")
  async def list_adapters():
    peft_dir = os.path.join(ctx.tmp_dir, "peft")
    adapters = []
    if os.path.exists(peft_dir):
      for entry in sorted(os.scandir(peft_dir), key=lambda e: e.stat().st_ctime, reverse=True):
        if not entry.is_dir():
          continue
        info = {"model_id": entry.name, "created_at": entry.stat().st_ctime, "timestamp": entry.stat().st_ctime, "alias": None}
        metadata_path = os.path.join(entry.path, "metadata.json")
        if os.path.exists(metadata_path):
          try:
            with open(metadata_path) as f:
              info.update(json.load(f))
          except Exception:
            pass
        adapters.append(info)
    return {"adapters": adapters}

  @api.post("/api/v1/telemetry")
  async def telemetry_endpoint(body: dict):
    run_records.sdk_events(body)
    return {"status": "accepted"}

  return api
