"""Open-RL FastAPI entrypoint and backward-compatible module facade over `server.api.gateway`.

Production deployments run `uvicorn server.api_server:app`. Programmatic callers and tests
can also build isolated apps directly via `server.api.gateway.create_app(...)`.
"""

from __future__ import annotations

import asyncio
import os
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

from fastapi import Request

from server.api import gateway
from server.api.gateway import (
  EXAMPLE_MODELS,
  RECONCILE_CONCURRENCY,
  RELEASES_URL,
  SAVED_WEIGHTS_WAIT_SECONDS,
  SESSION_REAP_INTERVAL_SEC,
  SETTINGS_PREFIX,
  USER_METADATA_DOC,
  AsampleRequest,
  CreateModelFromStateRequest,
  CreateModelRequest,
  CreateSamplingSessionRequest,
  CreateSessionRequest,
  FilterNoisyEndpoints,
  ForwardBackwardInput,
  ForwardBackwardRequest,
  GatewayState,
  GetInfoRequest,
  LoadWeightsRequest,
  ModelRequest,
  OptimStepRequest,
  RetrieveFutureRequest,
  SamplingParams,
  SaveWeightsForSamplerRequest,
  SaveWeightsRequest,
  SessionHeartbeatRequest,
  Settings,
  WeightsInfoRequest,
  base_model_id_from_sampling_ref,
  check_lora_jobs_per_runtime,
  content_type,
  create_app,
  env_int,
  get_default_model_name,
  get_sampler_backend,
  http_error,
  is_fft_enabled,
  is_single_process_mode,
  new_request_id,
  parse_bool,
  parse_trainer_backend,
  request_validation_error,
  resolve_settings,
  runtime_jobs_key,
  sample_sequence_ids,
  sampler_session_id,
  sampler_weights_path,
  sampling_ref,
  settings_schema,
  tag_metadata,
  visible,
)
from server.model_metadata import TrainingModelMetadata
from server.sampling_store import SamplingStore
from server.session_registry import SessionRegistry
from server.store import get_state_store, get_store
from server.worker_manager import WorkerManager
from training.commands import Command
from training.types import FineTuningType

# Module-level state kept for backward compatibility with existing callers and tests
# that monkeypatch `api_server.store`, `api_server.state`, `api_server.worker_manager`, etc.
store = get_store()
state = get_state_store()
worker_manager: WorkerManager | None = None
session_registry = SessionRegistry(state)
owner_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
slot_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
TMP_DIR = os.getenv("OPEN_RL_TMP_DIR", "/tmp/open-rl")
OPEN_RL_VERSION = os.getenv("OPEN_RL_VERSION") or "dev"

gateway.configure_telemetry()


def check_sampler_backend() -> None:
  gateway.check_sampler_backend(sampler_backend=get_sampler_backend())


def resolve_sampler_weights_path(model_id: str) -> str:
  return gateway.resolve_sampler_weights_path(model_id, tmp_dir=TMP_DIR)


def tinker_checkpoint_dir(path: str) -> str | None:
  return gateway.tinker_checkpoint_dir(path, tmp_dir=TMP_DIR)


def checkpoint_state_path(model_id: str, name: str) -> str:
  return gateway.checkpoint_state_path(model_id, name, tmp_dir=TMP_DIR)


def tinker_state_path(state_path: str) -> str:
  return gateway.tinker_state_path(state_path, tmp_dir=TMP_DIR)


def translate_future_result(result: dict) -> dict:
  return gateway.translate_future_result(result, tmp_dir=TMP_DIR)


def checkpoint_info(path: str) -> dict[str, Any]:
  return gateway.checkpoint_info(path, tmp_dir=TMP_DIR)


class _ModuleGatewayState(GatewayState):
  """Delegates runtime state lookups to `server.api_server` module globals."""

  def __init__(self) -> None:
    pass

  @property
  def store(self):
    return store

  @store.setter
  def store(self, value) -> None:
    global store
    store = value

  @property
  def state(self):
    return state

  @state.setter
  def state(self, value) -> None:
    global state
    state = value

  @property
  def session_registry(self):
    return session_registry

  @session_registry.setter
  def session_registry(self, value) -> None:
    global session_registry
    session_registry = value

  @property
  def worker_manager(self):
    return worker_manager

  @worker_manager.setter
  def worker_manager(self, value) -> None:
    global worker_manager
    worker_manager = value

  @property
  def owner_locks(self):
    return owner_locks

  @owner_locks.setter
  def owner_locks(self, value) -> None:
    global owner_locks
    owner_locks = value

  @property
  def slot_locks(self):
    return slot_locks

  @slot_locks.setter
  def slot_locks(self, value) -> None:
    global slot_locks
    slot_locks = value

  @property
  def tmp_dir(self) -> str:
    return TMP_DIR

  @property
  def version(self) -> str:
    return OPEN_RL_VERSION

  @property
  def saved_weights_wait_seconds(self) -> float:
    return SAVED_WEIGHTS_WAIT_SECONDS

  @property
  def session_reap_interval_sec(self) -> float:
    return SESSION_REAP_INTERVAL_SEC

  @property
  def reconcile_concurrency(self) -> int:
    return RECONCILE_CONCURRENCY

  def sampler_backend(self) -> str:
    return get_sampler_backend()

  def single_process_mode(self) -> bool:
    return is_single_process_mode()

  def read_checkpoint_info(self, path: str) -> dict[str, Any]:
    return checkpoint_info(path)

  def log_exception(self) -> None:
    traceback.print_exc()


_ctx = _ModuleGatewayState()
lifespan = gateway.make_lifespan(_ctx)
app = create_app(ctx=_ctx, instrument=True)


async def bind_session(session_id: str | None, model_id: str) -> None:
  await gateway.bind_session(_ctx, session_id, model_id)


async def reap_owner(owner: str) -> None:
  await gateway.reap_owner(_ctx, owner)


async def job_is_live(model_id: str) -> bool:
  return await gateway.job_is_live(_ctx, model_id)


async def prune_runtime_jobs(runtime: str) -> int:
  return await gateway.prune_runtime_jobs(_ctx, runtime)


async def persist_on_runtime_slot(model_id: str, meta: TrainingModelMetadata) -> None:
  await gateway.persist_on_runtime_slot(_ctx, model_id, meta)


async def _extract_and_persist_model_metadata(
  req: CreateModelRequest | CreateModelFromStateRequest,
  request: Request | None = None,
  checkpoint_kind: FineTuningType | None = None,
) -> tuple[str, TrainingModelMetadata]:
  return await gateway.extract_and_persist_model_metadata(_ctx, req, request, checkpoint_kind=checkpoint_kind)


async def _resolve_active_set_id(model_id: str | None) -> str | None:
  return await gateway.resolve_active_set_id(_ctx, model_id)


async def enqueue(command: Command) -> str:
  return await gateway.enqueue(_ctx, command)


async def launch_worker_and_enqueue(command: Command, start_sampler: bool = False) -> str:
  return await gateway.launch_worker_and_enqueue(_ctx, command, start_sampler=start_sampler)


async def ensure_sampler_launched(model_id: str) -> None:
  await gateway.ensure_sampler_launched(_ctx, model_id)


async def reap_dead_sessions() -> None:
  await gateway.reap_dead_sessions(_ctx)


async def reconcile_pools() -> None:
  await gateway.reconcile_pools(_ctx)


async def reconcile_job(sampling: SamplingStore, pool: str, job: str) -> None:
  await gateway.reconcile_job(_ctx, sampling, pool, job)


async def sweep_sample_deadlines() -> None:
  await gateway.sweep_sample_deadlines(_ctx)


async def wait_until_visible(path: Path) -> bool:
  return await gateway.wait_until_visible(_ctx, path)


async def accept_pool_sample(req: AsampleRequest, prompt: list[int]) -> str:
  return await gateway.accept_pool_sample(_ctx, req, prompt)


def worker_launcher_mode() -> str:
  return gateway.worker_launcher_mode(_ctx)


def openrl_info() -> dict[str, Any]:
  return gateway.openrl_info(_ctx)


__all__ = [
  "EXAMPLE_MODELS",
  "OPEN_RL_VERSION",
  "RECONCILE_CONCURRENCY",
  "RELEASES_URL",
  "SAVED_WEIGHTS_WAIT_SECONDS",
  "SESSION_REAP_INTERVAL_SEC",
  "SETTINGS_PREFIX",
  "TMP_DIR",
  "USER_METADATA_DOC",
  "AsampleRequest",
  "CreateModelFromStateRequest",
  "CreateModelRequest",
  "CreateSamplingSessionRequest",
  "CreateSessionRequest",
  "FilterNoisyEndpoints",
  "ForwardBackwardInput",
  "ForwardBackwardRequest",
  "GatewayState",
  "GetInfoRequest",
  "LoadWeightsRequest",
  "ModelRequest",
  "OptimStepRequest",
  "RetrieveFutureRequest",
  "SamplingParams",
  "SaveWeightsForSamplerRequest",
  "SaveWeightsRequest",
  "SessionHeartbeatRequest",
  "Settings",
  "WeightsInfoRequest",
  "accept_pool_sample",
  "app",
  "base_model_id_from_sampling_ref",
  "bind_session",
  "check_lora_jobs_per_runtime",
  "check_sampler_backend",
  "checkpoint_info",
  "checkpoint_state_path",
  "content_type",
  "create_app",
  "enqueue",
  "ensure_sampler_launched",
  "env_int",
  "get_default_model_name",
  "get_sampler_backend",
  "get_state_store",
  "get_store",
  "http_error",
  "is_fft_enabled",
  "is_single_process_mode",
  "job_is_live",
  "launch_worker_and_enqueue",
  "lifespan",
  "new_request_id",
  "openrl_info",
  "owner_locks",
  "parse_bool",
  "parse_trainer_backend",
  "persist_on_runtime_slot",
  "prune_runtime_jobs",
  "reap_dead_sessions",
  "reap_owner",
  "reconcile_job",
  "reconcile_pools",
  "request_validation_error",
  "resolve_sampler_weights_path",
  "resolve_settings",
  "runtime_jobs_key",
  "sample_sequence_ids",
  "sampler_session_id",
  "sampler_weights_path",
  "sampling_ref",
  "session_registry",
  "settings_schema",
  "slot_locks",
  "state",
  "store",
  "sweep_sample_deadlines",
  "tag_metadata",
  "tinker_checkpoint_dir",
  "tinker_state_path",
  "translate_future_result",
  "visible",
  "wait_until_visible",
  "worker_launcher_mode",
  "worker_manager",
]
