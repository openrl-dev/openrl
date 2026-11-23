"""The Tinker wire format for OpenRL.

`forward_backward` requests, `ForwardBackwardOutput`, and `SampleResponse` travel as binary
protobuf end to end: the gateway validates and forwards protobuf bytes untouched, and workers
read or write tensors straight to them via lazy-imported NumPy buffers (zero `.tolist()` JSON
tensor overhead). Every other API call stays JSON.

CRITICAL: This module MUST stay importable without `numpy` installed (`Dockerfile.api_server`
does not ship `numpy`). Gateway functions use stdlib `array`/`struct` + vendored protobuf only;
worker helpers (`datums`, `output`, `sample_output`, `pack_topk`, `pack_stability`,
`unpack_stability`) lazy-import `numpy` when called on workers.
"""

from __future__ import annotations

import array
import math
import struct
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from google.protobuf.message import DecodeError

from server.proto import tinker_public_pb2 as pb

PROTO = "application/x-protobuf"
PROTO_CONTENT_TYPE = PROTO
JSON = "application/json"
MASK_LOGPROB = -99999.0  # SDK sentinel logprob for an unscored top-k slot

# Result types (as stored by workers) that the Tinker SDK expects as protobuf.
PROTO_RESULT_TYPES = frozenset({"forward_backward_completed", "forward_backward", "sample_completed", "sample"})


class ProtoDecodeError(ValueError):
  """A protobuf request that failed to parse or asks for something workers cannot run."""


_LITTLE_ENDIAN = sys.byteorder == "little"

# Little-endian wire layout: int32 tokens, int64/float32 tensors.
_INT32 = "i"
_INT64 = "q"
_FLOAT32 = "f"
for _code, _size in ((_INT32, 4), (_INT64, 8), (_FLOAT32, 4)):
  if array.array(_code).itemsize != _size:  # pragma: no cover - platform sanity check
    raise RuntimeError(f"array typecode {_code!r} is {array.array(_code).itemsize} bytes on this platform, expected {_size}")

# Proto dtype -> (array typecode, public JSON dtype label, byte itemsize).
# int32 widens to int64 on the JSON path, where every integer tensor is labelled "int64".
_TENSOR_DTYPES: dict[int, tuple[str, str, int]] = {
  pb.DTYPE_FLOAT32: (_FLOAT32, "float32", 4),
  pb.DTYPE_INT64: (_INT64, "int64", 8),
  pb.DTYPE_INT32: (_INT32, "int64", 4),
}

_STOP_REASONS: dict[str, int] = {
  "stop": pb.STOP_REASON_STOP,
  "length": pb.STOP_REASON_LENGTH,
}

_STOP_REASON_NAMES: dict[int, str] = {
  pb.STOP_REASON_STOP: "stop",
  pb.STOP_REASON_LENGTH: "length",
}


def _np():
  import numpy as np

  return np


def _from_bytes(code: str, raw: bytes) -> array.array:
  arr = array.array(code)
  if len(raw) % arr.itemsize:
    raise ProtoDecodeError(f"tensor payload of {len(raw)} bytes is not a multiple of {arr.itemsize}")
  arr.frombytes(raw)
  if not _LITTLE_ENDIAN:
    arr.byteswap()
  return arr


def _to_bytes(code: str, values: Iterable[Any]) -> bytes:
  arr = array.array(code, values)
  if not _LITTLE_ENDIAN:
    arr.byteswap()
  return arr.tobytes()


def _float_or_nan(value: Any) -> float:
  # Samplers emit None for positions without a logprob (e.g. first prompt token).
  # The protobuf wire is a flat float32 buffer, so keep the slot as NaN.
  return math.nan if value is None else float(value)


# ***** binary result envelope helpers *****


def proto_result(result_type: str, proto_bytes: bytes, *, metrics: dict[str, float] | None = None) -> dict[str, Any]:
  """Wrap raw protobuf output bytes in a store/telemetry-compatible result envelope."""
  env: dict[str, Any] = {"type": result_type, "proto_bytes": bytes(proto_bytes)}
  if metrics is not None:
    env["metrics"] = metrics
  return env


def is_proto_result(result: Any) -> bool:
  """Whether `result` is a binary protobuf result envelope produced by a worker."""
  return isinstance(result, dict) and isinstance(result.get("proto_bytes"), (bytes, bytearray, memoryview))


def proto_result_bytes(result: dict[str, Any]) -> bytes:
  """Extract the raw protobuf payload bytes from a `proto_result` envelope."""
  return bytes(result["proto_bytes"])


# ***** spans & worker dataclasses *****


@dataclass(frozen=True)
class Span:
  kind: str  # "prompt" | "sampled"
  sequence_id: str
  offset: int
  length: int


@dataclass(frozen=True)
class Datum:
  tokens: Any  # np.ndarray[int32]
  inputs: dict[str, Any]  # dict[str, np.ndarray]
  model_spans: tuple[Span, ...] = ()
  loss_spans: tuple[Span, ...] = ()


@dataclass(frozen=True)
class SampleOut:
  stop: bool
  tokens: Any  # np.ndarray[int32] or sequence of int
  logprobs: Any = None  # np.ndarray[float32] or None
  topk: tuple[Any, Any] | None = None


def parse_spans(spans) -> tuple[Span, ...]:
  out = []
  for s in spans:
    if (which := s.WhichOneof("span")) in ("sampled_tokens", "prompt_tokens"):
      p = getattr(s, which)
      out.append(Span("sampled" if which == "sampled_tokens" else "prompt", p.sequence_id, p.offset, p.length))
  return tuple(out)


def referenced_sequences(req: pb.ForwardBackwardRequest) -> set[str]:
  return {s.sampled_tokens.sequence_id for d in req.data for s in d.loss_fn_input_spans if s.WhichOneof("span") == "sampled_tokens"}


# ***** requests (gateway — zero numpy) *****


def _validate_chunk(chunk: pb.Chunk) -> None:
  kind = chunk.WhichOneof("chunk")
  if kind != "encoded_text":
    raise ProtoDecodeError(f"model_input chunk of kind {kind or 'unset'!r} is not supported; this server only runs text models")
  if len(chunk.encoded_text.tokens) % 4:
    raise ProtoDecodeError(f"encoded_text.tokens of {len(chunk.encoded_text.tokens)} bytes is not a multiple of 4")


def _validate_tensor(name: str, tensor: pb.Tensor) -> None:
  if tensor.WhichOneof("encoding") == "sparse_csr":
    raise ProtoDecodeError(f"loss_fn_inputs[{name!r}] is a sparse tensor; only dense tensors are supported")
  spec = _TENSOR_DTYPES.get(tensor.dtype)
  if spec is None:
    raise ProtoDecodeError(f"loss_fn_inputs[{name!r}] has dtype {pb.DType.Name(tensor.dtype)}; only float32, int64, and int32 are supported")
  _, _, itemsize = spec
  if len(tensor.dense) % itemsize:
    raise ProtoDecodeError(f"loss_fn_inputs[{name!r}] payload of {len(tensor.dense)} bytes is not a multiple of {itemsize}")
  if len(tensor.shape) > 1:
    raise ProtoDecodeError(f"loss_fn_inputs[{name!r}] expected 1-D tensor, got shape {list(tensor.shape)}")


def parse_forward_backward(raw: bytes, *, validate: bool = True) -> pb.ForwardBackwardRequest:
  """Parse (and optionally validate) a protobuf `ForwardBackwardRequest` without expanding tensors."""
  msg = pb.ForwardBackwardRequest()
  try:
    msg.ParseFromString(raw)
  except DecodeError as exc:
    raise ProtoDecodeError(f"malformed ForwardBackwardRequest protobuf: {exc}") from exc
  if validate:
    for datum in msg.data:
      for chunk in datum.model_input:
        _validate_chunk(chunk)
      for name, tensor in datum.loss_fn_inputs.items():
        _validate_tensor(name, tensor)
  return msg


parse = parse_forward_backward


def cost(req: pb.ForwardBackwardRequest) -> int:
  """Total input tokens a `ForwardBackwardRequest` trains on."""
  return sum(len(chunk.encoded_text.tokens) // 4 for d in req.data for chunk in d.model_input)


def loss_config(req: pb.ForwardBackwardRequest) -> dict[str, Any]:
  # v2 carries both numbers and strings and is preferred when present; the
  # SDK dual-writes floats into the legacy map for older servers.
  if req.loss_fn_config_v2:
    out: dict[str, Any] = {}
    for key, value in req.loss_fn_config_v2.items():
      out[key] = value.text if value.WhichOneof("value") == "text" else value.number
    return out
  return dict(req.loss_fn_config)


def _decode_tensor(name: str, tensor: pb.Tensor) -> dict[str, Any]:
  _validate_tensor(name, tensor)
  code, label, _ = _TENSOR_DTYPES[tensor.dtype]
  data = _from_bytes(code, tensor.dense).tolist()
  shape = list(tensor.shape) or [len(data)]
  return {"data": data, "dtype": label, "shape": shape}


def _decode_chunk(chunk: pb.Chunk) -> dict[str, Any]:
  _validate_chunk(chunk)
  return {"type": "encoded_text", "tokens": _from_bytes(_INT32, chunk.encoded_text.tokens).tolist()}


def decode_forward_backward(body: bytes) -> dict[str, Any]:
  """Decode a protobuf `ForwardBackwardRequest` into the legacy JSON request shape."""
  msg = parse_forward_backward(body, validate=False)
  data = []
  for datum in msg.data:
    data.append(
      {
        "model_input": {"chunks": [_decode_chunk(chunk) for chunk in datum.model_input]},
        "loss_fn_inputs": {name: _decode_tensor(name, tensor) for name, tensor in datum.loss_fn_inputs.items()},
      }
    )
  return {
    "model_id": msg.model_id,
    "seq_id": msg.seq_id,
    "forward_only": msg.forward_only,
    "forward_backward_input": {
      "data": data,
      "loss_fn": msg.loss_fn,
      "loss_fn_config": loss_config(msg),
    },
  }


def _infer_tensor_dtype(name: str, raw_tensor: Any) -> tuple[int, str, Sequence[Any], list[int] | None]:
  if isinstance(raw_tensor, dict):
    values = raw_tensor.get("data") or []
    shape = raw_tensor.get("shape")
    dtype_label = str(raw_tensor.get("dtype") or "").lower()
    if dtype_label == "float32":
      return pb.DTYPE_FLOAT32, _FLOAT32, values, shape
    if dtype_label == "int64":
      return pb.DTYPE_INT64, _INT64, values, shape
    if dtype_label == "int32":
      return pb.DTYPE_INT32, _INT32, values, shape
  else:
    values = raw_tensor or []
    shape = None
  # Infer dtype when omitted on bare JSON lists.
  if any(isinstance(v, float) for v in values):
    return pb.DTYPE_FLOAT32, _FLOAT32, values, shape
  if values and all(isinstance(v, int) and not isinstance(v, bool) for v in values):
    if "token" in name or name in ("target_tokens", "input_ids"):
      return pb.DTYPE_INT64, _INT64, values, shape
    return pb.DTYPE_FLOAT32, _FLOAT32, values, shape
  if "token" in name or name in ("target_tokens", "input_ids"):
    return pb.DTYPE_INT64, _INT64, values, shape
  return pb.DTYPE_FLOAT32, _FLOAT32, values, shape


def json_to_forward_backward_proto(payload: dict[str, Any], *, forward_only: bool | None = None) -> bytes:
  """Encode a JSON `ForwardBackwardRequest` dict into binary protobuf `ForwardBackwardRequest` bytes (no numpy)."""
  fwd_input = payload.get("forward_backward_input") or payload.get("forward_input") or {}
  msg = pb.ForwardBackwardRequest(
    model_id=str(payload.get("model_id") or ""),
    seq_id=int(payload.get("seq_id") or 0),
    loss_fn=str(fwd_input.get("loss_fn") or "cross_entropy"),
    forward_only=bool(payload.get("forward_only", False) if forward_only is None else forward_only),
  )
  for key, value in (fwd_input.get("loss_fn_config") or {}).items():
    if isinstance(value, str):
      msg.loss_fn_config_v2[key].text = value
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
      num = float(value)
      msg.loss_fn_config[key] = num
      msg.loss_fn_config_v2[key].number = num
  for raw_datum in fwd_input.get("data") or []:
    datum = msg.data.add()
    model_input = raw_datum.get("model_input")
    if isinstance(model_input, dict):
      chunks = model_input.get("chunks") or []
      for chunk in chunks:
        tokens = chunk.get("tokens") or []
        datum.model_input.add().encoded_text.tokens = _to_bytes(_INT32, tokens)
    elif isinstance(model_input, (list, tuple)):
      datum.model_input.add().encoded_text.tokens = _to_bytes(_INT32, model_input)
    for name, raw_tensor in (raw_datum.get("loss_fn_inputs") or {}).items():
      dtype_pb, code, values, shape = _infer_tensor_dtype(name, raw_tensor)
      tensor = datum.loss_fn_inputs[name]
      tensor.dtype = dtype_pb
      if code == _FLOAT32:
        tensor.dense = _to_bytes(_FLOAT32, (_float_or_nan(v) for v in values))
      else:
        tensor.dense = _to_bytes(code, (int(v) for v in values))
      if shape is not None:
        tensor.shape.extend(int(s) for s in shape)
      else:
        tensor.shape.append(len(values))
  return msg.SerializeToString()


# ***** worker request helpers (lazy numpy, zero .tolist()) *****


def to_numpy(dtype: int, raw: bytes):
  np = _np()
  dtypes = {
    pb.DTYPE_FLOAT32: np.float32,
    pb.DTYPE_INT64: np.int64,
    pb.DTYPE_INT32: np.int32,
  }
  if dtype not in dtypes:
    raise ProtoDecodeError(f"tensors of {pb.DType.Name(dtype)} are not supported")
  np_dtype = dtypes[dtype]
  if len(raw) % np.dtype(np_dtype).itemsize:
    raise ProtoDecodeError(f"tensor payload of {len(raw)} bytes is not a multiple of {np.dtype(np_dtype).itemsize}")
  arr = np.frombuffer(raw, np_dtype)
  if not _LITTLE_ENDIAN:
    arr = arr.byteswap()
  return arr.copy()


def chunk_tokens(chunk: pb.Chunk):
  np = _np()
  _validate_chunk(chunk)
  arr = np.frombuffer(chunk.encoded_text.tokens, np.int32)
  if not _LITTLE_ENDIAN:
    arr = arr.byteswap()
  return arr.copy()


def loss_input(tensor: pb.Tensor):
  if tensor.WhichOneof("encoding") == "sparse_csr":
    raise ProtoDecodeError("sparse tensors are not supported")
  if len(tensor.shape) > 1:
    raise ProtoDecodeError(f"expected 1-D tensor, got shape {list(tensor.shape)}")
  return to_numpy(tensor.dtype, tensor.dense)


def datums(req: pb.ForwardBackwardRequest | bytes) -> list[Datum]:
  """Decode `ForwardBackwardRequest` (or raw bytes) directly into NumPy `Datum` objects without `.tolist()`."""
  np = _np()
  msg = parse_forward_backward(req, validate=False) if isinstance(req, (bytes, bytearray, memoryview)) else req
  out: list[Datum] = []
  for d in msg.data:
    tokens = [chunk_tokens(chunk) for chunk in d.model_input]
    inputs = {name: loss_input(tensor) for name, tensor in d.loss_fn_inputs.items()}
    out.append(
      Datum(
        np.concatenate(tokens) if tokens else np.zeros(0, np.int32),
        inputs,
        parse_spans(d.model_input_spans),
        parse_spans(d.loss_fn_input_spans),
      )
    )
  return out


def pack_stability(ids, lps) -> bytes:
  np = _np()
  ids_arr = np.asarray(ids, dtype=np.int32)
  lps_arr = np.asarray(lps, dtype=np.float32)
  return struct.pack("<II", *ids_arr.shape) + ids_arr.tobytes() + lps_arr.tobytes()


def unpack_stability(raw: bytes):
  np = _np()
  n, k = struct.unpack("<II", raw[:8])
  ids = np.frombuffer(raw, np.int32, n * k, 8).reshape(n, k).copy()
  lps = np.frombuffer(raw, np.float32, n * k, 8 + 4 * n * k).reshape(n, k).copy()
  return ids, lps


# ***** worker output encoders (lazy numpy, zero .tolist()) *****


def output(
  logprobs: Sequence[Any],
  metrics: dict[str, float],
  *,
  loss_fn_output_type: str = "ArrayRecord",
) -> bytes:
  """Pack per-datum logprobs directly into binary protobuf `ForwardBackwardOutput` bytes."""
  np = _np()
  out = pb.ForwardBackwardOutput(loss_fn_output_type=loss_fn_output_type, metrics={k: float(v) for k, v in metrics.items()})
  if logprobs:
    arrays = [np.asarray(x, dtype=np.float32).reshape(-1) for x in logprobs]
    field = out.loss_fn_outputs.add(num_datums=len(arrays)).fields["logprobs"]
    field.dtype = pb.DTYPE_FLOAT32
    field.data = np.concatenate(arrays).astype(np.float32, copy=False).tobytes() if arrays else b""
    field.offsets = np.cumsum([0] + [4 * len(a) for a in arrays], dtype=np.int64).tobytes()
  return out.SerializeToString()


def pack_topk(dst, topk: tuple[Any, Any]) -> None:
  np = _np()
  tk_ids, tk_lps = topk
  ids_arr = np.asarray(tk_ids, dtype=np.int32)
  lps_arr = np.asarray(tk_lps, dtype=np.float32)
  dst.token_ids = ids_arr.tobytes()
  dst.logprobs = lps_arr.tobytes()
  dst.k = int(ids_arr.shape[-1])
  dst.length = int(ids_arr.shape[0])


def sample_output(
  sequences: Sequence[SampleOut],
  prompt_logprobs: Any = None,
  topk_prompt: tuple[Any, Any] | None = None,
  prompt_cache_hit_tokens: int = 0,
) -> bytes:
  """Pack sampled sequences directly into binary protobuf `SampleResponse` bytes."""
  np = _np()
  out = pb.SampleResponse()
  for s in sequences:
    seq = out.sequences.add()
    seq.stop_reason = pb.STOP_REASON_STOP if s.stop else pb.STOP_REASON_LENGTH
    seq.tokens = np.asarray(s.tokens, dtype=np.int32).reshape(-1).tobytes()
    if s.logprobs is not None:
      lps = np.asarray(s.logprobs, dtype=np.float32).reshape(-1)
      if lps.size > 0:
        seq.logprobs = lps.tobytes()
    if s.topk is not None:
      pack_topk(seq.topk_sampled_logprobs, s.topk)
  if prompt_logprobs is not None:
    plps = np.asarray(prompt_logprobs, dtype=np.float32).reshape(-1)
    if plps.size > 0:
      out.prompt_logprobs = plps.tobytes()
  if topk_prompt is not None:
    pack_topk(out.topk_prompt_logprobs, topk_prompt)
  if prompt_cache_hit_tokens:
    out.prompt_cache_hit_tokens = int(prompt_cache_hit_tokens)
  return out.SerializeToString()


# ***** gateway response conversions (stdlib array only, zero numpy) *****


def encode_sample_response(result: dict[str, Any]) -> bytes:
  """Encode a sampler result dict (or pass through a `proto_result` envelope) as `SampleResponse` bytes."""
  if is_proto_result(result):
    return proto_result_bytes(result)
  msg = pb.SampleResponse()
  for seq in result.get("sequences") or []:
    out = msg.sequences.add()
    out.stop_reason = _STOP_REASONS.get(str(seq.get("stop_reason")), pb.STOP_REASON_LENGTH)
    out.tokens = _to_bytes(_INT32, seq.get("tokens") or [])
    logprobs = seq.get("logprobs")
    if logprobs:
      out.logprobs = _to_bytes(_FLOAT32, (_float_or_nan(v) for v in logprobs))
  prompt_logprobs = result.get("prompt_logprobs")
  if prompt_logprobs:
    msg.prompt_logprobs = _to_bytes(_FLOAT32, (_float_or_nan(v) for v in prompt_logprobs))
  msg.prompt_cache_hit_tokens = int(result.get("prompt_cache_hit_tokens") or 0)
  return msg.SerializeToString()


def _tensor_parts(value: Any) -> tuple[Sequence[Any], str, list[int] | None]:
  if isinstance(value, dict):
    data = value.get("data") or []
    return data, str(value.get("dtype") or "float32"), value.get("shape")
  return value or [], "float32", None


def encode_forward_backward_output(result: dict[str, Any]) -> bytes:
  """Encode a trainer result dict (or pass through a `proto_result` envelope) as `ForwardBackwardOutput` bytes."""
  if is_proto_result(result):
    return proto_result_bytes(result)
  msg = pb.ForwardBackwardOutput()
  msg.loss_fn_output_type = str(result.get("loss_fn_output_type") or "ArrayRecord")
  for key, value in (result.get("metrics") or {}).items():
    msg.metrics[key] = float(value)

  outputs: list[dict[str, Any]] = result.get("loss_fn_outputs") or []
  if not outputs:
    return msg.SerializeToString()

  record = msg.loss_fn_outputs.add()
  record.num_datums = len(outputs)
  names: list[str] = []
  for out_item in outputs:
    for name in out_item:
      if name not in names:
        names.append(name)

  for name in names:
    buf = bytearray()
    offsets = [0]
    label = "float32"
    trailing: list[int] = []
    for index, out_item in enumerate(outputs):
      data, dtype, shape = _tensor_parts(out_item.get(name))
      if index == 0:
        label = dtype
        if shape and len(shape) > 1:
          trailing = [int(s) for s in shape[1:]]
      code = _INT64 if label == "int64" else _FLOAT32
      buf += _to_bytes(code, data if label == "int64" else (_float_or_nan(v) for v in data))
      offsets.append(len(buf))
    field = record.fields[name]
    field.data = bytes(buf)
    field.offsets = _to_bytes(_INT64, offsets)
    field.dtype = pb.DTYPE_INT64 if label == "int64" else pb.DTYPE_FLOAT32
    field.trailing_shape.extend(trailing)
  return msg.SerializeToString()


def forward_backward_output_to_json(raw: bytes, *, result_type: str = "forward_backward_completed") -> dict[str, Any]:
  """Decode binary `ForwardBackwardOutput` bytes into the JSON response shape (gateway edge, no numpy)."""
  msg = pb.ForwardBackwardOutput()
  try:
    msg.ParseFromString(raw)
  except DecodeError as exc:
    raise ProtoDecodeError(f"malformed ForwardBackwardOutput protobuf: {exc}") from exc

  loss_fn_outputs: list[dict[str, Any]] = []
  for record in msg.loss_fn_outputs:
    n = int(record.num_datums)
    batch_items: list[dict[str, Any]] = [{} for _ in range(n)]
    for name, field in record.fields.items():
      spec = _TENSOR_DTYPES.get(field.dtype)
      if spec is None:
        raise ProtoDecodeError(f"unsupported output dtype {pb.DType.Name(field.dtype)}")
      code, label, itemsize = spec
      offsets = _from_bytes(_INT64, field.offsets).tolist()
      trailing = list(field.trailing_shape)
      trailing_prod = math.prod(trailing) if trailing else 1
      for idx in range(n):
        start, end = offsets[idx], offsets[idx + 1]
        vals = _from_bytes(code, field.data[start:end]).tolist()
        leading = len(vals) // trailing_prod if trailing_prod else len(vals)
        batch_items[idx][name] = {"data": vals, "dtype": label, "shape": [leading, *trailing]}
    loss_fn_outputs.extend(batch_items)

  return {
    "type": result_type,
    "loss_fn_output_type": msg.loss_fn_output_type or "ArrayRecord",
    "metrics": dict(msg.metrics),
    "loss_fn_outputs": loss_fn_outputs,
  }


def sample_response_to_json(raw: bytes, *, result_type: str = "sample_completed") -> dict[str, Any]:
  """Decode binary `SampleResponse` bytes into the JSON response shape (gateway edge, no numpy)."""
  msg = pb.SampleResponse()
  try:
    msg.ParseFromString(raw)
  except DecodeError as exc:
    raise ProtoDecodeError(f"malformed SampleResponse protobuf: {exc}") from exc

  sequences: list[dict[str, Any]] = []
  for seq in msg.sequences:
    item: dict[str, Any] = {
      "tokens": _from_bytes(_INT32, seq.tokens).tolist(),
      "stop_reason": _STOP_REASON_NAMES.get(seq.stop_reason, "length"),
    }
    if seq.HasField("logprobs") or seq.logprobs:
      item["logprobs"] = [None if math.isnan(v) else v for v in _from_bytes(_FLOAT32, seq.logprobs)]
    sequences.append(item)

  out: dict[str, Any] = {"type": result_type, "sequences": sequences}
  if msg.HasField("prompt_logprobs") or msg.prompt_logprobs:
    out["prompt_logprobs"] = [None if math.isnan(v) else v for v in _from_bytes(_FLOAT32, msg.prompt_logprobs)]
  if msg.prompt_cache_hit_tokens:
    out["prompt_cache_hit_tokens"] = int(msg.prompt_cache_hit_tokens)
  return out


def encode_future_result(result: dict[str, Any]) -> bytes | None:
  """Encode a completed future for a client that asked for protobuf (passes raw `proto_bytes` through untouched)."""
  if is_proto_result(result):
    return proto_result_bytes(result)
  result_type = result.get("type")
  if result_type in ("forward_backward_completed", "forward_backward"):
    return encode_forward_backward_output(result)
  if result_type in ("sample_completed", "sample"):
    return encode_sample_response(result)
  return None


def decode_future_result_json(result: dict[str, Any]) -> dict[str, Any]:
  """Convert a `proto_result` envelope to JSON dict at the gateway edge when client wants JSON."""
  if not is_proto_result(result):
    return result
  result_type = str(result.get("type") or "")
  raw = proto_result_bytes(result)
  if result_type in ("forward_backward_completed", "forward_backward"):
    return forward_backward_output_to_json(raw, result_type=result_type)
  if result_type in ("sample_completed", "sample"):
    return sample_response_to_json(raw, result_type=result_type)
  return result
