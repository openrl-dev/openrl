"""Tests for `server.api.wire`, `training.commands` proto passthrough, and binary store framing."""

from __future__ import annotations

import array
import json
import math
import os
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Any

import numpy as np

from server import proto_codec
from server.api import wire
from server.proto import tinker_public_pb2 as pb
from server.sampling_store import SamplingStore
from server.store import InMemoryStore, RedisStore, decode_wire_dict, encode_wire_dict
from training import commands
from training.types import Datum as JsonDatum

FIXTURES = Path(__file__).resolve().parent / "fixtures"
SRC_DIR = Path(__file__).resolve().parents[1] / "src"


class NoNumpyImportTest(unittest.TestCase):
  def test_wire_and_gateway_modules_work_without_numpy_installed(self) -> None:
    """`Dockerfile.api_server` does not install numpy; gateway wire helpers must work without it."""
    script = f"""
import json, sys
from pathlib import Path

class BlockNumpy:
  def find_spec(self, fullname, path, target=None):
    if fullname == "numpy" or fullname.startswith("numpy."):
      raise ModuleNotFoundError("numpy blocked in gateway test")
    return None

sys.meta_path.insert(0, BlockNumpy())

from server.api import wire
from server import proto_codec
from training import commands
from server.store import encode_wire_dict, decode_wire_dict

assert "numpy" not in sys.modules
raw = Path({str(FIXTURES / "fwdbwd_request_tinker_0.29.0.pb")!r}).read_bytes()
expected = json.loads(Path({str(FIXTURES / "fwdbwd_request_tinker_0.29.0.json")!r}).read_text())

req = wire.parse_forward_backward(raw)
assert req.model_id == expected["model_id"]
assert wire.cost(req) > 0
assert wire.loss_config(req) == expected["forward_backward_input"]["loss_fn_config"]
assert wire.decode_forward_backward(raw) == expected

reencoded = wire.json_to_forward_backward_proto(expected)
assert wire.decode_forward_backward(reencoded) == expected

cmd = commands.ForwardBackward(request_id="r1", model_id=req.model_id, loss_fn=req.loss_fn, proto_bytes=raw)
wired = commands.wire(cmd)
assert wired["payload"]["proto_bytes"] == raw
restored = commands.parse_command(decode_wire_dict(encode_wire_dict(wired)))
assert restored.proto_bytes == raw

fb_json = {{
  "type": "forward_backward_completed",
  "loss_fn_output_type": "ArrayRecord",
  "metrics": {{"loss:mean": 1.25}},
  "loss_fn_outputs": [{{"logprobs": {{"data": [-0.5, -1.0], "dtype": "float32", "shape": [2]}}}}],
}}
fb_raw = wire.encode_forward_backward_output(fb_json)
env = wire.proto_result("forward_backward_completed", fb_raw, metrics={{"loss:mean": 1.25}})
assert wire.encode_future_result(env) == fb_raw
assert wire.decode_future_result_json(env) == fb_json
assert "numpy" not in sys.modules
print("OK_NO_NUMPY")
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC_DIR)
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env, check=False)
    self.assertEqual(proc.returncode, 0, f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}")
    self.assertIn("OK_NO_NUMPY", proc.stdout)


class WireRequestAndWorkerTest(unittest.TestCase):
  def test_sdk_0_29_fixture_parse_cost_datums_and_json_roundtrip(self) -> None:
    raw = (FIXTURES / "fwdbwd_request_tinker_0.29.0.pb").read_bytes()
    expected = json.loads((FIXTURES / "fwdbwd_request_tinker_0.29.0.json").read_text())

    req = wire.parse_forward_backward(raw)
    self.assertEqual(req.model_id, expected["model_id"])
    self.assertEqual(req.seq_id, expected["seq_id"])
    self.assertEqual(wire.loss_config(req), expected["forward_backward_input"]["loss_fn_config"])

    expected_tokens = sum(len(chunk["tokens"]) for d in expected["forward_backward_input"]["data"] for chunk in d["model_input"]["chunks"])
    self.assertEqual(wire.cost(req), expected_tokens)

    # Worker zero-.tolist() NumPy decoding
    parsed_datums = wire.datums(raw)
    self.assertEqual(len(parsed_datums), len(expected["forward_backward_input"]["data"]))
    for datum, exp_d in zip(parsed_datums, expected["forward_backward_input"]["data"]):
      exp_toks = [t for ch in exp_d["model_input"]["chunks"] for t in ch["tokens"]]
      self.assertEqual(datum.tokens.dtype, np.int32)
      self.assertTrue(datum.tokens.flags.writeable)
      self.assertEqual(datum.tokens.tolist(), exp_toks)
      for name, exp_t in exp_d["loss_fn_inputs"].items():
        arr = datum.inputs[name]
        self.assertTrue(arr.flags.writeable)
        self.assertEqual(arr.tolist(), exp_t["data"])

    # JSON -> protobuf -> decode roundtrip matches fixture JSON
    reencoded = wire.json_to_forward_backward_proto(expected)
    self.assertEqual(wire.decode_forward_backward(reencoded), expected)
    re_datums = wire.datums(reencoded)
    for d1, d2 in zip(parsed_datums, re_datums):
      np.testing.assert_array_equal(d1.tokens, d2.tokens)
      self.assertEqual(set(d1.inputs), set(d2.inputs))
      for k in d1.inputs:
        np.testing.assert_array_equal(d1.inputs[k], d2.inputs[k])

  def test_parse_validation_rejects_bad_chunks_tensors_and_shapes(self) -> None:
    with self.assertRaises(wire.ProtoDecodeError):
      wire.parse_forward_backward(b"\xff\xff\xff not protobuf")

    # 2-D tensor rejected
    msg = pb.ForwardBackwardRequest(model_id="m", seq_id=1, loss_fn="cross_entropy")
    d = msg.data.add()
    d.model_input.add().encoded_text.tokens = array.array("i", [1, 2]).tobytes()
    t = d.loss_fn_inputs["weights"]
    t.dtype = pb.DTYPE_FLOAT32
    t.dense = array.array("f", [1.0, 2.0, 3.0, 4.0]).tobytes()
    t.shape.extend([2, 2])
    with self.assertRaises(wire.ProtoDecodeError):
      wire.parse_forward_backward(msg.SerializeToString())

    # Misaligned token bytes rejected
    msg2 = pb.ForwardBackwardRequest(model_id="m", seq_id=1, loss_fn="cross_entropy")
    msg2.data.add().model_input.add().encoded_text.tokens = b"\x01\x02\x03"
    with self.assertRaises(wire.ProtoDecodeError):
      wire.parse_forward_backward(msg2.SerializeToString())

  def test_spans_referenced_sequences_and_stability_packing(self) -> None:
    msg = pb.ForwardBackwardRequest(model_id="m", seq_id=1, loss_fn="importance_sampling")
    d = msg.data.add()
    d.model_input.add().encoded_text.tokens = array.array("i", [10, 20, 30]).tobytes()
    sp1 = d.model_input_spans.add()
    sp1.prompt_tokens.sequence_id = "seq-0"
    sp1.prompt_tokens.offset = 0
    sp1.prompt_tokens.length = 2
    sp2 = d.loss_fn_input_spans.add()
    sp2.sampled_tokens.sequence_id = "seq-1"
    sp2.sampled_tokens.offset = 1
    sp2.sampled_tokens.length = 2

    self.assertEqual(wire.referenced_sequences(msg), {"seq-1"})
    [datum] = wire.datums(msg)
    self.assertEqual(datum.model_spans, (wire.Span("prompt", "seq-0", 0, 2),))
    self.assertEqual(datum.loss_spans, (wire.Span("sampled", "seq-1", 1, 2),))

    ids = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    lps = np.array([[-0.1, -0.2, -0.3], [-0.4, -0.5, -0.6]], dtype=np.float32)
    ids2, lps2 = wire.unpack_stability(wire.pack_stability(ids, lps))
    np.testing.assert_array_equal(ids2, ids)
    np.testing.assert_allclose(lps2, lps)

  def test_worker_output_and_sample_output_roundtrip_and_passthrough(self) -> None:
    lp0 = np.array([-0.5, -1.5], dtype=np.float32)
    lp1 = np.array([-0.25, -0.75, -2.0], dtype=np.float32)
    metrics = {"loss:mean": 1.5, "loss:sum": 3.0}
    raw_fb = wire.output([lp0, lp1], metrics)

    expected_fb = {
      "type": "forward_backward_completed",
      "loss_fn_output_type": "ArrayRecord",
      "metrics": metrics,
      "loss_fn_outputs": [
        {"logprobs": {"data": [-0.5, -1.5], "dtype": "float32", "shape": [2]}},
        {"logprobs": {"data": [-0.25, -0.75, -2.0], "dtype": "float32", "shape": [3]}},
      ],
    }
    self.assertEqual(raw_fb, proto_codec.encode_forward_backward_output(expected_fb))
    self.assertEqual(wire.forward_backward_output_to_json(raw_fb), expected_fb)

    fb_env = wire.proto_result("forward_backward_completed", raw_fb, metrics=metrics)
    self.assertTrue(wire.is_proto_result(fb_env))
    self.assertEqual(wire.encode_future_result(fb_env), raw_fb)
    self.assertEqual(wire.decode_future_result_json(fb_env), expected_fb)

    # SampleResponse worker pack + gateway JSON decode (with NaN -> None + topk)
    tk_ids = np.array([[10, 11], [12, 13]], dtype=np.int32)
    tk_lps = np.array([[-0.1, -0.2], [-0.3, -0.4]], dtype=np.float32)
    raw_s = wire.sample_output(
      [
        wire.SampleOut(
          stop=True,
          tokens=np.array([9707, 11], dtype=np.int32),
          logprobs=np.array([-0.5, -1.0], dtype=np.float32),
          topk=(tk_ids, tk_lps),
        ),
        wire.SampleOut(stop=False, tokens=np.array([151645], dtype=np.int32), logprobs=None),
      ],
      prompt_logprobs=np.array([math.nan, -3.0, -0.125], dtype=np.float32),
      topk_prompt=(tk_ids, tk_lps),
      prompt_cache_hit_tokens=7,
    )
    msg_s = pb.SampleResponse()
    msg_s.ParseFromString(raw_s)
    self.assertEqual(msg_s.sequences[0].topk_sampled_logprobs.k, 2)
    self.assertEqual(msg_s.sequences[0].topk_sampled_logprobs.length, 2)
    self.assertEqual(msg_s.topk_prompt_logprobs.k, 2)

    s_env = wire.proto_result("sample_completed", raw_s)
    self.assertEqual(wire.encode_future_result(s_env), raw_s)
    decoded_s = wire.decode_future_result_json(s_env)
    self.assertEqual(decoded_s["type"], "sample_completed")
    self.assertEqual(decoded_s["sequences"][0], {"tokens": [9707, 11], "logprobs": [-0.5, -1.0], "stop_reason": "stop"})
    self.assertEqual(decoded_s["sequences"][1], {"tokens": [151645], "stop_reason": "length"})
    self.assertEqual(decoded_s["prompt_logprobs"], [None, -3.0, -0.125])
    self.assertEqual(decoded_s["prompt_cache_hit_tokens"], 7)


class _FakeUtf8Redis:
  """Minimal async Redis stub enforcing UTF-8 str-only values like `decode_responses=True`."""

  def __init__(self) -> None:
    self.lists: dict[str, list[str]] = {}
    self.sets: dict[str, set[str]] = {}
    self.streams: dict[str, list[tuple[str, dict[str, str]]]] = {}
    self._seq = 0

  def _check_str(self, val: Any) -> str:
    if not isinstance(val, str):
      raise TypeError(f"Redis decode_responses=True expects str, got {type(val).__name__}")
    return val

  async def rpush(self, key: str, value: str) -> int:
    self.lists.setdefault(key, []).append(self._check_str(value))
    return len(self.lists[key])

  async def lpush(self, key: str, *values: str) -> int:
    for v in values:
      self.lists.setdefault(key, []).insert(0, self._check_str(v))
    return len(self.lists[key])

  async def lpop(self, key: str) -> str | None:
    lst = self.lists.get(key)
    return lst.pop(0) if lst else None

  async def lindex(self, key: str, idx: int) -> str | None:
    lst = self.lists.get(key)
    if not lst:
      return None
    try:
      return lst[idx]
    except IndexError:
      return None

  async def llen(self, key: str) -> int:
    return len(self.lists.get(key, []))

  async def ltrim(self, key: str, start: int, end: int) -> None:
    lst = self.lists.get(key, [])
    if start < 0:
      start = max(0, len(lst) + start)
    if end < 0:
      end = len(lst) + end
    self.lists[key] = lst[start : end + 1]

  async def lmove(self, src: str, dst: str, wherefrom: str, whereto: str) -> str | None:
    val = await self.lpop(src)
    if val is not None:
      await self.rpush(dst, val)
    return val

  async def lrem(self, key: str, count: int, value: str) -> int:
    lst = self.lists.get(key, [])
    before = len(lst)
    self.lists[key] = [x for x in lst if x != value]
    return before - len(self.lists[key])

  async def sadd(self, key: str, member: str) -> int:
    s = self.sets.setdefault(key, set())
    if member in s:
      return 0
    s.add(member)
    return 1

  async def srem(self, key: str, member: str) -> int:
    s = self.sets.get(key, set())
    if member in s:
      s.remove(member)
      return 1
    return 0

  async def expire(self, key: str, seconds: int) -> bool:
    return True

  async def xadd(self, stream: str, fields: dict[str, str]) -> str:
    self._seq += 1
    eid = f"1700000000000-{self._seq}"
    self.streams.setdefault(stream, []).append((eid, {k: self._check_str(v) for k, v in fields.items()}))
    return eid

  async def xrange(self, stream: str, min: str = "-", max: str = "+", count: int | None = None):
    entries = list(self.streams.get(stream, []))
    if min.startswith("("):
      entries = [e for e in entries if e[0] > min[1:]]
    elif min != "-":
      entries = [e for e in entries if e[0] == min or e[0] >= min]
    if max != "+":
      entries = [e for e in entries if e[0] <= max]
    return entries[:count] if count is not None else entries

  async def xdel(self, stream: str, entry_id: str) -> int:
    entries = self.streams.get(stream, [])
    self.streams[stream] = [e for e in entries if e[0] != entry_id]
    return 1

  def pipeline(self, transaction: bool = True):
    redis_self = self

    class _Pipe:
      def __init__(self) -> None:
        self.ops: list[tuple[str, tuple[Any, ...]]] = []

      async def __aenter__(self):
        return self

      async def __aexit__(self, *exc):
        return False

      def multi(self) -> None:
        pass

      def rpush(self, key: str, value: str) -> None:
        self.ops.append(("rpush", (key, value)))

      def ltrim(self, key: str, start: int, end: int) -> None:
        self.ops.append(("ltrim", (key, start, end)))

      def expire(self, key: str, seconds: int) -> None:
        self.ops.append(("expire", (key, seconds)))

      def xdel(self, stream: str, entry_id: str) -> None:
        self.ops.append(("xdel", (stream, entry_id)))

      async def xrange(self, stream: str, min_id: str, max_id: str):
        return await redis_self.xrange(stream, min=min_id, max=max_id)

      async def execute(self):
        out = []
        for op, args in self.ops:
          out.append(await getattr(redis_self, op)(*args))
        return out

    return _Pipe()

  async def transaction(self, func, *watches, value_from_callable: bool = False):
    pipe = self.pipeline(transaction=True)
    res = await func(pipe)
    await pipe.execute()
    return res if value_from_callable else None


class BinaryCommandsAndStoreTest(unittest.IsolatedAsyncioTestCase):
  async def test_forward_backward_proto_bytes_in_memory_and_redis_store(self) -> None:
    raw = (FIXTURES / "fwdbwd_request_tinker_0.29.0.pb").read_bytes()
    cmd = commands.ForwardBackward(request_id="req-pb", model_id="m1", loss_fn="importance_sampling", forward_only=False, proto_bytes=raw)
    wired = commands.wire(cmd)
    self.assertNotIn("data", wired["payload"])
    self.assertIs(wired["payload"]["proto_bytes"], raw)

    # Legacy JSON ForwardBackward still omits `proto_bytes` on wire for backward compatibility
    json_cmd = commands.ForwardBackward(request_id="req-j", model_id="m1", data=[JsonDatum(model_input=[1, 2], loss_fn_inputs={})])
    self.assertNotIn("proto_bytes", commands.wire(json_cmd)["payload"])

    # 1. InMemoryStore: zero-copy raw bytes passthrough
    mem = InMemoryStore()
    await mem.put_request(wired)
    [popped_mem] = await mem.get_requests()
    self.assertIs(popped_mem["payload"]["proto_bytes"], raw)
    parsed_mem = commands.parse_command(popped_mem)
    self.assertIsInstance(parsed_mem, commands.ForwardBackward)
    self.assertEqual(parsed_mem.proto_bytes, raw)

    fb_raw = wire.output([np.array([-0.5, -1.0], dtype=np.float32)], {"loss:mean": 0.75})
    fb_env = wire.proto_result("forward_backward_completed", fb_raw, metrics={"loss:mean": 0.75})
    await mem.set_result("req-pb", fb_env)
    got_mem = await mem.get_result("req-pb", timeout=1.0)
    self.assertIs(got_mem["proto_bytes"], fb_raw)

    # 2. RedisStore (UTF-8 str-only wire): base64-framed without mutating caller dicts
    fake_redis = _FakeUtf8Redis()
    rstore = RedisStore(fake_redis)
    await rstore.put_request(wired)
    self.assertIs(wired["payload"]["proto_bytes"], raw)  # caller dict untouched
    [popped_redis] = await rstore.get_requests()
    self.assertEqual(popped_redis["payload"]["proto_bytes"], raw)
    self.assertEqual(commands.parse_command(popped_redis).proto_bytes, raw)

    await rstore.set_result("req-pb", fb_env)
    self.assertIs(fb_env["proto_bytes"], fb_raw)  # caller dict untouched
    got_redis = await rstore.get_result("req-pb", timeout=1.0)
    self.assertEqual(got_redis["type"], "forward_backward_completed")
    self.assertEqual(got_redis["metrics"], {"loss:mean": 0.75})
    self.assertEqual(got_redis["proto_bytes"], fb_raw)

    # 3. SamplingStore.set_result -> RedisStore.get_result binary SampleResponse passthrough
    sstore = SamplingStore(fake_redis)
    rec = {"request_id": "req-s", "pool": "pool-1", "revision": "r1", "model": "base", "prompt_token_ids": [1, 2]}
    await sstore.accept(rec)
    [pending] = await sstore.pending("pool-1")
    samp_raw = wire.sample_output(
      [wire.SampleOut(stop=True, tokens=np.array([5, 6], dtype=np.int32), logprobs=np.array([-0.2, -0.3], dtype=np.float32))]
    )
    saved = await sstore.set_result(pending, wire.proto_result("sample_completed", samp_raw))
    self.assertTrue(saved)
    got_samp = await rstore.get_result("req-s", timeout=1.0)
    self.assertEqual(got_samp["type"], "sample_completed")
    self.assertEqual(got_samp["proto_bytes"], samp_raw)
    self.assertEqual(wire.encode_future_result(got_samp), samp_raw)

    # 4. Plain JSON dicts still serialize identically (zero proto_b64 noise)
    plain = {"type": "optim_step_completed", "metrics": {"lr": 1e-4}}
    self.assertEqual(encode_wire_dict(plain), json.dumps(plain))
    self.assertEqual(decode_wire_dict(json.dumps(plain)), plain)


if __name__ == "__main__":
  unittest.main()
