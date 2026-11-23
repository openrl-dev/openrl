"""Tests for `server.api.gateway.create_app(...)` dependency injection, binary protobuf passthrough, and no-numpy isolation."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from server import api_server
from server.api import wire
from server.api.gateway import create_app
from server.session_registry import SessionRegistry
from server.store import InMemoryStateStore, InMemoryStore

FIXTURE_PB = Path(__file__).parent / "fixtures" / "fwdbwd_request_tinker_0.29.0.pb"
FIXTURE_JSON = Path(__file__).parent / "fixtures" / "fwdbwd_request_tinker_0.29.0.json"


class GatewayFactoryTest(unittest.TestCase):
  def test_create_app_isolates_dependencies_without_monkeypatching_globals(self) -> None:
    with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
      store_a, state_a = InMemoryStore(), InMemoryStateStore()
      store_b, state_b = InMemoryStore(), InMemoryStateStore()
      app_a = create_app(
        store=store_a,
        state=state_a,
        session_registry=SessionRegistry(state_a),
        tmp_dir=tmp_a,
        version="v1.0.0",
        enable_lifespan=False,
      )
      app_b = create_app(
        store=store_b,
        state=state_b,
        session_registry=SessionRegistry(state_b),
        tmp_dir=tmp_b,
        version="v2.0.0",
        enable_lifespan=False,
      )
      client_a, client_b = TestClient(app_a), TestClient(app_b)

      caps_a = client_a.get("/api/v1/get_server_capabilities").json()
      caps_b = client_b.get("/api/v1/get_server_capabilities").json()
      self.assertEqual(caps_a["openrl"]["version"], "v1.0.0")
      self.assertEqual(caps_b["openrl"]["version"], "v2.0.0")

      sess_a = client_a.post("/api/v1/create_session", json={"tags": ["openrl.recipe=sft"]}).json()["session_id"]
      model_a = client_a.post("/api/v1/create_model", json={"base_model": "model-a", "session_id": sess_a}).json()["request_id"]
      client_a.post("/api/v1/save_weights", json={"model_id": model_a, "path": "step-1"})

      # App B and the global api_server facade saw none of App A's state or queued commands.
      self.assertEqual(store_b.queues, {})
      self.assertEqual(state_b.kv_store, {})
      self.assertNotIn(f"open_rl:model_meta:{model_a}", getattr(api_server.state, "kv_store", {}))

      # App A's save_weights resolved against tmp_a (not api_server.TMP_DIR).
      queued = asyncio.run(store_a.get_requests())
      self.assertEqual([r["op"] for r in queued], ["create_model", "save_state"])
      self.assertEqual(queued[1]["payload"]["state_path"], os.path.join(tmp_a, "checkpoints", model_a, "weights", "step-1"))

      # OpenAPI document exposes OpenRLSettings on factory-built apps.
      schema = client_a.get("/openapi.json").json()
      self.assertIn("OpenRLSettings", schema["components"]["schemas"])

  def test_forward_backward_protobuf_passthrough_and_json_edge_conversion(self) -> None:
    store, state = InMemoryStore(), InMemoryStateStore()
    client = TestClient(create_app(store=store, state=state, enable_lifespan=False))

    raw_pb = FIXTURE_PB.read_bytes()
    resp_pb = client.post("/api/v1/forward_backward", content=raw_pb, headers={"Content-Type": wire.PROTO})
    self.assertEqual(resp_pb.status_code, 200, resp_pb.text)
    queued_pb = asyncio.run(store.get_requests())[0]

    # Exact raw bytes forwarded untouched, zero Python `data` list expansion.
    self.assertEqual(queued_pb["op"], "forward_backward")
    self.assertEqual(queued_pb["model_id"], "model-abc")
    self.assertNotIn("data", queued_pb["payload"])
    self.assertEqual(queued_pb["payload"]["proto_bytes"], raw_pb)
    self.assertEqual(queued_pb["payload"]["loss_fn"], "importance_sampling")
    self.assertEqual(queued_pb["payload"]["loss_config"], {"clip_range": 0.2, "kl_coeff": 0.01, "mode": "token"})

    # Legacy JSON request is validated and packed to protobuf bytes at the gateway edge.
    json_body = json.loads(FIXTURE_JSON.read_text())
    resp_json = client.post("/api/v1/forward_backward", json=json_body)
    self.assertEqual(resp_json.status_code, 200, resp_json.text)
    queued_json = asyncio.run(store.get_requests())[0]
    self.assertNotIn("data", queued_json["payload"])
    self.assertIsInstance(queued_json["payload"]["proto_bytes"], bytes)
    self.assertEqual(
      wire.decode_forward_backward(queued_pb["payload"]["proto_bytes"]),
      wire.decode_forward_backward(queued_json["payload"]["proto_bytes"]),
    )

    # /api/v1/forward forces forward_only=True on protobuf bodies while preserving raw proto_bytes.
    resp_fwd = client.post("/api/v1/forward", content=raw_pb, headers={"Content-Type": wire.PROTO})
    self.assertEqual(resp_fwd.status_code, 200, resp_fwd.text)
    queued_fwd = asyncio.run(store.get_requests())[0]
    self.assertTrue(queued_fwd["payload"]["forward_only"])
    self.assertEqual(queued_fwd["payload"]["proto_bytes"], raw_pb)

  def test_retrieve_future_passes_proto_bytes_through_and_decodes_for_json_clients(self) -> None:
    store, state = InMemoryStore(), InMemoryStateStore()
    client = TestClient(create_app(store=store, state=state, enable_lifespan=False))

    raw_fb = wire.encode_forward_backward_output(
      {
        "type": "forward_backward_completed",
        "metrics": {"loss:mean": 1.25},
        "loss_fn_outputs": [{"logprobs": {"data": [-0.5, -1.5], "dtype": "float32", "shape": [2]}}],
      }
    )
    asyncio.run(store.set_result("fb-1", wire.proto_result("forward_backward_completed", raw_fb, metrics={"loss:mean": 1.25})))

    # Protobuf client gets exact worker bytes with zero JSON round-trip.
    as_proto = client.post("/api/v1/retrieve_future", json={"request_id": "fb-1"}, headers={"Accept": wire.PROTO})
    self.assertEqual(as_proto.status_code, 200)
    self.assertEqual(as_proto.headers["content-type"], wire.PROTO)
    self.assertEqual(as_proto.content, raw_fb)

    # JSON client gets edge-decoded JSON.
    as_json = client.post("/api/v1/retrieve_future", json={"request_id": "fb-1"})
    self.assertEqual(as_json.status_code, 200)
    self.assertTrue(as_json.headers["content-type"].startswith(wire.JSON))
    body = as_json.json()
    self.assertEqual(body["type"], "forward_backward")
    self.assertEqual(body["metrics"], {"loss:mean": 1.25})
    self.assertEqual(body["loss_fn_outputs"][0]["logprobs"]["data"], [-0.5, -1.5])

    # Binary SampleResponse passthrough + JSON edge conversion.
    raw_sample = wire.encode_sample_response(
      {
        "type": "sample_completed",
        "sequences": [{"tokens": [10, 20], "logprobs": [-0.25, -0.5], "stop_reason": "stop"}],
        "prompt_logprobs": [None, -1.0],
      }
    )
    asyncio.run(store.set_result("samp-1", wire.proto_result("sample_completed", raw_sample)))

    samp_proto = client.post("/api/v1/retrieve_future", json={"request_id": "samp-1"}, headers={"Accept": wire.PROTO})
    self.assertEqual(samp_proto.status_code, 200)
    self.assertEqual(samp_proto.headers["content-type"], wire.PROTO)
    self.assertEqual(samp_proto.content, raw_sample)

    samp_json = client.post("/api/v1/retrieve_future", json={"request_id": "samp-1"})
    self.assertEqual(samp_json.status_code, 200)
    sbody = samp_json.json()
    self.assertEqual(sbody["type"], "sample")
    self.assertEqual(sbody["sequences"][0]["tokens"], [10, 20])
    self.assertEqual(sbody["sequences"][0]["stop_reason"], "stop")
    self.assertEqual(sbody["prompt_logprobs"], [None, -1.0])

    # Non-proto replies remain JSON even when client sends Accept: application/x-protobuf.
    asyncio.run(store.set_result("opt-1", {"type": "optim_step_completed", "metrics": {"grad_norm:mean": 0.5}}))
    opt = client.post("/api/v1/retrieve_future", json={"request_id": "opt-1"}, headers={"Accept": wire.PROTO})
    self.assertTrue(opt.headers["content-type"].startswith(wire.JSON))
    self.assertEqual(opt.json(), {"type": "optim_step", "metrics": {"grad_norm:mean": 0.5}})

  def test_gateway_imports_and_serves_protobuf_without_numpy(self) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    script = """
import asyncio, sys
from importlib.abc import MetaPathFinder

class BlockNumpy(MetaPathFinder):
  def find_spec(self, fullname, path, target=None):
    if fullname == "numpy" or fullname.startswith("numpy."):
      raise ImportError("numpy is intentionally blocked in gateway image")
    return None

sys.meta_path.insert(0, BlockNumpy())

from fastapi.testclient import TestClient
from server.api import wire
from server.api.gateway import create_app
from server.proto import tinker_public_pb2 as pb
from server.store import InMemoryStateStore, InMemoryStore

store, state = InMemoryStore(), InMemoryStateStore()
client = TestClient(create_app(store=store, state=state, enable_lifespan=False))
msg = pb.ForwardBackwardRequest(model_id="m-no-np", seq_id=1, loss_fn="cross_entropy")
raw = msg.SerializeToString()
r = client.post("/api/v1/forward_backward", content=raw, headers={"Content-Type": wire.PROTO})
assert r.status_code == 200, r.text
queued = asyncio.run(store.get_requests())[0]
assert queued["payload"]["proto_bytes"] == raw
assert "numpy" not in sys.modules
print("NO_NUMPY_GATEWAY_OK")
"""
    env = {**os.environ, "PYTHONPATH": f"{repo_root / 'src'}:{repo_root}"}
    proc = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, check=False)
    self.assertEqual(proc.returncode, 0, f"stdout={proc.stdout}\nstderr={proc.stderr}")
    self.assertIn("NO_NUMPY_GATEWAY_OK", proc.stdout)


if __name__ == "__main__":
  unittest.main()
