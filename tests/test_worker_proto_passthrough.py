"""End-to-end worker tests for binary protobuf passthrough (ForwardBackwardRequest -> Output & SampleResponse)."""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from server.api import wire
from server.proto import tinker_public_pb2 as pb
from server.sampling_dispatcher import to_result
from server.store import InMemoryStore, decode_wire_dict, encode_wire_dict
from server.training_requests_processor import TrainingRequestsProcessor
from server.vllm_sampler import Sampler
from tests.test_automodel_worker import FileCheckpointer, PeftConfig, tiny_lora_model
from tests.test_vllm_sampler import FakeEngine, fake_vllm
from training import automodel_worker, commands
from training.automodel_worker import AutomodelTrainingWorker
from training.fft_trainer_worker import FFTTrainingWorker
from training.lora_trainer_worker import LoraTrainingWorker
from training.types import Datum, LoraConfig, TensorData


def _tiny_llama() -> LlamaForCausalLM:
  torch.manual_seed(0)
  config = LlamaConfig(
    vocab_size=64,
    hidden_size=16,
    intermediate_size=32,
    num_hidden_layers=1,
    num_attention_heads=2,
    num_key_value_heads=2,
    max_position_embeddings=64,
  )
  return LlamaForCausalLM(config)


def _fb_json_payload(loss_fn: str = "cross_entropy", *, forward_only: bool = False) -> dict:
  rows = [[3, 5, 8, 13, 21, 34], [2, 7, 1, 8]]
  data = []
  for row in rows:
    length = len(row) - 1
    inputs = {
      "target_tokens": {"data": row[1:], "dtype": "int64", "shape": [length]},
      "weights": {"data": [1.0] * length, "dtype": "float32", "shape": [length]},
    }
    if loss_fn in ("importance_sampling", "ppo"):
      inputs["logprobs"] = {"data": [-0.5] * length, "dtype": "float32", "shape": [length]}
      inputs["advantages"] = {"data": [0.25] * length, "dtype": "float32", "shape": [length]}
    data.append({"model_input": {"chunks": [{"tokens": row[:-1]}]}, "loss_fn_inputs": inputs})
  loss_cfg = {"clip_range": 0.2, "kl_coeff": 0.01} if loss_fn == "ppo" else {}
  return {
    "model_id": "run-1",
    "seq_id": 1,
    "forward_only": forward_only,
    "forward_backward_input": {"data": data, "loss_fn": loss_fn, "loss_fn_config": loss_cfg},
  }


def _legacy_datums(payload: dict) -> list[Datum]:
  out = []
  for item in payload["forward_backward_input"]["data"]:
    tokens = item["model_input"]["chunks"][0]["tokens"]
    inputs = {k: TensorData(data=v["data"]) for k, v in item["loss_fn_inputs"].items()}
    out.append(Datum(model_input=tokens, loss_fn_inputs=inputs))
  return out


class TrainerProtoPassthroughTest(unittest.IsolatedAsyncioTestCase):
  async def test_fft_processor_proto_matches_legacy_json_for_all_losses_and_forward_only(self) -> None:
    for loss_fn, forward_only in (
      ("cross_entropy", False),
      ("cross_entropy", True),
      ("importance_sampling", False),
      ("ppo", False),
    ):
      with self.subTest(loss_fn=loss_fn, forward_only=forward_only):
        payload = _fb_json_payload(loss_fn, forward_only=forward_only)
        proto_req = wire.json_to_forward_backward_proto(payload, forward_only=forward_only)

        # Reference legacy JSON path
        ref_worker = FFTTrainingWorker()
        ref_worker.device = torch.device("cpu")
        ref_worker.model = _tiny_llama()
        ref_worker.base_model_name = "tiny"
        ref_res = ref_worker.forward_backward(
          _legacy_datums(payload),
          loss_fn,
          payload["forward_backward_input"]["loss_fn_config"],
          model_id="run-1",
          forward_only=forward_only,
        )

        # Binary protobuf passthrough path via TrainingRequestsProcessor + UTF-8 Redis wire codec
        proto_worker = FFTTrainingWorker()
        proto_worker.device = torch.device("cpu")
        proto_worker.model = _tiny_llama()
        proto_worker.base_model_name = "tiny"
        store = InMemoryStore()
        proc = TrainingRequestsProcessor(store, proto_worker)

        cmd = commands.ForwardBackward(
          request_id="req-1",
          model_id="run-1",
          loss_fn=loss_fn,
          loss_config=payload["forward_backward_input"]["loss_fn_config"],
          forward_only=forward_only,
          proto_bytes=proto_req,
        )
        wire_msg = decode_wire_dict(encode_wire_dict(commands.wire(cmd)))
        self.assertIn("proto_bytes", wire_msg["payload"])
        self.assertNotIn("data", wire_msg["payload"])

        await proc.process_request(wire_msg)
        stored = await store.get_result("req-1", timeout=5)
        self.assertTrue(wire.is_proto_result(stored), f"expected proto_result envelope, got {stored}")
        self.assertEqual(stored["type"], "forward_backward_completed")
        self.assertIn("metrics", stored)

        # Verify raw protobuf parses cleanly via ForwardBackwardOutput AND matches legacy JSON output.
        raw_out = wire.encode_future_result(stored)
        self.assertEqual(raw_out, stored["proto_bytes"])
        pb_out = pb.ForwardBackwardOutput()
        pb_out.ParseFromString(raw_out)
        self.assertEqual(len(pb_out.loss_fn_outputs), 1)
        self.assertEqual(pb_out.loss_fn_outputs[0].num_datums, 2)

        decoded_json = wire.decode_future_result_json(stored)
        self.assertAlmostEqual(decoded_json["metrics"]["loss:sum"], ref_res["metrics"]["loss:sum"], places=5)
        for lhs, rhs in zip(decoded_json["loss_fn_outputs"], ref_res["loss_fn_outputs"], strict=True):
          for a, b in zip(lhs["logprobs"]["data"], rhs["logprobs"]["data"], strict=True):
            self.assertAlmostEqual(a, b, places=5)

        if forward_only:
          self.assertTrue(all(p.grad is None for p in proto_worker.model.parameters()))
        else:
          for p_ref, p_proto in zip(ref_worker.model.parameters(), proto_worker.model.parameters(), strict=True):
            torch.testing.assert_close(p_proto.grad, p_ref.grad)

  async def test_lora_and_automodel_workers_handle_proto_passthrough(self) -> None:
    payload = _fb_json_payload("cross_entropy")
    proto_req = wire.json_to_forward_backward_proto(payload)
    cmd = commands.ForwardBackward(request_id="req-lora", model_id="a", loss_fn="cross_entropy", proto_bytes=proto_req)

    # 1. LoRA worker
    lora_worker = LoraTrainingWorker()
    lora_worker.device = torch.device("cpu")
    lora_worker.base_model = _tiny_llama()
    lora_worker.base_model_name = "tiny"
    lora_worker.save_adapter = lambda *_a, **_kw: None
    lora_worker.create_adapter("a", LoraConfig(rank=2, seed=1))
    lora_store = InMemoryStore()
    await TrainingRequestsProcessor(lora_store, lora_worker).process_request(commands.wire(cmd))
    lora_res = await lora_store.get_result("req-lora", timeout=5)
    self.assertTrue(wire.is_proto_result(lora_res))
    self.assertEqual(len(wire.decode_future_result_json(lora_res)["loss_fn_outputs"]), 2)

    # 2. Automodel worker
    with patch.object(automodel_worker, "MAX_LORA_RANK", 4):
      am_worker = AutomodelTrainingWorker()
      am_worker.device = torch.device("cpu")

      def load_model(base_model_name: str) -> None:
        am_worker.base_model_name = base_model_name
        am_worker.model = tiny_lora_model(4)

      am_worker.load_model = load_model
      am_worker.build_peft_config = lambda cfg: PeftConfig(dim=cfg.rank, alpha=cfg.lora_alpha)
      am_worker.checkpointer = FileCheckpointer()
      am_worker.create_model("tiny", "a", LoraConfig(rank=2, seed=1))

      am_store = InMemoryStore()
      await TrainingRequestsProcessor(am_store, am_worker).process_request(commands.wire(cmd))
      am_res = await am_store.get_result("req-lora", timeout=5)
      self.assertTrue(wire.is_proto_result(am_res))
      self.assertEqual(len(wire.decode_future_result_json(am_res)["loss_fn_outputs"]), 2)

  async def test_torch_sampler_command_emits_proto_sample_response(self) -> None:
    worker = FFTTrainingWorker()
    worker.device = torch.device("cpu")
    worker.model = _tiny_llama()
    worker.tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=0)
    store = InMemoryStore()
    proc = TrainingRequestsProcessor(store, worker)
    cmd = commands.Sample(
      request_id="samp-1",
      model_id="run-1",
      prompt_tokens=[3, 5, 8],
      max_tokens=2,
      num_samples=1,
      temperature=0.0,
      prompt_logprobs=True,
    )
    await proc.process_request(commands.wire(cmd))
    stored = await store.get_result("samp-1", timeout=5)
    self.assertTrue(wire.is_proto_result(stored))
    self.assertEqual(stored["type"], "sample_completed")

    pb_resp = pb.SampleResponse()
    pb_resp.ParseFromString(wire.proto_result_bytes(stored))
    self.assertEqual(len(pb_resp.sequences), 1)

    decoded = wire.decode_future_result_json(stored)
    self.assertEqual(decoded["type"], "sample_completed")
    self.assertEqual(len(decoded["sequences"][0]["tokens"]), 2)
    self.assertEqual(len(decoded["prompt_logprobs"]), 3)
    self.assertIsNone(decoded["prompt_logprobs"][0])


class SamplerProtoPassthroughTest(unittest.IsolatedAsyncioTestCase):
  async def test_vllm_sampler_emits_binary_sample_response_proto(self) -> None:
    class _PromptLpEngine(FakeEngine):
      async def generate(self, prompt: dict, sampling_params: dict, request_id: str, lora_request: str | None):
        choice = SimpleNamespace(
          token_ids=[10, 11],
          logprobs=[{10: SimpleNamespace(logprob=-0.25)}, {11: SimpleNamespace(logprob=-0.5)}],
          finish_reason="stop",
        )
        plps = [None, {2: SimpleNamespace(logprob=-0.1)}, {3: SimpleNamespace(logprob=-0.2)}]
        yield SimpleNamespace(outputs=[choice], prompt_logprobs=plps)

    with patch.dict(sys.modules, fake_vllm()):
      sampler = Sampler(_PromptLpEngine(), "base")
      res = await sampler.generate(
        {
          "request_id": "r1",
          "model": "base",
          "prompt_token_ids": [1, 2, 3],
          "sampling_params": {"n": 1, "max_tokens": 2},
          "include_prompt_logprobs": True,
        }
      )

    self.assertTrue(wire.is_proto_result(res))
    self.assertEqual(res["type"], "sample")
    decoded = wire.decode_future_result_json(res)
    self.assertEqual(decoded["sequences"][0]["tokens"], [10, 11])
    self.assertEqual(decoded["sequences"][0]["stop_reason"], "stop")
    self.assertAlmostEqual(decoded["sequences"][0]["logprobs"][0], -0.25, places=5)
    self.assertAlmostEqual(decoded["sequences"][0]["logprobs"][1], -0.5, places=5)
    self.assertIsNone(decoded["prompt_logprobs"][0])
    self.assertAlmostEqual(decoded["prompt_logprobs"][1], -0.1, places=5)
    self.assertAlmostEqual(decoded["prompt_logprobs"][2], -0.2, places=5)

  def test_sampling_dispatcher_to_result_emits_proto_without_numpy(self) -> None:
    body = {
      "choices": [
        {"token_ids": [100, 101], "finish_reason": "length", "logprobs": {"content": [{"logprob": -0.5}, {"logprob": -0.25}]}},
      ],
      "prompt_logprobs": [None, {"2": {"logprob": -0.1}}],
    }
    record = {"prompt_token_ids": [1, 2], "include_prompt_logprobs": True}
    res = to_result(body, record)
    self.assertTrue(wire.is_proto_result(res))
    self.assertEqual(res["type"], "sample")
    decoded = wire.decode_future_result_json(res)
    self.assertEqual(decoded["sequences"][0]["tokens"], [100, 101])
    self.assertIsNone(decoded["prompt_logprobs"][0])
    self.assertAlmostEqual(decoded["prompt_logprobs"][1], -0.1, places=5)

    # Prove sampling_dispatcher.to_result works in Dockerfile.api_server (no numpy installed).
    code = """
import sys
sys.modules["numpy"] = None
from server.api import wire
from server.sampling_dispatcher import to_result
body = {"choices": [{"token_ids": [7], "finish_reason": "stop", "logprobs": {"content": [{"logprob": -0.5}]}}]}
res = to_result(body, {"prompt_token_ids": [1], "include_prompt_logprobs": False})
assert wire.is_proto_result(res)
assert wire.decode_future_result_json(res)["sequences"][0]["tokens"] == [7]
"""
    env = dict(os.environ)
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    self.assertEqual(proc.returncode, 0, proc.stderr)


if __name__ == "__main__":
  unittest.main()
