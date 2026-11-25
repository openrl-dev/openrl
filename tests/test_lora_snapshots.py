import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from safetensors.numpy import load_file, save_file

from server import lora_snapshots
from server.lora_snapshots import freeze_adapter, served_as_multimodal, snapshot_name, snapshot_root

LAYER_A = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
LAYER_B = "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"
HEAD_A = "base_model.model.lm_head.lora_A.weight"


def served(name: str) -> str:
  return name.replace("base_model.model.model.", "base_model.model.model.language_model.", 1)


class Qwen35SnapshotLayoutTest(unittest.TestCase):
  """A Qwen3.5 LoRA saved from the text model reaches vLLM in the multimodal layout it serves the model in."""

  def setUp(self) -> None:
    self.enterContext(patch.dict(os.environ, {"OPEN_RL_TMP_DIR": self.enterContext(tempfile.TemporaryDirectory())}))
    self.enterContext(patch.dict(os.environ)).pop("VLLM_ARCHITECTURE_OVERRIDE", None)
    served_as_multimodal.cache_clear()
    self.addCleanup(served_as_multimodal.cache_clear)
    self.tensors = {
      LAYER_A: np.arange(16, dtype=np.float32).reshape(2, 8),
      LAYER_B: np.ones((8, 2), dtype=np.float32),
      HEAD_A: np.full((2, 8), 3.0, dtype=np.float32),
    }

  def freeze(self, tensors: dict, ref: str = "tinker://job/sampler_weights/sampler-1") -> Path:
    live = Path(os.environ["OPEN_RL_TMP_DIR"], "peft", "job", "job")
    live.mkdir(parents=True, exist_ok=True)
    (live / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "/models/qwen", "r": 2}))
    save_file(tensors, str(live / "adapter_model.safetensors"))
    freeze_adapter("job", "Qwen/Qwen3.5-0.8B", ref)
    return snapshot_root() / snapshot_name(ref) / "adapter_model.safetensors"

  def test_a_text_model_lora_is_renamed_for_a_multimodal_base_and_keeps_its_weights(self) -> None:
    with patch.object(lora_snapshots, "served_as_multimodal", return_value=True):
      frozen = load_file(str(self.freeze(self.tensors)))
    self.assertEqual(sorted(frozen), sorted([served(LAYER_A), served(LAYER_B), HEAD_A]))
    np.testing.assert_array_equal(frozen[served(LAYER_A)], self.tensors[LAYER_A])
    np.testing.assert_array_equal(frozen[HEAD_A], self.tensors[HEAD_A])

  def test_other_bases_and_already_renamed_adapters_are_left_as_saved(self) -> None:
    with patch.object(lora_snapshots, "served_as_multimodal", return_value=False):
      self.assertEqual(sorted(load_file(str(self.freeze(self.tensors)))), sorted(self.tensors))
    renamed = {served(name): value for name, value in self.tensors.items()}
    with patch.object(lora_snapshots, "served_as_multimodal", side_effect=AssertionError("no config read needed")):
      self.assertEqual(sorted(load_file(str(self.freeze(renamed, "tinker://job/sampler_weights/sampler-2")))), sorted(renamed))

  def test_the_served_class_comes_from_the_base_models_config(self) -> None:
    base = Path(self.enterContext(tempfile.TemporaryDirectory()))
    (base / "config.json").write_text(json.dumps({"model_type": "qwen3_5", "architectures": ["Qwen3_5ForConditionalGeneration"]}))
    self.assertTrue(served_as_multimodal(str(base)))
    served_as_multimodal.cache_clear()
    with patch.dict(os.environ, {"VLLM_ARCHITECTURE_OVERRIDE": "Qwen3_5ForCausalLM"}):
      self.assertFalse(served_as_multimodal(str(base)))
    served_as_multimodal.cache_clear()
    (base / "config.json").write_text(json.dumps({"model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"]}))
    self.assertFalse(served_as_multimodal(str(base)))


class LoraSnapshotTest(unittest.TestCase):
  def test_a_frozen_snapshot_keeps_its_weights_is_never_overwritten_and_names_the_served_base(self) -> None:
    self.enterContext(patch.dict(os.environ, {"OPEN_RL_TMP_DIR": self.enterContext(tempfile.TemporaryDirectory())}))
    live = Path(os.environ["OPEN_RL_TMP_DIR"], "peft", "job", "job")

    def save(weights: bytes) -> None:
      """What the trainer writes before each freeze."""
      live.mkdir(parents=True, exist_ok=True)
      (live / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "/models/qwen", "r": 8}))
      (live / "adapter_model.safetensors").write_bytes(weights)

    ref, session = "tinker://job/sampler_weights/sampler-1", "session-1:0"
    save(b"version-one")
    freeze_adapter("job", "Qwen/Qwen2.5-0.5B", ref, session)
    # The same weights saved under the same name again, with no training between, are already frozen.
    save(b"version-one")
    freeze_adapter("job", "Qwen/Qwen2.5-0.5B", ref)
    save(b"version-two")
    with self.assertRaisesRegex(FileExistsError, "new name or sequence ID"):
      freeze_adapter("job", "Qwen/Qwen2.5-0.5B", ref)
    # The refused save can still go out under a new name.
    renamed = "tinker://job/sampler_weights/sampler-2"
    freeze_adapter("job", "Qwen/Qwen2.5-0.5B", renamed)
    self.assertEqual((snapshot_root() / snapshot_name(renamed) / "adapter_model.safetensors").read_bytes(), b"version-two")
    # Both names of one save serve its weights.
    self.assertEqual((snapshot_root() / snapshot_name(session) / "adapter_model.safetensors").read_bytes(), b"version-one")
    frozen = snapshot_root() / snapshot_name(ref)
    self.assertEqual((frozen / "adapter_model.safetensors").read_bytes(), b"version-one")
    self.assertEqual(json.loads((frozen / "adapter_config.json").read_text()), {"base_model_name_or_path": "Qwen/Qwen2.5-0.5B", "r": 8})
