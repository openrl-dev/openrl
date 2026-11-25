"""Each saved LoRA reference gets an immutable copy of the adapter in vLLM's
VLLM_LORA_RESOLVER_CACHE_DIR, so its directory name is the served model name."""

import filecmp
import functools
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

# Qwen3.5 checkpoints are multimodal: vLLM serves them as Qwen3_5(Moe)ForConditionalGeneration, which looks for LoRA
# weights under model.language_model.*. The trainer loads the text model through AutoModelForCausalLM, so PEFT saves
# them under model.*, and vLLM before 0.33 applies none of them, without an error (vllm-project/vllm#62224).
# Drop this once the sampler runs vLLM 0.33 or later.
TEXT_MODEL_PREFIX = "base_model.model.model."
LANGUAGE_MODEL_PREFIX = "base_model.model.model.language_model."
MULTIMODAL_ARCHITECTURES = frozenset({"Qwen3_5ForConditionalGeneration", "Qwen3_5MoeForConditionalGeneration"})
MAX_SAFETENSORS_HEADER = 100 * 1024 * 1024


def snapshot_root() -> Path:
  return Path(os.getenv("OPEN_RL_TMP_DIR", "/tmp/open-rl"), "sampler_lora")


def snapshot_name(ref: str) -> str:
  # A hash, because a reference holds a user-chosen alias and slashes.
  return hashlib.sha256(ref.encode()).hexdigest()


def same_files(a: Path, b: Path) -> bool:
  names = sorted(p.relative_to(a) for p in a.rglob("*") if p.is_file())
  return names == sorted(p.relative_to(b) for p in b.rglob("*") if p.is_file()) and all(filecmp.cmp(a / n, b / n, shallow=False) for n in names)


def safetensors_header(path: Path) -> tuple[dict, int] | None:
  """A safetensors file's header and where its tensor bytes start, or None when it is not one."""
  try:
    with open(path, "rb") as f:
      size = int.from_bytes(f.read(8), "little")
      if not 0 < size <= MAX_SAFETENSORS_HEADER:
        return None
      header = json.loads(f.read(size))
  except (OSError, ValueError):
    return None
  return (header, 8 + size) if isinstance(header, dict) else None


@functools.cache
def served_as_multimodal(base_model: str) -> bool:
  """Whether vLLM serves base_model through a multimodal class that keeps the language model under language_model."""
  if os.getenv("VLLM_ARCHITECTURE_OVERRIDE"):
    return False  # the sampler serves the class named there
  from transformers import AutoConfig

  try:
    architectures = AutoConfig.from_pretrained(base_model).architectures or []
  except (OSError, ValueError):
    return False
  return bool(MULTIMODAL_ARCHITECTURES.intersection(architectures))


def to_served_layout(adapter: Path, base_model: str) -> None:
  """Renames a text-model LoRA's weights to the layout vLLM serves the base model in. Only the header changes, so the
  tensor bytes are copied as they are."""
  path = adapter / "adapter_model.safetensors"
  parsed = safetensors_header(path)
  if parsed is None:
    return
  header, data_start = parsed
  names = [name for name in header if name != "__metadata__"]
  text_layout = any(name.startswith(TEXT_MODEL_PREFIX) for name in names) and not any(name.startswith(LANGUAGE_MODEL_PREFIX) for name in names)
  if not text_layout or not served_as_multimodal(base_model):
    return

  def served_name(name: str) -> str:
    return LANGUAGE_MODEL_PREFIX + name.removeprefix(TEXT_MODEL_PREFIX) if name.startswith(TEXT_MODEL_PREFIX) else name

  renamed = {served_name(name): value for name, value in header.items()}
  encoded = json.dumps(renamed, separators=(",", ":")).encode()
  encoded += b" " * (-len(encoded) % 8)  # tensor data stays 8-byte aligned
  rewritten = path.with_name(path.name + ".renamed")
  with open(path, "rb") as old, open(rewritten, "wb") as new:
    new.write(len(encoded).to_bytes(8, "little"))
    new.write(encoded)
    old.seek(data_start)
    shutil.copyfileobj(old, new)
  os.replace(rewritten, path)


def freeze_adapter(model_id: str, base_model: str, *refs: str | None) -> None:
  """Freezes the adapter the trainer just saved under each reference. The trainer's copy moves into place instead of
  being copied, because copying tens to hundreds of MiB over NFS held up a trainer that every job on a base model
  shares. The trainer writes a new copy on its next save."""
  paths = {snapshot_root() / snapshot_name(ref) for ref in refs if ref}
  source = Path(os.getenv("OPEN_RL_TMP_DIR", "/tmp/open-rl"), "peft", model_id, model_id)
  snapshot_root().mkdir(parents=True, exist_ok=True)
  with tempfile.TemporaryDirectory(prefix=".saving-", dir=snapshot_root()) as staging:
    staged = Path(staging, "adapter")
    source.rename(staged)
    # vLLM's filesystem resolver loads an adapter only when this names the
    # served model exactly, and a trainer may have loaded it from a local path.
    config_path = staged / "adapter_config.json"
    config = json.loads(config_path.read_text())
    config["base_model_name_or_path"] = base_model
    config_path.write_text(json.dumps(config))
    to_served_layout(staged, base_model)
    # Never overwrite a snapshot, because vLLM caches adapters and prefixes by name.
    # Saving the same weights under a name again, as an eval before and after no training does, changes nothing.
    if any(path.exists() and not same_files(staged, path) for path in paths):
      # Hands the copy back, so the trainer still has its weights to save under another name.
      staged.rename(source)
      raise FileExistsError("Sampler weights reference already exists; save with a new name or sequence ID")
    for path in paths:
      if not path.exists():
        copy = Path(staging, path.name)
        # Hard links, since nothing writes these files again.
        shutil.copytree(staged, copy, copy_function=os.link)
        copy.rename(path)
