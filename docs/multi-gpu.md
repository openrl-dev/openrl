# Multi-GPU Training and Sampling

OpenRL scales across GPUs on Kubernetes with no changes to your training loop.
Set `openrl.*` tags in `TINKER_TAGS` (or `user_metadata` on `create_model`):

```bash
export TINKER_TAGS=openrl.trainer_backend=automodel,openrl.trainer_device_count=4,openrl.trainer_cp=4,openrl.sampler_replicas=4,openrl.recipe=rl
```

That single line runs a 4-GPU FSDP2 + CP4 Automodel trainer alongside 4
prefix-routed vLLM sampler replicas — the exact 8-GPU layout used by the
[Harvey LAB RL recipe](../examples/harvey_labs/README.md).

## Settings

| Setting | Values | Default | Effect |
| --- | --- | --- | --- |
| `openrl.trainer_backend` | `pytorch`, `automodel`, or image ref | `pytorch` | `automodel` runs NeMo Automodel (FSDP2 + context parallelism) for multi-GPU LoRA. |
| `openrl.trainer_device_count` | `1`–`8` | `1` | GPUs claimed on one node for a single-node `torchrun` trainer (`openrl.trainer_gpus` is still accepted as a deprecated alias). |
| `openrl.trainer_cp` | `1`–`8` | `1` | GPUs sharing one sequence (context parallelism); the remaining `trainer_device_count / trainer_cp` GPUs run data parallel. |
| `openrl.sampler_replicas` | `1`–`8` | `1` | Single-GPU `vllm serve` Workloads behind a CPU dispatcher + prefix-aware router (Envoy + llm-d). |
| `openrl.exclusive` | `true` / `false` | `false` | Dedicated trainer and sampler with no runtime sharing or GPU time-slicing. |
| `openrl.recipe` | `rl` / `sft` | unset | `rl` starts sampler replicas together with the trainer instead of waiting for the first sample call. |

## Common layouts

| Goal | `TINKER_TAGS` | GPUs |
| --- | --- | --- |
| Default LoRA (shared workers) | *(none)* | 2 (1 trainer + 1 sampler) |
| Sampling-bound RL | `openrl.sampler_replicas=4,openrl.recipe=rl` | 5 (1 trainer + 4 samplers) |
| Sharded SFT trainer (e.g. 27B on 4×L4) | `openrl.trainer_backend=automodel,openrl.trainer_device_count=4` | 4 (4 trainer) |
| Long-context RL (Harvey LAB on 8×H100 / 8×B200) | `openrl.trainer_backend=automodel,openrl.trainer_device_count=4,openrl.trainer_cp=4,openrl.sampler_replicas=4,openrl.recipe=rl` | 8 (4 CP4 trainer + 4 samplers) |

## Restrictions and exclusivity

- **Multi-GPU trainers are always exclusive.** `openrl.trainer_device_count > 1`
  gives the job its own trainer and sampling pool — no sharing with other LoRA
  jobs and no GPU time-slicing.
- **Automodel + LoRA only.** `openrl.trainer_device_count > 1` requires
  `openrl.trainer_backend=automodel` and `openrl.fine_tuning_type=lora`. Full
  fine-tuning (`openrl.fine_tuning_type=full`) is 1 GPU only.
- **Single-node trainer (2–8 GPUs).** All trainer GPUs land on one node, and
  `openrl.trainer_cp` must divide `openrl.trainer_device_count`.
- **1 GPU per sampler replica.** Samplers scale horizontally (`1`–`8` replicas),
  not via tensor parallelism, so the model and its KV cache must fit one GPU.
- **Sampler exclusivity:**
  - **FFT:** `openrl.sampler_replicas > 1` requires `openrl.exclusive=true`.
  - **LoRA (`trainer_device_count > 1`):** already exclusive; all replicas serve
    only that job.
  - **LoRA (`trainer_device_count=1`, non-exclusive):** jobs on the same base
    model can share a multi-replica sampling pool (sized to the max
    `sampler_replicas` requested); each replica still holds its GPU exclusively.
- **Kubernetes scheduler only.** `trainer_device_count > 1` and
  `sampler_replicas > 1` require `OPEN_RL_WORKER_MANAGER=scheduler` (not
  supported locally).
- **Automodel image.** `openrl.trainer_backend=automodel` pulls
  `OPEN_RL_AUTOMODEL_IMAGE` (default
  `ghcr.io/openrl-dev/openrl/automodel:latest`), separate from the default
  worker image.

## Example: Harvey LAB on 8 GPUs

The [Harvey LAB RL recipe](../examples/harvey_labs/README.md) trains
`Qwen/Qwen3.5-9B` with GRPO over a 262,144-token context window (4 CP4 trainer
GPUs + 4 prefix-routed vLLM samplers), lifting the held-out criterion pass rate
from 0.54 to 0.78 in 35 steps.

1. Create an 8-GPU DRA node pool (see the 8×H100 snippet in
   [GKE setup](setup/gke-setup.md)).
2. Raise the API server's context window limits before creating workers:
   ```bash
   kubectl -n openrl-system set env deployment/open-rl-api-server \
     VLLM_MAX_MODEL_LEN=262144 OPEN_RL_TRAIN_TOKEN_BUDGET=262144
   ```
3. Run the recipe with the 8-GPU tags:
   ```bash
   export TINKER_API_KEY=tml-dummy-key
   export TINKER_TAGS=openrl.trainer_backend=automodel,openrl.trainer_device_count=4,openrl.trainer_cp=4,openrl.sampler_replicas=4,openrl.recipe=rl
   uv --project examples run harvey-train base_url=http://127.0.0.1:9003 log_path=artifacts/harvey-labs/my-run
   ```

For the full in-cluster deployment (driver Job, Agent Sandbox warm pool, and
judge), see [Harvey LAB RL on GKE](setup/harvey-lab-on-gke.md).

## Verifying placement and utilization

```bash
# Cluster view: multi-GPU DRA claims and sampler replicas
kubectl -n openrl-system get workloads,resourceclaims,pods -o wide

# Client view (no kubectl needed; <model_id> is printed after create_model)
curl -s http://127.0.0.1:9003/api/v1/openrl/models/<model_id> | jq .summary
```

For per-GPU assignments and DCGM utilization, use the
[inspection API](configuration.md#inspection-api) (`/api/v1/inspect/activity`
and `/api/v1/inspect/query`).
