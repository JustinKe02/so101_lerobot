# PI0.5 TensorRT Prefix Acceleration

This deployment path keeps Real-Time Chunking (RTC) unchanged. TensorRT runs
the two-camera vision encoder and PaliGemma prefix prefill, then returns the 18
layers of K/V cache to the PyTorch action expert. PyTorch remains responsible
for the denoising loop and RTC VJP.

## Requirements

- Build the engine on the same GPU model used for rollout.
- TensorRT 10 Python bindings must be available. `trtexec` is optional; the
  exporter falls back to the TensorRT Python Builder.
- ONNX is required while exporting.
- BF16 is the preferred precision for this checkpoint. Use FP16 if the GPU or
  TensorRT build does not support BF16.

Verify the deployment environment before exporting:

```bash
ROOT=/data/cqy_workspace/tk/lerobot_src
PY=/home/cqy/miniconda3/envs/lerobot_tk/bin/python
TRT_PYTHON=$ROOT/.pi05_tensorrt/python
TRT_ROOT=/data/cqy_workspace/third_party/tensorrt_10_13_0_35
export PYTHONPATH="$TRT_PYTHON:$TRT_ROOT:$ROOT/src"
export LD_LIBRARY_PATH="$TRT_ROOT/tensorrt_libs:${LD_LIBRARY_PATH:-}"

$PY -c 'import onnx, tensorrt; print(onnx.__version__, tensorrt.__version__)'
```

## Export And Build

```bash
ROOT=/data/cqy_workspace/tk/lerobot_src
PY=/home/cqy/miniconda3/envs/lerobot_tk/bin/python
MODEL=$ROOT/outputs/train/pi05_so101_local_10epochs_bs32/checkpoints/005613/pretrained_model

cd "$ROOT"
TRT_PYTHON=$ROOT/.pi05_tensorrt/python
TRT_ROOT=/data/cqy_workspace/third_party/tensorrt_10_13_0_35
export PYTHONPATH="$TRT_PYTHON:$TRT_ROOT:$ROOT/src"
export LD_LIBRARY_PATH="$TRT_ROOT/tensorrt_libs:${LD_LIBRARY_PATH:-}"
export HF_HOME=/data/cqy_workspace/tk/hf_cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

$PY -m lerobot.scripts.lerobot_export_pi05_tensorrt \
  --checkpoint="$MODEL" \
  --precision=bf16 \
  --build-engine
```

The default artifacts are:

```text
pretrained_model/pi05_tensorrt/
  prefix_cache.onnx
  prefix_cache.onnx.data
  prefix_cache_metadata.json
  prefix_cache_bf16.plan
```

If export and engine construction are done in separate environments:

```bash
$PY -m lerobot.scripts.lerobot_export_pi05_tensorrt \
  --checkpoint="$MODEL" \
  --precision=bf16

$PY -m lerobot.scripts.lerobot_export_pi05_tensorrt \
  --checkpoint="$MODEL" \
  --precision=bf16 \
  --skip-export \
  --build-engine
```

Before connecting the robot, compare the TensorRT K/V outputs with PyTorch:

```bash
$PY -m lerobot.scripts.lerobot_export_pi05_tensorrt \
  --checkpoint="$MODEL" \
  --precision=bf16 \
  --skip-export \
  --verify-engine
```

## Verification Thresholds

`--verify-engine` gates on four action-visible and three KV-diagnostic checks
(defaults, all recorded in `<engine>.plan.verified.json`):

| Check | Threshold | Nature |
| --- | --- | --- |
| action mean abs (10-step denoise, TRT KV) | `1e-2` | functional gate |
| action max abs (10-step denoise, TRT KV) | `1e-1` | functional gate |
| KV mean abs (worst layer) | `1e-1` | diagnostic |
| KV max abs (worst element) | `3.0` | diagnostic |
| KV outlier element fraction (`atol=rtol=5e-2`) | `5%` | diagnostic |

The KV max threshold was recalibrated from 2.0 to 3.0 on 2026-07-26 based on
measured behavior of the strongly-typed bf16 engine for the `005613` old-full
checkpoint: divergence concentrates in the deepest layers (`value_14..17`,
worst 2.44 at `value_17`) and grows with depth — the signature of differing
bf16 accumulation order between TensorRT fused attention and PyTorch SDPA,
not a weight or graph defect. Outlier elements measured 3.07%. Three
independent action-level measurements showed no functional footprint:
10-step denoise parity max `0.007` (14x margin), real-frame RTC parity
normalized max `<= 0.0039` across delays 2-6 (plan gate `0.020`), robot-unit
max `0.24` (5% of the per-tick hardware clamp). The outlier-fraction check
was added at the same time so a genuinely divergent engine cannot pass by
staying under the absolute caps alone. The RTC-level contract remains the
plan §8.2 normalized parity gate, checked by
`examples/inference/verify_pi05_tensorrt_rtc.py`.

## Rollout

`run_pi05_rollout.sh` checks for `prefix_cache_bf16.plan`, then
`prefix_cache_fp16.plan`, and enables TensorRT automatically. An explicit
engine can be selected with:

```bash
PI05_TRT_PREFIX_ENGINE=/path/to/prefix_cache_bf16.plan ./run_pi05_rollout.sh
```

The startup log must include:

```text
TensorRT PI0.5 prefix enabled: ... cameras=2, cache_layers=18
```

If this line is absent, the prefix is still running in PyTorch. The current
rollout uses 10 denoising steps. TensorRT does not replace the action expert or
the RTC gradient calculation in this backend.

The rollout script only auto-loads an engine with a newer
`.plan.verified.json` marker. Rebuilding an engine invalidates its previous
verification until `--verify-engine` succeeds again.

For the supervised T1 long run (TensorRT prefix + action output filter +
Q45 RTC), use `run_pi05_full_rollout_rtc_q45_t1_logged.sh`. It chains three
offline gates (filtered replay report, engine verification marker, §8.2 RTC
parity report) before any hardware is opened, enforces a 30-60 s duration,
and analyzes the log with TensorRT/filter expectations while keeping the
166.667 ms latency gate. `PI05_PREFLIGHT_ONLY=1
./run_pi05_full_rollout_rtc_q45_t1.sh` validates the whole chain without
touching the robot.
