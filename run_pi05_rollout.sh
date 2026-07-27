#!/usr/bin/env bash

set -euo pipefail

ROOT=/data/cqy_workspace/tk/lerobot_src
PY=/home/cqy/miniconda3/envs/lerobot_tk/bin/python
MODEL=/data/cqy_workspace/tk/lerobot_src/outputs/train/pi05_so101_local_10epochs_bs32/checkpoints/005613/pretrained_model
CALIB_ROOT=/data/cqy_workspace/tk/hf_cache/huggingface/lerobot/calibration
CALIB_DIR=$CALIB_ROOT/robots/so_follower
ROBOT_PORT=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123192-if00
TOP_CAMERA=/dev/video4
WRIST_CAMERA=/dev/video6

TRT_ARGS=()
is_verified_engine() {
  local engine=$1
  local marker=$engine.verified.json
  [[ -f "$engine" && -f "$marker" && "$marker" -nt "$engine" ]]
}

if [[ -n "${PI05_TRT_PREFIX_ENGINE:-}" ]]; then
  [[ -f "$PI05_TRT_PREFIX_ENGINE" ]] || {
    echo "TensorRT prefix engine not found: $PI05_TRT_PREFIX_ENGINE" >&2
    exit 1
  }
  if ! is_verified_engine "$PI05_TRT_PREFIX_ENGINE" && [[ "${PI05_ALLOW_UNVERIFIED_TRT:-0}" != 1 ]]; then
    echo "TensorRT prefix engine has not passed verification: $PI05_TRT_PREFIX_ENGINE" >&2
    echo "Run the exporter with --skip-export --verify-engine first." >&2
    exit 1
  fi
  TRT_ARGS+=(--pi05_tensorrt_prefix_engine="$PI05_TRT_PREFIX_ENGINE")
elif is_verified_engine "$MODEL/pi05_tensorrt/prefix_cache_bf16.plan"; then
  TRT_ARGS+=(--pi05_tensorrt_prefix_engine="$MODEL/pi05_tensorrt/prefix_cache_bf16.plan")
elif is_verified_engine "$MODEL/pi05_tensorrt/prefix_cache_fp16.plan"; then
  TRT_ARGS+=(--pi05_tensorrt_prefix_engine="$MODEL/pi05_tensorrt/prefix_cache_fp16.plan")
else
  echo "No verified TensorRT prefix engine found; using the PyTorch prefix backend." >&2
fi

[[ -x "$PY" ]] || { echo "Python executable not found: $PY" >&2; exit 1; }
[[ -f "$MODEL/model.safetensors" ]] || { echo "Model not found: $MODEL" >&2; exit 1; }
[[ -f "$CALIB_DIR/tk_follower.json" ]] || { echo "Calibration not found: $CALIB_DIR/tk_follower.json" >&2; exit 1; }
[[ -e "$ROBOT_PORT" ]] || { echo "Robot port not found: $ROBOT_PORT" >&2; exit 1; }
[[ -e "$TOP_CAMERA" ]] || { echo "Top camera not found: $TOP_CAMERA" >&2; exit 1; }
[[ -e "$WRIST_CAMERA" ]] || { echo "Wrist camera not found: $WRIST_CAMERA" >&2; exit 1; }

CAMERAS='{"top":{"type":"opencv","index_or_path":"/dev/video4","width":640,"height":480,"fps":30,"backend":200},"wrist":{"type":"opencv","index_or_path":"/dev/video6","width":640,"height":480,"fps":30,"backend":200}}'

export CUDA_VISIBLE_DEVICES=0
TRT_PYTHON=$ROOT/.pi05_tensorrt/python
TRT_ROOT=/data/cqy_workspace/third_party/tensorrt_10_13_0_35
if [[ -d "$TRT_PYTHON" && -d "$TRT_ROOT/tensorrt" ]]; then
  export PYTHONPATH="$TRT_PYTHON:$TRT_ROOT:$ROOT/src"
  export LD_LIBRARY_PATH="$TRT_ROOT/tensorrt_libs:${LD_LIBRARY_PATH:-}"
else
  export PYTHONPATH="$ROOT/src"
fi
export HF_HOME=/data/cqy_workspace/tk/hf_cache/huggingface
export HF_LEROBOT_CALIBRATION="$CALIB_ROOT"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH="/home/cqy/miniconda3/envs/lerobot_tk/lib:${LD_LIBRARY_PATH:-}"

cd "$ROOT"

exec "$PY" -m lerobot.scripts.lerobot_rollout \
  --strategy.type=base \
  --inference.type=rtc \
  --inference.rtc.execution_horizon=10 \
  --inference.rtc.max_guidance_weight=10.0 \
  --inference.rtc.prefix_attention_schedule=EXP \
  --inference.queue_threshold=20 \
  --interpolation_multiplier=1 \
  "${TRT_ARGS[@]}" \
  --policy.path="$MODEL" \
  --policy.gradient_checkpointing=false \
  --policy.num_inference_steps=10 \
  --robot.type=so101_follower \
  --robot.port="$ROBOT_PORT" \
  --robot.id=tk_follower \
  --robot.calibration_dir="$CALIB_DIR" \
  --robot.max_relative_target=5.0 \
  --robot.cameras="$CAMERAS" \
  --task="Put the block in the bin" \
  --duration=15 \
  --fps=30 \
  --device=cuda \
  --display_data=false \
  --play_sounds=false \
  --return_to_initial_position=true
