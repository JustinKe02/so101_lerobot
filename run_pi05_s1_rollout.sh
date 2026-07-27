#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cqy_workspace/tk/lerobot_src
PY=/home/cqy/miniconda3/envs/lerobot_tk/bin/python
MODEL=${PI05_MODEL:-$ROOT/outputs/train/pi05_so101_expert_only_10epochs_bs32_seed1000/checkpoints/005613/pretrained_model}
CALIB_ROOT=/data/cqy_workspace/tk/hf_cache/huggingface/lerobot/calibration
CALIB_DIR=$CALIB_ROOT/robots/so_follower
ROBOT_PORT=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123192-if00
TOP_CAMERA=/dev/video4
WRIST_CAMERA=/dev/video6
DURATION=${PI05_DURATION:-30}
MAX_RELATIVE_TARGET=${PI05_MAX_RELATIVE_TARGET:-5.0}
USE_TRT=${PI05_USE_TRT:-0}
PREFLIGHT_ONLY=${PI05_PREFLIGHT_ONLY:-0}
TRT_ENGINE=${PI05_TRT_PREFIX_ENGINE:-$MODEL/pi05_tensorrt/prefix_cache_bf16.plan}
TRT_REPORT=$TRT_ENGINE.dataset_parity.json

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -x "$PY" ]] || fail "Python executable not found: $PY"
[[ -f "$MODEL/model.safetensors" ]] || fail "Model not found: $MODEL"
[[ -f "$CALIB_DIR/tk_follower.json" ]] || fail "Calibration not found: $CALIB_DIR/tk_follower.json"
[[ -e "$ROBOT_PORT" ]] || fail "Robot port not found: $ROBOT_PORT"
[[ -r "$ROBOT_PORT" && -w "$ROBOT_PORT" ]] || fail "Robot port is not readable and writable: $ROBOT_PORT"
[[ -r "$TOP_CAMERA" ]] || fail "Top camera is not readable: $TOP_CAMERA"
[[ -r "$WRIST_CAMERA" ]] || fail "Wrist camera is not readable: $WRIST_CAMERA"

if fuser "$ROBOT_PORT" >/dev/null 2>&1; then
  fail "Robot port is already in use: $ROBOT_PORT"
fi
if fuser "$TOP_CAMERA" >/dev/null 2>&1; then
  fail "Top camera is already in use: $TOP_CAMERA"
fi
if fuser "$WRIST_CAMERA" >/dev/null 2>&1; then
  fail "Wrist camera is already in use: $WRIST_CAMERA"
fi

TRT_ARGS=()
BACKEND=pytorch
if [[ "$USE_TRT" == 1 ]]; then
  [[ -f "$TRT_ENGINE" ]] || fail "TensorRT engine not found: $TRT_ENGINE"
  [[ -f "$TRT_REPORT" ]] || fail "TensorRT dataset parity report not found: $TRT_REPORT"
  [[ "$TRT_REPORT" -nt "$TRT_ENGINE" ]] || fail "TensorRT parity report is older than the engine"
  "$PY" - "$TRT_ENGINE" "$TRT_REPORT" <<'PY'
import json
import sys
from pathlib import Path

engine = Path(sys.argv[1]).resolve()
report_path = Path(sys.argv[2]).resolve()
report = json.loads(report_path.read_text())
if Path(report.get("engine", "")).resolve() != engine:
    raise SystemExit("TensorRT parity report references a different engine")
if not report.get("aggregate", {}).get("passed", False):
    raise SystemExit("TensorRT dataset parity report did not pass")
PY
  TRT_ARGS+=(--pi05_tensorrt_prefix_engine="$TRT_ENGINE")
  BACKEND=tensorrt
fi

CAMERAS='{"top":{"type":"opencv","index_or_path":"/dev/video4","width":640,"height":480,"fps":30,"backend":200},"wrist":{"type":"opencv","index_or_path":"/dev/video6","width":640,"height":480,"fps":30,"backend":200}}'

export CUDA_VISIBLE_DEVICES=0
TRT_PYTHON=$ROOT/.pi05_tensorrt/python
TRT_ROOT=/data/cqy_workspace/third_party/tensorrt_10_13_0_35
if [[ "$USE_TRT" == 1 ]]; then
  [[ -d "$TRT_PYTHON" ]] || fail "TensorRT Python environment not found: $TRT_PYTHON"
  [[ -d "$TRT_ROOT/tensorrt" ]] || fail "TensorRT installation not found: $TRT_ROOT"
  export PYTHONPATH="$TRT_PYTHON:$TRT_ROOT:$ROOT/src"
  export LD_LIBRARY_PATH="$TRT_ROOT/tensorrt_libs:${LD_LIBRARY_PATH:-}"
else
  export PYTHONPATH="$ROOT/src"
fi
export HF_HOME=/data/cqy_workspace/tk/hf_cache/huggingface
export HF_LEROBOT_CALIBRATION="$CALIB_ROOT"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH="/home/cqy/miniconda3/envs/lerobot_tk/lib:${LD_LIBRARY_PATH:-}"

echo "S1 rollout preflight passed"
echo "model=$MODEL"
echo "backend=$BACKEND"
echo "duration=$DURATION"
echo "max_relative_target=$MAX_RELATIVE_TARGET"

if [[ "$PREFLIGHT_ONLY" == 1 ]]; then
  exit 0
fi

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
  --robot.max_relative_target="$MAX_RELATIVE_TARGET" \
  --robot.cameras="$CAMERAS" \
  --task="Put the block in the bin" \
  --duration="$DURATION" \
  --fps=30 \
  --device=cuda \
  --display_data=false \
  --play_sounds=false \
  --return_to_initial_position=true
