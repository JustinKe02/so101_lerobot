#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cqy_workspace/tk/lerobot_src
PY=/home/cqy/miniconda3/envs/lerobot_tk/bin/python
MODEL=$ROOT/outputs/train/pi05_so101_local_10epochs_bs32/checkpoints/005613/pretrained_model
CALIB_ROOT=/data/cqy_workspace/tk/hf_cache/huggingface/lerobot/calibration
CALIB_DIR=$CALIB_ROOT/robots/so_follower
ROBOT_PORT=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123192-if00
TOP_CAMERA=/dev/video4
WRIST_CAMERA=/dev/video6

DURATION=${PI05_DURATION:-5}
MAX_ACTIONS=${PI05_MAX_ACTIONS_PER_CHUNK:-30}
CLAMP_THRESHOLD=${PI05_CLAMP_REPLAN_THRESHOLD:-5.0}
MAX_RELATIVE_TARGET=${PI05_MAX_RELATIVE_TARGET:-5.0}
RETURN_TO_INITIAL=${PI05_RETURN_TO_INITIAL_POSITION:-true}
LOG_DIR=${PI05_SYNC_GUARD_LOG_DIR:-$ROOT/outputs/rollout/sync_guarded}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_PATH=$LOG_DIR/sync_guarded_${TIMESTAMP}.log

[[ -x "$PY" ]] || { echo "Python executable not found: $PY" >&2; exit 1; }
[[ -f "$MODEL/model.safetensors" ]] || { echo "Model not found: $MODEL" >&2; exit 1; }
[[ -f "$CALIB_DIR/tk_follower.json" ]] || { echo "Calibration not found: $CALIB_DIR/tk_follower.json" >&2; exit 1; }

CAMERAS='{"top":{"type":"opencv","index_or_path":"/dev/video4","width":640,"height":480,"fps":30,"backend":200},"wrist":{"type":"opencv","index_or_path":"/dev/video6","width":640,"height":480,"fps":30,"backend":200}}'

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$ROOT/src"
export HF_HOME=/data/cqy_workspace/tk/hf_cache/huggingface
export HF_LEROBOT_CALIBRATION="$CALIB_ROOT"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH="/home/cqy/miniconda3/envs/lerobot_tk/lib:${LD_LIBRARY_PATH:-}"

COMMAND=(
  "$PY" -m lerobot.scripts.lerobot_rollout
  --strategy.type=base
  --inference.type=sync
  --inference.max_actions_per_chunk="$MAX_ACTIONS"
  --inference.replan_on_clamp=true
  --inference.clamp_replan_threshold="$CLAMP_THRESHOLD"
  --interpolation_multiplier=1
  --pi05_prefix_backend=pytorch
  --pi05_action_backend=pytorch
  --policy.path="$MODEL"
  --policy.gradient_checkpointing=false
  --policy.n_action_steps=50
  --policy.num_inference_steps=10
  --robot.type=so101_follower
  --robot.port="$ROBOT_PORT"
  --robot.id=tk_follower
  --robot.calibration_dir="$CALIB_DIR"
  --robot.max_relative_target="$MAX_RELATIVE_TARGET"
  --robot.cameras="$CAMERAS"
  --task="Put the block in the bin"
  --duration="$DURATION"
  --fps=30
  --seed=1000
  --device=cuda
  --display_data=false
  --play_sounds=false
  --return_to_initial_position="$RETURN_TO_INITIAL"
)

echo "policy=old_full_005613"
echo "inference=sync_guarded"
echo "prefix_backend=pytorch"
echo "action_backend=pytorch"
echo "max_actions_per_chunk=$MAX_ACTIONS"
echo "replan_on_clamp=true"
echo "clamp_replan_threshold=$CLAMP_THRESHOLD"
echo "max_relative_target=$MAX_RELATIVE_TARGET"
echo "duration=$DURATION"
echo "return_to_initial_position=$RETURN_TO_INITIAL"

if [[ "${PI05_PREFLIGHT_ONLY:-0}" == 1 || "${PI05_PREFLIGHT_ONLY:-0}" == true ]]; then
  printf 'command='
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

[[ -e "$ROBOT_PORT" ]] || { echo "Robot port not found: $ROBOT_PORT" >&2; exit 1; }
[[ -r "$ROBOT_PORT" && -w "$ROBOT_PORT" ]] || {
  echo "Robot port is not readable/writable: $ROBOT_PORT" >&2
  echo "Add the current user to dialout or fix the device udev permissions before running." >&2
  exit 1
}
[[ -e "$TOP_CAMERA" ]] || { echo "Top camera not found: $TOP_CAMERA" >&2; exit 1; }
[[ -e "$WRIST_CAMERA" ]] || { echo "Wrist camera not found: $WRIST_CAMERA" >&2; exit 1; }

mkdir -p "$LOG_DIR"
echo "log=$LOG_PATH"
cd "$ROOT"

set +e
"${COMMAND[@]}" 2>&1 | tee "$LOG_PATH"
STATUS=${PIPESTATUS[0]}
set -e

echo "rollout_exit_code=$STATUS"
echo "log=$LOG_PATH"
exit "$STATUS"
