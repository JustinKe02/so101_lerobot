#!/usr/bin/env bash
set -Eeuo pipefail

readonly ROOT=/data/cqy_workspace/tk/lerobot_src
readonly PY_ENV=/home/cqy/miniconda3/envs/lerobot_tk
readonly PY="$PY_ENV/bin/python"
readonly BASE_MODEL=/data/cqy_workspace/tk/model_assets/lerobot/pi05_base
readonly DATASET_ROOT="$ROOT/data/so101_test_data"
readonly JOB_NAME=pi05_so101_expert_only_10epochs_5epoch_ckpt_seed1000
readonly OUTPUT_DIR="$ROOT/outputs/train/$JOB_NAME"
readonly STATE_DIR="$ROOT/outputs/train/.run_state/$JOB_NAME"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -x "$PY" ]] || fail "Python executable not found: $PY"
[[ -f "$BASE_MODEL/config.json" ]] || fail "Base model config not found: $BASE_MODEL/config.json"
[[ -f "$BASE_MODEL/model.safetensors" ]] || fail "Base model weights not found: $BASE_MODEL/model.safetensors"
[[ -f "$DATASET_ROOT/meta/info.json" ]] || fail "Dataset metadata not found: $DATASET_ROOT/meta/info.json"
[[ ! -e "$OUTPUT_DIR" ]] || fail "Output directory already exists: $OUTPUT_DIR"

if [[ -f "$STATE_DIR/run.pid" ]]; then
  previous_pid=$(<"$STATE_DIR/run.pid")
  if [[ "$previous_pid" =~ ^[0-9]+$ ]] && kill -0 "$previous_pid" 2>/dev/null; then
    fail "Training is already running with PID $previous_pid"
  fi
fi

mkdir -p "$ROOT/tmp" "$(dirname "$OUTPUT_DIR")" "$STATE_DIR"
rm -f "$STATE_DIR/exit_code" "$STATE_DIR/finished_at"
date --iso-8601=seconds >"$STATE_DIR/started_at"
echo "starting" >"$STATE_DIR/status"
git -C "$ROOT" rev-parse HEAD >"$STATE_DIR/git_commit.txt"
git -C "$ROOT" status --short >"$STATE_DIR/git_status.txt"

on_exit() {
  rc=$?
  echo "$rc" >"$STATE_DIR/exit_code"
  date --iso-8601=seconds >"$STATE_DIR/finished_at"
  if [[ $rc -eq 0 ]]; then
    echo "completed" >"$STATE_DIR/status"
  else
    echo "failed" >"$STATE_DIR/status"
  fi
}
trap on_exit EXIT

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$ROOT/src"
export HF_HOME=/data/cqy_workspace/tk/hf_cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TMPDIR="$ROOT/tmp"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export LD_LIBRARY_PATH="$PY_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

command=(
  "$PY" -m lerobot.scripts.lerobot_train
  --dataset.repo_id=admin123/so101_test_data
  --dataset.root="$DATASET_ROOT"
  --dataset.video_backend=pyav
  --dataset.eval_split=0.0
  --policy.type=pi05
  --policy.pretrained_path="$BASE_MODEL"
  --policy.input_features={}
  --policy.device=cuda
  --policy.dtype=bfloat16
  --policy.gradient_checkpointing=true
  --policy.compile_model=false
  --policy.freeze_vision_encoder=false
  --policy.train_expert_only=true
  --policy.optimizer_lr=2.5e-5
  --policy.optimizer_weight_decay=0.01
  --policy.optimizer_grad_clip_norm=1.0
  --policy.scheduler_warmup_steps=1000
  --policy.scheduler_decay_steps=30000
  --policy.scheduler_decay_lr=2.5e-6
  --policy.push_to_hub=false
  --output_dir="$OUTPUT_DIR"
  --job_name="$JOB_NAME"
  --resume=false
  --seed=1000
  --cudnn_deterministic=false
  --batch_size=32
  --num_workers=2
  --prefetch_factor=2
  --persistent_workers=true
  --steps=5613
  --log_freq=10
  --save_checkpoint=true
  --save_freq=2806
  --env_eval_freq=0
  --eval_steps=0
  --wandb.enable=false
)

printf '%q ' "${command[@]}" >"$STATE_DIR/command.txt"
printf '\n' >>"$STATE_DIR/command.txt"

echo "running" >"$STATE_DIR/status"
"${command[@]}" &
child_pid=$!
echo "$child_pid" >"$STATE_DIR/run.pid"

forward_signal() {
  if kill -0 "$child_pid" 2>/dev/null; then
    kill -TERM "$child_pid"
  fi
}
trap forward_signal INT TERM

wait "$child_pid"
