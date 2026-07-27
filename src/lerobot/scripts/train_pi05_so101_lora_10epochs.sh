#!/usr/bin/env bash
set -Eeuo pipefail

# S2: PI0.5 LoRA fine-tuning for the 40-episode SO-101 dataset.
# The adapter targets are intentionally explicit. In particular, the model's
# modules are named time_mlp_in/out, not action_time_mlp_in/out; this S2 plan
# keeps the planned 38 targets and does not adapt the time MLP.

readonly ROOT=/data/cqy_workspace/tk/lerobot_src
readonly PY_ENV=/home/cqy/miniconda3/envs/lerobot_tk
readonly PY="$PY_ENV/bin/python"
readonly BASE_MODEL=/data/cqy_workspace/tk/model_assets/lerobot/pi05_base
readonly DATASET_ROOT="$ROOT/data/so101_test_data"
readonly JOB_NAME=pi05_so101_lora_r16_alpha16_10epochs_bs32_seed1000
readonly OUTPUT_DIR="${PI05_LORA_OUTPUT_DIR:-$ROOT/outputs/train/$JOB_NAME}"
readonly STATE_DIR="$ROOT/outputs/train/.run_state/$JOB_NAME"
readonly LORA_R=16
readonly LORA_ALPHA=16
readonly LORA_LR=1e-4
readonly TARGET_MODULES='(.*\.gemma_expert\..*\.self_attn\.(q|v)_proj|model\.(action_in_proj|action_out_proj))'

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -x "$PY" ]] || fail "Python executable not found: $PY"
[[ -f "$BASE_MODEL/config.json" ]] || fail "Base model config not found: $BASE_MODEL/config.json"
[[ -f "$BASE_MODEL/model.safetensors" ]] || fail "Base model weights not found: $BASE_MODEL/model.safetensors"
[[ -f "$DATASET_ROOT/meta/info.json" ]] || fail "Dataset metadata not found: $DATASET_ROOT/meta/info.json"
[[ -f "$DATASET_ROOT/data/chunk-000/file-000.parquet" ]] || fail "Dataset parquet file not found"
[[ -f "$DATASET_ROOT/videos/observation.images.top/chunk-000/file-000.mp4" ]] || fail "Top camera video not found"
[[ -f "$DATASET_ROOT/videos/observation.images.wrist/chunk-000/file-000.mp4" ]] || fail "Wrist camera video not found"

export TARGET_MODULES
export LORA_R

# Validate the target regex and the exact adapter size without loading the
# 9-GB checkpoint into GPU memory. This catches renamed modules or an omitted
# projection before a multi-hour run starts.
"$PY" - "$BASE_MODEL/model.safetensors" <<'PY'
import os
import re
import sys
from safetensors import safe_open

path = sys.argv[1]
pattern = re.compile(os.environ["TARGET_MODULES"])
rank = int(os.environ["LORA_R"])

with safe_open(path, framework="pt", device="cpu") as handle:
    keys = list(handle.keys())
    module_names = {
        "model." + key.rsplit(".", 1)[0]
        for key in keys
        if key.endswith(".weight")
    }
    matched = sorted(name for name in module_names if pattern.fullmatch(name))
    missing_weights = []
    adapter_params = 0
    for name in matched:
        state_key = name.removeprefix("model.") + ".weight"
        if state_key not in keys:
            missing_weights.append(state_key)
            continue
        out_features, in_features = handle.get_slice(state_key).get_shape()
        adapter_params += rank * (out_features + in_features)

if missing_weights:
    raise SystemExit("Missing target weights: " + ", ".join(missing_weights))
if len(matched) != 38:
    raise SystemExit(f"Expected 38 LoRA targets, found {len(matched)}: {matched}")
if adapter_params != 1_287_168:
    raise SystemExit(f"Expected 1,287,168 LoRA parameters, found {adapter_params}")
if any("time_mlp" in name for name in matched):
    raise SystemExit("The S2 target set must not include time_mlp modules")
if not any(name.endswith("model.action_in_proj") for name in matched):
    raise SystemExit("action_in_proj was not matched")
if not any(name.endswith("model.action_out_proj") for name in matched):
    raise SystemExit("action_out_proj was not matched")

print(f"LoRA target preflight passed: targets={len(matched)}, trainable_adapter_params={adapter_params}")
PY

if ! "$PY" -c 'import peft; print(f"PEFT {peft.__version__}")'; then
  fail "The training environment cannot import peft"
fi

if [[ -e "$OUTPUT_DIR" ]]; then
  fail "Output directory already exists: $OUTPUT_DIR"
fi

if [[ "${PI05_LORA_PREFLIGHT_ONLY:-0}" == 1 ]]; then
  echo "S2 LoRA preflight passed"
  echo "base_model=$BASE_MODEL"
  echo "output_dir=$OUTPUT_DIR"
  echo "targets=$TARGET_MODULES"
  echo "r=$LORA_R alpha=$LORA_ALPHA lr=$LORA_LR"
  exit 0
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
  --policy.train_expert_only=false
  --policy.optimizer_lr="$LORA_LR"
  --policy.optimizer_weight_decay=0.01
  --policy.optimizer_grad_clip_norm=1.0
  --policy.scheduler_warmup_steps=1000
  --policy.scheduler_decay_steps=30000
  --policy.scheduler_decay_lr=1e-5
  --policy.push_to_hub=false
  --peft.method_type=LORA
  --peft.target_modules="$TARGET_MODULES"
  --peft.full_training_modules=[]
  --peft.r="$LORA_R"
  --peft.lora_alpha="$LORA_ALPHA"
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
  --save_freq=1123
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
