#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cqy_workspace/tk/lerobot_src
PY_ENV=/home/cqy/miniconda3/envs/lerobot_tk
BASELINE=$ROOT/outputs/train/pi05_so101_local_10epochs_bs32/checkpoints/005613/pretrained_model
PHASE=${1:-full}

case "$PHASE" in
  smoke)
    STEPS=100
    SAVE_CHECKPOINT=false
    SAVE_FREQ=100
    OUTPUT_DIR=$ROOT/outputs/train/pi05_so101_vlash_expert_only_smoke100_bs32_seed1000
    ;;
  full)
    STEPS=5613
    SAVE_CHECKPOINT=true
    SAVE_FREQ=1123
    OUTPUT_DIR=$ROOT/outputs/train/pi05_so101_vlash_expert_only_10epochs_bs32_seed1000
    ;;
  *)
    echo "Usage: $0 [smoke|full]" >&2
    exit 2
    ;;
esac

if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Output directory already exists: $OUTPUT_DIR" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$ROOT/src"
export HF_HOME=/data/cqy_workspace/tk/hf_cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TMPDIR="$ROOT/tmp"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export LD_LIBRARY_PATH="$PY_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec "$PY_ENV/bin/python" -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=admin123/so101_test_data \
  --dataset.root="$ROOT/data/so101_test_data" \
  --dataset.video_backend=pyav \
  --dataset.eval_split=0.0 \
  --dataset.image_transforms.enable=false \
  --policy.path="$BASELINE" \
  --policy.state_cond=true \
  --policy.temporal_offset_max_steps=8 \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=false \
  --policy.train_expert_only=true \
  --policy.optimizer_lr=5e-6 \
  --policy.push_to_hub=false \
  --policy.fuse_qkv=false \
  --policy.fuse_gate_up=false \
  --output_dir="$OUTPUT_DIR" \
  --job_name="$(basename "$OUTPUT_DIR")" \
  --seed=1000 \
  --batch_size=32 \
  --num_workers=2 \
  --prefetch_factor=2 \
  --persistent_workers=true \
  --steps="$STEPS" \
  --log_freq=10 \
  --save_checkpoint="$SAVE_CHECKPOINT" \
  --save_freq="$SAVE_FREQ" \
  --env_eval_freq=0 \
  --eval_steps=0 \
  --wandb.enable=false
