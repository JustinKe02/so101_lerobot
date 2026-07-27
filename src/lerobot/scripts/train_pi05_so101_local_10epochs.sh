#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cqy_workspace/tk/lerobot_src
PY_ENV=/home/cqy/miniconda3/envs/lerobot_tk

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
  --policy.type=pi05 \
  --policy.pretrained_path=/data/cqy_workspace/tk/model_assets/lerobot/pi05_base \
  --policy.input_features='{}' \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=false \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.push_to_hub=false \
  --output_dir="$ROOT/outputs/train/pi05_so101_local_10epochs_bs32" \
  --job_name=pi05_so101_local_10epochs_bs32 \
  --batch_size=32 \
  --num_workers=2 \
  --prefetch_factor=2 \
  --persistent_workers=true \
  --steps=5613 \
  --log_freq=10 \
  --save_checkpoint=true \
  --save_freq=1123 \
  --env_eval_freq=0 \
  --eval_steps=0 \
  --wandb.enable=false
