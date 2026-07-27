#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/tk/lerobot_src_train
PY_ENV=/opt/conda/envs/lerobot

export CUDA_VISIBLE_DEVICES=1,2,3,4,5
export PYTHONPATH="$ROOT/src"
export HF_HOME=/workspace/tk/vla/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TMPDIR="$ROOT/tmp"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export LD_LIBRARY_PATH="$PY_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec "$PY_ENV/bin/accelerate" launch \
  --config_file="$ROOT/configs/pi05_fsdp_5gpu.yaml" \
  --main_process_port=29551 \
  --module lerobot.scripts.lerobot_train \
  --dataset.repo_id=admin123/so101_test_data \
  --dataset.root="$ROOT/data/so101_test_data" \
  --dataset.video_backend=pyav \
  --policy.type=pi05 \
  --policy.pretrained_path=/workspace/tk/model_assets/lerobot/pi05_base \
  --policy.input_features='{}' \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=false \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.push_to_hub=false \
  --output_dir="$ROOT/outputs/train/pi05_so101_1k_5gpu_bs16" \
  --job_name=pi05_so101_1k_5gpu_bs16 \
  --batch_size=16 \
  --num_workers=2 \
  --prefetch_factor=2 \
  --persistent_workers=true \
  --steps=1000 \
  --log_freq=5 \
  --save_checkpoint=true \
  --save_freq=250 \
  --env_eval_freq=0 \
  --eval_steps=0 \
  --wandb.enable=false
