#!/usr/bin/env bash
set -euo pipefail

dataset_root="${1:?usage: train_act_no_vae.sh DATASET_ROOT REPO_ID OUTPUT_DIR}"
repo_id="${2:?usage: train_act_no_vae.sh DATASET_ROOT REPO_ID OUTPUT_DIR}"
output_dir="${3:?usage: train_act_no_vae.sh DATASET_ROOT REPO_ID OUTPUT_DIR}"
device="${LEROBOT_DEVICE:-cuda}"
steps="${LEROBOT_STEPS:-4000}"
batch_size="${LEROBOT_BATCH_SIZE:-16}"
num_workers="${LEROBOT_NUM_WORKERS:-4}"
save_freq="${LEROBOT_SAVE_FREQ:-1000}"

lerobot-train \
  --dataset.repo_id="$repo_id" \
  --dataset.root="$dataset_root" \
  --dataset.video_backend=pyav \
  --policy.type=act \
  --policy.device="$device" \
  --policy.push_to_hub=false \
  --policy.dim_model=256 \
  --policy.n_heads=4 \
  --policy.dim_feedforward=1024 \
  --policy.n_encoder_layers=2 \
  --policy.chunk_size=32 \
  --policy.use_vae=false \
  --policy.n_action_steps=8 \
  --batch_size="$batch_size" \
  --num_workers="$num_workers" \
  --steps="$steps" \
  --save_freq="$save_freq" \
  --log_freq=100 \
  --seed=20261010 \
  --output_dir="$output_dir" \
  --wandb.enable=false
