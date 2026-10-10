#!/usr/bin/env bash
# Run inside a provisioned AMD ROCm + LeRobot 0.4.2 Python environment.
set -euo pipefail

dataset_root="${1:?usage: amd_cloud_smoke.sh DATASET_ROOT REPO_ID CHECKPOINT OUTPUT_DIR}"
repo_id="${2:?usage: amd_cloud_smoke.sh DATASET_ROOT REPO_ID CHECKPOINT OUTPUT_DIR}"
checkpoint="${3:?usage: amd_cloud_smoke.sh DATASET_ROOT REPO_ID CHECKPOINT OUTPUT_DIR}"
output_dir="${4:?usage: amd_cloud_smoke.sh DATASET_ROOT REPO_ID CHECKPOINT OUTPUT_DIR}"

python scripts/verify_lerobot_bundle.py --dataset "$dataset_root" --repo-id "$repo_id" \
  --checkpoint "$checkpoint" --require-rocm
LEROBOT_DEVICE=cuda LEROBOT_STEPS=10 LEROBOT_BATCH_SIZE=2 \
  LEROBOT_NUM_WORKERS=0 LEROBOT_SAVE_FREQ=10 \
  bash scripts/train_act_no_vae.sh "$dataset_root" "$repo_id" "$output_dir"
