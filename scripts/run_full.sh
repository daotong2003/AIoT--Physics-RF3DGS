#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/data/aiot/raw}"
PROCESSED_ROOT="${PROCESSED_ROOT:-/data/aiot/processed}"
RUN_ROOT="${RUN_ROOT:-/runs/rf3dgs/run_001}"

cd "${PROJECT_ROOT}"
python prepare_rf3dgs_scene.py \
  --data-root "${DATA_ROOT}" \
  --output-root "${PROCESSED_ROOT}" \
  --max-gaussians 200000 \
  --fine-voxel-m 0.10 \
  --coarse-voxel-m 0.20 \
  --object-spacing-m 0.08

python train_physics_rf3dgs.py \
  --processed-root "${PROCESSED_ROOT}" \
  --output-root "${RUN_ROOT}" \
  --grid-step-m 0.25 \
  --batch-size 64 \
  --patience-steps 3000 \
  --seed 19

python evaluate_inverse_localization.py \
  --processed-root "${PROCESSED_ROOT}" \
  --model-dir "${RUN_ROOT}" \
  --output-root "${RUN_ROOT}/evaluation" \
  --top-k 32 \
  --refine-steps 50

