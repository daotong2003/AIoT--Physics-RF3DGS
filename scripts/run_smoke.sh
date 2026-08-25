#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_ROOT="${DATA_ROOT:-/data/aiot/raw}"
PROCESSED_ROOT="${PROCESSED_ROOT:-/data/aiot/processed_smoke}"
RUN_ROOT="${RUN_ROOT:-/runs/rf3dgs/smoke}"

cd "${PROJECT_ROOT}"
command -v "${PYTHON_BIN}" >/dev/null || {
  echo "找不到Python解释器：${PYTHON_BIN}" >&2
  exit 1
}

"${PYTHON_BIN}" prepare_rf3dgs_scene.py \
  --data-root "${DATA_ROOT}" \
  --output-root "${PROCESSED_ROOT}" \
  --max-gaussians 5000 \
  --fine-voxel-m 0.5 \
  --coarse-voxel-m 1.0 \
  --object-spacing-m 0.25

"${PYTHON_BIN}" train_physics_rf3dgs.py \
  --processed-root "${PROCESSED_ROOT}" \
  --output-root "${RUN_ROOT}" \
  --grid-step-m 4.0 \
  --smoke-test

"${PYTHON_BIN}" evaluate_inverse_localization.py \
  --processed-root "${PROCESSED_ROOT}" \
  --model-dir "${RUN_ROOT}" \
  --output-root "${RUN_ROOT}/evaluation" \
  --smoke-test
