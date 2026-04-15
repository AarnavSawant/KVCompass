#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-${ROOT_DIR}/results/benchmark_eval}"
FRACTION="${FRACTION:-1.0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [matrix|budget4k|budget8k|assignments]

Examples:
  $(basename "$0") matrix
  RESULTS_DIR=/path/to/benchmark_eval $(basename "$0") budget4k
EOF
}

MODE="${1:-matrix}"

cd "${ROOT_DIR}"
mkdir -p logs "${RESULTS_DIR}"

"${PYTHON_BIN}" scripts/write_pace_configs.py --dest-dir configs --output-dir "${RESULTS_DIR}" --fraction "${FRACTION}"

submit_job() {
  local config_path="$1"
  local label="$2"
  echo "Submitting ${label} using ${config_path}"
  sbatch \
    --export=ALL,REPO_ROOT="${ROOT_DIR}",CONFIG_PATH="${config_path}",RESULTS_DIR="${RESULTS_DIR}",LOG_LABEL="${label}",FRACTION="${FRACTION}",PYTHON_BIN="${PYTHON_BIN}" \
    scripts/pace_sweep.sbatch
}

case "${MODE}" in
  matrix)
    submit_job "configs/benchmark_sweeps.pace_full_matrix.yaml" "pace-full-matrix"
    ;;
  budget4k)
    submit_job "configs/benchmark_sweeps.pace_budget_sensitivity_4k.yaml" "pace-budget-4k"
    ;;
  budget8k)
    submit_job "configs/benchmark_sweeps.pace_budget_sensitivity_8k.yaml" "pace-budget-8k"
    ;;
  assignments)
    for assignment in 1 2 3 4 5; do
      submit_job "configs/benchmark_sweeps.pace_assignment_${assignment}.yaml" "pace-assignment-${assignment}"
    done
    ;;
  *)
    usage
    exit 1
    ;;
esac
