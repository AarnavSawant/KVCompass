#!/bin/bash
# Usage:
#   sbatch --export=ALL,CONFIG_PATH=configs/benchmark_sweeps.pace_full_matrix.yaml,RESULTS_DIR=/path/to/results,LOG_LABEL=pace_full_matrix scripts/pace_sweep.sbatch

#SBATCH --job-name=kvpress-sweep
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
CONFIG_PATH="${CONFIG_PATH:?CONFIG_PATH must be set}"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/results/benchmark_eval}"
LOG_LABEL="${LOG_LABEL:-$(basename "${CONFIG_PATH}" .yaml)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "${REPO_ROOT}"
mkdir -p logs "${RESULTS_DIR}"

echo "[PACE] Host: $(hostname)"
echo "[PACE] Working directory: ${REPO_ROOT}"
echo "[PACE] Config: ${CONFIG_PATH}"
echo "[PACE] Results dir: ${RESULTS_DIR}"
echo "[PACE] Label: ${LOG_LABEL}"

"${PYTHON_BIN}" scripts/write_pace_configs.py --dest-dir configs --output-dir "${RESULTS_DIR}" --fraction "${FRACTION:-1.0}"
"${PYTHON_BIN}" scripts/run_kvpress_benchmark_sweep.py --config "${CONFIG_PATH}"
