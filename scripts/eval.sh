#!/usr/bin/env bash
# SLURM job: sampler × NFE evaluation grid.
#
# Reads cluster config from .env in the project root.
# Generates samples, computes ECG-FID / text cosine sim / joint quality,
# and saves a grouped bar chart to FIGURES_DIR.
#
# Submit: sbatch scripts/eval.sh
# Custom checkpoint: sbatch scripts/eval.sh --checkpoint step_0050000
# ──────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=gdss_eval
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

set -euo pipefail

# ── Load environment ──────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$PROJECT_DIR/.env"

if [[ -f "$ENV_FILE" ]]; then
    set -a; source "$ENV_FILE"; set +a
fi

[[ -n "${SLURM_PARTITION:-}" ]] && SBATCH_PARTITION="$SLURM_PARTITION"
[[ -n "${SLURM_ACCOUNT:-}"   ]] && SBATCH_ACCOUNT="$SLURM_ACCOUNT"

# ── Activate virtualenv ───────────────────────────────────────────────────────
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
source "$VENV_DIR/bin/activate"

mkdir -p "$PROJECT_DIR/logs"

# ── GPU diagnostics ───────────────────────────────────────────────────────────
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ── Run ───────────────────────────────────────────────────────────────────────
cd "$PROJECT_DIR"

echo "=== eval  $(date) ==="
echo "DATA_DIR        = ${DATA_DIR:-data/ptbxl}"
echo "CACHE_DIR       = ${CACHE_DIR:-cache}"
echo "CHECKPOINT_DIR  = ${CHECKPOINT_DIR:-checkpoints}"
echo "FIGURES_DIR     = ${FIGURES_DIR:-figures}"

EXTRA_ARGS="${@:-}"

python apps/eval/main.py \
    --data-dir        "${DATA_DIR:-data/ptbxl}" \
    --cache-dir       "${CACHE_DIR:-cache}" \
    --checkpoint-dir  "${CHECKPOINT_DIR:-checkpoints}" \
    --figures-dir     "${FIGURES_DIR:-figures}" \
    --checkpoint      final \
    --n-samples       1000 \
    --nfe             100 500 1000 \
    --samplers        s4 pc em \
    --corrector-snr   0.16 \
    --cfg-scale       1.5 \
    --device          cuda \
    $EXTRA_ARGS

echo "=== done $(date) ==="
