#!/usr/bin/env bash
# SLURM job: generate publication-quality visualisation PNGs.
#
# Reads cluster config from .env in the project root.
# Writes ecg_waveforms.png, psd_comparison.png, text_neighbors.png
# to FIGURES_DIR (or --output-dir override).
#
# Submit: sbatch scripts/visualize.sh
# Custom checkpoint: sbatch scripts/visualize.sh --checkpoint step_0050000.pt
# ──────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=gdss_viz
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --output=logs/visualize_%j.out
#SBATCH --error=logs/visualize_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
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

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR="${LOG_DIR:-$PROJECT_DIR/logs}"
mkdir -p "$LOG_DIR"
exec >"$LOG_DIR/visualize_${SLURM_JOB_ID:-local}.out" \
    2>"$LOG_DIR/visualize_${SLURM_JOB_ID:-local}.err"

# ── Activate virtualenv ───────────────────────────────────────────────────────
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
source "$VENV_DIR/bin/activate"

# ── GPU diagnostics ───────────────────────────────────────────────────────────
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ── Run ───────────────────────────────────────────────────────────────────────
cd "$PROJECT_DIR"

CKPT="${CHECKPOINT_DIR:-checkpoints}/final.pt"

echo "=== visualize  $(date) ==="
echo "DATA_DIR    = ${DATA_DIR:-data/ptbxl}"
echo "CACHE_DIR   = ${CACHE_DIR:-cache}"
echo "FIGURES_DIR = ${FIGURES_DIR:-figures}"
echo "CHECKPOINT  = $CKPT"
echo "LOG_DIR     = $LOG_DIR"

EXTRA_ARGS="${@:-}"

python apps/visualize/main.py \
    --checkpoint  "$CKPT" \
    --data-dir    "${DATA_DIR:-data/ptbxl}" \
    --cache-dir   "${CACHE_DIR:-cache}" \
    --output-dir  "${FIGURES_DIR:-figures}" \
    --n-real      400 \
    --n-gen       300 \
    --device      cuda \
    --sampler     pc \
    --n-steps     500 \
    --cfg-scale   1.5 \
    $EXTRA_ARGS

echo "=== done $(date) ==="
