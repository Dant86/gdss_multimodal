#!/usr/bin/env bash
# SLURM job: generate publication-quality visualisation PNGs.
#
# Reads cluster config from .env in the project root.
# Writes ecg_waveforms.png, psd_comparison.png, text_neighbors.png
# to FIGURES_DIR (or --output-dir override).
#
# Submit: sbatch scripts/visualize.sh
# Custom checkpoint: sbatch scripts/visualize.sh --checkpoint step_0050000.pt
#
# Log paths are set after sourcing .env — SLURM directives can't read env files,
# so we start with /dev/null and exec-redirect once LOG_DIR is known.
# ──────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=gdss_viz
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -euo pipefail

export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR"

if [[ ! -f .env ]]; then
    echo "ERROR: .env not found in ${SLURM_SUBMIT_DIR}. Copy .env.sample and fill in your values." >&2
    exit 1
fi
source .env

[[ -n "${SLURM_PARTITION:-}" ]] && SBATCH_PARTITION="$SLURM_PARTITION"
[[ -n "${SLURM_ACCOUNT:-}"   ]] && SBATCH_ACCOUNT="$SLURM_ACCOUNT"

mkdir -p "${LOG_DIR}"
exec > "${LOG_DIR}/visualize_${SLURM_JOB_ID}.out" \
     2> "${LOG_DIR}/visualize_${SLURM_JOB_ID}.err"

# ── Activate virtualenv ───────────────────────────────────────────────────────
source "${VENV_DIR}/bin/activate"

# ── GPU diagnostics ───────────────────────────────────────────────────────────
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ── Run ───────────────────────────────────────────────────────────────────────
echo "[$(date)] Starting visualize (job ${SLURM_JOB_ID})"
echo "DATA_DIR    = ${DATA_DIR}"
echo "CACHE_DIR   = ${CACHE_DIR}"
echo "FIGURES_DIR = ${FIGURES_DIR}"
echo "LOG_DIR     = ${LOG_DIR}"

python apps/visualize/main.py \
    --checkpoint  "${CHECKPOINT_DIR}/final.pt" \
    --data-dir    "${DATA_DIR}" \
    --cache-dir   "${CACHE_DIR}" \
    --output-dir  "${FIGURES_DIR}" \
    --n-real      400 \
    --n-gen       300 \
    --device      cuda \
    --sampler     pc \
    --n-steps     500 \
    --cfg-scale   1.5 \
    "$@"

echo "[$(date)] Done."
