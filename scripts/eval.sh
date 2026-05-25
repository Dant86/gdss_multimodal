#!/usr/bin/env bash
# SLURM job: sampler × NFE evaluation grid.
#
# Reads cluster config from .env in the project root.
# Generates samples, computes ECG-FID / text cosine sim / joint quality,
# and saves a grouped bar chart to FIGURES_DIR.
#
# Submit: sbatch scripts/eval.sh
# Custom checkpoint: sbatch scripts/eval.sh --checkpoint step_0050000
#
# Log paths are set after sourcing .env — SLURM directives can't read env files,
# so we start with /dev/null and exec-redirect once LOG_DIR is known.
# ──────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=gdss_eval
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
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
exec > "${LOG_DIR}/eval_${SLURM_JOB_ID}.out" \
     2> "${LOG_DIR}/eval_${SLURM_JOB_ID}.err"

# ── Activate virtualenv ───────────────────────────────────────────────────────
source "${VENV_DIR}/bin/activate"

# ── GPU diagnostics ───────────────────────────────────────────────────────────
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ── Run ───────────────────────────────────────────────────────────────────────
echo "[$(date)] Starting eval (job ${SLURM_JOB_ID})"
echo "DATA_DIR       = ${DATA_DIR}"
echo "CACHE_DIR      = ${CACHE_DIR}"
echo "CHECKPOINT_DIR = ${CHECKPOINT_DIR}"
echo "FIGURES_DIR    = ${FIGURES_DIR}"
echo "LOG_DIR        = ${LOG_DIR}"

python apps/eval/main.py \
    --data-dir        "${DATA_DIR}" \
    --cache-dir       "${CACHE_DIR}" \
    --checkpoint-dir  "${CHECKPOINT_DIR}" \
    --figures-dir     "${FIGURES_DIR}" \
    --checkpoint      final \
    --n-samples       1000 \
    --nfe             100 500 1000 \
    --samplers        s4 pc em \
    --corrector-snr   0.16 \
    --cfg-scale       1.5 \
    --device          cuda \
    "$@"

echo "[$(date)] Done."
