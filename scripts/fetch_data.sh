#!/usr/bin/env bash
# SLURM job: download, translate, and process PTB-XL.
#
# Reads cluster config from .env in the project root.
# Typical runtime: 15–30 min (concurrent sync API calls, concurrency=50).
#
# Submit: sbatch scripts/fetch_data.sh
#
# Log paths are set after sourcing .env — SLURM directives can't read env files,
# so we start with /dev/null and exec-redirect once LOG_DIR is known.
# ──────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=gdss_fetch
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
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
exec > "${LOG_DIR}/fetch_data_${SLURM_JOB_ID}.out" \
     2> "${LOG_DIR}/fetch_data_${SLURM_JOB_ID}.err"

# ── Activate virtualenv ───────────────────────────────────────────────────────
source "${VENV_DIR}/bin/activate"

# ── Run ───────────────────────────────────────────────────────────────────────
echo "[$(date)] Starting fetch_data (job ${SLURM_JOB_ID})"
echo "DATA_DIR  = ${DATA_DIR}"
echo "CACHE_DIR = ${CACHE_DIR}"
echo "LOG_DIR   = ${LOG_DIR}"

python apps/fetch_data/main.py \
    --data-dir  "${DATA_DIR}" \
    --cache-dir "${CACHE_DIR}" \
    --bert-device cpu \
    "$@"

echo "[$(date)] Done."
