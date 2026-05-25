#!/usr/bin/env bash
# SLURM job: download, translate, and process PTB-XL.
#
# Reads cluster config from .env in the project root.
# Typical runtime: 30–90 min depending on batch API latency.
#
# Submit: sbatch scripts/fetch_data.sh
# ──────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=gdss_fetch
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

# ── Load environment ──────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$PROJECT_DIR/.env"

if [[ -f "$ENV_FILE" ]]; then
    set -a; source "$ENV_FILE"; set +a
fi

# Apply SLURM-level overrides for partition / account if provided.
[[ -n "${SLURM_PARTITION:-}" ]] && SBATCH_PARTITION="$SLURM_PARTITION"
[[ -n "${SLURM_ACCOUNT:-}"   ]] && SBATCH_ACCOUNT="$SLURM_ACCOUNT"

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR="${LOG_DIR:-$PROJECT_DIR/logs}"
mkdir -p "$LOG_DIR"
exec >"$LOG_DIR/fetch_data_${SLURM_JOB_ID:-local}.out" \
    2>"$LOG_DIR/fetch_data_${SLURM_JOB_ID:-local}.err"

# ── Activate virtualenv ───────────────────────────────────────────────────────
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
source "$VENV_DIR/bin/activate"

# ── Run ───────────────────────────────────────────────────────────────────────
cd "$PROJECT_DIR"

echo "=== fetch_data  $(date) ==="
echo "DATA_DIR  = ${DATA_DIR:-data/ptbxl}"
echo "CACHE_DIR = ${CACHE_DIR:-cache}"
echo "LOG_DIR   = $LOG_DIR"

python apps/fetch_data/main.py \
    --data-dir  "${DATA_DIR:-data/ptbxl}" \
    --cache-dir "${CACHE_DIR:-cache}" \
    --bert-device cpu

echo "=== done $(date) ==="
