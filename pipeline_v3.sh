#!/usr/bin/env bash
# pipeline_v3.sh — R-peak conditioned model pipeline:
#   1. Pretrain ECGUNet with R-peak conditioning (30K steps)
#   2. DSM train with R-peak + lead + CFG (100K steps)
#   3. Sample 1000 ECGs at 72 bpm, Lead II, cfg_scale=1.5
#   4. Visualize → figures/12lead_v3_rpeak/
#
# Usage: bash pipeline_v3.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv/bin/modal"

echo "=== V3 R-peak pipeline started at $(date) ==="

# ── Step 1: pretrain ────────────────────────────────────────────────────────
echo ""
echo ">>> [v3] Pretraining new ECGUNet (12-lead + CFG + R-peak)…"
"$VENV" run "$SCRIPT_DIR/pretrain.py"

# ── Step 2: DSM train ───────────────────────────────────────────────────────
echo ""
echo ">>> [v3] DSM training (12-lead + CFG + loss clamp + R-peak)…"
"$VENV" run "$SCRIPT_DIR/train.py" \
    --pretrain-checkpoint /vol/checkpoints/pretrain_final.pt \
    --max-steps 100000

# ── Step 3: sample ──────────────────────────────────────────────────────────
echo ""
echo ">>> [v3] Generating 1000 samples (Lead II, s4, NFE=1000, cfg=1.5, HR=72bpm)…"
"$VENV" run "$SCRIPT_DIR/sample.py" \
    --checkpoint final \
    --sampler s4 \
    --n-steps 1000 \
    --n-samples 1000 \
    --batch-size 32 \
    --cfg-scale 1.5 \
    --heart-rate-bpm 72.0

# ── Step 4: visualize ───────────────────────────────────────────────────────
echo ""
echo ">>> [v3] Generating figures → figures/12lead_v3_rpeak/"
"$VENV" run "$SCRIPT_DIR/visualize.py" \
    --checkpoint final \
    --n-gen 300 \
    --tag 12lead_v3_rpeak

echo ""
echo "=== V3 pipeline complete at $(date) ==="
echo "    figures/12lead_v3_rpeak/ — 12-lead + CFG + R-peak conditioning"
