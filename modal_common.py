"""Shared Modal infrastructure: app, image, volumes, remote paths.

Image strategy
--------------
The image is split into two layers:

  base_image  — all pip installs (slow, ~10 min to build, rarely changes)
  image       — base_image + local source files (fast, rebuilds on every
                code change but takes only a few seconds)

Modal caches each layer by its content hash.  As long as package versions in
base_image don't change, only the fast layer rebuilds when you edit code.
To force a base rebuild (e.g. after bumping a package version), run:
    modal run train.py   # Modal detects the hash change and rebuilds

Run
---
    modal run train.py
    modal run sample.py    -- --checkpoint final --sampler s4 --n-steps 1000
    modal run evaluate.py  -- --checkpoint final --nfe 100,500,1000
"""
from __future__ import annotations

from pathlib import Path

import modal

PROJECT_ROOT = Path(__file__).parent

app = modal.App("gdss-multimodal")

# ---------------------------------------------------------------------------
# Base image — pip installs only, no local files
# Rebuilds only when package versions change.
# ---------------------------------------------------------------------------
base_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04",
        add_python="3.11",
    )
    .pip_install(
        "torch==2.4.0+cu121",
        "torchvision==0.19.0+cu121",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    # momentfm pins old transformers/huggingface_hub — install without deps
    # and add its only non-torch runtime dep manually.
    .pip_install("momentfm", extra_options="--no-deps")
    .pip_install(
        "einops",
        "transformers==4.44.2",   # last version compatible with torch 2.4
        "huggingface_hub>=0.22.0",
        "accelerate>=0.29.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "pandas>=2.0.0",
        "wfdb>=4.1.0",
    )
    .env({"PYTHONPATH": "/app"})
)

# ---------------------------------------------------------------------------
# Final image — base + local source (rebuilds in seconds on code changes)
# ---------------------------------------------------------------------------
image = base_image.add_local_dir(
    PROJECT_ROOT,
    remote_path="/app",
    ignore=["cache", "checkpoints", "samples", ".git", "__pycache__", "*.pyc"],
)

# ---------------------------------------------------------------------------
# Persistent volumes
# ---------------------------------------------------------------------------
cache_vol   = modal.Volume.from_name("gdss-cache",       create_if_missing=True)
ckpt_vol    = modal.Volume.from_name("gdss-checkpoints", create_if_missing=True)
samples_vol = modal.Volume.from_name("gdss-samples",     create_if_missing=True)

REMOTE_CACHE   = "/vol/cache"
REMOTE_CKPTS   = "/vol/checkpoints"
REMOTE_SAMPLES = "/vol/samples"

# HuggingFace model cache lives in the volume so MOMENT-1-large (~7 GB) and
# BioClinicalBERT are downloaded once and reused across all runs.
HF_CACHE_DIR = f"{REMOTE_CACHE}/huggingface"

VOLUME_MAP = {
    REMOTE_CACHE:   cache_vol,
    REMOTE_CKPTS:   ckpt_vol,
    REMOTE_SAMPLES: samples_vol,
}

try:
    HF_SECRETS = [modal.Secret.from_name("huggingface-secret")]
except Exception:
    HF_SECRETS = []

GPU = "H100"
