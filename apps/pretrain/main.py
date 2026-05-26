"""Autoencoder pre-training for ECGUNet.

Trains ECGUNet as a pure reconstructor (zero conditioning) on clean PTB-XL
ECGs before the joint DSM fine-tuning phase.  This gives the U-Net a strong
ECG-morphology prior — QRS complexes, 1/f PSD, baseline wander — before
the denoising task introduces the diffusion objective.

Loss: L = MSE(recon, ecg) + λ_spec · MSE(|FFT(recon)|/L, |FFT(ecg)|/L)

After training, pass the checkpoint to apps/train via --pretrain-checkpoint
(or cfg.train.pretrain_checkpoint) so that DSM fine-tuning starts from
morphology-aware weights.

Environment variables (loaded from .env):
    DATA_DIR         Processed PTB-XL directory.
    CACHE_DIR        BERT embeddings and ECG stats.
    CHECKPOINT_DIR   Where to save checkpoints.
    HF_TOKEN         Optional HuggingFace token.

Usage
-----
    python apps/pretrain/main.py [--config experiments/pretrain.yaml]
                                 [--data-dir DIR] [--cache-dir DIR]
                                 [--checkpoint-dir DIR]
                                 [--max-steps 30000] [--batch-size 512]
                                 [--lr 1e-3] [--spectral-weight 0.1]
                                 [--device cuda] [--seed 42]
                                 [--resume CHECKPOINT_PATH]
                                 [KEY=VALUE ...]
"""

from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path
from typing import Callable, Optional

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import ConcatDataset, DataLoader

import gdss_multimodal.config as config_module
import gdss_multimodal.data as data_module
import gdss_multimodal.models as models_module

# cuDNN sublibrary version mismatch on some cluster GPU nodes — disable and
# fall back to native CUDA kernels (still fast on H200/A100).
torch.backends.cudnn.enabled = False


# ---------------------------------------------------------------------------
# Spectral loss
# ---------------------------------------------------------------------------

def _spectral_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """FFT magnitude MSE, normalised by sequence length.

    Encourages matching the power spectral density (1/f profile for ECGs).

    Args:
        pred:   Reconstructed tensor of shape (B, n_leads, L).
        target: Ground-truth tensor of the same shape.

    Returns:
        Scalar MSE over normalised FFT magnitudes.
    """
    L = pred.shape[-1]
    p = torch.fft.rfft(pred.float(), dim=-1).abs() / L
    t = torch.fft.rfft(target.float(), dim=-1).abs() / L
    return F.mse_loss(p, t)


# ---------------------------------------------------------------------------
# Core training function
# ---------------------------------------------------------------------------

def pretrain(
    cfg: config_module.Config,
    on_checkpoint: Optional[Callable[[int], None]] = None,
) -> str:
    """Run the ECGUNet autoencoder pre-training loop.

    Uses all PTB-XL splits (train + val + test) since we are learning ECG
    morphology rather than fitting the joint distribution.

    Args:
        cfg: Experiment configuration (reads cfg.pretrain and cfg.ecg_score).
        on_checkpoint: Optional callback invoked after each checkpoint save.

    Returns:
        Path to the final pretrain checkpoint as a string.
    """
    pcfg = cfg.pretrain
    ecfg = cfg.ecg_score
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    _seed(cfg.train.seed)

    multi_lead = ecfg.lead_emb_dim > 0
    print(f"Loading datasets… (multi_lead={multi_lead})")
    train_ds, val_ds, test_ds = data_module.build_datasets(
        cfg.train.data_dir,
        cfg.train.data_cache_dir,
        bert_device=str(device),
        multi_lead=multi_lead,
    )
    # Use all splits — pretraining only learns ECG morphology
    all_ds = ConcatDataset([train_ds, val_ds, test_ds])
    loader = DataLoader(
        all_ds,
        batch_size=pcfg.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    print("Building ECGUNet…")
    model = models_module.ECGUNet(
        text_dim=ecfg.text_dim,
        n_leads=ecfg.n_leads,
        seq_len=ecfg.seq_len,
        timestep_dim=ecfg.timestep_dim,
        channels=ecfg.channels,
        bottleneck_ch=ecfg.bottleneck_ch,
        lead_emb_dim=ecfg.lead_emb_dim,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ECGUNet params: {n_params / 1e6:.2f}M")

    if pcfg.resume:
        resume_path = Path(pcfg.resume)
        print(f"  Resuming from {resume_path}…")
        state = torch.load(resume_path, map_location=device, weights_only=True)
        model.load_state_dict(state["s_theta"])

    use_amp = device.type == "cuda"
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else torch.autocast("cpu")

    optimiser = AdamW(model.parameters(), lr=pcfg.lr, weight_decay=pcfg.weight_decay)
    scheduler = OneCycleLR(
        optimiser,
        max_lr=pcfg.lr,
        total_steps=pcfg.max_steps,
        pct_start=pcfg.warmup_steps / max(pcfg.max_steps, 1),
        anneal_strategy="cos",
    )

    ckpt_dir = Path(pcfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    loss_ema: Optional[float] = None
    model.train()
    loader_iter = iter(loader)
    t0 = time.time()
    print(f"Pre-training on {device}, max_steps={pcfg.max_steps}, bf16={use_amp}")

    while step < pcfg.max_steps:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        ecg = batch["ecg"].to(device)
        lead_idx = batch.get("lead_idx")
        if lead_idx is not None:
            lead_idx = lead_idx.to(device)

        optimiser.zero_grad()
        with autocast:
            recon = model.reconstruct(ecg, lead_idx)
            mse = F.mse_loss(recon, ecg)
            spec = _spectral_loss(recon, ecg)
            loss = mse + pcfg.spectral_weight * spec

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), pcfg.grad_clip)
        optimiser.step()
        scheduler.step()
        step += 1

        raw = loss.item()
        loss_ema = raw if loss_ema is None else 0.98 * loss_ema + 0.02 * raw

        if step % pcfg.log_every == 0:
            elapsed = time.time() - t0
            print(
                f"step {step:6d} | loss {raw:.5f} | ema {loss_ema:.5f} "
                f"| mse {mse.item():.5f} | spec {spec.item():.5f} "
                f"| {elapsed / 60:.1f} min"
            )

        if step % pcfg.save_every == 0 or step == pcfg.max_steps:
            _save_pretrain(model, optimiser, step, ckpt_dir)
            if on_checkpoint:
                on_checkpoint(step)

    final_path = _save_pretrain(model, optimiser, step, ckpt_dir, name="pretrain_final")
    print(f"Pre-training complete → {final_path}")
    return str(final_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_pretrain(model, optimiser, step, ckpt_dir, name=None):
    """Save a pretrain checkpoint and return its path."""
    fname = name or f"pretrain_step_{step:07d}"
    path = Path(ckpt_dir) / f"{fname}.pt"
    torch.save(
        {
            "step": step,
            "s_theta": model.state_dict(),
            "optimiser": optimiser.state_dict(),
        },
        path,
    )
    print(f"  checkpoint saved → {path}")
    return path


def _seed(seed: int) -> None:
    """Seed all relevant RNGs for reproducibility."""
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ECGUNet autoencoder pre-training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="", help="Path to YAML config file.")
    parser.add_argument("--data-dir",        default=os.environ.get("DATA_DIR", "data/ptbxl"))
    parser.add_argument("--cache-dir",       default=os.environ.get("CACHE_DIR", "cache"))
    parser.add_argument("--checkpoint-dir",  default=os.environ.get("CHECKPOINT_DIR", "checkpoints"))
    parser.add_argument("--max-steps",       type=int,   default=None)
    parser.add_argument("--batch-size",      type=int,   default=None)
    parser.add_argument("--lr",              type=float, default=None)
    parser.add_argument("--spectral-weight", type=float, default=None)
    parser.add_argument("--device",          default=None)
    parser.add_argument("--seed",            type=int,   default=None)
    parser.add_argument("--resume",          default=None, help="Path to pretrain checkpoint to resume.")
    # Allow dot-notation overrides: e.g.  ecg_score.channels=64,128,256
    parser.add_argument("overrides", nargs="*", metavar="KEY=VALUE")
    return parser.parse_args()


if __name__ == "__main__":
    load_dotenv()

    args = _parse_args()

    cfg = config_module.Config.from_yaml(args.config) if args.config else config_module.Config()

    # Path / device overrides from CLI flags
    cfg.train.data_dir       = args.data_dir
    cfg.train.data_cache_dir = args.cache_dir
    cfg.pretrain.checkpoint_dir = args.checkpoint_dir
    if args.max_steps       is not None: cfg.pretrain.max_steps       = args.max_steps
    if args.batch_size      is not None: cfg.pretrain.batch_size      = args.batch_size
    if args.lr              is not None: cfg.pretrain.lr              = args.lr
    if args.spectral_weight is not None: cfg.pretrain.spectral_weight = args.spectral_weight
    if args.device          is not None: cfg.train.device             = args.device
    if args.seed            is not None: cfg.train.seed               = args.seed
    if args.resume          is not None: cfg.pretrain.resume          = args.resume

    # Dot-notation key=value overrides (e.g. ecg_score.channels=64,128,256)
    if args.overrides:
        cfg.override({k: v for k, v in (o.split("=", 1) for o in args.overrides)})

    pretrain(cfg)
