"""Joint (ECG, text) diffusion model training loop.

Trains s_θ (ECGUNet) and s_φ (TextScoreNet) jointly via denoising score
matching with VP-SDE.  Supports single-GPU and multi-GPU DDP, mixed-precision
bfloat16, EMA, and linear warmup + cosine annealing.

Environment variables (loaded from .env):
    DATA_DIR         Processed PTB-XL directory.
    CACHE_DIR        BERT embeddings and ECG stats.
    CHECKPOINT_DIR   Where to save checkpoints.
    HF_TOKEN         Optional HuggingFace token.

Usage
-----
    python apps/train/main.py [--config experiments/train.yaml]
                              [--data-dir DIR] [--cache-dir DIR]
                              [--checkpoint-dir DIR]
                              [--max-steps 100000] [--batch-size 256]
                              [--lr 3e-4] [--device cuda]
                              [--pretrain-checkpoint PATH]
                              [--resume-checkpoint PATH]
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
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from dotenv import load_dotenv
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler

import gdss_multimodal.config as config_module
import gdss_multimodal.data as data_module
import gdss_multimodal.models as models_module
import gdss_multimodal.sde as sde_module


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _unwrap(model):
    """Strip DataParallel / DDP wrapper, returning the base module."""
    return model.module if isinstance(model, (nn.DataParallel, DDP)) else model


def _init_ema(model) -> dict:
    """Initialise an EMA shadow as a clone of the current state dict."""
    return {k: v.clone() for k, v in _unwrap(model).state_dict().items()}


def _update_ema(ema: dict, model, decay: float = 0.999) -> None:
    """Update EMA shadow in-place: shadow = decay·shadow + (1−decay)·param."""
    with torch.no_grad():
        for k, v in _unwrap(model).state_dict().items():
            ema[k].mul_(decay).add_(v, alpha=1.0 - decay)


def _seed(seed: int) -> None:
    """Seed all relevant RNGs for reproducibility."""
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# DSM loss
# ---------------------------------------------------------------------------

def dsm_loss(
    s_theta,
    s_phi,
    ecg0: torch.Tensor,
    text0: torch.Tensor,
    vpsde,
    likelihood_weighting: bool = True,
    lead_idx: Optional[torch.Tensor] = None,
    cfg_drop_prob: float = 0.0,
) -> torch.Tensor:
    """Joint denoising score matching loss with optional Min-SNR weighting.

    Args:
        s_theta: ECG score network.
        s_phi: Text score network.
        ecg0: Clean ECG batch of shape (B, n_leads, seq_len).
        text0: Clean text embedding batch of shape (B, text_dim).
        vpsde: VP-SDE instance.
        likelihood_weighting: If True, weight by σ²·min(SNR, 5).
        lead_idx: Optional lead indices of shape (B,) for lead conditioning.
        cfg_drop_prob: Probability of nulling lead conditioning (CFG drop).

    Returns:
        Scalar DSM loss.
    """
    B, device = ecg0.shape[0], ecg0.device
    t = torch.empty(B, device=device).uniform_(vpsde.eps, vpsde.T)

    eps1, eps2 = torch.randn_like(ecg0), torch.randn_like(text0)
    mean1, std1 = vpsde.marginal_prob(ecg0, t)
    mean2, std2 = vpsde.marginal_prob(text0, t)
    ecg_t  = mean1 + std1 * eps1
    text_t = mean2 + std2 * eps2

    # Classifier-free guidance: randomly null the lead conditioning.
    cfg_lead = lead_idx
    if lead_idx is not None and cfg_drop_prob > 0.0:
        drop_mask = torch.rand(B, device=device) < cfg_drop_prob
        cfg_lead = lead_idx.clone()
        cfg_lead[drop_mask] = -1   # sentinel → _cond zeroes the lead slot

    score_ecg  = s_theta(ecg_t, text_t, t, cfg_lead)
    score_text = s_phi(text_t, _ecg_rep(s_theta, ecg_t, text_t, t, cfg_lead), t)

    shape1 = (-1,) + (1,) * (ecg0.dim() - 1)
    shape2 = (-1,) + (1,) * (text0.dim() - 1)
    target_ecg  = -eps1 / std1.view(shape1).clamp(min=1e-8)
    target_text = -eps2 / std2.view(shape2).clamp(min=1e-8)

    loss_ecg  = (score_ecg  - target_ecg)  ** 2
    loss_text = (score_text - target_text) ** 2

    if likelihood_weighting:
        # Min-SNR weighting (Hang et al. 2023, γ=5).
        # w = σ² · min(SNR, γ) — peaks near the SNR=γ crossover.
        _GAMMA = 5.0
        alpha_t = vpsde.alpha(t)
        sigma_t = vpsde.sigma(t).clamp(min=1e-8)
        snr = (alpha_t / sigma_t).pow(2)
        w = sigma_t.pow(2) * snr.clamp(max=_GAMMA)
        loss_ecg  = w.view(shape1) * loss_ecg
        loss_text = w.view(shape2) * loss_text

    return loss_ecg.mean() + loss_text.mean()


def _ecg_rep(s_theta, ecg_t, text_t, t, lead_idx=None):
    """Mean-pooled, L2-normalised ECG bottleneck (detached, no text bias)."""
    m = _unwrap(s_theta)
    with torch.no_grad():
        h = m.encode(ecg_t, m.t_embed(t), lead_idx)
    return h.detach()


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train(cfg: config_module.Config, on_checkpoint: Optional[Callable[[int], None]] = None) -> None:
    """Run the full training loop, spawning DDP workers when >1 GPU is present.

    Args:
        cfg: Experiment configuration.
        on_checkpoint: Callback invoked after each checkpoint save (single-GPU).
    """
    use_cuda = cfg.train.device == "cuda" and torch.cuda.is_available()
    world_size = torch.cuda.device_count() if use_cuda else 1

    if world_size > 1:
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "12355")
        mp.spawn(_train_worker, args=(world_size, cfg), nprocs=world_size, join=True)
    else:
        _train_worker(0, 1, cfg, on_checkpoint=on_checkpoint)


def _train_worker(
    rank: int,
    world_size: int,
    cfg: config_module.Config,
    on_checkpoint: Optional[Callable[[int], None]] = None,
) -> None:
    """Per-process training worker (single GPU)."""
    ddp = world_size > 1
    if ddp:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

    device   = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    is_main  = rank == 0

    if is_main:
        torch.backends.cudnn.benchmark = True

    _seed(cfg.train.seed + rank)

    # ── Data ─────────────────────────────────────────────────────────────────
    multi_lead = cfg.ecg_score.lead_emb_dim > 0
    if is_main:
        print(f"Loading datasets… (multi_lead={multi_lead})")
    train_ds, val_ds, _ = data_module.build_datasets(
        cfg.train.data_dir,
        cfg.train.data_cache_dir,
        bert_device=f"cuda:{rank}" if torch.cuda.is_available() else "cpu",
        multi_lead=multi_lead,
    )

    train_sampler = (
        DistributedSampler(train_ds, num_replicas=world_size, rank=rank,
                           shuffle=True, drop_last=True)
        if ddp else None
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
    ) if is_main else None

    # ── Models ───────────────────────────────────────────────────────────────
    if is_main:
        print("Building models…")
    s_theta = models_module.ECGUNet(
        text_dim=cfg.ecg_score.text_dim,
        n_leads=cfg.ecg_score.n_leads,
        seq_len=cfg.ecg_score.seq_len,
        timestep_dim=cfg.ecg_score.timestep_dim,
        channels=cfg.ecg_score.channels,
        bottleneck_ch=cfg.ecg_score.bottleneck_ch,
        lead_emb_dim=cfg.ecg_score.lead_emb_dim,
    ).to(device)
    s_phi = models_module.TextScoreNet(
        text_dim=cfg.text_score.text_dim,
        moment_hidden=cfg.text_score.moment_hidden_dim,
        timestep_dim=cfg.text_score.timestep_embed_dim,
        hidden_dim=cfg.text_score.hidden_dim,
        n_layers=cfg.text_score.n_layers,
    ).to(device)
    vpsde = sde_module.VPSDE(cfg.sde.beta_min, cfg.sde.beta_max, cfg.sde.T, cfg.sde.eps)

    if is_main:
        n1 = sum(p.numel() for p in s_theta.parameters() if p.requires_grad)
        n2 = sum(p.numel() for p in s_phi.parameters()   if p.requires_grad)
        print(f"  ECGUNet: {n1/1e6:.2f}M params  |  TextScoreNet: {n2/1e6:.2f}M params")

    # ── Load pretrained / resumed weights ────────────────────────────────────
    if cfg.train.pretrain_checkpoint and is_main:
        ckpt_path = Path(cfg.train.pretrain_checkpoint)
        print(f"Loading pretrained ECGUNet weights from {ckpt_path}…")
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        current_sd = s_theta.state_dict()
        filtered = {k: v for k, v in state["s_theta"].items()
                    if k in current_sd and current_sd[k].shape == v.shape}
        skipped = [k for k in state["s_theta"] if k not in filtered]
        s_theta.load_state_dict(filtered, strict=False)
        print(f"  loaded {len(filtered)} keys, skipped {len(skipped)}.")

    if cfg.train.resume_checkpoint and is_main:
        ckpt_path = Path(cfg.train.resume_checkpoint)
        print(f"Resuming from {ckpt_path} (EMA weights)…")
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        s_theta.load_state_dict(state.get("s_theta_ema", state["s_theta"]))
        s_phi.load_state_dict(state.get("s_phi_ema",   state["s_phi"]))

    # ── DDP wrap ─────────────────────────────────────────────────────────────
    if ddp:
        s_theta = DDP(s_theta, device_ids=[rank])
        s_phi   = DDP(s_phi,   device_ids=[rank])
        if is_main:
            print(f"  DDP active across {world_size} GPUs.")

    # ── Optimiser / scheduler ────────────────────────────────────────────────
    use_amp  = device.type == "cuda"
    autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                if use_amp else torch.autocast("cpu"))

    params    = list(s_theta.parameters()) + list(s_phi.parameters())
    optimiser = AdamW(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = CosineAnnealingLR(optimiser, T_max=cfg.train.max_steps,
                                   eta_min=cfg.train.lr * 0.05)

    ckpt_dir = Path(cfg.train.checkpoint_dir)
    if is_main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    ema_theta = _init_ema(s_theta) if is_main else None
    ema_phi   = _init_ema(s_phi)   if is_main else None

    # ── Training loop ────────────────────────────────────────────────────────
    step, loss_ema = 0, None
    s_theta.train()
    s_phi.train()
    loader_iter = iter(train_loader)
    t0 = time.time()

    if is_main:
        print(f"Training on {device} (world_size={world_size}), "
              f"max_steps={cfg.train.max_steps}, bf16={use_amp}")

    grad_accum = max(1, cfg.train.grad_accum)

    while step < cfg.train.max_steps:
        if ddp and train_sampler is not None:
            train_sampler.set_epoch(step // max(1, len(train_loader)))

        optimiser.zero_grad()
        accum_loss = 0.0
        for _ in range(grad_accum):
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(train_loader)
                batch = next(loader_iter)

            ecg0  = batch["ecg"].to(device)
            text0 = batch["text_emb"].to(device)
            lead_idx = batch["lead_idx"].to(device) if "lead_idx" in batch else None

            with autocast:
                loss = dsm_loss(
                    s_theta, s_phi, ecg0, text0, vpsde,
                    cfg.train.likelihood_weighting,
                    lead_idx,
                    cfg_drop_prob=cfg.train.cfg_drop_prob,
                )
            loss = (loss / grad_accum).clamp(max=50.0).nan_to_num(0.0)
            loss.backward()
            accum_loss += loss.item()

        nn.utils.clip_grad_norm_(params, cfg.train.grad_clip)
        optimiser.step()
        scheduler.step()
        step += 1

        if is_main:
            _update_ema(ema_theta, s_theta)
            _update_ema(ema_phi,   s_phi)
            raw = accum_loss
            loss_ema = raw if loss_ema is None else 0.98 * loss_ema + 0.02 * raw

            if step % cfg.train.log_every == 0:
                elapsed = time.time() - t0
                print(f"step {step:6d} | loss {raw:.4f} | ema {loss_ema:.4f} "
                      f"| {elapsed / 60:.1f} min")

            if step % cfg.train.val_every == 0:
                val_loss = _validate(s_theta, s_phi, val_loader, vpsde, device, cfg)
                print(f"step {step:6d} | val {val_loss:.4f}")

            if step % cfg.train.save_every == 0:
                _save(s_theta, s_phi, optimiser, step, ckpt_dir,
                      ema_theta=ema_theta, ema_phi=ema_phi)
                if on_checkpoint:
                    on_checkpoint(step)

    if is_main:
        _save(s_theta, s_phi, optimiser, step, ckpt_dir, name="final",
              ema_theta=ema_theta, ema_phi=ema_phi)
        if on_checkpoint:
            on_checkpoint(step)
        print("Training complete.")

    if ddp:
        dist.destroy_process_group()


def _validate(s_theta, s_phi, loader, vpsde, device, cfg) -> float:
    """Compute validation loss without gradients."""
    s_theta.eval()
    s_phi.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            ecg0     = batch["ecg"].to(device)
            text0    = batch["text_emb"].to(device)
            lead_idx = batch["lead_idx"].to(device) if "lead_idx" in batch else None
            loss = dsm_loss(s_theta, s_phi, ecg0, text0, vpsde,
                            cfg.train.likelihood_weighting, lead_idx)
            total += loss.item() * ecg0.shape[0]
            count += ecg0.shape[0]
    s_theta.train()
    s_phi.train()
    return total / max(count, 1)


def _save(s_theta, s_phi, optimiser, step, ckpt_dir, name=None,
          ema_theta=None, ema_phi=None) -> None:
    """Save a training checkpoint (base + EMA weights)."""
    path = ckpt_dir / f"{name or f'step_{step:07d}'}.pt"
    payload = {
        "step": step,
        "s_theta": _unwrap(s_theta).state_dict(),
        "s_phi":   _unwrap(s_phi).state_dict(),
        "optimiser": optimiser.state_dict(),
    }
    if ema_theta is not None:
        payload["s_theta_ema"] = ema_theta
    if ema_phi is not None:
        payload["s_phi_ema"] = ema_phi
    torch.save(payload, path)
    print(f"  checkpoint saved → {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Joint ECG–text diffusion training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config",               default="",    help="Path to YAML config file.")
    parser.add_argument("--data-dir",             default=os.environ.get("DATA_DIR", "data/ptbxl"))
    parser.add_argument("--cache-dir",            default=os.environ.get("CACHE_DIR", "cache"))
    parser.add_argument("--checkpoint-dir",       default=os.environ.get("CHECKPOINT_DIR", "checkpoints"))
    parser.add_argument("--max-steps",            type=int,   default=None)
    parser.add_argument("--batch-size",           type=int,   default=None)
    parser.add_argument("--lr",                   type=float, default=None)
    parser.add_argument("--device",               default=None)
    parser.add_argument("--pretrain-checkpoint",  default=None, help="Path to pretrain .pt checkpoint.")
    parser.add_argument("--resume-checkpoint",    default=None, help="Path to resume .pt checkpoint.")
    parser.add_argument("overrides", nargs="*", metavar="KEY=VALUE",
                        help="Dot-notation config overrides, e.g. train.lr=1e-4")
    return parser.parse_args()


if __name__ == "__main__":
    load_dotenv()

    args = _parse_args()

    cfg = config_module.Config.from_yaml(args.config) if args.config else config_module.Config()

    cfg.train.data_dir         = args.data_dir
    cfg.train.data_cache_dir   = args.cache_dir
    cfg.train.checkpoint_dir   = args.checkpoint_dir
    if args.max_steps            is not None: cfg.train.max_steps           = args.max_steps
    if args.batch_size           is not None: cfg.train.batch_size          = args.batch_size
    if args.lr                   is not None: cfg.train.lr                  = args.lr
    if args.device               is not None: cfg.train.device              = args.device
    if args.pretrain_checkpoint  is not None: cfg.train.pretrain_checkpoint = args.pretrain_checkpoint
    if args.resume_checkpoint    is not None: cfg.train.resume_checkpoint   = args.resume_checkpoint

    if args.overrides:
        cfg.override({k: v for k, v in (o.split("=", 1) for o in args.overrides)})

    train(cfg)
