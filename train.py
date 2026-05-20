"""Training loop for the joint (ECG, text) diffusion model.

Local:  python train.py [--device cuda] [--max-steps 100000] ...
Modal:  modal run train.py [-- --max-steps 100000]
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Callable, Optional

import config as cfg_module
import modal_common


def _init_ema(model) -> dict:
    """Initialise an EMA shadow as a clone of the current state dict."""
    import torch
    return {k: v.clone() for k, v in model.state_dict().items()}


def _update_ema(ema: dict, model, decay: float = 0.9999) -> None:
    """Update EMA shadow in-place: shadow = decay·shadow + (1−decay)·param."""
    import torch
    with torch.no_grad():
        for k, v in model.state_dict().items():
            ema[k].mul_(decay).add_(v, alpha=1.0 - decay)


def dsm_loss(s_theta, s_phi, ecg0, text0, vpsde, likelihood_weighting=True, lead_idx=None,
             cfg_drop_prob: float = 0.0, r_peak_mask=None, r_peak_drop_prob: float = 0.0):
    """Compute joint denoising score matching loss.

    Args:
        s_theta: ECG score network.
        s_phi: Text score network.
        ecg0: Clean ECG batch of shape (B, n_leads, seq_len).
        text0: Clean text embedding batch of shape (B, text_dim).
        vpsde: VP-SDE instance.
        likelihood_weighting: Whether to weight by σ(t)².
        lead_idx: Optional lead indices of shape (B,) for lead-conditioned scoring.

    Returns:
        Scalar DSM loss.
    """
    import torch

    B, device = ecg0.shape[0], ecg0.device
    t = torch.empty(B, device=device).uniform_(vpsde.eps, vpsde.T)

    eps1, eps2 = torch.randn_like(ecg0), torch.randn_like(text0)
    mean1, std1 = vpsde.marginal_prob(ecg0, t)
    mean2, std2 = vpsde.marginal_prob(text0, t)
    ecg_t = mean1 + std1 * eps1
    text_t = mean2 + std2 * eps2

    # Classifier-free guidance: randomly null the lead conditioning so the
    # model also learns the lead-agnostic (unconditional) score.
    import torch as _torch
    cfg_lead = lead_idx
    if lead_idx is not None and cfg_drop_prob > 0.0:
        drop_mask = _torch.rand(B, device=device) < cfg_drop_prob
        cfg_lead = lead_idx.clone()
        cfg_lead[drop_mask] = -1   # sentinel → _cond will zero the lead slot

    # R-peak CFG drop: zero out the mask for randomly chosen samples so the
    # model also learns the R-peak-unconditional score.
    r_peak_cond = r_peak_mask
    if r_peak_mask is not None and r_peak_drop_prob > 0.0:
        rp_drop = _torch.rand(B, device=device) < r_peak_drop_prob
        if rp_drop.any():
            r_peak_cond = r_peak_mask.clone()
            r_peak_cond[rp_drop] = 0.0

    score_ecg = s_theta(ecg_t, text_t, t, cfg_lead if cfg_lead is not None else lead_idx, r_peak_cond)
    score_text = s_phi(text_t, _ecg_rep(s_theta, ecg_t, text_t, t, cfg_lead if cfg_lead is not None else lead_idx, r_peak_cond), t)

    shape1 = (-1,) + (1,) * (ecg0.dim() - 1)
    shape2 = (-1,) + (1,) * (text0.dim() - 1)
    target_ecg = -eps1 / std1.view(shape1).clamp(min=1e-8)
    target_text = -eps2 / std2.view(shape2).clamp(min=1e-8)

    loss_ecg = (score_ecg - target_ecg) ** 2
    loss_text = (score_text - target_text) ** 2
    if likelihood_weighting:
        # Min-SNR weighting (Hang et al. 2023, γ=5).
        # Correct form: σ² · min(SNR, γ) = min(α², γσ²).
        # This retains σ² damping so the weight → 0 at both t→0 and t→T,
        # but peaks at the SNR=γ transition (~t=0.13 with our schedule)
        # where ECG structure is still partially visible in the noisy signal.
        # Without the σ² factor, weight=γ at low-t where target=-ε/σ blows up.
        _MINSNR_GAMMA = 5.0
        alpha_t = vpsde.alpha(t)                            # (B,)
        sigma_t = vpsde.sigma(t).clamp(min=1e-8)            # (B,)
        snr = (alpha_t / sigma_t).pow(2)                    # (B,)
        w = sigma_t.pow(2) * snr.clamp(max=_MINSNR_GAMMA)  # σ²·min(SNR,γ)
        loss_ecg  = w.view(shape1) * loss_ecg
        loss_text = w.view(shape2) * loss_text
    return loss_ecg.mean() + loss_text.mean()


def _ecg_rep(s_theta, ecg_t, text_t, t, lead_idx=None, r_peak_mask=None):
    import torch

    with torch.no_grad():
        h = s_theta.encode(ecg_t, s_theta.t_embed(t), lead_idx, r_peak_mask)
    return h.detach()


def train(cfg: cfg_module.Config, on_checkpoint: Optional[Callable[[int], None]] = None) -> None:
    """Run the full training loop.

    Args:
        cfg: Experiment configuration.
        on_checkpoint: Optional callback invoked after each checkpoint save.
    """
    import time

    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.utils.data import DataLoader

    import data as data_module
    import models as models_module
    import sde as sde_module

    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    _seed(cfg.train.seed)

    multi_lead = cfg.ecg_score.lead_emb_dim > 0
    print(f"Loading datasets… (multi_lead={multi_lead})")
    train_ds, val_ds, _ = data_module.build_datasets(
        cfg.train.data_dir, cfg.train.data_cache_dir,
        bert_device=cfg.train.device, multi_lead=multi_lead,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
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
    )

    print("Building models…")
    s_theta = models_module.ECGUNet(
        text_dim=cfg.ecg_score.text_dim,
        n_leads=cfg.ecg_score.n_leads,
        seq_len=cfg.ecg_score.seq_len,
        timestep_dim=cfg.ecg_score.timestep_dim,
        channels=cfg.ecg_score.channels,
        bottleneck_ch=cfg.ecg_score.bottleneck_ch,
        lead_emb_dim=cfg.ecg_score.lead_emb_dim,
        r_peak_enc_dim=cfg.ecg_score.r_peak_enc_dim,
    ).to(device)
    s_phi = models_module.TextScoreNet(
        text_dim=cfg.text_score.text_dim,
        moment_hidden=cfg.text_score.moment_hidden_dim,
        timestep_dim=cfg.text_score.timestep_embed_dim,
        hidden_dim=cfg.text_score.hidden_dim,
        n_layers=cfg.text_score.n_layers,
    ).to(device)
    vpsde = sde_module.VPSDE(cfg.sde.beta_min, cfg.sde.beta_max, cfg.sde.T, cfg.sde.eps)

    if cfg.train.pretrain_checkpoint:
        ckpt_path = Path(cfg.train.pretrain_checkpoint)
        print(f"Loading pretrained ECGUNet weights from {ckpt_path}…")
        state = torch.load(ckpt_path, map_location=device)
        missing, unexpected = s_theta.load_state_dict(state["s_theta"], strict=False)
        if missing:
            print(f"  missing keys: {missing}")
        if unexpected:
            print(f"  unexpected keys: {unexpected}")
        print("  pretrained weights loaded.")

    if cfg.train.resume_checkpoint:
        ckpt_path = Path(cfg.train.resume_checkpoint)
        print(f"Resuming from training checkpoint {ckpt_path} (EMA weights, fresh optimiser)…")
        state = torch.load(ckpt_path, map_location=device)
        # Prefer EMA weights — they are smoother than the final training weights
        s_theta.load_state_dict(state.get("s_theta_ema", state["s_theta"]))
        s_phi.load_state_dict(state.get("s_phi_ema", state["s_phi"]))
        print("  resumed.")

    use_amp = device.type == "cuda"
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else torch.autocast("cpu")

    params = list(s_theta.parameters()) + list(s_phi.parameters())
    optimiser = AdamW(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = CosineAnnealingLR(
        optimiser, T_max=cfg.train.max_steps, eta_min=cfg.train.lr * 0.05
    )

    ckpt_dir = Path(cfg.train.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ema_theta = _init_ema(s_theta)
    ema_phi   = _init_ema(s_phi)

    step = 0
    loss_ema = None
    s_theta.train()
    s_phi.train()
    loader_iter = iter(train_loader)
    print(f"Training on {device}, max_steps={cfg.train.max_steps}, bf16={use_amp}")
    t0 = time.time()

    while step < cfg.train.max_steps:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch = next(loader_iter)

        ecg0 = batch["ecg"].to(device)
        text0 = batch["text_emb"].to(device)
        lead_idx = batch["lead_idx"].to(device) if "lead_idx" in batch else None
        r_peak_mask = batch["r_peak_mask"].to(device) if "r_peak_mask" in batch else None
        optimiser.zero_grad()
        with autocast:
            loss = dsm_loss(s_theta, s_phi, ecg0, text0, vpsde, cfg.train.likelihood_weighting, lead_idx,
                            cfg_drop_prob=cfg.train.cfg_drop_prob,
                            r_peak_mask=r_peak_mask, r_peak_drop_prob=cfg.train.r_peak_drop_prob)
        # Guard against rare catastrophic batches (bfloat16 small-t instability).
        # clamp before backward so the gradient signal is also bounded.
        loss = loss.clamp(max=50.0).nan_to_num(0.0)
        loss.backward()
        nn.utils.clip_grad_norm_(params, cfg.train.grad_clip)
        optimiser.step()
        scheduler.step()
        _update_ema(ema_theta, s_theta)
        _update_ema(ema_phi, s_phi)
        step += 1

        raw = loss.item()
        loss_ema = raw if loss_ema is None else 0.98 * loss_ema + 0.02 * raw

        if step % cfg.train.val_every == 0:
            elapsed = time.time() - t0
            val_loss = _validate(s_theta, s_phi, val_loader, vpsde, device, cfg)
            print(f"step {step:6d} | val {val_loss:.4f} | ema {loss_ema:.4f} | {elapsed / 60:.1f} min")
        if step % cfg.train.save_every == 0:
            _save(s_theta, s_phi, optimiser, step, ckpt_dir, ema_theta=ema_theta, ema_phi=ema_phi)
            if on_checkpoint:
                on_checkpoint(step)

    _save(s_theta, s_phi, optimiser, step, ckpt_dir, name="final", ema_theta=ema_theta, ema_phi=ema_phi)
    if on_checkpoint:
        on_checkpoint(step)
    print("Training complete.")


def _validate(s_theta, s_phi, loader, vpsde, device, cfg):
    import torch

    s_theta.eval()
    s_phi.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            ecg0 = batch["ecg"].to(device)
            text0 = batch["text_emb"].to(device)
            lead_idx = batch["lead_idx"].to(device) if "lead_idx" in batch else None
            r_peak_mask = batch["r_peak_mask"].to(device) if "r_peak_mask" in batch else None
            loss = dsm_loss(s_theta, s_phi, ecg0, text0, vpsde, cfg.train.likelihood_weighting,
                            lead_idx, r_peak_mask=r_peak_mask)
            total += loss.item() * ecg0.shape[0]
            count += ecg0.shape[0]
    s_theta.train()
    s_phi.train()
    return total / count


def _save(s_theta, s_phi, optimiser, step, ckpt_dir, name=None, ema_theta=None, ema_phi=None):
    import torch

    path = ckpt_dir / f"{name or f'step_{step:07d}'}.pt"
    payload = {
        "step": step,
        "s_theta": s_theta.state_dict(),
        "s_phi": s_phi.state_dict(),
        "optimiser": optimiser.state_dict(),
    }
    if ema_theta is not None:
        payload["s_theta_ema"] = ema_theta
    if ema_phi is not None:
        payload["s_phi_ema"] = ema_phi
    torch.save(payload, path)
    print(f"  checkpoint saved → {path}")


def _seed(seed: int) -> None:
    import numpy
    import torch

    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@modal_common.app.function(
    image=modal_common.image,
    gpu=modal_common.GPU,
    volumes=modal_common.VOLUME_MAP,
    timeout=72_000,
    secrets=modal_common.HF_SECRETS,
)
def train_on_modal(max_steps=100_000, batch_size=256, lr=2e-4, pretrain_checkpoint="", resume_checkpoint=""):
    """Modal entry point for distributed training.

    Args:
        max_steps: Total training steps.
        batch_size: Per-GPU batch size.
        lr: Initial learning rate.
        pretrain_checkpoint: Optional path to a pretrained ECGUNet checkpoint.
    """
    import os

    os.environ["HF_HOME"] = modal_common.HF_CACHE_DIR

    cfg = cfg_module.Config()
    cfg.train.device = "cuda"
    cfg.train.batch_size = batch_size
    cfg.train.max_steps = max_steps
    cfg.train.lr = lr
    cfg.train.data_dir = f"{modal_common.REMOTE_CACHE}/ptbxl"
    cfg.train.data_cache_dir = modal_common.REMOTE_CACHE
    cfg.train.checkpoint_dir = modal_common.REMOTE_CKPTS
    cfg.train.pretrain_checkpoint = pretrain_checkpoint
    cfg.train.resume_checkpoint = resume_checkpoint

    def commit_after_save(step):
        print(f"  committing checkpoint volume at step {step}…")
        modal_common.ckpt_vol.commit()

    train(cfg, on_checkpoint=commit_after_save)
    modal_common.ckpt_vol.commit()
    modal_common.cache_vol.commit()


@modal_common.app.local_entrypoint(name="train")
def main(max_steps: int = 100_000, batch_size: int = 256, lr: float = 2e-4, pretrain_checkpoint: str = "", resume_checkpoint: str = ""):
    train_on_modal.remote(max_steps=max_steps, batch_size=batch_size, lr=lr, pretrain_checkpoint=pretrain_checkpoint, resume_checkpoint=resume_checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--data-dir", default="data/ptbxl")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    args = parser.parse_args()

    cfg = cfg_module.Config()
    cfg.train.device = args.device
    cfg.train.batch_size = args.batch_size
    cfg.train.max_steps = args.max_steps
    cfg.train.lr = args.lr
    cfg.train.data_dir = args.data_dir
    cfg.train.data_cache_dir = args.cache_dir
    cfg.train.checkpoint_dir = args.checkpoint_dir
    train(cfg)
