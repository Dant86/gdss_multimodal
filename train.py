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


def _unwrap(model):
    """Return the underlying module, stripping DataParallel or DDP wrappers."""
    import torch.nn as nn
    from torch.nn.parallel import DistributedDataParallel as DDP
    return model.module if isinstance(model, (nn.DataParallel, DDP)) else model


def _try_commit_volume(step: int) -> None:
    """Commit the Modal checkpoint volume from rank-0 worker (no-op locally)."""
    try:
        import modal_common as _mc
        print(f"  committing checkpoint volume at step {step}…")
        _mc.ckpt_vol.commit()
    except Exception:
        pass


def _init_ema(model) -> dict:
    """Initialise an EMA shadow as a clone of the current state dict."""
    import torch
    return {k: v.clone() for k, v in _unwrap(model).state_dict().items()}


def _update_ema(ema: dict, model, decay: float = 0.9999) -> None:
    """Update EMA shadow in-place: shadow = decay·shadow + (1−decay)·param."""
    import torch
    with torch.no_grad():
        for k, v in _unwrap(model).state_dict().items():
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

    m = _unwrap(s_theta)
    with torch.no_grad():
        h = m.encode(ecg_t, m.t_embed(t), lead_idx, r_peak_mask)
    return h.detach()


def train(cfg: cfg_module.Config, on_checkpoint: Optional[Callable[[int], None]] = None) -> None:
    """Run the full training loop, using DDP when multiple GPUs are present.

    Args:
        cfg: Experiment configuration.
        on_checkpoint: Callback invoked after each checkpoint save (single-GPU only;
            in DDP mode rank-0 commits the Modal volume directly via _try_commit_volume).
    """
    import torch
    import os

    use_cuda = cfg.train.device == "cuda" and torch.cuda.is_available()
    world_size = torch.cuda.device_count() if use_cuda else 1

    if world_size > 1:
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "12355")
        import torch.multiprocessing as mp
        mp.spawn(_train_worker, args=(world_size, cfg), nprocs=world_size, join=True)
    else:
        _train_worker(0, 1, cfg, on_checkpoint=on_checkpoint)


def _train_worker(
    rank: int,
    world_size: int,
    cfg: cfg_module.Config,
    on_checkpoint: Optional[Callable[[int], None]] = None,
) -> None:
    """Per-process training worker (runs on a single GPU).

    In single-GPU mode rank=0, world_size=1 and on_checkpoint is forwarded normally.
    In DDP mode each GPU runs this in its own process; rank-0 handles all I/O.
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

    # ── Distributed setup ────────────────────────────────────────────────────
    ddp = world_size > 1
    if ddp:
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP
        from torch.utils.data import DistributedSampler
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0   # only this rank prints, saves, and validates

    if is_main:
        torch.backends.cudnn.benchmark = True  # fast conv kernels for fixed-size inputs

    _seed(cfg.train.seed + rank)  # different seed per worker for data diversity

    # ── Data ─────────────────────────────────────────────────────────────────
    multi_lead = cfg.ecg_score.lead_emb_dim > 0
    if is_main:
        print(f"Loading datasets… (multi_lead={multi_lead})")
    train_ds, val_ds, _ = data_module.build_datasets(
        cfg.train.data_dir, cfg.train.data_cache_dir,
        bert_device=f"cuda:{rank}" if torch.cuda.is_available() else "cpu",
        multi_lead=multi_lead,
    )

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank,
                                       shuffle=True, drop_last=True) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    # Validation only on rank 0
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

    # ── Pretrain / resume ────────────────────────────────────────────────────
    if cfg.train.pretrain_checkpoint:
        ckpt_path = Path(cfg.train.pretrain_checkpoint)
        if is_main:
            print(f"Loading pretrained ECGUNet weights from {ckpt_path}…")
        state = torch.load(ckpt_path, map_location=device)
        current_sd = s_theta.state_dict()
        filtered = {
            k: v for k, v in state["s_theta"].items()
            if k in current_sd and current_sd[k].shape == v.shape
        }
        skipped = [k for k in state["s_theta"] if k not in filtered]
        s_theta.load_state_dict(filtered, strict=False)
        if is_main:
            print(f"  pretrained weights loaded ({len(filtered)} matched, {len(skipped)} skipped).")
            if skipped:
                print(f"  skipped keys: {skipped[:5]}{'…' if len(skipped) > 5 else ''}")

    if cfg.train.resume_checkpoint:
        ckpt_path = Path(cfg.train.resume_checkpoint)
        if is_main:
            print(f"Resuming from {ckpt_path} (EMA weights, fresh optimiser)…")
        state = torch.load(ckpt_path, map_location=device)
        s_theta.load_state_dict(state.get("s_theta_ema", state["s_theta"]))
        s_phi.load_state_dict(state.get("s_phi_ema", state["s_phi"]))
        if is_main:
            print("  resumed.")

    # ── DDP ──────────────────────────────────────────────────────────────────
    if ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        s_theta = DDP(s_theta, device_ids=[rank])
        s_phi   = DDP(s_phi,   device_ids=[rank])
        if is_main:
            print(f"  DDP active across {world_size} GPUs.")

    # ── Optimiser / scheduler ────────────────────────────────────────────────
    use_amp = device.type == "cuda"
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else torch.autocast("cpu")

    params = list(s_theta.parameters()) + list(s_phi.parameters())
    optimiser = AdamW(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = CosineAnnealingLR(
        optimiser, T_max=cfg.train.max_steps, eta_min=cfg.train.lr * 0.05
    )

    ckpt_dir = Path(cfg.train.checkpoint_dir)
    if is_main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    # EMA maintained only on rank 0 (params are identical across ranks via DDP)
    ema_theta = _init_ema(s_theta) if is_main else None
    ema_phi   = _init_ema(s_phi)   if is_main else None

    # ── Training loop ────────────────────────────────────────────────────────
    step = 0
    loss_ema = None
    s_theta.train()
    s_phi.train()
    loader_iter = iter(train_loader)
    if is_main:
        print(f"Training on {device} (world_size={world_size}), "
              f"max_steps={cfg.train.max_steps}, bf16={use_amp}")
    t0 = time.time()

    while step < cfg.train.max_steps:
        if ddp and train_sampler is not None:
            # Set epoch so each DDP worker sees a different shuffle each epoch
            epoch = step // max(1, len(train_loader))
            train_sampler.set_epoch(epoch)

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
        loss = loss.clamp(max=50.0).nan_to_num(0.0)
        loss.backward()
        nn.utils.clip_grad_norm_(params, cfg.train.grad_clip)
        optimiser.step()
        scheduler.step()
        step += 1

        if is_main:
            _update_ema(ema_theta, s_theta)
            _update_ema(ema_phi, s_phi)
            raw = loss.item()
            loss_ema = raw if loss_ema is None else 0.98 * loss_ema + 0.02 * raw

            if step % cfg.train.val_every == 0:
                elapsed = time.time() - t0
                val_loss = _validate(s_theta, s_phi, val_loader, vpsde, device, cfg)
                print(f"step {step:6d} | val {val_loss:.4f} | ema {loss_ema:.4f} | {elapsed / 60:.1f} min")

            if step % cfg.train.save_every == 0:
                _save(s_theta, s_phi, optimiser, step, ckpt_dir, ema_theta=ema_theta, ema_phi=ema_phi)
                if ddp:
                    _try_commit_volume(step)
                elif on_checkpoint:
                    on_checkpoint(step)

    if is_main:
        _save(s_theta, s_phi, optimiser, step, ckpt_dir, name="final",
              ema_theta=ema_theta, ema_phi=ema_phi)
        if ddp:
            _try_commit_volume(step)
        elif on_checkpoint:
            on_checkpoint(step)
        print("Training complete.")

    if ddp:
        dist.destroy_process_group()


def _validate(s_theta, s_phi, loader, vpsde, device, cfg):
    import torch

    s_theta.eval()
    s_phi.eval()
    total, count = 0.0, 0
    _first_batch_diag = True
    with torch.no_grad():
        for batch in loader:
            ecg0 = batch["ecg"].to(device)
            text0 = batch["text_emb"].to(device)
            lead_idx = batch["lead_idx"].to(device) if "lead_idx" in batch else None
            r_peak_mask = batch["r_peak_mask"].to(device) if "r_peak_mask" in batch else None

            # ── First-batch diagnostics (printed once per val call) ──────────
            if _first_batch_diag:
                _first_batch_diag = False
                t_diag = torch.empty(ecg0.shape[0], device=device).uniform_(vpsde.eps, vpsde.T)
                m1, s1 = vpsde.marginal_prob(ecg0, t_diag)
                m2, s2 = vpsde.marginal_prob(text0, t_diag)
                e1 = torch.randn_like(ecg0)
                e2 = torch.randn_like(text0)
                ecg_t_d = m1 + s1 * e1
                text_t_d = m2 + s2 * e2
                sc_ecg = s_theta(ecg_t_d, text_t_d, t_diag, lead_idx, r_peak_mask)
                _st = _unwrap(s_theta)
                h_bot = _st.encode(ecg_t_d, _st.t_embed(t_diag), lead_idx, r_peak_mask)
                sc_txt = s_phi(text_t_d, h_bot, t_diag)
                a_t = vpsde.alpha(t_diag); sig_t = vpsde.sigma(t_diag).clamp(min=1e-8)
                w_t = sig_t.pow(2) * (a_t / sig_t).pow(2).clamp(max=5.0)
                tgt_ecg = -e1 / s1.clamp(min=1e-8)
                tgt_txt = -e2 / s2.clamp(min=1e-8)
                l_ecg = (w_t.view(-1,1,1) * (sc_ecg - tgt_ecg)**2).mean()
                l_txt = (w_t.view(-1,1) * (sc_txt - tgt_txt)**2).mean()
                print(
                    f"  [diag] ecg0 norm={ecg0.norm(dim=-1).mean():.2f} "
                    f"text0 norm={text0.norm(dim=-1).mean():.2f} "
                    f"h_bot mean={h_bot.mean():.2e} std={h_bot.std():.2e} max={h_bot.abs().max():.2e} | "
                    f"score_ecg max={sc_ecg.abs().max():.2e} | score_text max={sc_txt.abs().max():.2e} | "
                    f"t_diag min={t_diag.min():.3e} max={t_diag.max():.3e} | "
                    f"sigma min={sig_t.min():.3e} | w max={w_t.max():.3e} | "
                    f"loss_ecg={l_ecg:.4e} loss_text={l_txt:.4e}"
                )

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
        "s_theta": _unwrap(s_theta).state_dict(),
        "s_phi": _unwrap(s_phi).state_dict(),
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
def train_on_modal(max_steps=100_000, batch_size=512, lr=2e-4, pretrain_checkpoint="", resume_checkpoint=""):
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

    # In DDP mode, rank-0 commits via _try_commit_volume after each save.
    # In single-GPU mode, pass a callback so the volume is committed too.
    def commit_after_save(step):
        print(f"  committing checkpoint volume at step {step}…")
        modal_common.ckpt_vol.commit()

    train(cfg, on_checkpoint=commit_after_save)
    modal_common.ckpt_vol.commit()
    modal_common.cache_vol.commit()


@modal_common.app.local_entrypoint(name="train")
def main(max_steps: int = 100_000, batch_size: int = 512, lr: float = 2e-4, pretrain_checkpoint: str = "", resume_checkpoint: str = ""):
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
