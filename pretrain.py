"""Autoencoder pre-training for ECGUNet.

Trains ECGUNet as a pure reconstructor (zero conditioning) on clean PTB-XL
ECGs before the joint DSM fine-tuning phase.  This gives the U-Net a strong
ECG-morphology prior — QRS complexes, 1/f PSD, baseline wander — before
the denoising task introduces the diffusion objective.

Loss: L = MSE(recon, ecg) + λ_spec · MSE(|FFT(recon)|/L, |FFT(ecg)|/L)

After training, the checkpoint can be passed to train.py via
--pretrain-checkpoint (or cfg.train.pretrain_checkpoint) so that DSM
fine-tuning begins from morphology-aware weights.

Local:  python pretrain.py
Modal:  modal run pretrain.py
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Optional, Callable

import config as cfg_module
import modal_common


# ---------------------------------------------------------------------------
# Spectral loss helper
# ---------------------------------------------------------------------------

def _spectral_loss(pred: "torch.Tensor", target: "torch.Tensor") -> "torch.Tensor":
    """FFT magnitude MSE, normalised by sequence length.

    Encourages the model to match the power spectral density of the target,
    which for ECGs has a characteristic 1/f profile.

    Args:
        pred:   Reconstructed tensor of shape (B, n_leads, L).
        target: Ground-truth tensor of the same shape.

    Returns:
        Scalar MSE over normalised FFT magnitudes.
    """
    import torch
    import torch.nn.functional as F

    L = pred.shape[-1]
    # rfft doesn't support bfloat16 — cast to float32 for the FFT
    p = torch.fft.rfft(pred.float(), dim=-1).abs() / L
    t = torch.fft.rfft(target.float(), dim=-1).abs() / L
    return F.mse_loss(p, t)


# ---------------------------------------------------------------------------
# Core training function
# ---------------------------------------------------------------------------

def pretrain(
    cfg: cfg_module.Config,
    on_checkpoint: Optional[Callable[[int], None]] = None,
) -> str:
    """Run the autoencoder pre-training loop.

    Uses all PTB-XL splits (train + val + test) since we are learning ECG
    morphology, not fitting the joint distribution.

    Args:
        cfg: Experiment configuration (reads cfg.pretrain and cfg.ecg_score).
        on_checkpoint: Optional callback invoked after each checkpoint save.

    Returns:
        Path to the final pretrain checkpoint.
    """
    import time

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import OneCycleLR
    from torch.utils.data import ConcatDataset, DataLoader

    import data as data_module
    import models as models_module

    pcfg = cfg.pretrain
    ecfg = cfg.ecg_score
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    _seed(cfg.train.seed)

    multi_lead = ecfg.lead_emb_dim > 0
    print(f"Loading datasets… (multi_lead={multi_lead})")
    train_ds, val_ds, test_ds = data_module.build_datasets(
        cfg.train.data_dir, cfg.train.data_cache_dir,
        bert_device=cfg.train.device, multi_lead=multi_lead,
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
        r_peak_enc_dim=ecfg.r_peak_enc_dim,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ECGUNet params: {n_params / 1e6:.2f}M")

    if pcfg.resume:
        resume_path = Path(pcfg.resume)
        print(f"  Resuming from {resume_path}…")
        state = torch.load(resume_path, map_location=device)
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

        ecg = batch["ecg"].to(device)          # (B, 1, seq_len)
        lead_idx = batch["lead_idx"].to(device) if "lead_idx" in batch else None
        r_peak_mask = batch["r_peak_mask"].to(device) if "r_peak_mask" in batch else None

        optimiser.zero_grad()
        with autocast:
            recon = model.reconstruct(ecg, lead_idx, r_peak_mask)
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
            ckpt_path = _save_pretrain(model, optimiser, step, ckpt_dir)
            if on_checkpoint:
                on_checkpoint(step)

    final_path = _save_pretrain(model, optimiser, step, ckpt_dir, name="pretrain_final")
    print(f"Pre-training complete → {final_path}")
    return str(final_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_pretrain(model, optimiser, step, ckpt_dir, name=None):
    import torch

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
    import numpy
    import torch

    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Modal
# ---------------------------------------------------------------------------

@modal_common.app.function(
    image=modal_common.image,
    gpu=modal_common.GPU,
    volumes=modal_common.VOLUME_MAP,
    timeout=14_400,   # 4 hours — 30K steps ≈ 20 min on H100
    secrets=modal_common.HF_SECRETS,
)
def pretrain_on_modal(
    max_steps: int = 30_000,
    batch_size: int = 512,
    lr: float = 1e-3,
    spectral_weight: float = 0.1,
    resume: str = "",
):
    """Modal entry point for ECGUNet autoencoder pre-training.

    Args:
        max_steps: Total pre-training steps (30K ≈ 20 min on H100).
        batch_size: Per-GPU batch size.
        lr: Peak learning rate for the OneCycleLR schedule.
        spectral_weight: Weight of the FFT magnitude loss.
        resume: Path to an existing pretrain checkpoint to resume from.
    """
    import os

    os.environ["HF_HOME"] = modal_common.HF_CACHE_DIR

    cfg = cfg_module.Config()
    cfg.train.device = "cuda"
    cfg.train.data_dir = f"{modal_common.REMOTE_CACHE}/ptbxl"
    cfg.train.data_cache_dir = modal_common.REMOTE_CACHE
    cfg.pretrain.max_steps = max_steps
    cfg.pretrain.batch_size = batch_size
    cfg.pretrain.lr = lr
    cfg.pretrain.spectral_weight = spectral_weight
    cfg.pretrain.checkpoint_dir = modal_common.REMOTE_CKPTS
    cfg.pretrain.resume = resume

    def commit_after_save(step):
        print(f"  committing checkpoint volume at step {step}…")
        modal_common.ckpt_vol.commit()

    pretrain(cfg, on_checkpoint=commit_after_save)
    modal_common.ckpt_vol.commit()
    modal_common.cache_vol.commit()


@modal_common.app.local_entrypoint(name="pretrain")
def main(
    max_steps: int = 30_000,
    batch_size: int = 512,
    lr: float = 1e-3,
    spectral_weight: float = 0.1,
    resume: str = "",
):
    pretrain_on_modal.remote(
        max_steps=max_steps,
        batch_size=batch_size,
        lr=lr,
        spectral_weight=spectral_weight,
        resume=resume,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ECGUNet autoencoder pre-training")
    parser.add_argument("--max-steps", type=int, default=30_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--spectral-weight", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-dir", default="data/ptbxl")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--resume", default="")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = cfg_module.Config()
    cfg.train.device = args.device
    cfg.train.data_dir = args.data_dir
    cfg.train.data_cache_dir = args.cache_dir
    cfg.train.seed = args.seed
    cfg.pretrain.max_steps = args.max_steps
    cfg.pretrain.batch_size = args.batch_size
    cfg.pretrain.lr = args.lr
    cfg.pretrain.spectral_weight = args.spectral_weight
    cfg.pretrain.checkpoint_dir = args.checkpoint_dir
    cfg.pretrain.resume = args.resume
    pretrain(cfg)
