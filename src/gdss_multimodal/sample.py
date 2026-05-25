"""Generation utilities for joint (ECG, text) reverse diffusion."""

from __future__ import annotations

from pathlib import Path

import numpy
import torch

import gdss_multimodal.config as config_module
import gdss_multimodal.models as models_module
import gdss_multimodal.sde as sde_module
import gdss_multimodal.solvers as solvers_module


def load_models(ckpt_path: str | Path, cfg: "config_module.Config", device) -> tuple:
    """Load trained ECGUNet and TextScoreNet from a checkpoint.

    Args:
        ckpt_path: Path to the .pt checkpoint file.
        cfg: Experiment configuration.
        device: Torch device.

    Returns:
        Tuple of (s_theta, s_phi) in eval mode.
    """
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

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    s_theta.load_state_dict(ckpt.get("s_theta_ema", ckpt["s_theta"]))
    s_phi.load_state_dict(ckpt.get("s_phi_ema", ckpt["s_phi"]))
    s_theta.eval()
    s_phi.eval()
    return s_theta, s_phi


def generate(
    s_theta,
    s_phi,
    vpsde,
    sampler_name: str,
    n_samples: int,
    batch_size: int,
    n_steps: int,
    snr: float,
    device,
    cfg: "config_module.Config",
    lead_idx: int = 1,
    cfg_scale: float = 0.0,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Run reverse diffusion to generate (ECG, text) pairs.

    Args:
        s_theta: Trained ECG score network.
        s_phi: Trained text score network.
        vpsde: VP-SDE instance.
        sampler_name: One of "s4", "pc", "em".
        n_samples: Total number of samples to generate.
        batch_size: Batch size for generation.
        n_steps: Number of reverse diffusion steps (NFE).
        snr: Langevin corrector SNR.
        device: Torch device.
        cfg: Experiment configuration.
        lead_idx: Which lead to generate (0–11). -1 = sample randomly.
        cfg_scale: Classifier-free guidance scale w ≥ 0. 0 disables CFG.

    Returns:
        Tuple of (ecg_array, text_array) as numpy arrays.
    """
    sampler = solvers_module.SAMPLERS[sampler_name]
    _use_lead_cond = cfg.ecg_score.lead_emb_dim > 0
    all_ecg, all_text = [], []

    generated = 0
    with torch.no_grad():
        while generated < n_samples:
            B = min(batch_size, n_samples - generated)

            if _use_lead_cond:
                lidx = (
                    torch.randint(0, 12, (B,), device=device)
                    if lead_idx < 0
                    else torch.full((B,), lead_idx, dtype=torch.long, device=device)
                )
            else:
                lidx = None

            def score_ecg(m1, m2, t, _lidx=lidx):
                cond = s_theta(m1, m2, t, _lidx)
                if cfg_scale > 0.0 and _lidx is not None:
                    null_lidx = torch.full_like(_lidx, -1)
                    uncond = s_theta(m1, m2, t, null_lidx)
                    return (1.0 + cfg_scale) * cond - cfg_scale * uncond
                return cond

            def score_text(m2, m1, t, _lidx=lidx):
                h = s_theta.encode(m1, s_theta.t_embed(t), _lidx)
                return s_phi(m2, h, t)

            m1_T = torch.randn(B, cfg.ecg_score.n_leads, cfg.ecg_score.seq_len, device=device)
            m2_T = torch.randn(B, cfg.text_score.text_dim, device=device)
            m1, m2 = sampler(m1_T, m2_T, score_ecg, score_text, vpsde, n_steps=n_steps, snr=snr)
            all_ecg.append(m1.cpu().numpy())
            all_text.append(m2.cpu().numpy())
            generated += B
            print(f"  generated {generated}/{n_samples}")

    return numpy.concatenate(all_ecg), numpy.concatenate(all_text)
