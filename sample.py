"""Generation script for joint (ECG, text) samples.

Local:  python sample.py --checkpoint final --sampler s4 --n-steps 1000
Modal:  modal run sample.py -- --checkpoint final --sampler s4 --n-steps 1000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import config as cfg_module
import modal_common


def make_rpeak_mask(
    n_samples: int,
    heart_rate_bpm: float = 72.0,
    seq_len: int = 1000,
    fs: float = 100.0,
    jitter_frac: float = 0.05,
    device="cpu",
) -> "torch.Tensor":
    """Generate synthetic R-peak binary masks for conditional sampling.

    Places R-peaks at evenly spaced intervals corresponding to the requested
    heart rate, with a small random jitter (±jitter_frac × RR interval) and a
    random phase offset so the first beat doesn't always land at sample 0.

    Args:
        n_samples: Batch size.
        heart_rate_bpm: Desired heart rate in beats per minute (default 72).
        seq_len: Sequence length (1000 for 10 s at 100 Hz).
        fs: Sampling frequency in Hz.
        jitter_frac: Jitter as a fraction of the RR interval (default 0.05 = ±5%).
        device: Torch device.

    Returns:
        Float tensor of shape (n_samples, 1, seq_len) with 1.0 at R-peak positions.
    """
    import torch

    rr = fs * 60.0 / heart_rate_bpm          # RR interval in samples
    jitter = max(1, int(rr * jitter_frac))   # jitter in samples

    masks = torch.zeros(n_samples, 1, seq_len, device=device)
    for b in range(n_samples):
        # Random phase offset: start anywhere in the first half-RR window
        pos = float(torch.randint(0, max(1, int(rr / 2)), (1,)).item())
        while pos < seq_len:
            idx = int(round(pos))
            if 0 <= idx < seq_len:
                masks[b, 0, idx] = 1.0
            j = float(torch.randint(-jitter, jitter + 1, (1,)).item())
            pos += max(rr / 2, rr + j)
    return masks


def load_models(ckpt_path: str | Path, cfg: cfg_module.Config, device):
    """Load trained ECGScoreNet and TextScoreNet from a checkpoint.

    Args:
        ckpt_path: Path to the .pt checkpoint file.
        cfg: Experiment configuration.
        device: Torch device.

    Returns:
        Tuple of (s_theta, s_phi) in eval mode.
    """
    import torch

    import models as models_module

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

    ckpt = torch.load(ckpt_path, map_location=device)
    # Prefer EMA weights for sampling — they are smoother and produce better samples
    s_theta.load_state_dict(ckpt.get("s_theta_ema", ckpt["s_theta"]))
    s_phi.load_state_dict(ckpt.get("s_phi_ema", ckpt["s_phi"]))
    s_theta.eval()
    s_phi.eval()
    return s_theta, s_phi


def generate(s_theta, s_phi, vpsde, sampler_name, n_samples, batch_size, n_steps, snr, device, cfg,
             lead_idx: int = 1, cfg_scale: float = 0.0, heart_rate_bpm: float = 72.0):
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
        lead_idx: Which lead to generate (0–11). Default 1 = Lead II.
            Set to -1 to sample randomly across all 12 leads.
        cfg_scale: Classifier-free guidance scale w ≥ 0. Score is computed as
            (1+w)*score(lead=l) − w*score(lead=null). 0 disables CFG.
        heart_rate_bpm: Heart rate used to generate synthetic R-peak masks at
            inference time (default 72 bpm). Ignored if model has r_peak_enc_dim=0.

    Returns:
        Tuple of (ecg_array, text_array) as numpy arrays.
    """
    import numpy
    import torch

    import solvers as solvers_module

    sampler = solvers_module.SAMPLERS[sampler_name]
    all_ecg, all_text = [], []
    _use_lead_cond = cfg.ecg_score.lead_emb_dim > 0
    _use_rpeak_cond = cfg.ecg_score.r_peak_enc_dim > 0

    generated = 0
    with torch.no_grad():
        while generated < n_samples:
            B = min(batch_size, n_samples - generated)

            if _use_lead_cond:
                if lead_idx < 0:
                    lidx = torch.randint(0, 12, (B,), device=device)
                else:
                    lidx = torch.full((B,), lead_idx, dtype=torch.long, device=device)
            else:
                lidx = None

            rpmask = (
                make_rpeak_mask(B, heart_rate_bpm, cfg.ecg_score.seq_len, device=device)
                if _use_rpeak_cond else None
            )

            def score_ecg(m1, m2, t, _lidx=lidx, _rp=rpmask):
                cond = s_theta(m1, m2, t, _lidx, _rp)
                if cfg_scale > 0.0 and _lidx is not None:
                    null_lidx = torch.full_like(_lidx, -1)
                    uncond = s_theta(m1, m2, t, null_lidx, _rp)
                    return (1.0 + cfg_scale) * cond - cfg_scale * uncond
                return cond

            def score_text(m2, m1, t, _lidx=lidx, _rp=rpmask):
                h = s_theta.encode(m1, s_theta.t_embed(t), _lidx, _rp)
                return s_phi(m2, h, t)

            m1_T = torch.randn(B, cfg.ecg_score.n_leads, cfg.ecg_score.seq_len, device=device)
            m2_T = torch.randn(B, cfg.text_score.text_dim, device=device)
            m1, m2 = sampler(m1_T, m2_T, score_ecg, score_text, vpsde, n_steps=n_steps, snr=snr)
            all_ecg.append(m1.cpu().numpy())
            all_text.append(m2.cpu().numpy())
            generated += B
            print(f"  generated {generated}/{n_samples}")

    return numpy.concatenate(all_ecg), numpy.concatenate(all_text)


@modal_common.app.function(
    image=modal_common.image,
    gpu=modal_common.GPU,
    volumes=modal_common.VOLUME_MAP,
    timeout=10_800,
    secrets=modal_common.HF_SECRETS,
)
def sample_on_modal(
    checkpoint="final",
    sampler="s4",
    n_steps=1000,
    n_samples=1000,
    batch_size=32,
    corrector_snr=0.16,
    lead_idx=1,
    cfg_scale=0.0,
    heart_rate_bpm=72.0,
):
    """Modal entry point for sample generation.

    Args:
        checkpoint: Checkpoint name (stem of the .pt file).
        sampler: Sampler name — "s4", "pc", or "em".
        n_steps: Number of reverse diffusion steps.
        n_samples: Total samples to generate.
        batch_size: Generation batch size.
        corrector_snr: Langevin corrector SNR.
        lead_idx: Lead to generate (0–11, default 1 = Lead II; -1 = random).
    """
    import os

    import numpy
    import torch

    import sde as sde_module

    os.environ["HF_HOME"] = modal_common.HF_CACHE_DIR
    device = torch.device("cuda")
    cfg = cfg_module.Config()
    vpsde = sde_module.VPSDE(cfg.sde.beta_min, cfg.sde.beta_max, cfg.sde.T, cfg.sde.eps)

    s_theta, s_phi = load_models(Path(modal_common.REMOTE_CKPTS) / f"{checkpoint}.pt", cfg, device)
    ecgs, texts = generate(
        s_theta, s_phi, vpsde, sampler, n_samples, batch_size, n_steps, corrector_snr, device, cfg,
        lead_idx=lead_idx, cfg_scale=cfg_scale, heart_rate_bpm=heart_rate_bpm,
    )

    out = Path(modal_common.REMOTE_SAMPLES)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{sampler}_nfe{n_steps}"
    numpy.save(out / f"ecg_{tag}.npy", ecgs)
    numpy.save(out / f"text_{tag}.npy", texts)
    print(f"Saved to {out}/ecg_{tag}.npy and text_{tag}.npy")
    modal_common.samples_vol.commit()


@modal_common.app.local_entrypoint(name="sample")
def main(
    checkpoint: str = "final",
    sampler: str = "s4",
    n_steps: int = 1000,
    n_samples: int = 1000,
    batch_size: int = 32,
    corrector_snr: float = 0.16,
    lead_idx: int = 1,
    cfg_scale: float = 0.0,
    heart_rate_bpm: float = 72.0,
):
    sample_on_modal.remote(
        checkpoint=checkpoint,
        sampler=sampler,
        n_steps=n_steps,
        n_samples=n_samples,
        batch_size=batch_size,
        corrector_snr=corrector_snr,
        lead_idx=lead_idx,
        cfg_scale=cfg_scale,
        heart_rate_bpm=heart_rate_bpm,
    )


if __name__ == "__main__":
    import numpy
    import torch

    import sde as sde_module

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="final")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--sampler", default="s4", choices=["s4", "pc", "em"])
    parser.add_argument("--n-steps", type=int, default=1000)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--corrector-snr", type=float, default=0.16)
    parser.add_argument("--lead-idx", type=int, default=1,
                        help="Lead to generate (0-11; -1=random). Default 1=Lead II.")
    parser.add_argument("--output-dir", default="samples")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = cfg_module.Config()
    vpsde = sde_module.VPSDE(cfg.sde.beta_min, cfg.sde.beta_max, cfg.sde.T, cfg.sde.eps)

    s_theta, s_phi = load_models(
        Path(args.checkpoint_dir) / f"{args.checkpoint}.pt", cfg, device
    )
    ecgs, texts = generate(
        s_theta, s_phi, vpsde, args.sampler, args.n_samples,
        args.batch_size, args.n_steps, args.corrector_snr, device, cfg,
        lead_idx=args.lead_idx,
    )
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.sampler}_nfe{args.n_steps}"
    numpy.save(out / f"ecg_{tag}.npy", ecgs)
    numpy.save(out / f"text_{tag}.npy", texts)
    print(f"Saved to {out}/ecg_{tag}.npy and text_{tag}.npy")
