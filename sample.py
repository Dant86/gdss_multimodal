"""Generation script for joint (ECG, text) samples.

Local:  python sample.py --checkpoint final --sampler s4 --n-steps 1000
Modal:  modal run sample.py -- --checkpoint final --sampler s4 --n-steps 1000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import config as cfg_module
import modal_common


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

    s_theta = models_module.ECGScoreNet(
        text_dim=cfg.ecg_score.text_dim,
        moment_hidden=cfg.ecg_score.moment.moment_hidden_dim,
        timestep_dim=cfg.ecg_score.moment.timestep_embed_dim,
        film_hidden=cfg.ecg_score.moment.film_hidden_dim,
        score_head_hidden=cfg.ecg_score.score_head_hidden,
        n_leads=cfg.ecg_score.n_leads,
        seq_len=cfg.ecg_score.seq_len,
    ).to(device)
    s_phi = models_module.TextScoreNet(
        text_dim=cfg.text_score.text_dim,
        moment_hidden=cfg.text_score.moment_hidden_dim,
        timestep_dim=cfg.text_score.timestep_embed_dim,
        hidden_dim=cfg.text_score.hidden_dim,
        n_layers=cfg.text_score.n_layers,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    s_theta.load_state_dict(ckpt["s_theta"])
    s_phi.load_state_dict(ckpt["s_phi"])
    s_theta.eval()
    s_phi.eval()
    return s_theta, s_phi


def generate(s_theta, s_phi, vpsde, sampler_name, n_samples, batch_size, n_steps, snr, device, cfg):
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

    Returns:
        Tuple of (ecg_array, text_array) as numpy arrays.
    """
    import numpy
    import torch

    import solvers as solvers_module

    sampler = solvers_module.SAMPLERS[sampler_name]
    all_ecg, all_text = [], []

    def score_ecg(m1, m2, t):
        return s_theta(m1, m2, t)

    def score_text(m2, m1, t):
        h = s_theta._moment_forward_with_film(m1, s_theta.t_embed(t))
        return s_phi(m2, h, t)

    generated = 0
    with torch.no_grad():
        while generated < n_samples:
            B = min(batch_size, n_samples - generated)
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
):
    """Modal entry point for sample generation.

    Args:
        checkpoint: Checkpoint name (stem of the .pt file).
        sampler: Sampler name — "s4", "pc", or "em".
        n_steps: Number of reverse diffusion steps.
        n_samples: Total samples to generate.
        batch_size: Generation batch size.
        corrector_snr: Langevin corrector SNR.
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
        s_theta, s_phi, vpsde, sampler, n_samples, batch_size, n_steps, corrector_snr, device, cfg
    )

    out = Path(modal_common.REMOTE_SAMPLES)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{sampler}_nfe{n_steps}"
    numpy.save(out / f"ecg_{tag}.npy", ecgs)
    numpy.save(out / f"text_{tag}.npy", texts)
    print(f"Saved to {out}/ecg_{tag}.npy and text_{tag}.npy")
    modal_common.samples_vol.commit()


@modal_common.app.local_entrypoint()
def main(
    checkpoint: str = "final",
    sampler: str = "s4",
    n_steps: int = 1000,
    n_samples: int = 1000,
    batch_size: int = 32,
    corrector_snr: float = 0.16,
):
    sample_on_modal.remote(
        checkpoint=checkpoint,
        sampler=sampler,
        n_steps=n_steps,
        n_samples=n_samples,
        batch_size=batch_size,
        corrector_snr=corrector_snr,
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
    )
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.sampler}_nfe{args.n_steps}"
    numpy.save(out / f"ecg_{tag}.npy", ecgs)
    numpy.save(out / f"text_{tag}.npy", texts)
    print(f"Saved to {out}/ecg_{tag}.npy and text_{tag}.npy")
