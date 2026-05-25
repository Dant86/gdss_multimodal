"""Evaluation metrics and sampler ablation table.

Runs a grid of (sampler × NFE) combinations and reports:
  - ECG-FID via MOMENT-1-large embeddings
  - Text cosine similarity (max nearest-neighbour)
  - Joint quality (ECG label ↔ nearest text label match rate)

Environment variables (loaded from .env):
    DATA_DIR         Processed PTB-XL directory.
    CACHE_DIR        BERT embeddings and ECG stats.
    CHECKPOINT_DIR   Checkpoint directory.
    FIGURES_DIR      Where to save the FID bar chart.

Usage
-----
    python apps/eval/main.py --checkpoint final
                             [--checkpoint-dir checkpoints]
                             [--data-dir data/ptbxl] [--cache-dir cache]
                             [--figures-dir figures]
                             [--n-samples 1000] [--device cuda]
                             [--nfe 100 500 1000]
                             [--samplers pc s4 em]
                             [--corrector-snr 0.16]
                             [--lead-idx 1]
                             [--cfg-scale 1.5]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import momentfm
import numpy
import plotly.graph_objects as go
import scipy.linalg
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv
from torch.utils.data import DataLoader

import gdss_multimodal.config as config_module
import gdss_multimodal.data as data_module
import gdss_multimodal.sample as sample_module
import gdss_multimodal.sde as sde_module


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _make_classifier(n_classes: int, n_leads: int = 1):
    """Build a lightweight 1D ResNet classifier for ECG label prediction."""

    class _Res(nn.Module):
        def __init__(self, ch):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(ch, ch, 3, padding=1, bias=False),
                nn.BatchNorm1d(ch), nn.ReLU(),
                nn.Conv1d(ch, ch, 3, padding=1, bias=False),
                nn.BatchNorm1d(ch),
            )
        def forward(self, x):
            return F.relu(x + self.net(x))

    class Clf(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv1d(n_leads, 64, 7, stride=4, padding=3, bias=False),
                nn.BatchNorm1d(64), nn.ReLU(),
            )
            self.blocks = nn.Sequential(
                _Res(64), _Res(64),
                nn.Conv1d(64, 128, 3, stride=4, padding=1), _Res(128), _Res(128),
                nn.Conv1d(128, 256, 3, stride=4, padding=1), _Res(256), _Res(256),
            )
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(256, n_classes),
            )
        def forward(self, x):
            return self.head(self.blocks(self.stem(x)))

    return Clf()


def train_classifier(train_ds, n_classes: int, device, epochs: int = 20,
                     batch_size: int = 64):
    """Train a 1D ResNet classifier on training ECGs for joint-quality scoring.

    Args:
        train_ds: PTBXLDataset training split.
        n_classes: Number of SCP code classes.
        device: Torch device.
        epochs: Training epochs.
        batch_size: Training batch size.

    Returns:
        Trained classifier in eval mode.
    """
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    model = _make_classifier(n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ce  = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        for batch in loader:
            ecg, label = batch["ecg"].to(device), batch["label"].to(device)
            if label.min() < 0:
                continue
            loss = ce(model(ecg), label)
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"  classifier epoch {epoch + 1}/{epochs}")
    model.eval()
    return model


def extract_moment_features(ecgs, device, batch_size: int = 32):
    """Extract MOMENT-1-large embeddings for FID computation.

    Args:
        ecgs: ECG array of shape (N, n_leads, seq_len).
        device: Torch device.
        batch_size: Inference batch size.

    Returns:
        Feature array of shape (N, moment_hidden).
    """
    moment = momentfm.MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large", model_kwargs={"task_name": "embedding"}
    ).to(device).eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(ecgs), batch_size):
            x = torch.from_numpy(ecgs[i: i + batch_size]).to(device)
            inner = moment.model if hasattr(moment, "model") else moment
            mask  = torch.ones(x.shape[0], x.shape[-1], device=device, dtype=torch.long)
            out   = inner.embed(x_enc=x, input_mask=mask, reduction="mean")
            feats.append(out.embeddings.cpu().numpy())
    return numpy.concatenate(feats)


def ecg_fid(real_ecgs, gen_ecgs, device) -> float:
    """Fréchet Inception Distance using MOMENT-1-large features.

    Args:
        real_ecgs: Real ECG array of shape (N, n_leads, seq_len).
        gen_ecgs:  Generated ECG array of the same shape.
        device:    Torch device.

    Returns:
        FID scalar (lower is better).
    """
    def _fid(m1, s1, m2, s2):
        diff = m1 - m2
        eps  = numpy.eye(s1.shape[0]) * 1e-6
        cov, _ = scipy.linalg.sqrtm((s1 + eps) @ (s2 + eps), disp=False)
        if numpy.iscomplexobj(cov):
            cov = cov.real
        return float(diff @ diff + numpy.trace(s1 + s2 - 2 * cov))

    rf = extract_moment_features(real_ecgs, device).reshape(len(real_ecgs), -1)
    gf = extract_moment_features(gen_ecgs,  device).reshape(len(gen_ecgs),  -1)
    return _fid(rf.mean(0), numpy.cov(rf, rowvar=False),
                gf.mean(0), numpy.cov(gf, rowvar=False))


def text_cosine_sim(gen_texts, real_texts) -> float:
    """Mean max cosine similarity: generated → nearest real text embedding.

    Args:
        gen_texts:  Generated text embeddings of shape (N, D).
        real_texts: Real text embeddings of shape (M, D).

    Returns:
        Scalar in [−1, 1] (higher is better).
    """
    gn = gen_texts  / (numpy.linalg.norm(gen_texts,  axis=1, keepdims=True) + 1e-8)
    rn = real_texts / (numpy.linalg.norm(real_texts, axis=1, keepdims=True) + 1e-8)
    return float((gn @ rn.T).max(axis=1).mean())


def joint_quality(gen_ecgs, gen_texts, real_texts, real_labels,
                  classifier, device, batch_size: int = 64) -> float:
    """Fraction of generated samples whose ECG label matches their nearest-text label.

    Args:
        gen_ecgs:    Generated ECG array.
        gen_texts:   Generated text embeddings.
        real_texts:  Real text embeddings.
        real_labels: Integer labels for real texts.
        classifier:  Trained ECG classifier.
        device:      Torch device.
        batch_size:  Inference batch size.

    Returns:
        Scalar in [0, 1] (higher is better).
    """
    preds = []
    with torch.no_grad():
        for i in range(0, len(gen_ecgs), batch_size):
            x = torch.from_numpy(gen_ecgs[i: i + batch_size]).to(device)
            preds.append(classifier(x).argmax(dim=-1).cpu().numpy())
    preds = numpy.concatenate(preds)

    gn = gen_texts  / (numpy.linalg.norm(gen_texts,  axis=1, keepdims=True) + 1e-8)
    rn = real_texts / (numpy.linalg.norm(real_texts, axis=1, keepdims=True) + 1e-8)
    nn_labels = real_labels[(gn @ rn.T).argmax(axis=1)]
    return float((preds == nn_labels).mean())


# ---------------------------------------------------------------------------
# Orchestration helpers
# ---------------------------------------------------------------------------

def _build_real_data(data_dir: str, cache_dir: str, n_samples: int):
    """Load the training split and extract real ECG/text arrays for eval."""
    train_ds, _, _ = data_module.build_datasets(data_dir, cache_dir)
    idx = range(min(n_samples, len(train_ds)))
    real_ecgs   = numpy.stack([train_ds[i]["ecg"].numpy()    for i in idx])
    real_texts  = numpy.stack([train_ds[i]["text_emb"].numpy() for i in range(len(train_ds))])
    real_labels = numpy.array([train_ds[i]["label"]           for i in range(len(train_ds))])
    return train_ds, real_ecgs, real_texts, real_labels


def _run_cell(sampler_name, nfe, s_theta, s_phi, vpsde, real_ecgs, real_texts,
              real_labels, classifier, device, cfg, n_samples, lead_idx, cfg_scale):
    """Run one (sampler, NFE) cell and return metric dict."""
    print(f"\n{sampler_name.upper()} NFE={nfe}")
    gen_ecgs, gen_texts = sample_module.generate(
        s_theta, s_phi, vpsde, sampler_name,
        n_samples, cfg.eval.batch_size, nfe,
        cfg.eval.sampler.corrector_snr, device, cfg,
        lead_idx=lead_idx, cfg_scale=cfg_scale,
    )
    fid = ecg_fid(real_ecgs, gen_ecgs, device)
    cos = text_cosine_sim(gen_texts, real_texts)
    jq  = joint_quality(gen_ecgs, gen_texts, real_texts, real_labels, classifier, device)
    print(f"  FID={fid:.2f}  cos_sim={cos:.4f}  joint_quality={jq:.4f}")
    return {"fid": fid, "cos_sim": cos, "joint_quality": jq}


def print_ablation_table(results: dict) -> None:
    """Print sampler × NFE ablation results as a formatted table."""
    hdr = f"{'Sampler':>8} {'NFE':>6} {'FID':>10} {'CosSim':>10} {'JointQ':>10}"
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for (s, n), m in sorted(results.items()):
        print(f"{s:>8} {n:>6} {m['fid']:>10.2f} {m['cos_sim']:>10.4f} {m['joint_quality']:>10.4f}")


def plot_fid_bar(results: dict, out_path: str) -> None:
    """Save a grouped bar chart of FID scores to out_path."""
    _BG, _DARK, _GRID, _GRAY3 = "#FFFFFF", "#1A1A1A", "#EEEEEE", "#AAAAAA"
    _FONT   = "Fira Sans, Helvetica Neue, Arial, sans-serif"
    _COLORS = ["#C0392B", "#7F8C8D", "#CC8800", "#005555", "#2C3E50", "#8E44AD"]

    samplers = sorted({s for s, _ in results})
    nfes     = sorted({n for _, n in results})

    fig = go.Figure()
    for i, sampler in enumerate(samplers):
        fids = [results.get((sampler, n), {}).get("fid", None) for n in nfes]
        fig.add_trace(go.Bar(
            name=sampler.upper(), x=nfes, y=fids,
            marker_color=_COLORS[i % len(_COLORS)],
            marker_line_color=_DARK, marker_line_width=0.8,
        ))

    fig.update_layout(
        paper_bgcolor=_BG, plot_bgcolor=_BG,
        font=dict(family=_FONT, color=_DARK, size=13),
        margin=dict(l=70, r=30, t=20, b=65),
        barmode="group", width=700, height=450,
        xaxis=dict(
            title_text="NFE", type="category",
            categoryorder="array", categoryarray=[str(n) for n in nfes],
            tickvals=nfes, ticktext=[str(n) for n in nfes],
            showgrid=False, linecolor=_GRAY3, linewidth=1, showline=True,
            ticks="outside", tickcolor=_GRAY3,
        ),
        yaxis=dict(
            title_text="FID ↓",
            showgrid=True, gridcolor=_GRID, gridwidth=1,
            zeroline=False, linecolor=_GRAY3, linewidth=1, showline=True,
            ticks="outside", tickcolor=_GRAY3,
        ),
        legend=dict(bgcolor=_BG, bordercolor=_GRID, borderwidth=1, font=dict(size=12)),
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(out_path), scale=2)
    print(f"  FID bar chart saved → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the joint diffusion model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint",      default="final",   help="Checkpoint stem (no .pt).")
    parser.add_argument("--checkpoint-dir",  default=os.environ.get("CHECKPOINT_DIR", "checkpoints"))
    parser.add_argument("--data-dir",        default=os.environ.get("DATA_DIR", "data/ptbxl"))
    parser.add_argument("--cache-dir",       default=os.environ.get("CACHE_DIR", "cache"))
    parser.add_argument("--figures-dir",     default=os.environ.get("FIGURES_DIR", "figures"))
    parser.add_argument("--config",          default="",        help="Path to YAML config.")
    parser.add_argument("--device",          default="cuda")
    parser.add_argument("--n-samples",       type=int,   default=1000)
    parser.add_argument("--nfe",             nargs="+",  type=int,   default=[100, 500, 1000])
    parser.add_argument("--samplers",        nargs="+",  default=["s4", "pc", "em"])
    parser.add_argument("--corrector-snr",   type=float, default=0.16)
    parser.add_argument("--lead-idx",        type=int,   default=1,
                        help="Lead to evaluate (0–11). -1 = random.")
    parser.add_argument("--cfg-scale",       type=float, default=1.5,
                        help="Classifier-free guidance scale.")
    return parser.parse_args()


if __name__ == "__main__":
    load_dotenv()

    args   = _parse_args()
    cfg    = config_module.Config.from_yaml(args.config) if args.config else config_module.Config()
    cfg.train.data_dir            = args.data_dir
    cfg.train.data_cache_dir      = args.cache_dir
    cfg.eval.sampler.corrector_snr = args.corrector_snr

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    vpsde  = sde_module.VPSDE(cfg.sde.beta_min, cfg.sde.beta_max, cfg.sde.T, cfg.sde.eps)

    ckpt_path = Path(args.checkpoint_dir) / f"{args.checkpoint}.pt"
    s_theta, s_phi = sample_module.load_models(ckpt_path, cfg, device)

    train_ds, real_ecgs, real_texts, real_labels = _build_real_data(
        args.data_dir, args.cache_dir, args.n_samples
    )
    n_classes  = int(real_labels.max()) + 1 if real_labels.min() >= 0 else 5
    classifier = train_classifier(train_ds, n_classes, device)

    results = {}
    for s in args.samplers:
        for n in args.nfe:
            results[(s, n)] = _run_cell(
                s, n, s_theta, s_phi, vpsde,
                real_ecgs, real_texts, real_labels,
                classifier, device, cfg, args.n_samples,
                args.lead_idx, args.cfg_scale,
            )

    print_ablation_table(results)
    fig_path = Path(args.figures_dir) / args.checkpoint / "fid_bar.png"
    plot_fid_bar(results, str(fig_path))
