"""Publication figures for the joint (ECG, text) diffusion model.

Styled for LaTeX Metropolis theme with Beaver colour scheme.
Exports high-resolution PNGs suitable for \\includegraphics.

Outputs
-------
    ecg_waveforms.png   Grid of real vs generated ECGs by diagnosis class.
    psd_comparison.png  Mean power spectral density: real (—) vs generated (- -).
    text_neighbors.png  Table of generated text embedding → nearest real report.

Environment variables (loaded from .env):
    DATA_DIR      Processed PTB-XL directory.
    CACHE_DIR     BERT embeddings and ECG stats.
    FIGURES_DIR   Where to save output PNGs.

Usage
-----
    python apps/visualize/main.py --checkpoint checkpoints/final.pt
                                  [--data-dir data/ptbxl]
                                  [--cache-dir cache]
                                  [--output-dir figures/run1]
                                  [--n-real 400] [--n-gen 300]
                                  [--device cpu]
                                  [--sampler pc] [--n-steps 500]
                                  [--cfg-scale 1.5] [--lead-idx 1]
"""

from __future__ import annotations

import argparse
import ast
import collections
import os
import pickle
import random
from pathlib import Path

import numpy
import pandas
import plotly.graph_objects as go
import plotly.subplots as subplots
import scipy.signal
import torch
from dotenv import load_dotenv

import gdss_multimodal.config as config_module
import gdss_multimodal.data as data_module
import gdss_multimodal.sample as sample_module
import gdss_multimodal.sde as sde_module


# ── Colour theme (Metropolis / Beaver) ──────────────────────────────────────

_RED    = "#800000"
_RUST   = "#A63200"
_DARK   = "#1A1A1A"
_GRAY1  = "#404040"
_GRAY2  = "#707070"
_GRAY3  = "#A8A8A8"
_GRID   = "#EEEEEE"
_BG     = "#FFFFFF"
_FONT   = "Fira Sans, Helvetica Neue, Arial, sans-serif"
_PALETTE = [_RED, _RUST, _GRAY1, _GRAY2, "#CC8800", "#005555"]


def _layout(**kw) -> dict:
    base = dict(
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        font=dict(family=_FONT, color=_DARK, size=13),
        margin=dict(l=70, r=30, t=20, b=65),
    )
    base.update(kw)
    return base


def _axis(**kw) -> dict:
    return dict(
        showgrid=True, gridcolor=_GRID, gridwidth=1,
        zeroline=False,
        linecolor=_GRAY3, linewidth=1, showline=True,
        ticks="outside", tickcolor=_GRAY3,
        **kw,
    )


# ── Data helpers ─────────────────────────────────────────────────────────────

def _scp_name_map(data_dir: str | Path) -> dict[int, str]:
    """Rebuild label-index → SCP code name from ptbxl_database.csv.

    Args:
        data_dir: PTB-XL root directory.

    Returns:
        Dict mapping integer label to SCP code string.
    """
    df = pandas.read_csv(Path(data_dir) / data_module.PTBXL_CSV, index_col="ecg_id")
    parsed = df["scp_codes"].apply(ast.literal_eval)
    all_codes: list[str] = []
    for codes in parsed:
        all_codes.extend(codes.keys())
    return {i: name for i, (name, _) in enumerate(collections.Counter(all_codes).most_common())}


def _load_real_subset(
    data_dir: str | Path,
    cache_dir: str | Path,
    n: int = 400,
    seed: int = 0,
) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, list[str]]:
    """Load a random subset of real ECGs without a full dataset preload.

    Args:
        data_dir:  PTB-XL root directory.
        cache_dir: Directory with bert_embeddings.pkl and ecg_stats pkl.
        n:         Number of records to sample.
        seed:      Random seed.

    Returns:
        Tuple of (ecgs, text_embs, labels, reports).
        ecgs has shape (N, 1, seq_len), text_embs has shape (N, 768).
    """
    records = data_module.load_ptbxl_records(data_dir)
    train_recs, _, _ = data_module.split_by_fold(records)

    cache_path = Path(cache_dir) / "bert_embeddings.pkl"
    with open(cache_path, "rb") as f:
        emb_cache: dict[str, numpy.ndarray] = pickle.load(f)

    stats_path = Path(cache_dir) / f"ecg_stats_{data_module.SEQ_LEN}hz.pkl"
    with open(stats_path, "rb") as f:
        mean, std = pickle.load(f)
    mean_t = torch.from_numpy(mean[:, None])
    std_t  = torch.from_numpy(std[:, None])

    random.seed(seed)
    sample_recs = random.sample(train_recs, min(n, len(train_recs)))

    ecgs, text_embs, labels, reports = [], [], [], []
    for rec in sample_recs:
        try:
            raw = data_module._load_waveform(rec)
            ecg = (torch.from_numpy(raw) - mean_t) / std_t
            ecg = ecg[data_module.LEAD_IDX: data_module.LEAD_IDX + 1].numpy()
            rid = str(rec["ecg_id"])
            ecgs.append(ecg)
            text_embs.append(emb_cache[rid])
            labels.append(int(rec["label"]))
            reports.append(str(rec.get("report", "")))
        except Exception:
            continue

    return (
        numpy.stack(ecgs),
        numpy.stack(text_embs),
        numpy.array(labels),
        reports,
    )


def _assign_labels(
    gen_texts: numpy.ndarray,
    real_texts: numpy.ndarray,
    real_labels: numpy.ndarray,
) -> numpy.ndarray:
    """Assign each generated sample the label of its nearest real text embedding.

    Args:
        gen_texts:   Generated text embeddings of shape (N, D).
        real_texts:  Real text embeddings of shape (M, D).
        real_labels: Integer labels for real texts of shape (M,).

    Returns:
        Integer labels for generated samples of shape (N,).
    """
    gn = gen_texts  / (numpy.linalg.norm(gen_texts,  axis=1, keepdims=True) + 1e-8)
    rn = real_texts / (numpy.linalg.norm(real_texts, axis=1, keepdims=True) + 1e-8)
    nn_idx = (gn @ rn.T).argmax(axis=1)
    return real_labels[nn_idx]


# ── Figures ───────────────────────────────────────────────────────────────────

def plot_waveforms(
    real_ecgs: numpy.ndarray,
    real_labels: numpy.ndarray,
    gen_ecgs: numpy.ndarray,
    gen_labels: numpy.ndarray,
    name_map: dict[int, str],
    out_dir: Path,
    n_classes: int = 4,
    n_per_class: int = 3,
    fs: float = 100.0,
) -> Path:
    """ECG waveform grid: real vs generated, organised by diagnosis class.

    Args:
        real_ecgs:    Real ECGs of shape (N, 1, seq_len).
        real_labels:  Integer labels of shape (N,).
        gen_ecgs:     Generated ECGs of shape (M, 1, seq_len).
        gen_labels:   Integer labels (via text NN) of shape (M,).
        name_map:     Label index → SCP code string.
        out_dir:      Directory to write the PNG.
        n_classes:    Number of diagnosis classes to display.
        n_per_class:  Waveforms to overlay per panel.
        fs:           ECG sampling frequency in Hz.

    Returns:
        Path to the saved PNG.
    """
    unique, counts = numpy.unique(real_labels, return_counts=True)
    top_labels = unique[numpy.argsort(-counts)][:n_classes].tolist()
    t = numpy.linspace(0, real_ecgs.shape[-1] / fs, real_ecgs.shape[-1])

    row_titles = [name_map.get(lbl, str(lbl)) for lbl in top_labels]
    fig = subplots.make_subplots(
        rows=n_classes, cols=2,
        shared_xaxes=True,
        column_titles=["<b>Real</b>", "<b>Generated</b>"],
        row_titles=row_titles,
        horizontal_spacing=0.07,
        vertical_spacing=0.05,
    )

    for row, lbl in enumerate(top_labels, start=1):
        color = _PALETTE[(row - 1) % len(_PALETTE)]
        for i in numpy.where(real_labels == lbl)[0][:n_per_class]:
            fig.add_trace(go.Scatter(
                x=t, y=real_ecgs[i, 0], mode="lines",
                line=dict(color=color, width=1.2), opacity=0.80, showlegend=False,
            ), row=row, col=1)
        for i in numpy.where(gen_labels == lbl)[0][:n_per_class]:
            fig.add_trace(go.Scatter(
                x=t, y=gen_ecgs[i, 0], mode="lines",
                line=dict(color=_GRAY1, width=1.2), opacity=0.80, showlegend=False,
            ), row=row, col=2)

    fig.update_layout(**_layout(height=210 * n_classes, width=1100))
    fig.update_xaxes(**_axis(title_text="Time (s)"))
    fig.update_yaxes(**_axis(title_text="Amplitude (norm.)"))

    path = out_dir / "ecg_waveforms.png"
    fig.write_image(str(path), scale=2)
    print(f"  saved {path}")
    return path


def plot_psd(
    real_ecgs: numpy.ndarray,
    real_labels: numpy.ndarray,
    gen_ecgs: numpy.ndarray,
    gen_labels: numpy.ndarray,
    name_map: dict[int, str],
    out_dir: Path,
    n_classes: int = 4,
    fs: float = 100.0,
) -> Path:
    """Mean power spectral density: real (solid) vs generated (dashed).

    Args:
        real_ecgs:   Real ECGs of shape (N, 1, seq_len).
        real_labels: Integer labels of shape (N,).
        gen_ecgs:    Generated ECGs of shape (M, 1, seq_len).
        gen_labels:  Integer labels of shape (M,).
        name_map:    Label index → SCP code string.
        out_dir:     Output directory.
        n_classes:   Number of diagnosis classes.
        fs:          Sampling frequency in Hz.

    Returns:
        Path to saved PNG.
    """
    unique, counts = numpy.unique(real_labels, return_counts=True)
    top_labels = unique[numpy.argsort(-counts)][:n_classes].tolist()

    fig = go.Figure()
    for j, lbl in enumerate(top_labels):
        color = _PALETTE[j % len(_PALETTE)]
        name  = name_map.get(lbl, str(lbl))

        r_idx = numpy.where(real_labels == lbl)[0]
        if len(r_idx):
            freqs = scipy.signal.welch(real_ecgs[r_idx[0], 0], fs=fs, nperseg=256)[0]
            psds  = [scipy.signal.welch(real_ecgs[i, 0], fs=fs, nperseg=256)[1] for i in r_idx]
            mpsd  = 10 * numpy.log10(numpy.mean(psds, axis=0) + 1e-12)
            fig.add_trace(go.Scatter(
                x=freqs, y=mpsd, name=f"{name} (real)",
                mode="lines", line=dict(color=color, width=2.5, dash="solid"),
            ))

        g_idx = numpy.where(gen_labels == lbl)[0]
        if len(g_idx):
            psds = [scipy.signal.welch(gen_ecgs[i, 0], fs=fs, nperseg=256)[1] for i in g_idx]
            mpsd = 10 * numpy.log10(numpy.mean(psds, axis=0) + 1e-12)
            fig.add_trace(go.Scatter(
                x=freqs, y=mpsd, name=f"{name} (gen)",
                mode="lines", line=dict(color=color, width=2.5, dash="dash"),
            ))

    fig.update_layout(**_layout(
        width=950, height=500,
        xaxis=_axis(title_text="Frequency (Hz)", range=[0, 50]),
        yaxis=_axis(title_text="Power (dB)"),
        legend=dict(bgcolor=_BG, bordercolor=_GRID, borderwidth=1, font=dict(size=11)),
    ))

    path = out_dir / "psd_comparison.png"
    fig.write_image(str(path), scale=2)
    print(f"  saved {path}")
    return path


def plot_text_neighbors(
    gen_texts: numpy.ndarray,
    gen_labels: numpy.ndarray,
    real_texts: numpy.ndarray,
    real_reports: list[str],
    name_map: dict[int, str],
    out_dir: Path,
    n_rows: int = 16,
    max_report_chars: int = 80,
) -> Path:
    """Table: generated text embedding → nearest real clinical report.

    Args:
        gen_texts:        Generated text embeddings of shape (N, D).
        gen_labels:       Assigned diagnosis labels of shape (N,).
        real_texts:       Real text embeddings of shape (M, D).
        real_reports:     Raw report strings for each real sample.
        name_map:         Label → SCP code string.
        out_dir:          Output directory.
        n_rows:           Number of rows to display.
        max_report_chars: Truncation length for long reports.

    Returns:
        Path to saved PNG.
    """
    gn   = gen_texts  / (numpy.linalg.norm(gen_texts,  axis=1, keepdims=True) + 1e-8)
    rn   = real_texts / (numpy.linalg.norm(real_texts, axis=1, keepdims=True) + 1e-8)
    sims = gn @ rn.T  # (N_gen, N_real)

    unique_labels = sorted({int(l) for l in gen_labels})
    chosen: list[int] = []
    for lbl in unique_labels:
        idxs = numpy.where(gen_labels == lbl)[0]
        if len(idxs):
            chosen.append(int(idxs[0]))
        if len(chosen) >= n_rows:
            break
    remainder = n_rows - len(chosen)
    if remainder > 0:
        extra = numpy.linspace(0, len(gen_texts) - 1, remainder + 2, dtype=int)[1:-1]
        chosen += [i for i in extra if i not in chosen][:remainder]

    col_class, col_sim, col_report = [], [], []
    for i in chosen[:n_rows]:
        nn     = int(sims[i].argmax())
        report = real_reports[nn].strip()
        if len(report) > max_report_chars:
            report = report[:max_report_chars] + "…"
        col_class.append(name_map.get(int(gen_labels[i]), str(gen_labels[i])))
        col_sim.append(f"{float(sims[i, nn]):.3f}")
        col_report.append(report)

    fill = [_GRID if k % 2 == 0 else _BG for k in range(len(col_class))]

    fig = go.Figure(data=[go.Table(
        columnwidth=[120, 80, 500],
        header=dict(
            values=["<b>Assigned class</b>", "<b>Cos sim</b>", "<b>Nearest real report</b>"],
            fill_color=_RED,
            font=dict(color="white", family=_FONT, size=13),
            align="left", height=36, line_color=_RED,
        ),
        cells=dict(
            values=[col_class, col_sim, col_report],
            fill_color=[fill, fill, fill],
            font=dict(color=_DARK, family=_FONT, size=11),
            align="left", height=30, line_color=_GRID,
        ),
    )])
    fig.update_layout(**_layout(
        width=900,
        height=90 + 32 * len(col_class),
        margin=dict(l=20, r=20, t=20, b=20),
    ))

    path = out_dir / "text_neighbors.png"
    fig.write_image(str(path), scale=2)
    print(f"  saved {path}")
    return path


# ── Orchestration ─────────────────────────────────────────────────────────────

def visualize(
    ckpt_path: str | Path,
    data_dir: str | Path,
    cache_dir: str | Path,
    out_dir: str | Path,
    n_real: int = 400,
    n_gen: int = 300,
    device_str: str = "cpu",
    sampler: str = "pc",
    n_steps: int = 500,
    cfg_scale: float = 1.5,
    lead_idx: int = 1,
) -> None:
    """Run all three visualisations and write PNGs to out_dir.

    Args:
        ckpt_path:  Path to the model checkpoint (.pt).
        data_dir:   PTB-XL root directory.
        cache_dir:  Directory with BERT embeddings and ECG stats caches.
        out_dir:    Directory to write PNG files.
        n_real:     Number of real ECGs to sample for comparison.
        n_gen:      Number of ECG pairs to generate.
        device_str: Torch device string.
        sampler:    Sampler name — "pc", "s4", or "em".
        n_steps:    Number of reverse diffusion steps.
        cfg_scale:  Classifier-free guidance scale.
        lead_idx:   Lead to generate (0–11; -1 = random).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_str)

    print("Loading real data subset…")
    real_ecgs, real_texts, real_labels, real_reports = _load_real_subset(
        data_dir, cache_dir, n=n_real
    )

    print("Loading models…")
    cfg   = config_module.Config()
    vpsde = sde_module.VPSDE(cfg.sde.beta_min, cfg.sde.beta_max, cfg.sde.T, cfg.sde.eps)
    s_theta, s_phi = sample_module.load_models(ckpt_path, cfg, device)

    print(f"Generating {n_gen} samples with {sampler.upper()} NFE={n_steps}…")
    gen_ecgs, gen_texts = sample_module.generate(
        s_theta, s_phi, vpsde, sampler, n_gen,
        batch_size=min(64, n_gen), n_steps=n_steps,
        snr=0.16, device=device, cfg=cfg,
        cfg_scale=cfg_scale, lead_idx=lead_idx,
    )

    print("Assigning labels via text nearest-neighbour…")
    gen_labels = _assign_labels(gen_texts, real_texts, real_labels)

    name_map = _scp_name_map(data_dir)

    print("Plotting…")
    plot_waveforms(real_ecgs, real_labels, gen_ecgs, gen_labels, name_map, out_dir)
    plot_psd(real_ecgs, real_labels, gen_ecgs, gen_labels, name_map, out_dir)
    plot_text_neighbors(gen_texts, gen_labels, real_texts, real_reports, name_map, out_dir)
    print("Done.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate publication-quality visualisation PNGs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint",  required=True, help="Path to .pt checkpoint file.")
    parser.add_argument("--data-dir",    default=os.environ.get("DATA_DIR", "data/ptbxl"))
    parser.add_argument("--cache-dir",   default=os.environ.get("CACHE_DIR", "cache"))
    parser.add_argument("--output-dir",  default=os.environ.get("FIGURES_DIR", "figures"))
    parser.add_argument("--n-real",      type=int,   default=400)
    parser.add_argument("--n-gen",       type=int,   default=300)
    parser.add_argument("--device",      default="cpu")
    parser.add_argument("--sampler",     default="pc", choices=["pc", "s4", "em"])
    parser.add_argument("--n-steps",     type=int,   default=500)
    parser.add_argument("--cfg-scale",   type=float, default=1.5)
    parser.add_argument("--lead-idx",    type=int,   default=1,
                        help="Lead to generate (0–11; -1 = random).")
    return parser.parse_args()


if __name__ == "__main__":
    load_dotenv()

    args = _parse_args()
    visualize(
        ckpt_path=args.checkpoint,
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        out_dir=args.output_dir,
        n_real=args.n_real,
        n_gen=args.n_gen,
        device_str=args.device,
        sampler=args.sampler,
        n_steps=args.n_steps,
        cfg_scale=args.cfg_scale,
        lead_idx=args.lead_idx,
    )
