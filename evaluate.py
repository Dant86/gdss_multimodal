"""Evaluation metrics and sampler ablation table.

Local:  python evaluate.py --checkpoint final --nfe 100 500 1000
Modal:  modal run evaluate.py -- --checkpoint final --nfe 100,500,1000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import config as cfg_module
import modal_common


def _make_classifier(n_classes: int, n_leads: int = 1):
    import torch.nn as nn
    import torch.nn.functional as F

    class ResBlock1D(nn.Module):
        def __init__(self, ch):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(ch, ch, 3, padding=1, bias=False),
                nn.BatchNorm1d(ch),
                nn.ReLU(),
                nn.Conv1d(ch, ch, 3, padding=1, bias=False),
                nn.BatchNorm1d(ch),
            )

        def forward(self, x):
            return F.relu(x + self.net(x))

    class ECGClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv1d(n_leads, 64, 7, stride=4, padding=3, bias=False),
                nn.BatchNorm1d(64),
                nn.ReLU(),
            )
            self.blocks = nn.Sequential(
                ResBlock1D(64),
                ResBlock1D(64),
                nn.Conv1d(64, 128, 3, stride=4, padding=1),
                ResBlock1D(128),
                ResBlock1D(128),
                nn.Conv1d(128, 256, 3, stride=4, padding=1),
                ResBlock1D(256),
                ResBlock1D(256),
            )
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(256, n_classes),
            )

        def forward(self, x):
            return self.head(self.blocks(self.stem(x)))

    return ECGClassifier()


def train_classifier(train_ds, n_classes: int, device, epochs: int = 20, batch_size: int = 64):
    """Train a 1D ResNet classifier on training ECGs for evaluation.

    Args:
        train_ds: PTBXLDataset training split.
        n_classes: Number of SCP code classes.
        device: Torch device.
        epochs: Training epochs.
        batch_size: Training batch size.

    Returns:
        Trained classifier in eval mode.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    model = _make_classifier(n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        for batch in loader:
            ecg, label = batch["ecg"].to(device), batch["label"].to(device)
            if label.min() < 0:
                continue
            loss = ce(model(ecg), label)
            opt.zero_grad()
            loss.backward()
            opt.step()
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
    import numpy
    import torch
    import momentfm

    moment = momentfm.MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large", model_kwargs={"task_name": "embedding"}
    ).to(device).eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(ecgs), batch_size):
            x = torch.from_numpy(ecgs[i: i + batch_size]).to(device)
            inner = moment.model if hasattr(moment, "model") else moment
            mask = torch.ones(x.shape[0], x.shape[-1], device=device, dtype=torch.long)
            out = inner.embed(x_enc=x, input_mask=mask, reduction="mean")
            feats.append(out.embeddings.cpu().numpy())  # (B, d_model) — already pooled
    return numpy.concatenate(feats)


def ecg_fid(real_ecgs, gen_ecgs, device) -> float:
    """Compute ECG Fréchet Inception Distance using MOMENT features.

    Args:
        real_ecgs: Real ECG array of shape (N, n_leads, seq_len).
        gen_ecgs: Generated ECG array of the same shape.
        device: Torch device.

    Returns:
        FID scalar.
    """
    import numpy
    import scipy.linalg

    def _fid(m1, s1, m2, s2):
        diff = m1 - m2
        eps = numpy.eye(s1.shape[0]) * 1e-6
        cov, _ = scipy.linalg.sqrtm((s1 + eps) @ (s2 + eps), disp=False)
        if numpy.iscomplexobj(cov):
            cov = cov.real
        return float(diff @ diff + numpy.trace(s1 + s2 - 2 * cov))

    rf = extract_moment_features(real_ecgs, device)
    gf = extract_moment_features(gen_ecgs, device)
    rf = rf.reshape(len(rf), -1)
    gf = gf.reshape(len(gf), -1)
    return _fid(rf.mean(0), numpy.cov(rf, rowvar=False), gf.mean(0), numpy.cov(gf, rowvar=False))


def text_cosine_sim(gen_texts, real_texts) -> float:
    """Max nearest-neighbour cosine similarity between generated and real texts.

    Args:
        gen_texts: Generated text embeddings of shape (N, text_dim).
        real_texts: Real text embeddings of shape (M, text_dim).

    Returns:
        Mean max cosine similarity scalar.
    """
    import numpy

    gn = gen_texts / (numpy.linalg.norm(gen_texts, axis=1, keepdims=True) + 1e-8)
    rn = real_texts / (numpy.linalg.norm(real_texts, axis=1, keepdims=True) + 1e-8)
    return float((gn @ rn.T).max(axis=1).mean())


def joint_quality(gen_ecgs, gen_texts, real_texts, real_labels, classifier, device, batch_size: int = 64) -> float:
    """Fraction of generated samples whose ECG label matches their nearest-text label.

    Args:
        gen_ecgs: Generated ECG array.
        gen_texts: Generated text embeddings.
        real_texts: Real text embeddings.
        real_labels: Integer labels for real texts.
        classifier: Trained ECG classifier.
        device: Torch device.
        batch_size: Inference batch size.

    Returns:
        Joint quality scalar in [0, 1].
    """
    import numpy
    import torch

    preds = []
    with torch.no_grad():
        for i in range(0, len(gen_ecgs), batch_size):
            x = torch.from_numpy(gen_ecgs[i: i + batch_size]).to(device)
            preds.append(classifier(x).argmax(dim=-1).cpu().numpy())
    preds = numpy.concatenate(preds)

    gn = gen_texts / (numpy.linalg.norm(gen_texts, axis=1, keepdims=True) + 1e-8)
    rn = real_texts / (numpy.linalg.norm(real_texts, axis=1, keepdims=True) + 1e-8)
    nn_labels = real_labels[(gn @ rn.T).argmax(axis=1)]
    return float((preds == nn_labels).mean())


def _build_real_data(data_dir: str, cache_dir: str, n_samples: int):
    import numpy

    import data as data_module

    train_ds, _, _ = data_module.build_datasets(data_dir, cache_dir)
    real_ecgs = numpy.stack([train_ds[i]["ecg"].numpy() for i in range(min(n_samples, len(train_ds)))])
    real_texts = numpy.stack([train_ds[i]["text_emb"].numpy() for i in range(len(train_ds))])
    real_labels = numpy.array([train_ds[i]["label"] for i in range(len(train_ds))])
    return train_ds, real_ecgs, real_texts, real_labels


def _run_one_cell(sampler_name, nfe, s_theta, s_phi, vpsde, real_ecgs, real_texts,
                  real_labels, classifier, device, cfg, n_samples, lead_idx: int = 1):
    import sample as sample_module

    print(f"\n{sampler_name.upper()} NFE={nfe}")
    gen_ecgs, gen_texts = sample_module.generate(
        s_theta, s_phi, vpsde, sampler_name, n_samples,
        cfg.eval.batch_size, nfe, cfg.eval.sampler.corrector_snr, device, cfg,
        lead_idx=lead_idx,
    )
    fid = ecg_fid(real_ecgs, gen_ecgs, device)
    cos = text_cosine_sim(gen_texts, real_texts)
    jq = joint_quality(gen_ecgs, gen_texts, real_texts, real_labels, classifier, device)
    print(f"  FID={fid:.2f}  cos_sim={cos:.4f}  joint_quality={jq:.4f}")
    return {"fid": fid, "cos_sim": cos, "joint_quality": jq}


def print_ablation_table(results: dict) -> None:
    """Print sampler × NFE ablation results as a formatted table.

    Args:
        results: Dict mapping (sampler_name, nfe) → metrics dict.
    """
    hdr = f"{'Sampler':>8} {'NFE':>6} {'FID':>10} {'CosSim':>10} {'JointQ':>10}"
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for (s, n), m in sorted(results.items()):
        print(f"{s:>8} {n:>6} {m['fid']:>10.2f} {m['cos_sim']:>10.4f} {m['joint_quality']:>10.4f}")


@modal_common.app.function(
    image=modal_common.image,
    gpu=modal_common.GPU,
    volumes=modal_common.VOLUME_MAP,
    timeout=10_800,
    secrets=modal_common.HF_SECRETS,
)
def eval_one_cell(sampler_name, nfe, checkpoint, n_samples, corrector_snr, lead_idx: int = 1):
    """Modal entry point for a single ablation cell.

    Args:
        sampler_name: Sampler to evaluate.
        nfe: Number of function evaluations.
        checkpoint: Checkpoint name stem.
        n_samples: Number of generated samples.
        corrector_snr: Langevin corrector SNR.
        lead_idx: Which lead to generate (0–11, default 1 = Lead II).
            Real comparison data is always Lead II, so keep this at 1 for FID.

    Returns:
        Tuple of (sampler_name, nfe, metrics).
    """
    import os

    import torch

    import sample as sample_module
    import sde as sde_module

    os.environ["HF_HOME"] = modal_common.HF_CACHE_DIR
    device = torch.device("cuda")
    cfg = cfg_module.Config()
    cfg.train.data_dir = f"{modal_common.REMOTE_CACHE}/ptbxl"
    cfg.train.data_cache_dir = modal_common.REMOTE_CACHE
    cfg.eval.sampler.corrector_snr = corrector_snr
    vpsde = sde_module.VPSDE(cfg.sde.beta_min, cfg.sde.beta_max, cfg.sde.T, cfg.sde.eps)

    s_theta, s_phi = sample_module.load_models(
        Path(modal_common.REMOTE_CKPTS) / f"{checkpoint}.pt", cfg, device
    )
    train_ds, real_ecgs, real_texts, real_labels = _build_real_data(
        cfg.train.data_dir, modal_common.REMOTE_CACHE, n_samples
    )
    n_classes = int(real_labels.max()) + 1 if real_labels.min() >= 0 else 5
    classifier = train_classifier(train_ds, n_classes, device)
    metrics = _run_one_cell(
        sampler_name, nfe, s_theta, s_phi, vpsde,
        real_ecgs, real_texts, real_labels, classifier, device, cfg, n_samples,
        lead_idx=lead_idx,
    )
    return sampler_name, nfe, metrics


@modal_common.app.local_entrypoint(name="evaluate")
def main(
    checkpoint: str = "final",
    n_samples: int = 1000,
    corrector_snr: float = 0.16,
    nfe: str = "100,500,1000",
    samplers: str = "s4,pc,em",
    lead_idx: int = 1,
):
    nfe_list = [int(n) for n in nfe.split(",")]
    sampler_list = samplers.split(",")
    cells = [
        (s, n, checkpoint, n_samples, corrector_snr, lead_idx)
        for s in sampler_list
        for n in nfe_list
    ]
    print(f"Launching {len(cells)} parallel eval cells…")
    results = {}
    for sampler_name, nfe_val, metrics in eval_one_cell.starmap(cells):
        results[(sampler_name, nfe_val)] = metrics
    print_ablation_table(results)


if __name__ == "__main__":
    import torch

    import sample as sample_module
    import sde as sde_module

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="final")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--data-dir", default="data/ptbxl")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--nfe", nargs="+", type=int, default=[100, 500, 1000])
    parser.add_argument("--samplers", nargs="+", default=["s4", "pc", "em"])
    parser.add_argument("--corrector-snr", type=float, default=0.16)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = cfg_module.Config()
    cfg.train.data_dir = args.data_dir
    cfg.train.data_cache_dir = args.cache_dir
    cfg.eval.sampler.corrector_snr = args.corrector_snr
    vpsde = sde_module.VPSDE(cfg.sde.beta_min, cfg.sde.beta_max, cfg.sde.T, cfg.sde.eps)

    s_theta, s_phi = sample_module.load_models(
        Path(args.checkpoint_dir) / f"{args.checkpoint}.pt", cfg, device
    )
    train_ds, real_ecgs, real_texts, real_labels = _build_real_data(
        args.data_dir, args.cache_dir, args.n_samples
    )
    n_classes = int(real_labels.max()) + 1 if real_labels.min() >= 0 else 5
    classifier = train_classifier(train_ds, n_classes, device)

    results = {}
    for s in args.samplers:
        for n in args.nfe:
            results[(s, n)] = _run_one_cell(
                s, n, s_theta, s_phi, vpsde,
                real_ecgs, real_texts, real_labels, classifier, device, cfg, args.n_samples,
            )
    print_ablation_table(results)
