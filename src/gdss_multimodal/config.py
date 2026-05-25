"""Dataclass configs for all hyperparameters, with YAML load/save support.

Usage
-----
Load defaults:
    cfg = Config()

Load from YAML:
    cfg = Config.from_yaml("experiments/run1.yaml")

Save to YAML:
    cfg.to_yaml("experiments/run1.yaml")

Override individual fields from an argparse Namespace:
    cfg.override({"train.lr": 1e-4, "train.batch_size": 256})
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class SDEConfig:
    beta_min: float = 0.1
    beta_max: float = 12.0
    T: float = 1.0
    eps: float = 1e-5


@dataclass
class ECGScoreConfig:
    # Text embedding dim (BioClinicalBERT CLS token)
    text_dim: int = 768
    # ECG shape
    n_leads: int = 1
    seq_len: int = 1000
    # U-Net architecture
    timestep_dim: int = 128
    channels: tuple = (128, 256, 512)
    bottleneck_ch: int = 1024
    # Lead identity conditioning: learned embedding injected into FiLM cond.
    # Set to 0 to disable (no lead conditioning).
    lead_emb_dim: int = 64


@dataclass
class TextScoreConfig:
    text_dim: int = 768
    # Must match ECGScoreConfig.bottleneck_ch
    moment_hidden_dim: int = 1024
    timestep_embed_dim: int = 256
    hidden_dim: int = 1024
    n_layers: int = 6


@dataclass
class PretrainConfig:
    """Hyperparameters for the ECGUNet autoencoder pre-training phase."""
    max_steps: int = 30_000
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    grad_clip: float = 1.0
    spectral_weight: float = 0.1
    log_every: int = 500
    save_every: int = 10_000
    checkpoint_dir: str = "checkpoints"
    resume: str = ""


@dataclass
class TrainConfig:
    # Paths (overridden by env vars in SLURM scripts)
    data_dir: str = "data/ptbxl"
    data_cache_dir: str = "cache"
    checkpoint_dir: str = "checkpoints"
    # Data
    batch_size: int = 256
    num_workers: int = 4
    # Optimisation
    lr: float = 3e-4
    weight_decay: float = 1e-4
    max_steps: int = 100_000
    warmup_steps: int = 2_000
    grad_clip: float = 1.0
    # Loss
    likelihood_weighting: bool = True
    # Classifier-free guidance drop prob for lead conditioning
    cfg_drop_prob: float = 0.10
    # Logging
    log_every: int = 500
    val_every: int = 1_000
    save_every: int = 5_000
    # Device / seed
    device: str = "cuda"
    seed: int = 42
    # Checkpoint paths
    pretrain_checkpoint: str = ""
    resume_checkpoint: str = ""


@dataclass
class SamplerConfig:
    n_steps: int = 1000
    corrector_snr: float = 0.16
    n_corrector_steps: int = 1
    sampler: str = "pc"


@dataclass
class EvalConfig:
    n_samples: int = 1000
    batch_size: int = 64
    figures_dir: str = "figures"
    k_neighbors: int = 1
    sampler: SamplerConfig = field(default_factory=SamplerConfig)


@dataclass
class Config:
    sde: SDEConfig = field(default_factory=SDEConfig)
    ecg_score: ECGScoreConfig = field(default_factory=ECGScoreConfig)
    text_score: TextScoreConfig = field(default_factory=TextScoreConfig)
    pretrain: PretrainConfig = field(default_factory=PretrainConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # ── YAML I/O ────────────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load a Config from a YAML file.  Missing keys use dataclass defaults."""
        with open(path) as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}
        return _from_dict(cls, raw)

    def to_yaml(self, path: str | Path) -> None:
        """Serialise the Config to a YAML file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(_to_dict(self), f, default_flow_style=False, sort_keys=False)

    # ── Field override ───────────────────────────────────────────────────────

    def override(self, overrides: dict[str, Any]) -> None:
        """Apply dot-separated overrides, e.g. {"train.lr": 1e-4}.

        Supports int, float, bool, and str leaves.  Tuple fields (e.g.
        ecg_score.channels) accept comma-separated ints: "128,256,512".
        """
        for key, value in overrides.items():
            parts = key.split(".")
            obj = self
            for part in parts[:-1]:
                obj = getattr(obj, part)
            leaf = parts[-1]
            current = getattr(obj, leaf)
            if isinstance(current, tuple):
                value = tuple(int(x) for x in str(value).split(","))
            elif isinstance(current, bool):
                value = str(value).lower() not in ("0", "false", "no")
            elif isinstance(current, int):
                value = int(value)
            elif isinstance(current, float):
                value = float(value)
            setattr(obj, leaf, value)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _to_dict(obj: Any) -> Any:
    """Recursively convert dataclasses to plain dicts; pass through primitives."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, tuple):
        return list(obj)
    return obj


def _from_dict(cls: type, data: dict[str, Any]) -> Any:
    """Recursively reconstruct a dataclass from a plain dict."""
    if not dataclasses.is_dataclass(cls):
        return data
    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        val = data[f.name]
        ft = f.type
        # Resolve string annotations
        if isinstance(ft, str):
            ft = _FIELD_TYPES.get(f"{cls.__name__}.{f.name}", ft)
        if dataclasses.is_dataclass(ft) and isinstance(val, dict):
            kwargs[f.name] = _from_dict(ft, val)
        elif f.name == "channels" or (isinstance(val, list) and
                                       isinstance(getattr(_default_instance(cls), f.name, None), tuple)):
            kwargs[f.name] = tuple(val) if isinstance(val, list) else val
        else:
            kwargs[f.name] = val
    return cls(**kwargs)


# Map "ClassName.field_name" → actual type for nested dataclass fields.
_FIELD_TYPES: dict[str, type] = {
    "Config.sde": SDEConfig,
    "Config.ecg_score": ECGScoreConfig,
    "Config.text_score": TextScoreConfig,
    "Config.pretrain": PretrainConfig,
    "Config.train": TrainConfig,
    "Config.eval": EvalConfig,
    "EvalConfig.sampler": SamplerConfig,
}


def _default_instance(cls: type) -> Any:
    try:
        return cls()
    except Exception:
        return None
