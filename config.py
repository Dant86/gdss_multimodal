"""Dataclass configs for all hyperparameters."""
from dataclasses import dataclass, field


@dataclass
class SDEConfig:
    beta_min: float = 0.1
    beta_max: float = 20.0
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
    channels: tuple = (32, 64, 128)
    bottleneck_ch: int = 256


@dataclass
class TextScoreConfig:
    text_dim: int = 768
    # Must match ECGScoreConfig.bottleneck_ch
    moment_hidden_dim: int = 256
    timestep_embed_dim: int = 256
    hidden_dim: int = 512
    n_layers: int = 4


@dataclass
class TrainConfig:
    # Data
    data_dir: str = "data/ptbxl"   # PhysioNet PTB-XL root (contains ptbxl_database.csv)
    data_cache_dir: str = "cache"  # BERT embeddings + ECG stats
    batch_size: int = 128
    num_workers: int = 0
    # Optimization
    lr: float = 2e-4
    weight_decay: float = 1e-4
    max_steps: int = 2_000
    warmup_steps: int = 1000
    grad_clip: float = 1.0
    # Loss
    likelihood_weighting: bool = True  # λ(t) = σ(t)²
    # Logging
    log_every: int = 500
    val_every: int = 500
    save_every: int = 5000
    checkpoint_dir: str = "checkpoints"
    # Device
    device: str = "cuda"
    seed: int = 42


@dataclass
class SamplerConfig:
    # Number of function evaluations
    n_steps: int = 1000
    # Corrector SNR for adaptive Langevin step size (GDSS Appendix C)
    corrector_snr: float = 0.16
    # Number of corrector steps per predictor step
    n_corrector_steps: int = 1
    sampler: str = "s4"  # "s4" | "pc" | "em"


@dataclass
class EvalConfig:
    n_samples: int = 1000
    batch_size: int = 32
    # FID neighbour count for text cosine sim
    k_neighbors: int = 1
    sampler: SamplerConfig = field(default_factory=SamplerConfig)


@dataclass
class Config:
    sde: SDEConfig = field(default_factory=SDEConfig)
    ecg_score: ECGScoreConfig = field(default_factory=ECGScoreConfig)
    text_score: TextScoreConfig = field(default_factory=TextScoreConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
