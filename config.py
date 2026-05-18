"""Dataclass configs for all hyperparameters."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SDEConfig:
    beta_min: float = 0.1
    beta_max: float = 20.0
    T: float = 1.0
    eps: float = 1e-5


@dataclass
class MOMENTConfig:
    # Number of transformer layers to freeze (first half); rest are fine-tuned
    freeze_first_n: Optional[int] = None  # None = auto (half of total layers)
    # Sinusoidal noise-level embedding dim
    timestep_embed_dim: int = 256
    # FiLM MLP hidden dim (2-layer MLP: timestep_embed_dim -> film_hidden -> 2*d_model)
    film_hidden_dim: int = 512
    # MOMENT's hidden dim for MOMENT-1-large
    moment_hidden_dim: int = 1024


@dataclass
class ECGScoreConfig:
    moment: MOMENTConfig = field(default_factory=MOMENTConfig)
    # Text embedding dim (BioClinicalBERT CLS token)
    text_dim: int = 768
    # Score head hidden dim
    score_head_hidden: int = 512
    # ECG shape
    n_leads: int = 1
    seq_len: int = 1000


@dataclass
class TextScoreConfig:
    text_dim: int = 768
    moment_hidden_dim: int = 1024
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
    log_every: int = 10
    val_every: int = 2000
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
