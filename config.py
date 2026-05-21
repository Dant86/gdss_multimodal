"""Dataclass configs for all hyperparameters."""
from dataclasses import dataclass, field


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
    # ECG shape (model always processes 1 lead at a time)
    n_leads: int = 1
    seq_len: int = 1000
    # U-Net architecture
    timestep_dim: int = 128
    channels: tuple = (128, 256, 512)
    bottleneck_ch: int = 1024
    # Lead identity conditioning: learned embedding injected into FiLM cond.
    # Set to 0 to disable (single-lead mode, no lead embedding).
    lead_emb_dim: int = 64
    # R-peak conditioning: small 1D-CNN encoder output dim injected into FiLM.
    # Set to 0 to disable.
    r_peak_enc_dim: int = 0


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
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    grad_clip: float = 1.0
    spectral_weight: float = 0.1   # λ_spec: weight of FFT magnitude loss
    log_every: int = 500
    save_every: int = 10_000
    checkpoint_dir: str = "checkpoints"
    # Path to a finished pretrain checkpoint to skip re-training
    resume: str = ""


@dataclass
class TrainConfig:
    # Data
    data_dir: str = "data/ptbxl"   # PhysioNet PTB-XL root (contains ptbxl_database.csv)
    data_cache_dir: str = "cache"  # BERT embeddings + ECG stats
    batch_size: int = 512
    num_workers: int = 4
    # Optimization
    lr: float = 3e-4
    weight_decay: float = 1e-4
    max_steps: int = 2_000
    warmup_steps: int = 2000
    grad_clip: float = 1.0
    # Loss
    likelihood_weighting: bool = True  # λ(t) = σ(t)²
    # Classifier-free guidance: probability of zeroing lead conditioning per sample.
    # Set to 0 to disable CFG (default). Typical value: 0.10–0.15.
    cfg_drop_prob: float = 0.10
    # R-peak CFG drop: probability of zeroing R-peak mask per sample during training
    # so the model also learns the R-peak-unconditional score.
    r_peak_drop_prob: float = 0.10
    # Logging
    log_every: int = 500
    val_every: int = 1000
    save_every: int = 5000
    checkpoint_dir: str = "checkpoints"
    # Device
    device: str = "cuda"
    seed: int = 42
    # Optional path to a pretrained ECGUNet checkpoint (s_theta weights only).
    # When set, DSM training starts from these weights instead of random init.
    pretrain_checkpoint: str = ""
    # Optional path to a full training checkpoint to resume from.
    # Loads EMA weights for both s_theta and s_phi; resets the optimiser
    # so the LR schedule restarts fresh from cfg.train.lr.
    resume_checkpoint: str = ""


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
    pretrain: PretrainConfig = field(default_factory=PretrainConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
