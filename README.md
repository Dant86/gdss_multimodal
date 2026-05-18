# GDSS Multimodal — Joint ECG–Text Diffusion

Score-based generative model over paired (ECG waveform, clinical report embedding) data, using a **Symmetric Splitting Score Sampler (S4)** adapted from GDSS.

> Jo, J., Lee, S., & Hwang, S. J. (2022). *Score-based Generative Modeling of Graphs via the System of Stochastic Differential Equations.* ICML 2022. [arXiv:2202.02514](https://arxiv.org/abs/2202.02514)

---

## Architecture

### Forward process

Both modalities share a **Variance-Preserving SDE** with linear schedule:

```
β(t) = β_min + t·(β_max − β_min)
p(x_t | x_0) = N(α(t)·x_0, σ(t)²·I)
α(t) = exp(−0.5 · ∫₀ᵗ β(s) ds)
σ(t) = sqrt(1 − α(t)²)
```

### Score networks

**ECG score network s_θ** (`ECGScoreNet`):
1. [MOMENT-1-large](https://huggingface.co/AutonLab/MOMENT-1-large) backbone processes the noisy ECG waveform via channel-independent patch attention. The first half of transformer blocks is frozen; the second half is fine-tuned.
2. FiLM conditioning on the pooled MOMENT output using the sinusoidal timestep embedding.
3. Cross-modal FiLM conditioning on the noisy text embedding.
4. 2-layer MLP score head projecting to `(B, n_leads, seq_len)`.

**Text score network s_φ** (`TextScoreNet`):
4-layer residual MLP with SiLU activations. Timestep and ECG conditioning are injected at every layer via FiLM.

### Training objective

Joint denoising score matching with likelihood weighting λ(t) = σ(t)²:

```
L = E[σ²(t) · ‖s_θ(x_t, y_t, t) + ε₁/σ(t)‖²]
  + E[σ²(t) · ‖s_φ(y_t, h_θ(x_t, t), t) + ε₂/σ(t)‖²]
```

where `h_θ` is the timestep-conditioned mean-pooled MOMENT embedding of the noisy ECG.

### Reverse sampler — S4

The S4 sampler generalises the symmetric Strang splitting from GDSS (graph nodes ↔ adjacency) to arbitrary modality pairs (ECG ↔ text). One S4 step at time t:

1. Half-corrector on ECG — Langevin MCMC with adaptive step size (snr·‖x‖/‖score‖)²
2. Full predictor on text — Euler-Maruyama reverse SDE step
3. Half-corrector on ECG — symmetric counterpart of step 1

The symmetric arrangement reduces the local operator splitting error from **O(δt²)** (plain alternating PC) to **O(δt³)** via the Baker-Campbell-Hausdorff argument (GDSS Appendix B). Ablation baselines `pc_sampler` and `em_sampler` are provided with the same interface.

---

## Data — PTB-XL

**PTB-XL** is a large publicly available ECG dataset from PhysioNet:

> Wagner, P., Strodthoff, N., Bousseljot, R., Kreiseler, D., Lunze, F. I., Samek, W., & Schaeffter, T. (2020). *PTB-XL, a large publicly available electrocardiography dataset.* Scientific Data, 7(1), 154. [PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/)

- **21,837 records**, 10 seconds each, 12 leads
- Free text clinical reports in German (auto-translated to English in the CSV)
- Stratified 10-fold cross-validation splits: folds 1–8 train, fold 9 val, fold 10 test
- Two resolutions: 500 Hz (`records500`, 5000 samples) and 100 Hz (`records100`, 1000 samples)

This project uses **100 Hz, Lead II only** for fast iteration. Switch `SEQ_LEN`, `RECORDS_DIR`, and `LEAD_IDX` in `data.py` to restore full 12-lead 500 Hz training.

Text embeddings are produced by [Bio_ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT) ([CLS] token, 768-dim), cached to disk on first run.

---

## Setup

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment and install deps
uv venv && source .venv/bin/activate
uv pip install -e .

# momentfm pins old transformers — must be installed without its deps
uv pip install momentfm --no-deps
```

---

## Data preparation

PTB-XL requires a free PhysioNet account. The fastest source is the [Kaggle mirror](https://www.kaggle.com/datasets/khyeh0719/ptb-xl-dataset).

```bash
# Upload the zip to the Modal cache volume (single file = fast)
modal volume put gdss-cache ~/Downloads/ptbxl.zip ptbxl.zip

# Unzip inside the container
modal run download_ptbxl.py
```

---

## Training

```bash
# Modal (H100, recommended)
modal run train.py                          # defaults: 100k steps, batch 128, lr 2e-4
modal run train.py -- --max-steps 20000     # quick run

# Local
python train.py --device cuda --max-steps 5000 --data-dir /path/to/ptbxl
```

Checkpoints are saved every 5,000 steps to `checkpoints/` (local) or the `gdss-checkpoints` Modal volume.

---

## Sampling

```bash
# Modal
modal run sample.py -- --checkpoint final --sampler s4 --n-steps 1000 --n-samples 1000

# Local
python sample.py --checkpoint final --sampler s4 --n-steps 1000
```

Sampler options: `s4` (default), `pc`, `em`.

---

## Evaluation

```bash
# Modal — runs all sampler × NFE combinations in parallel
modal run evaluate.py -- --checkpoint final --nfe 100,500,1000 --samplers s4,pc,em

# Local
python evaluate.py --checkpoint final --nfe 100 500 1000 --samplers s4 pc em
```

Metrics reported:
- **ECG FID** — Fréchet distance in MOMENT-1-large feature space
- **Text cosine similarity** — mean max nearest-neighbour cosine sim
- **Joint quality** — fraction of generated pairs where the ECG classifier label matches the nearest real text label

---

## Project layout

```
config.py           Dataclass hyperparameter configuration
data.py             PTB-XL loading, BioClinicalBERT cache, PTBXLDataset
models.py           FiLM, SinusoidalTimestepEmbed, ECGScoreNet, TextScoreNet
sde.py              VP-SDE schedule and marginal probability helpers
solvers.py          S4, PC, and EM samplers
train.py            DSM training loop (local + Modal)
sample.py           Reverse diffusion generation (local + Modal)
evaluate.py         FID / cosine-sim / joint-quality + ablation table (local + Modal)
download_ptbxl.py   Unzip PTB-XL inside a Modal container
modal_common.py     Shared Modal image, volumes, and GPU config
```
