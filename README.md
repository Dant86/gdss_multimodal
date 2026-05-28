# GDSS Multimodal — Joint ECG–Text Diffusion

Score-based generative model over paired (ECG waveform, clinical report embedding) data, using a **Symmetric Splitting Score Sampler (S4)** adapted from GDSS.

> Jo, J., Lee, S., & Hwang, S. J. (2022). *Score-based Generative Modeling of Graphs via the System of Stochastic Differential Equations.* ICML 2022. [arXiv:2202.02514](https://arxiv.org/abs/2202.02514)

---

## Fitted distribution

The model jointly fits a **conditional generative distribution** over ECG waveforms and clinical text embeddings:

$$p_\theta\left(\mathbf{x},\, \mathbf{y} \;\middle|\; \ell\right)$$

where:

| Symbol | Meaning |
|--------|---------|
| $\mathbf{x} \in \mathbb{R}^{L}$ | ECG waveform for a single lead ($L = 1000$ samples at 100 Hz) |
| $\mathbf{y} \in \mathbb{R}^{768}$ | BioClinicalBERT [CLS] token embedding of the paired clinical report |
| $\ell \in \{0,\ldots,11\}$ | Lead identity (which of the 12 ECG leads is being modelled) |

The full PTB-XL dataset contains ~21 K recordings; the training split (~17 K recordings, 12 leads each) yields **~209 K training samples**. The model learns a shared score function across all leads, distinguished by the lead embedding $\ell$.

---

## Architecture

### Forward process

Both modalities share a **Variance-Preserving SDE** (VP-SDE) with linear noise schedule:

$$\beta(t) = \beta_{\min} + t\,(\beta_{\max} - \beta_{\min}), \qquad t \in [0, T]$$

The marginal distribution at time $t$ given clean data $\mathbf{x}_0$ is:

$$p(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}\left(\alpha(t)\,\mathbf{x}_0,\; \sigma(t)^2 \mathbf{I}\right)$$

where the signal and noise scales are:

$$\alpha(t) = \exp\left(-\frac{1}{2}\int_0^t \beta(s)\,ds\right), \qquad \sigma(t) = \sqrt{1 - \alpha(t)^2}$$

Parameters: $\beta_{\min} = 0.1$, $\beta_{\max} = 1.0$ (following GDSS; higher values create excessively noisy DSM targets). The same schedule is applied independently to $\mathbf{x}$ (ECG) and $\mathbf{y}$ (text embedding), yielding noisy pairs $(\mathbf{x}_t, \mathbf{y}_t)$ at each diffusion time $t$.

---

### Conditioning signal — FiLM cond vector

Every residual block in $s_\theta$ receives a shared conditioning vector $\mathbf{c}$ assembled by concatenation:

$$\mathbf{c} = \begin{bmatrix} \mathbf{e}_t \\ \phi(\mathbf{y}_t) \\ \psi(\ell) \end{bmatrix} \in \mathbb{R}^{3d}$$

where $d = 128$ (timestep dim):

| Component | Description |
|-----------|-------------|
| $\mathbf{e}_t = \mathrm{SinEmbed}(t) \in \mathbb{R}^d$ | Sinusoidal timestep embedding (Ho et al., 2020) |
| $\phi(\mathbf{y}_t) = \mathrm{SiLU}(W_\phi\,\mathbf{y}_t) \in \mathbb{R}^d$ | Linear projection of noisy text embedding |
| $\psi(\ell) = \mathrm{SiLU}(W_\psi\,\mathrm{Embed}(\ell)) \in \mathbb{R}^d$ | Lead identity embedding (nn.Embedding(12, 64) → Linear → SiLU) |

This vector is consumed by **FiLM** (Feature-wise Linear Modulation) at every residual block:

$$\mathrm{FiLM}(\mathbf{h},\, \mathbf{c}) = \gamma(\mathbf{c}) \odot \mathbf{h} + \beta(\mathbf{c})$$

where $\gamma, \beta$ are produced by a 2-layer MLP with zero-initialised output, so all blocks start as identity maps at initialisation.

**Classifier-free guidance (CFG)** drops the lead identity $\ell$ during training (probability $p_\ell = 0.10$), replacing it with a null sentinel so the model simultaneously learns the conditional and unconditional scores. At inference:

$$\tilde{s}_\theta(\mathbf{x}_t, \mathbf{y}_t, t, \ell) = (1 + w)\,s_\theta(\mathbf{x}_t, \mathbf{y}_t, t, \ell) - w\,s_\theta(\mathbf{x}_t, \mathbf{y}_t, t, \varnothing)$$

with guidance scale $w = 1.5$.

---

### ECG score network $s_\theta$ — ECGUNet

1D U-Net with strided convolutions, residual blocks, and FiLM conditioning at every layer. **Self-attention is applied at the top encoder level, the bottleneck, and every decoder level**, giving the network global context at all resolutions for modelling periodic QRS structure.

| Component | Config |
|-----------|--------|
| Input conv | 1→128, k=3 |
| Encoder levels | 128→256→512, stride-2 each |
| Encoder attention | Self-attention before final downsample |
| Bottleneck | 1024 channels, self-attention |
| Decoder levels | skip-connected, stride-2 transpose conv + **self-attention at each level** |
| Output conv | 128→1, k=1 |

**Pretraining** (15 K steps, lr=1e-3, OneCycleLR): ECGUNet is first trained as a pure autoencoder with reconstruction loss $\mathcal{L}_{\mathrm{pre}} = \mathcal{L}_{\mathrm{MSE}} + \lambda_{\mathrm{spec}}\,\mathcal{L}_{\mathrm{FFT}}$, where $\mathcal{L}_{\mathrm{FFT}}$ is MSE over normalised FFT magnitudes. This builds in ECG morphology — QRS complexes, 1/f PSD, baseline wander — before any diffusion training.

The bottleneck also serves as an ECG context extractor: `encode()` returns an L2-normalised mean-pooled vector passed to $s_\phi$.

---

### Text score network $s_\phi$

6-layer residual MLP. The ECG context is extracted by mean-pooling the ECGUNet bottleneck:

$$\mathbf{h}_\theta(\mathbf{x}_t, t, \ell) = \mathrm{L2Norm}\left(\mathrm{MeanPool}\left(\mathrm{Bottleneck}_{s_\theta}(\mathbf{x}_t, t, \ell)\right)\right) \in \mathbb{R}^{1024}$$

The text score is then:

$$s_\phi(\mathbf{y}_t, \mathbf{h}_\theta, t) = \mathrm{MLP}_\phi\left(\mathbf{y}_t;\; \bigl[\mathbf{e}_t \;\|\; \mathrm{SiLU}(W_{\mathrm{ecg}}\,\mathbf{h}_\theta)\bigr]\right)$$

FiLM conditioning with $(t,\, \mathbf{h}_\theta)$ applied at every residual layer.

**End-to-end alignment gradients**: the ECG representation $\mathbf{h}_\theta$ is computed within the joint backward pass (no `detach`), so $s_\phi$'s text alignment loss trains $s_\theta$'s encoder to produce alignment-useful representations, not just ECG-denoising-useful ones.

---

### Training objective

Joint **denoising score matching (DSM)** with min-SNR-5 likelihood weighting $w(t)$:

$$\mathcal{L} = \mathbb{E}_{t,\,\mathbf{x}_0,\,\mathbf{y}_0,\,\boldsymbol{\epsilon}}\left[w(t)\left(\left\|s_\theta(\mathbf{x}_t, \mathbf{y}_t, t, \ell) + \frac{\boldsymbol{\epsilon}_1}{\sigma(t)}\right\|^2 + \left\|s_\phi(\mathbf{y}_t, \mathbf{h}_\theta, t) + \frac{\boldsymbol{\epsilon}_2}{\sigma(t)}\right\|^2\right)\right]$$

The min-SNR-5 weight (Hang et al., 2023) balances the loss across diffusion times:

$$w(t) = \sigma(t)^2 \cdot \min\left(\mathrm{SNR}(t),\; 5\right), \qquad \mathrm{SNR}(t) = \frac{\alpha(t)^2}{\sigma(t)^2}$$

This prevents the loss from blowing up at small $t$ (where $\sigma(t) \to 0$) and downweights noisy large-$t$ samples, focusing capacity on the intermediate regime where ECG structure is learnable.

**Loss spike guard**: Rare bfloat16 instability spikes are absorbed before the backward pass:

$$\mathcal{L}_{\mathrm{clipped}} = \mathrm{clamp}(\mathcal{L},\; \max=50) \cdot \mathbf{1}[\mathcal{L} \neq \mathrm{NaN}]$$

**Gradient accumulation**: gradients are accumulated over 4 micro-batches (effective batch size 1024) to reduce DSM loss variance without exceeding GPU memory.

**EMA**: Both networks maintain exponential moving average weights (decay = 0.999), used exclusively at inference.

---

### Reverse sampler — S4

The S4 sampler generalises the symmetric Strang splitting from GDSS (graph nodes ↔ adjacency) to arbitrary modality pairs (ECG ↔ text). One S4 step at time $t$:

1. **Half-corrector** on ECG — Langevin MCMC with adaptive step size $(\mathrm{snr} \cdot \|\mathbf{x}\| / \|s_\theta\|)^2$
2. **Full predictor** on text — Euler-Maruyama reverse SDE step
3. **Half-corrector** on ECG — symmetric counterpart of step 1

The symmetric arrangement reduces the local operator splitting error from $\mathcal{O}(\delta t^2)$ (plain alternating PC) to $\mathcal{O}(\delta t^3)$ via the Baker-Campbell-Hausdorff argument (GDSS Appendix B).

---

## Data — PTB-XL

**PTB-XL** is a large publicly available ECG dataset from PhysioNet:

> Wagner, P., Strodthoff, N., Bousseljot, R., Kreiseler, D., Lunze, F. I., Samek, W., & Schaeffter, T. (2020). *PTB-XL, a large publicly available electrocardiography dataset.* Scientific Data, 7(1), 154. [PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/)

- **21,837 records**, 10 seconds each, 12 leads at 100 Hz (1000 samples/lead)
- Clinical reports originally in German; translated to English via Anthropic Batch API (stored as `report_en` column)
- Contaminated API responses (refusals, preambles) are automatically filtered from both the CSV and embedding cache
- Stratified 10-fold cross-validation splits: folds 1–8 train, fold 9 val, fold 10 test
- Multi-lead training expands each recording to 12 samples → **~209 K train samples**

**Text embeddings**: [Bio_ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT) [CLS] token, 768-dim, cached to disk on first run.

---

## Setup

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment with Python 3.13 (PyTorch requires <3.14)
uv venv --python 3.13 && source .venv/bin/activate
uv pip install -e ".[dev]"

# momentfm pins old transformers/huggingface_hub; install without its deps
uv pip install momentfm --no-deps
```

Copy `.env.example` → `.env` and fill in your cluster settings (used by SLURM scripts):

```bash
cp .env.example .env
```

---

## Data preparation

PTB-XL is available via [Kaggle](https://www.kaggle.com/datasets/khyeh0719/ptb-xl-dataset). The `fetch_data` app downloads it, translates reports to English via the Anthropic Batch API, and caches BioClinicalBERT embeddings:

```bash
# Requires KAGGLE_USERNAME / KAGGLE_KEY and ANTHROPIC_API_KEY in .env
python apps/fetch_data/main.py \
    --data-dir data/ptbxl \
    --cache-dir data/cache

# Or via SLURM (6 h, CPU-only job)
sbatch scripts/fetch_data.sh
```

The pipeline:
1. Download PTB-XL from Kaggle → extract to `--data-dir`
2. Submit all non-empty reports to the Anthropic Batch API for normalisation (Claude detects language and translates German → English)
3. Filter contaminated API responses; write `report_en` column to CSV
4. Compute per-lead signal statistics
5. Cache BioClinicalBERT embeddings to `--cache-dir`

Re-running translation is not required if waveforms and CSV already exist:

```bash
sbatch scripts/fetch_data.sh --skip-translation
```

---

## Training

```bash
# Step 1: pretrain ECGUNet as autoencoder (15 K steps)
sbatch scripts/pretrain.sh

# Step 2: joint DSM fine-tune from pretrained weights (100 K steps)
sbatch scripts/train.sh
# pretrain_final.pt is detected automatically; pass --pretrain-checkpoint to override
```

Config values can be overridden on the command line:

```bash
python apps/train/main.py \
    train.lr=3e-4 train.max_steps=200000
```

Checkpoints are saved every 5 K steps. Resume from the latest checkpoint automatically via `--resume-checkpoint`.

---

## Sampling

```bash
sbatch scripts/visualize.sh
# or locally:
python apps/visualize/main.py \
    --checkpoint checkpoints/final.pt \
    --sampler s4 \
    --n-steps 1000 \
    --n-gen 300 \
    --cfg-scale 1.5 \
    --output-dir figures
```

Sampler options: `s4` (default, $\mathcal{O}(\delta t^3)$ splitting error), `pc` (predictor-corrector), `em` (Euler-Maruyama).

---

## Evaluation

```bash
sbatch scripts/eval.sh
# or locally:
python apps/eval/main.py \
    --data-dir data/ptbxl \
    --cache-dir data/cache \
    --checkpoint-dir checkpoints \
    --figures-dir figures \
    --n-samples 1000
```

Metrics reported:
- **ECG FID** — Fréchet distance in MOMENT-1-large feature space
- **Text cosine similarity** — mean max nearest-neighbour cosine sim between generated and real text embeddings
- **Joint quality** — label agreement between generated ECGs and their nearest text neighbours

---

## Project layout

```
src/gdss_multimodal/
    __init__.py         Package marker
    config.py           Dataclass hyperparameter configuration (YAML ↔ dataclass)
    sde.py              VP-SDE schedule and marginal probability helpers
    models.py           FiLM, SinEmbed, ECGUNet (s_θ), TextScoreNet (s_φ)
    solvers.py          S4, PC, and EM samplers + SAMPLERS dispatch dict
    data.py             PTB-XL loading, BioClinicalBERT cache, PyTorch datasets
    sample.py           load_models() and generate() entry points

apps/
    fetch_data/main.py  Kaggle download → Anthropic Batch API → BERT cache
    pretrain/main.py    ECGUNet autoencoder pretraining (MSE + spectral loss)
    train/main.py       Joint DSM training loop with EMA, CFG, min-SNR-5 weighting
    eval/main.py        FID / cosine-sim / joint-quality metrics + bar chart
    visualize/main.py   Publication figures: waveform grid, PSD, text neighbours

scripts/
    fetch_data.sh       SLURM: 6 h, CPU — data download and preprocessing
    pretrain.sh         SLURM: 12 h, 1 GPU — ECGUNet autoencoder pretraining
    train.sh            SLURM: 12 h, 1 GPU — joint DSM training
    eval.sh             SLURM: 12 h, 1 GPU — evaluation metrics
    visualize.sh        SLURM: 2 h, 1 GPU — figure generation

tests/                  pytest suite (100% src/ coverage)
```
