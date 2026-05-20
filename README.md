# GDSS Multimodal — Joint ECG–Text Diffusion

Score-based generative model over paired (ECG waveform, clinical report embedding) data, using a **Symmetric Splitting Score Sampler (S4)** adapted from GDSS.

> Jo, J., Lee, S., & Hwang, S. J. (2022). *Score-based Generative Modeling of Graphs via the System of Stochastic Differential Equations.* ICML 2022. [arXiv:2202.02514](https://arxiv.org/abs/2202.02514)

---

## Fitted distribution

The model jointly fits a **conditional generative distribution** over ECG waveforms and clinical text embeddings:

$$p_\theta\!\left(\mathbf{x},\, \mathbf{y} \;\middle|\; \ell,\, \mathbf{r}\right)$$

where:

| Symbol | Meaning |
|--------|---------|
| $\mathbf{x} \in \mathbb{R}^{L}$ | ECG waveform for a single lead ($L = 1000$ samples at 100 Hz) |
| $\mathbf{y} \in \mathbb{R}^{768}$ | BioClinicalBERT [CLS] token embedding of the paired clinical report |
| $\ell \in \{0,\ldots,11\}$ | Lead identity (which of the 12 ECG leads is being modelled) |
| $\mathbf{r} \in \{0,1\}^{L}$ | Binary R-peak mask — 1 at QRS spike positions, 0 elsewhere |

Training uses all 12 leads of each PTB-XL recording, expanding the effective dataset from 17 K to **209 K samples**. The model learns a shared score function across all leads, distinguished by the lead embedding $\ell$, and anchors QRS spike positions via the R-peak mask $\mathbf{r}$.

---

## Architecture

### Forward process

Both modalities share a **Variance-Preserving SDE** (VP-SDE) with linear noise schedule:

$$\beta(t) = \beta_{\min} + t\,(\beta_{\max} - \beta_{\min}), \qquad t \in [0, T]$$

The marginal distribution at time $t$ given clean data $\mathbf{x}_0$ is:

$$p(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}\!\left(\alpha(t)\,\mathbf{x}_0,\; \sigma(t)^2 \mathbf{I}\right)$$

where the signal and noise scales are:

$$\alpha(t) = \exp\!\left(-\tfrac{1}{2}\int_0^t \beta(s)\,\mathrm{d}s\right), \qquad \sigma(t) = \sqrt{1 - \alpha(t)^2}$$

The same schedule is applied independently to $\mathbf{x}$ (ECG) and $\mathbf{y}$ (text embedding), yielding noisy pairs $(\mathbf{x}_t, \mathbf{y}_t)$ at each diffusion time $t$.

---

### Conditioning signal — FiLM cond vector

Every residual block in $s_\theta$ receives a shared conditioning vector $\mathbf{c}$ assembled by concatenation:

$$\mathbf{c} = \bigl[\,\mathbf{e}_t \;\|\; \phi(\mathbf{y}_t) \;\|\; \psi(\ell) \;\|\; \rho(\mathbf{r})\,\bigr] \;\in\; \mathbb{R}^{3d + d_r}$$

where $d = 128$ (timestep dim) and $d_r = 64$ (R-peak encoder dim):

| Component | Description |
|-----------|-------------|
| $\mathbf{e}_t = \mathrm{SinEmbed}(t) \in \mathbb{R}^d$ | Sinusoidal timestep embedding (Ho et al., 2020) |
| $\phi(\mathbf{y}_t) = \mathrm{SiLU}(W_\phi\,\mathbf{y}_t) \in \mathbb{R}^d$ | Linear projection of noisy text embedding |
| $\psi(\ell) = \mathrm{SiLU}(W_\psi\,\mathrm{Embed}(\ell)) \in \mathbb{R}^d$ | Lead identity embedding (nn.Embedding(12, 64) → Linear → SiLU) |
| $\rho(\mathbf{r}) = \mathrm{RPeakEncoder}(\mathbf{r}) \in \mathbb{R}^{d_r}$ | R-peak CNN encoder (see below) |

This vector is consumed by **FiLM** (Feature-wise Linear Modulation) at every residual block:

$$\mathrm{FiLM}(\mathbf{h},\, \mathbf{c}) = \gamma(\mathbf{c}) \odot \mathbf{h} + \beta(\mathbf{c})$$

where $\gamma, \beta$ are produced by a 2-layer MLP with zero-initialised output, so all blocks start as identity maps at initialisation.

**Classifier-free guidance (CFG)** is implemented by independently dropping $\ell$ (with probability $p_\ell = 0.10$) and $\mathbf{r}$ (with probability $p_r = 0.10$) during training, replacing them with zero vectors so the model simultaneously learns the conditional and unconditional scores. At inference, CFG is applied to the lead dimension:

$$\tilde{s}_\theta(\mathbf{x}_t, \mathbf{y}_t, t, \ell, \mathbf{r}) = (1 + w)\,s_\theta(\mathbf{x}_t, \mathbf{y}_t, t, \ell, \mathbf{r}) - w\,s_\theta(\mathbf{x}_t, \mathbf{y}_t, t, \varnothing, \mathbf{r})$$

with guidance scale $w = 1.5$.

---

### R-peak encoder $\rho$

The R-peak mask $\mathbf{r} \in \{0,1\}^L$ (detected on Lead II via XQRS, shared across all leads) is encoded by a lightweight 1D CNN:

$$\rho(\mathbf{r}) = \mathrm{AvgPool}(\mathrm{Conv}_3 \to \mathrm{Conv}_2 \to \mathrm{Conv}_1)(\mathbf{r})$$

| Layer | Config | Output shape |
|-------|--------|-------------|
| Conv1d | 1→32, k=9, stride=1 | $(B,\,32,\,L)$ |
| Conv1d | 32→64, k=7, stride=4 | $(B,\,64,\,L/4)$ |
| Conv1d | 64→64, k=7, stride=4 | $(B,\,64,\,L/16)$ |
| AdaptiveAvgPool1d(1) | — | $(B,\,64,\,1)$ |
| Flatten | — | $(B,\,64)$ |

The encoder compresses sparse beat positions into a dense vector that captures heart rate, rhythm regularity, and beat phase — giving the score network a stable anchor for QRS spike generation.

At inference, R-peak masks are synthesised at a specified heart rate $f_\mathrm{HR}$ with small jitter $\epsilon \sim \mathrm{Uniform}(-\delta, \delta)$ where $\delta = 0.05 \times T_\mathrm{RR}$:

$$T_\mathrm{RR} = \frac{f_s \cdot 60}{f_\mathrm{HR}}, \qquad r_k = \left\lfloor T_\mathrm{phase} + k\,(T_\mathrm{RR} + \epsilon_k) \right\rceil$$

---

### ECG score network $s_\theta$ — ECGUNet

1D U-Net with strided convolutions, residual blocks, and FiLM conditioning at every layer.

| Component | Config |
|-----------|--------|
| Input conv | 1→64, k=3 |
| Encoder levels | 64→128→256, stride-2 each |
| Stride-4 attention | 8 heads, seq_len/4 positions (~2 cardiac cycles) |
| Bottleneck | 512 channels, self-attention (8 heads) |
| Decoder levels | skip-connected, stride-2 transpose conv |
| Output conv | 64→1, k=1 |
| Parameters | ~9.3 M |

**Pretraining** (30 K steps, lr=1e-3, OneCycleLR): ECGUNet is first trained as a pure autoencoder with reconstruction loss $\mathcal{L}_\mathrm{pre} = \mathcal{L}_\mathrm{MSE} + \lambda_\mathrm{spec}\,\mathcal{L}_\mathrm{FFT}$, where $\mathcal{L}_\mathrm{FFT}$ is MSE over normalised FFT magnitudes. This builds in ECG morphology — QRS complexes, 1/f PSD, baseline wander — before any diffusion training.

---

### Text score network $s_\phi$

4-layer residual MLP. The ECG context is extracted by mean-pooling the ECGUNet bottleneck:

$$\mathbf{h}_\theta(\mathbf{x}_t, t, \ell, \mathbf{r}) = \mathrm{MeanPool}\!\left(\mathrm{Bottleneck}_{s_\theta}(\mathbf{x}_t, t, \ell, \mathbf{r})\right) \in \mathbb{R}^{512}$$

The text score is then:

$$s_\phi(\mathbf{y}_t, \mathbf{h}_\theta, t) = \mathrm{MLP}_\phi\!\left(\mathbf{y}_t;\; \bigl[\mathbf{e}_t \;\|\; \mathrm{SiLU}(W_\mathrm{ecg}\,\mathbf{h}_\theta)\bigr]\right)$$

FiLM conditioning with $(t,\, \mathbf{h}_\theta)$ applied at every residual layer.

---

### Training objective

Joint **denoising score matching (DSM)** with min-SNR-5 likelihood weighting $w(t)$:

$$\mathcal{L} = \mathbb{E}_{t,\,\mathbf{x}_0,\,\mathbf{y}_0,\,\boldsymbol{\epsilon}}\!\left[w(t)\left(\left\|s_\theta(\mathbf{x}_t, \mathbf{y}_t, t, \ell, \mathbf{r}) + \frac{\boldsymbol{\epsilon}_1}{\sigma(t)}\right\|^2 + \left\|s_\phi(\mathbf{y}_t, \mathbf{h}_\theta, t) + \frac{\boldsymbol{\epsilon}_2}{\sigma(t)}\right\|^2\right)\right]$$

The min-SNR-5 weight (Hang et al., 2023) balances the loss across diffusion times:

$$w(t) = \sigma(t)^2 \cdot \min\!\left(\mathrm{SNR}(t),\; 5\right), \qquad \mathrm{SNR}(t) = \frac{\alpha(t)^2}{\sigma(t)^2}$$

This prevents the loss from blowing up at small $t$ (where $\sigma(t) \to 0$) and downweights noisy large-$t$ samples, focusing capacity on the intermediate regime where ECG structure is learnable.

**Loss spike guard**: Rare bfloat16 instability spikes are absorbed before the backward pass:

$$\mathcal{L}_\mathrm{clipped} = \mathrm{clamp}(\mathcal{L},\; \max=50) \cdot \mathbf{1}[\mathcal{L} \neq \mathrm{NaN}]$$

**EMA**: Both networks maintain exponential moving average weights (decay = 0.9999), used exclusively at inference.

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
- Free text clinical reports in German (auto-translated to English in the CSV)
- Stratified 10-fold cross-validation splits: folds 1–8 train, fold 9 val, fold 10 test
- Multi-lead training expands each recording to 12 samples → **~209 K train samples**

**Text embeddings**: [Bio_ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT) [CLS] token, 768-dim, cached on first run.

**R-peak masks**: Computed once via XQRS (wfdb) on Lead II of each recording, cached to `rpeak_masks.pkl`. Detection failures fall back to an all-zeros mask (unconditional).

---

## Setup

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment and install deps
uv venv && source .venv/bin/activate
uv pip install -e .
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
# Step 1: pretrain ECGUNet as autoencoder (30K steps, ~20 min on H100)
modal run pretrain.py

# Step 2: DSM fine-tune from pretrained weights (100K steps, ~80 min on H100)
modal run train.py --pretrain-checkpoint /vol/checkpoints/pretrain_final.pt --max-steps 100000

# Or run the full pipeline unattended
bash pipeline_v3.sh
```

Checkpoints saved every 5 K steps to the `gdss-checkpoints` Modal volume.

---

## Sampling

```bash
# Modal — conditional on Lead II at 72 bpm, CFG scale 1.5
modal run sample.py \
    --checkpoint final \
    --sampler s4 \
    --n-steps 1000 \
    --n-samples 1000 \
    --cfg-scale 1.5 \
    --heart-rate-bpm 72.0
```

The `--heart-rate-bpm` argument controls the synthetic R-peak mask used at inference. Try 50 (bradycardia), 72 (normal sinus rhythm), 100 (tachycardia).

Sampler options: `s4` (default), `pc`, `em`.

---

## Evaluation

```bash
modal run evaluate.py -- --checkpoint final --nfe 100,500,1000 --samplers s4,pc,em
```

Metrics:
- **ECG FID** — Fréchet distance in MOMENT-1-large feature space
- **Text cosine similarity** — mean max nearest-neighbour cosine sim
- **Joint quality** — label agreement between generated ECGs and nearest-text labels

---

## Project layout

```
config.py           Dataclass hyperparameter configuration
data.py             PTB-XL loading, BioClinicalBERT + R-peak caches, datasets
models.py           FiLM, SinEmbed, _RPeakEncoder, ECGUNet (s_θ), TextScoreNet (s_φ)
sde.py              VP-SDE schedule and marginal probability helpers
solvers.py          S4, PC, and EM samplers
pretrain.py         ECGUNet autoencoder pretraining (local + Modal)
train.py            DSM training loop (local + Modal)
sample.py           Reverse diffusion generation + make_rpeak_mask (local + Modal)
evaluate.py         FID / cosine-sim / joint-quality metrics (local + Modal)
visualize.py        Publication figures: waveform grid, PSD, text neighbours
pipeline_v3.sh      Unattended: pretrain → DSM train → sample → visualize
download_ptbxl.py   Unzip PTB-XL inside a Modal container
modal_common.py     Shared Modal image, volumes, and GPU config
```
