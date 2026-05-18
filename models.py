"""Score networks for joint (ECG, text) diffusion.

Classes:
    FiLM: Feature-wise Linear Modulation conditioning layer.
    SinusoidalTimestepEmbed: Sinusoidal embedding for continuous timesteps.
    ECGScoreNet: s_θ — ECG score network backed by MOMENT-1-large.
    TextScoreNet: s_φ — text score MLP with cross-modal FiLM conditioning.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class FiLM(nn.Module):
    """2-layer MLP mapping a conditioning vector to (scale, shift) for FiLM.

    Applies FiLM(x, c) = γ(c) ⊙ x + β(c) to a feature tensor x.
    Last layer is zero-initialised so γ=1, β=0 at the start of training,
    preserving pretrained features during fine-tuning.

    Args:
        cond_dim: Dimensionality of the conditioning input.
        feature_dim: Dimensionality of the feature tensor to modulate.
        hidden_dim: Hidden layer width of the MLP.
    """

    def __init__(self, cond_dim: int, feature_dim: int, hidden_dim: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * feature_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias[:feature_dim], 1.0)
        nn.init.zeros_(self.net[-1].bias[feature_dim:])

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Apply FiLM modulation.

        Args:
            x: Feature tensor of shape (..., feature_dim).
            cond: Conditioning vector of shape (B, cond_dim).

        Returns:
            Modulated tensor of the same shape as x.
        """
        params = self.net(cond)
        gamma, beta = params.chunk(2, dim=-1)
        for _ in range(x.dim() - gamma.dim()):
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        return gamma * x + beta


class SinusoidalTimestepEmbed(nn.Module):
    """Maps scalar timestep t ∈ [0, 1] to a d-dimensional sinusoidal embedding.

    Follows Ho et al. (DDPM, 2020), scaled to [0, 10000] internally.

    Args:
        dim: Output embedding dimensionality (must be even).
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        assert dim % 2 == 0, "embedding dim must be even"
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed a batch of timesteps.

        Args:
            t: Timesteps of shape (B,), values in [0, 1].

        Returns:
            Embeddings of shape (B, dim).
        """
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t[:, None] * 1000 * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ECGScoreNet(nn.Module):
    """s_θ(M1_t, M2_t, t) — ECG score network.

    Architecture:
        1. MOMENT-1-large backbone (first half frozen, second half fine-tuned).
           Gradient checkpointing is disabled so gradients flow correctly
           across the frozen/unfrozen boundary.
        2. FiLM noise conditioning on the pooled MOMENT output.
        3. Cross-modal FiLM conditioning on the noisy text embedding.
        4. Score head: 2-layer MLP projecting to (B, n_leads, seq_len).

    Args:
        text_dim: Text embedding dimensionality (768 for BioClinicalBERT).
        moment_hidden: MOMENT-1-large hidden dimensionality (1024).
        timestep_dim: Sinusoidal timestep embedding dimensionality.
        film_hidden: Hidden width of FiLM MLPs.
        score_head_hidden: Hidden width of the score head MLP.
        n_leads: Number of ECG leads to generate.
        seq_len: Number of time steps per lead.
        freeze_first_n: Number of MOMENT transformer blocks to freeze.
            Defaults to half of all blocks when None.
    """

    def __init__(
        self,
        text_dim: int = 768,
        moment_hidden: int = 1024,
        timestep_dim: int = 256,
        film_hidden: int = 512,
        score_head_hidden: int = 512,
        n_leads: int = 12,
        seq_len: int = 5000,
        freeze_first_n: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.n_leads = n_leads
        self.seq_len = seq_len

        self.t_embed = SinusoidalTimestepEmbed(timestep_dim)
        self.moment = self._load_moment()
        self._freeze_moment(freeze_first_n)

        self.t_film = FiLM(timestep_dim, moment_hidden, film_hidden)
        self.text_proj = nn.Linear(text_dim, moment_hidden)
        self.cross_film = FiLM(moment_hidden, moment_hidden, film_hidden)

        self.score_head = nn.Sequential(
            nn.Linear(moment_hidden, score_head_hidden),
            nn.SiLU(),
            nn.Linear(score_head_hidden, n_leads * seq_len),
        )

    def _load_moment(self) -> nn.Module:
        import momentfm  # noqa: PLC0415

        return momentfm.MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large",
            model_kwargs={"task_name": "embedding"},
        )

    def _freeze_moment(self, freeze_first_n: Optional[int]) -> None:
        """Freeze patch/pos embeddings and the first N transformer blocks.

        Args:
            freeze_first_n: Number of blocks to freeze. Defaults to half.
        """
        for name, param in self.moment.named_parameters():
            if any(k in name for k in (
                "patch_embed", "pos_embed", "cls_token",
                "patch_embedding", "positional_encoding",
            )):
                param.requires_grad_(False)

        blocks = [
            m for m in self.moment.modules()
            if m.__class__.__name__ in (
                "Block", "BertLayer", "TransformerLayer", "EncoderLayer"
            )
        ]
        n_freeze = freeze_first_n if freeze_first_n is not None else len(blocks) // 2
        for block in blocks[:n_freeze]:
            for param in block.parameters():
                param.requires_grad_(False)

        inner = self.moment.model if hasattr(self.moment, "model") else self.moment
        if hasattr(inner, "config"):
            inner.config.gradient_checkpointing = False
        for m in inner.modules():
            if hasattr(m, "gradient_checkpointing"):
                m.gradient_checkpointing = False

    def _moment_encode(self, ecg: torch.Tensor) -> torch.Tensor:
        """Run MOMENT embed and return mean-pooled representation.

        Args:
            ecg: ECG waveform of shape (B, n_leads, seq_len).

        Returns:
            Pooled embedding of shape (B, moment_hidden).
        """
        B = ecg.shape[0]
        mask = torch.ones(B, ecg.shape[-1], device=ecg.device, dtype=torch.long)
        inner = self.moment.model if hasattr(self.moment, "model") else self.moment
        out = inner.embed(x_enc=ecg, input_mask=mask, reduction="mean")
        return out.embeddings

    def _moment_forward_with_film(
        self, ecg: torch.Tensor, t_emb: torch.Tensor
    ) -> torch.Tensor:
        """Extract timestep-conditioned ECG representation for cross-modal use.

        Args:
            ecg: Noisy ECG waveform of shape (B, n_leads, seq_len).
            t_emb: Timestep embedding of shape (B, timestep_dim).

        Returns:
            Conditioned representation of shape (B, moment_hidden).
        """
        h = self._moment_encode(ecg)
        return self.t_film(h, t_emb)

    def forward(
        self, ecg_t: torch.Tensor, text_t: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Estimate the score of the ECG marginal at time t.

        Args:
            ecg_t: Noisy ECG of shape (B, n_leads, seq_len).
            text_t: Noisy text embedding of shape (B, text_dim).
            t: Timesteps of shape (B,).

        Returns:
            Score estimate of shape (B, n_leads, seq_len).
        """
        t_emb = self.t_embed(t)
        h = self._moment_forward_with_film(ecg_t, t_emb)
        text_cond = self.text_proj(text_t)
        h = self.cross_film(h, text_cond)
        return self.score_head(h).view(-1, self.n_leads, self.seq_len)


class TextScoreNet(nn.Module):
    """s_φ(M2_t, M1_t, t) — text score network.

    4-layer residual MLP with SiLU activations. Timestep and cross-modal
    (ECG) conditioning are both injected via FiLM at every layer.

    Args:
        text_dim: Text embedding dimensionality (768).
        moment_hidden: ECG mean-pooled representation dimensionality (1024).
        timestep_dim: Sinusoidal timestep embedding dimensionality.
        hidden_dim: MLP hidden width.
        n_layers: Number of residual MLP layers.
    """

    def __init__(
        self,
        text_dim: int = 768,
        moment_hidden: int = 1024,
        timestep_dim: int = 256,
        hidden_dim: int = 512,
        n_layers: int = 4,
    ) -> None:
        super().__init__()
        self.t_embed = SinusoidalTimestepEmbed(timestep_dim)
        self.ecg_proj = nn.Linear(moment_hidden, hidden_dim)
        self.input_proj = nn.Linear(text_dim, hidden_dim)
        cond_dim = timestep_dim + hidden_dim
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)]
        )
        self.film_layers = nn.ModuleList(
            [FiLM(cond_dim, hidden_dim, hidden_dim) for _ in range(n_layers)]
        )
        self.act = nn.SiLU()
        self.output_proj = nn.Linear(hidden_dim, text_dim)

    def forward(
        self, text_t: torch.Tensor, ecg_rep: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Estimate the score of the text marginal at time t.

        Args:
            text_t: Noisy text embedding of shape (B, text_dim).
            ecg_rep: Mean-pooled ECG representation of shape (B, moment_hidden).
            t: Timesteps of shape (B,).

        Returns:
            Score estimate of shape (B, text_dim).
        """
        t_emb = self.t_embed(t)
        ecg_cond = self.act(self.ecg_proj(ecg_rep))
        cond = torch.cat([t_emb, ecg_cond], dim=-1)
        h = self.act(self.input_proj(text_t))
        for layer, film in zip(self.layers, self.film_layers):
            residual = h
            h = layer(h)
            h = film(h, cond)
            h = self.act(h) + residual
        return self.output_proj(h)
