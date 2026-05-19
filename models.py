"""Score networks for joint (ECG, text) diffusion.

Classes:
    FiLM: Feature-wise Linear Modulation conditioning layer.
    SinusoidalTimestepEmbed: Sinusoidal embedding for continuous timesteps.
    ECGUNet: s_θ — ECG score network; small 1D U-Net with FiLM conditioning.
    TextScoreNet: s_φ — text score MLP with cross-modal FiLM conditioning.
"""

from __future__ import annotations

import math

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


class _ResBlock1D(nn.Module):
    """GroupNorm → Conv1d → FiLM → GroupNorm → Conv1d with residual."""

    def __init__(self, channels: int, cond_dim: int) -> None:
        super().__init__()
        groups = min(8, channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.film = FiLM(cond_dim, channels, hidden_dim=max(64, channels))
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        # FiLM expects (..., C); transpose (B, C, L) → (B, L, C) and back
        h = self.film(h.transpose(1, 2), cond).transpose(1, 2)
        h = self.conv2(self.act(self.norm2(h)))
        return x + h


class _Down1D(nn.Module):
    """Stride-2 Conv1d followed by a residual block."""

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv1d(in_ch, out_ch, 4, stride=2, padding=1)
        self.res = _ResBlock1D(out_ch, cond_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.res(self.proj(x), cond)


class _Up1D(nn.Module):
    """ConvTranspose1d upsample, merge skip via 1×1 conv, residual block."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, cond_dim: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, out_ch, 4, stride=2, padding=1)
        self.merge = nn.Conv1d(out_ch + skip_ch, out_ch, 1)
        self.res = _ResBlock1D(out_ch, cond_dim)

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-1] != skip.shape[-1]:
            x = x[..., : skip.shape[-1]]
        h = self.merge(torch.cat([x, skip], dim=1))
        return self.res(h, cond)


class ECGUNet(nn.Module):
    """s_θ(M1_t, M2_t, t) — ECG score network.

    Small 1D U-Net with strided convolutions and FiLM conditioning.
    Skip connections preserve spatial resolution through the score function,
    avoiding the bottleneck-then-expand issue of a pooled backbone approach.

    Conditioning signal: sinusoidal timestep embedding concatenated with a
    linear projection of the noisy text embedding → all U-Net residual blocks
    receive (time, text) jointly via FiLM.

    Args:
        text_dim: Text embedding dimensionality (768 for BioClinicalBERT).
        n_leads: Number of ECG leads.
        seq_len: Number of time steps per lead.
        timestep_dim: Sinusoidal timestep embedding dimensionality.
        channels: Channel widths for each encoder level (length = n_levels).
        bottleneck_ch: Channel width of the U-Net bottleneck.
    """

    def __init__(
        self,
        text_dim: int = 768,
        n_leads: int = 1,
        seq_len: int = 1000,
        timestep_dim: int = 128,
        channels: tuple = (32, 64, 128),
        bottleneck_ch: int = 256,
    ) -> None:
        super().__init__()
        self.n_leads = n_leads
        self.seq_len = seq_len
        self.bottleneck_ch = bottleneck_ch
        self._text_proj_dim = timestep_dim

        self.t_embed = SinusoidalTimestepEmbed(timestep_dim)
        self.text_proj = nn.Linear(text_dim, timestep_dim)
        cond_dim = timestep_dim * 2

        self.input_conv = nn.Conv1d(n_leads, channels[0], 3, padding=1)
        self.input_res = _ResBlock1D(channels[0], cond_dim)

        ch_list = list(channels)
        self.downs = nn.ModuleList()
        for i in range(len(ch_list) - 1):
            self.downs.append(_Down1D(ch_list[i], ch_list[i + 1], cond_dim))
        self.downs.append(_Down1D(ch_list[-1], bottleneck_ch, cond_dim))

        self.mid = _ResBlock1D(bottleneck_ch, cond_dim)

        self.ups = nn.ModuleList()
        # First up: bottleneck → last channel
        self.ups.append(_Up1D(bottleneck_ch, ch_list[-1], ch_list[-1], cond_dim))
        for i in range(len(ch_list) - 1, 0, -1):
            self.ups.append(_Up1D(ch_list[i], ch_list[i - 1], ch_list[i - 1], cond_dim))

        self.output_norm = nn.GroupNorm(min(8, channels[0]), channels[0])
        self.output_conv = nn.Conv1d(channels[0], n_leads, 1)

    def _cond(self, t_emb: torch.Tensor, text_t: torch.Tensor) -> torch.Tensor:
        return torch.cat([t_emb, nn.functional.silu(self.text_proj(text_t))], dim=-1)

    def _encode(self, ecg_t: torch.Tensor, cond: torch.Tensor) -> tuple:
        """Run encoder, return (bottleneck, list_of_skips)."""
        h = self.input_res(self.input_conv(ecg_t), cond)
        skips = [h]
        for down in self.downs[:-1]:
            h = down(h, cond)
            skips.append(h)
        h = self.downs[-1](h, cond)
        h = self.mid(h, cond)
        return h, skips

    def encode(self, ecg_t: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """Mean-pool bottleneck for cross-modal text conditioning.

        Args:
            ecg_t: Noisy ECG of shape (B, n_leads, seq_len).
            t_emb: Timestep embedding of shape (B, timestep_dim).

        Returns:
            Pooled bottleneck of shape (B, bottleneck_ch).
        """
        B = ecg_t.shape[0]
        text_zero = torch.zeros(B, self._text_proj_dim, device=ecg_t.device)
        cond = torch.cat([t_emb, text_zero], dim=-1)
        h, _ = self._encode(ecg_t, cond)
        return h.mean(dim=-1)

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
        cond = self._cond(t_emb, text_t)
        h, skips = self._encode(ecg_t, cond)
        h = self.ups[0](h, skips[-1], cond)
        for i, up in enumerate(self.ups[1:]):
            h = up(h, skips[-(i + 2)], cond)
        h = nn.functional.silu(self.output_norm(h))
        return self.output_conv(h)


class TextScoreNet(nn.Module):
    """s_φ(M2_t, M1_t, t) — text score network.

    4-layer residual MLP with SiLU activations. Timestep and cross-modal
    (ECG) conditioning are both injected via FiLM at every layer.

    Args:
        text_dim: Text embedding dimensionality (768).
        moment_hidden: ECG bottleneck dimensionality (matches ECGUNet.bottleneck_ch).
        timestep_dim: Sinusoidal timestep embedding dimensionality.
        hidden_dim: MLP hidden width.
        n_layers: Number of residual MLP layers.
    """

    def __init__(
        self,
        text_dim: int = 768,
        moment_hidden: int = 256,
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
            ecg_rep: Mean-pooled ECG bottleneck of shape (B, moment_hidden).
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
