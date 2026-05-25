"""Variance-Preserving SDE (VP-SDE) with a linear noise schedule.

Schedule:
    β(t) = β_min + t·(β_max − β_min)
    ∫₀ᵗ β(s) ds = β_min·t + 0.5·(β_max − β_min)·t²

Forward marginal:
    p(x_t | x_0) = N(α(t)·x_0, σ(t)²·I)
    α(t) = exp(−0.5 · ∫₀ᵗ β(s) ds)
    σ(t) = sqrt(1 − α(t)²)

DSM score target:
    ∇_x log p(x_t | x_0) = −ε / σ(t)   where x_t = α(t)·x_0 + σ(t)·ε

Reverse SDE (Itô):
    dx = [f(x,t) − g(t)²·∇_x log p] dt + g(t) dW̄
    f(x, t) = −0.5·β(t)·x
    g(t)    = sqrt(β(t))
"""

from __future__ import annotations

import torch


class VPSDE:
    """VP-SDE with linear noise schedule.

    Operates on arbitrary tensor shapes; both ECG and text share this SDE.

    Args:
        beta_min: Minimum noise level.
        beta_max: Maximum noise level.
        T: End time of the forward process.
        eps: Start time of the reverse process (small positive value).
    """

    def __init__(
        self,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        T: float = 1.0,
        eps: float = 1e-5,
    ) -> None:
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.T = T
        self.eps = eps

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """Instantaneous noise level β(t).

        Args:
            t: Timesteps of shape (B,).

        Returns:
            β(t) of shape (B,).
        """
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def _log_alpha(self, t: torch.Tensor) -> torch.Tensor:
        return -0.5 * (self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t**2)

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Mean coefficient α(t) = exp(−0.5 · ∫₀ᵗ β(s) ds).

        Args:
            t: Timesteps of shape (B,).

        Returns:
            α(t) of shape (B,).
        """
        return torch.exp(self._log_alpha(t))

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Noise standard deviation σ(t) = sqrt(1 − α(t)²).

        Args:
            t: Timesteps of shape (B,).

        Returns:
            σ(t) of shape (B,), clamped away from zero.
        """
        var = (1.0 - torch.exp(2.0 * self._log_alpha(t))).clamp(min=1e-10)
        return var.sqrt()

    def marginal_prob(
        self, x0: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean and std of the forward marginal p(x_t | x_0).

        Args:
            x0: Clean sample of any shape with batch dim first.
            t: Timesteps of shape (B,).

        Returns:
            Tuple of (mean, std), each broadcast to the shape of x0.
        """
        shape = (-1,) + (1,) * (x0.dim() - 1)
        a = self.alpha(t).view(shape)
        s = self.sigma(t).view(shape)
        return a * x0, s

    def reverse_drift(
        self, x: torch.Tensor, score: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Drift of the reverse SDE: f(x,t) − g(t)²·score.

        Args:
            x: Current sample.
            score: Score estimate at (x, t).
            t: Current timestep of shape (B,).

        Returns:
            Drift tensor of the same shape as x.
        """
        shape = (-1,) + (1,) * (x.dim() - 1)
        b = self.beta(t).view(shape)
        return -0.5 * b * x - b * score

    def diffusion_coeff(self, t: torch.Tensor) -> torch.Tensor:
        """Diffusion coefficient g(t) = sqrt(β(t)).

        Args:
            t: Timesteps of shape (B,).

        Returns:
            g(t) of shape (B,).
        """
        return self.beta(t).sqrt()
