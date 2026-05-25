"""Tests for src/sde.py — VPSDE."""

from __future__ import annotations

import pytest
import torch

import gdss_multimodal.sde as sde_module

VPSDE = sde_module.VPSDE


@pytest.fixture()
def sde():
    return VPSDE(beta_min=0.1, beta_max=12.0, T=1.0, eps=1e-5)


@pytest.fixture()
def t_batch():
    return torch.linspace(1e-4, 1.0 - 1e-4, 8)


class TestVPSDE:
    # ── beta ─────────────────────────────────────────────────────────────────

    def test_beta_monotone(self, sde):
        t = torch.linspace(0.0, 1.0, 50)
        b = sde.beta(t)
        assert (b[1:] >= b[:-1]).all(), "beta must be non-decreasing"

    def test_beta_endpoint_min(self, sde):
        assert sde.beta(torch.tensor([0.0])).item() == pytest.approx(sde.beta_min)

    def test_beta_endpoint_max(self, sde):
        assert sde.beta(torch.tensor([1.0])).item() == pytest.approx(sde.beta_max)

    # ── alpha / sigma ─────────────────────────────────────────────────────────

    def test_alpha_in_01(self, sde, t_batch):
        a = sde.alpha(t_batch)
        assert ((a > 0) & (a <= 1)).all()

    def test_alpha_decreasing(self, sde):
        t = torch.linspace(0.0, 1.0, 20)
        a = sde.alpha(t)
        assert (a[1:] <= a[:-1]).all()

    def test_sigma_non_negative(self, sde, t_batch):
        s = sde.sigma(t_batch)
        assert (s >= 0).all()

    def test_alpha_sigma_pythagoras(self, sde, t_batch):
        """α² + σ² ≈ 1 for VP-SDE (unit-variance preservation)."""
        a = sde.alpha(t_batch)
        s = sde.sigma(t_batch)
        lhs = a ** 2 + s ** 2
        assert torch.allclose(lhs, torch.ones_like(lhs), atol=1e-5)

    def test_sigma_near_zero_at_t0(self, sde):
        t = torch.full((4,), sde.eps)
        s = sde.sigma(t)
        assert (s < 0.01).all()

    def test_sigma_near_one_at_t1(self, sde):
        t = torch.full((4,), sde.T - 1e-4)
        s = sde.sigma(t)
        assert (s > 0.9).all()

    # ── marginal_prob ─────────────────────────────────────────────────────────

    def test_marginal_prob_mean_shape_3d(self, sde, t_batch):
        x = torch.randn(8, 1, 64)
        mean, std = sde.marginal_prob(x, t_batch)
        assert mean.shape == x.shape

    def test_marginal_prob_std_shape_3d(self, sde, t_batch):
        x = torch.randn(8, 1, 64)
        mean, std = sde.marginal_prob(x, t_batch)
        # std is broadcast: (B, 1, 1) for 3D input
        assert std.shape[0] == 8

    def test_marginal_prob_mean_shape_2d(self, sde, t_batch):
        x = torch.randn(8, 32)
        mean, std = sde.marginal_prob(x, t_batch)
        assert mean.shape == x.shape

    def test_marginal_prob_mean_scales_with_alpha(self, sde, t_batch):
        x = torch.ones(8, 4)
        mean, std = sde.marginal_prob(x, t_batch)
        alpha = sde.alpha(t_batch).view(-1, 1)
        assert torch.allclose(mean, alpha * x, atol=1e-6)

    def test_marginal_prob_noisy_sample_finite(self, sde, t_batch):
        x = torch.randn(8, 1, 64)
        mean, std = sde.marginal_prob(x, t_batch)
        noisy = mean + std * torch.randn_like(x)
        assert torch.isfinite(noisy).all()

    # ── reverse_drift ─────────────────────────────────────────────────────────

    def test_reverse_drift_shape_3d(self, sde, t_batch):
        x     = torch.randn(8, 1, 64)
        score = torch.randn_like(x)
        drift = sde.reverse_drift(x, score, t_batch)
        assert drift.shape == x.shape

    def test_reverse_drift_shape_2d(self, sde, t_batch):
        x     = torch.randn(8, 32)
        score = torch.randn_like(x)
        drift = sde.reverse_drift(x, score, t_batch)
        assert drift.shape == x.shape

    def test_reverse_drift_is_finite(self, sde, t_batch):
        x     = torch.randn(8, 1, 64)
        score = torch.randn_like(x)
        drift = sde.reverse_drift(x, score, t_batch)
        assert torch.isfinite(drift).all()

    def test_reverse_drift_zero_score_equals_forward_drift(self, sde, t_batch):
        """With zero score, reverse drift = −0.5·β·x (forward drift)."""
        x     = torch.randn(8, 1, 64)
        score = torch.zeros_like(x)
        drift = sde.reverse_drift(x, score, t_batch)
        b     = sde.beta(t_batch).view(-1, 1, 1)
        expected = -0.5 * b * x
        assert torch.allclose(drift, expected, atol=1e-6)

    # ── diffusion_coeff ───────────────────────────────────────────────────────

    def test_diffusion_coeff_positive(self, sde, t_batch):
        g = sde.diffusion_coeff(t_batch)
        assert (g > 0).all()

    def test_diffusion_coeff_shape(self, sde, t_batch):
        g = sde.diffusion_coeff(t_batch)
        assert g.shape == t_batch.shape

    def test_diffusion_coeff_equals_sqrt_beta(self, sde, t_batch):
        g = sde.diffusion_coeff(t_batch)
        expected = sde.beta(t_batch).sqrt()
        assert torch.allclose(g, expected, atol=1e-6)

    # ── Custom schedule ───────────────────────────────────────────────────────

    def test_custom_beta_range(self):
        sde = VPSDE(beta_min=1.0, beta_max=5.0)
        assert sde.beta_min == 1.0
        assert sde.beta_max == 5.0

    def test_default_eps(self):
        sde = VPSDE()
        assert sde.eps == 1e-5
