"""Unit tests for gdss_multimodal.

Run with:  pytest tests.py -v
Requires only torch, numpy, scipy — no Modal or momentfm needed.
"""

from __future__ import annotations

import numpy
import pytest
import torch
import torch.nn as nn

import sde as sde_module
import solvers as solvers_module


B = 4
TEXT_DIM = 768
BOTTLENECK_DIM = 256
N_LEADS = 1
SEQ_LEN = 64  # small for fast tests


@pytest.fixture
def vpsde():
    return sde_module.VPSDE(beta_min=0.1, beta_max=20.0, T=1.0, eps=1e-5)


class TestVPSDE:
    def test_marginal_prob_at_zero(self, vpsde):
        x0 = torch.randn(B, N_LEADS, SEQ_LEN)
        t = torch.full((B,), vpsde.eps)
        mean, std = vpsde.marginal_prob(x0, t)
        # mean must broadcast to x0 shape; std is (B, 1, 1) by design
        assert mean.shape == x0.shape
        assert (mean + std).shape == x0.shape  # std broadcasts correctly
        assert torch.allclose(mean, x0, atol=1e-2)
        assert (std < 0.01).all()

    def test_marginal_prob_at_T(self, vpsde):
        x0 = torch.randn(B, TEXT_DIM)
        t = torch.full((B,), vpsde.T)
        mean, std = vpsde.marginal_prob(x0, t)
        # std is (B, 1) by design — check it broadcasts and is near 1
        assert mean.shape == x0.shape
        assert (mean + std).shape == x0.shape
        assert (std > 0.9).all()

    def test_alpha_monotone_decreasing(self, vpsde):
        ts = torch.linspace(vpsde.eps, vpsde.T, 50)
        alphas = vpsde.alpha(ts)
        assert (alphas[1:] <= alphas[:-1]).all()

    def test_sigma_monotone_increasing(self, vpsde):
        ts = torch.linspace(vpsde.eps, vpsde.T, 50)
        sigmas = vpsde.sigma(ts)
        assert (sigmas[1:] >= sigmas[:-1]).all()

    def test_reverse_drift_shape(self, vpsde):
        x = torch.randn(B, TEXT_DIM)
        score = torch.randn(B, TEXT_DIM)
        t = torch.rand(B)
        drift = vpsde.reverse_drift(x, score, t)
        assert drift.shape == x.shape

    def test_diffusion_coeff_positive(self, vpsde):
        t = torch.rand(B)
        assert (vpsde.diffusion_coeff(t) > 0).all()

    def test_broadcast_over_leading_dims(self, vpsde):
        x0 = torch.randn(B, N_LEADS, SEQ_LEN)
        t = torch.rand(B)
        mean, std = vpsde.marginal_prob(x0, t)
        # std is (B, 1, 1) — check it broadcasts to x0 shape
        assert mean.shape == (B, N_LEADS, SEQ_LEN)
        assert (mean + std).shape == (B, N_LEADS, SEQ_LEN)


def _zero_score(x, x_cross, t):
    return torch.zeros_like(x)


def _gaussian_score(x, x_cross, t):
    """Score of a standard Gaussian: -x. Keeps eps = snr² (constant, small)."""
    return -x


class TestSolvers:
    @pytest.fixture
    def setup(self, vpsde):
        m1_T = torch.randn(B, N_LEADS, SEQ_LEN)
        m2_T = torch.randn(B, TEXT_DIM)
        return m1_T, m2_T, vpsde

    @pytest.mark.parametrize("name", ["s4", "pc", "em"])
    def test_output_shape(self, name, setup):
        m1_T, m2_T, vpsde = setup
        sampler = solvers_module.SAMPLERS[name]
        m1, m2 = sampler(m1_T, m2_T, _gaussian_score, _gaussian_score, vpsde, n_steps=3)
        assert m1.shape == m1_T.shape
        assert m2.shape == m2_T.shape

    @pytest.mark.parametrize("name", ["s4", "pc", "em"])
    def test_output_finite(self, name, setup):
        """Gaussian score (-x) keeps adaptive Langevin eps = snr² (constant, small)."""
        m1_T, m2_T, vpsde = setup
        sampler = solvers_module.SAMPLERS[name]
        m1, m2 = sampler(m1_T, m2_T, _gaussian_score, _gaussian_score, vpsde, n_steps=3)
        assert torch.isfinite(m1).all()
        assert torch.isfinite(m2).all()

    def test_zero_score_stays_finite(self, setup):
        """Langevin eps is clamped at 1.0, so zero score degrades to a random walk (finite)."""
        m1_T, m2_T, vpsde = setup
        m1, m2 = solvers_module.s4_sampler(
            m1_T, m2_T, _zero_score, _zero_score, vpsde, n_steps=3
        )
        assert torch.isfinite(m1).all()

    def test_samplers_dict_complete(self):
        assert set(solvers_module.SAMPLERS.keys()) == {"s4", "pc", "em"}


class TestECGScoreNet:
    @pytest.fixture
    def model(self):
        import models as models_module

        return models_module.ECGUNet(
            text_dim=TEXT_DIM,
            n_leads=N_LEADS,
            seq_len=SEQ_LEN,
            timestep_dim=32,
            channels=(8, 16),
            bottleneck_ch=32,
        )

    def test_forward_shape(self, model):
        ecg_t = torch.randn(B, N_LEADS, SEQ_LEN)
        text_t = torch.randn(B, TEXT_DIM)
        t = torch.rand(B)
        out = model(ecg_t, text_t, t)
        assert out.shape == (B, N_LEADS, SEQ_LEN)

    def test_forward_finite(self, model):
        ecg_t = torch.randn(B, N_LEADS, SEQ_LEN)
        text_t = torch.randn(B, TEXT_DIM)
        t = torch.rand(B)
        out = model(ecg_t, text_t, t)
        assert torch.isfinite(out).all()

    def test_encode_shape(self, model):
        ecg_t = torch.randn(B, N_LEADS, SEQ_LEN)
        t_emb = model.t_embed(torch.rand(B))
        rep = model.encode(ecg_t, t_emb)
        assert rep.shape == (B, model.bottleneck_ch)


class TestTextScoreNet:
    @pytest.fixture
    def model(self):
        import models as models_module
        return models_module.TextScoreNet(
            text_dim=TEXT_DIM,
            moment_hidden=BOTTLENECK_DIM,
            timestep_dim=256,
            hidden_dim=128,
            n_layers=2,
        )

    def test_forward_shape(self, model):
        text_t = torch.randn(B, TEXT_DIM)
        ecg_rep = torch.randn(B, BOTTLENECK_DIM)
        t = torch.rand(B)
        out = model(text_t, ecg_rep, t)
        assert out.shape == (B, TEXT_DIM)

    def test_forward_finite(self, model):
        text_t = torch.randn(B, TEXT_DIM)
        ecg_rep = torch.randn(B, BOTTLENECK_DIM)
        t = torch.rand(B)
        assert torch.isfinite(model(text_t, ecg_rep, t)).all()


class TestFiLM:
    def test_identity_init(self):
        import models as models_module
        film = models_module.FiLM(cond_dim=32, feature_dim=64, hidden_dim=128)
        x = torch.randn(B, 64)
        cond = torch.randn(B, 32)
        out = film(x, cond)
        # at init γ=1, β=0 so output should equal input
        assert torch.allclose(out, x, atol=1e-5)

    def test_output_shape(self):
        import models as models_module
        film = models_module.FiLM(cond_dim=32, feature_dim=64)
        x = torch.randn(B, 10, 64)
        cond = torch.randn(B, 32)
        assert film(x, cond).shape == (B, 10, 64)


class TestFID:
    def test_identical_distributions_near_zero(self):
        """FID between identical distributions should be close to zero."""
        from evaluate import ecg_fid

        class MockDevice:
            type = "cpu"

        rng = numpy.random.default_rng(0)
        feats = rng.standard_normal((200, 32)).astype(numpy.float32)

        # patch extract_moment_features to return known features
        import evaluate
        original = evaluate.extract_moment_features
        evaluate.extract_moment_features = lambda ecgs, device, **kw: feats
        try:
            fid = ecg_fid(feats, feats, "cpu")
        finally:
            evaluate.extract_moment_features = original

        assert fid < 1.0

    def test_different_distributions_positive(self):
        """FID between shifted distributions should be clearly positive."""
        from evaluate import ecg_fid

        rng = numpy.random.default_rng(1)
        real = rng.standard_normal((200, 32)).astype(numpy.float32)
        gen = (rng.standard_normal((200, 32)) + 5).astype(numpy.float32)

        import evaluate
        calls = iter([real, gen])
        evaluate.extract_moment_features = lambda ecgs, device, **kw: next(calls)
        try:
            fid = ecg_fid(real, gen, "cpu")
        finally:
            del evaluate.extract_moment_features

        assert fid > 1.0

    def test_text_cosine_sim_identical(self):
        from evaluate import text_cosine_sim
        vecs = numpy.random.randn(50, TEXT_DIM).astype(numpy.float32)
        sim = text_cosine_sim(vecs, vecs)
        assert abs(sim - 1.0) < 1e-4

    def test_text_cosine_sim_orthogonal(self):
        from evaluate import text_cosine_sim
        rng = numpy.random.default_rng(42)
        # Two orthogonal unit vectors
        gen = numpy.zeros((1, 2), dtype=numpy.float32)
        gen[0, 0] = 1.0
        real = numpy.zeros((1, 2), dtype=numpy.float32)
        real[0, 1] = 1.0
        sim = text_cosine_sim(gen, real)
        assert abs(sim) < 1e-4
