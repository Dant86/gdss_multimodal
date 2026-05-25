"""Tests for src/solvers.py — s4, pc, em samplers."""

from __future__ import annotations

import pytest
import torch

import gdss_multimodal.sde as sde_module
import gdss_multimodal.solvers as solvers_module

VPSDE = sde_module.VPSDE
SAMPLERS = solvers_module.SAMPLERS
_clip_score = solvers_module._clip_score
_langevin_step = solvers_module._langevin_step
_reverse_sde_step = solvers_module._reverse_sde_step
_s4_step = solvers_module._s4_step
s4_sampler = solvers_module.s4_sampler
pc_sampler = solvers_module.pc_sampler
em_sampler = solvers_module.em_sampler


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sde():
    return VPSDE(beta_min=0.1, beta_max=12.0, T=1.0, eps=1e-5)


@pytest.fixture()
def B():
    return 3


@pytest.fixture()
def m1(B):
    return torch.randn(B, 1, 16)


@pytest.fixture()
def m2(B):
    return torch.randn(B, 8)


@pytest.fixture()
def t_vec(B):
    return torch.full((B,), 0.5)


# Dummy score networks: return zero score (simplest stable behaviour)
def zero_score(x, x_cross, t):
    return torch.zeros_like(x)


# ---------------------------------------------------------------------------
# _clip_score
# ---------------------------------------------------------------------------

class TestClipScore:
    def test_finite_input_unchanged_when_small(self):
        x     = torch.randn(4, 8)
        score = x * 0.1   # small; well within limit
        out   = _clip_score(score, x)
        # Should not be clipped
        assert torch.allclose(out, score, atol=1e-5)

    def test_large_score_is_clipped(self):
        x     = torch.ones(4, 8)
        score = x * 1000.0   # way beyond max_ratio default of 5
        out   = _clip_score(score, x)
        B     = x.shape[0]
        out_norm = out.view(B, -1).norm(dim=1)
        x_norm   = x.view(B, -1).norm(dim=1)
        assert (out_norm <= x_norm * 5.0 + 1e-4).all()

    def test_nan_input_is_zeroed(self):
        x     = torch.ones(4, 8)
        score = torch.full_like(x, float("nan"))
        out   = _clip_score(score, x)
        assert torch.isfinite(out).all()

    def test_inf_input_is_zeroed(self):
        x     = torch.ones(4, 8)
        score = torch.full_like(x, float("inf"))
        out   = _clip_score(score, x)
        assert torch.isfinite(out).all()

    def test_output_shape_preserved(self):
        x     = torch.randn(4, 1, 16)
        score = torch.randn_like(x)
        out   = _clip_score(score, x)
        assert out.shape == score.shape


# ---------------------------------------------------------------------------
# _reverse_sde_step
# ---------------------------------------------------------------------------

class TestReverseSdeStep:
    def test_output_shape_3d(self, sde, m1, t_vec):
        score = torch.zeros_like(m1)
        out   = _reverse_sde_step(m1, score, t_vec, dt=0.01, vpsde=sde)
        assert out.shape == m1.shape

    def test_output_shape_2d(self, sde, m2, t_vec):
        score = torch.zeros_like(m2)
        out   = _reverse_sde_step(m2, score, t_vec, dt=0.01, vpsde=sde)
        assert out.shape == m2.shape

    def test_output_is_finite(self, sde, m1, t_vec):
        score = torch.zeros_like(m1)
        out   = _reverse_sde_step(m1, score, t_vec, dt=0.01, vpsde=sde)
        assert torch.isfinite(out).all()

    def test_zero_score_changes_sample(self, sde, m1, t_vec):
        """With zero score the sample still changes due to noise injection."""
        torch.manual_seed(0)
        score = torch.zeros_like(m1)
        out   = _reverse_sde_step(m1, score, t_vec, dt=0.01, vpsde=sde)
        # The sample should change (due to drift and noise)
        assert not torch.allclose(out, m1)


# ---------------------------------------------------------------------------
# _langevin_step
# ---------------------------------------------------------------------------

class TestLangevinStep:
    def test_output_shape(self, m1, m2, t_vec):
        out = _langevin_step(m1, m2, t_vec, zero_score, snr=0.16)
        assert out.shape == m1.shape

    def test_output_is_finite(self, m1, m2, t_vec):
        out = _langevin_step(m1, m2, t_vec, zero_score, snr=0.16)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# _s4_step
# ---------------------------------------------------------------------------

class TestS4Step:
    def test_output_shapes(self, sde, m1, m2, t_vec):
        out_m1, out_m2 = _s4_step(
            m1, m2, t_vec, dt=0.01,
            s_theta=zero_score, s_phi=zero_score,
            vpsde=sde, snr=0.16, n_corrector=1,
        )
        assert out_m1.shape == m1.shape
        assert out_m2.shape == m2.shape

    def test_outputs_are_finite(self, sde, m1, m2, t_vec):
        out_m1, out_m2 = _s4_step(
            m1, m2, t_vec, dt=0.01,
            s_theta=zero_score, s_phi=zero_score,
            vpsde=sde, snr=0.16, n_corrector=1,
        )
        assert torch.isfinite(out_m1).all()
        assert torch.isfinite(out_m2).all()

    def test_zero_corrector_steps(self, sde, m1, m2, t_vec):
        out_m1, out_m2 = _s4_step(
            m1, m2, t_vec, dt=0.01,
            s_theta=zero_score, s_phi=zero_score,
            vpsde=sde, snr=0.16, n_corrector=0,
        )
        assert out_m1.shape == m1.shape
        assert out_m2.shape == m2.shape


# ---------------------------------------------------------------------------
# Full samplers — smoke tests (n_steps=3 for speed)
# ---------------------------------------------------------------------------

class TestSamplers:
    @pytest.mark.parametrize("sampler_fn", [s4_sampler, pc_sampler, em_sampler])
    def test_output_shapes(self, sde, m1, m2, sampler_fn):
        out_m1, out_m2 = sampler_fn(
            m1, m2, zero_score, zero_score, sde, n_steps=3, snr=0.16
        )
        assert out_m1.shape == m1.shape
        assert out_m2.shape == m2.shape

    @pytest.mark.parametrize("sampler_fn", [s4_sampler, pc_sampler, em_sampler])
    def test_outputs_finite(self, sde, m1, m2, sampler_fn):
        out_m1, out_m2 = sampler_fn(
            m1, m2, zero_score, zero_score, sde, n_steps=3, snr=0.16
        )
        assert torch.isfinite(out_m1).all()
        assert torch.isfinite(out_m2).all()

    def test_s4_log_steps_flag(self, sde, m1, m2, capsys):
        s4_sampler(m1, m2, zero_score, zero_score, sde, n_steps=5, log_steps=True)
        # Should print diagnostics without error
        captured = capsys.readouterr()
        assert "[s4 step" in captured.out

    def test_samplers_dict_keys(self):
        assert set(SAMPLERS.keys()) == {"s4", "pc", "em"}

    def test_samplers_dict_callable(self):
        for fn in SAMPLERS.values():
            assert callable(fn)

    @pytest.mark.parametrize("name", ["s4", "pc", "em"])
    def test_samplers_dict_produces_correct_shape(self, sde, m1, m2, name):
        fn = SAMPLERS[name]
        out_m1, out_m2 = fn(m1, m2, zero_score, zero_score, sde, n_steps=2)
        assert out_m1.shape == m1.shape
        assert out_m2.shape == m2.shape
