"""Tests for src/sample.py — load_models and generate."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy
import pytest
import torch

import gdss_multimodal.models as models_module
import gdss_multimodal.sample as sample_module
import gdss_multimodal.sde as sde_module

generate = sample_module.generate
load_models = sample_module.load_models
VPSDE = sde_module.VPSDE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def saved_checkpoint(tiny_s_theta, tiny_s_phi, tmp_path):
    """Save a minimal checkpoint and return its path."""
    path = tmp_path / "model.pt"
    torch.save({
        "s_theta":     tiny_s_theta.state_dict(),
        "s_phi":       tiny_s_phi.state_dict(),
        "s_theta_ema": tiny_s_theta.state_dict(),
        "s_phi_ema":   tiny_s_phi.state_dict(),
    }, path)
    return path


# ---------------------------------------------------------------------------
# load_models
# ---------------------------------------------------------------------------

class TestLoadModels:
    def test_returns_two_models(self, saved_checkpoint, tiny_cfg, device):
        s_theta, s_phi = load_models(saved_checkpoint, tiny_cfg, device)
        assert s_theta is not None
        assert s_phi is not None

    def test_models_in_eval_mode(self, saved_checkpoint, tiny_cfg, device):
        s_theta, s_phi = load_models(saved_checkpoint, tiny_cfg, device)
        assert not s_theta.training
        assert not s_phi.training

    def test_prefers_ema_weights(self, tiny_s_theta, tiny_s_phi, tmp_path, tiny_cfg, device):
        """load_models should prefer s_theta_ema over s_theta when both present."""
        # Modify EMA weights so they differ from base
        ema_sd = {k: v + 1.0 for k, v in tiny_s_theta.state_dict().items()}
        path = tmp_path / "ema.pt"
        torch.save({
            "s_theta":     tiny_s_theta.state_dict(),
            "s_phi":       tiny_s_phi.state_dict(),
            "s_theta_ema": ema_sd,
            "s_phi_ema":   tiny_s_phi.state_dict(),
        }, path)
        s_theta_loaded, _ = load_models(path, tiny_cfg, device)
        # The loaded weights should match ema_sd (value + 1), not the base
        loaded_sd = s_theta_loaded.state_dict()
        orig_sd   = tiny_s_theta.state_dict()
        # At least one parameter should differ
        any_differ = any(
            not torch.allclose(loaded_sd[k], orig_sd[k]) for k in orig_sd
        )
        assert any_differ, "EMA weights should differ from base weights"

    def test_fallback_to_s_theta_when_no_ema(self, tiny_s_theta, tiny_s_phi,
                                              tmp_path, tiny_cfg, device):
        """If no EMA key, load_models falls back to s_theta key."""
        path = tmp_path / "no_ema.pt"
        torch.save({
            "s_theta": tiny_s_theta.state_dict(),
            "s_phi":   tiny_s_phi.state_dict(),
        }, path)
        s_theta_loaded, _ = load_models(path, tiny_cfg, device)
        loaded_sd = s_theta_loaded.state_dict()
        orig_sd   = tiny_s_theta.state_dict()
        for k in orig_sd:
            assert torch.allclose(loaded_sd[k], orig_sd[k])


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------

class TestGenerate:
    @pytest.fixture()
    def vpsde(self):
        return VPSDE(beta_min=0.1, beta_max=12.0)

    @pytest.mark.parametrize("sampler_name", ["s4", "pc", "em"])
    def test_output_shapes(self, tiny_s_theta, tiny_s_phi, vpsde, tiny_cfg,
                           device, sampler_name):
        n_samples = 2
        ecgs, texts = generate(
            tiny_s_theta, tiny_s_phi, vpsde, sampler_name,
            n_samples=n_samples, batch_size=2, n_steps=2,
            snr=0.16, device=device, cfg=tiny_cfg,
            lead_idx=0, cfg_scale=0.0,
        )
        assert ecgs.shape == (n_samples, tiny_cfg.ecg_score.n_leads,
                               tiny_cfg.ecg_score.seq_len)
        assert texts.shape == (n_samples, tiny_cfg.ecg_score.text_dim)

    def test_batching_works(self, tiny_s_theta, tiny_s_phi, vpsde, tiny_cfg, device):
        """n_samples > batch_size should produce correct total count."""
        n_samples, batch_size = 5, 2
        ecgs, texts = generate(
            tiny_s_theta, tiny_s_phi, vpsde, "em",
            n_samples=n_samples, batch_size=batch_size, n_steps=2,
            snr=0.16, device=device, cfg=tiny_cfg,
            lead_idx=1, cfg_scale=0.0,
        )
        assert ecgs.shape[0] == n_samples
        assert texts.shape[0] == n_samples

    def test_random_lead_idx(self, tiny_s_theta, tiny_s_phi, vpsde, tiny_cfg, device):
        """lead_idx=-1 triggers random lead selection — should not error."""
        ecgs, texts = generate(
            tiny_s_theta, tiny_s_phi, vpsde, "em",
            n_samples=2, batch_size=2, n_steps=2,
            snr=0.16, device=device, cfg=tiny_cfg,
            lead_idx=-1, cfg_scale=0.0,
        )
        assert ecgs.shape[0] == 2

    def test_cfg_scale_nonzero(self, tiny_s_theta, tiny_s_phi, vpsde, tiny_cfg, device):
        """cfg_scale > 0 activates CFG — should produce finite output."""
        ecgs, texts = generate(
            tiny_s_theta, tiny_s_phi, vpsde, "em",
            n_samples=2, batch_size=2, n_steps=2,
            snr=0.16, device=device, cfg=tiny_cfg,
            lead_idx=0, cfg_scale=1.5,
        )
        assert numpy.isfinite(ecgs).all()
        assert numpy.isfinite(texts).all()

    def test_no_lead_emb_mode(self, tiny_cfg, device, vpsde):
        """When lead_emb_dim=0, lead conditioning is disabled."""
        ECGUNet = models_module.ECGUNet
        TextScoreNet = models_module.TextScoreNet

        cfg2 = tiny_cfg
        cfg2.ecg_score.lead_emb_dim = 0

        s_theta = ECGUNet(
            text_dim=cfg2.ecg_score.text_dim,
            n_leads=cfg2.ecg_score.n_leads,
            seq_len=cfg2.ecg_score.seq_len,
            timestep_dim=cfg2.ecg_score.timestep_dim,
            channels=cfg2.ecg_score.channels,
            bottleneck_ch=cfg2.ecg_score.bottleneck_ch,
            lead_emb_dim=0,
        ).eval()
        s_phi = TextScoreNet(
            text_dim=cfg2.text_score.text_dim,
            moment_hidden=cfg2.text_score.moment_hidden_dim,
            timestep_dim=cfg2.text_score.timestep_embed_dim,
            hidden_dim=cfg2.text_score.hidden_dim,
            n_layers=cfg2.text_score.n_layers,
        ).eval()

        ecgs, texts = generate(
            s_theta, s_phi, vpsde, "em",
            n_samples=2, batch_size=2, n_steps=2,
            snr=0.16, device=device, cfg=cfg2,
            lead_idx=0, cfg_scale=0.0,
        )
        assert ecgs.shape[0] == 2
