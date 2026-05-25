"""Tests for src/models.py — ECGUNet and TextScoreNet."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import gdss_multimodal.models as models_module

ECGUNet = models_module.ECGUNet
TextScoreNet = models_module.TextScoreNet
FiLM = models_module.FiLM
SinusoidalTimestepEmbed = models_module.SinusoidalTimestepEmbed


# ---------------------------------------------------------------------------
# FiLM
# ---------------------------------------------------------------------------

class TestFiLM:
    def test_forward_shape_2d(self):
        film = FiLM(cond_dim=16, feature_dim=8)
        x    = torch.randn(4, 8)
        cond = torch.randn(4, 16)
        out  = film(x, cond)
        assert out.shape == x.shape

    def test_forward_shape_3d(self):
        # FiLM is applied after transpose(1,2) in _ResBlock1D, so the
        # 3D input arrives as (B, L, C), not (B, C, L).
        film = FiLM(cond_dim=16, feature_dim=8)
        x    = torch.randn(4, 32, 8)   # (B, L, C)
        cond = torch.randn(4, 16)
        out  = film(x, cond)
        assert out.shape == x.shape

    def test_identity_at_init(self):
        """Zero-init last layer → γ=1, β=0 at init → FiLM(x, c) ≈ x."""
        film = FiLM(cond_dim=8, feature_dim=4)
        x    = torch.randn(2, 4)
        cond = torch.zeros(2, 8)
        out  = film(x, cond)
        assert torch.allclose(out, x, atol=1e-5), "FiLM should be identity at init"

    def test_output_finite(self):
        film = FiLM(cond_dim=16, feature_dim=8)
        x    = torch.randn(4, 8)
        cond = torch.randn(4, 16)
        assert torch.isfinite(film(x, cond)).all()


# ---------------------------------------------------------------------------
# SinusoidalTimestepEmbed
# ---------------------------------------------------------------------------

class TestSinusoidalTimestepEmbed:
    def test_shape(self):
        emb = SinusoidalTimestepEmbed(dim=64)
        t   = torch.rand(8)
        out = emb(t)
        assert out.shape == (8, 64)

    def test_output_finite(self):
        emb = SinusoidalTimestepEmbed(dim=32)
        t   = torch.rand(4)
        assert torch.isfinite(emb(t)).all()

    def test_different_t_different_embedding(self):
        emb = SinusoidalTimestepEmbed(dim=32)
        t1  = torch.tensor([0.1])
        t2  = torch.tensor([0.9])
        assert not torch.allclose(emb(t1), emb(t2))

    def test_odd_dim_raises(self):
        with pytest.raises(AssertionError):
            SinusoidalTimestepEmbed(dim=33)


# ---------------------------------------------------------------------------
# ECGUNet (s_θ)
# ---------------------------------------------------------------------------

class TestECGUNet:
    def test_forward_shape(self, tiny_s_theta, tiny_ecg, tiny_text, tiny_t, lead_idx):
        out = tiny_s_theta(tiny_ecg, tiny_text, tiny_t, lead_idx)
        assert out.shape == tiny_ecg.shape

    def test_forward_no_lead_idx(self, tiny_s_theta, tiny_ecg, tiny_text, tiny_t):
        out = tiny_s_theta(tiny_ecg, tiny_text, tiny_t, None)
        assert out.shape == tiny_ecg.shape

    def test_forward_finite(self, tiny_s_theta, tiny_ecg, tiny_text, tiny_t, lead_idx):
        out = tiny_s_theta(tiny_ecg, tiny_text, tiny_t, lead_idx)
        assert torch.isfinite(out).all()

    def test_encode_shape(self, tiny_s_theta, tiny_ecg, tiny_t, lead_idx, tiny_cfg):
        t_emb = tiny_s_theta.t_embed(tiny_t)
        h     = tiny_s_theta.encode(tiny_ecg, t_emb, lead_idx)
        assert h.shape == (tiny_ecg.shape[0], tiny_cfg.ecg_score.bottleneck_ch)

    def test_encode_no_lead_idx(self, tiny_s_theta, tiny_ecg, tiny_t, tiny_cfg):
        t_emb = tiny_s_theta.t_embed(tiny_t)
        h     = tiny_s_theta.encode(tiny_ecg, t_emb, None)
        assert h.shape == (tiny_ecg.shape[0], tiny_cfg.ecg_score.bottleneck_ch)

    def test_encode_l2_normalised(self, tiny_s_theta, tiny_ecg, tiny_t, lead_idx):
        t_emb = tiny_s_theta.t_embed(tiny_t)
        h     = tiny_s_theta.encode(tiny_ecg, t_emb, lead_idx)
        norms = h.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_reconstruct_shape(self, tiny_s_theta, tiny_ecg, lead_idx):
        out = tiny_s_theta.reconstruct(tiny_ecg, lead_idx)
        assert out.shape == tiny_ecg.shape

    def test_reconstruct_no_lead_idx(self, tiny_s_theta, tiny_ecg):
        out = tiny_s_theta.reconstruct(tiny_ecg, None)
        assert out.shape == tiny_ecg.shape

    def test_reconstruct_is_finite(self, tiny_s_theta, tiny_ecg, lead_idx):
        out = tiny_s_theta.reconstruct(tiny_ecg, lead_idx)
        assert torch.isfinite(out).all()

    def test_no_r_peak_params(self, tiny_cfg):
        """ECGUNet must not accept r_peak_enc_dim (r_peak removed)."""
        with pytest.raises(TypeError):
            ECGUNet(
                text_dim=tiny_cfg.ecg_score.text_dim,
                n_leads=tiny_cfg.ecg_score.n_leads,
                seq_len=tiny_cfg.ecg_score.seq_len,
                timestep_dim=tiny_cfg.ecg_score.timestep_dim,
                channels=tiny_cfg.ecg_score.channels,
                bottleneck_ch=tiny_cfg.ecg_score.bottleneck_ch,
                lead_emb_dim=tiny_cfg.ecg_score.lead_emb_dim,
                r_peak_enc_dim=64,   # must raise
            )

    def test_no_lead_emb_mode(self, tiny_cfg, tiny_ecg, tiny_text, tiny_t):
        """lead_emb_dim=0 disables lead conditioning — model must still run."""
        model = ECGUNet(
            text_dim=tiny_cfg.ecg_score.text_dim,
            n_leads=tiny_cfg.ecg_score.n_leads,
            seq_len=tiny_cfg.ecg_score.seq_len,
            timestep_dim=tiny_cfg.ecg_score.timestep_dim,
            channels=tiny_cfg.ecg_score.channels,
            bottleneck_ch=tiny_cfg.ecg_score.bottleneck_ch,
            lead_emb_dim=0,
        ).eval()
        out = model(tiny_ecg, tiny_text, tiny_t, None)
        assert out.shape == tiny_ecg.shape

    def test_cfg_null_lead_idx(self, tiny_s_theta, tiny_ecg, tiny_text, tiny_t, batch_size):
        """Negative lead index sentinel (-1) should produce finite output."""
        null_lidx = torch.full((batch_size,), -1, dtype=torch.long)
        out = tiny_s_theta(tiny_ecg, tiny_text, tiny_t, null_lidx)
        assert torch.isfinite(out).all()

    def test_gradient_flows(self, tiny_cfg, tiny_ecg, tiny_text, tiny_t, lead_idx):
        model = ECGUNet(
            text_dim=tiny_cfg.ecg_score.text_dim,
            n_leads=tiny_cfg.ecg_score.n_leads,
            seq_len=tiny_cfg.ecg_score.seq_len,
            timestep_dim=tiny_cfg.ecg_score.timestep_dim,
            channels=tiny_cfg.ecg_score.channels,
            bottleneck_ch=tiny_cfg.ecg_score.bottleneck_ch,
            lead_emb_dim=tiny_cfg.ecg_score.lead_emb_dim,
        ).train()
        out = model(tiny_ecg, tiny_text, tiny_t, lead_idx)
        loss = out.pow(2).mean()
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No grad for {name}"


# ---------------------------------------------------------------------------
# TextScoreNet (s_φ)
# ---------------------------------------------------------------------------

class TestTextScoreNet:
    def test_forward_shape(self, tiny_s_phi, tiny_text, tiny_cfg, tiny_t, batch_size):
        ecg_rep = torch.randn(batch_size, tiny_cfg.text_score.moment_hidden_dim)
        out = tiny_s_phi(tiny_text, ecg_rep, tiny_t)
        assert out.shape == tiny_text.shape

    def test_forward_finite(self, tiny_s_phi, tiny_text, tiny_cfg, tiny_t, batch_size):
        ecg_rep = torch.randn(batch_size, tiny_cfg.text_score.moment_hidden_dim)
        out = tiny_s_phi(tiny_text, ecg_rep, tiny_t)
        assert torch.isfinite(out).all()

    def test_gradient_flows(self, tiny_cfg, tiny_text, tiny_t, batch_size):
        model = TextScoreNet(
            text_dim=tiny_cfg.text_score.text_dim,
            moment_hidden=tiny_cfg.text_score.moment_hidden_dim,
            timestep_dim=tiny_cfg.text_score.timestep_embed_dim,
            hidden_dim=tiny_cfg.text_score.hidden_dim,
            n_layers=tiny_cfg.text_score.n_layers,
        ).train()
        ecg_rep = torch.randn(batch_size, tiny_cfg.text_score.moment_hidden_dim)
        out  = model(tiny_text, ecg_rep, tiny_t)
        loss = out.pow(2).mean()
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No grad for {name}"

    def test_different_text_input_different_output(self, tiny_s_phi, tiny_cfg,
                                                    tiny_t, batch_size):
        """Different text_t inputs should produce different score estimates."""
        ecg_rep = torch.randn(batch_size, tiny_cfg.text_score.moment_hidden_dim)
        text1   = torch.zeros(batch_size, tiny_cfg.text_score.text_dim)
        text2   = torch.ones(batch_size,  tiny_cfg.text_score.text_dim)
        out1 = tiny_s_phi(text1, ecg_rep, tiny_t)
        out2 = tiny_s_phi(text2, ecg_rep, tiny_t)
        assert not torch.allclose(out1, out2)
