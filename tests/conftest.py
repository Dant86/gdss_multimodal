"""Shared pytest fixtures for the gdss_multimodal test suite."""

from __future__ import annotations

import numpy
import pytest
import torch

import gdss_multimodal.config as config_module
import gdss_multimodal.models as models_module
import gdss_multimodal.sde as sde_module


# ---------------------------------------------------------------------------
# Tiny model dimensions — fast enough for CPU tests
# ---------------------------------------------------------------------------

@pytest.fixture()
def tiny_cfg():
    """A Config with very small model dimensions for fast CPU testing."""
    cfg = config_module.Config()
    cfg.ecg_score.channels      = (8, 16)
    cfg.ecg_score.bottleneck_ch = 32
    cfg.ecg_score.timestep_dim  = 16
    cfg.ecg_score.text_dim      = 32
    cfg.ecg_score.n_leads       = 1
    cfg.ecg_score.seq_len       = 64
    cfg.ecg_score.lead_emb_dim  = 8
    cfg.text_score.text_dim          = 32
    cfg.text_score.moment_hidden_dim = 32
    cfg.text_score.timestep_embed_dim = 16
    cfg.text_score.hidden_dim        = 32
    cfg.text_score.n_layers          = 2
    return cfg


@pytest.fixture()
def device():
    return torch.device("cpu")


@pytest.fixture()
def batch_size():
    return 4


@pytest.fixture()
def tiny_ecg(tiny_cfg, batch_size):
    """Random (B, n_leads, seq_len) ECG tensor."""
    return torch.randn(batch_size, tiny_cfg.ecg_score.n_leads, tiny_cfg.ecg_score.seq_len)


@pytest.fixture()
def tiny_text(tiny_cfg, batch_size):
    """Random (B, text_dim) text embedding tensor."""
    return torch.randn(batch_size, tiny_cfg.ecg_score.text_dim)


@pytest.fixture()
def tiny_t(batch_size):
    """Random timesteps in (0, 1)."""
    return torch.rand(batch_size).clamp(1e-4, 1 - 1e-4)


@pytest.fixture()
def lead_idx(batch_size):
    """Random lead indices 0–11."""
    return torch.randint(0, 12, (batch_size,))


@pytest.fixture()
def vpsde():
    return sde_module.VPSDE(beta_min=0.1, beta_max=12.0, T=1.0, eps=1e-5)


@pytest.fixture()
def tiny_s_theta(tiny_cfg, device):
    return models_module.ECGUNet(
        text_dim=tiny_cfg.ecg_score.text_dim,
        n_leads=tiny_cfg.ecg_score.n_leads,
        seq_len=tiny_cfg.ecg_score.seq_len,
        timestep_dim=tiny_cfg.ecg_score.timestep_dim,
        channels=tiny_cfg.ecg_score.channels,
        bottleneck_ch=tiny_cfg.ecg_score.bottleneck_ch,
        lead_emb_dim=tiny_cfg.ecg_score.lead_emb_dim,
    ).to(device).eval()


@pytest.fixture()
def tiny_s_phi(tiny_cfg, device):
    return models_module.TextScoreNet(
        text_dim=tiny_cfg.text_score.text_dim,
        moment_hidden=tiny_cfg.text_score.moment_hidden_dim,
        timestep_dim=tiny_cfg.text_score.timestep_embed_dim,
        hidden_dim=tiny_cfg.text_score.hidden_dim,
        n_layers=tiny_cfg.text_score.n_layers,
    ).to(device).eval()
