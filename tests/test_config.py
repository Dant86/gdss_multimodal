"""Tests for src/config.py — Config dataclass and YAML I/O."""

from __future__ import annotations

import io
import textwrap
from pathlib import Path

import pytest

import gdss_multimodal.config as config_module
from gdss_multimodal.config import _to_dict, _from_dict

Config = config_module.Config
ECGScoreConfig = config_module.ECGScoreConfig
SDEConfig = config_module.SDEConfig
TextScoreConfig = config_module.TextScoreConfig
TrainConfig = config_module.TrainConfig
PretrainConfig = config_module.PretrainConfig
EvalConfig = config_module.EvalConfig
SamplerConfig = config_module.SamplerConfig


# ---------------------------------------------------------------------------
# Default construction
# ---------------------------------------------------------------------------

class TestConfigDefaults:
    def test_default_construction(self):
        cfg = Config()
        assert isinstance(cfg.sde, SDEConfig)
        assert isinstance(cfg.ecg_score, ECGScoreConfig)
        assert isinstance(cfg.text_score, TextScoreConfig)
        assert isinstance(cfg.train, TrainConfig)
        assert isinstance(cfg.pretrain, PretrainConfig)
        assert isinstance(cfg.eval, EvalConfig)

    def test_sde_defaults(self):
        cfg = Config()
        assert cfg.sde.beta_min == 0.1
        assert cfg.sde.beta_max == 12.0
        assert cfg.sde.T == 1.0
        assert cfg.sde.eps == 1e-5

    def test_ecg_score_defaults(self):
        cfg = Config()
        assert cfg.ecg_score.n_leads == 1
        assert cfg.ecg_score.seq_len == 1000
        assert cfg.ecg_score.lead_emb_dim == 64
        # r_peak fields must NOT exist
        assert not hasattr(cfg.ecg_score, "r_peak_enc_dim")

    def test_train_defaults(self):
        cfg = Config()
        assert cfg.train.batch_size == 256
        assert cfg.train.max_steps == 100_000
        # r_peak_drop_prob must NOT exist
        assert not hasattr(cfg.train, "r_peak_drop_prob")

    def test_channels_is_tuple(self):
        cfg = Config()
        assert isinstance(cfg.ecg_score.channels, tuple)


# ---------------------------------------------------------------------------
# YAML round-trip
# ---------------------------------------------------------------------------

class TestYAML:
    def test_to_yaml_creates_file(self, tmp_path):
        cfg = Config()
        out = tmp_path / "cfg.yaml"
        cfg.to_yaml(out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_from_yaml_round_trip(self, tmp_path):
        cfg = Config()
        cfg.sde.beta_max = 20.0
        cfg.train.batch_size = 512
        cfg.ecg_score.channels = (16, 32)
        out = tmp_path / "cfg.yaml"
        cfg.to_yaml(out)

        cfg2 = Config.from_yaml(out)
        assert cfg2.sde.beta_max == 20.0
        assert cfg2.train.batch_size == 512
        assert cfg2.ecg_score.channels == (16, 32)

    def test_from_yaml_uses_defaults_for_missing_keys(self, tmp_path):
        yaml_text = textwrap.dedent("""\
            sde:
              beta_min: 0.5
        """)
        out = tmp_path / "partial.yaml"
        out.write_text(yaml_text)
        cfg = Config.from_yaml(out)
        assert cfg.sde.beta_min == 0.5
        # Untouched field keeps default
        assert cfg.sde.beta_max == 12.0

    def test_from_yaml_empty_file(self, tmp_path):
        out = tmp_path / "empty.yaml"
        out.write_text("")
        cfg = Config.from_yaml(out)
        # Should fall back to all defaults
        assert cfg.sde.beta_min == 0.1

    def test_channels_roundtrip_as_tuple(self, tmp_path):
        cfg = Config()
        cfg.ecg_score.channels = (64, 128, 256)
        out = tmp_path / "cfg.yaml"
        cfg.to_yaml(out)
        cfg2 = Config.from_yaml(out)
        assert cfg2.ecg_score.channels == (64, 128, 256)
        assert isinstance(cfg2.ecg_score.channels, tuple)


# ---------------------------------------------------------------------------
# Override
# ---------------------------------------------------------------------------

class TestOverride:
    def test_override_float(self):
        cfg = Config()
        cfg.override({"train.lr": "5e-4"})
        assert cfg.train.lr == pytest.approx(5e-4)

    def test_override_int(self):
        cfg = Config()
        cfg.override({"train.batch_size": "128"})
        assert cfg.train.batch_size == 128

    def test_override_bool_false(self):
        cfg = Config()
        cfg.override({"train.likelihood_weighting": "false"})
        assert cfg.train.likelihood_weighting is False

    def test_override_bool_true(self):
        cfg = Config()
        cfg.override({"train.likelihood_weighting": "true"})
        assert cfg.train.likelihood_weighting is True

    def test_override_tuple(self):
        cfg = Config()
        cfg.override({"ecg_score.channels": "16,32,64"})
        assert cfg.ecg_score.channels == (16, 32, 64)

    def test_override_nested(self):
        cfg = Config()
        cfg.override({"sde.beta_min": "0.2"})
        assert cfg.sde.beta_min == pytest.approx(0.2)

    def test_override_multiple_keys(self):
        cfg = Config()
        cfg.override({"train.lr": "1e-3", "train.max_steps": "50000"})
        assert cfg.train.lr == pytest.approx(1e-3)
        assert cfg.train.max_steps == 50_000

    def test_override_unknown_key_raises(self):
        cfg = Config()
        with pytest.raises(AttributeError):
            cfg.override({"train.nonexistent_key": "1"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_to_dict_is_plain(self):
        cfg = Config()
        d = _to_dict(cfg)
        assert isinstance(d, dict)
        assert "sde" in d
        assert isinstance(d["sde"], dict)

    def test_to_dict_tuple_becomes_list(self):
        cfg = Config()
        d = _to_dict(cfg)
        assert isinstance(d["ecg_score"]["channels"], list)

    def test_from_dict_reconstructs(self):
        cfg = Config()
        d = _to_dict(cfg)
        cfg2 = _from_dict(Config, d)
        assert cfg2.sde.beta_min == cfg.sde.beta_min
        assert cfg2.ecg_score.channels == cfg.ecg_score.channels

    def test_sampler_config_nested_in_eval(self):
        cfg = Config()
        assert isinstance(cfg.eval.sampler, SamplerConfig)
        d = _to_dict(cfg)
        assert "sampler" in d["eval"]
        cfg2 = _from_dict(Config, d)
        assert isinstance(cfg2.eval.sampler, SamplerConfig)
        assert cfg2.eval.sampler.n_steps == cfg.eval.sampler.n_steps
