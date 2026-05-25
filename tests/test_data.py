"""Tests for src/data.py — record loading, splits, stats, and datasets.

Heavy I/O (WFDB reads, BioClinicalBERT) is mocked so these tests run fast
on any machine without the PTB-XL dataset.
"""

from __future__ import annotations

import pickle
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy
import pytest
import torch

import gdss_multimodal.data as data_module
from gdss_multimodal.data import (
    N_LEADS,
    SEQ_LEN,
    LEAD_IDX,
    PTBXL_CSV,
    RECORDS_DIR,
    split_by_fold,
    load_ptbxl_records,
    compute_train_stats,
    build_embedding_cache,
    PTBXLDataset,
    PTBXLMultiLeadDataset,
    build_datasets,
    _load_waveform,
)


# ---------------------------------------------------------------------------
# Helpers to build a minimal fake PTB-XL directory
# ---------------------------------------------------------------------------

_FAKE_CSV = textwrap.dedent("""\
    ecg_id,filename_lr,report,report_en,strat_fold,scp_codes
    1,records100/00000/00001_lr,Sinusrhythmus,Sinus rhythm,1,"{'NORM': 1.0}"
    2,records100/00000/00002_lr,Sinusrhythmus,Sinus rhythm,9,"{'NORM': 1.0}"
    3,records100/00000/00003_lr,,           ,10,"{'AFIB': 0.5, 'NORM': 0.5}"
    4,records100/00000/00004_lr,Vorhofflimmern,Atrial fibrillation,2,"{'AFIB': 1.0}"
    5,records100/00000/00005_lr,Atrial fibrillation,Atrial fibrillation,10,"{'AFIB': 1.0}"
""")


def _make_fake_data_dir(tmp_path: Path) -> Path:
    """Write a minimal ptbxl_database.csv to a temp directory."""
    data_dir = tmp_path / "ptbxl"
    data_dir.mkdir()
    (data_dir / PTBXL_CSV).write_text(_FAKE_CSV)
    return data_dir


def _fake_waveform() -> numpy.ndarray:
    return numpy.random.randn(N_LEADS, SEQ_LEN).astype(numpy.float32)


def _fake_records(n: int = 4, folds=(1, 2, 9, 10)) -> list[dict]:
    return [
        {
            "ecg_id": i + 1,
            "filename_lr": f"records100/00000/0000{i+1}_lr",
            "report": f"report {i}",
            "strat_fold": folds[i % len(folds)],
            "label": i % 3,
            "_data_dir": "/fake/dir",
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# split_by_fold
# ---------------------------------------------------------------------------

class TestSplitByFold:
    def test_train_folds_1_to_8(self):
        recs = [{"strat_fold": f} for f in range(1, 11)]
        train, val, test = split_by_fold(recs)
        assert len(train) == 8
        assert len(val)   == 1
        assert len(test)  == 1

    def test_val_is_fold_9(self):
        recs = [{"strat_fold": 9}]
        _, val, _ = split_by_fold(recs)
        assert len(val) == 1

    def test_test_is_fold_10(self):
        recs = [{"strat_fold": 10}]
        _, _, test = split_by_fold(recs)
        assert len(test) == 1

    def test_empty_input(self):
        train, val, test = split_by_fold([])
        assert train == val == test == []

    def test_all_train(self):
        recs = [{"strat_fold": f} for f in range(1, 9)]
        train, val, test = split_by_fold(recs)
        assert len(train) == 8
        assert len(val) == len(test) == 0


# ---------------------------------------------------------------------------
# load_ptbxl_records
# ---------------------------------------------------------------------------

class TestLoadPtbxlRecords:
    def test_skips_empty_reports(self, tmp_path):
        data_dir = _make_fake_data_dir(tmp_path)
        records = load_ptbxl_records(data_dir)
        # Record 3 has empty report_en → should be skipped
        ecg_ids = [r["ecg_id"] for r in records]
        assert 3 not in ecg_ids

    def test_includes_non_empty_reports(self, tmp_path):
        data_dir = _make_fake_data_dir(tmp_path)
        records = load_ptbxl_records(data_dir)
        assert len(records) == 4   # records 1, 2, 4, 5 (record 3 has empty report)

    def test_record_keys(self, tmp_path):
        data_dir = _make_fake_data_dir(tmp_path)
        records = load_ptbxl_records(data_dir)
        for r in records:
            assert "ecg_id"      in r
            assert "filename_lr" in r
            assert "report"      in r
            assert "strat_fold"  in r
            assert "label"       in r
            assert "_data_dir"   in r

    def test_prefers_report_en_column(self, tmp_path):
        """If report_en is present it should be used, not raw report."""
        data_dir = _make_fake_data_dir(tmp_path)
        records = load_ptbxl_records(data_dir)
        for r in records:
            # report_en values from CSV
            assert r["report"] in ("Sinus rhythm", "Atrial fibrillation")

    def test_falls_back_to_report_column(self, tmp_path):
        """Without report_en, raw report is used."""
        csv_no_en = textwrap.dedent("""\
            ecg_id,filename_lr,report,strat_fold,scp_codes
            1,records100/00000/00001_lr,Sinusrhythmus,1,"{'NORM': 1.0}"
        """)
        data_dir = tmp_path / "ptbxl2"
        data_dir.mkdir()
        (data_dir / PTBXL_CSV).write_text(csv_no_en)
        records = load_ptbxl_records(data_dir)
        assert records[0]["report"] == "Sinusrhythmus"

    def test_label_assigned(self, tmp_path):
        data_dir = _make_fake_data_dir(tmp_path)
        records = load_ptbxl_records(data_dir)
        for r in records:
            assert isinstance(r["label"], int)

    def test_data_dir_as_string(self, tmp_path):
        data_dir = _make_fake_data_dir(tmp_path)
        records = load_ptbxl_records(str(data_dir))
        assert len(records) > 0


# ---------------------------------------------------------------------------
# _load_waveform (mocked)
# ---------------------------------------------------------------------------

class TestLoadWaveform:
    # wfdb is imported lazily inside _load_waveform; patch the top-level module.

    def test_output_shape(self):
        rec = _fake_records(1)[0]
        with patch("wfdb.rdsamp", return_value=(
            numpy.random.randn(SEQ_LEN, N_LEADS).astype(numpy.float32), {}
        )):
            arr = _load_waveform(rec)
        assert arr.shape == (N_LEADS, SEQ_LEN)
        assert arr.dtype == numpy.float32

    def test_pads_short_signal(self):
        rec = _fake_records(1)[0]
        short_len = SEQ_LEN - 100
        with patch("wfdb.rdsamp", return_value=(
            numpy.random.randn(short_len, N_LEADS).astype(numpy.float32), {}
        )):
            arr = _load_waveform(rec)
        assert arr.shape == (N_LEADS, SEQ_LEN)
        assert numpy.all(arr[:, short_len:] == 0.0)

    def test_truncates_long_signal(self):
        rec = _fake_records(1)[0]
        long_len = SEQ_LEN + 200
        with patch("wfdb.rdsamp", return_value=(
            numpy.random.randn(long_len, N_LEADS).astype(numpy.float32), {}
        )):
            arr = _load_waveform(rec)
        assert arr.shape == (N_LEADS, SEQ_LEN)


# ---------------------------------------------------------------------------
# compute_train_stats (mocked)
# ---------------------------------------------------------------------------

class TestComputeTrainStats:
    def _make_fut(self, result_tuple):
        """Build a mock Future that returns result_tuple from .result()."""
        fut = MagicMock()
        fut.result.return_value = result_tuple
        return fut

    def test_output_shapes(self):
        records  = _fake_records(4)
        fake_w   = _fake_waveform()
        s  = fake_w.sum(axis=1)
        sq = (fake_w ** 2).sum(axis=1)
        c  = fake_w.shape[1]

        futs = [self._make_fut((s, sq, c)) for _ in records]

        executor = MagicMock()
        executor.submit.side_effect = lambda fn, rec: futs.pop(0)

        with patch("gdss_multimodal.data.ProcessPoolExecutor") as pool_cls, \
             patch("gdss_multimodal.data.as_completed", side_effect=lambda fs: iter(fs)):
            pool_cls.return_value.__enter__.return_value = executor
            pool_cls.return_value.__exit__ = MagicMock(return_value=False)
            mean, std = compute_train_stats(records)

        assert mean.shape == (N_LEADS,)
        assert std.shape  == (N_LEADS,)

    def test_std_positive(self):
        records  = _fake_records(4)
        fake_w   = numpy.random.randn(N_LEADS, SEQ_LEN).astype(numpy.float32)
        s  = fake_w.sum(axis=1)
        sq = (fake_w ** 2).sum(axis=1)
        c  = fake_w.shape[1]

        futs = [self._make_fut((s, sq, c)) for _ in records]
        executor = MagicMock()
        executor.submit.side_effect = lambda fn, rec: futs.pop(0)

        with patch("gdss_multimodal.data.ProcessPoolExecutor") as pool_cls, \
             patch("gdss_multimodal.data.as_completed", side_effect=lambda fs: iter(fs)):
            pool_cls.return_value.__enter__.return_value = executor
            pool_cls.return_value.__exit__ = MagicMock(return_value=False)
            mean, std = compute_train_stats(records)

        assert (std > 0).all()


# ---------------------------------------------------------------------------
# build_embedding_cache (mocked)
# ---------------------------------------------------------------------------

class TestBuildEmbeddingCache:
    def test_returns_existing_cache(self, tmp_path):
        """If cache already exists it should be returned without running BERT."""
        cache_path = tmp_path / "emb.pkl"
        prebuilt = {"99": numpy.zeros(768, dtype=numpy.float32)}
        with open(cache_path, "wb") as f:
            pickle.dump(prebuilt, f)
        result = build_embedding_cache([], cache_path, device="cpu")
        assert set(result.keys()) == {"99"}
        assert numpy.allclose(result["99"], prebuilt["99"])

    def test_existing_cache_correct_content(self, tmp_path):
        """Existing cache should be returned with correct keys and values."""
        cache_path = tmp_path / "emb.pkl"
        arr = numpy.ones(768, dtype=numpy.float32) * 3.14
        prebuilt = {"42": arr}
        with open(cache_path, "wb") as f:
            pickle.dump(prebuilt, f)
        result = build_embedding_cache(_fake_records(1), cache_path, device="cpu")
        assert "42" in result
        assert numpy.allclose(result["42"], arr)


# ---------------------------------------------------------------------------
# PTBXLDataset (mocked waveforms + prebuilt embeddings)
# ---------------------------------------------------------------------------

def _make_cache_and_stats(n: int, text_dim: int = 768):
    emb_cache = {str(i + 1): numpy.random.randn(text_dim).astype(numpy.float32)
                 for i in range(n)}
    mean = numpy.zeros(N_LEADS, dtype=numpy.float32)
    std  = numpy.ones(N_LEADS,  dtype=numpy.float32)
    return emb_cache, mean, std


def _make_preloaded_waveforms(n: int):
    return {i: _fake_waveform() for i in range(n)}


class TestPTBXLDataset:
    def test_len(self):
        recs = _fake_records(4)
        emb_cache, mean, std = _make_cache_and_stats(4)
        fake_waves = _make_preloaded_waveforms(4)
        with patch("gdss_multimodal.data._preload_waveforms", return_value=fake_waves):
            ds = PTBXLDataset(recs, emb_cache, mean, std)
        assert len(ds) == 4

    def test_getitem_keys(self):
        recs = _fake_records(2)
        emb_cache, mean, std = _make_cache_and_stats(2)
        fake_waves = _make_preloaded_waveforms(2)
        with patch("gdss_multimodal.data._preload_waveforms", return_value=fake_waves):
            ds = PTBXLDataset(recs, emb_cache, mean, std)
        item = ds[0]
        assert "ecg"       in item
        assert "text_emb"  in item
        assert "record_id" in item
        assert "label"     in item

    def test_ecg_shape(self):
        recs = _fake_records(2)
        emb_cache, mean, std = _make_cache_and_stats(2)
        fake_waves = _make_preloaded_waveforms(2)
        with patch("gdss_multimodal.data._preload_waveforms", return_value=fake_waves):
            ds = PTBXLDataset(recs, emb_cache, mean, std)
        assert ds[0]["ecg"].shape == (1, SEQ_LEN)

    def test_text_emb_shape(self):
        recs = _fake_records(2)
        emb_cache, mean, std = _make_cache_and_stats(2)
        fake_waves = _make_preloaded_waveforms(2)
        with patch("gdss_multimodal.data._preload_waveforms", return_value=fake_waves):
            ds = PTBXLDataset(recs, emb_cache, mean, std)
        assert ds[0]["text_emb"].shape == (768,)

    def test_text_emb_normalised(self):
        recs = _fake_records(2)
        emb_cache, mean, std = _make_cache_and_stats(2)
        fake_waves = _make_preloaded_waveforms(2)
        with patch("gdss_multimodal.data._preload_waveforms", return_value=fake_waves):
            ds = PTBXLDataset(recs, emb_cache, mean, std)
        for i in range(len(ds)):
            norm = ds[i]["text_emb"].norm().item()
            assert abs(norm - 1.0) < 1e-5, f"text_emb norm={norm} != 1.0"

    def test_no_r_peak_mask_key(self):
        recs = _fake_records(2)
        emb_cache, mean, std = _make_cache_and_stats(2)
        fake_waves = _make_preloaded_waveforms(2)
        with patch("gdss_multimodal.data._preload_waveforms", return_value=fake_waves):
            ds = PTBXLDataset(recs, emb_cache, mean, std)
        assert "r_peak_mask" not in ds[0]


# ---------------------------------------------------------------------------
# PTBXLMultiLeadDataset
# ---------------------------------------------------------------------------

class TestPTBXLMultiLeadDataset:
    def test_len_is_12x_recordings(self):
        n_recs = 3
        recs = _fake_records(n_recs)
        emb_cache, mean, std = _make_cache_and_stats(n_recs)
        fake_waves = _make_preloaded_waveforms(n_recs)
        with patch("gdss_multimodal.data._preload_waveforms", return_value=fake_waves):
            ds = PTBXLMultiLeadDataset(recs, emb_cache, mean, std)
        assert len(ds) == n_recs * N_LEADS

    def test_getitem_keys(self):
        recs = _fake_records(2)
        emb_cache, mean, std = _make_cache_and_stats(2)
        fake_waves = _make_preloaded_waveforms(2)
        with patch("gdss_multimodal.data._preload_waveforms", return_value=fake_waves):
            ds = PTBXLMultiLeadDataset(recs, emb_cache, mean, std)
        item = ds[0]
        assert "ecg"       in item
        assert "text_emb"  in item
        assert "lead_idx"  in item
        assert "record_id" in item
        assert "label"     in item

    def test_lead_idx_range(self):
        recs = _fake_records(2)
        emb_cache, mean, std = _make_cache_and_stats(2)
        fake_waves = _make_preloaded_waveforms(2)
        with patch("gdss_multimodal.data._preload_waveforms", return_value=fake_waves):
            ds = PTBXLMultiLeadDataset(recs, emb_cache, mean, std)
        for i in range(len(ds)):
            assert 0 <= ds[i]["lead_idx"] < N_LEADS

    def test_ecg_shape(self):
        recs = _fake_records(2)
        emb_cache, mean, std = _make_cache_and_stats(2)
        fake_waves = _make_preloaded_waveforms(2)
        with patch("gdss_multimodal.data._preload_waveforms", return_value=fake_waves):
            ds = PTBXLMultiLeadDataset(recs, emb_cache, mean, std)
        for i in range(len(ds)):
            assert ds[i]["ecg"].shape == (1, SEQ_LEN)

    def test_no_r_peak_mask_key(self):
        recs = _fake_records(2)
        emb_cache, mean, std = _make_cache_and_stats(2)
        fake_waves = _make_preloaded_waveforms(2)
        with patch("gdss_multimodal.data._preload_waveforms", return_value=fake_waves):
            ds = PTBXLMultiLeadDataset(recs, emb_cache, mean, std)
        assert "r_peak_mask" not in ds[0]


# ---------------------------------------------------------------------------
# build_datasets (integration-level mock)
# ---------------------------------------------------------------------------

class TestBuildDatasets:
    def test_returns_three_datasets(self, tmp_path):
        data_dir = _make_fake_data_dir(tmp_path)
        # ecg_ids with non-empty reports: 1 (fold 1), 2 (fold 9), 4 (fold 2), 5 (fold 10)
        emb = {str(i): numpy.random.randn(768).astype(numpy.float32)
               for i in [1, 2, 4, 5]}
        mean = numpy.zeros(N_LEADS, dtype=numpy.float32)
        std  = numpy.ones(N_LEADS,  dtype=numpy.float32)

        # Preloaded waveform dict keys are 0-based indices into each split
        fake_waves = _make_preloaded_waveforms(4)

        with patch("gdss_multimodal.data._preload_waveforms", return_value=fake_waves), \
             patch("gdss_multimodal.data.build_embedding_cache", return_value=emb), \
             patch("gdss_multimodal.data.compute_train_stats", return_value=(mean, std)):
            train_ds, val_ds, test_ds = build_datasets(
                str(data_dir), str(tmp_path), bert_device="cpu", multi_lead=False
            )
        assert train_ds is not None
        assert val_ds   is not None
        assert test_ds  is not None
        assert len(train_ds) == 2   # folds 1, 2
        assert len(val_ds)   == 1   # fold 9
        assert len(test_ds)  == 1   # fold 10

    def test_multi_lead_returns_multilead_dataset(self, tmp_path):
        data_dir = _make_fake_data_dir(tmp_path)
        emb = {str(i): numpy.random.randn(768).astype(numpy.float32)
               for i in [1, 2, 4, 5]}
        mean = numpy.zeros(N_LEADS, dtype=numpy.float32)
        std  = numpy.ones(N_LEADS,  dtype=numpy.float32)
        fake_waves = _make_preloaded_waveforms(4)

        with patch("gdss_multimodal.data._preload_waveforms", return_value=fake_waves), \
             patch("gdss_multimodal.data.build_embedding_cache", return_value=emb), \
             patch("gdss_multimodal.data.compute_train_stats", return_value=(mean, std)):
            train_ds, _, _ = build_datasets(
                str(data_dir), str(tmp_path), bert_device="cpu", multi_lead=True
            )
        assert isinstance(train_ds, PTBXLMultiLeadDataset)
