"""PTB-XL dataset loading, normalisation, and BioClinicalBERT embedding cache.

PTB-XL directory layout (after fetch_data app runs):

    DATA_DIR/
        ptbxl_database.csv      metadata with report_en column (English translations)
        records100/             100 Hz WFDB recordings (12 leads × 1000 samples)
            00000/
                00001_lr.dat
                00001_lr.hea

Fold convention (PTB-XL stratified splits):
    folds 1–8  → train
    fold 9     → validation
    fold 10    → test

Records with empty report_en are excluded from training.
"""

from __future__ import annotations

import ast
import collections
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy
import pandas
import torch
import transformers
import wfdb
from torch.utils.data import Dataset


N_LEADS = 12
SEQ_LEN = 1000
LEAD_IDX = 1           # Lead II — default for single-lead visualisation
BERT_MODEL = "emilyalsentzer/Bio_ClinicalBERT"
PTBXL_CSV = "ptbxl_database.csv"
RECORDS_DIR = "records100"

_TRAIN_FOLDS = set(range(1, 9))
_VAL_FOLDS = {9}


def split_by_fold(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Partition records into (train, val, test) by strat_fold.

    Args:
        records: List of record dicts from load_ptbxl_records.

    Returns:
        Tuple of (train, val, test) record lists.
    """
    train, val, test = [], [], []
    for rec in records:
        fold = int(rec["strat_fold"])
        if fold in _TRAIN_FOLDS:
            train.append(rec)
        elif fold in _VAL_FOLDS:
            val.append(rec)
        else:
            test.append(rec)
    return train, val, test


def load_ptbxl_records(data_dir: str | Path) -> list[dict[str, Any]]:
    """Read ptbxl_database.csv and return a list of record dicts.

    Uses the ``report_en`` column (English translations written by fetch_data).
    Records with an empty report are excluded.

    Each dict contains:
        ecg_id (int), filename_lr (str), report (str),
        strat_fold (int), label (int, −1 if unavailable).

    Args:
        data_dir: Root directory of the processed PTB-XL data.

    Returns:
        List of record dicts (only records with a non-empty report).
    """
    data_dir = Path(data_dir)
    df = pandas.read_csv(data_dir / PTBXL_CSV, index_col="ecg_id")

    # Determine the report column: prefer translated English, fall back to raw.
    if "report_en" in df.columns:
        report_col = "report_en"
    else:
        report_col = "report"

    try:
        parsed = df["scp_codes"].apply(ast.literal_eval)
        all_codes: list[str] = []
        for codes in parsed:
            all_codes.extend(codes.keys())
        code_index = {
            c: i for i, (c, _) in enumerate(collections.Counter(all_codes).most_common())
        }
        df["label"] = parsed.apply(
            lambda d: code_index[max(d, key=d.get)] if d else -1
        )
    except Exception:
        df["label"] = -1

    records = []
    skipped = 0
    for ecg_id, row in df.iterrows():
        report = str(row.get(report_col, "")).strip()
        if not report or report.lower() in ("nan", "none", ""):
            skipped += 1
            continue
        records.append({
            "ecg_id": int(ecg_id),
            "filename_lr": str(row["filename_lr"]),
            "report": report,
            "strat_fold": int(row["strat_fold"]),
            "label": int(row["label"]),
            "_data_dir": str(data_dir),
        })

    if skipped:
        print(f"  skipped {skipped} records with empty reports.")
    return records


def _load_waveform(rec: dict[str, Any]) -> numpy.ndarray:
    """Read one WFDB recording and return float32 array of shape (12, SEQ_LEN)."""
    path = Path(rec["_data_dir"]) / rec["filename_lr"]
    signal, _ = wfdb.rdsamp(str(path))
    arr = signal.T.astype(numpy.float32)
    if arr.shape[1] < SEQ_LEN:
        pad = numpy.zeros((N_LEADS, SEQ_LEN - arr.shape[1]), dtype=numpy.float32)
        arr = numpy.concatenate([arr, pad], axis=1)
    else:
        arr = arr[:, :SEQ_LEN]
    return arr


def _waveform_stats(rec: dict[str, Any]) -> tuple[numpy.ndarray, numpy.ndarray, int]:
    ecg = _load_waveform(rec)
    return ecg.sum(axis=1), (ecg**2).sum(axis=1), ecg.shape[1]


def compute_train_stats(
    records: list[dict[str, Any]],
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Compute per-lead (mean, std) over the training set.

    Args:
        records: Training split record dicts.

    Returns:
        Tuple of (mean, std), each of shape (N_LEADS,).
    """
    sums = numpy.zeros(N_LEADS, dtype=numpy.float64)
    sq_sums = numpy.zeros(N_LEADS, dtype=numpy.float64)
    count = 0
    n = len(records)
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor() as ex:
        futures = {ex.submit(_waveform_stats, rec): rec for rec in records}
        for fut in as_completed(futures):
            s, sq, c = fut.result()
            sums += s
            sq_sums += sq
            count += c
            done += 1
            if done % 1000 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed
                print(f"  stats {done}/{n} | {rate:.0f} rec/s | ETA {(n - done) / rate:.0f}s")
    mean = sums / count
    std = numpy.sqrt(sq_sums / count - mean**2).clip(min=1e-6)
    return mean.astype(numpy.float32), std.astype(numpy.float32)


def build_embedding_cache(
    records: list[dict[str, Any]],
    cache_path: str | Path,
    device: str = "cpu",
    batch_size: int = 256,
) -> dict[str, numpy.ndarray]:
    """Encode all reports with BioClinicalBERT and persist to disk.

    Cache maps str(ecg_id) → float32 ndarray of shape (768,).
    Returns the cache immediately if the file already exists.

    Args:
        records: All record dicts (train + val + test).
        cache_path: Path to the pickle cache file.
        device: Device string for the BioClinicalBERT model.
        batch_size: Tokeniser batch size.

    Returns:
        Dict mapping ecg_id strings to CLS-token embeddings.
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"Building BioClinicalBERT embedding cache → {cache_path}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(BERT_MODEL)
    model = transformers.AutoModel.from_pretrained(BERT_MODEL).eval().to(device)

    texts = [rec["report"] for rec in records]
    ids = [str(rec["ecg_id"]) for rec in records]
    cache: dict[str, numpy.ndarray] = {}

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            enc = tokenizer(
                texts[start: start + batch_size],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)
            out = model(**enc)
            cls_vecs = out.last_hidden_state[:, 0, :].cpu().numpy().astype(numpy.float32)
            for rid, vec in zip(ids[start: start + batch_size], cls_vecs):
                cache[rid] = vec
            if (start // batch_size) % 10 == 0:
                print(f"  encoded {min(start + batch_size, len(texts))}/{len(texts)}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)
    print("  cache saved.")
    return cache


def _preload_waveforms(records: list[dict[str, Any]]) -> dict[int, numpy.ndarray]:
    """Parallel-load all waveforms, returning {index: ndarray}."""
    n = len(records)
    t0 = time.time()
    waveforms: dict[int, numpy.ndarray] = {}
    with ProcessPoolExecutor() as ex:
        futures = {ex.submit(_load_waveform, rec): i for i, rec in enumerate(records)}
        for done, fut in enumerate(as_completed(futures), 1):
            waveforms[futures[fut]] = fut.result()
            if done % 2000 == 0 or done == n:
                rate = done / (time.time() - t0)
                print(
                    f"  preload {done}/{n} | {rate:.0f} rec/s"
                    f" | ETA {(n - done) / rate:.0f}s"
                )
    return waveforms


class PTBXLDataset(Dataset):
    """PTB-XL (ECG, text embedding) dataset — single-lead, fully pre-loaded.

    All waveforms are normalised at construction time; __getitem__ is a
    pure tensor index with no disk I/O.

    Args:
        records: Record dicts for this split.
        embedding_cache: Dict mapping ecg_id → BioClinicalBERT embedding.
        mean: Per-lead mean of shape (N_LEADS,) from the training set.
        std: Per-lead std of shape (N_LEADS,) from the training set.
    """

    def __init__(
        self,
        records: list[dict[str, Any]],
        embedding_cache: dict[str, numpy.ndarray],
        mean: numpy.ndarray,
        std: numpy.ndarray,
    ) -> None:
        mean_t = torch.from_numpy(mean[:, None])
        std_t = torch.from_numpy(std[:, None])

        n = len(records)
        print(f"  preloading {n} waveforms into RAM…")
        waveforms = _preload_waveforms(records)

        ecgs, text_embs, rids, labels = [], [], [], []
        for i, rec in enumerate(records):
            rid = str(rec["ecg_id"])
            ecg = torch.from_numpy(waveforms[i])
            ecg = (ecg - mean_t) / std_t
            ecg = ecg[LEAD_IDX: LEAD_IDX + 1]
            ecgs.append(ecg)
            text_embs.append(torch.from_numpy(embedding_cache[rid]))
            rids.append(rid)
            labels.append(int(rec["label"]))

        self.ecgs = torch.stack(ecgs)
        self.text_embs = torch.nn.functional.normalize(torch.stack(text_embs), dim=-1)
        self.rids = rids
        self.labels = labels

    def __len__(self) -> int:
        return len(self.rids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {
            "ecg": self.ecgs[idx],
            "text_emb": self.text_embs[idx],
            "record_id": self.rids[idx],
            "label": self.labels[idx],
        }


class PTBXLMultiLeadDataset(Dataset):
    """PTB-XL dataset expanded to all 12 leads.

    Each recording yields 12 training samples — one per lead — with a
    ``lead_idx`` key (0–11) for lead-identity conditioning.
    Effective dataset size is 12× the number of recordings.

    Args:
        records: Record dicts for this split.
        embedding_cache: Dict mapping ecg_id → BioClinicalBERT embedding.
        mean: Per-lead mean of shape (N_LEADS,) from the training set.
        std: Per-lead std of shape (N_LEADS,) from the training set.
    """

    def __init__(
        self,
        records: list[dict[str, Any]],
        embedding_cache: dict[str, numpy.ndarray],
        mean: numpy.ndarray,
        std: numpy.ndarray,
    ) -> None:
        mean_t = torch.from_numpy(mean[:, None])
        std_t = torch.from_numpy(std[:, None])

        n = len(records)
        print(f"  preloading {n} waveforms (all 12 leads) into RAM…")
        waveforms = _preload_waveforms(records)

        ecgs_all, text_embs_all, rids_all, labels_all = [], [], [], []
        for i, rec in enumerate(records):
            rid = str(rec["ecg_id"])
            ecg = torch.from_numpy(waveforms[i])
            ecg = (ecg - mean_t) / std_t
            ecgs_all.append(ecg)
            text_embs_all.append(torch.from_numpy(embedding_cache[rid]))
            rids_all.append(rid)
            labels_all.append(int(rec["label"]))

        self._ecgs = torch.stack(ecgs_all)                                      # (N, 12, SEQ_LEN)
        self._text_embs = torch.nn.functional.normalize(
            torch.stack(text_embs_all), dim=-1
        )                                                                        # (N, 768)
        self._rids = rids_all
        self._labels = labels_all
        self._n_recordings = n

    def __len__(self) -> int:
        return self._n_recordings * N_LEADS

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec_idx = idx // N_LEADS
        lead = idx % N_LEADS
        return {
            "ecg": self._ecgs[rec_idx, lead: lead + 1, :],   # (1, SEQ_LEN)
            "text_emb": self._text_embs[rec_idx],
            "lead_idx": lead,
            "record_id": self._rids[rec_idx],
            "label": self._labels[rec_idx],
        }


def build_datasets(
    data_dir: str = "data/ptbxl",
    cache_dir: str = "cache",
    bert_device: str = "cpu",
    multi_lead: bool = False,
) -> tuple[Dataset, Dataset, Dataset]:
    """Load PTB-XL, cache BERT embeddings and ECG stats, return all three splits.

    Args:
        data_dir: Root of the processed PTB-XL directory.
        cache_dir: Directory for bert_embeddings.pkl and ecg_stats.pkl.
        bert_device: Device for BioClinicalBERT encoding.
        multi_lead: If True return PTBXLMultiLeadDataset (12× data, with
            lead_idx per sample) instead of single-lead PTBXLDataset.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
    """
    records = load_ptbxl_records(data_dir)
    train_recs, val_recs, test_recs = split_by_fold(records)

    emb_cache = build_embedding_cache(
        train_recs + val_recs + test_recs,
        Path(cache_dir) / "bert_embeddings.pkl",
        device=bert_device,
    )

    stats_path = Path(cache_dir) / f"ecg_stats_{SEQ_LEN}hz.pkl"
    if stats_path.exists():
        with open(stats_path, "rb") as f:
            mean, std = pickle.load(f)
    else:
        print("Computing per-lead normalisation stats from training set…")
        mean, std = compute_train_stats(train_recs)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_path, "wb") as f:
            pickle.dump((mean, std), f)

    cls = PTBXLMultiLeadDataset if multi_lead else PTBXLDataset
    return (
        cls(train_recs, emb_cache, mean, std),
        cls(val_recs,   emb_cache, mean, std),
        cls(test_recs,  emb_cache, mean, std),
    )
