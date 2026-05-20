"""PTB-XL dataset loading, normalisation, and BioClinicalBERT embedding cache.

PTB-XL is sourced from PhysioNet and stored in WFDB format:

    data_dir/
        ptbxl_database.csv          metadata: ecg_id, report, strat_fold, …
        records100/                 100 Hz WFDB recordings (12 leads × 1000 samples)
            00000/
                00001_lr.dat
                00001_lr.hea

Download instructions:
    1. Create a free PhysioNet account at https://physionet.org
    2. Download PTB-XL 1.0.3 (fastest via Kaggle mirror)
    3. modal volume put gdss-cache ~/Downloads/ptbxl.zip ptbxl.zip
    4. modal run download_ptbxl.py

Fold convention (PTB-XL stratified splits):
    folds 1–8  → train
    fold 9     → validation
    fold 10    → test
"""

from __future__ import annotations

import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any
import time

import numpy
import torch
from torch.utils.data import Dataset


N_LEADS = 12
SEQ_LEN = 1000
LEAD_IDX = 1           # default single-lead index (Lead II) used in single-lead mode
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

    Each dict contains:
        ecg_id (int), filename_lr (str), report (str),
        strat_fold (int), label (int, −1 if unavailable).

    Args:
        data_dir: Root directory of the PTB-XL download.

    Returns:
        List of record dicts.
    """
    import ast
    import collections
    import pandas

    data_dir = Path(data_dir)
    df = pandas.read_csv(data_dir / PTBXL_CSV, index_col="ecg_id")

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
    for ecg_id, row in df.iterrows():
        records.append({
            "ecg_id": int(ecg_id),
            "filename_lr": str(row["filename_lr"]),
            "report": str(row.get("report", "")),
            "strat_fold": int(row["strat_fold"]),
            "label": int(row["label"]),
            "_data_dir": str(data_dir),
        })
    return records


def _load_waveform(rec: dict[str, Any]) -> numpy.ndarray:
    """Read one WFDB recording and return float32 array of shape (12, SEQ_LEN).

    Args:
        rec: Record dict from load_ptbxl_records.

    Returns:
        ECG array of shape (N_LEADS, SEQ_LEN), padded or truncated as needed.
    """
    import wfdb

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
    import transformers

    cache_path = Path(cache_path)
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"Building BioClinicalBERT embedding cache → {cache_path}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(BERT_MODEL)
    model = transformers.AutoModel.from_pretrained(BERT_MODEL).eval().to(device)

    texts = [rec.get("report", "") or "" for rec in records]
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


def _detect_rpeak_mask(waveform_12lead: numpy.ndarray, fs: int = 100, det_lead: int = 1) -> numpy.ndarray:
    """Detect R-peaks in a 12-lead waveform and return a binary mask.

    Uses Lead II (index 1) for detection — it has the clearest QRS complex.
    The resulting mask is shared across all leads since R-peaks are a property
    of the cardiac cycle, not of the recording angle.

    Args:
        waveform_12lead: Array of shape (N_LEADS, SEQ_LEN).
        fs: Sampling frequency in Hz.
        det_lead: Lead index to run XQRS on (default 1 = Lead II).

    Returns:
        Float32 binary mask of shape (SEQ_LEN,) with 1.0 at R-peak positions.
    """
    try:
        import wfdb.processing
        sig = waveform_12lead[det_lead].astype(float)
        qrs_inds = wfdb.processing.xqrs_detect(sig=sig, fs=fs, verbose=False)
        mask = numpy.zeros(waveform_12lead.shape[1], dtype=numpy.float32)
        valid = qrs_inds[qrs_inds < mask.shape[0]]
        mask[valid] = 1.0
        return mask
    except Exception:
        return numpy.zeros(waveform_12lead.shape[1], dtype=numpy.float32)


def build_rpeak_cache(
    records: list[dict[str, Any]],
    cache_path: str | Path,
    fs: int = 100,
) -> dict[str, numpy.ndarray]:
    """Compute or load R-peak binary masks for all records.

    Maps str(ecg_id) → float32 array of shape (SEQ_LEN,).
    Loads from disk if the cache file already exists.

    Args:
        records: All record dicts (any split).
        cache_path: Path to store/load the pickle cache.
        fs: ECG sampling frequency in Hz.

    Returns:
        Dict mapping ecg_id strings to binary R-peak masks.
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"Computing R-peak masks for {len(records)} recordings → {cache_path}")
    t0 = time.time()
    rpeak_cache: dict[str, numpy.ndarray] = {}
    for i, rec in enumerate(records):
        waveform = _load_waveform(rec)
        mask = _detect_rpeak_mask(waveform, fs=fs)
        rpeak_cache[str(rec["ecg_id"])] = mask
        if (i + 1) % 2000 == 0 or (i + 1) == len(records):
            rate = (i + 1) / (time.time() - t0)
            print(f"  rpeak {i + 1}/{len(records)} | {rate:.0f} rec/s")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(rpeak_cache, f)
    print("  R-peak cache saved.")
    return rpeak_cache


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
    """PTB-XL (ECG, text embedding) dataset, fully pre-loaded into RAM.

    All waveforms are normalised and sliced to LEAD_IDX at construction time
    so __getitem__ is a pure tensor index with no disk I/O.

    Args:
        records: Record dicts for this split.
        embedding_cache: Dict mapping ecg_id → BioClinicalBERT embedding.
        mean: Per-lead mean of shape (N_LEADS,) from the training set.
        std: Per-lead std of shape (N_LEADS,) from the training set.
        rpeak_cache: Dict mapping ecg_id → binary R-peak mask of shape (SEQ_LEN,).
    """

    def __init__(
        self,
        records: list[dict[str, Any]],
        embedding_cache: dict[str, numpy.ndarray],
        mean: numpy.ndarray,
        std: numpy.ndarray,
        rpeak_cache: dict[str, numpy.ndarray] | None = None,
    ) -> None:
        mean_t = torch.from_numpy(mean[:, None])
        std_t = torch.from_numpy(std[:, None])

        n = len(records)
        print(f"  preloading {n} waveforms into RAM…")
        waveforms = _preload_waveforms(records)

        ecgs, text_embs, rids, labels, rpeaks = [], [], [], [], []
        for i, rec in enumerate(records):
            rid = str(rec["ecg_id"])
            ecg = torch.from_numpy(waveforms[i])
            ecg = (ecg - mean_t) / std_t
            ecg = ecg[LEAD_IDX: LEAD_IDX + 1]
            ecgs.append(ecg)
            text_embs.append(torch.from_numpy(embedding_cache[rid]))
            rids.append(rid)
            labels.append(int(rec["label"]))
            if rpeak_cache is not None and rid in rpeak_cache:
                rpeaks.append(torch.from_numpy(rpeak_cache[rid]).unsqueeze(0))  # (1, SEQ_LEN)
            else:
                rpeaks.append(torch.zeros(1, SEQ_LEN))

        self.ecgs = torch.stack(ecgs)
        self.text_embs = torch.stack(text_embs)
        self.rpeaks = torch.stack(rpeaks)
        self.rids = rids
        self.labels = labels

    def __len__(self) -> int:
        return len(self.rids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "ecg": self.ecgs[idx],
            "text_emb": self.text_embs[idx],
            "r_peak_mask": self.rpeaks[idx],
            "record_id": self.rids[idx],
            "label": self.labels[idx],
        }


class PTBXLMultiLeadDataset(Dataset):
    """PTB-XL dataset expanded to all 12 leads.

    Each PTB-XL recording yields 12 training samples — one per lead — with a
    ``lead_idx`` key (0–11) so the model can condition on lead identity.

    R-peak masks are shared across all 12 leads of the same recording: the mask
    is always detected from Lead II (strongest QRS) since cardiac beat positions
    are the same regardless of recording angle.

    Effective dataset size is 12× the number of recordings.

    Args:
        records: Record dicts for this split.
        embedding_cache: Dict mapping ecg_id → BioClinicalBERT embedding.
        mean: Per-lead mean of shape (N_LEADS,) from the training set.
        std: Per-lead std of shape (N_LEADS,) from the training set.
        rpeak_cache: Dict mapping ecg_id → binary R-peak mask of shape (SEQ_LEN,).
    """

    def __init__(
        self,
        records: list[dict[str, Any]],
        embedding_cache: dict[str, numpy.ndarray],
        mean: numpy.ndarray,
        std: numpy.ndarray,
        rpeak_cache: dict[str, numpy.ndarray] | None = None,
    ) -> None:
        mean_t = torch.from_numpy(mean[:, None])   # (N_LEADS, 1)
        std_t = torch.from_numpy(std[:, None])     # (N_LEADS, 1)

        n = len(records)
        print(f"  preloading {n} waveforms (all 12 leads) into RAM…")
        waveforms = _preload_waveforms(records)

        # Store all 12 leads normalised: (N, 12, SEQ_LEN)
        ecgs_all, text_embs_all, rids_all, labels_all, rpeaks_all = [], [], [], [], []
        for i, rec in enumerate(records):
            rid = str(rec["ecg_id"])
            ecg = torch.from_numpy(waveforms[i])   # (12, SEQ_LEN)
            ecg = (ecg - mean_t) / std_t           # per-lead normalisation
            ecgs_all.append(ecg)
            text_embs_all.append(torch.from_numpy(embedding_cache[rid]))
            rids_all.append(rid)
            labels_all.append(int(rec["label"]))
            if rpeak_cache is not None and rid in rpeak_cache:
                rpeaks_all.append(torch.from_numpy(rpeak_cache[rid]).unsqueeze(0))  # (1, SEQ_LEN)
            else:
                rpeaks_all.append(torch.zeros(1, SEQ_LEN))

        # (N, 12, SEQ_LEN)
        self._ecgs = torch.stack(ecgs_all)
        # (N, 768)
        self._text_embs = torch.stack(text_embs_all)
        # (N, 1, SEQ_LEN) — shared across all 12 leads per recording
        self._rpeaks = torch.stack(rpeaks_all)
        self._rids = rids_all
        self._labels = labels_all
        self._n_recordings = n

    def __len__(self) -> int:
        return self._n_recordings * N_LEADS

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rec_idx = idx // N_LEADS
        lead = idx % N_LEADS
        return {
            "ecg": self._ecgs[rec_idx, lead: lead + 1, :],   # (1, SEQ_LEN)
            "text_emb": self._text_embs[rec_idx],
            "lead_idx": lead,
            "r_peak_mask": self._rpeaks[rec_idx],             # (1, SEQ_LEN)
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
        data_dir: Root of the PTB-XL PhysioNet directory.
        cache_dir: Directory for bert_embeddings.pkl and ecg_stats.pkl.
        bert_device: Device for BioClinicalBERT encoding.
        multi_lead: If True, return PTBXLMultiLeadDataset (12× data, with
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

    rpeak_cache = build_rpeak_cache(
        train_recs + val_recs + test_recs,
        Path(cache_dir) / "rpeak_masks.pkl",
    )

    cls = PTBXLMultiLeadDataset if multi_lead else PTBXLDataset
    return (
        cls(train_recs, emb_cache, mean, std, rpeak_cache),
        cls(val_recs, emb_cache, mean, std, rpeak_cache),
        cls(test_recs, emb_cache, mean, std, rpeak_cache),
    )
