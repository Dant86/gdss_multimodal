"""Fetch, translate, and process PTB-XL data.

Pipeline
--------
1. Download PTB-XL from Kaggle (khyeh0719/ptb-xl-dataset).
2. Extract into a staging directory.
3. Load ptbxl_database.csv; skip records with an empty report.
4. Submit all non-empty reports to the Anthropic Batch API for normalisation:
   Claude returns the text in English regardless of source language.
5. Write a new ``report_en`` column to ptbxl_database.csv.
6. Compute per-lead normalisation statistics from the training split.
7. Build the BioClinicalBERT embedding cache.
8. Delete the staging directory.

Environment variables (loaded from .env):
    DATA_DIR           Destination for the processed PTB-XL directory.
    CACHE_DIR          Destination for BERT embeddings and ECG stats.
    KAGGLE_USERNAME    Kaggle API credentials.
    KAGGLE_KEY
    ANTHROPIC_API_KEY  For the translation batch.
    HF_TOKEN           Optional HuggingFace token.

Usage
-----
    python apps/fetch_data/main.py [--data-dir DIR] [--cache-dir DIR]
                                   [--bert-device cpu|cuda]
                                   [--translation-model claude-haiku-4-5-20251001]
"""

from __future__ import annotations

import argparse
import os
import pickle
import shutil
import subprocess
import time
from pathlib import Path

import anthropic
import pandas
from dotenv import load_dotenv

import gdss_multimodal.data as data_module


# ---------------------------------------------------------------------------
# Kaggle download
# ---------------------------------------------------------------------------

def _download_kaggle(staging_dir: Path) -> Path:
    """Download PTB-XL from Kaggle and return the path to the extracted directory."""
    staging_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading PTB-XL from Kaggle…")
    subprocess.run(
        [
            "kaggle", "datasets", "download",
            "-d", "khyeh0719/ptb-xl-dataset",
            "-p", str(staging_dir),
            "--unzip",
        ],
        env=os.environ.copy(),
        check=True,
    )

    candidates = list(staging_dir.rglob("ptbxl_database.csv"))
    if not candidates:
        raise FileNotFoundError(
            "ptbxl_database.csv not found after extraction. "
            f"Contents of {staging_dir}: {list(staging_dir.iterdir())}"
        )
    return candidates[0].parent


# ---------------------------------------------------------------------------
# Normalisation via Anthropic Batch API
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a medical language specialist. The following is a cardiology ECG "
    "report. It may be written in German or English. If it is in German, "
    "translate it to English. If it is already in English, return it unchanged. "
    "Output only the final English text with no preamble, no explanation, and "
    "no quotation marks."
)


def _normalise_batch(
    texts: list[str],
    ecg_ids: list[int],
    model: str = "claude-haiku-4-5-20251001",
) -> dict[int, str]:
    """Submit all reports to the Batch API and return English text for each.

    Claude handles language detection internally — German reports are
    translated, English reports are returned unchanged.

    Args:
        texts: Report strings (any language).
        ecg_ids: Corresponding ECG IDs used as custom_id keys.
        model: Anthropic model to use.

    Returns:
        Dict mapping ecg_id → English report text.
    """
    client = anthropic.Anthropic()

    requests = [
        {
            "custom_id": str(eid),
            "params": {
                "model": model,
                "max_tokens": 512,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": text}],
            },
        }
        for eid, text in zip(ecg_ids, texts)
    ]

    print(f"  Submitting batch of {len(requests)} reports…")
    batch = client.messages.batches.create(requests=requests)
    batch_id = batch.id
    print(f"  Batch ID: {batch_id}")

    while True:
        batch = client.messages.batches.retrieve(batch_id)
        status = batch.processing_status
        counts = batch.request_counts
        print(
            f"  [{status}] processing={counts.processing} "
            f"succeeded={counts.succeeded} errored={counts.errored}"
        )
        if status == "ended":
            break
        time.sleep(60)

    results: dict[int, str] = {}
    errors: list[str] = []
    for result in client.messages.batches.results(batch_id):
        eid = int(result.custom_id)
        if result.result.type == "succeeded":
            results[eid] = result.result.message.content[0].text.strip()
        else:
            results[eid] = ""
            if len(errors) < 5:
                errors.append(f"  ecg_id={eid}: {result.result}")

    n_err = sum(1 for v in results.values() if v == "")
    n_ok = len(results) - n_err
    print(f"  Batch complete: {len(results)} results, {n_ok} succeeded, {n_err} errored.")
    if errors:
        print("  Sample errors (first 5):")
        for e in errors:
            print(e)

    return results


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(
    data_dir: Path,
    cache_dir: Path,
    bert_device: str = "cpu",
    translation_model: str = "claude-haiku-4-5-20251001",
    staging_dir: Path | None = None,
    skip_download: bool = False,
    skip_translation: bool = False,
) -> None:
    """Run the full fetch-and-process pipeline.

    Args:
        data_dir: Destination for processed PTB-XL data.
        cache_dir: Destination for BERT embeddings and ECG stats.
        bert_device: Device string for BioClinicalBERT encoding.
        translation_model: Anthropic model for report normalisation.
        staging_dir: Temporary directory for the Kaggle download.
        skip_download: If True, skip Kaggle download and read the CSV
            directly from data_dir (waveforms must already be in place).
        skip_translation: If True, skip the Batch API translation step and
            read report_en directly from the existing CSV in data_dir.
    """
    if staging_dir is None:
        staging_dir = data_dir.parent / "_ptbxl_staging"

    # ── 1. Download ──────────────────────────────────────────────────────────
    if skip_download or skip_translation:
        print("Skipping Kaggle download — reading existing data from data_dir.")
        csv_path = data_dir / data_module.PTBXL_CSV
    else:
        raw_dir = _download_kaggle(staging_dir)
        print(f"Extracted to: {raw_dir}")
        csv_path = raw_dir / data_module.PTBXL_CSV

    # ── 2. Load CSV ──────────────────────────────────────────────────────────
    df = pandas.read_csv(csv_path, index_col="ecg_id")
    print(f"Loaded {len(df)} records from {csv_path}")

    if skip_translation:
        print("Skipping translation — using existing report_en column.")
        n_usable = (df.get("report_en", pandas.Series(dtype=str)).fillna("").str.strip() != "").sum()
        print(f"  Usable records with report_en: {n_usable}/{len(df)}")
    else:
        # ── 3. Collect non-empty reports ─────────────────────────────────────
        reports = df.get("report", pandas.Series(dtype=str)).fillna("").astype(str)

        batch_ids, batch_texts = [], []
        skipped = 0
        for ecg_id, text in reports.items():
            text = text.strip()
            if not text or text.lower() in ("nan", "none"):
                skipped += 1
                continue
            batch_ids.append(ecg_id)
            batch_texts.append(text)

        print(f"  Sending {len(batch_ids)} reports to Claude ({skipped} empty, skipped).")

        # ── 4. Normalise via Batch API ───────────────────────────────────────
        normalised = _normalise_batch(batch_texts, batch_ids, model=translation_model)

        # ── 5. Write report_en column ────────────────────────────────────────
        df["report_en"] = pandas.Series(normalised).reindex(df.index, fill_value="")
        n_usable = (df["report_en"].str.strip() != "").sum()
        print(f"  Usable records with report_en: {n_usable}/{len(df)}")

    # ── 6. Write CSV (and copy waveforms if downloaded) ─────────────────────
    data_dir.mkdir(parents=True, exist_ok=True)
    if not skip_translation:
        dest_csv = data_dir / data_module.PTBXL_CSV
        df.to_csv(dest_csv)
        print(f"  Saved updated CSV → {dest_csv}")

    if not skip_download and not skip_translation:
        records_src = raw_dir / data_module.RECORDS_DIR
        records_dst = data_dir / data_module.RECORDS_DIR
        if records_dst.exists():
            shutil.rmtree(records_dst)
        shutil.copytree(records_src, records_dst)
        print(f"  Copied waveforms → {records_dst}")

    # ── 7. Normalisation stats ───────────────────────────────────────────────
    cache_dir.mkdir(parents=True, exist_ok=True)
    stats_path = cache_dir / f"ecg_stats_{data_module.SEQ_LEN}hz.pkl"
    if not stats_path.exists():
        print("Computing per-lead normalisation statistics…")
        all_records = data_module.load_ptbxl_records(data_dir)
        train_recs, _, _ = data_module.split_by_fold(all_records)
        mean, std = data_module.compute_train_stats(train_recs)
        with open(stats_path, "wb") as f:
            pickle.dump((mean, std), f)
        print(f"  Stats saved → {stats_path}")
    else:
        print(f"  Stats already exist at {stats_path}, skipping.")

    # ── 8. BERT embedding cache ──────────────────────────────────────────────
    emb_cache_path = cache_dir / "bert_embeddings.pkl"
    if not emb_cache_path.exists():
        print("Building BioClinicalBERT embedding cache…")
        all_records = data_module.load_ptbxl_records(data_dir)
        data_module.build_embedding_cache(all_records, emb_cache_path, device=bert_device)
    else:
        print(f"  Embedding cache already exists at {emb_cache_path}, skipping.")

    # ── 9. Delete staging directory (only if we created it) ─────────────────
    if not skip_download and not skip_translation and staging_dir.exists():
        print(f"Deleting staging directory {staging_dir}…")
        shutil.rmtree(staging_dir)
    print("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download, translate, and process PTB-XL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("DATA_DIR", "data/ptbxl"),
        help="Destination for processed PTB-XL data.",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get("CACHE_DIR", "cache"),
        help="Destination for BERT embeddings and ECG stats.",
    )
    parser.add_argument(
        "--bert-device",
        default="cpu",
        help="Device for BioClinicalBERT encoding.",
    )
    parser.add_argument(
        "--translation-model",
        default="claude-haiku-4-5-20251001",
        help="Anthropic model for report normalisation.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip Kaggle download; read CSV and waveforms from --data-dir directly.",
    )
    parser.add_argument(
        "--skip-translation",
        action="store_true",
        help="Skip Batch API translation; use existing report_en in --data-dir CSV.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    load_dotenv()

    args = _parse_args()
    run(
        data_dir=Path(args.data_dir),
        cache_dir=Path(args.cache_dir),
        bert_device=args.bert_device,
        translation_model=args.translation_model,
        skip_download=args.skip_download,
        skip_translation=args.skip_translation,
    )
