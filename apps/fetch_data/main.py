"""Fetch, translate, and process PTB-XL data.

Pipeline
--------
1. Download PTB-XL from Kaggle (khyeh0719/ptb-xl-dataset).
2. Extract into a staging directory.
3. Load ptbxl_database.csv and detect the language of each report.
4. Submit German reports to the Anthropic Batch API (claude-haiku-3-5) for
   translation; skip records with an empty report.
5. Write a new ``report_en`` column to ptbxl_database.csv.
6. Compute per-lead normalisation statistics from the training split.
7. Build the BioClinicalBERT embedding cache.
8. Delete the raw Kaggle download zip.

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
                                   [--translation-model claude-haiku-3-5-20241022]
"""

from __future__ import annotations

import argparse
import os
import pickle
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

import anthropic
import langdetect
import pandas
from dotenv import load_dotenv

import gdss_multimodal.data as data_module


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

def _detect_language(text: str) -> str:
    """Return ISO-639-1 language code, or 'unknown' on failure."""
    try:
        return langdetect.detect(text)
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Kaggle download
# ---------------------------------------------------------------------------

def _download_kaggle(staging_dir: Path) -> Path:
    """Download PTB-XL from Kaggle and return the path to the extracted directory."""
    staging_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading PTB-XL from Kaggle…")
    env = os.environ.copy()
    subprocess.run(
        [
            "kaggle", "datasets", "download",
            "-d", "khyeh0719/ptb-xl-dataset",
            "-p", str(staging_dir),
            "--unzip",
        ],
        env=env,
        check=True,
    )

    # After --unzip the files are extracted in place; find the root CSV.
    candidates = list(staging_dir.rglob("ptbxl_database.csv"))
    if not candidates:
        raise FileNotFoundError(
            "ptbxl_database.csv not found after extraction. "
            f"Contents of {staging_dir}: {list(staging_dir.iterdir())}"
        )
    return candidates[0].parent


# ---------------------------------------------------------------------------
# Translation via Anthropic Batch API
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a medical translator. Translate the following German cardiology "
    "ECG report to English. Output only the translated text with no preamble, "
    "no explanation, and no quotation marks."
)


def _translate_batch(
    texts: list[str],
    ecg_ids: list[int],
    model: str = "claude-haiku-3-5-20241022",
) -> dict[int, str]:
    """Submit a batch translation job and poll until complete.

    Args:
        texts: German report strings.
        ecg_ids: Corresponding ECG IDs (used as custom_id).
        model: Anthropic model to use for translation.

    Returns:
        Dict mapping ecg_id → translated English text.
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

    print(f"  Submitting batch of {len(requests)} translation requests…")
    batch = client.messages.batches.create(requests=requests)
    batch_id = batch.id
    print(f"  Batch ID: {batch_id}")

    # Poll until complete
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
        time.sleep(30)

    # Collect results
    translations: dict[int, str] = {}
    for result in client.messages.batches.results(batch_id):
        eid = int(result.custom_id)
        if result.result.type == "succeeded":
            translations[eid] = result.result.message.content[0].text.strip()
        else:
            translations[eid] = ""   # errored — will be skipped later as empty

    print(f"  Translation complete: {len(translations)} results.")
    return translations


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(
    data_dir: Path,
    cache_dir: Path,
    bert_device: str = "cpu",
    translation_model: str = "claude-haiku-3-5-20241022",
    staging_dir: Path | None = None,
) -> None:
    """Run the full fetch-and-process pipeline.

    Args:
        data_dir: Destination for processed PTB-XL data.
        cache_dir: Destination for BERT embeddings and ECG stats.
        bert_device: Device string for BioClinicalBERT encoding.
        translation_model: Anthropic model for German → English translation.
        staging_dir: Temporary directory for the Kaggle download.
    """
    if staging_dir is None:
        staging_dir = data_dir.parent / "_ptbxl_staging"

    # ── 1. Download ──────────────────────────────────────────────────────────
    raw_dir = _download_kaggle(staging_dir)
    print(f"Extracted to: {raw_dir}")

    # ── 2. Load CSV ──────────────────────────────────────────────────────────
    csv_path = raw_dir / data_module.PTBXL_CSV
    df = pandas.read_csv(csv_path, index_col="ecg_id")
    print(f"Loaded {len(df)} records from {csv_path}")

    # ── 3. Language detection ────────────────────────────────────────────────
    print("Detecting report languages…")
    reports = df.get("report", pandas.Series(dtype=str)).fillna("").astype(str)

    german_ids, german_texts = [], []
    english_ids, english_texts = [], []
    skipped_ids = []

    for ecg_id, text in reports.items():
        text = text.strip()
        if not text or text.lower() in ("nan", "none"):
            skipped_ids.append(ecg_id)
            continue
        lang = _detect_language(text)
        if lang == "de":
            german_ids.append(ecg_id)
            german_texts.append(text)
        else:
            # Treat anything non-German as usable English (or close enough)
            english_ids.append(ecg_id)
            english_texts.append(text)

    print(
        f"  German: {len(german_ids)}  "
        f"Other/English: {len(english_ids)}  "
        f"Empty (skipped): {len(skipped_ids)}"
    )

    # ── 4. Translate German reports ──────────────────────────────────────────
    translations: dict[int, str] = {}
    if german_ids:
        translations = _translate_batch(german_texts, german_ids, model=translation_model)

    # ── 5. Write report_en column ────────────────────────────────────────────
    report_en: dict[int, str] = {}
    for eid, text in zip(english_ids, english_texts):
        report_en[eid] = text
    for eid, text in translations.items():
        report_en[eid] = text
    # Skipped records get an empty string → will be filtered by data.py

    df["report_en"] = pandas.Series(report_en)
    df["report_en"] = df["report_en"].fillna("")

    n_usable = (df["report_en"].str.strip() != "").sum()
    print(f"  Usable records with report_en: {n_usable}/{len(df)}")

    # ── 6. Copy processed data to DATA_DIR ───────────────────────────────────
    data_dir.mkdir(parents=True, exist_ok=True)
    dest_csv = data_dir / data_module.PTBXL_CSV
    df.to_csv(dest_csv)
    print(f"  Saved updated CSV → {dest_csv}")

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

    # ── 9. Delete staging directory ──────────────────────────────────────────
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
        default="claude-haiku-3-5-20241022",
        help="Anthropic model for German → English translation.",
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
    )
