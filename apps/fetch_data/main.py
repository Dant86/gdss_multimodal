"""Fetch, translate, and process PTB-XL data.

Pipeline
--------
1. Download PTB-XL from Kaggle (khyeh0719/ptb-xl-dataset).
2. Extract into a staging directory.
3. Load ptbxl_database.csv; skip records with an empty report.
4. Send all non-empty reports to the Anthropic Messages API concurrently
   for normalisation: Claude returns the text in English regardless of
   source language.
5. Write a new ``report_en`` column to ptbxl_database.csv.
6. Compute per-lead normalisation statistics from the training split.
7. Build the BioClinicalBERT embedding cache.
8. Delete the staging directory.

Environment variables (loaded from .env):
    DATA_DIR           Destination for the processed PTB-XL directory.
    CACHE_DIR          Destination for BERT embeddings and ECG stats.
    KAGGLE_USERNAME    Kaggle API credentials.
    KAGGLE_KEY
    ANTHROPIC_API_KEY  For the translation requests.
    HF_TOKEN           Optional HuggingFace token.

Usage
-----
    python apps/fetch_data/main.py [--data-dir DIR] [--cache-dir DIR]
                                   [--bert-device cpu|cuda]
                                   [--translation-model claude-haiku-4-5-20251001]
                                   [--concurrency 50]
"""

from __future__ import annotations

import argparse
import asyncio
import os
import pickle
import shutil
import subprocess
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
# Normalisation via concurrent async Messages API calls
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a medical language specialist. The following is a cardiology ECG "
    "report. It may be written in German or English. If it is in German, "
    "translate it to English. If it is already in English, return it unchanged. "
    "Output only the final English text with no preamble, no explanation, and "
    "no quotation marks."
)

_MAX_RETRIES = 5
_RETRY_BASE_DELAY = 2.0  # seconds; doubled on each retry


async def _normalise_one(
    client: anthropic.AsyncAnthropic,
    sem: asyncio.Semaphore,
    ecg_id: int,
    text: str,
    model: str,
) -> tuple[int, str]:
    """Translate/normalise a single report, retrying on rate-limit errors."""
    delay = _RETRY_BASE_DELAY
    for attempt in range(_MAX_RETRIES):
        try:
            async with sem:
                msg = await client.messages.create(
                    model=model,
                    max_tokens=512,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": text}],
                )
            return ecg_id, msg.content[0].text.strip()
        except anthropic.RateLimitError:
            if attempt == _MAX_RETRIES - 1:
                return ecg_id, ""
            await asyncio.sleep(delay)
            delay *= 2
        except Exception as exc:  # noqa: BLE001
            return ecg_id, f"__error__: {exc}"
    return ecg_id, ""


async def _normalise_async(
    texts: list[str],
    ecg_ids: list[int],
    model: str,
    concurrency: int,
) -> dict[int, str]:
    """Run all normalisation requests concurrently and return results."""
    client = anthropic.AsyncAnthropic()
    sem = asyncio.Semaphore(concurrency)

    tasks = [
        _normalise_one(client, sem, eid, text, model)
        for eid, text in zip(ecg_ids, texts)
    ]

    results: dict[int, str] = {}
    errors: list[str] = []
    total = len(tasks)
    done = 0

    for coro in asyncio.as_completed(tasks):
        eid, result_text = await coro
        results[eid] = result_text
        done += 1
        if not result_text or result_text.startswith("__error__"):
            if len(errors) < 5:
                errors.append(f"  ecg_id={eid}: {result_text!r}")
        if done % 500 == 0 or done == total:
            n_err = sum(1 for v in results.values() if not v or v.startswith("__error__"))
            print(f"  [{done}/{total}] done — {n_err} errors so far")

    n_err = sum(1 for v in results.values() if not v or v.startswith("__error__"))
    print(f"  Complete: {total} requests, {total - n_err} succeeded, {n_err} errored.")
    if errors:
        print("  Sample errors (first 5):")
        for e in errors:
            print(e)

    # Replace error sentinels with empty string for downstream use
    return {eid: ("" if (not t or t.startswith("__error__")) else t) for eid, t in results.items()}


def _normalise_batch(
    texts: list[str],
    ecg_ids: list[int],
    model: str = "claude-haiku-4-5-20251001",
    concurrency: int = 50,
) -> dict[int, str]:
    """Normalise all reports concurrently and return English text for each.

    Claude handles language detection internally — German reports are
    translated, English reports are returned unchanged.

    Args:
        texts: Report strings (any language).
        ecg_ids: Corresponding ECG IDs.
        model: Anthropic model to use.
        concurrency: Maximum simultaneous in-flight API requests.

    Returns:
        Dict mapping ecg_id → English report text.
    """
    print(f"  Normalising {len(texts)} reports (concurrency={concurrency})…")
    return asyncio.run(_normalise_async(texts, ecg_ids, model, concurrency))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(
    data_dir: Path,
    cache_dir: Path,
    bert_device: str = "cpu",
    translation_model: str = "claude-haiku-4-5-20251001",
    concurrency: int = 50,
    staging_dir: Path | None = None,
    skip_download: bool = False,
) -> None:
    """Run the full fetch-and-process pipeline.

    Args:
        data_dir: Destination for processed PTB-XL data.
        cache_dir: Destination for BERT embeddings and ECG stats.
        bert_device: Device string for BioClinicalBERT encoding.
        translation_model: Anthropic model for report normalisation.
        concurrency: Max simultaneous Anthropic API requests.
        staging_dir: Temporary directory for the Kaggle download.
        skip_download: If True, skip Kaggle download and read the CSV
            directly from data_dir (waveforms must already be in place).
    """
    if staging_dir is None:
        staging_dir = data_dir.parent / "_ptbxl_staging"

    # ── 1. Download ──────────────────────────────────────────────────────────
    if skip_download:
        print("Skipping Kaggle download — reading existing data from data_dir.")
        csv_path = data_dir / data_module.PTBXL_CSV
    else:
        raw_dir = _download_kaggle(staging_dir)
        print(f"Extracted to: {raw_dir}")
        csv_path = raw_dir / data_module.PTBXL_CSV

    # ── 2. Load CSV ──────────────────────────────────────────────────────────
    df = pandas.read_csv(csv_path, index_col="ecg_id")
    print(f"Loaded {len(df)} records from {csv_path}")

    # ── 3. Collect non-empty reports ─────────────────────────────────────────
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

    # ── 4. Normalise via concurrent API calls ────────────────────────────────
    normalised = _normalise_batch(
        batch_texts, batch_ids, model=translation_model, concurrency=concurrency
    )

    # ── 5. Write report_en column ────────────────────────────────────────────
    df["report_en"] = pandas.Series(normalised).reindex(df.index, fill_value="")
    n_usable = (df["report_en"].str.strip() != "").sum()
    print(f"  Usable records with report_en: {n_usable}/{len(df)}")

    # ── 6. Write CSV (and copy waveforms if downloaded) ─────────────────────
    data_dir.mkdir(parents=True, exist_ok=True)
    dest_csv = data_dir / data_module.PTBXL_CSV
    df.to_csv(dest_csv)
    print(f"  Saved updated CSV → {dest_csv}")

    if not skip_download:
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
    if not skip_download and staging_dir.exists():
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
        "--concurrency",
        type=int,
        default=50,
        help="Max simultaneous Anthropic API requests.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip Kaggle download; read CSV and waveforms from --data-dir directly.",
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
        concurrency=args.concurrency,
        skip_download=args.skip_download,
    )
