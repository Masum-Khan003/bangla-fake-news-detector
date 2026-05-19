"""
Preprocessing pipeline — BanFakeNews → clean, deduplicated,
labelled DataFrame ready for temporal splitting.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from tqdm import tqdm

from src.utils.text_utils  import clean_text, text_hash, bangla_char_ratio
from src.utils.label_map   import raw_to_int, INT_TO_LABEL
from src.model.chunker     import get_training_text, needs_chunking
from src.model.language_detect import detect_language

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_banfakenews(raw_dir: Path) -> pd.DataFrame:
    """Load and concatenate all BanFakeNews CSVs."""
    csv_files = list(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files in {raw_dir}")

    dfs = []
    for f in csv_files:
        df_tmp = pd.read_csv(f, low_memory=False)
        df_tmp.columns = [c.lower().strip() for c in df_tmp.columns]
        df_tmp["source_file"] = f.name
        dfs.append(df_tmp)
        log.info(f"Loaded {f.name}: {len(df_tmp):,} rows")

    df = pd.concat(dfs, ignore_index=True)
    log.info(f"Total raw rows: {len(df):,}")
    return df


def preprocess_banfakenews(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline:
    1. Standardize columns
    2. Clean text
    3. Map labels
    4. Detect language
    5. Deduplicate
    6. Add metadata columns
    """
    # ── 1. Identify key columns ───────────────────────────
    # BanFakeNews columns: 'articleid','domain','date','category',
    #                      'source','headline','content','label'
    text_col  = "content"
    label_col = "label"
    head_col  = "headline" if "headline" in df.columns else None
    date_col  = "date"     if "date"     in df.columns else None

    log.info(f"Columns: {df.columns.tolist()}")
    log.info(f"Text col: {text_col}, Label col: {label_col}")

    # ── 2. Drop nulls in text or label ────────────────────
    before = len(df)
    df = df.dropna(subset=[text_col, label_col])
    log.info(f"Dropped {before - len(df):,} null-text/label rows")

    # ── 3. Clean text ─────────────────────────────────────
    log.info("Cleaning text...")
    tqdm.pandas(desc="Cleaning")
    df["clean_content"] = df[text_col].progress_apply(
        lambda x: clean_text(str(x))
    )

    if head_col:
        df["clean_headline"] = df[head_col].fillna("").apply(
            lambda x: clean_text(str(x))
        )
    else:
        df["clean_headline"] = ""

    # Training text (headline + first 512 tokens worth)
    df["train_text"] = df.apply(
        lambda r: get_training_text(r["clean_headline"], r["clean_content"]),
        axis=1
    )

    # ── 4. Map labels ─────────────────────────────────────
    df["label_int"] = df[label_col].apply(raw_to_int)
    df["label_str"] = df["label_int"].map(INT_TO_LABEL)

    # ── 5. Language detection (sample 20% for speed) ──────
    log.info("Detecting language (sampling 20%)...")
    sample_mask = np.random.rand(len(df)) < 0.20
    lang_results = df.loc[sample_mask, "clean_content"].apply(detect_language)
    df.loc[sample_mask, "detected_lang"]   = lang_results.apply(lambda r: r.lang)
    df.loc[sample_mask, "model_branch"]    = lang_results.apply(lambda r: r.model_branch)
    df.loc[sample_mask, "lang_uncertain"]  = lang_results.apply(lambda r: r.is_uncertain)
    df["bangla_ratio"] = df["clean_content"].apply(bangla_char_ratio)

    # ── 6. Metadata columns ───────────────────────────────
    df["char_len"]      = df["clean_content"].str.len()
    df["approx_tokens"] = (df["char_len"] / 2.5).astype(int)
    df["needs_chunking"] = df["char_len"] > 1280
    df["text_hash"]     = df["clean_content"].apply(text_hash)

    # Parse dates if available
    if date_col and date_col in df.columns:
        df["pub_date"] = pd.to_datetime(df[date_col], errors="coerce")
    else:
        df["pub_date"] = pd.NaT
        log.warning("No date column found — temporal split will use index order")

    # ── 7. Deduplicate by text hash ───────────────────────
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["text_hash"], keep="first")
    log.info(f"Removed {before_dedup - len(df):,} duplicates")

    # ── 8. Select output columns ──────────────────────────
    output_cols = [
        "label_int", "label_str", "clean_headline",
        "clean_content", "train_text",
        "char_len", "approx_tokens", "needs_chunking",
        "bangla_ratio", "text_hash", "pub_date",
    ]
    # Keep optional columns if they exist
    for col in ["domain", "category", "source", "detected_lang",
                "model_branch", "lang_uncertain", "source_file"]:
        if col in df.columns:
            output_cols.append(col)

    df = df[output_cols].reset_index(drop=True)

    log.info(f"Final clean rows: {len(df):,}")
    log.info(f"Label distribution:\n{df['label_str'].value_counts()}")
    log.info(f"Needs chunking: {df['needs_chunking'].sum():,} "
             f"({df['needs_chunking'].mean()*100:.1f}%)")

    return df


def save_processed(df: pd.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "banfakenews_clean.parquet"
    df.to_parquet(out_path, index=False)
    log.info(f"Saved {len(df):,} rows → {out_path}")
    return out_path


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    raw_dir      = PROJECT_ROOT / "data" / "raw" / "banfakenews"
    proc_dir     = PROJECT_ROOT / "data" / "processed"

    raw_df  = load_banfakenews(raw_dir)
    proc_df = preprocess_banfakenews(raw_df)
    out     = save_processed(proc_df, proc_dir)

    # Quick sanity check
    print("\n── Final DataFrame Info ──────────────────────────")
    print(proc_df.info())
    print("\nSample row:")
    print(proc_df[proc_df["label_int"] == 0].iloc[0][
        ["label_str","clean_headline","approx_tokens","needs_chunking","bangla_ratio"]
    ])
