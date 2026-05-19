"""
Temporal train/validation/test split.

Primary strategy  : sort by publication date, split chronologically.
Fallback strategy : stratified split (preserves class ratios).

Why stratified fallback instead of index-based:
  BanFakeNews stores fake and authentic articles in separate CSV files.
  After concatenation, fake samples cluster at rows 7K-10K.
  Index-based 70/15/15 puts ALL fakes in train → val/test have zero fakes.
  Stratified split guarantees fake samples in every split.

NOTE: Pure random stratified split technically allows data leakage
(articles from the same event in train+test). This is acceptable here
because we have no reliable per-article dates to do better.
We document this limitation explicitly in the model card.
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)

TRAIN_CUTOFF = "2022-01-01"
VAL_CUTOFF   = "2024-01-01"
RANDOM_STATE = 42


def _check_split_quality(splits: dict) -> bool:
    """
    Returns True if every split has at least 10 fake samples.
    """
    for name, df in splits.items():
        n_fake = (df["label_int"] == 0).sum()
        if n_fake < 10:
            log.warning(f"Split '{name}' has only {n_fake} fake samples.")
            return False
    return True


def temporal_split(df: pd.DataFrame,
                   date_col: str = "pub_date") -> dict[str, pd.DataFrame]:
    """
    Split df — tries temporal first, falls back to stratified.
    Returns {"train": df, "val": df, "test": df}
    """

    # ── Attempt 1: Date-based temporal split ─────────────
    has_dates = (
        date_col in df.columns
        and df[date_col].notna().sum() > len(df) * 0.30
    )

    if has_dates:
        log.info("Attempting date-based temporal split...")
        df_sorted = df.sort_values(date_col).reset_index(drop=True)

        train_mask = df_sorted[date_col] < TRAIN_CUTOFF
        val_mask   = ((df_sorted[date_col] >= TRAIN_CUTOFF) &
                      (df_sorted[date_col] <  VAL_CUTOFF))
        test_mask  = df_sorted[date_col] >= VAL_CUTOFF

        splits = {
            "train": df_sorted[train_mask].reset_index(drop=True),
            "val":   df_sorted[val_mask].reset_index(drop=True),
            "test":  df_sorted[test_mask].reset_index(drop=True),
        }

        if _check_split_quality(splits):
            log.info("✓ Date-based temporal split OK.")
            _log_split_stats(splits)
            return splits
        else:
            log.warning("Date-based split failed quality check — "
                        "falling back to stratified split.")

    # ── Attempt 2: Stratified split ───────────────────────
    log.info("Using stratified split (preserves class ratios across splits).")
    log.warning(
        "Stratified split cannot guarantee temporal ordering. "
        "This is a known limitation documented in the model card."
    )

    # Step 1: train (70%) vs temp (30%)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=df["label_int"],
    )

    # Step 2: val (15%) vs test (15%) from the 30% temp
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temp_df["label_int"],
    )

    splits = {
        "train": train_df.reset_index(drop=True),
        "val":   val_df.reset_index(drop=True),
        "test":  test_df.reset_index(drop=True),
    }

    if not _check_split_quality(splits):
        raise RuntimeError(
            "Stratified split also failed — dataset may be too small "
            "or severely imbalanced. Check label distribution."
        )

    log.info("✓ Stratified split OK.")
    _log_split_stats(splits)
    return splits


def _log_split_stats(splits: dict) -> None:
    for name, df in splits.items():
        n_fake = int((df["label_int"] == 0).sum())
        n_cred = int((df["label_int"] == 1).sum())
        fake_pct = round(n_fake / len(df) * 100, 2)
        log.info(
            f"{name:6s}: {len(df):,} rows | "
            f"Fake: {n_fake:,} ({fake_pct}%) | "
            f"Credible: {n_cred:,}"
        )


def save_splits(splits: dict, out_dir: Path) -> dict[str, Path]:
    """Save each split as parquet and return paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    for name, df in splits.items():
        p = out_dir / f"{name}.parquet"
        df.to_parquet(p, index=False)
        paths[name] = p
        log.info(f"Saved {name} → {p}")

    # Metadata JSON — tracked by git (not DVC) for reproducibility
    meta = {
        split: {
            "n_rows":              len(df),
            "n_fake":              int((df["label_int"] == 0).sum()),
            "n_credible":          int((df["label_int"] == 1).sum()),
            "fake_pct":            float(round(
                                       (df["label_int"] == 0).mean() * 100, 2)),
            "needs_chunking_pct":  float(round(
                                       df["needs_chunking"].mean() * 100, 2))
                                   if "needs_chunking" in df.columns else None,
        }
        for split, df in splits.items()
    }

    meta_path = out_dir / "split_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info(f"Metadata → {meta_path}")

    return paths


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    proc_dir     = PROJECT_ROOT / "data" / "processed"
    split_dir    = PROJECT_ROOT / "data" / "splits"

    df     = pd.read_parquet(proc_dir / "banfakenews_clean.parquet")
    splits = temporal_split(df)
    paths  = save_splits(splits, split_dir)

    print("\n── Split Summary ─────────────────────────────────")
    meta = json.load(open(split_dir / "split_metadata.json"))
    for name, m in meta.items():
        print(f"{name:6s}: {m['n_rows']:,} rows | "
              f"Fake: {m['n_fake']:,} ({m['fake_pct']}%) | "
              f"Credible: {m['n_credible']:,} | "
              f"Needs chunking: {m['needs_chunking_pct']}%")
