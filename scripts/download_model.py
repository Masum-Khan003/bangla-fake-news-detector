"""
Downloads the fine-tuned model from HuggingFace Hub at container startup.
Skips download if model already exists (e.g. Railway volume persistence).

Why download at runtime instead of baking into image:
  - Model is ~1.1GB — would push image over Railway's 2GB limit
  - Allows model updates without rebuilding the image
  - HF Hub caches to /app/models/ which can be a mounted volume
"""

import os
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MODEL_DIR = Path("/app/models/xlmr-bangla-fn")
HF_REPO   = os.getenv("HF_MODEL_REPO", "maksays-003/bangla-fake-news-xlmr")
HF_TOKEN  = os.getenv("HUGGINGFACE_TOKEN", "")

REQUIRED_FILES = ["config.json", "tokenizer.json", "model.safetensors"]


def model_exists() -> bool:
    return all((MODEL_DIR / f).exists() for f in REQUIRED_FILES)


def download():
    log.info(f"Downloading model: {HF_REPO} → {MODEL_DIR}")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id         = HF_REPO,
            local_dir       = str(MODEL_DIR),
            token           = HF_TOKEN or None,
            ignore_patterns = ["*.msgpack", "flax_model*",
                               "tf_model*", "rust_model*"],
        )
        log.info("✓ Model downloaded successfully")
    except Exception as e:
        log.error(f"Model download failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    if model_exists():
        log.info(f"✓ Model already exists at {MODEL_DIR} — skipping download")
    else:
        download()
