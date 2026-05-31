"""
Inference wrapper — loads the fine-tuned XLM-R model and runs
calibrated prediction with chunking support.

Used by the FastAPI routers. Model is loaded ONCE at startup,
never per-request.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import json
import logging
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from scipy.special import softmax

from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

from src.model.language_detect import detect_language
from src.model.chunker import chunk_text_by_chars, needs_chunking
from src.utils.text_utils import clean_text, truncate_to_chars
from src.utils.label_map import INT_TO_LABEL, INT_TO_COLOR

log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parents[2]
MODEL_DIR      = PROJECT_ROOT / "models" / "xlmr-bangla-fn"
CALIBRATION    = PROJECT_ROOT / "calibration"


@dataclass
class PredictionResult:
    label:             str       # "Fake" | "Credible"
    label_int:         int
    confidence:        float     # calibrated probability of predicted class
    fake_prob:         float     # calibrated probability of Fake class
    credible_prob:     float
    language:          str
    model_branch:      str
    is_uncertain_lang: bool
    token_count:       int
    was_chunked:       bool
    n_chunks:          int
    color:             str       # hex color for UI


class FakeNewsPredictor:
    """
    Singleton predictor — instantiate once, call predict() many times.
    Thread-safe for read-only inference.
    """

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir   = model_dir
        self.tokenizer   = None
        self.model       = None
        self.temperature = 1.0
        self.thresh_fake = 0.50
        # ONNX Runtime handles device selection automatically
        self._loaded = False

    def load(self) -> None:
        """
        Load model, tokenizer, and calibration config.
        Call this ONCE at API startup — takes 10–30s.
        """
        log.info(f"Loading model from {self.model_dir}")
        log.info("Device: CPU (ONNX Runtime)")

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.model     = ORTModelForSequenceClassification.from_pretrained(
            str(self.model_dir),
            file_name = "model_quantized.onnx",
        )

        # Load calibration config
        thresh_path = CALIBRATION / "thresholds.json"
        temp_path   = CALIBRATION / "temperature.json"

        if thresh_path.exists():
            config = json.loads(thresh_path.read_text())
            self.thresh_fake = config.get("threshold_fake", 0.05)
            log.info(f"Threshold (Fake): {self.thresh_fake}")

        if temp_path.exists():
            config = json.loads(temp_path.read_text())
            self.temperature = config.get("temperature", 0.5308)
            log.info(f"Temperature: {self.temperature}")

        # Warm-up pass — pre-JIT the model on dummy input
        log.info("Running warm-up inference...")
        dummy = self.tokenizer(
            "সরকার ঘোষণা করেছে।",
            return_tensors="pt",
            max_length=64,
            truncation=True,
            padding=True,
        )
        _ = self.model(**dummy)

        self._loaded = True
        log.info("✓ ONNX model loaded and warmed up")

    def _run_inference(self, text: str) -> np.ndarray:
        """Run tokenization + forward pass. Returns raw logits (1D)."""
        inputs = self.tokenizer(
            text,
            return_tensors  = "pt",
            max_length      = 512,
            truncation      = True,
            padding         = True,
        )
        outputs = self.model(**inputs)
        return outputs.logits[0].numpy()

    def _calibrate(self, logits: np.ndarray) -> tuple[float, float]:
        """
        Apply temperature scaling and return (fake_prob, credible_prob).
        """
        scaled = logits / self.temperature
        probs  = softmax(scaled)
        return float(probs[0]), float(probs[1])

    def predict(self, text: str, max_chars: int = 10_000) -> PredictionResult:
        """
        Full prediction pipeline:
        1. Clean + truncate input
        2. Detect language
        3. Chunk if needed
        4. Run inference (per chunk if chunked)
        5. Aggregate via max-pooling on fake_prob
        6. Apply calibrated threshold
        """
        if not self._loaded:
            raise RuntimeError("Call predictor.load() before predict()")

        # 1. Clean and truncate
        text = clean_text(truncate_to_chars(text, max_chars))
        if not text.strip():
            raise ValueError("Input text is empty after cleaning")

        # 2. Language detection
        lang_result = detect_language(text)

        # 3. Token count estimate
        approx_tokens = int(len(text) / 2.5)

        # 4. Chunking
        chunked    = needs_chunking(text)
        all_logits = []

        if chunked:
            chunks = chunk_text_by_chars(
                text,
                tokenizer  = self.tokenizer,
                chunk_size = 384,
                overlap    = 64,
            )
            for chunk in chunks:
                if chunk.text.strip():
                    logits = self._run_inference(chunk.text)
                    all_logits.append(logits)
        else:
            all_logits = [self._run_inference(text)]

        n_chunks = len(all_logits)

        # 5. Aggregate — max-pool on fake_prob across chunks
        #    (fake signals are front-loaded; we want the most alarming chunk)
        best_logits   = max(all_logits,
                            key=lambda l: softmax(l / self.temperature)[0])
        fake_p, cred_p = self._calibrate(best_logits)

        # 6. Apply threshold
        predicted_int = 0 if fake_p >= self.thresh_fake else 1
        confidence    = fake_p if predicted_int == 0 else cred_p

        return PredictionResult(
            label             = INT_TO_LABEL[predicted_int],
            label_int         = predicted_int,
            confidence        = round(confidence, 4),
            fake_prob         = round(fake_p, 4),
            credible_prob     = round(cred_p, 4),
            language          = lang_result.lang,
            model_branch      = lang_result.model_branch,
            is_uncertain_lang = lang_result.is_uncertain,
            token_count       = approx_tokens,
            was_chunked       = chunked,
            n_chunks          = n_chunks,
            color             = INT_TO_COLOR[predicted_int],
        )

    def predict_batch(self, texts: list[str]) -> list[PredictionResult]:
        """Run predict() on a list of texts. Used by /v1/predict/batch."""
        return [self.predict(t) for t in texts]


# ── Module-level singleton ────────────────────────────────
predictor = FakeNewsPredictor()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    print("Loading model...")
    predictor.load()

    test_cases = [
        ("Bangla fake",    "সরকার ঘোষণা করেছে সমস্ত বিদ্যালয় স্থায়ীভাবে বন্ধ হবে।"),
        ("Bangla credible","প্রধানমন্ত্রী আজ জাতীয় সংসদে বার্ষিক বাজেট পেশ করেছেন।"),
        ("English news",   "The government announced new economic policies today."),
        ("Banglish",       "Government school band korbe bole news ashche."),
    ]

    print("\n── Prediction Results ────────────────────────────────")
    for name, text in test_cases:
        r = predictor.predict(text)
        print(f"\n{name}:")
        print(f"  Label      : {r.label} ({r.confidence:.1%} confidence)")
        print(f"  Fake prob  : {r.fake_prob:.4f} (threshold: {predictor.thresh_fake})")
        print(f"  Language   : {r.language} → {r.model_branch}")
        print(f"  Chunked    : {r.was_chunked} ({r.n_chunks} chunk(s))")
