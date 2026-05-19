"""
Language detection — routes text to the correct model branch.

Bangla  (bn) → csebuetnlp/banglabert
English (en) → xlm-roberta-base  
Banglish     → xlm-roberta-base (flagged as uncertain)

Uses langdetect (fast, no model file needed) with a Bangla char
ratio fallback — because langdetect can fail on short Bangla text.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from langdetect import detect, LangDetectException
from src.utils.text_utils import bangla_char_ratio


# Thresholds
BANGLA_RATIO_HIGH    = 0.35   # clearly Bangla
BANGLA_RATIO_LOW     = 0.10   # mixed / Banglish
MIN_TEXT_LENGTH      = 20     # too short for reliable detection


class LanguageResult:
    __slots__ = ("lang", "model_branch", "bangla_ratio", "is_uncertain")

    def __init__(self, lang, model_branch, bangla_ratio, is_uncertain=False):
        self.lang          = lang
        self.model_branch  = model_branch
        self.bangla_ratio  = bangla_ratio
        self.is_uncertain  = is_uncertain

    def __repr__(self):
        return (f"LanguageResult(lang={self.lang!r}, "
                f"branch={self.model_branch!r}, "
                f"bn_ratio={self.bangla_ratio:.2f}, "
                f"uncertain={self.is_uncertain})")


def detect_language(text: str) -> LanguageResult:
    """
    Detect language and return which model branch to use.

    Returns:
        LanguageResult with .lang and .model_branch
        model_branch: "banglabert" | "xlm-roberta" | "xlm-roberta-uncertain"
    """
    if not text or len(text.strip()) < MIN_TEXT_LENGTH:
        return LanguageResult("unknown", "xlm-roberta", 0.0, is_uncertain=True)

    bn_ratio = bangla_char_ratio(text)

    # Strong Bangla signal — trust the char ratio over langdetect
    if bn_ratio >= BANGLA_RATIO_HIGH:
        return LanguageResult("bn", "banglabert", bn_ratio)

    # Mixed / Banglish
    if BANGLA_RATIO_LOW <= bn_ratio < BANGLA_RATIO_HIGH:
        return LanguageResult("banglish", "xlm-roberta", bn_ratio,
                              is_uncertain=True)

    # Low Bangla content — use langdetect
    try:
        detected = detect(text)
        branch = "banglabert" if detected == "bn" else "xlm-roberta"
        return LanguageResult(detected, branch, bn_ratio)
    except LangDetectException:
        return LanguageResult("unknown", "xlm-roberta", bn_ratio,
                              is_uncertain=True)


def batch_detect(texts: list[str]) -> list[LanguageResult]:
    return [detect_language(t) for t in texts]


if __name__ == "__main__":
    samples = [
        ("Pure Bangla",    "সরকার ঘোষণা করেছে যে সমস্ত বিদ্যালয় বন্ধ থাকবে।"),
        ("Pure English",   "The government announced that all schools will remain closed."),
        ("Banglish",       "Ami jani je government schools band korbe."),
        ("Short text",     "হ্যাঁ"),
        ("Mixed",          "Today সরকার announced নতুন policy."),
    ]
    print(f"{'Type':<15} {'Lang':<10} {'Branch':<22} {'BN%':<8} {'Uncertain'}")
    print("─" * 65)
    for name, text in samples:
        r = detect_language(text)
        print(f"{name:<15} {r.lang:<10} {r.model_branch:<22} "
              f"{r.bangla_ratio:<8.2%} {r.is_uncertain}")
