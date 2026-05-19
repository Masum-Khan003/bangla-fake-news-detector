"""
Text utilities — Bangla-specific cleaning and normalization.
"""

import re
import unicodedata
import hashlib


# ── Bangla Unicode ranges ─────────────────────────────────
BANGLA_RANGE = re.compile(r'[\u0980-\u09FF]')

# Patterns to strip
HTML_TAG        = re.compile(r'<[^>]+>')
MULTIPLE_SPACES = re.compile(r'\s+')
URL_PATTERN     = re.compile(r'https?://\S+|www\.\S+')
EMAIL_PATTERN   = re.compile(r'\S+@\S+\.\S+')
SPECIAL_CHARS   = re.compile(r'[^\u0980-\u09FF\u0020-\u007E\u00A0\n]')


def normalize_unicode(text: str) -> str:
    """Normalize to NFC — required for consistent Bangla tokenization."""
    return unicodedata.normalize("NFC", text)


def strip_html(text: str) -> str:
    return HTML_TAG.sub(" ", text)


def strip_urls(text: str) -> str:
    return URL_PATTERN.sub(" ", text)


def strip_emails(text: str) -> str:
    return EMAIL_PATTERN.sub(" ", text)


def collapse_whitespace(text: str) -> str:
    return MULTIPLE_SPACES.sub(" ", text).strip()


def bangla_char_ratio(text: str) -> float:
    """Fraction of characters that are Bangla Unicode."""
    if not text:
        return 0.0
    bangla_chars = len(BANGLA_RANGE.findall(text))
    return bangla_chars / len(text)


def clean_text(text: str,
               strip_html: bool = True,
               strip_urls: bool = True,
               normalize: bool = True) -> str:
    """
    Full cleaning pipeline for a single article text.
    Order matters — HTML first, then URLs, then normalize.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    if strip_html:
        text = HTML_TAG.sub(" ", text)
    if strip_urls:
        text = URL_PATTERN.sub(" ", text)
        text = EMAIL_PATTERN.sub(" ", text)
    if normalize:
        text = normalize_unicode(text)

    text = collapse_whitespace(text)
    return text


def text_hash(text: str) -> str:
    """SHA-256 of cleaned, lowercased text — for deduplication."""
    normalized = clean_text(text).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def truncate_to_chars(text: str, max_chars: int = 10_000) -> str:
    """Hard cap on input length — protects the API tokenizer."""
    return text[:max_chars] if len(text) > max_chars else text


def headline_plus_body(headline: str, body: str,
                       max_chars: int = 10_000) -> str:
    """
    Concatenate headline + body with separator.
    Headline gets priority — always included.
    """
    headline = clean_text(headline or "")
    body     = clean_text(body or "")
    combined = f"{headline} [SEP] {body}" if headline else body
    return truncate_to_chars(combined, max_chars)


if __name__ == "__main__":
    sample = "  <p>সরকার ঘোষণা করেছে।</p> Visit https://example.com  "
    cleaned = clean_text(sample)
    print(f"Original : {repr(sample)}")
    print(f"Cleaned  : {repr(cleaned)}")
    print(f"Bangla % : {bangla_char_ratio(cleaned):.2%}")
    print(f"Hash     : {text_hash(cleaned)[:16]}...")
