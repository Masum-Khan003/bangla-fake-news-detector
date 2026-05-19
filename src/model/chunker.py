"""
Article chunker — handles BERT's 512-token hard limit.

Strategy (from EDA):
  - 54.8% of articles exceed 512 tokens
  - Median article = 553 tokens
  - Fake signals are often front-loaded (headline + first paragraphs)

Training : use headline + first 512 tokens (fast, effective for news)
Inference: sliding window (384 tok, 64 overlap) → max-pool across chunks
"""

from __future__ import annotations
from dataclasses import dataclass, field


CHUNK_SIZE    = 384   # tokens per chunk (leaves room for special tokens)
OVERLAP       = 64    # overlapping tokens between consecutive chunks
MAX_SEQ_LEN   = 512   # hard BERT limit


@dataclass
class Chunk:
    text:       str
    chunk_idx:  int
    is_first:   bool
    char_start: int
    char_end:   int


def chunk_text_by_chars(text: str,
                         tokenizer=None,
                         chunk_size: int = CHUNK_SIZE,
                         overlap: int = OVERLAP) -> list[Chunk]:
    """
    Split text into overlapping chunks.

    If tokenizer is provided: split by tokens (accurate).
    If not (e.g., during preprocessing): split by estimated chars.

    Char estimate: 1 token ≈ 2.5 Bangla chars (from EDA).
    """
    if not text or not text.strip():
        return [Chunk(text="", chunk_idx=0, is_first=True,
                      char_start=0, char_end=0)]

    # ── Token-aware splitting (used during inference) ──────
    if tokenizer is not None:
        return _chunk_with_tokenizer(text, tokenizer, chunk_size, overlap)

    # ── Character-based estimate (used during preprocessing) ──
    chars_per_chunk   = int(chunk_size * 2.5)
    chars_per_overlap = int(overlap * 2.5)
    step              = chars_per_chunk - chars_per_overlap

    if len(text) <= chars_per_chunk:
        return [Chunk(text=text, chunk_idx=0, is_first=True,
                      char_start=0, char_end=len(text))]

    chunks = []
    start  = 0
    idx    = 0
    while start < len(text):
        end  = min(start + chars_per_chunk, len(text))
        chunk_text = text[start:end]
        chunks.append(Chunk(
            text       = chunk_text,
            chunk_idx  = idx,
            is_first   = (idx == 0),
            char_start = start,
            char_end   = end,
        ))
        if end == len(text):
            break
        start += step
        idx   += 1

    return chunks


def _chunk_with_tokenizer(text: str, tokenizer,
                           chunk_size: int, overlap: int) -> list[Chunk]:
    """Token-accurate chunking using a HuggingFace tokenizer."""
    # Tokenize without special tokens first
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    step      = chunk_size - overlap

    if len(token_ids) <= chunk_size:
        return [Chunk(text=text, chunk_idx=0, is_first=True,
                      char_start=0, char_end=len(text))]

    chunks = []
    for idx, start in enumerate(range(0, len(token_ids), step)):
        end       = min(start + chunk_size, len(token_ids))
        chunk_ids = token_ids[start:end]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(Chunk(
            text       = chunk_text,
            chunk_idx  = idx,
            is_first   = (idx == 0),
            char_start = start,   # token index, not char
            char_end   = end,
        ))
        if end >= len(token_ids):
            break

    return chunks


def get_training_text(headline: str, body: str,
                       max_chars: int = 1280) -> str:
    """
    For TRAINING only — headline + first N chars of body.
    Fake signals are front-loaded; this is faster than chunking
    and avoids positional bias from chunk aggregation during training.
    ~1280 chars ≈ 512 tokens for Bangla.
    """
    headline = (headline or "").strip()
    body     = (body or "").strip()

    if headline:
        combined = f"{headline} । {body}"
    else:
        combined = body

    return combined[:max_chars]


def needs_chunking(text: str, char_threshold: int = 1280) -> bool:
    """Quick check — does this text need chunking?"""
    return len(text) > char_threshold


if __name__ == "__main__":
    sample_short = "সরকার ঘোষণা করেছে।" * 10
    sample_long  = "সরকার ঘোষণা করেছে যে সমস্ত বিদ্যালয় বন্ধ থাকবে।" * 60

    print(f"Short text ({len(sample_short)} chars):")
    short_chunks = chunk_text_by_chars(sample_short)
    print(f"  → {len(short_chunks)} chunk(s)\n")

    print(f"Long text ({len(sample_long)} chars):")
    long_chunks = chunk_text_by_chars(sample_long)
    print(f"  → {len(long_chunks)} chunks")
    for c in long_chunks:
        print(f"  chunk {c.chunk_idx}: chars {c.char_start}–{c.char_end} "
              f"({'first' if c.is_first else 'cont.'})")

    print(f"\nneeds_chunking(short): {needs_chunking(sample_short)}")
    print(f"needs_chunking(long):  {needs_chunking(sample_long)}")
