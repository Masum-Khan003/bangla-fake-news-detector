"""
Pydantic request/response schemas — all API contracts defined here.
Validation happens automatically before any route handler runs.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional
import re

# ── Strip HTML helper ─────────────────────────────────────
_HTML_RE = re.compile(r'<[^>]+>')

class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length  = 50,
        max_length  = 10_000,
        description = "Article text to classify. Min 50 chars, max 10,000.",
    )

    @field_validator("text")
    @classmethod
    def strip_html_and_clean(cls, v: str) -> str:
        v = _HTML_RE.sub(" ", v).strip()
        if not v:
            raise ValueError("Text is empty after HTML stripping")
        return v


class URLRequest(BaseModel):
    url: str = Field(
        ...,
        min_length  = 10,
        max_length  = 2048,
        description = "URL of the news article to scrape and classify.",
    )

    @field_validator("url")
    @classmethod
    def must_be_http(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return v


class BatchRequest(BaseModel):
    texts: list[str] = Field(
        ...,
        min_length  = 1,
        max_length  = 32,
        description = "List of texts to classify. Max 32 items.",
    )

    @field_validator("texts")
    @classmethod
    def validate_each(cls, texts: list[str]) -> list[str]:
        cleaned = []
        for i, t in enumerate(texts):
            t = _HTML_RE.sub(" ", t).strip()
            if len(t) < 20:
                raise ValueError(f"Text at index {i} is too short (min 20 chars)")
            if len(t) > 10_000:
                t = t[:10_000]
            cleaned.append(t)
        return cleaned


class FeedbackRequest(BaseModel):
    request_id:    str           = Field(..., min_length=8, max_length=64)
    correct_label: str           = Field(..., pattern="^(Fake|Credible|Unverified)$")
    comment:       Optional[str] = Field(None, max_length=500)


class PredictionResponse(BaseModel):
    request_id:        str
    label:             str
    label_int:         int
    confidence:        float
    fake_prob:         float
    credible_prob:     float
    language_detected: str
    model_branch:      str
    is_uncertain_lang: bool
    token_count:       int
    was_chunked:       bool
    n_chunks:          int
    threshold_applied: float
    inference_ms:      float
    model_version:     str
    warning:           Optional[str] = None   # shown when input may be unreliable


class URLPredictionResponse(PredictionResponse):
    url:          str
    article_title: Optional[str]
    domain:       str
    scrape_method: str
    cached:       bool


class BatchPredictionResponse(BaseModel):
    results:    list[PredictionResponse]
    total:      int
    failed:     int


class FeedbackResponse(BaseModel):
    accepted:   bool
    message:    str


class HealthResponse(BaseModel):
    status:       str   # "ready" | "loading" | "degraded"
    model_loaded: bool
    model_version: str
    uptime_seconds: float
    temperature:  float
    threshold_fake: float
