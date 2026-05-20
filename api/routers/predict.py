"""
POST /v1/predict      — single text prediction
POST /v1/predict-url  — scrape URL then predict
"""

import time
import uuid
import logging
from fastapi import APIRouter, Depends, Request, HTTPException

from api.schemas import (PredictRequest, URLRequest,
                          PredictionResponse, URLPredictionResponse)
from api.auth   import require_api_key
from api.middleware import limiter

log    = APIRouter()
router = APIRouter(prefix="/v1", tags=["Prediction"])
logger = logging.getLogger(__name__)


def _build_response(result, request_id: str,
                    inference_ms: float) -> dict:
    from api.main import predictor
    return dict(
        request_id        = request_id,
        label             = result.label,
        label_int         = result.label_int,
        confidence        = result.confidence,
        fake_prob         = result.fake_prob,
        credible_prob     = result.credible_prob,
        language_detected = result.language,
        model_branch      = result.model_branch,
        is_uncertain_lang = result.is_uncertain_lang,
        token_count       = result.token_count,
        was_chunked       = result.was_chunked,
        n_chunks          = result.n_chunks,
        threshold_applied = predictor.thresh_fake,
        inference_ms      = round(inference_ms, 1),
        model_version     = "maksays-003/bangla-fake-news-xlmr",
    )


@router.post("/predict", response_model=PredictionResponse)
@limiter.limit("60/minute")
async def predict_text(
    request:  Request,
    body:     PredictRequest,
    _api_key: str = Depends(require_api_key),
):
    """
    Classify a single text as Fake or Credible.
    Runs inference in a thread pool — never blocks the event loop.
    """
    import asyncio
    from api.main import predictor

    request_id = getattr(request.state, "request_id", str(uuid.uuid4())[:8])

    t0 = time.perf_counter()
    try:
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, predictor.predict, body.text
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"[{request_id}] Inference error: {e}")
        raise HTTPException(status_code=500, detail="Inference failed")

    ms = (time.perf_counter() - t0) * 1000
    return PredictionResponse(**_build_response(result, request_id, ms))


@router.post("/predict-url", response_model=URLPredictionResponse)
@limiter.limit("30/minute")
async def predict_url(
    request:  Request,
    body:     URLRequest,
    _api_key: str = Depends(require_api_key),
):
    """
    Scrape a URL and classify the article.
    Checks Redis cache first — cached results return instantly.
    """
    import asyncio
    from urllib.parse import urlparse
    from api.main import predictor
    from api.cache import cache

    request_id = getattr(request.state, "request_id", str(uuid.uuid4())[:8])

    # Check cache
    cached = await cache.get(body.url)
    if cached:
        logger.info(f"[{request_id}] Cache hit: {body.url}")
        cached["cached"] = True
        return URLPredictionResponse(**cached)

    # Scrape
    t0 = time.perf_counter()
    try:
        import trafilatura
        html = trafilatura.fetch_url(body.url)
        text = trafilatura.extract(html) if html else None
        method = "trafilatura"

        if not text or len(text) < 50:
            try:
                from newspaper import Article
                art = Article(body.url)
                art.download(); art.parse()
                text   = art.text
                method = "newspaper4k"
            except Exception:
                text = None

        if not text or len(text) < 50:
            raise HTTPException(
                status_code = 422,
                detail      = "Could not extract article text from URL. "
                              "The site may block scrapers or require JavaScript."
            )

        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, predictor.predict, text
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] URL prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    ms     = (time.perf_counter() - t0) * 1000
    domain = urlparse(body.url).netloc

    response_data = dict(
        **_build_response(result, request_id, ms),
        url           = body.url,
        article_title = None,
        domain        = domain,
        scrape_method = method,
        cached        = False,
    )

    # Store in cache
    await cache.set(body.url, response_data)

    return URLPredictionResponse(**response_data)
