"""
POST /v1/predict/batch — classify up to 32 texts at once.
"""

import time
import uuid
import logging
import asyncio
from fastapi import APIRouter, Depends, Request, HTTPException

from api.schemas      import BatchRequest, BatchPredictionResponse, PredictionResponse
from api.auth         import require_api_key
from api.middleware   import limiter

router = APIRouter(prefix="/v1", tags=["Batch"])
logger = logging.getLogger(__name__)


@router.post("/predict/batch", response_model=BatchPredictionResponse)
@limiter.limit("10/minute")
async def predict_batch(
    request:  Request,
    body:     BatchRequest,
    _api_key: str = Depends(require_api_key),
):
    from api.main import predictor

    request_id = getattr(request.state, "request_id", str(uuid.uuid4())[:8])
    logger.info(f"[{request_id}] Batch request: {len(body.texts)} texts")

    results = []
    failed  = 0

    loop = asyncio.get_event_loop()

    for i, text in enumerate(body.texts):
        t0 = time.perf_counter()
        try:
            result = await loop.run_in_executor(
                None, predictor.predict, text
            )
            ms = (time.perf_counter() - t0) * 1000
            results.append(PredictionResponse(
                request_id        = f"{request_id}-{i}",
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
                inference_ms      = round(ms, 1),
                model_version     = "maksays-003/bangla-fake-news-xlmr",
            ))
        except Exception as e:
            logger.warning(f"[{request_id}] Item {i} failed: {e}")
            failed += 1

    return BatchPredictionResponse(
        results = results,
        total   = len(body.texts),
        failed  = failed,
    )
