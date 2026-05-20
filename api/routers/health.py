"""
GET /health/ready — Railway health check endpoint.
Returns 503 while model is loading, 200 when ready.
This is the ONLY endpoint Railway pings to decide if the
container is healthy. It must block until model is loaded.
"""

import time
from fastapi import APIRouter, Response
from api.schemas import HealthResponse

router     = APIRouter()
_start_time = time.time()


@router.get("/health/ready", response_model=HealthResponse, tags=["Health"])
async def health_ready(response: Response):
    """
    Returns 200 + model info when ready.
    Returns 503 while model is still loading.
    Railway health check must point here.
    """
    # Import here to avoid circular imports
    from api.main import predictor

    if not predictor._loaded:
        response.status_code = 503
        return HealthResponse(
            status         = "loading",
            model_loaded   = False,
            model_version  = "unknown",
            uptime_seconds = round(time.time() - _start_time, 1),
            temperature    = 0.0,
            threshold_fake = 0.0,
        )

    return HealthResponse(
        status         = "ready",
        model_loaded   = True,
        model_version  = "maksays-003/bangla-fake-news-xlmr",
        uptime_seconds = round(time.time() - _start_time, 1),
        temperature    = predictor.temperature,
        threshold_fake = predictor.thresh_fake,
    )
