"""
Middleware: CORS, request logging, rate limiting setup.
"""

import time
import logging
import uuid
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address

log = logging.getLogger(__name__)

# Rate limiter — keyed by IP (upgrade to API key in production)
limiter = Limiter(key_func=get_remote_address)


def add_cors(app: FastAPI) -> None:
    app.add_middleware(
        CORSMiddleware,
        allow_origins     = ["*"],   # tighten in production
        allow_credentials = False,
        allow_methods     = ["GET", "POST"],
        allow_headers     = ["*"],
    )


async def log_requests(request: Request, call_next):
    """Log every request with timing and request ID."""
    request_id = str(uuid.uuid4())[:8]
    start      = time.perf_counter()

    # Attach request_id so routes can read it
    request.state.request_id = request_id

    response = await call_next(request)

    ms = (time.perf_counter() - start) * 1000
    log.info(
        f"[{request_id}] {request.method} {request.url.path} "
        f"→ {response.status_code} ({ms:.0f}ms)"
    )
    response.headers["X-Request-ID"] = request_id
    return response
