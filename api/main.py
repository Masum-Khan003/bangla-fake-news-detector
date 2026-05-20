"""
FastAPI application entry point.

Startup sequence (critical):
  1. Load model into memory (30–90s)
  2. Run warm-up inference
  3. Connect to Redis cache
  4. Mark /health/ready as 200

Railway health check MUST point to /health/ready.
The default check timeout must be set to ≥120s in Railway settings.
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from src.model.predict import FakeNewsPredictor
from api.middleware import limiter, add_cors, log_requests
from api.cache import cache
from api.routers import predict, batch, feedback, health

# ── Logging ───────────────────────────────────────────────
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger(__name__)

# ── Global predictor singleton ────────────────────────────
# Imported by all routers via: from api.main import predictor
predictor = FakeNewsPredictor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: load model + connect cache.
    Shutdown: close cache connections.
    Railway health check returns 503 until this completes.
    """
    log.info("=" * 55)
    log.info("Starting Bangla Fake News Detector API")
    log.info("=" * 55)

    # Load model (blocking — intentional, must finish before serving)
    log.info("Loading model — /health/ready returns 503 until complete")
    predictor.load()

    # Connect cache
    await cache.connect()

    log.info("✓ Startup complete — /health/ready now returns 200")
    yield

    # Shutdown
    log.info("Shutting down...")
    await cache.close()


# ── App factory ───────────────────────────────────────────
app = FastAPI(
    title       = "Bangla Fake News Detector API",
    description = (
        "Detects fake news in Bangla and English using a fine-tuned "
        "XLM-RoBERTa model. All prediction endpoints require X-API-Key header."
    ),
    version     = "1.0.0",
    lifespan    = lifespan,
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

# ── Middleware ────────────────────────────────────────────
add_cors(app)
app.middleware("http")(log_requests)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── Routers ───────────────────────────────────────────────
app.include_router(health.router)
app.include_router(predict.router)
app.include_router(batch.router)
app.include_router(feedback.router)


@app.get("/", include_in_schema=False)
async def root():
    return {
        "name":    "Bangla Fake News Detector",
        "version": "1.0.0",
        "docs":    "/docs",
        "health":  "/health/ready",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host    = "0.0.0.0",
        port    = int(os.getenv("PORT", 8000)),
        reload  = False,
        workers = 1,    # single worker — model is too large to fork
    )
