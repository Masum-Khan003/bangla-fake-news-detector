"""
API key authentication.
All prediction endpoints require X-API-Key header.
Keys are stored in .env as a comma-separated list.
"""

import os
import logging
from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader

log = logging.getLogger(__name__)

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def _load_valid_keys() -> set[str]:
    """Load API keys from environment. Supports multiple keys."""
    raw = os.getenv("API_SECRET_KEY", "")
    if not raw:
        log.warning("API_SECRET_KEY not set — all requests will be rejected")
        return set()
    keys = {k.strip() for k in raw.split(",") if k.strip()}
    log.info(f"Loaded {len(keys)} API key(s)")
    return keys


VALID_KEYS: set[str] = _load_valid_keys()


async def require_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    """
    FastAPI dependency — inject into any route that needs auth.
    Usage: async def my_route(key: str = Depends(require_api_key))
    """
    if not api_key:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail      = "Missing X-API-Key header",
            headers     = {"WWW-Authenticate": "ApiKey"},
        )
    if api_key not in VALID_KEYS:
        raise HTTPException(
            status_code = status.HTTP_403_FORBIDDEN,
            detail      = "Invalid API key",
        )
    return api_key
