"""
Redis cache for URL predictions.
Falls back to in-memory dict if Redis is unavailable (dev mode).
"""

import os
import json
import logging
import hashlib
from typing import Optional

log = logging.getLogger(__name__)


class PredictionCache:
    """
    Simple cache wrapper.
    Uses Redis in production, in-memory dict in development.
    TTL: 3600 seconds (1 hour) for URL predictions.
    """

    def __init__(self, ttl: int = 3600):
        self.ttl         = ttl
        self._redis      = None
        self._memory: dict[str, str] = {}
        self._using_redis = False

    async def connect(self) -> None:
        redis_url = os.getenv("REDIS_URL", "")
        if redis_url:
            try:
                import redis.asyncio as aioredis
                self._redis       = aioredis.from_url(redis_url,
                                                       decode_responses=True)
                await self._redis.ping()
                self._using_redis = True
                log.info(f"Redis connected: {redis_url}")
            except Exception as e:
                log.warning(f"Redis unavailable ({e}) — using in-memory cache")
        else:
            log.info("REDIS_URL not set — using in-memory cache (dev mode)")

    def _url_key(self, url: str) -> str:
        return "pred:" + hashlib.md5(url.encode()).hexdigest()

    async def get(self, url: str) -> Optional[dict]:
        key = self._url_key(url)
        try:
            if self._using_redis:
                raw = await self._redis.get(key)
            else:
                raw = self._memory.get(key)
            return json.loads(raw) if raw else None
        except Exception as e:
            log.warning(f"Cache get failed: {e}")
            return None

    async def set(self, url: str, data: dict) -> None:
        key = self._url_key(url)
        raw = json.dumps(data)
        try:
            if self._using_redis:
                await self._redis.setex(key, self.ttl, raw)
            else:
                self._memory[key] = raw
        except Exception as e:
            log.warning(f"Cache set failed: {e}")

    async def close(self) -> None:
        if self._redis:
            await self._redis.close()


# Module-level singleton
cache = PredictionCache()
