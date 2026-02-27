"""Async token-bucket rate limiter for polite scraping."""

import asyncio
import time


class RateLimiter:
    """Token-bucket rate limiter for async HTTP requests.

    Usage:
        limiter = RateLimiter(rate=2.0)  # 2 requests per second
        async with limiter:
            await session.get(url)
    """

    def __init__(self, rate: float, burst: int = 1) -> None:
        """Initialize rate limiter.

        Args:
            rate: Maximum requests per second (sustained).
            burst: Maximum burst size (tokens available at once).
        """
        self._rate = rate
        self._burst = burst
        self._tokens = float(burst)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
        self._last_refill = now

    async def acquire(self) -> None:
        """Wait until a token is available."""
        while True:
            async with self._lock:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
            # Sleep for the time needed to get one token
            await asyncio.sleep(1.0 / self._rate)

    async def __aenter__(self) -> "RateLimiter":
        await self.acquire()
        return self

    async def __aexit__(self, *exc: object) -> None:
        pass
