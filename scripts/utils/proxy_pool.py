"""Round-robin proxy pool with per-proxy rate limiting."""

from __future__ import annotations

import itertools

from scripts.utils.rate_limiter import RateLimiter


class ProxyPool:
    """Round-robin proxy rotator with per-proxy rate limiting.

    Each proxy gets its own :class:`RateLimiter` so that the total throughput
    scales linearly with the number of proxies (N proxies × rate = total req/s).

    Usage::

        pool = ProxyPool(["http://p1:8080", "socks5://p2:1080"], rate=0.5)
        proxy_url = await pool.acquire()  # waits for that proxy's rate limiter
        async with session.get(url, proxy=proxy_url) as resp:
            ...

    If constructed with an empty list, :meth:`acquire` always returns ``None``
    (direct connection).
    """

    def __init__(self, proxies: list[str], rate: float) -> None:
        self._entries: list[tuple[str, RateLimiter]] = [
            (url, RateLimiter(rate=rate)) for url in proxies
        ]
        self._cycle = itertools.cycle(self._entries) if self._entries else None

    def __len__(self) -> int:
        return len(self._entries)

    async def acquire(self) -> str | None:
        """Return the next proxy URL after its rate limiter allows a request.

        Returns ``None`` when the pool is empty (no proxies configured).
        """
        if self._cycle is None:
            return None
        proxy_url, limiter = next(self._cycle)
        await limiter.acquire()
        return proxy_url
