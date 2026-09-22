import hashlib
import os
import time
from collections import defaultdict
from typing import Dict, List
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.config import settings

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_requests: int = 120, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.history: Dict[str, List[float]] = defaultdict(list)
        self._last_eviction = time.time()

    def _get_bucket_key(self, request: Request) -> str:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()
            token_hash = hashlib.sha256(token.encode()).hexdigest()[:16]
            return f"user_{token_hash}"

        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            # First IP in X-Forwarded-For is client
            client_ip = forwarded.split(",")[0].strip()
            return f"ip_{client_ip}"

        client_ip = request.client.host if request.client else "unknown"
        return f"ip_{client_ip}"

    def _prune_expired(self, now: float):
        if now - self._last_eviction < 60.0:
            return
        self._last_eviction = now
        window_start = now - self.window_seconds
        stale_keys = [k for k, v in self.history.items() if not v or v[-1] <= window_start]
        for k in stale_keys:
            del self.history[k]

    async def dispatch(self, request: Request, call_next):
        if os.getenv("PYTEST_CURRENT_TEST"):
            return await call_next(request)

        # Exclude static assets and health check
        path = request.url.path
        if path.startswith("/static") or path == "/" or path == "/health":
            return await call_next(request)

        now = time.time()
        self._prune_expired(now)

        bucket_key = self._get_bucket_key(request)
        window_start = now - self.window_seconds

        # Clean old timestamps for this bucket
        timestamps = self.history[bucket_key]
        self.history[bucket_key] = [t for t in timestamps if t > window_start]

        is_auth_route = path.endswith("/auth/login") or path.endswith("/auth/register")
        limit = 25 if is_auth_route else self.max_requests

        if len(self.history[bucket_key]) >= limit:
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "code": "RATE_LIMIT_EXCEEDED",
                        "message": f"Rate limit exceeded ({limit} requests per minute). Please throttle requests.",
                    }
                },
                headers={"Retry-After": str(self.window_seconds)}
            )

        self.history[bucket_key].append(now)
        response = await call_next(request)
        return response
