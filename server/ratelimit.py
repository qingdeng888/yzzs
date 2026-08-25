"""
server/ratelimit.py - 内存限流(每分钟/每天) + 配额检查
"""
from __future__ import annotations
import time
import threading
from collections import defaultdict, deque
from typing import Optional
from datetime import datetime, timedelta
from fastapi import HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from .db import get_db, session_scope
from .models import ApiKey, Usage
from .config import AppConfig


class RateLimiter:
    """按 API Key 限流(每分钟 + 每天)"""

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self._lock = threading.RLock()
        # {api_key_id: deque[timestamp]}
        self._minute_buckets: dict[int, deque] = defaultdict(deque)
        # {api_key_id: {date_str: count}}
        self._day_buckets: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def check(self, api_key: ApiKey) -> None:
        """校验 + 配额,不通过抛 HTTPException(429)"""
        now = time.time()
        today = datetime.utcnow().strftime("%Y-%m-%d")
        kid = api_key.id
        with self._lock:
            # 1) 每分钟限流
            mb = self._minute_buckets[kid]
            while mb and now - mb[0] > 60:
                mb.popleft()
            if len(mb) >= self.cfg.api.rate_limit.requests_per_minute:
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": {
                            "message": f"Rate limit exceeded: {self.cfg.api.rate_limit.requests_per_minute} req/min",
                            "type": "rate_limit",
                            "code": "rate_limit_exceeded",
                        }
                    },
                    headers={"Retry-After": "30"},
                )
            # 2) 每天限流(全局)
            day_total = self._day_buckets[kid].get(today, 0)
            if self.cfg.api.rate_limit.requests_per_day and day_total >= self.cfg.api.rate_limit.requests_per_day:
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": {
                            "message": f"Daily rate limit exceeded: {self.cfg.api.rate_limit.requests_per_day}",
                            "type": "rate_limit",
                            "code": "daily_limit_exceeded",
                        }
                    },
                )
            # 3) API Key 配额(查 DB,因为进程崩溃要持久)
            if api_key.daily_quota or api_key.monthly_quota:
                with session_scope() as db:
                    if api_key.daily_quota:
                        day_used = db.query(func.count(Usage.id)).filter(
                            Usage.api_key_id == kid,
                            Usage.created_at >= datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0),
                        ).scalar() or 0
                        if day_used >= api_key.daily_quota:
                            raise HTTPException(
                                status_code=429,
                                detail={
                                    "error": {
                                        "message": f"API key daily quota ({api_key.daily_quota}) exceeded",
                                        "type": "quota",
                                        "code": "daily_quota_exceeded",
                                    }
                                },
                            )
                    if api_key.monthly_quota:
                        first_of_month = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                        month_used = db.query(func.count(Usage.id)).filter(
                            Usage.api_key_id == kid,
                            Usage.created_at >= first_of_month,
                        ).scalar() or 0
                        if month_used >= api_key.monthly_quota:
                            raise HTTPException(
                                status_code=429,
                                detail={
                                    "error": {
                                        "message": f"API key monthly quota ({api_key.monthly_quota}) exceeded",
                                        "type": "quota",
                                        "code": "monthly_quota_exceeded",
                                    }
                                },
                            )
        # 通过:记账
        with self._lock:
            self._minute_buckets[kid].append(now)
            self._day_buckets[kid][today] += 1

    def get_stats(self, api_key_id: int) -> dict:
        """查实时计数"""
        now = time.time()
        today = datetime.utcnow().strftime("%Y-%m-%d")
        with self._lock:
            mb = self._minute_buckets.get(api_key_id, deque())
            recent = sum(1 for t in mb if now - t <= 60)
            return {
                "requests_last_minute": recent,
                "requests_today": self._day_buckets.get(api_key_id, {}).get(today, 0),
            }


_limiter: Optional[RateLimiter] = None


def init_limiter(cfg: AppConfig) -> RateLimiter:
    global _limiter
    _limiter = RateLimiter(cfg)
    return _limiter


def get_limiter() -> RateLimiter:
    if _limiter is None:
        raise RuntimeError("limiter not initialized")
    return _limiter
