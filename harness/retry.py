"""
Exponential backoff retry wrapper for API calls.

Handles:
- 429 (rate limit) — long backoff
- 500/502/503/529 (server errors) — medium backoff
- Network errors — short backoff
- Other — immediate raise (no retry)
"""
from __future__ import annotations

import random
import time
from typing import Callable, TypeVar

T = TypeVar("T")


def call_with_retry(
    fn: Callable[[], T],
    *,
    max_retries: int = 5,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    jitter: float = 0.25,
    context: str = "",
) -> T:
    """
    Call fn() with exponential backoff on transient errors.

    Exceptions we retry on: any exception whose class name or message contains
    rate-limit or server-error signals from anthropic / openai SDKs.
    """
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == max_retries:
                raise
            if not _is_retryable(e):
                raise
            delay = min(max_delay, base_delay * (2**attempt))
            delay += random.uniform(0, jitter * delay)
            msg = f"[retry {attempt + 1}/{max_retries}] {context}: {type(e).__name__}: {str(e)[:200]}"
            print(f"{msg}  — sleeping {delay:.1f}s", flush=True)
            time.sleep(delay)
    raise RuntimeError("unreachable")  # pragma: no cover


def _is_retryable(e: BaseException) -> bool:
    name = type(e).__name__.lower()
    msg = str(e).lower()
    retryable_names = {
        "ratelimiterror",
        "apitimeouterror",
        "apiconnectionerror",
        "internalservererror",
        "serviceunavailableerror",
        "apistatuserror",  # might be transient 5xx
    }
    if name in retryable_names:
        return True
    if any(code in msg for code in ["429", "500", "502", "503", "529", "overloaded", "timeout", "timed out"]):
        return True
    return False
