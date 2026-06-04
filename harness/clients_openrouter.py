"""OpenRouter wrapper (OpenAI-compatible endpoint).

OpenRouter aggregates many providers (Together, DeepInfra, Fireworks, Anyscale,
Moonshot direct, etc.) behind a single OpenAI-compatible endpoint at
https://openrouter.ai/api/v1, with automatic provider failover.

This is used as the canonical path for Kimi K2.6 because the previous Together-
only path returned empty content on ~91% of long-context completions. With
OpenRouter we can let the router pick a working upstream automatically.

Env: OPEN_ROUTER__API_KEY (matches the project .env naming).
"""
from __future__ import annotations

import os
import time
from typing import Optional


OPENROUTER_DEFAULT_MAX_TOKENS = 1024
OPENROUTER_DEFAULT_TEMPERATURE = 0.0
OPENROUTER_DEFAULT_TIMEOUT_SEC = 180.0
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def get_openrouter_api_key() -> str:
    key = (
        os.environ.get("OPEN_ROUTER__API_KEY", "").strip()
        or os.environ.get("OPENROUTER_API_KEY", "").strip()
    )
    if not key:
        raise RuntimeError(
            "OPEN_ROUTER__API_KEY (or OPENROUTER_API_KEY) is not set. "
            "Source ../.env or export it."
        )
    return key


def make_openrouter_client():
    from openai import OpenAI
    return OpenAI(
        api_key=get_openrouter_api_key(),
        base_url=OPENROUTER_BASE_URL,
    )


def call_openrouter(
    client,
    model_id: str,
    messages: list[dict],
    system: Optional[str] = None,
    max_retries: int = 4,
    max_tokens: int = OPENROUTER_DEFAULT_MAX_TOKENS,
    temperature: float = OPENROUTER_DEFAULT_TEMPERATURE,
    request_timeout: float = OPENROUTER_DEFAULT_TIMEOUT_SEC,
) -> tuple[str, int, int]:
    """Call an OpenRouter-hosted model. Returns (text, in_tok, out_tok)."""
    oai_msgs: list[dict] = []
    if system is not None:
        oai_msgs.append({"role": "system", "content": system})
    for m in messages:
        oai_msgs.append({"role": m["role"], "content": m["content"]})

    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=oai_msgs,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=request_timeout,
            )
            text = resp.choices[0].message.content or ""
            usage = resp.usage
            in_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
            out_tok = int(getattr(usage, "completion_tokens", 0) or 0)
            if not text.strip():
                raise RuntimeError(
                    f"OpenRouter {model_id} returned empty visible output "
                    f"(in={in_tok}, out={out_tok})."
                )
            return text, in_tok, out_tok
        except Exception as e:
            etype = type(e).__name__
            status = (
                getattr(e, "status_code", None)
                or getattr(getattr(e, "response", None), "status_code", None)
            )
            permanent = isinstance(status, int) and 400 <= status < 500 and status != 429
            if permanent:
                raise
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(2 * (2 ** attempt))
                continue
            raise RuntimeError(
                f"OpenRouter {model_id} exhausted {max_retries} retries "
                f"({etype}): {e}"
            ) from e

    raise RuntimeError(
        f"OpenRouter {model_id} call failed after {max_retries} attempts; "
        f"last error: {last_err}"
    )
