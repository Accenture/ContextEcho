"""Cohere wrapper for the panel-extension scripts.

Provides ``call_cohere`` matching the call signature of ``call_anthropic``,
``call_openai``, ``call_gemini``, ``call_together``, ``call_mistral``, and
``call_nvidia``.

Cohere's OpenAI-compatible endpoint at
``https://api.cohere.com/compatibility/v1`` accepts the same
``chat/completions`` request shape we use for every other panel-extension
provider. API key from ``COHERE_API_KEY`` in ``me/projects/.env``.

Locked under the panel-extension Convention B: temperature=0.0,
max_tokens=4096, request_timeout=120s.
"""
from __future__ import annotations

import os
import time
from typing import Optional


COHERE_DEFAULT_MAX_TOKENS = 4096
COHERE_DEFAULT_TEMPERATURE = 0.0
COHERE_DEFAULT_TIMEOUT_SEC = 120.0
COHERE_BASE_URL = "https://api.cohere.com/compatibility/v1"


def get_cohere_api_key() -> str:
    key = os.environ.get("COHERE_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "COHERE_API_KEY is not set. Source ../.env or export it."
        )
    return key


def make_cohere_client():
    """Build an OpenAI-compatible client pointed at Cohere."""
    from openai import OpenAI
    return OpenAI(
        api_key=get_cohere_api_key(),
        base_url=COHERE_BASE_URL,
    )


def call_cohere(
    client,
    model_id: str,
    messages: list[dict],
    system: Optional[str] = None,
    max_retries: int = 4,
    max_tokens: int = COHERE_DEFAULT_MAX_TOKENS,
    temperature: float = COHERE_DEFAULT_TEMPERATURE,
    request_timeout: float = COHERE_DEFAULT_TIMEOUT_SEC,
) -> tuple[str, int, int]:
    """Call a Cohere-hosted model and return (text, in_tok, out_tok).

    Args:
      client: ``OpenAI(...)`` instance pointed at Cohere's compatibility URL.
      model_id: e.g., ``"command-a-03-2025"`` or
        ``"command-a-reasoning-08-2025"``.
      messages: list of ``{"role": "user"|"assistant", "content": str}``.
      system: system prompt (prepended as a system role message).
      max_retries: retry budget for transient errors (429, 5xx, network).
      max_tokens: visible-output token cap. Default 4096.
      temperature: 0.0 for deterministic decoding (panel convention).
      request_timeout: per-call HTTP timeout in seconds.

    Returns:
      ``(response_text, prompt_tokens, completion_tokens)``.

    Raises:
      RuntimeError: if all retries exhausted, or if response is empty.
      Permanent 4xx errors propagate immediately without retry.
    """
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
            in_tok = int(resp.usage.prompt_tokens or 0)
            out_tok = int(resp.usage.completion_tokens or 0)
            if not text.strip():
                raise RuntimeError(
                    f"Cohere {model_id} returned empty visible output "
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
                f"Cohere {model_id} exhausted {max_retries} retries "
                f"({etype}): {e}"
            ) from e

    raise RuntimeError(
        f"Cohere {model_id} call failed after {max_retries} attempts; "
        f"last error: {last_err}"
    )
