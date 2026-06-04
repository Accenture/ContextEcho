"""Together AI wrapper (OpenAI-compatible endpoint) for the panel-extension scripts.

Provides ``call_together`` matching the call signature of ``call_anthropic``,
``call_openai``, and ``call_gemini`` in the panel-extension scripts.

Together AI hosts Mistral, Cohere, Llama, Qwen, DeepSeek, Gemma, and many
other open-weights and open-API models behind a single OpenAI-compatible
endpoint at https://api.together.xyz/v1.

Note: this differs from the existing ``scripts/non_anthropic_extension.py``
``call_together`` in two ways, both locked by the panel-extension family
of pre-registration amendments:

  1. ``temperature=0.0`` (deterministic decoding) instead of 0.7.
  2. ``max_tokens=4096`` instead of 400.

These match the convention locked in PREREG_AMENDMENT_GEMINI.md §2.3 and
used across all post-Gemini panel-extension work. The original
``non_anthropic_extension.py`` predates that amendment and retains its
original parameters for reproducing the existing 12-target panel.
"""
from __future__ import annotations

import os
import time
from typing import Optional


# Match the locked sampling parameters from the panel-extension convention.
TOGETHER_DEFAULT_MAX_TOKENS = 4096
TOGETHER_DEFAULT_TEMPERATURE = 0.0
TOGETHER_DEFAULT_TIMEOUT_SEC = 120.0
TOGETHER_BASE_URL = "https://api.together.xyz/v1"


def get_together_api_key() -> str:
    """Pull the Together API key from env. Both `TOGETHER_AI_KEY` (primary,
    in this project's .env) and `TOGETHER_API_KEY` (used by some older
    scripts) are accepted to avoid env-var-name confusion."""
    key = (
        os.environ.get("TOGETHER_AI_KEY", "").strip()
        or os.environ.get("TOGETHER_API_KEY", "").strip()
    )
    if not key:
        raise RuntimeError(
            "TOGETHER_AI_KEY (or TOGETHER_API_KEY) is not set in the "
            "environment. Source ../.env or export it."
        )
    return key


def make_together_client():
    """Build an OpenAI-compatible client pointed at Together AI."""
    from openai import OpenAI
    return OpenAI(
        api_key=get_together_api_key(),
        base_url=TOGETHER_BASE_URL,
    )


def call_together(
    client,
    model_id: str,
    messages: list[dict],
    system: Optional[str] = None,
    max_retries: int = 4,
    max_tokens: int = TOGETHER_DEFAULT_MAX_TOKENS,
    temperature: float = TOGETHER_DEFAULT_TEMPERATURE,
    request_timeout: float = TOGETHER_DEFAULT_TIMEOUT_SEC,
) -> tuple[str, int, int]:
    """Call a Together-hosted model and return (text, in_tok, out_tok).

    Args:
      client: ``OpenAI(...)`` instance pointed at Together's base_url.
      model_id: Together model identifier, e.g.,
        ``"mistralai/Mistral-Large-Instruct-2411"``.
      messages: list of ``{"role": "user"|"assistant", "content": str}``.
      system: system prompt (prepended as a system role message).
      max_retries: retry budget for transient errors (429, 5xx, network).
      max_tokens: visible-output token cap. Default 4096.
      temperature: 0.0 for deterministic decoding (panel convention).
      request_timeout: per-call HTTP timeout in seconds.

    Returns:
      Tuple of ``(response_text, prompt_tokens, completion_tokens)``.

    Raises:
      RuntimeError: if all retries exhausted, or if response is empty.
      Underlying ``openai`` errors with status 4xx (auth / billing) are
      propagated immediately without retry.
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
                    f"Together {model_id} returned empty visible output "
                    f"(in={in_tok}, out={out_tok})."
                )
            return text, in_tok, out_tok
        except Exception as e:
            etype = type(e).__name__
            # Permanent client errors should NOT retry.
            status = (
                getattr(e, "status_code", None)
                or getattr(getattr(e, "response", None), "status_code", None)
            )
            permanent = isinstance(status, int) and 400 <= status < 500 and status != 429
            if permanent:
                raise
            # Transient: retry with backoff.
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(2 * (2 ** attempt))
                continue
            raise RuntimeError(
                f"Together {model_id} exhausted {max_retries} retries "
                f"({etype}): {e}"
            ) from e

    # Defensive — should be unreachable.
    raise RuntimeError(
        f"Together {model_id} call failed after {max_retries} attempts; "
        f"last error: {last_err}"
    )
