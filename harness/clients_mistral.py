"""Mistral la Plateforme wrapper for the panel-extension scripts.

Provides ``call_mistral`` matching the call signature of ``call_anthropic``,
``call_openai``, ``call_gemini``, and ``call_together``.

Mistral's own API at https://api.mistral.ai/v1 is OpenAI-compatible (uses
the same `chat/completions` endpoint shape). We use the OpenAI Python SDK
pointed at Mistral's base URL — same pattern as Together AI.

API key from ``MISTRAL_API_KEY`` in ``me/projects/.env``.

Locked under PREREG_AMENDMENT_MISTRAL.md (sha256 30d3890f90d1...). Sampling
parameters match the panel-extension Convention B: temperature=0.0,
max_tokens=4096, request_timeout=120s. Deviation §6 of the amendment
will note the target update from `Mistral-Large-Instruct-2411` (Together,
gated) to `mistral-large-latest` (Mistral API serverless, which resolves
to `mistral-large-2512` as of 2026-04-29).
"""
from __future__ import annotations

import os
import time
from typing import Optional


MISTRAL_DEFAULT_MAX_TOKENS = 4096
MISTRAL_DEFAULT_TEMPERATURE = 0.0
MISTRAL_DEFAULT_TIMEOUT_SEC = 120.0
MISTRAL_BASE_URL = "https://api.mistral.ai/v1"


def get_mistral_api_key() -> str:
    """Pull the Mistral API key from env."""
    key = os.environ.get("MISTRAL_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "MISTRAL_API_KEY is not set in the environment. "
            "Source ../.env or export it."
        )
    return key


def make_mistral_client():
    """Build an OpenAI-compatible client pointed at Mistral la Plateforme."""
    from openai import OpenAI
    return OpenAI(
        api_key=get_mistral_api_key(),
        base_url=MISTRAL_BASE_URL,
    )


def call_mistral(
    client,
    model_id: str,
    messages: list[dict],
    system: Optional[str] = None,
    max_retries: int = 4,
    max_tokens: int = MISTRAL_DEFAULT_MAX_TOKENS,
    temperature: float = MISTRAL_DEFAULT_TEMPERATURE,
    request_timeout: float = MISTRAL_DEFAULT_TIMEOUT_SEC,
) -> tuple[str, int, int]:
    """Call a Mistral-hosted model and return (text, in_tok, out_tok).

    Args:
      client: ``OpenAI(...)`` instance pointed at Mistral's base_url.
      model_id: Mistral model identifier, e.g., ``"mistral-large-latest"``.
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
            # Mistral reasoning models (e.g., magistral-*) return content
            # as a LIST of structured blocks rather than a string. Extract
            # only the visible text parts; ignore "thinking" / reasoning
            # blocks (those are billed but invisible). Diagnosed 2026-04-29
            # on magistral-medium-latest smoke test.
            raw_content = resp.choices[0].message.content
            if isinstance(raw_content, list):
                text_parts = []
                for block in raw_content:
                    if isinstance(block, dict):
                        # Mistral block shape varies: {"type":"text","text":"..."}
                        # or just {"text":"..."}. Be defensive.
                        if block.get("type") == "text" and block.get("text"):
                            text_parts.append(block["text"])
                        elif "text" in block and block.get("type") not in (
                            "thinking", "reasoning"
                        ):
                            text_parts.append(block["text"])
                    elif isinstance(block, str):
                        text_parts.append(block)
                    else:
                        # SDK object form: try .text attribute
                        t = getattr(block, "text", None)
                        if t and getattr(block, "type", None) not in (
                            "thinking", "reasoning"
                        ):
                            text_parts.append(t)
                text = "\n".join(text_parts)
            else:
                text = raw_content or ""
            in_tok = int(resp.usage.prompt_tokens or 0)
            out_tok = int(resp.usage.completion_tokens or 0)
            if not text.strip():
                raise RuntimeError(
                    f"Mistral {model_id} returned empty visible output "
                    f"(in={in_tok}, out={out_tok}). raw_content type="
                    f"{type(raw_content).__name__}; reasoning models "
                    f"may need higher max_tokens to leave room for "
                    f"visible output after reasoning."
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
                f"Mistral {model_id} exhausted {max_retries} retries "
                f"({etype}): {e}"
            ) from e

    raise RuntimeError(
        f"Mistral {model_id} call failed after {max_retries} attempts; "
        f"last error: {last_err}"
    )
