"""NVIDIA NIM API wrapper for the panel-extension scripts.

Provides ``call_nvidia`` matching the call signature of ``call_anthropic``,
``call_openai``, ``call_gemini``, ``call_together``, and ``call_mistral``.

NVIDIA's NIM API at https://integrate.api.nvidia.com/v1 is OpenAI-compatible
(same `chat/completions` endpoint shape). API key from ``NVIDIA_API_KEY`` in
``me/projects/.env`` (key prefix: ``nvapi-``, length 70).

Same pattern as Together AI / Mistral wrappers — just point an OpenAI
client at the NIM base URL.
"""
from __future__ import annotations

import os
import time
from typing import Optional


NVIDIA_DEFAULT_MAX_TOKENS = 4096
NVIDIA_DEFAULT_TEMPERATURE = 0.0
NVIDIA_DEFAULT_TIMEOUT_SEC = 120.0
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"


def get_nvidia_api_key() -> str:
    key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "NVIDIA_API_KEY is not set. Source ../.env or export it."
        )
    return key


def make_nvidia_client():
    """Build an OpenAI-compatible client pointed at NVIDIA NIM."""
    from openai import OpenAI
    return OpenAI(
        api_key=get_nvidia_api_key(),
        base_url=NVIDIA_BASE_URL,
    )


def call_nvidia(
    client,
    model_id: str,
    messages: list[dict],
    system: Optional[str] = None,
    max_retries: int = 4,
    max_tokens: int = NVIDIA_DEFAULT_MAX_TOKENS,
    temperature: float = NVIDIA_DEFAULT_TEMPERATURE,
    request_timeout: float = NVIDIA_DEFAULT_TIMEOUT_SEC,
) -> tuple[str, int, int]:
    """Call an NVIDIA-hosted NIM model and return (text, in_tok, out_tok).

    Args:
      client: ``OpenAI(...)`` instance pointed at NIM base_url.
      model_id: e.g., ``"nvidia/nemotron-3-super-120b-a12b"``.
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
      Permanent 4xx errors (auth / billing / not-found) propagate
      immediately without retry.
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
                    f"NVIDIA {model_id} returned empty visible output "
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
                f"NVIDIA {model_id} exhausted {max_retries} retries "
                f"({etype}): {e}"
            ) from e

    raise RuntimeError(
        f"NVIDIA {model_id} call failed after {max_retries} attempts; "
        f"last error: {last_err}"
    )
