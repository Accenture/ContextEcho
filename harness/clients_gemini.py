"""Google Gemini API wrapper for the persona-drift panel-extension scripts.

This module provides ``call_gemini`` matching the call signature of the
inline ``call_anthropic`` / ``call_openai`` functions in the
``scripts/cross_judge_12model.py``-style standalone runners.

Usage:
    from google import genai
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    text, in_tok, out_tok = call_gemini(
        client, "gemini-2.5-pro",
        messages=[{"role": "user", "content": "..."}],
        system="You are a helpful AI assistant.",
    )

Locked under PREREG_AMENDMENT_GEMINI.md (2026-04-28). Do not modify
``max_output_tokens`` minimum, role translation, or retry behavior post-hoc
without a logged deviation.
"""
from __future__ import annotations

import time
from typing import Optional


# Minimum max_output_tokens for Gemini 2.5 Pro. At lower values, Pro's
# internal reasoning tokens consume the budget before any visible output is
# produced (silent empty-output failure mode). 4096 verified clean in the
# 2026-04-28 smoke test (Pro produced 25 visible tokens after ~933 reasoning
# tokens; total well within 4096). DO NOT lower without re-validating.
GEMINI_MIN_MAX_OUTPUT_TOKENS = 4096


# Per-request timeout (seconds). The google-genai SDK has NO default request
# timeout, so a stalled connection hangs forever. Phase 2 hit a 70-min
# silent hang on Pro recent3K_filler (long-context call) — diagnosed
# 2026-04-28. We add a hard ceiling per call: any call that doesn't return
# within this many seconds raises and triggers retry. Pro's slowest observed
# call so far was ~15s (recent3K with reasoning), so 120s gives ~8x margin.
GEMINI_REQUEST_TIMEOUT_SEC = 120.0


def call_gemini(
    client,
    model_id: str,
    messages: list[dict],
    system: Optional[str] = None,
    max_retries: int = 4,
    max_output_tokens: int = GEMINI_MIN_MAX_OUTPUT_TOKENS,
    temperature: float = 0.0,
) -> tuple[str, int, int]:
    """Call a Gemini model and return (text, input_tokens, output_tokens).

    Mirrors the ``call_anthropic`` / ``call_openai`` shape used by the
    panel-extension scripts (e.g., ``scripts/cross_judge_12model.py``,
    ``scripts/a1_context_source_ablation.py``,
    ``scripts/b2_content_position_crossmodel.py``).

    Args:
      client: ``google.genai.Client`` instance, already authenticated.
      model_id: e.g., ``"gemini-2.5-pro"`` or ``"gemini-2.5-flash"``.
      messages: list of ``{"role": "user"|"assistant", "content": str}``.
        Translated to Gemini's ``user``/``model`` role naming internally.
      system: system instruction (Gemini's ``system_instruction`` field).
      max_retries: retry budget for transient errors (429, 5xx).
      max_output_tokens: hard floor at ``GEMINI_MIN_MAX_OUTPUT_TOKENS``;
        smaller values are silently raised to that floor with a warning.
      temperature: sampling temperature; default 0.0 for deterministic decoding,
        matching primary panel protocol convention.

    Returns:
      Tuple of (response_text, input_tokens, output_tokens).
        ``output_tokens`` is the *visible* output count (``candidates_token_count``
        on Gemini's response), NOT ``total_token_count`` which includes
        invisible reasoning tokens billed at output rates.

    Raises:
      RuntimeError: if all retries are exhausted, or if the API returns an
        empty response despite ``max_output_tokens`` being above the floor.
      Underlying ``google.genai`` errors (auth / billing / 400 / 403) are
        propagated immediately without retry — these indicate state problems
        not solvable by waiting.
    """
    # Lazy import so the rest of the codebase doesn't pay for google-genai
    # unless a Gemini call actually happens.
    from google.genai import types
    from google.genai import errors as genai_errors

    if max_output_tokens < GEMINI_MIN_MAX_OUTPUT_TOKENS:
        # Silent floor: enforced by the pre-reg amendment, not user-overridable.
        max_output_tokens = GEMINI_MIN_MAX_OUTPUT_TOKENS

    contents = _translate_messages(messages)
    # Per-request timeout via http_options on the config. The SDK takes
    # timeout in MILLISECONDS, not seconds. Without this, a stalled
    # connection hangs forever (observed 70-min hang on Pro recent3K_filler
    # 2026-04-28).
    config_kwargs: dict = {
        "max_output_tokens": max_output_tokens,
        "temperature": temperature,
        "http_options": types.HttpOptions(
            timeout=int(GEMINI_REQUEST_TIMEOUT_SEC * 1000),
        ),
    }
    if system is not None:
        config_kwargs["system_instruction"] = system

    # Belt-and-suspenders timeout: SDK's http_options.timeout was observed
    # NOT to fire on long-context calls. Wrap each attempt in a *fresh*
    # ThreadPoolExecutor — using a shared executor means a stuck zombie
    # thread blocks subsequent submits even after the local
    # future.result() timeout fires. A fresh executor per attempt
    # guarantees the retry actually gets its own thread to run on; the
    # stuck thread leaks but at least doesn't block forward progress.
    # Diagnosed 2026-04-29 after Flash long-context hangs of 35+ min.
    import concurrent.futures

    def _do_call():
        return client.models.generate_content(
            model=model_id,
            contents=contents,
            config=types.GenerateContentConfig(**config_kwargs),
        )

    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        # NEW executor per attempt — avoids zombie-thread queue blocking.
        _executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            future = _executor.submit(_do_call)
            try:
                resp = future.result(timeout=GEMINI_REQUEST_TIMEOUT_SEC)
            except concurrent.futures.TimeoutError:
                # Don't wait for the stuck thread; abandon the executor.
                # shutdown(wait=False) lets the zombie thread leak without
                # blocking us. The thread will eventually finish or be
                # garbage collected when the process exits.
                _executor.shutdown(wait=False)
                raise TimeoutError(
                    f"Gemini {model_id} call exceeded "
                    f"{GEMINI_REQUEST_TIMEOUT_SEC}s thread timeout "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
            else:
                # Success path: clean shutdown.
                _executor.shutdown(wait=False)
            text = resp.text or ""
            usage = resp.usage_metadata
            input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
            # Note: candidates_token_count is the VISIBLE output count.
            # Reasoning tokens (Pro only) are billed at output rates but live
            # in (total_token_count - input_tokens - candidates_token_count).
            output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)

            if not text.strip():
                # Empty output despite a high max_output_tokens cap is the
                # canonical Pro failure mode — usually means reasoning ate
                # the budget. Surface this rather than returning ""
                # silently; callers can decide whether to retry with a
                # higher cap.
                raise RuntimeError(
                    f"Gemini {model_id} returned empty visible output "
                    f"(input={input_tokens}, "
                    f"output={output_tokens}, "
                    f"total={getattr(usage, 'total_token_count', 'NA')}). "
                    f"Likely reasoning consumed max_output_tokens={max_output_tokens}."
                )
            return text, input_tokens, output_tokens

        except genai_errors.ClientError as e:
            # 429 = rate / billing; retry with backoff.
            # 5xx = transient server; retry with backoff.
            # 400 / 403 = auth / billing state / project denied — DO NOT retry.
            status = getattr(e, "status_code", None) or getattr(e, "code", None)
            if status == 429 or (status is not None and 500 <= status < 600):
                last_err = e
                if attempt < max_retries - 1:
                    wait = 2 * (2 ** attempt)
                    time.sleep(wait)
                    continue
                raise RuntimeError(
                    f"Gemini {model_id} exhausted {max_retries} retries on "
                    f"transient error: {e}"
                ) from e
            # 4xx other than 429 → propagate immediately.
            raise
        except Exception as e:
            # Timeout exceptions (httpx.ReadTimeout, ConnectError, etc.) and
            # other transient network errors fall through to here. Retry
            # them with backoff up to max_retries. Diagnosed 2026-04-28
            # after 70-min silent hang on Pro recent3K_filler.
            etype = type(e).__name__
            if "Timeout" in etype or "Connect" in etype or "ServerError" in etype:
                last_err = e
                if attempt < max_retries - 1:
                    wait = 2 * (2 ** attempt)
                    time.sleep(wait)
                    continue
                raise RuntimeError(
                    f"Gemini {model_id} exhausted {max_retries} retries on "
                    f"network error ({etype}): {e}"
                ) from e
            # Anything else (RuntimeError from empty-output, programming
            # errors, etc.) propagates immediately.
            raise

    # Should be unreachable (loop either returns or raises), but defensive:
    raise RuntimeError(
        f"Gemini {model_id} call failed after {max_retries} attempts; "
        f"last error: {last_err}"
    )


def _translate_messages(messages: list[dict]) -> list[dict]:
    """Translate Anthropic-style messages to Gemini ``contents`` format.

    Anthropic / OpenAI use ``role: "user"`` and ``role: "assistant"``.
    Gemini uses ``role: "user"`` and ``role: "model"``. Content goes into
    a ``parts: [{text: ...}]`` list rather than a flat string.
    """
    out: list[dict] = []
    for m in messages:
        role = m["role"]
        gemini_role = "model" if role == "assistant" else "user"
        content = m["content"]
        if isinstance(content, str):
            parts = [{"text": content}]
        elif isinstance(content, list):
            # Anthropic-style block content: extract text blocks only;
            # tool-use blocks are not used in the probe-call pattern this
            # wrapper targets (panel extension is single-turn probe calls).
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    text_parts.append(block)
            parts = [{"text": "\n".join(text_parts)}]
        else:
            parts = [{"text": str(content)}]
        out.append({"role": gemini_role, "parts": parts})
    return out
