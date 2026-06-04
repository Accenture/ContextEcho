"""LiteLLM subclass that records per-call wall-clock (LLM-only).

Wraps `LiteLLM.call` to time the round-trip from request to response.
Excludes tmux/container/tool execution time, unlike the inter-episode
delta we compute from start_time fields. This is the cleanest available
measurement of "LLM time per turn" for the H1 sec/output_token metric.

TTFT (time-to-first-token) is NOT captured because TerminalBench's
LiteLLM wrapper raises NotImplementedError on streaming responses.
Per signed amendment §4.5, the fallback is to omit TTFT and proceed
with total LLM call wall-clock, which is what this class provides.

Per-turn timings are stored on the instance as `self.last_call_seconds`
(latest call) and accumulated into `self.call_seconds_history` (one
entry per call across the lifetime of the instance). Saved to disk
by the panel orchestrator as `llm_seconds_per_turn[]` in results.json.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from litellm import Message
from pydantic import BaseModel

from terminal_bench.llms.lite_llm import LiteLLM


class TimedLiteLLM(LiteLLM):
    """Subclass of LiteLLM that records per-call wall-clock seconds."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.call_seconds_history: list[float] = []
        self.last_call_seconds: float | None = None

    def call(
        self,
        prompt: str,
        message_history: list[dict[str, Any] | Message] = [],
        response_format: dict | type[BaseModel] | None = None,
        logging_path: Path | None = None,
        **kwargs,
    ) -> str:
        t0 = time.perf_counter()
        try:
            return super().call(
                prompt=prompt,
                message_history=message_history,
                response_format=response_format,
                logging_path=logging_path,
                **kwargs,
            )
        finally:
            elapsed = time.perf_counter() - t0
            self.last_call_seconds = elapsed
            self.call_seconds_history.append(elapsed)
