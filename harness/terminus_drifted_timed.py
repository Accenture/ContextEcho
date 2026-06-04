"""TerminusDrifted variant with per-LLM-call wall-clock recording.

Authorized by signed PREREG_AMENDMENT_TERMINALBENCH §4.5 ("the harness will
wrap each `chat.chat()` call to record the wall-clock between request send
and first token received"). Because TerminalBench's LiteLLM wrapper raises
NotImplementedError on streaming, true TTFT is unavailable; per the §4.5
fallback clause, this class records **total LLM call wall-clock per turn**
instead — strictly more useful than the inter-episode `start_time` delta
because it excludes container/tool execution.

The smoke-locked `TerminusDrifted` (SHA-256 4f2f5547...) is unchanged. This
class is a sibling, used only for Phase 1+ scaled runs.

Per-turn LLM wall-clock seconds are dumped to `<logging_dir>/llm_seconds_per_turn.json`
when `logging_dir` is non-None. The panel orchestrator reads that sidecar
file and merges into the cell's results.json.

Wires into TerminalBench:
  tb run --agent-import-path harness.terminus_drifted_timed:TerminusDriftedTimed \\
         --model anthropic/claude-sonnet-4-6 \\
         -k recent3K_path=/path/to/recent3k.txt \\
         -k condition_label=recent3K
"""
from __future__ import annotations

import json
from pathlib import Path

from terminal_bench.agents.base_agent import AgentResult
from terminal_bench.agents.failure_mode import FailureMode
from terminal_bench.llms.chat import Chat
from terminal_bench.terminal.tmux_session import TmuxSession

from harness.terminus_drifted import ACK_MESSAGE, TerminusDrifted
from harness.timed_lite_llm import TimedLiteLLM


class TerminusDriftedTimed(TerminusDrifted):
    """TerminusDrifted with TimedLiteLLM + multi-arm context injection.

    Adds support for filler3K and GPT5_3K control arms beyond the parent's
    scratch / recent3K. The condition_label drives which file to load:
      - "scratch": no injection (parent default)
      - "recent3K": load recent3K_path (parent default)
      - "filler3K": load context_path
      - "gpt5_3K": load context_path
      - any other label with context_path set: load context_path

    context_path is a generic kwarg that overrides recent3K_path-based loading
    when condition_label is not "recent3K". This lets us run cross-session
    SWE-Bench tests where the injected context is independent of the locked
    Claude-derived recent3K.
    """

    def __init__(self, *args, context_path: str | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Replace the parent's LiteLLM with a timed subclass.
        self._llm = TimedLiteLLM(
            model_name=self._model_name,
            api_base=self._llm._api_base,
            temperature=self._llm._temperature,
        )

        # If parent set up scratch (no recent3K_text loaded) but the user
        # supplied context_path with a non-scratch condition label, load it.
        if (self._recent3K_text is None
                and context_path
                and self._condition_label not in ("scratch", "recent3K")):
            p = Path(context_path)
            if not p.exists():
                raise FileNotFoundError(f"context_path does not exist: {context_path}")
            self._recent3K_text = p.read_text()
            if not self._recent3K_text.strip():
                raise ValueError(f"context_path is empty: {context_path}")
            self._logger.info(
                f"TerminusDriftedTimed: loaded {len(self._recent3K_text)} chars "
                f"from context_path={context_path}; condition={self._condition_label}"
            )

    def perform_task(
        self,
        instruction: str,
        session: TmuxSession,
        logging_dir: Path | None = None,
    ) -> AgentResult:
        chat = Chat(self._llm)
        if self._recent3K_text is not None:
            chat._messages = [
                {"role": "user", "content": self._recent3K_text},
                {"role": "assistant", "content": ACK_MESSAGE},
            ]

        initial_prompt = self._prompt_template.format(
            response_schema=self._response_schema,
            instruction=instruction,
            history="",
            terminal_state=session.capture_pane(),
        )

        self._run_agent_loop(initial_prompt, session, chat, logging_dir)

        if logging_dir is not None:
            sidecar = logging_dir / "llm_seconds_per_turn.json"
            sidecar.write_text(json.dumps({
                "llm_seconds_per_turn": list(self._llm.call_seconds_history),
                "total_llm_seconds": sum(self._llm.call_seconds_history),
                "n_calls": len(self._llm.call_seconds_history),
                "note": "wall-clock for each litellm.completion() round-trip; excludes container/tool exec",
            }, indent=2))

        return AgentResult(
            total_input_tokens=chat.total_input_tokens,
            total_output_tokens=chat.total_output_tokens,
            failure_mode=FailureMode.NONE,
            timestamped_markers=self._timestamped_markers,
        )
