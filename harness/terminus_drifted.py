"""TerminalBench Terminus agent with persona-drift `recent3K` injection.

Subclasses TerminalBench's stock `Terminus` agent. Pre-seeds the chat
history with our donated Claude-Code-derived `recent3K` context (last
3K characters of a real Claude Code agentic-coding session) and an
acknowledgment message before the task instruction is given.

Wires into TerminalBench via:
  tb run --agent-import-path harness.terminus_drifted:TerminusDrifted \\
         --model anthropic/claude-sonnet-4-6 \\
         -k recent3K_path=/path/to/recent3k.txt

The recent3K text is read from `recent3K_path` (provided as agent-kwarg)
because passing 3000 characters via CLI is impractical. If
`recent3K_path` is not supplied or empty, the agent behaves identically
to the stock Terminus (this matches the `scratch` condition).

Locked under the panel-extension Convention B for any Claude calls
performed via this agent (temperature handled by terminus_1's default,
which we do NOT override — the agent's research question is multi-turn
task efficiency under context perturbation, not deterministic decoding).
"""
from __future__ import annotations

from pathlib import Path

# terminal-bench is installed via pip; import as normal.
from terminal_bench.agents.terminus import Terminus
from terminal_bench.agents.base_agent import AgentResult
from terminal_bench.agents.failure_mode import FailureMode
from terminal_bench.llms.chat import Chat
from terminal_bench.terminal.tmux_session import TmuxSession


ACK_MESSAGE = "Acknowledged. How can I help continue this work?"


class TerminusDrifted(Terminus):
    """Terminus with optional `recent3K` history pre-injection."""

    def __init__(
        self,
        model_name: str,
        max_episodes: int = 50,
        api_base: str | None = None,
        temperature: float = 0.7,
        recent3K_path: str | None = None,
        condition_label: str = "scratch",
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            max_episodes=max_episodes,
            api_base=api_base,
            temperature=temperature,
            **kwargs,
        )
        self._condition_label = condition_label
        self._recent3K_text: str | None = None
        if recent3K_path and condition_label == "recent3K":
            p = Path(recent3K_path)
            if not p.exists():
                raise FileNotFoundError(
                    f"recent3K_path does not exist: {recent3K_path}"
                )
            self._recent3K_text = p.read_text()
            if not self._recent3K_text.strip():
                raise ValueError(f"recent3K_path is empty: {recent3K_path}")
            self._logger.info(
                f"TerminusDrifted: loaded {len(self._recent3K_text)} chars "
                f"of recent3K context from {recent3K_path}; condition={condition_label}"
            )
        else:
            self._logger.info(
                f"TerminusDrifted: no injection (condition={condition_label}, "
                f"recent3K_path={'unset' if not recent3K_path else 'set-but-skipped'})"
            )

    def perform_task(
        self,
        instruction: str,
        session: TmuxSession,
        logging_dir: Path | None = None,
    ) -> AgentResult:
        # Build the same Chat the parent does, but pre-seed history if
        # we're in the recent3K condition.
        chat = Chat(self._llm)
        if self._recent3K_text is not None:
            # Match the panel-extension protocol exactly: user-message
            # carrying the recent3K text, then assistant-acknowledgment.
            # The agent's own first prompt then comes after.
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

        return AgentResult(
            total_input_tokens=chat.total_input_tokens,
            total_output_tokens=chat.total_output_tokens,
            failure_mode=FailureMode.NONE,
            timestamped_markers=self._timestamped_markers,
        )
