"""
Thin wrappers around Anthropic and OpenAI SDKs.

Normalizes tool-use APIs so agent_loop.py doesn't care which provider is the target.
Both return a TargetResponse with the same shape.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal

from .cost import CostTracker
from .retry import call_with_retry


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass
class TargetResponse:
    text: str
    tool_calls: list[ToolCall]
    stop_reason: str
    raw_usage: dict


@dataclass
class ToolResult:
    tool_call_id: str
    content: str
    is_error: bool = False


ProviderName = Literal["anthropic", "openai", "together", "gemini", "openrouter",
                       "cohere", "nvidia"]


class TargetClient:
    """Unified interface for the target model."""

    _client: Any
    cost_tracker: CostTracker | None
    session_id: str

    def __init__(
        self,
        provider: ProviderName,
        model_id: str,
        cost_tracker: CostTracker | None = None,
        session_id: str = "",
    ):
        self.provider = provider
        self.model_id = model_id
        self.cost_tracker = cost_tracker
        self.session_id = session_id
        if provider == "anthropic":
            from anthropic import Anthropic
            self._client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        elif provider == "openai":
            from openai import OpenAI
            self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        elif provider == "together":
            from openai import OpenAI
            tg_key = (
                os.environ.get("TOGETHER_AI_KEY", "").strip()
                or os.environ.get("TOGETHER_API_KEY", "").strip()
            )
            if not tg_key:
                raise RuntimeError("TOGETHER_AI_KEY (or TOGETHER_API_KEY) not set")
            self._client = OpenAI(
                api_key=tg_key,
                base_url="https://api.together.xyz/v1",
            )
        elif provider == "gemini":
            from google import genai  # type: ignore
            if not os.environ.get("GOOGLE_API_KEY"):
                raise RuntimeError("GOOGLE_API_KEY not set")
            self._client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        elif provider == "openrouter":
            from openai import OpenAI
            or_key = (
                os.environ.get("OPEN_ROUTER__API_KEY", "").strip()
                or os.environ.get("OPENROUTER_API_KEY", "").strip()
            )
            if not or_key:
                raise RuntimeError("OPEN_ROUTER__API_KEY (or OPENROUTER_API_KEY) not set")
            self._client = OpenAI(
                api_key=or_key,
                base_url="https://openrouter.ai/api/v1",
            )
        elif provider == "cohere":
            from openai import OpenAI
            co_key = os.environ.get("COHERE_API_KEY", "").strip()
            if not co_key:
                raise RuntimeError("COHERE_API_KEY not set")
            # Cohere ships an OpenAI-compatible chat-completions endpoint
            # at this base URL; same call shape as OpenAI / OpenRouter.
            self._client = OpenAI(
                api_key=co_key,
                base_url="https://api.cohere.com/compatibility/v1",
            )
        elif provider == "nvidia":
            from openai import OpenAI
            nv_key = os.environ.get("NVIDIA_API_KEY", "").strip()
            if not nv_key:
                raise RuntimeError("NVIDIA_API_KEY not set")
            # NVIDIA build.nvidia.com exposes OpenAI-compatible chat-
            # completions; works with the same call shape as OpenAI.
            self._client = OpenAI(
                api_key=nv_key,
                base_url="https://integrate.api.nvidia.com/v1",
            )
        elif provider == "mistral":
            from openai import OpenAI
            mi_key = os.environ.get("MISTRAL_API_KEY", "").strip()
            if not mi_key:
                raise RuntimeError("MISTRAL_API_KEY not set")
            # Mistral la Plateforme exposes an OpenAI-compatible
            # chat-completions endpoint at this base URL.
            self._client = OpenAI(
                api_key=mi_key,
                base_url="https://api.mistral.ai/v1",
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def step(
        self,
        system_prompt: str,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int = 16384,
    ) -> TargetResponse:
        # Note: OpenAI uses this as a ceiling (non-streaming OK up to full completion).
        # Anthropic requires streaming for max_tokens > ~16384 due to their 10-min non-streaming limit.
        # Keep at 16384 unless we implement streaming in _step_anthropic.
        """
        messages: standardized list of {"role": "user"|"assistant", "content": [...]}
                  where content is a list of blocks:
                  - {"type": "text", "text": "..."}
                  - {"type": "tool_use", "id": "...", "name": "...", "input": {...}}  (assistant)
                  - {"type": "tool_result", "tool_use_id": "...", "content": "...", "is_error": bool}  (user)

        Returns a provider-normalized TargetResponse.
        """
        if self.provider == "anthropic":
            return self._step_anthropic(system_prompt, messages, tools, max_tokens)
        if self.provider == "gemini":
            return self._step_gemini(system_prompt, messages, max_tokens)
        # together uses an OpenAI-compatible client; same code path
        return self._step_openai(system_prompt, messages, tools, max_tokens)

    def _step_anthropic(
        self, system_prompt: str, messages: list[dict], tools: list[dict], max_tokens: int
    ) -> TargetResponse:
        response = call_with_retry(
            lambda: self._client.messages.create(
                model=self.model_id,
                system=system_prompt,
                messages=messages,
                tools=tools,
                max_tokens=max_tokens,
            ),
            context=f"anthropic.messages.create({self.model_id})",
        )
        text_parts = []
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, input=dict(block.input)))
        if self.cost_tracker is not None:
            self.cost_tracker.log(
                session_id=self.session_id,
                role="target",
                model_id=self.model_id,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )
        return TargetResponse(
            text="\n".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=str(response.stop_reason or ""),
            raw_usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        )

    def _step_openai(
        self, system_prompt: str, messages: list[dict], tools: list[dict], max_tokens: int
    ) -> TargetResponse:
        # Translate standardized messages to OpenAI chat format.
        openai_messages = [{"role": "system", "content": system_prompt}]
        for m in messages:
            role = m["role"]
            content = m["content"]
            if role == "assistant":
                text_parts = [c["text"] for c in content if c.get("type") == "text"]
                tool_calls = [
                    {
                        "id": c["id"],
                        "type": "function",
                        "function": {"name": c["name"], "arguments": _json_dumps(c["input"])},
                    }
                    for c in content
                    if c.get("type") == "tool_use"
                ]
                msg = {"role": "assistant", "content": "\n".join(text_parts) or None}
                if tool_calls:
                    msg["tool_calls"] = tool_calls
                openai_messages.append(msg)
            elif role == "user":
                tool_results = [c for c in content if c.get("type") == "tool_result"]
                text_parts = [c["text"] for c in content if c.get("type") == "text"]
                if tool_results:
                    for tr in tool_results:
                        openai_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tr["tool_use_id"],
                                "content": tr["content"],
                            }
                        )
                if text_parts:
                    openai_messages.append({"role": "user", "content": "\n".join(text_parts)})

        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                },
            }
            for t in tools
        ]

        # OpenAI: bump max_tokens for GPT-5 family (reasoning tokens count against
        # this budget). Note: reasoning_effort is NOT supported with tools on
        # /v1/chat/completions for gpt-5.4+; would require /v1/responses API
        # (future refactor). For now, just give a very large budget.
        is_reasoning_family = (
            self.model_id.startswith("gpt-5")
            or self.model_id.startswith("o3")
            or self.model_id.startswith("o4")
        )
        openai_max = max(max_tokens, 128000) if is_reasoning_family else max_tokens

        # Together uses standard `max_tokens`; OpenAI's gpt-5 family accepts
        # `max_completion_tokens` (and reasoning tokens count against it).
        # OpenRouter accepts both, but `max_tokens` is more reliable across
        # the diverse upstream models it routes to (gpt-5 needs a larger
        # budget to leave room for reasoning tokens).
        if self.provider == "together":
            create_kwargs = dict(
                model=self.model_id,
                messages=openai_messages,
                max_tokens=min(openai_max, 4096),
            )
            if openai_tools:
                create_kwargs["tools"] = openai_tools
        elif self.provider in ("cohere", "nvidia", "mistral"):
            # Cohere, NVIDIA, and Mistral expose OpenAI-compatible endpoints
            # that accept standard `max_tokens` (not `max_completion_tokens`).
            # NVIDIA Nemotron models emit reasoning tokens that count
            # against this budget, so leave 8K floor for them; Cohere and
            # Mistral are non-reasoning, 4K is fine.
            cn_max = max(openai_max, 8192) if self.provider == "nvidia" else min(openai_max, 4096)
            create_kwargs = dict(
                model=self.model_id,
                messages=openai_messages,
                max_tokens=cn_max,
            )
            if openai_tools:
                create_kwargs["tools"] = openai_tools
        elif self.provider == "openrouter":
            # OpenRouter's GPT-5 reasoning tokens count against the budget;
            # use a generous floor so visible output isn't starved.
            or_max = max(max_tokens, 4096)
            if "gpt-5" in self.model_id or "/o3" in self.model_id or "/o4" in self.model_id:
                or_max = max(or_max, 8192)
            create_kwargs = dict(
                model=self.model_id,
                messages=openai_messages,
                max_tokens=or_max,
            )
            if openai_tools:
                create_kwargs["tools"] = openai_tools
        else:
            create_kwargs = dict(
                model=self.model_id,
                messages=openai_messages,
                tools=openai_tools,
                max_completion_tokens=openai_max,
            )
        response = call_with_retry(
            lambda: self._client.chat.completions.create(**create_kwargs),
            context=f"{self.provider}.chat.completions.create({self.model_id})",
        )
        choice = response.choices[0]
        text = choice.message.content or ""
        tool_calls = []
        for tc in choice.message.tool_calls or []:
            fn = getattr(tc, "function", None)
            if fn is None:
                continue
            tool_calls.append(
                ToolCall(id=tc.id, name=fn.name, input=_json_loads(fn.arguments))
            )
        if self.cost_tracker is not None:
            self.cost_tracker.log(
                session_id=self.session_id,
                role="target",
                model_id=self.model_id,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            )
        return TargetResponse(
            text=text,
            tool_calls=tool_calls,
            stop_reason=choice.finish_reason,
            raw_usage={
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            },
        )


    def _step_gemini(
        self, system_prompt: str, messages: list[dict], max_tokens: int
    ) -> TargetResponse:
        """Gemini path. Tools not supported in this thin wrapper.

        Translates our standardized messages into the simple
        ``[{"role": "user"|"assistant", "content": str}]`` shape that
        ``harness.clients_gemini.call_gemini`` expects.
        """
        from harness.clients_gemini import call_gemini  # type: ignore
        flat: list[dict] = []
        for m in messages:
            role = m["role"]
            text_parts = [c["text"] for c in m["content"]
                           if c.get("type") == "text"]
            if not text_parts:
                continue
            flat.append({"role": role, "content": "\n".join(text_parts)})

        # Gemini reasoning tokens (Pro especially) count against the
        # max_output_tokens budget. 4096 is empirically too tight on
        # long-prefix probes; 16384 is the safe floor for stressor
        # paraphrases that occasionally consume 9000+ reasoning tokens.
        # Note: a small number of (prefix, paraphrase) combinations on
        # Gemini 2.5 Flash deterministically return output=0 even with
        # 32K budget — these are accepted as missing cells (≤1 of 120
        # per target on the stressor surface) rather than over-fit.
        text, in_tok, out_tok = call_gemini(
            self._client, self.model_id, flat,
            system=system_prompt or None,
            max_output_tokens=max(max_tokens, 16384),
        )
        if self.cost_tracker is not None:
            self.cost_tracker.log(
                session_id=self.session_id,
                role="target",
                model_id=self.model_id,
                input_tokens=in_tok,
                output_tokens=out_tok,
            )
        return TargetResponse(
            text=text,
            tool_calls=[],
            stop_reason="",
            raw_usage={"input_tokens": in_tok, "output_tokens": out_tok},
        )


def _json_dumps(obj) -> str:
    import json
    return json.dumps(obj)


def _json_loads(s: str):
    import json
    try:
        return json.loads(s)
    except Exception:
        return {}
