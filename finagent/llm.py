"""One client for every OpenAI-compatible endpoint FinAgent targets.

The NVIDIA API catalog, a self-hosted NIM container, Ollama and llama.cpp all
speak the same chat-completions protocol, so the agents never know which one is
answering. The differences that do matter are smoothed over here:

* how reasoning ("thinking") is switched off,
* small models that print a tool call as text instead of returning it in
  ``tool_calls``,
* reasoning text returned inline as ``<think>...</think>``,
* structured output (JSON schema), which constrains decoding on llama.cpp and
  Ollama and makes a 4B model's JSON reliable.
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

from openai import BadRequestError, OpenAI

from .config import Settings


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    native: bool = True  # False when recovered from text rather than the API's tool_calls field


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    reasoning: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)


class LLM(Protocol):
    def chat(
        self,
        messages: list[dict],
        *,
        role: str,
        tools: list[dict] | None = None,
        json_schema: dict | None = None,
        max_tokens: int = 1024,
    ) -> LLMResponse: ...


# --------------------------------------------------------------------------- parsing helpers

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_TOOL_BLOCK_RE = re.compile(r"<(tool_call|TOOLCALL)>(.*?)</\1>", re.DOTALL)
_XML_FUNC_RE = re.compile(r"<function=([\w.\-]+)>(.*?)</function>", re.DOTALL)
_XML_PARAM_RE = re.compile(r"<parameter=([\w.\-]+)>(.*?)</parameter>", re.DOTALL)


def split_reasoning(content: str) -> tuple[str, str | None]:
    """Remove ``<think>`` blocks (and a dangling unterminated one) from model output."""
    thoughts = [m.strip() for m in _THINK_RE.findall(content)]
    content = _THINK_RE.sub("", content)
    if "</think>" in content:  # template opened <think> in the prompt; only the close tag is emitted
        head, _, content = content.partition("</think>")
        thoughts.append(head.strip())
    return content.strip(), ("\n".join(t for t in thoughts if t) or None)


def _coerce_param(raw: str) -> Any:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw


def parse_text_tool_calls(content: str) -> tuple[str, list[ToolCall]]:
    """Recover tool calls that a model wrote into its message text.

    Handles the JSON form used by Hermes/Nemotron templates
    (``<tool_call>{"name": ..., "arguments": {...}}</tool_call>`` and
    ``<TOOLCALL>[...]</TOOLCALL>``) and the XML form used by Qwen3-Coder style
    parsers (``<function=name><parameter=x>...</parameter></function>``).
    """
    calls: list[ToolCall] = []
    for _, body in _TOOL_BLOCK_RE.findall(content):
        body = body.strip()
        xml_funcs = _XML_FUNC_RE.findall(body)
        if xml_funcs:
            for name, params in xml_funcs:
                args = {k: _coerce_param(v) for k, v in _XML_PARAM_RE.findall(params)}
                calls.append(ToolCall(f"text_{uuid.uuid4().hex[:8]}", name, args, native=False))
            continue
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            continue
        for item in payload if isinstance(payload, list) else [payload]:
            if not isinstance(item, dict) or "name" not in item:
                continue
            args = item.get("arguments", item.get("parameters", {}))
            if isinstance(args, str):
                args = _coerce_param(args)
            calls.append(
                ToolCall(f"text_{uuid.uuid4().hex[:8]}", str(item["name"]), args if isinstance(args, dict) else {}, native=False)
            )
    cleaned = _TOOL_BLOCK_RE.sub("", content).strip() if calls else content
    return cleaned, calls


def extract_json(text: str) -> Any:
    """Parse the first JSON object or array in ``text`` (tolerates code fences and preamble)."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if not starts:
        raise ValueError("no JSON found in model output")
    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(text[min(starts):])
    return obj


# --------------------------------------------------------------------------- client


def reasoning_style(settings: Settings) -> str:
    """How this endpoint expects reasoning to be toggled."""
    if ":11434" in settings.base_url:
        return "ollama"  # Ollama's OpenAI layer maps reasoning_effort onto its `think` option
    return "template"  # NIM, vLLM and llama.cpp forward chat_template_kwargs to the chat template


class OpenAICompatLLM:
    """Chat client with per-role model routing and graceful capability fallbacks."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = OpenAI(
            base_url=settings.base_url,
            api_key=settings.api_key or "not-needed",
            timeout=settings.timeout_s,
            max_retries=2,
        )
        self._style = reasoning_style(settings)
        self._unsupported: set[str] = set()
        self._lock = threading.Lock()
        # Agents may run in parallel, but a CPU inference server handles one request
        # at a time well; queue requests here rather than overloading the server.
        self._slots = threading.BoundedSemaphore(max(1, settings.max_concurrency))

    def _reasoning_params(self, enabled: bool) -> tuple[dict, dict]:
        """Return (top-level kwargs, extra_body) that switch reasoning on or off."""
        if self._style == "ollama" and "reasoning_effort" not in self._unsupported:
            return {"reasoning_effort": "medium" if enabled else "none"}, {}
        if self._style == "template" and "chat_template_kwargs" not in self._unsupported:
            return {}, {"chat_template_kwargs": {"enable_thinking": enabled}}
        return {}, {}

    def _create(self, kwargs: dict, extra_body: dict):
        """Call the endpoint, dropping optional parameters the server rejects."""
        while True:
            try:
                return self._client.chat.completions.create(**kwargs, extra_body=extra_body or None)
            except BadRequestError as err:
                message = str(err).lower()
                for optional in ("chat_template_kwargs", "reasoning_effort", "response_format"):
                    if optional in message and (optional in kwargs or optional in extra_body):
                        with self._lock:
                            self._unsupported.add(optional)
                        kwargs.pop(optional, None)
                        extra_body.pop(optional, None)
                        break
                else:
                    raise

    def chat(
        self,
        messages: list[dict],
        *,
        role: str,
        tools: list[dict] | None = None,
        json_schema: dict | None = None,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        model = self.settings.model_for(role)
        top, extra = self._reasoning_params(self.settings.reasoning)
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": self.settings.temperature,
            "top_p": self.settings.top_p,
            "max_tokens": max_tokens,
            **top,
        }
        if tools:
            kwargs["tools"] = tools
        if json_schema and "response_format" not in self._unsupported:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": json_schema.get("title", "output"), "schema": json_schema},
            }

        with self._slots:
            started = time.perf_counter()  # measured inside the slot: excludes queueing
            resp = self._create(kwargs, extra)
            latency = time.perf_counter() - started

        msg = resp.choices[0].message
        content, inline_reasoning = split_reasoning(msg.content or "")
        reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or inline_reasoning

        calls = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw": tc.function.arguments}
            calls.append(ToolCall(tc.id or f"call_{uuid.uuid4().hex[:8]}", tc.function.name, args if isinstance(args, dict) else {}))
        if tools and not calls:
            content, calls = parse_text_tool_calls(content)

        usage = resp.usage
        return LLMResponse(
            content=content,
            tool_calls=calls,
            reasoning=reasoning,
            usage={
                "role": role,
                "model": model,
                "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                "latency_s": round(latency, 2),
            },
        )


def tool_call_message(content: str, calls: list[ToolCall]) -> dict:
    """The assistant turn that must precede tool results in the conversation."""
    return {
        "role": "assistant",
        "content": content or "",
        "tool_calls": [
            {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
            for c in calls
        ],
    }


def usage_record(response: LLMResponse, agent: str) -> dict:
    return {"agent": agent, **response.usage}


__all__ = [
    "LLM",
    "LLMResponse",
    "OpenAICompatLLM",
    "ToolCall",
    "extract_json",
    "parse_text_tool_calls",
    "split_reasoning",
    "tool_call_message",
    "usage_record",
]
