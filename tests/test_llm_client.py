"""The real OpenAI SDK driven through a mocked HTTP transport: probing, reasoning
toggles, the parameter-fallback loop, tool-call parsing and concurrency."""

import json
import threading
import time

import httpx
import pytest
from openai import BadRequestError

from finagent.config import load_settings
from finagent.llm import OpenAICompatLLM

TOOL = {"type": "function", "function": {"name": "price_performance", "parameters": {"type": "object", "properties": {}}}}


def completion(content="ok", tool_calls=None, **message_extra):
    message = {"role": "assistant", "content": content, **message_extra}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": "c1", "object": "chat.completion", "created": 0, "model": "m",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
    }


class Server:
    """Records requests; ``replies`` is a list of (status, json) for successive chat calls."""

    def __init__(self, replies=None, ollama=False):
        self.replies = list(replies or [])
        self.ollama = ollama
        self.bodies: list[dict] = []
        self.probes = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/api/version"):
            self.probes += 1
            return httpx.Response(200, json={"version": "0.17.6"}) if self.ollama else httpx.Response(404)
        self.bodies.append(json.loads(request.content))
        status, payload = self.replies.pop(0) if self.replies else (200, completion())
        return httpx.Response(status, json=payload)


def make(server, **overrides):
    settings = load_settings("local").with_overrides(base_url="http://test/v1", **overrides)
    return OpenAICompatLLM(settings, http_client=httpx.Client(transport=httpx.MockTransport(server)))


def test_ollama_is_detected_by_probe_and_thinking_switched_off():
    server = Server(ollama=True)
    llm = make(server)
    llm.chat([{"role": "user", "content": "hi"}], role="specialist")
    llm.chat([{"role": "user", "content": "hi"}], role="specialist")
    assert server.probes == 1  # probed once, then remembered
    assert server.bodies[0]["reasoning_effort"] == "none" and "chat_template_kwargs" not in server.bodies[0]


def test_other_servers_get_chat_template_kwargs():
    server = Server()
    make(server).chat([{"role": "user", "content": "hi"}], role="planner")
    assert server.bodies[0]["chat_template_kwargs"] == {"enable_thinking": False}


def test_explicit_style_skips_the_probe():
    server = Server()
    make(server, reasoning_style="effort", reasoning=True).chat([{"role": "user", "content": "hi"}], role="writer")
    assert server.probes == 0 and server.bodies[0]["reasoning_effort"] == "medium"


def test_named_rejection_is_dropped_and_remembered():
    rejection = {"error": {"message": "Unsupported parameter: chat_template_kwargs", "type": "invalid_request_error"}}
    server = Server([(400, rejection), (200, completion("fine"))])
    llm = make(server, reasoning_style="template")
    first = llm.chat([{"role": "user", "content": "hi"}], role="specialist")
    assert first.content == "fine" and first.usage["dropped_params"] == ["chat_template_kwargs"]
    llm.chat([{"role": "user", "content": "again"}], role="specialist")
    assert "chat_template_kwargs" not in server.bodies[-1]  # not sent again
    assert llm.unsupported_params == {"chat_template_kwargs"}


def test_unnamed_rejection_drops_structured_output_first():
    server = Server([(400, {"error": {"message": "bad request"}}), (200, completion('{"tasks": []}'))])
    llm = make(server, reasoning_style="none")
    out = llm.chat([{"role": "user", "content": "plan"}], role="planner", json_schema={"title": "plan", "type": "object"})
    assert "response_format" in server.bodies[0] and "response_format" not in server.bodies[1]
    assert out.usage["dropped_params"] == ["response_format"]


def test_rejection_without_optional_parameters_is_raised():
    server = Server([(400, {"error": {"message": "context too long"}})])
    with pytest.raises(BadRequestError):
        make(server, reasoning_style="none").chat([{"role": "user", "content": "hi"}], role="specialist")


def test_native_and_text_tool_calls_and_reasoning_are_parsed():
    native = [{"id": "t1", "type": "function", "function": {"name": "price_performance", "arguments": '{"ticker": "NVDA"}'}}]
    text = 'Calling.<tool_call>{"name": "price_performance", "arguments": {"ticker": "AMD"}}</tool_call>'
    server = Server([(200, completion("", tool_calls=native)), (200, completion(text)), (200, completion("Answer", reasoning_content="thought"))])
    llm = make(server, reasoning_style="none")
    a = llm.chat([{"role": "user", "content": "x"}], role="specialist", tools=[TOOL])
    b = llm.chat([{"role": "user", "content": "x"}], role="specialist", tools=[TOOL])
    c = llm.chat([{"role": "user", "content": "x"}], role="specialist")
    assert a.tool_calls[0].arguments == {"ticker": "NVDA"} and a.tool_calls[0].native
    assert b.tool_calls[0].arguments == {"ticker": "AMD"} and not b.tool_calls[0].native and b.content == "Calling."
    assert c.content == "Answer" and c.reasoning == "thought"
    assert a.usage["prompt_tokens"] == 7 and a.usage["model"] == "nemotron-3-nano:4b"


def test_requests_queue_when_the_server_takes_one_at_a_time():
    in_flight, peak, lock = [0], [0], threading.Lock()

    def slow(request):
        if request.url.path.endswith("/api/version"):
            return httpx.Response(404)
        with lock:
            in_flight[0] += 1
            peak[0] = max(peak[0], in_flight[0])
        time.sleep(0.05)
        with lock:
            in_flight[0] -= 1
        return httpx.Response(200, json=completion())

    llm = make(slow, max_concurrency=1, reasoning_style="none")
    threads = [threading.Thread(target=llm.chat, args=([{"role": "user", "content": "hi"}],), kwargs={"role": "specialist"}) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak[0] == 1
