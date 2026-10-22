from finagent.llm import extract_json, parse_text_tool_calls, split_reasoning


def test_json_tool_call_in_text():
    content, calls = parse_text_tool_calls('Let me check.<tool_call>{"name": "price_performance", "arguments": {"ticker": "NVDA"}}</tool_call>')
    assert content == "Let me check."
    assert calls[0].name == "price_performance" and calls[0].arguments == {"ticker": "NVDA"} and not calls[0].native


def test_nemotron_toolcall_list():
    _, calls = parse_text_tool_calls('<TOOLCALL>[{"name": "a", "arguments": {"x": 1}}, {"name": "b", "arguments": "{\\"y\\": 2}"}]</TOOLCALL>')
    assert [(c.name, c.arguments) for c in calls] == [("a", {"x": 1}), ("b", {"y": 2})]


def test_xml_style_tool_call():
    text = "<tool_call><function=value_at_risk><parameter=tickers>[\"NVDA\", \"AMD\"]</parameter><parameter=horizon_days>10</parameter></function></tool_call>"
    _, calls = parse_text_tool_calls(text)
    assert calls[0].name == "value_at_risk"
    assert calls[0].arguments == {"tickers": ["NVDA", "AMD"], "horizon_days": 10}


def test_plain_text_is_left_alone():
    content, calls = parse_text_tool_calls("No tools needed.")
    assert content == "No tools needed." and calls == []


def test_reasoning_is_split_out():
    assert split_reasoning("<think>check data</think>Answer") == ("Answer", "check data")
    assert split_reasoning("hidden plan</think>Answer") == ("Answer", "hidden plan")


def test_extract_json_tolerates_fences_and_preamble():
    assert extract_json('Here you go:\n```json\n{"tasks": []}\n```') == {"tasks": []}
    assert extract_json('Plan: {"tasks": [{"agent": "risk"}]} done') == {"tasks": [{"agent": "risk"}]}
