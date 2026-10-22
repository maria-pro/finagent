import json

import pandas as pd
import pytest
from fakes import ScriptedLLM

from finagent import cli
from finagent.runs import list_runs, load_run, save_run


@pytest.fixture
def scripted(monkeypatch, offline, tmp_path):
    """Route the CLI's model and data to the scripted model and synthetic data."""
    import finagent.llm

    monkeypatch.setattr(finagent.llm, "OpenAICompatLLM", lambda settings: ScriptedLLM(hallucinate=False))
    monkeypatch.setenv("FINAGENT_CACHE_DIR", str(tmp_path / "cache"))
    return tmp_path


def test_tools_command(capsys):
    assert cli.main(["tools"]) == 0
    assert "price_performance" in capsys.readouterr().out


def test_bench_command(capsys):
    assert cli.main(["bench-risk", "--paths", "2000", "--horizon", "2", "--assets", "2"]) == 0
    assert "CPU (NumPy)" in capsys.readouterr().out


def test_analyze_saves_note_and_run_record(scripted, capsys):
    out_dir = scripted / "reports"
    assert cli.main(["analyze", "NVDA", "-q", "How risky is NVDA?", "--out-dir", str(out_dir), "--quiet"]) == 0
    saved = list(out_dir.glob("*.json"))
    assert len(saved) == 1 and saved[0].with_suffix(".md").exists()
    record = json.loads(saved[0].read_text())
    assert record["stats"]["quality"]["claims_checked"] > 0
    assert any(e["event"] == "verify" for e in record["events"])
    assert "fact-check" in capsys.readouterr().out


def test_doctor_reports_each_check(monkeypatch, offline, tmp_path, capsys):
    import finagent.data.prices as prices
    import finagent.llm

    class FakeLLM(ScriptedLLM):
        reasoning_style = "ollama"
        unsupported_params = set()

        def list_models(self):
            return ["nemotron-3-nano:4b"]

        def chat(self, messages, *, role, tools=None, json_schema=None, max_tokens=1024):
            if tools:
                from finagent.llm import LLMResponse, ToolCall

                return LLMResponse("", tool_calls=[ToolCall("t", "price_performance", {"ticker": "NVDA"})], usage=self._usage(role))
            if role == "specialist":
                from finagent.llm import LLMResponse

                return LLMResponse("ready", usage=self._usage(role))
            return super().chat(messages, role=role, tools=tools, json_schema=json_schema, max_tokens=max_tokens)

    monkeypatch.setattr(finagent.llm, "OpenAICompatLLM", lambda settings: FakeLLM())
    monkeypatch.setattr(prices, "adjusted_close", lambda s, t, r="10y": pd.Series([1.0, 2.0]))
    monkeypatch.setattr(prices, "risk_free_rate", lambda s: 0.04)
    monkeypatch.setenv("FINAGENT_CACHE_DIR", str(tmp_path))
    assert cli.main(["doctor"]) == 0
    out = capsys.readouterr().out
    for check in ("Reasoning control", "Tool calling", "Planner (JSON schema)", "SEC EDGAR", "Risk-free rate"):
        assert check in out


def test_eval_command_writes_results_and_summary(scripted, capsys):
    out = scripted / "eval" / "r.jsonl"
    assert cli.main(["eval", "--tickers", "NVDA", "AMD", "--variants", "full,-coverage", "--out", str(out)]) == 0
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [(r["ticker"], r["variant"]) for r in rows] == [("NVDA", "full"), ("NVDA", "-coverage"), ("AMD", "full"), ("AMD", "-coverage")]
    assert all(r["ok"] for r in rows)
    assert out.with_suffix(".md").read_text().startswith("# FinAgent evaluation")
    assert cli.main(["eval", "--summarize", str(out)]) == 0


def test_run_records_round_trip_and_skip_macos_metadata(scripted, settings):
    final = {"tickers": ["NVDA"], "question": "q?", "report": "# note", "stats": {"llm_calls": 1}}
    path = save_run(final, [{"event": "done"}], settings, scripted / "reports")
    (scripted / "reports" / "._junk.json").write_bytes(b"\x00\x05\x16\x07")  # AppleDouble file on exFAT drives
    assert list_runs((str(scripted / "reports"),)) == [path]
    record = load_run(path)
    assert record["report"] == "# note" and record["events"] == [{"event": "done"}]
    assert record["settings"]["profile"] == "local" and path.with_suffix(".md").read_text() == "# note"
