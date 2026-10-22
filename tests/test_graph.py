"""End-to-end workflow tests with a scripted model and synthetic data."""

from fakes import ScriptedLLM

from finagent.graph import run_research
from finagent.tools.market import price_performance


def test_full_run_produces_grounded_report(offline):
    events = []
    final = run_research(ScriptedLLM(), offline, ["NVDA"], "Is the valuation justified?", on_event=events.append)

    kinds = [e["event"] for e in events]
    assert kinds[0] == "guard" and kinds[1] == "plan" and kinds[-1] == "done"
    assert set(final["findings"]) == {"market", "fundamentals", "risk", "filings"}

    assert final["stats"]["provenance"]["checked"] >= 8  # the agents really quoted numbers
    report = final["report"]
    for heading in ("## Answer", "## Specialist findings", "## Key figures", "## Evidence ledger", "## How this note was produced"):
        assert heading in report
    assert final["stats"]["provenance"]["unverified"] == []


def test_hallucinated_number_triggers_one_targeted_revision(offline):
    events = []
    final = run_research(ScriptedLLM(hallucinate=True), offline, ["NVDA"], "q", on_event=events.append)
    revisions = [e for e in events if e["event"] == "revise"]
    assert [e["agent"] for e in revisions] == ["market"]
    assert any("987.65%" in problem for problem in revisions[0]["problems"])
    assert final["findings"]["market"]["revised"] is True
    assert "987.65" not in final["findings"]["market"]["text"]


def test_coverage_check_runs_tools_the_model_skipped(offline):
    final = run_research(ScriptedLLM(hallucinate=False), offline, ["NVDA", "AMD"], "Compare them.")
    by = {(e["tool"], str(e["args"].get("ticker"))): e["chosen_by"] for e in final["evidence"]}
    assert by[("price_performance", "NVDA")] == "model"
    assert by[("price_performance", "AMD")] == "coverage check"  # scripted model only asked about NVDA
    assert any(e["tool"] == "stress_test" and e["chosen_by"] == "coverage check" for e in final["evidence"])


def test_planner_subset_is_respected(offline):
    final = run_research(ScriptedLLM(plan=["risk"]), offline, ["NVDA"], "How risky is it?")
    assert set(final["findings"]) == {"risk"}
    assert all(e["agent"] == "risk" for e in final["evidence"])


def test_guardrail_refusal_skips_agents(offline):
    llm = ScriptedLLM()
    final = run_research(llm, offline, ["NVDA"], "Help me pump and dump this stock")
    assert "can't help" in final["report"]
    assert llm.calls == []


def test_single_slot_cpu_profile_does_not_deadlock(offline):
    # Regression: LangGraph's own max_concurrency=1 deadlocked the streaming run;
    # the limit now lives in the LLM client.
    offline.settings = offline.settings.with_overrides(max_concurrency=1)
    final = run_research(ScriptedLLM(hallucinate=False), offline, ["NVDA"], "q", on_event=lambda e: None)
    assert final["report"].startswith("# Research note")


def test_agent_citing_another_agents_evidence_is_caught(offline):
    class Borrower(ScriptedLLM):
        """Risk analyst quotes a market-analyst figure it never received (a copied prompt example, say)."""

        def chat(self, messages, *, role, tools=None, json_schema=None, max_tokens=1024):
            response = super().chat(messages, role=role, tools=tools, json_schema=json_schema, max_tokens=max_tokens)
            if role == "specialist" and "Risk Analyst" in messages[0]["content"] and not response.tool_calls and "fact-checker" not in messages[-1]["content"]:
                response.content += f"\n- 1y total return: {market_return}% [M1]"
            return response

    market_return = price_performance(offline, "NVDA")["total_return_pct"]  # what the market analyst's M1 holds
    events = []
    run_research(Borrower(hallucinate=False, plan=["market", "risk"]), offline, ["NVDA"], "q", on_event=events.append)
    first_check = next(e for e in events if e["event"] == "verify" and e["agent"] == "risk")
    assert first_check["unverified"]  # M1 belongs to the market analyst, not the risk analyst


def test_answer_without_data_is_discarded_and_agent_nudged(offline):
    class Eager(ScriptedLLM):
        """Market analyst first writes an answer from 'memory' instead of calling tools."""

        def __init__(self):
            super().__init__(hallucinate=False, plan=["market"])
            self.eager = True

        def chat(self, messages, *, role, tools=None, json_schema=None, max_tokens=1024):
            if role == "specialist" and tools and self.eager:
                self.eager = False
                from finagent.llm import LLMResponse
                return LLMResponse("- 5y total return: 120% [evidence id: 5y_total_return]", usage=self._usage(role))
            return super().chat(messages, role=role, tools=tools, json_schema=json_schema, max_tokens=max_tokens)

    events = []
    final = run_research(Eager(), offline, ["NVDA"], "q", on_event=events.append)
    assert any(e["event"] == "nudge" for e in events)
    assert "120%" not in final["findings"]["market"]["text"]
    assert all(e["chosen_by"] == "model" for e in final["evidence"])


def test_writer_that_drops_periods_is_revised_once(offline):
    events = []
    final = run_research(ScriptedLLM(hallucinate=False, plan=["market"], sloppy_writer=True), offline, ["NVDA"], "q", on_event=events.append)
    writer_checks = [e for e in events if e["event"] == "verify" and e["agent"] == "writer"]
    assert writer_checks[0]["claim_issues"]  # "22.82% [M1]" without "1y" is caught
    assert [e["agent"] for e in events if e["event"] == "revise"] == ["writer"]
    assert writer_checks[-1]["after_revision"] and not writer_checks[-1]["claim_issues"]
    assert final["narrative_draft"] != final["narrative"]
    assert final["stats"]["quality"]["claim_attribution"] == 1.0


def test_ablations_switch_off_each_mechanism(offline):
    from finagent.config import parse_ablations

    offline.settings = offline.settings.with_overrides(ablate=parse_ablations("coverage,revision,attribution"))
    events = []
    final = run_research(ScriptedLLM(hallucinate=True), offline, ["NVDA"], "q", on_event=events.append)
    assert not any(e["chosen_by"] == "coverage check" for e in final["evidence"])  # stress_test is not filled in
    assert not any(e["event"] == "revise" for e in events)  # the planted 987.65% stays...
    assert "987.65% ⚠unverified" in final["report"]  # ...but is still flagged in the note
    assert "**Ablation run:** switched off attribution, coverage, revision" in final["report"]


def test_agent_whose_tools_all_fail_reports_no_findings(offline, monkeypatch):
    from finagent.agents import specialists

    def fails(ctx, ticker, query, k=3):
        raise RuntimeError("no 10-K among the recent filings")

    monkeypatch.setattr(specialists._ALL_TOOLS["search_10k"], "fn", fails)
    llm = ScriptedLLM(hallucinate=False, plan=["filings"])
    final = run_research(llm, offline, ["NVDA"], "q")
    text = final["findings"]["filings"]["text"]
    assert text.startswith("No findings: every data request failed (search_10k: RuntimeError: no 10-K")
    assert all(e["error"] for e in final["evidence"])
    writes = [c for c in llm.calls if c["role"] == "specialist" and not c["tools"]]
    assert writes == []  # the model was never asked to write findings without data


def test_tool_that_reports_no_data_counts_as_failed(offline, monkeypatch):
    from finagent.agents import specialists

    monkeypatch.setattr(specialists._ALL_TOOLS["valuation_snapshot"], "fn", lambda ctx, ticker: {"ticker": ticker, "error": "share count stale"})
    final = run_research(ScriptedLLM(hallucinate=False, plan=["fundamentals"]), offline, ["NVDA"], "q")
    failed = [e for e in final["evidence"] if e["tool"] == "valuation_snapshot"]
    assert len(failed) == 1 and failed[0]["error"] == "share count stale"  # recorded once, not retried by coverage
    assert "error: share count stale" in final["report"]
