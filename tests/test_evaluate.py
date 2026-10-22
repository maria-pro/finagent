import pytest

from finagent import evaluate
from finagent.config import ABLATIONS

EVIDENCE = [
    {"id": "F1", "tool": "financial_statements"},
    {"id": "F2", "tool": "valuation_snapshot"},
    {"id": "R1", "tool": "value_at_risk"},
]


def test_variants_cover_every_ablation():
    every = evaluate.variants(["all"])
    assert every["full"] == frozenset()
    assert {name[1:] for name in every if name != "full"} == set(ABLATIONS)
    with pytest.raises(ValueError):
        evaluate.variants(["-nonsense"])


def test_facet_coverage_needs_cited_evidence_of_the_right_kind():
    question = "Is NVDA's valuation justified by its fundamentals, and what are the main risks?"
    full = "## Answer\nP/E 27.73 [F2] on growth [F1].\n## Risks\n- VaR 14% [R1]"
    partial = "## Answer\nGrowth was strong [F1].\n## Risks\n- volatile"
    assert evaluate.facet_coverage(question, ["NVDA"], full, EVIDENCE)["score"] == 1.0
    out = evaluate.facet_coverage(question, ["NVDA"], partial, EVIDENCE)
    assert out["facets"] == ["valuation", "fundamentals", "risk"] and out["addressed"] == ["fundamentals"]


def test_comparison_facet_requires_every_company_in_the_answer():
    question = "Compare NVDA and AMD on growth"
    assert evaluate.facet_coverage(question, ["NVDA", "AMD"], "## Answer\nNVDA grew faster than AMD [F1].", EVIDENCE)["score"] == 1.0
    assert evaluate.facet_coverage(question, ["NVDA", "AMD"], "## Answer\nNVDA grew fast [F1].", EVIDENCE)["score"] == 0.5


def test_summary_aggregates_per_variant():
    rows = [
        {"ticker": "A", "variant": "full", "ok": True, "wall_seconds": 10, "number_provenance": 1.0, "claim_attribution": 1.0,
         "draft_claim_attribution": 0.8, "facet_coverage": 1.0, "tool_calls": 4, "tool_calls_by_model": 4, "nudges": 0,
         "revisions": 1, "tokens": 1000, "claim_issues": [], "model": "m"},
        {"ticker": "B", "variant": "full", "ok": False, "error": "boom", "model": "m"},
    ]
    table = evaluate.summarize(rows)["full"]
    assert table["runs"] == 2 and table["succeeded"] == 1 and table["draft_claim_attribution"] == 0.8
    md = evaluate.summary_markdown(rows)
    assert "| full | 2 (1) |" in md and "B/full: boom" in md


def test_saved_runs_can_be_rescored(offline, tmp_path):
    from fakes import ScriptedLLM

    rows = evaluate.run_eval(offline.settings, ["NVDA"], ["valuation"], ["full"], tmp_path / "r.jsonl",
                             llm_factory=lambda s: ScriptedLLM(hallucinate=False))
    assert rows[0]["record"].endswith(".json")
    again = evaluate.rescore(tmp_path / "r-runs")
    assert len(again) == 1 and again[0]["variant"] == "full" and again[0]["ticker"] == "NVDA"
    for key in ("number_provenance", "claim_attribution", "facet_coverage", "tool_calls"):
        assert again[0][key] == rows[0][key]
