import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("null_baseline", Path(__file__).parents[1] / "scripts" / "null_baseline.py")
null_baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(null_baseline)

LEDGER = [
    {"id": "M1", "result": {"facts": [f"metric {i}: {10 + i * 0.37:.2f}%" for i in range(200)]}},
    {"id": "F1", "result": {"pe_ratio_ttm": 23.1, "market_cap_bn": 5216.93}},
]


def test_whole_number_percentages_no_longer_trace_by_chance():
    uncited = null_baseline.false_pass_rates([LEDGER], draws=200, seed=1)
    assert uncited["23%"] == 0.0  # rounded numbers need a citation now
    cited = null_baseline.false_pass_rates([LEDGER], draws=200, seed=1, cite=True)
    assert cited["23%"] < 0.6  # only the cited evidence can match, and a P/E never does
    assert set(cited) == set(null_baseline.FORMATS)


def test_baseline_reads_saved_runs(tmp_path):
    (tmp_path / "run.json").write_text('{"evidence": [{"id": "M1", "result": {"x_pct": 12.5}}]}')
    (tmp_path / "._run.json").write_bytes(b"\x00")
    assert null_baseline.load_ledgers(tmp_path) == [[{"id": "M1", "result": {"x_pct": 12.5}}]]
    with pytest.raises(SystemExit):
        null_baseline.main(["--help"])
