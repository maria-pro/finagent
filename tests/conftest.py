"""Offline fixtures: synthetic prices, SEC facts and filing text, so tests need no network."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from finagent.config import load_settings  # noqa: E402
from finagent.tools import ToolContext  # noqa: E402


def synthetic_prices(seed: int, drift: float = 0.0004, vol: float = 0.02, days: int = 2600, start: float = 100.0) -> pd.Series:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(end="2026-10-21", periods=days)
    returns = rng.normal(drift, vol, size=days)
    series = pd.Series(start * np.cumprod(1 + returns), index=index)
    series.attrs["meta"] = {"name": "Synthetic Corp"}
    return series


def fact(start, end, val, form="10-K", filed="2026-02-25"):
    row = {"end": end, "val": val, "form": form, "filed": filed, "accn": "0000000000-26-000001"}
    if start:
        row["start"] = start
    return row


SYNTHETIC_FACTS = {
    "facts": {
        "dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": [fact(None, "2026-08-21", 10e9, "10-Q", "2026-08-26")]}}},
        "us-gaap": {
            "Revenues": {
                "units": {
                    "USD": [
                        fact("2023-01-30", "2024-01-28", 60e9, filed="2024-02-21"),
                        fact("2024-01-29", "2025-01-26", 130e9, filed="2025-02-26"),
                        fact("2025-01-27", "2026-01-25", 216e9),
                        fact("2025-01-27", "2025-07-27", 90e9, "10-Q", "2026-08-26"),
                        fact("2026-01-26", "2026-07-26", 178e9, "10-Q", "2026-08-26"),
                        fact("2026-04-27", "2026-07-26", 96e9, "10-Q", "2026-08-26"),
                    ]
                }
            },
            "NetIncomeLoss": {
                "units": {
                    "USD": [
                        fact("2023-01-30", "2024-01-28", 30e9, filed="2024-02-21"),
                        fact("2024-01-29", "2025-01-26", 73e9, filed="2025-02-26"),
                        fact("2025-01-27", "2026-01-25", 120e9),
                        fact("2025-01-27", "2025-07-27", 45e9, "10-Q", "2026-08-26"),
                        fact("2026-01-26", "2026-07-26", 95e9, "10-Q", "2026-08-26"),
                    ]
                }
            },
            "GrossProfit": {"units": {"USD": [fact("2025-01-27", "2026-01-25", 150e9)]}},
            "OperatingIncomeLoss": {"units": {"USD": [fact("2025-01-27", "2026-01-25", 130e9)]}},
            "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": [fact("2025-01-27", "2026-01-25", 100e9)]}},
            "PaymentsToAcquirePropertyPlantAndEquipment": {"units": {"USD": [fact("2025-01-27", "2026-01-25", 5e9)]}},
            "CashAndCashEquivalentsAtCarryingValue": {"units": {"USD": [fact(None, "2026-07-26", 20e9, "10-Q", "2026-08-26")]}},
            "LongTermDebt": {"units": {"USD": [fact(None, "2026-07-26", 8e9, "10-Q", "2026-08-26")]}},
        },
    }
}

SYNTHETIC_10K = {
    "risk_factors": (
        "Item 1A. Risk Factors\nOur revenue is concentrated: two customers accounted for 35% of revenue.\n"
        "Export controls on advanced chips could reduce demand for our products in China and other markets.\n"
        "We depend on a single foundry partner for manufacturing, which creates supply chain risk."
    ),
    "mda": "Item 7. Management's Discussion\nData center revenue grew strongly as customers built AI infrastructure.",
}


@pytest.fixture
def settings(tmp_path):
    return load_settings("local").with_overrides(cache_dir=tmp_path, max_concurrency=4)


@pytest.fixture
def offline(monkeypatch, settings):
    """Route every data access to synthetic data."""
    from finagent.data import sec
    from finagent.tools import fundamentals, market, risk

    seeds = {"SPY": 1, "NVDA": 2, "AMD": 3}

    def fake_close(_settings, ticker, range_="10y"):
        return synthetic_prices(seeds.get(ticker, 9), drift=0.0003 if ticker == "SPY" else 0.0008).rename(ticker)

    for module in (market, risk, fundamentals):
        monkeypatch.setattr(module, "adjusted_close", fake_close)
    monkeypatch.setattr(market, "risk_free_series", lambda _s, start: pd.Series(0.04, index=pd.bdate_range(start, "2026-10-21")))
    monkeypatch.setattr(sec, "ticker_to_cik", lambda _s, t: (1, f"{t} CORP"))
    monkeypatch.setattr(sec, "company_facts", lambda _s, _cik: SYNTHETIC_FACTS)
    monkeypatch.setattr(
        sec,
        "annual_report_sections",
        lambda _s, t: ({"form": "10-K", "accession": f"acc-{t}", "filed": "2026-02-25", "period": "2026-01-25", "url": "https://sec.example/10k"}, SYNTHETIC_10K),
    )
    return ToolContext(settings)
