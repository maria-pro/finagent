import numpy as np
import pandas as pd
import pytest
from conftest import synthetic_prices

from finagent import analytics as an


def test_drawdown_finds_peak_and_trough():
    prices = pd.Series([100, 120, 90, 60, 80, 130], index=pd.bdate_range("2026-01-01", periods=6))
    dd = an.max_drawdown(prices)
    assert dd.max_drawdown == pytest.approx(-0.5)
    assert dd.peak == prices.index[1] and dd.trough == prices.index[3]


def test_volatility_matches_generating_process():
    returns = an.daily_returns(synthetic_prices(0, drift=0.0, vol=0.01, days=5000))
    assert an.annualized_volatility(returns) == pytest.approx(0.01 * np.sqrt(252), rel=0.05)


def test_beta_recovers_known_exposure():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2020-01-01", periods=3000)
    market = pd.Series(rng.normal(0, 0.01, 3000), index=idx)
    asset = 1.8 * market + pd.Series(rng.normal(0, 0.002, 3000), index=idx)
    stats = an.benchmark_stats(asset, market)
    assert stats.beta == pytest.approx(1.8, rel=0.02)
    assert stats.correlation > 0.95


def test_historical_var_of_normal_returns():
    rng = np.random.default_rng(2)
    returns = pd.Series(rng.normal(0, 0.01, 200_000))
    result = an.historical_var(returns, confidence=0.95)
    assert result.var == pytest.approx(1.645 * 0.01, rel=0.03)
    assert result.cvar > result.var


def test_monte_carlo_is_reproducible_and_close_to_historical():
    rng = np.random.default_rng(3)
    hist = rng.normal(0, 0.01, size=(2000, 2))
    weights = np.array([0.5, 0.5])
    a = an.monte_carlo_var(hist, weights, horizon_days=1, n_paths=200_000, seed=11)
    b = an.monte_carlo_var(hist, weights, horizon_days=1, n_paths=200_000, seed=11)
    assert a.var == b.var
    direct = an.historical_var(pd.Series(hist @ weights), 0.95)
    assert a.var == pytest.approx(direct.var, rel=0.05)


def test_rsi_bounds():
    rising = pd.Series(np.arange(1, 60, dtype=float))
    assert an.rsi(rising) == 100.0
    assert 0 <= an.rsi(synthetic_prices(4)) <= 100


def test_stress_test_skips_windows_without_data():
    prices = synthetic_prices(5, days=600)  # starts in 2024: no 2018/2020/2022 windows
    assert an.stress_test(prices, synthetic_prices(6, days=600)) == []
    full = synthetic_prices(5)
    episodes = an.stress_test(full, synthetic_prices(6))
    assert {e["episode"] for e in episodes} == set(an.STRESS_WINDOWS)
