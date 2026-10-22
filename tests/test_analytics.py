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


def test_filtered_simulation_is_reproducible_and_matches_historical_when_volatility_is_constant():
    rng = np.random.default_rng(3)
    hist = rng.normal(0, 0.01, size=(2000, 2))
    weights = np.array([0.5, 0.5])
    a = an.filtered_historical_simulation(hist, weights, horizon_days=1, n_paths=200_000, seed=11)
    b = an.filtered_historical_simulation(hist, weights, horizon_days=1, n_paths=200_000, seed=11)
    assert a.var == b.var
    direct = an.historical_var(pd.Series(hist @ weights), 0.95)
    assert a.var == pytest.approx(direct.var, rel=0.08)


def _regime(calm_first: bool) -> np.ndarray:
    rng = np.random.default_rng(5)
    calm, wild = rng.normal(0, 0.005, 1000), rng.normal(0, 0.03, 120)
    return np.concatenate([calm, wild] if calm_first else [wild, calm])


def test_filtered_simulation_responds_to_the_current_volatility_regime():
    # Historical VaR weighs every day equally; FHS scales to today's volatility, so the
    # two disagree in the direction of the latest regime - which is the point of a second view.
    for calm_first, higher in ((True, True), (False, False)):
        r = _regime(calm_first)
        fhs = an.filtered_historical_simulation(r, np.array([1.0]), horizon_days=10, n_paths=50_000)
        hist = an.historical_var(pd.Series(r), 0.95, 10)
        assert (fhs.var > hist.var) is higher
        assert (fhs.current_vol_annual > fhs.average_vol_annual) is higher


def test_sharpe_uses_the_rate_that_applied_each_day():
    idx = pd.bdate_range("2021-01-01", periods=500)
    returns = pd.Series(np.random.default_rng(6).normal(0.0006, 0.01, 500), index=idx)
    assert an.sharpe_ratio(returns, pd.Series(0.03, index=idx)) == pytest.approx(an.sharpe_ratio(returns, 0.03))
    # Rates near zero for most of the window, then 5% at the end: today's rate overstates the hurdle.
    dated = pd.Series(np.where(np.arange(500) < 450, 0.0005, 0.05), index=idx)
    assert an.sharpe_ratio(returns, dated) > an.sharpe_ratio(returns, 0.05)
    # A sparse rate series (weekly fixes) is forward-filled onto trading days.
    weekly = dated.iloc[::5]
    assert an.sharpe_ratio(returns, weekly) == pytest.approx(an.sharpe_ratio(returns, dated), rel=0.02)


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
