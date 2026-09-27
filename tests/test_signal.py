"""The gap signal: market minus model in bp, and its trailing z."""

import numpy as np
import pandas as pd
import pytest
from policypath import config
from policypath.signal.gap import build, gaps, zscore

SPEC = config.currency("USD")["signal"]


def long(market, model, days):
    return pd.DataFrame({"session": np.repeat(days, 2), "k": [1, 2] * len(days),
                         "market": np.ravel(market), "model": np.ravel(model)})


def test_the_gap_is_market_minus_model_in_bp():
    days = pd.to_datetime(["2024-01-02", "2024-01-03"])
    g = gaps(long([[4.00, 3.80], [4.10, 3.90]], [[4.20, 4.25], [4.20, 4.25]], days))
    # the market prices 20 and 45bp less than the rule: negative
    assert g.loc["2024-01-02"].tolist() == pytest.approx([-20.0, -45.0])
    assert g.loc["2024-01-03"].tolist() == pytest.approx([-10.0, -35.0])


def test_a_session_is_not_in_its_own_window():
    days = pd.bdate_range("2020-01-01", periods=300)
    gap = pd.DataFrame({1: np.random.default_rng(1).normal(0, 20, 300)}, index=days)
    spec = {**SPEC, "min_periods": 50}
    z, mean, sd = zscore(gap, spec)
    t = days[200]
    prior = gap.loc[(gap.index < t) & (gap.index >= t - pd.Timedelta(spec["window"])), 1]
    assert mean.loc[t, 1] == pytest.approx(prior.mean())
    assert z.loc[t, 1] == pytest.approx((gap.loc[t, 1] - prior.mean()) / prior.std())
    # and a shock at t moves z at t, but not the window's moments at t
    shocked = gap.copy()
    shocked.loc[t, 1] += 500
    z2, mean2, _ = zscore(shocked, spec)
    assert mean2.loc[t, 1] == mean.loc[t, 1] and z2.loc[t, 1] > z.loc[t, 1]


def test_no_z_before_min_periods():
    days = pd.bdate_range("2020-01-01", periods=40)
    gap = pd.DataFrame({1: np.arange(40.0)}, index=days)
    z, _, _ = zscore(gap, {**SPEC, "min_periods": 10})
    assert z[1].iloc[:10].isna().all() and z[1].iloc[10:].notna().all()


def test_the_sd_floor_stops_a_quiet_window_from_blowing_up():
    days = pd.bdate_range("2013-01-01", periods=400)
    quiet = pd.DataFrame({1: np.random.default_rng(2).normal(0, 0.5, 400)}, index=days)
    quiet.iloc[-1, 0] = 3.0                                   # a 3bp move after a year of noise
    z, _, sd = zscore(quiet, {**SPEC, "min_periods": 250})
    assert sd[1].iloc[-1] == SPEC["sd_floor_bp"] == 5
    assert abs(z[1].iloc[-1]) < 1


def test_build_keeps_both_bp_and_z():
    days = pd.bdate_range("2020-01-01", periods=5)
    out = build(long(np.full((5, 2), 1.0), np.full((5, 2), 1.1), days), {**SPEC, "min_periods": 2})
    assert set(out.columns) >= {"session", "k", "gap_bp", "z", "window_mean_bp", "window_sd_bp"}
    assert out["gap_bp"].to_numpy() == pytest.approx(-10.0)
