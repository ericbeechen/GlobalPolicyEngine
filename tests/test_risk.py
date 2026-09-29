"""The EWMA sigma and its floor: hand-computed weights, strictly before the session, and the floor's median.

Synthetic series only. Checks: sigma at t is the lambda-weighted mean of the
squares before t, normalised over the observations there are, NaN until one
mean age of history; nothing at or after t moves it; a missing observation is
skipped, not taken as 0; the floor binds at the configured multiple of the
trailing two-year median and never lifts a missing sigma.
"""

import numpy as np
import pandas as pd
import pytest
from policypath import config
from policypath.strategy import risk

DAYS = pd.bdate_range("2020-01-01", periods=400)


def by_hand(x, lam, t):
    """sigma at position t: sum lam^(j-1) x_(t-j)^2 / sum lam^(j-1) over the non-missing x before t."""
    past = x[:t][::-1]
    w = lam ** np.arange(len(past))
    ok = ~np.isnan(past)
    return np.sqrt((w[ok] * past[ok] ** 2).sum() / w[ok].sum())


def test_min_periods_is_one_mean_age():
    assert risk.min_periods(0.5) == 2
    assert risk.min_periods(config.strategy()["risk"]["ewma_lambda"]) == 34       # 1 / 0.03 = 33.3


def test_sigma_is_the_weighted_mean_of_squares_before_the_session():
    lam = 0.9
    x = np.random.default_rng(1).normal(0, 2, len(DAYS))
    x[50] = np.nan
    sigma = risk.ewma_sigma(pd.Series(x, index=DAYS), lam)
    n = risk.min_periods(lam)
    assert sigma.iloc[:n].isna().all() and sigma.iloc[n:].notna().all()
    for t in (n, n + 5, 51, 52, 200, 399):
        assert sigma.iloc[t] == pytest.approx(by_hand(x, lam, t), rel=1e-12)


def test_sigma_at_t_does_not_see_t_or_later():
    x = pd.Series(np.random.default_rng(2).normal(0, 1, len(DAYS)), index=DAYS)
    lam = config.strategy()["risk"]["ewma_lambda"]
    clean = risk.ewma_sigma(x, lam)
    for cut in (100, 250):
        poisoned = x.copy()
        poisoned.iloc[cut:] = 1e6
        got = risk.ewma_sigma(poisoned, lam)
        pd.testing.assert_series_equal(got.iloc[:cut + 1], clean.iloc[:cut + 1])
        assert got.iloc[cut + 1] > 1e3


def test_the_floor_is_the_multiple_of_the_trailing_two_year_median():
    sigma = pd.Series(1.0, index=DAYS)
    sigma.iloc[300:] = 0.1                          # vol collapses
    sigma.iloc[:5] = np.nan
    floored = risk.sigma_floor(sigma, 0.5)
    assert floored.iloc[:5].isna().all()
    assert (floored.iloc[5:300] == 1.0).all()
    # the median over the trailing two years is still 1 for a while, so the floor binds at 0.5
    assert floored.iloc[301] == 0.5
    window = sigma.loc[DAYS[399] - pd.Timedelta(risk.FLOOR_WINDOW) + pd.Timedelta(days=1):DAYS[399]]
    assert floored.iloc[399] == pytest.approx(max(0.1, 0.5 * window.median()))


def test_the_floor_uses_no_later_sigma():
    sigma = pd.Series(np.random.default_rng(3).uniform(0.5, 1.5, len(DAYS)), index=DAYS)
    clean = risk.sigma_floor(sigma, 0.5)
    poisoned = sigma.copy()
    poisoned.iloc[200:] = 1e6
    pd.testing.assert_series_equal(risk.sigma_floor(poisoned, 0.5).iloc[:200], clean.iloc[:200])
