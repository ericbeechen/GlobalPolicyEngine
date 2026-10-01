"""The covariance scored as a forecast (`strategy/scoring.py`) on hand-made numbers.

Checks: QLIKE's ranking form differs from r/h - log(r/h) - 1 by a term free
of the forecast; the bias statistic, the kurtosis and the Diebold-Mariano t
are their formulas; a matrix at the horizon is the one decided that many
sessions back; the fixed portfolios are rescaled to a configured forecast of
1 and the random ones are the seed's; a correct covariance scores
realized-alpha portfolios at 1 and one that understates a direction's risk
above it; the configured candidate is the book's estimate and every other
moves one choice; the benchmark on iid Gaussian P&L is a little above 1, the
estimate's noise, and nothing up to a session reads the P&L after it.
"""

import numpy as np
import pandas as pd
from policypath.backtest import metrics
from policypath.strategy import portfolio, risk, scoring

DAYS = pd.bdate_range("2015-01-01", periods=900)


def gaussian(cov, t=len(DAYS), seed=3):
    x = np.random.default_rng(seed).multivariate_normal(np.zeros(len(cov)), cov, t)
    return pd.DataFrame(x, index=DAYS[:t] if t <= len(DAYS) else pd.bdate_range("2000-01-03", periods=t),
                        columns=list("abc")[:len(cov)])


COV = np.array([[1.0, 0.3, 0.0], [0.3, 2.0, 0.5], [0.0, 0.5, 0.5]])


def test_qlike_ranks_as_the_normalised_loss_does():
    r, h1, h2 = np.array([0.5, 2.0, 1.3]), np.array([1.0, 1.5, 0.7]), np.array([0.8, 2.5, 1.1])
    full = lambda h: r / h - np.log(r / h) - 1.0
    np.testing.assert_allclose(scoring.qlike(r, h1) - scoring.qlike(r, h2), full(h1) - full(h2))
    assert np.allclose(full(r), 0.0)
    assert np.isfinite(scoring.qlike(np.array([0.0]), np.array([1.0]))).all()       # an unchanged settle


def test_bias_kurtosis_and_dm_are_their_formulas():
    r, h = np.array([[1.0], [4.0], [0.25]]), np.array([[1.0], [2.0], [1.0]])
    assert scoring.bias(r, h) == np.sqrt((1.0 + 2.0 + 0.25) / 3)
    z2 = np.array([1.0, 2.0, 0.25])
    assert np.isclose(scoring.kurtosis(r, h)[0], (z2 ** 2).mean() / z2.mean() ** 2)
    d = np.random.default_rng(0).normal(0.1, 1.0, 400)
    assert np.isclose(scoring.dm(d, 5), d.mean() / np.sqrt(metrics.long_run_var(d, 5) / len(d)))
    assert np.isclose(scoring.dm(d, 0), d.mean() / d.std(ddof=0) * np.sqrt(len(d)))


def test_a_matrix_at_the_horizon_is_the_one_decided_that_many_sessions_back():
    m = np.arange(10.0)[:, None, None] * np.ones((10, 2, 2))
    got = scoring.at_horizon(m, 2)
    assert np.isnan(got[:2]).all()
    np.testing.assert_array_equal(got[2:], m[:8])


def test_fixed_portfolios_are_rescaled_to_a_configured_forecast_of_one():
    x = gaussian(COV, 300)
    mats = {scoring.CONFIGURED: np.broadcast_to(COV, (300, 3, 3)), "double": np.broadcast_to(2 * COV, (300, 3, 3))}
    w = scoring.random_weights(np.ones((300, 3)), draws=7, seed=1)
    rows = np.ones(300, dtype=bool)
    r, h = scoring.fixed(x.to_numpy(), mats, w, rows)
    np.testing.assert_allclose(h[scoring.CONFIGURED], 1.0)
    np.testing.assert_allclose(h["double"], 2.0)
    np.testing.assert_array_equal(w, scoring.random_weights(np.ones((300, 3)), draws=7, seed=1))
    s = scoring.score(r, h, 0)
    assert s["candidates"][scoring.CONFIGURED]["qlike"] == 0.0 and np.isnan(s["candidates"][scoring.CONFIGURED]["qlike_t"])
    assert 0.9 < s["candidates"][scoring.CONFIGURED]["b"] < 1.1                    # the right covariance
    assert np.isclose(s["candidates"]["double"]["b"], s["candidates"][scoring.CONFIGURED]["b"] / np.sqrt(2))


def test_realized_alpha_scores_a_correct_covariance_at_one_and_an_understated_direction_above():
    t = 21 * 600
    x = gaussian(COV, t, seed=5).to_numpy()
    low = COV.copy()
    low[2, 2] = 0.2                                     # understates the third sleeve's risk
    mats = {scoring.CONFIGURED: np.broadcast_to(COV, (t, 3, 3)), "low": np.broadcast_to(low, (t, 3, 3))}
    r, h, starts = scoring.realized_alpha(x, mats, np.ones(t, dtype=bool))
    assert len(starts) == 600 and np.array_equal(starts, np.arange(0, t, 21))
    assert abs(r[scoring.CONFIGURED].mean() - 1.0) < 0.06
    assert scoring.score(r, h, 0)["candidates"]["low"]["b"] > 1.3


def test_the_configured_candidate_is_the_books_estimate_and_each_other_moves_one_choice():
    x = gaussian(COV)
    mats = scoring.candidates(x, 0.97, 1, 0.5)
    assert list(mats) == [scoring.CONFIGURED, "lambda 0.94", "lambda 0.99", "unshrunk", "no lag term", "no floor"]
    np.testing.assert_array_equal(mats[scoring.CONFIGURED], portfolio.floored(risk.ewma_cov(x, 0.97, 1), True, 0.5)[1])
    np.testing.assert_array_equal(mats["no floor"], risk.ewma_cov(x, 0.97, 1).shrunk)
    np.testing.assert_array_equal(mats["no lag term"], portfolio.floored(risk.ewma_cov(x, 0.97, 0), True, 0.5)[1])
    assert list(scoring.candidates(x, 0.94, 0, None)) == [scoring.CONFIGURED, "lambda 0.99", "unshrunk"]


def test_the_benchmark_on_iid_gaussian_pnl_is_the_estimates_noise():
    x = gaussian(COV)
    cfg = scoring.estimate(x, 0.97, 1, True, 0.5)
    rows = scoring.sample(x, {scoring.CONFIGURED: scoring.at_horizon(cfg, 2)}, np.ones(len(x), dtype=bool), 200)
    pool = scoring.sample(x, {scoring.CONFIGURED: cfg}, np.ones(len(x), dtype=bool), 200)
    alone = scoring.sleeves_alone(len(x), 3)
    got = scoring.benchmark(x, cfg, pool, {"sleeves": rows}, lambda m: {"sleeves": alone}, 0.97, 1, 0.5, 2, reps=4)
    for kind in ("gaussian", "bootstrap"):
        b = got[kind]["sleeves"]
        assert b.shape == (3,) and (b > 1.0).all() and (b < 1.06).all(), (kind, b)


def test_nothing_up_to_a_session_reads_the_pnl_after_it():
    x = gaussian(COV)
    cut = 500
    poisoned = x.copy()
    poisoned.iloc[cut:] = 1e3
    a, b = scoring.candidates(x, 0.97, 1, 0.5), scoring.candidates(poisoned, 0.97, 1, 0.5)
    for c in a:
        ha, hb = scoring.at_horizon(a[c], 2), scoring.at_horizon(b[c], 2)
        np.testing.assert_array_equal(ha[:cut + 3], hb[:cut + 3])                 # decided by the close before cut + 1
        assert not np.allclose(ha[cut + 4:], hb[cut + 4:])
