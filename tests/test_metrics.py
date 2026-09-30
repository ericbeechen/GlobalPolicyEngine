"""The week 10 statistics in `backtest/metrics.py`, on hand-checkable series.

Trades: a trade's costs are the closes it is on after, and the close that takes
it off; on a flip that close's cost goes to the new trade. Drawdown: from a
peak never below the start. Turns: a round trip is 2 turns. The windows:
ex_2022 leaves out exactly the configured window's sessions and nothing else,
the positions not re-run. The level factor recovers known betas.
"""

import numpy as np
import pandas as pd
import pytest
from policypath import config
from policypath.backtest import metrics

DAYS = pd.bdate_range("2021-06-01", periods=12)


def test_a_trade_pays_its_opening_and_closing_costs_and_a_flip_charges_the_new_trade():
    #                 0    1    2    3    4    5    6    7    8    9   10   11
    held = pd.Series([0.0, 0.0, 1.0, 1.0, 1.0, -1.0, -1.0, 0.0, 0.0, 2.0, 2.0, 2.0], index=DAYS)
    gross = pd.Series([0.0, 0.0, 5.0, -1.0, 2.0, 3.0, 4.0, 0.0, 0.0, -2.0, 1.0, 1.0], index=DAYS)
    cost = pd.Series([0.0, 0.5, 0.0, 0.0, 0.7, 0.0, 0.3, 0.0, 0.2, 0.0, 0.0, 0.0], index=DAYS)
    t = metrics.trades(gross, cost, held)
    # receive: put on at close 1 (0.5), held into 2-4 (+6), flipped at close 4 (0.7 goes to the pay)
    # pay: held into 5-6 (+7), taken off at close 6 (0.3); a second receive from close 8 (0.2), still on at the end
    assert list(t.index) == [DAYS[1], DAYS[4], DAYS[8]]
    np.testing.assert_allclose(t.to_numpy(), [6.0 - 0.5, 7.0 - 0.7 - 0.3, 0.0 - 0.2])
    assert t.sum() == pytest.approx(gross.sum() - cost.sum())


def test_hit_rates_drawdown_turns_and_time_in_market():
    net = pd.Series([1.0, -2.0, 0.0, 3.0, -1.0, -1.0], index=DAYS[:6])
    on = pd.Series([True, True, False, True, True, True], index=DAYS[:6])
    assert metrics.hit_rate(net, on) == pytest.approx(2 / 5)
    assert metrics.max_drawdown(net) == pytest.approx(2.0)            # peak 1, trough -1; later peak 2, trough 0
    assert metrics.max_drawdown(pd.Series([-1.0, -1.0, 1.5])) == pytest.approx(2.0)   # the peak is never below 0
    dv01 = pd.Series([2.0, 2.0, 0.0, 0.0, 0.0, 0.0])
    traded = pd.Series([2.0, 0.0, 2.0, 0.0, 0.0, 0.0])                # one round trip of 2
    assert metrics.turnover(traded, dv01, years=1.0) == pytest.approx(2.0)
    assert metrics.time_in_market(dv01) == pytest.approx(1 / 3)


def _result(n=600, seed=3):
    days = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(seed)
    gross = pd.Series(rng.normal(0.1, 1.0, n), index=days)
    daily = pd.DataFrame({"gross": gross, "cost": 0.02, "traded": 0.04, "gross_dv01": 1.0,
                          "kept": days >= days[20]}, index=days)
    return metrics.Result(daily, 252.0)


def test_evaluate_is_week_8s_sharpe_on_the_counted_sessions():
    r = _result()
    d = r.daily[r.daily["kept"]]
    s = metrics.evaluate(r, capital=100.0)
    assert s["sessions"] == len(d)
    assert s["net_sr"] == pytest.approx(metrics.sharpe(d["gross"] - d["cost"], 252.0))
    assert s["net_se"] == pytest.approx(metrics.sharpe_se(s["net_sr"], len(d) / 252.0))
    assert s["net_year"] == pytest.approx((d["gross"] - d["cost"]).sum() / (len(d) / 252.0))   # % of 100
    assert s["turns_year"] == pytest.approx(0.04 * 252.0) and s["time_in_market"] == 1.0


def test_ex_2022_leaves_out_the_configured_window_only():
    r = _result()
    book = config.strategy()
    lo, hi = metrics.window("ex_2022", book)
    assert (lo, hi) == (pd.Timestamp(book["evaluation"]["exclude"]["ex_2022"][0]),
                        pd.Timestamp(book["evaluation"]["exclude"]["ex_2022"][1]))
    d = r.daily[r.daily["kept"]]
    rest = d[(d.index < lo) | (d.index > hi)]
    s = metrics.ex_2022(r, book)
    assert s["sessions"] == len(rest) < len(d)
    assert s["net_sr"] == pytest.approx(metrics.sharpe(rest["gross"] - rest["cost"], 252.0))
    assert metrics.ex_2022_23(r, book)["sessions"] < s["sessions"]         # the cycle is the longer window
    assert metrics.ex_2022(r)["sessions"] == s["sessions"]                   # the configured book by default


def test_the_level_factor_recovers_known_betas():
    rng = np.random.default_rng(5)
    xs = pd.DataFrame(rng.normal(size=(2000, 2)), columns=["a", "b"])
    y = 0.4 * xs["a"] - 0.7 * xs["b"] + 0.01 + rng.normal(0, 0.1, 2000)
    coef, r2, n = metrics.factor(y, xs)
    assert coef["a"] == pytest.approx(0.4, abs=0.01) and coef["b"] == pytest.approx(-0.7, abs=0.01)
    assert r2 > 0.95 and n == 2000


def test_ic_table_is_the_ic_on_the_masked_sessions_at_each_horizon():
    days = pd.bdate_range("2019-01-01", periods=400)
    rng = np.random.default_rng(8)
    x = pd.Series(rng.normal(size=400), index=days)
    z = x.rolling(5).sum().shift(-6)                      # z knows the next 5 sessions after a lag of 1
    mask = pd.Series(days.year == 2019, index=days)
    rows = metrics.ic_table(z, x, [5, 21], 1, mask)
    assert [r["h"] for r in rows] == [5, 21]
    assert rows[0]["ic"] > 0.95 and rows[0]["sessions"] <= int(mask.sum())
    v, t, n = metrics.ic(z.where(mask), metrics.forward(x, 21, 1), 21)
    assert rows[1]["ic"] == pytest.approx(v) and rows[1]["t"] == pytest.approx(t)
