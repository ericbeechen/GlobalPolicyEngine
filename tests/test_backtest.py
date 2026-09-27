"""The crude backtest: a signal-agnostic engine, and the instrument it is fed."""

import numpy as np
import pandas as pd
import pytest
from policypath.backtest.engine import run, stats
from policypath.backtest.instrument import held_rate_change

DAYS = pd.bdate_range("2024-01-01", periods=5)


def test_receiving_makes_money_when_rates_fall():
    position = pd.Series([1.0, 1.0, -2.0, 0.0, 0.0], index=DAYS)     # receive, receive, pay
    change = pd.Series([-0.10, 0.05, 0.02, 0.03], index=DAYS[1:])   # pp over each session
    r = run(position, change * 100, dv01=1.0, lag=0)                  # in bp
    # held into day 1 is day 0's +1: rates fell 10bp -> +10; day 2 +1, rates +5 -> -5; day 3 -2, +2 -> +4
    assert r["pnl"].tolist() == pytest.approx([10.0, -5.0, 4.0, 0.0])
    assert r["equity"].iloc[-1] == pytest.approx(9.0)


def test_the_lag_delays_the_position():
    position = pd.Series([1.0, 0.0, 0.0, 0.0, 0.0], index=DAYS)
    change = pd.Series([-1.0, -1.0, -1.0, -1.0], index=DAYS[1:])
    assert run(position, change, lag=0)["pnl"].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert run(position, change, lag=1)["pnl"].tolist() == [0.0, 1.0, 0.0, 0.0]


def test_any_signal_runs_through_unchanged():
    rng = np.random.default_rng(3)
    days = pd.bdate_range("2020-01-01", periods=250)
    position = pd.Series(rng.normal(size=250), index=days, name="credit_z")
    change = pd.Series(rng.normal(size=249), index=days[1:])
    r = run(position, change, dv01=2.0)
    assert r["pnl"].to_numpy() == pytest.approx(-2.0 * position.shift(1).iloc[1:].to_numpy() * change.to_numpy())
    s = stats(r)
    assert s["sessions"] == 249 and np.isfinite(s["sharpe"])


def panel_frames():
    """Two sessions either side of a meeting: k=1 on the first is gone by the second."""
    s0, s1, s2 = pd.to_datetime(["2024-09-17", "2024-09-18", "2024-09-19"])
    m1, m2, m3 = pd.to_datetime(["2024-09-19", "2024-11-08", "2024-12-19"])
    sessions = pd.DataFrame({"session": [s0, s1, s2], "error": None, "rate_now": [5.33, 5.33, 4.83]})
    meetings = pd.DataFrame({
        "session": [s0, s0, s1, s1, s2, s2],
        "k": [1, 2, 1, 2, 1, 2],
        "effective_date": [m1, m2, m1, m2, m2, m3],
        "rate": [4.95, 4.60, 4.90, 4.55, 4.60, 4.35],
    })
    return sessions, meetings


def test_the_change_follows_the_meeting_held_not_the_horizon():
    sessions, meetings = panel_frames()
    k2 = held_rate_change(sessions, meetings, 2)
    # held into 09-19 is 09-18's second meeting (Nov), priced 4.55 then 4.60, not the Dec meeting's 4.35
    assert k2.tolist() == pytest.approx([-0.05, 0.05])
    k1 = held_rate_change(sessions, meetings, 1)
    # the September meeting took effect on 09-19: its rate is then the rate in force
    assert k1.tolist() == pytest.approx([-0.05, 4.83 - 4.90])
