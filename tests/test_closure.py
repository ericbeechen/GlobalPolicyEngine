"""Who closes the gap (`closure.py`) on made-up panels where the answer is known.

A flat world: every meeting on a session's path carries the same market rate and the
same rule rate, so the change for the meeting held is the change in the level. In one
world the market is a random walk and the rule moves a fixed share `kappa` of the gap
toward it each session; in the other the rule is the walk and the market does the
moving. Over sessions t + 1 to t + 1 + h, the mover's expected change is
(1 - kappa)(1 - (1 - kappa)^h) times the gap at t, and the walker's is zero.
"""

import numpy as np
import pandas as pd
import pytest
from policypath import closure
from policypath.report import closure as report

K, H = 4, 21
MEETINGS = 8


def world(n=6000, kappa=0.04, mover="rule", seed=1, elb_until=0, meet_every=25):
    """(paths, signal, model) for a flat world in which `mover` closes the gap."""
    rng = np.random.default_rng(seed)
    sessions = pd.bdate_range("2012-01-02", periods=n)
    walk = 2.0 + np.cumsum(rng.normal(0.0, 0.02, n))
    noise = rng.normal(0.0, 0.02, n)
    other = np.empty(n)
    other[0] = walk[0] + 0.3
    for s in range(n - 1):
        other[s + 1] = other[s] + kappa * (walk[s] - other[s]) + (noise[s] if mover == "market" else 0.0)
    market, rule = (walk, other) if mover == "rule" else (other, walk)
    effective = pd.bdate_range(sessions[0] - pd.offsets.BDay(meet_every), periods=n // meet_every + MEETINGS + 3,
                               freq=pd.offsets.BDay(meet_every))
    rows = []
    for s, day in enumerate(sessions):
        ahead = effective[effective > day][:MEETINGS]
        rows += [{"session": day, "k": j + 1, "announcement_date": e - pd.Timedelta(days=1), "effective_date": e,
                  "market": market[s], "model": rule[s], "model_mid": rule[s]} for j, e in enumerate(ahead)]
    paths = pd.DataFrame(rows)
    gap = (paths["market"] - paths["model"]) * 100.0
    warm = paths["session"] >= sessions[50]
    signal = pd.DataFrame({"session": paths["session"], "k": paths["k"], "gap_bp": gap,
                           "z": (gap / 10.0).where(warm), "window_mean_bp": 0.0, "window_sd_bp": 10.0})
    elb = 0.1
    at_floor = np.arange(n) < elb_until
    model = pd.DataFrame({"session": sessions, "r0": np.where(at_floor, elb, rule), "elb": elb, "at_elb": at_floor,
                          "inflation": 2.0, "u_gap": 0.0, "rstar": 0.5, "notional": rule + 0.5})
    return paths, signal, model


def expected(kappa=0.04, h=H):
    return (1 - kappa) * (1 - (1 - kappa) ** h)


@pytest.fixture(scope="module")
def rule_moves():
    return closure.fit(closure.frame(*world(mover="rule"), K, H), H)


@pytest.fixture(scope="module")
def market_moves():
    return closure.fit(closure.frame(*world(mover="market"), K, H), H)


def test_a_rule_that_closes_the_gap_reads_as_the_rule(rule_moves):
    f = rule_moves
    assert f["rule"]["slope"] == pytest.approx(expected(), abs=0.08)
    assert abs(f["market"]["slope"]) < 0.08
    assert f["who"] == "the rule" and f["rule_share"] > closure.MOSTLY
    assert f["rate"]["slope"] == pytest.approx(f["rule"]["slope"])     # the rate in force is the rule's level here


def test_a_market_that_closes_the_gap_reads_as_the_market(market_moves):
    f = market_moves
    assert f["market"]["slope"] == pytest.approx(-expected(), abs=0.08)
    assert abs(f["rule"]["slope"]) < 0.08
    assert f["who"] == "the market" and f["rule_share"] < 1 - closure.MOSTLY


def test_the_gap_slope_is_the_market_less_the_rule(rule_moves, market_moves):
    for f in (rule_moves, market_moves):
        assert f["gap"]["slope"] == pytest.approx(f["market"]["slope"] - f["rule"]["slope"], abs=1e-9)
        assert f["closed"] == pytest.approx(-f["gap"]["slope"], abs=1e-9)
        assert f["market"]["n"] == f["rule"]["n"] == f["rate"]["n"] == f["sessions"]


def test_the_change_is_for_the_meeting_held_at_the_close():
    # Meeting j's rate on session s is j x (1 + s): a meeting passes between the first two sessions, so the
    # fourth at the first close is the third at the next. The change must be meeting 4's own.
    sessions = pd.bdate_range("2020-01-06", periods=3)
    effective = pd.DatetimeIndex([sessions[0] + pd.Timedelta(hours=12)]).append(
        pd.bdate_range(sessions[2] + pd.offsets.BDay(1), periods=10))
    rows = []
    for s, day in enumerate(sessions):
        for k, e in enumerate(effective[effective > day][:MEETINGS], start=1):
            j = int(np.flatnonzero(effective == e)[0]) + 1
            rows.append({"session": day, "k": k, "effective_date": e, "market": j * (1.0 + s), "model": 0.0})
    paths = pd.DataFrame(rows)
    at, start, end = (np.array([d], dtype="datetime64[ns]") for d in sessions)
    market, rule = closure.same_meeting(paths, 4, at, start, end)
    held = paths[(paths["session"] == sessions[0]) & (paths["k"] == 4)]["effective_date"].iloc[0]
    j = int(np.flatnonzero(effective == held)[0]) + 1
    assert market[0] == pytest.approx(j * 100.0) and rule[0] == 0.0


def test_sessions_in_the_elb_state_are_left_out():
    f = closure.frame(*world(elb_until=500), K, H)
    assert not f["keep"].iloc[:500].any() and f["keep"].iloc[500:].any()
    assert closure.fit(f, H)["sessions"] == f["keep"].sum()


def test_a_meeting_that_has_left_the_path_is_left_out():
    f = closure.frame(*world(n=600, meet_every=3), K, H)     # the fourth meeting is about 12 sessions out
    assert not f["keep"].any()
    fitted = closure.fit(f, H)
    assert fitted["sessions"] == 0 and fitted["who"] == "neither" and fitted["bins"] == []


def test_the_bins_split_the_sessions_evenly(rule_moves):
    b = rule_moves["bins"]
    assert len(b) == closure.BINS
    assert max(x["sessions"] for x in b) - min(x["sessions"] for x in b) <= 1
    assert all(b[i]["edge"] < b[i + 1]["edge"] for i in range(len(b) - 1))


def test_who_closes_it_in_words():
    assert closure.who(0.9, 0.5) == "the rule" and closure.who(closure.MOSTLY, 0.5) == "the rule"
    assert closure.who(0.1, 0.5) == "the market" and closure.who(1 - closure.MOSTLY, 0.5) == "the market"
    assert closure.who(0.5, 0.5) == "both"
    assert closure.who(np.nan, -0.1) == "neither" and closure.who(np.nan, np.nan) == "neither"


def test_the_report_prints_every_horizon_and_its_reading():
    panels = dict(zip(("paths", "signal", "model"), world(mover="rule")))
    labels = {"market": "the futures", "rule": "The rule", "policy_rate": "Policy rate"}
    cfg = {"backtest": {"horizon": K}, "report": {"labels": labels}}
    results = {"AAA": closure.run("AAA", load=lambda ccy, name: panels[name], cfg=cfg, horizons=[5, H])}
    sessions = panels["paths"]["session"]
    text = report.markdown(results, {"AAA": cfg}, {"AAA": (sessions.min(), sessions.max())})
    f = results["AAA"]["horizons"][H]
    assert "## AAA" in text and report.reading(f, labels) in text
    for h in (5, H):
        assert f"| {h} | " in text
    assert report.slope_t(f["rule"]) in text and "mostly by the rule" in text
    out = report.results_json(results, {"AAA": cfg})
    assert set(out["AAA"]["horizons"]) == {"5", str(H)} and out["AAA"]["reading"][str(H)] == report.reading(f, labels)
