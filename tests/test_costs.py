"""What a close costs: each leg's one-way cost from config, the charges by hand, the modes, and the breakeven.

Checks, hand-computed on synthetic positions: a futures leg's cost is its
ticks and fee over the derived DV01; a ZQ month roll charges the whole
position out and in, and a roll that is also an entry or a reversal is
charged as one; a forward window change with a resize splits into the resize
and the roll; a par leg is re-struck at the first session on or after each
quarterly anniversary of its entry, never inside the first quarter, and a
re-entry restarts the clock; the report's curve is the Sharpe of gross less
c/2 of what is traded (every leg, or only the assumed ones); its breakevens are
gross over traded on the kept sessions and the zeros of those curves; its net
is gross less cost there. And no cost or tick number is typed in code under
src/.
"""

from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from policypath import config
from policypath.backtest import metrics
from policypath.report import costs as report
from policypath.strategy import costs, instruments

from test_expression import code_tokens

SRC = Path(__file__).resolve().parents[1] / "src" / "policypath"
T = pd.Timestamp


def block(ticks=2, other=0.005, fee=1.0):
    """A currency block with one futures expression and one par expression, costs in ticks and in bp."""
    contract = {"notional": 5_000_000, "accrual": {"days": 30, "year_days": 360}, "tick": {"front": 0.0025,
                                                                                          "other": other},
                "fee_per_side": fee}
    return {"contracts": {"FF": contract},
            "expression": {"outright": {"instrument": "futures_month", "contract": "FF"},
                           "curve": {"instrument": "par_yield"}},
            "costs": {"outright": {"ticks_round_trip": ticks, "observed": True},
                      "curve": {"round_trip_bp": 0.8, "observed": False, "roll_every_months": 3}}}


def test_a_futures_legs_cost_is_its_ticks_and_fee_over_the_derived_dv01():
    dv01 = 5_000_000 * 30 / 360 * 1e-4                      # 41.67 a contract
    assert costs.one_way(block(), "outright") == pytest.approx(2 * 0.5 / 2 + 1.0 / dv01)     # 0.5 + 0.024bp
    assert costs.one_way(block(ticks=4, fee=0.0), "outright") == pytest.approx(1.0)
    assert costs.one_way(block(), "curve") == pytest.approx(0.4)


def test_the_configured_futures_cost_is_derived_from_the_contract():
    for ccy in config.enabled():
        cfg = config.currency(ccy)
        for e, c in cfg["costs"].items():
            if e == "tags":
                continue
            if "ticks_round_trip" in c:
                ct = cfg["contracts"][cfg["expression"][e]["contract"]]
                want = (c["ticks_round_trip"] * ct["tick"]["other"] / 0.01 / 2
                        + ct["fee_per_side"] / instruments.contract_dv01(ct))
            else:
                want = c["round_trip_bp"] / 2
            assert costs.one_way(cfg, e) == pytest.approx(want)


def run_of(days, q, ids, s=None, label="L", lag=0, months=None, one_way=0.5, observed=False):
    """costs.traded for one leg with the position `q` put on at each close, in instrument `ids`."""
    days = pd.DatetimeIndex(days)
    ids = pd.Series(ids, index=days, dtype=object)
    legs = pd.DataFrame({"session": days, "leg": label, "q": q, "selected": ids.to_numpy(),
                         "held_id": ids.shift(1).to_numpy()})
    s = pd.Series(np.sign(q) if s is None else s, index=days, dtype=float)
    lc = [costs.LegCost(label, "X", "e", one_way, observed, months)]
    return costs.traded(SimpleNamespace(sessions=days, lag=lag), legs, s, lc), lc


def test_a_zq_month_roll_charges_the_position_out_and_in():
    days = pd.bdate_range("2024-03-11", periods=6)
    q = [0.0, 2.0, 2.0, 2.0, 2.0, 0.0]
    ids = ["2024-06", "2024-06", "2024-07", "2024-07", "2024-07", "2024-07"]
    trades, lc = run_of(days, q, ids, one_way=0.524)
    by = trades.T.groupby(level="cause").sum().T
    assert by["signal"].tolist() == [0, 2, 0, 0, 0, 2]
    assert by["roll"].tolist() == [0, 0, 4, 0, 0, 0]             # |q_p| + |q_t| on the month change
    assert costs.charge(trades, costs.configured(lc)).tolist() == pytest.approx([0, 1.048, 2.096, 0, 0, 1.048])


def test_a_roll_that_is_an_entry_or_a_reversal_is_charged_once():
    days = pd.bdate_range("2024-03-11", periods=4)
    trades, _ = run_of(days, [0.0, 1.0, -1.0, -1.0], ["A", "B", "C", "C"])
    by = trades.T.groupby(level="cause").sum().T
    assert by["signal"].tolist() == [0, 1, 2, 0] and by["roll"].tolist() == [0, 0, 0, 0]


def test_a_forward_window_change_with_a_resize_splits_into_the_resize_and_the_roll():
    days = pd.bdate_range("2024-03-11", periods=3)
    windows = ["2024-05-09/2024-06-20", "2024-05-09/2024-06-20", "2024-06-20/2024-08-01"]
    trades, _ = run_of(days, [1.0, 1.0, 1.5], windows, s=[1, 1, 1])
    by = trades.T.groupby(level="cause").sum().T
    assert by["rescale"].tolist() == [0, 0, 0.5]                # s did not change: a resize
    assert by["roll"].tolist() == [0, 0, 2.0]                   # 2 min(1, 1.5): the rest of 1 + 1.5
    assert by.sum(axis=1).tolist() == [1.0, 0, 2.5]


def test_a_par_leg_is_re_struck_every_quarter_from_its_entry_while_held():
    days = pd.bdate_range("2024-01-02", "2025-03-31")
    q = pd.Series(0.0, index=days)
    q["2024-01-31":"2024-11-15"] = 2.0          # entry 31 Jan: anniversaries 30 Apr, 31 Jul, 31 Oct (DateOffset)
    q["2024-12-02":] = -1.0                     # a new entry 2 Dec restarts the clock: 2 Mar 2025 is a Sunday
    trades, lc = run_of(days, q.to_numpy(), ["10y"] * len(days), months=3, one_way=0.25)
    struck = trades["restrike"].sum(axis=1)
    assert list(struck[struck > 0].index) == [T("2024-04-30"), T("2024-07-31"), T("2024-10-31"), T("2025-03-03")]
    assert struck[struck > 0].tolist() == [4.0, 4.0, 4.0, 2.0]                     # 2|q|
    cost = costs.charge(trades, costs.configured(lc))
    assert cost["2024-04-30"] == pytest.approx(1.0) and cost["2024-02-01":"2024-04-29"].sum() == 0
    none, _ = run_of(days, q.to_numpy(), ["10y"] * len(days), months=None)
    assert none["restrike"].to_numpy().sum() == 0


def test_the_configured_rates_are_each_legs_own_one_way():
    lc = [costs.LegCost("F", "X", "outright", 0.524, True, None), costs.LegCost("P", "X", "curve", 0.25, False, 3)]
    assert costs.configured(lc) == {"F": 0.524, "P": 0.25}
    assert lc[0].round_trip_bp == pytest.approx(1.048)


def synthetic_run(seed=7, n=1500, drift=0.04):
    """A Run from made-up daily P&L and DV01 traded: two legs, one observed."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2015-01-01", periods=n)
    noise = rng.normal(0.0, 1.0, n)
    gross = drift + noise - noise.mean()                  # the mean is exactly the drift
    traded = rng.poisson(0.3, n) * 1.0
    observed = traded * rng.uniform(0, 1, n)
    daily = pd.DataFrame({"gross": gross, "traded": traded, "assumed": traded - observed,
                          "observed_cost": observed * 0.52, "kept": True}, index=days)
    daily["cost"] = daily["observed_cost"] + daily["assumed"] * 0.5
    daily = daily.assign(signal=traded, rescale=0.0, roll=0.0, restrike=0.0, gross_dv01=1.0)
    daily.iloc[:40, daily.columns.get_loc("kept")] = False          # left out, with their P&L
    daily.iloc[:40, daily.columns.get_loc("gross")] += 5.0          # so leaving them out matters
    return report.Run("x", None, daily, days[0], metrics.per_year(days))


def test_the_curve_is_the_sharpe_of_gross_less_c_over_2_traded():
    run = synthetic_run()
    d = run.daily[run.daily["kept"]]
    cs = [0.0, 0.3, 1.0, 4.0]
    for c, got in zip(cs, report.curve(run, cs)):
        assert got == pytest.approx(metrics.sharpe(d["gross"] - c / 2 * d["traded"], run.periods), rel=1e-12)
    for c, got in zip(cs, report.curve(run, cs, "assumed")):
        net = d["gross"] - d["observed_cost"] - c / 2 * d["assumed"]
        assert got == pytest.approx(metrics.sharpe(net, run.periods), rel=1e-12)


def test_the_closed_form_breakeven_is_the_zero_of_the_numeric_curve():
    run = synthetic_run()
    d = run.daily[run.daily["kept"]]
    star = costs.breakeven(d["gross"].sum(), d["traded"].sum())
    assert star > 0
    assert report.curve(run, [star])[0] == pytest.approx(0.0, abs=1e-12)
    grid = np.linspace(0, 2 * star, 2001)
    y = report.curve(run, grid)
    crossing = grid[np.flatnonzero(np.diff(np.sign(y)))[0]]
    assert abs(crossing - star) <= grid[1] - grid[0]
    loser = synthetic_run(drift=-0.02)
    dl = loser.daily[loser.daily["kept"]]
    assert costs.breakeven(dl["gross"].sum(), dl["traded"].sum()) is None
    assert (report.curve(loser, grid) < 0).all()


def test_the_reports_breakevens_are_gross_over_traded_and_the_zeros_of_its_curves():
    run = synthetic_run(drift=0.2)                          # gross clears the observed leg's cost
    d = run.daily[run.daily["kept"]]
    st = report.stats(run)
    assert st["cstar_bp"] == pytest.approx(2 * d["gross"].sum() / d["traded"].sum(), rel=1e-12)
    assert report.curve(run, [st["cstar_bp"]])[0] == pytest.approx(0.0, abs=1e-12)
    want = 2 * (d["gross"].sum() - d["observed_cost"].sum()) / d["assumed"].sum()
    assert st["cstar_assumed_bp"] == pytest.approx(want, rel=1e-12)
    assert report.curve(run, [st["cstar_assumed_bp"]], "assumed")[0] == pytest.approx(0.0, abs=1e-12)


def test_the_reports_net_is_gross_less_cost_over_the_kept_sessions():
    run = synthetic_run()
    d = run.daily[run.daily["kept"]]
    st = report.stats(run)
    assert st["net_sr"] == pytest.approx(metrics.sharpe(d["gross"] - d["cost"], run.periods), rel=1e-12)
    assert st["gross_sr"] == pytest.approx(metrics.sharpe(d["gross"], run.periods), rel=1e-12)
    assert st["net_year"] == pytest.approx(st["gross_year"] - st["cost_year"]) and st["cost_year"] > 0


COST_NUMBERS = {"0.01", "0.0", "0", "1", "2.0"}   # a bp in price points, the halves of a round trip, loop bookkeeping


def cost_tokens():
    """Every tick size, fee, tick value, one-way and round-trip cost the config implies, as code could type them."""
    out = set()
    for ccy in config.enabled():
        cfg = config.currency(ccy)
        for name, ct in cfg.get("contracts", {}).items():
            dv01 = instruments.contract_dv01(ct)
            out |= {repr(t) for t in ct["tick"].values()}
            out |= {f"{v:.{d}f}" for v in instruments.tick_values(ct).values() for d in (2, 3, 4)}
            out |= {f"{ct['fee_per_side'] / dv01:.{d}f}" for d in (3, 4)}
        for e, c in cfg["costs"].items():
            if e != "tags" and "ticks_round_trip" in c:
                one = costs.one_way(cfg, e)
                out |= {f"{x:.{d}f}" for x in (one, 2 * one) for d in (3, 4)}
    return out


def test_no_cost_or_tick_number_is_typed_in_code():
    numbers = [t for t in code_tokens(SRC / "strategy" / "costs.py") if t[0].isdigit()]
    assert set(numbers) <= COST_NUMBERS, set(numbers) - COST_NUMBERS
    typed = cost_tokens()
    assert {"0.005", "0.524", "1.048", "0.024", "20.83"} <= typed       # the probe: ZQ's tick, costs, fee, tick value
    hits = {str(p.relative_to(SRC)): t for p in SRC.rglob("*.py") for t in code_tokens(p)
            if any(x in t for x in typed)}
    assert not hits
