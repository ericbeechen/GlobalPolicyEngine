"""The instruments: marks as they stood, contract months, forwards off the path's own helper, par legs.

All synthetic. Each leg type is checked against the thing it must agree with:
`market._as_published` for the marks, `curves.forward.forward_path` for the
forward (bit for bit at k >= 2), the par-bond identities for the par legs.
"""

import numpy as np
import pandas as pd
import pytest
from policypath import config, market
from policypath.calendars import BDAYS
from policypath.curves.forward import forward_path
from policypath.strategy import instruments

T = pd.Timestamp
BDAY = BDAYS["uk"]


def vintage_log():
    """Two dates, two tenors, each with a same-day preliminary, a next-day final and a revision a week later."""
    rows = []
    for d in [T("2024-05-24"), T("2024-05-28")]:                   # Friday; Tuesday after the bank holiday
        for tenor in (1, 2):
            final = d + BDAY + pd.Timedelta(hours=12)
            for published, value in [(d + pd.Timedelta(hours=17), 1.0), (final, 2.0), (d + pd.Timedelta(days=9), 3.0)]:
                rows.append({"date": d, "tenor": tenor, "value": value + tenor, "published": published,
                             "retrieved": T("2026-09-28")})
    return pd.DataFrame(rows)


def test_prints_keeps_what_as_published_keeps_for_every_date():
    rows = vintage_log()
    got = instruments.prints(rows, ["tenor"], 1, BDAY)
    for d, day in got.groupby("date"):
        want = market._as_published(rows[rows["date"] == d], "tenor", d, BDAY, 1)
        assert day.sort_values("tenor")["value"].tolist() == want.sort_values("tenor")["value"].tolist() == [3.0, 4.0]


def test_a_session_without_a_print_carries_the_last_row_whole():
    frame = pd.DataFrame({1: [1.0, 1.1, 1.2], 6: [2.0, np.nan, 2.2]},
                         index=pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-05"]))
    days = pd.DatetimeIndex(["2024-01-03", "2024-01-04", "2024-01-05"])
    rows, stale = instruments.asof(frame, days)
    assert stale.tolist() == [False, True, False]
    assert rows.loc["2024-01-04", 1] == 1.1 and np.isnan(rows.loc["2024-01-04", 6])   # not filled node by node
    before, _ = instruments.asof(frame, pd.DatetimeIndex(["2024-01-01"]))
    assert before.isna().all(axis=None)


def test_the_lag_is_the_series_own_else_the_markets():
    usd, gbp = config.currency("USD"), config.currency("GBP")
    m = instruments.Marks("USD", usd)
    assert m.lag("fred", "DGS2") == usd["sources"]["daily"]["fred"]["DGS2"]
    assert m.lag("databento", "ZQ") == usd["market"]["final_after_bdays"]
    assert instruments.Marks("GBP", gbp).lag("fred", "DEXUSUK") == 0


# ---- futures month ------------------------------------------------------------------

def test_the_contract_is_the_first_month_starting_on_or_after_the_meeting():
    effective = pd.Series(pd.to_datetime(["2026-10-29", "2026-11-01", "2026-12-10"]))
    assert instruments.contract_month(effective).astype(str).tolist() == ["2026-11", "2026-11", "2027-01"]


def test_dv01_and_ticks_come_from_the_contract_spec():
    spec = {"notional": 5_000_000, "accrual": {"days": 30, "year_days": 360}, "tick": {"front": 0.0025, "other": 0.005}}
    assert instruments.contract_dv01(spec) == pytest.approx(5_000_000 * 30 / 360 * 1e-4)
    ticks = instruments.tick_values(spec)
    assert ticks["front"] == pytest.approx(instruments.contract_dv01(spec) / 4)
    assert ticks["other"] == pytest.approx(instruments.contract_dv01(spec) / 2)


def zq(days, strips, effective, k=1, h=91):
    prices = pd.DataFrame(strips, index=days)
    prices.columns = pd.PeriodIndex(prices.columns, freq="M")
    return instruments.futures_month(days, prices, effective, k, h, 1.0)


def test_a_futures_month_rolls_linearly_between_month_midpoints_and_flat_beyond():
    days = pd.DatetimeIndex(["2026-06-01", "2026-06-04"])
    strip = {"2026-07": 96.0, "2026-08": 95.7}                        # 4.00%, 4.30%
    effective = pd.DataFrame({1: T("2026-07-29"), 2: T("2026-09-17")}, index=days)
    frame, regimes = zq(days, [strip, strip], effective)
    assert frame["id"].tolist() == ["2026-08", "2026-08"]
    # mid-July to mid-August is 31 days (15.5 + 15.5), so three days roll 0.30 x 3/31 off the August rate
    assert frame["rolled"].iloc[1] == pytest.approx(4.30 - 0.30 * 3 / 31)
    assert frame["held_t"].iloc[1] == pytest.approx(4.30)
    # 91 days before mid-August is before mid-July, the strip's first month: flat at its rate
    assert frame["ahead"].iloc[0] == pytest.approx(4.00)
    assert regimes.loc[days[0], 1] == 1.0 and regimes.loc[days[0], 2] == 0.0


def test_a_month_that_straddles_the_next_meeting_weights_its_days():
    days = pd.DatetimeIndex(["2026-09-01"])
    effective = pd.DataFrame({1: T("2026-11-10"), 2: T("2026-12-10"), 3: T("2027-01-28")}, index=days)
    _, regimes = zq(days, [{"2026-12": 96.0}], effective)
    assert regimes.iloc[0].tolist() == pytest.approx([9 / 31, 22 / 31, 0.0])     # Dec 1-9 in k, 10-31 in k + 1


# ---- the forward off a spot curve -----------------------------------------------------

def spot_curve(level=4.0, seed=3):
    rng = np.random.default_rng(seed)
    months = np.arange(1, 25)
    return pd.Series(level + np.cumsum(rng.normal(0, 0.05, 24)), index=months)


MEETINGS = pd.to_datetime(["2025-02-06", "2025-03-20", "2025-05-08", "2025-06-19", "2025-08-07", "2025-09-18",
                           "2025-11-06", "2025-12-18", "2026-02-05"])


def test_the_forward_is_the_paths_rate_bit_for_bit_from_the_second_meeting():
    day = T("2025-01-21")
    spot = spot_curve()
    path = forward_path(spot, day, MEETINGS[:8], MEETINGS[8], 365, last_fixing=4.7, pin_always=True)
    days = pd.DatetimeIndex([day])
    for k in range(2, 9):
        frame = instruments.curve_forward(days, spot.to_frame().T.set_axis(days), pd.Series(MEETINGS[k - 1], index=days),
                                          pd.Series(MEETINGS[k], index=days), 91, 365, pd.Series(4.7, index=days))
        assert frame["rate"].iloc[0] == path[MEETINGS[k - 1]]


def test_a_rolled_window_below_the_first_node_reads_the_rate_in_force():
    day = T("2025-01-21")
    spot = pd.Series([4.0, 4.2, 4.4], index=[1, 6, 12])
    start, end = day + pd.Timedelta(days=100), day + pd.Timedelta(days=150)
    days = pd.DatetimeIndex([day])
    frame = instruments.curve_forward(days, spot.to_frame().T.set_axis(days), pd.Series(start, index=days),
                                      pd.Series(end, index=days), 91, 365, pd.Series(3.0, index=days))
    ta, tb = 9 / 365, 59 / 365                                        # 91 days on, the window starts inside a month
    lb = np.interp(tb, [0, 1 / 12, 0.5, 1], [0, 0.04 / 12, 0.021, 0.044])
    assert frame["ahead"].iloc[0] == pytest.approx((lb - 0.03 * ta) / (tb - ta) * 100)
    df = np.exp(-np.interp(150 / 365, [0, 1 / 12, 0.5, 1], [0, 0.04 / 12, 0.021, 0.044]))
    assert frame["dv01"].iloc[0] == pytest.approx(50 / 365 * df * 1e-4)


# ---- par legs -----------------------------------------------------------------------

def test_par_duration_and_the_bond_price_agree_at_par():
    assert instruments.par_duration(4.0, 2.0, 2) == pytest.approx((1 - 1.02 ** -4) / 0.04)
    assert instruments.par_duration(0.0, 2.0, 2) == 2.0
    for tau in (1.99, 2.0, 9.3):
        assert instruments.bond_price(4.0, 4.0, tau, 2) == pytest.approx(1.0, abs=1e-15)
        slope = (instruments.bond_price(4.0, 4.001, tau, 2) - instruments.bond_price(4.0, 3.999, tau, 2)) / 0.002
        assert slope * 100 == pytest.approx(-instruments.par_duration(4.0, tau, 2), rel=1e-6)


def test_par_yields_from_a_flat_curve_and_with_the_half_year_node_missing():
    months = np.arange(6, 121, 6)
    flat = pd.DataFrame([np.full(len(months), 4.0)], columns=months)
    par = instruments.par_curve(flat, 2, 10)
    assert par.iloc[0].to_numpy() == pytest.approx(2 * (np.exp(0.02) - 1) * 100)   # semiannual par of 4% continuous
    rng = np.random.default_rng(5)
    curve = pd.DataFrame([4.0 + np.cumsum(rng.normal(0, 0.03, len(months)))], columns=months)
    full = instruments.par_curve(curve, 2, 10).iloc[0]
    missing = curve.copy()
    missing[6] = np.nan
    gap = instruments.par_curve(missing, 2, 10).iloc[0]
    assert gap.notna().all()
    # without the 0.5y node, s t runs straight from 0 to the 1y point: DF(0.5) = exp(-s(1) x 0.5)
    df = np.exp(-curve.iloc[0].to_numpy() / 100 * months / 12)
    df_gap = np.r_[np.exp(-curve.iloc[0][12] / 100 * 0.5), df[1:]]
    assert gap[2.0] == pytest.approx(2 * (1 - df_gap[3]) / df_gap[:4].sum() * 100)
    assert abs(gap[2.0] - full[2.0]) < 0.05


def test_a_par_leg_is_struck_at_p_and_marked_at_its_shorter_maturity():
    days = pd.DatetimeIndex(["2025-01-06", "2025-01-09"])
    curves = pd.DataFrame([{1: 3.0, 2: 4.0, 10: 5.0}, {1: 3.1, 2: 4.2, 10: 5.0}], index=days)
    frame = instruments.par_leg(days, curves, 2, 2, 91, pd.Series([2.0, 2.1], index=days), 360)
    last = frame.iloc[1]
    assert last["tau"] == pytest.approx(2 - 3 / 365) and last["coupon"] == 4.0 and last["held_p"] == 4.0
    assert last["rolled"] == pytest.approx(4.0 - 1.0 * 3 / 365)                  # p's curve at the shorter maturity
    assert last["held_t"] == pytest.approx(4.2 - 1.1 * 3 / 365)                  # t's curve there
    assert last["funding_p"] == 2.0 and last["duration_p"] == pytest.approx(instruments.par_duration(4.0, 2, 2))
    assert frame["dv01"].iloc[0] == pytest.approx(instruments.par_duration(4.0, 2, 2) * 1e-4)


def test_fx_converts_at_spot_on_the_day_over_spot_at_p():
    frame = pd.DataFrame({"x": [0.0, 0.0, 0.0]}, index=pd.bdate_range("2025-01-06", periods=3))
    out = instruments.with_fx(frame, [1.25, 1.30, 1.20])
    assert out["fx"].iloc[1:].tolist() == pytest.approx([1.30 / 1.25, 1.20 / 1.30])


def test_a_spot_outside_the_plausible_range_is_refused():
    fx = config.currency("GBP")["expression"]["fx"]
    days = pd.bdate_range("2025-01-06", periods=3)
    usd_per_gbp = np.array([1.25, 1.30, 1.27])
    instruments.plausible_spot(usd_per_gbp, days, fx)
    with pytest.raises(ValueError, match="upside down"):
        instruments.plausible_spot(1.0 / usd_per_gbp, days, fx)       # GBP per USD read as USD per GBP
