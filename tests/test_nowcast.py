"""The macro nowcast on dated vintages: each input reads what was known on the day.

Values come from the committed ALFRED fixtures (tests/data/alfred), so they are
the real-time numbers, not today's revised ones.
"""

import numpy as np
import pandas as pd
import pytest
from policypath import config
from policypath.calendars import US_BDAY
from policypath.macro.activity import activity, history, inputs
from policypath.macro.gap import unemployment_gap
from policypath.macro.inflation import bridge, bridge_backtest, inflation
from policypath.macro.nowcast import build, nowcast
from policypath.macro.vintage import VintagePanel

T = pd.Timestamp
MACRO = config.currency("USD")["macro"]
GAP = MACRO["gap"]
INFLATION = MACRO["inflation"]
# Fixtures keep reference dates from 2018 only, so the moments start there; the
# tests below check which quarter and months are used, not the z's size.
ACTIVITY = {**MACRO["activity"], "moments_start": "2018-04-01"}
SPEC = {**MACRO, "activity": ACTIVITY, "inflation": {**INFLATION, "bridge_window": 24}}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise RuntimeError("the nowcast touched the network")
    monkeypatch.setattr("socket.socket.connect", refuse)


# ---- the unemployment gap ----------------------------------------------------

def test_the_gap_in_january_2021(panel):
    # November 2020's rate, against the natural rate CBO published on 2020-08-03.
    g = unemployment_gap(panel, "2021-01-04", GAP)
    assert g["gap_month"] == T("2020-11-01")
    assert g["u"] == pytest.approx(6.7)
    assert g["u_star"] == pytest.approx(4.373, abs=1e-3)
    assert g["u_gap"] == pytest.approx(2.327, abs=1e-3)


def test_slack_is_positive(panel):
    g = unemployment_gap(panel, "2021-01-04", GAP)
    assert g["u"] > g["u_star"] and g["u_gap"] > 0
    assert g["u_gap"] == g["u"] - g["u_star"]


def test_the_shutdown_gap_refers_to_november_not_the_missing_october(panel):
    # Until the November report, the latest month is September.
    before = unemployment_gap(panel, "2025-12-15", GAP)
    assert before["gap_month"] == T("2025-09-01") and before["u"] == pytest.approx(4.4)
    # October is published as missing; the gap skips it rather than failing or filling it.
    assert pd.isna(panel.series(GAP["unemployment"], "2025-12-16")[T("2025-10-01")])
    g = unemployment_gap(panel, "2025-12-16", GAP)
    assert g["gap_month"] == T("2025-11-01")
    assert g["u"] == pytest.approx(4.6)
    assert g["u_gap"] == pytest.approx(0.288, abs=1e-3)


def test_u_star_is_the_natural_rate_for_the_quarter_of_the_unemployment_month(panel):
    g = unemployment_gap(panel, "2025-12-16", GAP)
    star = panel.series(GAP["natural_rate"], "2025-12-16")
    assert g["u_star"] == star[T("2025-10-01")]   # November 2025 is in 2025Q4


def test_a_new_cbo_vintage_moves_the_gap_on_the_day_it_lands(panel):
    # UNRATE is unchanged across the two days; only u* for 2024Q4 is revised.
    before, after = (unemployment_gap(panel, d, GAP) for d in ["2025-01-16", "2025-01-17"])
    assert before["gap_month"] == after["gap_month"] == T("2024-12-01")
    assert before["u"] == after["u"] == pytest.approx(4.1)
    assert before["u_star"] == pytest.approx(4.409, abs=1e-3)
    assert after["u_star"] == pytest.approx(4.322, abs=1e-3)
    assert before["u_gap"] == pytest.approx(-0.309, abs=1e-3)
    assert after["u_gap"] == pytest.approx(-0.222, abs=1e-3)


def test_a_missing_natural_rate_raises(raw):
    no_nrou = VintagePanel(raw[raw["series"] != GAP["natural_rate"]], MACRO["projections"])
    with pytest.raises(ValueError, match=GAP["natural_rate"]):
        unemployment_gap(no_nrou, "2021-01-04", GAP)


def test_u_star_is_not_carried_forward_from_an_earlier_quarter(raw):
    # A vintage that stops at 2020Q3 cannot supply u* for November 2020.
    short = raw[(raw["series"] != GAP["natural_rate"]) | (raw["date"] < T("2020-10-01"))]
    with pytest.raises(ValueError, match="2020-10-01"):
        unemployment_gap(VintagePanel(short, MACRO["projections"]), "2021-01-04", GAP)


def test_no_unemployment_rate_raises(panel):
    # The fixtures' first UNRATE vintage is 2018-02-02.
    with pytest.raises(ValueError, match=GAP["unemployment"]):
        unemployment_gap(panel, "2018-01-15", GAP)


@pytest.mark.parametrize("day", ["2021-01-04", "2024-02-07", "2025-01-17", "2025-12-16", "2026-09-24"])
def test_the_gap_on_date_d_does_not_depend_on_anything_published_after_d(panel, day):
    assert unemployment_gap(panel, day, GAP) == unemployment_gap(panel.truncate(day), day, GAP)


# ---- inflation and the CPI -> PCE bridge --------------------------------------

def test_june_2022_cpi_day_bridges_one_month(panel):
    before, on = (inflation(panel, d, INFLATION) for d in ["2022-07-12", "2022-07-13"])
    assert (before["n_bridged"], before["infl_month"]) == (0, T("2022-05-01"))
    assert (on["n_bridged"], on["infl_month"], on["target_month"]) == (1, T("2022-06-01"), T("2022-05-01"))
    assert before["infl_12m"] == pytest.approx(4.685, abs=1e-3)
    assert on["infl_12m"] == pytest.approx(4.721, abs=1e-3)   # June's first print was 4.8


def test_january_20th_2024_bridges_december_from_cpi(panel):
    i = inflation(panel, "2024-01-20", INFLATION)
    assert (i["n_bridged"], i["infl_month"], i["target_month"]) == (1, T("2023-12-01"), T("2023-11-01"))
    assert i["infl_12m"] == pytest.approx(3.03, abs=0.005)
    assert i["infl_6m_ann"] == pytest.approx(2.05, abs=0.005)
    assert (i["bridge_intercept"], i["bridge_slope"]) == (pytest.approx(0.05, abs=0.005), pytest.approx(0.70, abs=0.005))


def test_the_shutdown_bridges_two_months_across_the_cpi_hole(panel):
    i = inflation(panel, "2025-12-19", INFLATION)
    assert (i["n_bridged"], i["infl_month"], i["target_month"]) == (2, T("2025-11-01"), T("2025-09-01"))
    assert np.isfinite([i["infl_12m"], i["infl_6m_ann"], i["infl_3m_ann"]]).all()
    assert i["infl_12m"] == pytest.approx(2.73, abs=0.005)


def synthetic(months=48, seed=0):
    """A target index and a proxy whose m/m is exactly twice the target's."""
    idx = pd.date_range("2020-01-01", periods=months, freq="MS")
    mm = np.random.default_rng(seed).normal(0.2, 0.1, months)
    target = pd.Series(100 * np.exp(np.cumsum(mm) / 100), idx)
    proxy = pd.Series(100 * np.exp(np.cumsum(2 * mm) / 100), idx)
    return target, proxy


def test_the_bridge_recovers_an_exact_linear_proxy():
    target, proxy = synthetic()
    out, filled, (a, b) = bridge(target.iloc[:-3], proxy, window=24)
    assert filled == list(target.index[-3:])
    assert (a, b) == (pytest.approx(0, abs=1e-10), pytest.approx(0.5, abs=1e-10))
    np.testing.assert_allclose(out.to_numpy(), target.to_numpy(), rtol=1e-10)


def test_a_proxy_hole_is_interpolated_not_forward_filled():
    target, proxy = synthetic()
    proxy.iloc[-2] = np.nan                     # October missing, November present
    out, filled, _ = bridge(target.iloc[:-2], proxy, window=24)
    assert filled == list(target.index[-2:])
    sep, oct_, nov = out.iloc[-3:]
    assert oct_ != sep                          # not carried forward
    assert np.log(oct_ / sep) == pytest.approx(np.log(nov / oct_), rel=1e-10)   # the change split evenly
    assert nov == pytest.approx(target.iloc[-1], rel=1e-10)                     # and none of it lost


def test_the_bridge_beats_both_naive_alternatives_in_real_time(panel):
    bt = bridge_backtest(panel, "2026-09-24", INFLATION, start="2023-01-01")
    assert (bt["released"] > bt["month"]).all()
    rmse = {c: np.sqrt(((bt[c] - bt["first_print"]) ** 2).mean()) for c in ["bridge", "proxy_1to1", "last_carried"]}
    assert rmse["bridge"] < rmse["proxy_1to1"] < rmse["last_carried"]
    assert bt.loc[bt["across_hole"], "month"].tolist() == [T("2025-10-01"), T("2025-11-01")]


# ---- activity over the ragged edge ------------------------------------------

def test_january_20th_2024_scores_2023q4_with_two_months_of_orders(panel):
    a = activity(panel, "2024-01-20", ACTIVITY)
    assert a["activity_quarter"] == T("2023-10-01")
    assert {s: a[f"{s}_months"] for s in ACTIVITY["series"]} == {"PAYEMS": 3, "INDPRO": 3, "RSAFS": 3, "NEWORDER": 2}
    assert a["activity_n"] == 4 and np.isfinite(a["activity_z"])


def test_early_in_a_quarter_it_falls_back_to_the_previous_one(panel):
    # January payrolls are out, IP and retail sales are still at December.
    assert panel.series("PAYEMS", "2024-02-05").dropna().index[-1] == T("2024-01-01")
    a = activity(panel, "2024-02-05", ACTIVITY)
    assert (a["activity_quarter"], a["activity_n"]) == (T("2023-10-01"), 4)


def test_the_shutdown_falls_back_to_the_third_quarter(panel):
    # Payrolls at November, IP at September, real retail sales blocked by the CPI hole.
    a = activity(panel, "2025-12-16", ACTIVITY)
    assert (a["activity_quarter"], a["activity_n"]) == (T("2025-07-01"), 4)
    assert np.isfinite(a["activity_z"])


def test_the_moments_ignore_the_target_quarter_and_everything_after(panel):
    data = inputs(panel, "2024-01-20", ACTIVITY)
    quarter = T("2023-10-01")
    for name, x in data.items():
        poisoned = x.mask(x.index >= quarter, x * 10)
        pd.testing.assert_series_equal(history(poisoned, quarter, ACTIVITY["moments_start"]),
                                       history(x, quarter, ACTIVITY["moments_start"]), check_names=False)


# ---- the nowcast end to end: no look-ahead --------------------------------------

# Release days, and the day before each: a leak of even one day shows up only there.
# June 2022 CPI, SA factors, the payroll benchmark, the shutdown CPI, the late PCE.
DAYS = ["2021-01-04", "2022-07-12", "2022-07-13", "2024-01-20", "2024-02-08", "2025-02-06", "2025-02-07",
        "2025-12-16", "2025-12-17", "2025-12-19", "2026-01-21", "2026-01-22"]


def same(a, b):
    assert a.keys() == b.keys()
    for k in a:
        assert a[k] == b[k] or (pd.isna(a[k]) and pd.isna(b[k])), k


@pytest.mark.parametrize("day", DAYS)
def test_the_nowcast_on_date_d_does_not_depend_on_anything_published_after_d(panel, day):
    same(nowcast(day, panel=panel, spec=SPEC), nowcast(day, panel=panel.truncate(day), spec=SPEC))


@pytest.mark.parametrize("day", DAYS)
def test_poisoned_later_vintages_do_not_reach_the_nowcast(raw, panel, day):
    later = raw["realtime_start"] > T(day)
    dirty = VintagePanel(raw.assign(value=raw["value"].mask(later, 99.0)), MACRO["projections"])
    same(nowcast(day, panel=dirty, spec=SPEC), nowcast(day, panel=panel, spec=SPEC))


def test_the_nowcast_carries_the_newest_vintage_behind_it(panel):
    n = nowcast("2024-01-20", panel=panel, spec=SPEC)
    assert n["published"] == T("2024-01-17")
    assert n["as_of"] == T("2024-01-20")


def test_build_gives_one_row_per_fed_business_day(panel):
    frame = build("USD", "2024-01-10", "2024-01-26", panel, SPEC)
    days = pd.date_range("2024-01-10", "2024-01-26", freq=US_BDAY)
    assert T("2024-01-15") not in days            # Martin Luther King Jr. Day
    assert frame["as_of"].tolist() == list(days)
