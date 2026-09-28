"""GBP: the London calendar, the forward-curve path, the rate in force, the MPR check, the ONS vintages.

The forward-curve extractor is simpler algebra than ZQ's: the rate between two
meetings is the forward between two points of a spot curve. The checks here
are that it recovers a path exactly where the curve pins it down, that the
parts where it cannot (below the first maturity, between maturities) are
handled as documented, and that on the committed MPR windows it lands on the
Bank of England's own conditioning path.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from policypath import config
from policypath.calendars import UK_BDAY
from policypath.curves.forward import forward_path
from policypath.macro.nowcast import nowcast
from policypath.market import rate_in_force, solve_curve
from policypath.model.path import operating_spread, target_range
from policypath.report import conditioning
from policypath.sources import ons

T = pd.Timestamp
DATA = Path(__file__).parent / "data"
GBP = config.currency("GBP")
PATH = GBP["path"]
YEAR_DAYS = GBP["market"]["curve"]["year_days"]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise RuntimeError("the GBP code touched the network")
    monkeypatch.setattr("socket.socket.connect", refuse)


@pytest.fixture(scope="module")
def mpc():
    return config.meetings("GBP")


def spot_on(curve, day):
    rows = curve[curve["date"] == T(day)]
    return rows.set_index("tenor")["value"].sort_index()


def solve(day, curve, mpc, sonia, bank_rate):
    return solve_curve(day, spot_on(curve, day), mpc, sonia, PATH["n_meetings"], UK_BDAY, YEAR_DAYS,
                       PATH["tail_days"], PATH["min_regime_days"], policy=bank_rate,
                       pin_always=GBP["market"]["pin_first_regime"])


# ---- the calendar -----------------------------------------------------------------

def test_london_business_days_are_exactly_the_days_sonia_is_fixed(sonia):
    days = pd.date_range(sonia["date"].min(), sonia["date"].max(), freq=UK_BDAY)
    assert days.equals(pd.DatetimeIndex(sonia["date"]))
    for one_off in ["2011-04-29", "2012-06-05", "2020-05-08", "2022-06-03", "2022-09-19", "2023-05-08"]:
        assert T(one_off) not in days


def test_the_mpc_calendar_has_both_schedules_and_the_march_2020_cuts(mpc):
    per_year = mpc.groupby(mpc["announcement_date"].dt.year).size()
    assert (per_year.loc[2009:2015] == 12).all() and (per_year.loc[2017:2019] == 8).all()
    assert per_year[2020] == 10
    unscheduled = mpc.loc[~mpc["scheduled"], "announcement_date"].dt.strftime("%Y-%m-%d").tolist()
    assert unscheduled == ["2020-03-11", "2020-03-19"]
    assert (mpc["effective_date"] == mpc["announcement_date"]).all(), "Bank Rate applies from the announcement"


def test_bank_rate_changes_only_on_decision_days(bank_rate, mpc):
    br = bank_rate.set_index("date")["value"]
    changes = br[br.diff().fillna(0) != 0].index
    assert len(changes) >= 27 and set(changes) <= set(mpc["announcement_date"])


# ---- the forward-curve path -----------------------------------------------------------

def spot_from(steps, as_of, months=48):
    """Spot rates at monthly maturities (Act/365 years) for a piecewise-flat forward curve."""
    t = np.arange(1, months + 1) / 12.0
    starts = [(T(d) - T(as_of)).days / 365.0 for d, _ in steps]
    integral = np.zeros_like(t)
    for (s, (_, r)), e in zip(zip(starts, steps), [*starts[1:], np.inf]):
        integral += r * np.clip(np.minimum(t, e) - s, 0, None)
    return pd.Series(integral / t, index=np.arange(1, months + 1))


def test_a_path_with_steps_on_the_curves_maturities_comes_back_exactly():
    as_of = T("2023-01-02")
    e1, e2, end = as_of + pd.Timedelta(days=365), as_of + pd.Timedelta(days=730), as_of + pd.Timedelta(days=1095)
    spot = spot_from([(as_of, 4.0), (e1, 3.0), (e2, 2.5)], as_of)
    path = forward_path(spot, as_of, [e1, e2], end, YEAR_DAYS)
    assert path.to_numpy() == pytest.approx([4.0, 3.0, 2.5], abs=1e-12)
    assert not path.attrs["pinned"] and np.isnan(path.attrs["residual_bp"])


def test_a_flat_curve_gives_a_flat_path_and_the_first_month_is_pinned():
    as_of = T("2024-01-02")
    spot = pd.Series(4.5, index=range(1, 25))
    meetings = [T("2024-01-18"), T("2024-03-07"), T("2024-04-25")]
    path = forward_path(spot, as_of, meetings, T("2024-06-06"), YEAR_DAYS, last_fixing=4.45)
    assert path.attrs["pinned"] and path.iloc[0] == pytest.approx(4.45)      # before the 1-month node
    later = forward_path(spot, as_of, meetings[1:], T("2024-06-06"), YEAR_DAYS, last_fixing=4.45)
    assert not later.attrs["pinned"] and later.to_numpy() == pytest.approx(4.5)
    always = forward_path(spot, as_of, meetings[1:], T("2024-06-06"), YEAR_DAYS, last_fixing=4.45, pin_always=True)
    assert always.attrs["pinned"] and always.iloc[0] == pytest.approx(4.45)


def test_a_curve_too_short_for_the_horizon_is_refused():
    spot = pd.Series(4.0, index=range(1, 7))
    with pytest.raises(ValueError, match="ends at 6 months"):
        forward_path(spot, T("2024-01-02"), [T("2024-03-01")], T("2024-09-01"), YEAR_DAYS)


def test_on_a_decision_day_the_rate_in_force_is_the_new_bank_rate(sonia, bank_rate):
    # 1 August 2024: cut 5.25 -> 5.00 at noon. The last published fixing is 31 July's, at the old rate.
    rate, last = rate_in_force(sonia, bank_rate, "2024-08-01", UK_BDAY)
    assert last == pytest.approx(5.20) and rate == pytest.approx(4.95)
    rate, last = rate_in_force(sonia, bank_rate, "2024-07-31", UK_BDAY)
    assert rate == last


def test_the_path_on_committed_sessions(curve, mpc, sonia, bank_rate):
    p = solve("2024-08-02", curve, mpc, sonia, bank_rate)
    m = p.meetings
    assert m["k"].tolist() == list(range(1, 9)) and m["announcement_date"].iloc[0] == T("2024-09-19")
    assert p.summary["rate_now"] == pytest.approx(4.95) and p.summary["pinned"]
    assert (m["step_bp"] < 0).all(), "the day after the first cut, a cutting cycle is priced"
    # the day the market priced an emergency hike after the mini-budget
    assert solve("2022-09-27", curve, mpc, sonia, bank_rate).meetings["step_bp"].iloc[0] > 100


# ---- the Bank's own conditioning paths --------------------------------------------------

def test_the_path_lands_on_the_banks_mpr_conditioning_path(curve, mpc, sonia, bank_rate):
    truth = conditioning.truth(DATA / "boe" / "conditioning_paths.csv")
    days = sorted(curve["date"].unique())
    solved = {d: solve(d, curve, mpc, sonia, bank_rate) for d in days}
    sessions = pd.DataFrame([{"session": d, **s.summary} for d, s in solved.items()])
    meetings = pd.concat([s.meetings.assign(session=d) for d, s in solved.items()], ignore_index=True)
    worst = {}
    for report in ["August 2019", "February 2023", "August 2024"]:
        got = conditioning.compare(truth[truth["report"] == report], sessions, meetings, sonia, UK_BDAY)
        assert len(got) == 3, report
        worst[report] = got["diff_bp"].abs().max()
    assert worst["August 2024"] < 0.5 and worst["August 2019"] < 2.0 and worst["February 2023"] < 5.0, worst


# ---- the SONIA - Bank Rate spread ---------------------------------------------------------

def test_the_spread_is_estimated_either_side_of_the_sonia_reform(sonia, bank_rate):
    target = target_range(bank_rate, bank_rate)
    breaks = GBP["rule"]["spread_breaks"]
    # on the reform day no reformed fixing is out yet: the pre-reform estimate stands
    assert operating_spread(sonia, target, "2018-04-23", 20, breaks) == operating_spread(sonia, target, "2018-04-23", 20)
    # three days later only the reformed fixings count
    after = operating_spread(sonia, target, "2018-04-26", 20, breaks)
    seen = sonia[(sonia["published"] <= T("2018-04-26")) & (sonia["date"] >= T("2018-04-23"))]
    assert len(seen) == 3                                  # the 23rd, 24th and 25th
    assert after == pytest.approx(float((seen["value"] - 0.5).median()))


# ---- the ONS vintages ----------------------------------------------------------------

def test_a_triangle_row_is_a_vintage_and_a_blank_is_not_a_revision():
    wide = pd.DataFrame([[5.0, np.nan], [5.1, 5.2], [np.nan, 5.2], [5.1, 5.3]],
                        index=pd.PeriodIndex(["2024-02", "2024-03", "2024-04", "2024-05"], freq="M"),
                        columns=pd.to_datetime(["2023-12-01", "2024-01-01"]))
    days = pd.Series(pd.to_datetime(["2024-02-13", "2024-03-12", "2024-04-16", "2024-05-14"]),
                     index=wide.index)
    v = ons.vintages_from_triangle(wide, days)
    dec = v[v["date"] == T("2023-12-01")]
    assert dec["value"].tolist() == [5.0, 5.1]
    assert dec["realtime_end"].tolist()[0] == T("2024-03-11") and pd.isna(dec["realtime_end"].iloc[-1])
    jan = v[v["date"] == T("2024-01-01")]
    assert jan["value"].tolist() == [5.2, 5.3] and jan["realtime_start"].tolist() == [T("2024-03-12"), T("2024-05-14")]


def test_before_the_ons_calendar_a_release_is_dated_late_never_early():
    days = pd.Series([T("2016-04-20")], index=pd.PeriodIndex(["2016-04"], freq="M"))
    got = ons.published_in(pd.PeriodIndex(["2013-06", "2016-04", "2015-12"], freq="M"), days)
    assert got[1] == T("2016-04-20")
    assert got[0] == T("2013-06-26") and got[2] == T("2015-12-29")   # 26 Dec 2015 is Boxing Day


def test_during_the_lfs_suspension_the_last_official_vintage_is_held(ons_panel):
    spec = GBP["macro"]
    for day in ["2023-10-24", "2023-12-01", "2024-02-12"]:
        n = nowcast(day, "GBP", ons_panel, spec)
        assert n["gap_month"] == T("2023-07-01"), day            # the September 2023 release
    assert nowcast("2024-02-13", "GBP", ons_panel, spec)["gap_month"] > T("2023-07-01")


def test_the_uk_nowcast_reads_published_cpi_and_a_constant_u_star(ons_panel):
    n = nowcast("2022-10-20", "GBP", ons_panel, GBP["macro"])
    assert n["infl_month"] == T("2022-09-01") and n["infl_12m"] == pytest.approx(10.1)
    assert n["u_star"] == GBP["macro"]["gap"]["natural_rate"]["constant"]
    assert "activity_z" not in n                     # no activity composite configured for GBP
    # CPI for September 2022 was released on 19 October: not known the day before
    assert nowcast("2022-10-18", "GBP", ons_panel, GBP["macro"])["infl_month"] == T("2022-08-01")
