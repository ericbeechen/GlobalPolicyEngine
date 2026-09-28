"""Appending future data must leave everything at date D unchanged, for every currency.

The publication boundary is the easiest place to leak: on session t the market
knew fixings through t - 1 only, and a settle or a curve is only as final as
the vintage that had arrived. Each test hands the chain extra data it should
not be able to see, poisons it, and checks nothing moves.

Every currency with fixtures (``tests/data/fixtures.yml``) runs the same tests,
through the same entry points the build scripts use (`policypath.regress`): the
config's extractor, `model.path.inputs`, `VintagePanel.from_cache`. A new
currency's fixtures block is all it takes to put it under them.
"""

import numpy as np
import pandas as pd
import pytest
from policypath import config, fixtures, market, panel, regress
from policypath.calendars import BDAYS, known_meetings

T = pd.Timestamp
CURRENCIES = fixtures.currencies()
POISON = 99.0


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise RuntimeError("the look-ahead tests touched the network")
    monkeypatch.setattr("socket.socket.connect", refuse)


def market_keys(ccy):
    """The cache keys the extractor reads as market data: settles or curves, final a business day late."""
    m = config.currency(ccy)["market"]
    return {(m[k]["source"], m[k]["series"]) for k in ("futures", "curve") if k in m}


def overnight_key(ccy):
    o = config.currency(ccy)["overnight"]
    return o["source"], o["series"]


def later(ccy, key, frame, day):
    """The rows a run at the end of `day` must not see.

    Market data by session: a session's own settles or curve arrive after it
    and are read on it, so what is after D is every later session. Everything
    else by when it was published.
    """
    return frame["date"] > day if key in market_keys(ccy) else frame["published"] > day


def truncate_logs(ccy, logs, day):
    return {key: f[~later(ccy, key, f, day)] for key, f in logs.items()}


def poison_logs(ccy, logs, day):
    """Everything after D set to POISON, and every session's market data revised to POISON just after its final.

    The revisions test the other side of the market boundary: a settle or curve
    is read as it stood when the final arrived (``market.final_after_bdays``),
    so a vintage landing the next morning must not reach any session, before D or after.
    """
    cfg = config.currency(ccy)
    after, bday = cfg["market"]["final_after_bdays"], BDAYS[cfg["calendar"]]
    out = {}
    for key, f in logs.items():
        f = f.assign(value=f["value"].mask(later(ccy, key, f, day), POISON))
        if key in market_keys(ccy):
            final = f["date"].map(lambda d: d + (after + 1) * bday) + pd.Timedelta(hours=9)
            f = pd.concat([f, f.assign(value=POISON, published=final)], ignore_index=True)
        out[key] = f
    return out


def truncate_vintages(vintages, day):
    """Each series as a pull at the end of `day` would have returned it (`VintagePanel.truncate`)."""
    out = {}
    for key, f in vintages.items():
        f = f[f["realtime_start"] <= day].copy()
        f.loc[f["realtime_end"] > day, "realtime_end"] = pd.NaT
        out[key] = f
    return out


def dirty_vintages(vintages, day, seed):
    """Every vintage after `day` scaled at random, and each series' latest print revised the day after D.

    Scaled row by row: a constant would flatten every growth rate after D and
    break the fits the later sessions run. The revision on D + 1 is what a one-day
    leak would see, whether or not anything was really released that day.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for key, f in vintages.items():
        f = f.assign(value=f["value"].where(f["realtime_start"] <= day, f["value"] * rng.uniform(1.5, 3.0, len(f))))
        live = f[(f["realtime_start"] <= day) & (f["realtime_end"].isna() | (f["realtime_end"] > day))]
        last = live[live["date"] == live["date"].max()]
        if len(last):
            i = last.index[-1]
            revised = f.loc[[i]].assign(value=f.at[i, "value"] * 2, realtime_start=day + pd.Timedelta(days=1))
            f.loc[i, "realtime_end"] = day
            f = pd.concat([f, revised], ignore_index=True)
        out[key] = f
    return out


def surprise(ccy, day):
    """The calendar with an unscheduled meeting announced after `day`, as the calendar would carry it later."""
    meetings = config.meetings(ccy)
    extra = pd.DataFrame({"announcement_date": [day + pd.Timedelta(days=10)],
                          "effective_date": [day + pd.Timedelta(days=11)],
                          "scheduled": [False], "cancelled": [pd.NaT]})
    return pd.concat([meetings, extra], ignore_index=True)


# ---- one session at a time: what the market path reads ---------------------------------

def solve(ccy, day, logs, tmp, calendar=None):
    fixtures.write(ccy, tmp, logs)
    return market.extractor(ccy, tmp, calendar).session(day)


@pytest.mark.parametrize("ccy, day", [(c, d) for c in CURRENCIES for d in fixtures.spec(c)["sessions"]])
def test_fixings_published_after_the_session_are_invisible(ccy, day, tmp_path):
    day = T(day)
    logs = fixtures.logs(ccy)
    key = overnight_key(ccy)
    fixings = logs[key]
    poisoned = {**logs, key: fixings.assign(value=fixings["value"].mask(fixings["published"] > day, POISON))}
    seen = {**logs, key: fixings[fixings["published"] <= day]}

    want = solve(ccy, day, seen, tmp_path / "seen")
    got = solve(ccy, day, poisoned, tmp_path / "poisoned")
    pd.testing.assert_frame_equal(got.meetings, want.meetings)
    assert got.summary == want.summary
    assert (poisoned[key].loc[poisoned[key]["date"] == day, "value"] == POISON).all(), \
        "the session's own fixing must be among the poisoned rows"


@pytest.mark.parametrize("ccy", CURRENCIES)
def test_an_unscheduled_meeting_announced_later_is_invisible(ccy, tmp_path):
    day = fixtures.sessions(ccy)[-2]
    logs = fixtures.logs(ccy)
    want = solve(ccy, day, logs, tmp_path)
    got = solve(ccy, day, logs, tmp_path, surprise(ccy, day))
    pd.testing.assert_frame_equal(got.meetings, want.meetings)
    assert got.summary == want.summary
    # ...and the same meeting, announced by the session, is a pillar.
    announced = surprise(ccy, day)
    announced.loc[announced.index[-1], "announcement_date"] = day
    known = solve(ccy, day, logs, tmp_path, announced)
    assert day + pd.Timedelta(days=11) in set(known.meetings["effective_date"])


def test_a_settle_revised_after_the_final_is_not_used():
    """A futures settle is read as it stood when the final arrived (``market.final_after_bdays``), not later."""
    ccy = next(c for c in config.enabled() if config.currency(c)["market"]["extractor"] == "futures_strip")
    cfg = config.currency(ccy)
    day = T("2024-09-17")
    vintages = pd.DataFrame({
        "date": [day] * 3,
        "contract": ["ZQV4"] * 3,
        "value": [95.20, 95.21, 99.99],
        "published": pd.to_datetime(["2024-09-17 16:01", "2024-09-17 19:40", "2024-09-25 09:00"]),
        "retrieved": T("2026-09-24"),
    })
    used = panel.settles_on(vintages, day, BDAYS[cfg["calendar"]], cfg["market"]["final_after_bdays"])
    assert used["value"].item() == 95.21


def test_the_calendar_on_each_day_of_march_2020():
    """Emergency cuts count from their announcement; the meeting they replaced until it was called off."""
    meetings = config.meetings("USD")

    def pillars(day):
        return set(known_meetings(meetings, day)["effective_date"].dt.strftime("%Y-%m-%d"))
    march = {"2020-03-04", "2020-03-16", "2020-03-19"}
    assert pillars("2020-03-02") & march == {"2020-03-19"}
    assert pillars("2020-03-03") & march == {"2020-03-04", "2020-03-19"}
    assert pillars("2020-03-13") & march == {"2020-03-04", "2020-03-19"}
    assert pillars("2020-03-16") & march == {"2020-03-04", "2020-03-16"}


# ---- the whole chain: market path, nowcast, r*, model path, gap, z, P&L -----------------

LOOKAHEAD = [(c, d) for c in CURRENCIES for d in fixtures.spec(c)["lookahead"]]


@pytest.mark.parametrize("ccy, day", LOOKAHEAD)
def test_the_whole_pipeline_at_d_ignores_everything_after_d(ccy, day):
    """Truncate every input at D and run end to end; then append everything after D, then poison it.

    The output for every session up to D must not move. This is what covers the
    r* step-fill (a later SEP must not reach back), hold-flat conditioning (a
    later print must not reach the path at D), the calendar (a meeting
    announced after D is not a pillar before it) and the market data's own
    publication lag (D's settles or curve are read as they stood by the final).
    """
    day = T(day)
    logs, vintages = fixtures.logs(ccy), fixtures.vintages(ccy)
    sessions = fixtures.sessions(ccy)
    early = sessions[sessions <= day]
    assert 3 <= len(early) < len(sessions), "D needs sessions before it and at least one after"

    want = regress.fixture_outputs(ccy, truncate_logs(ccy, logs, day), truncate_vintages(vintages, day),
                                   macro_days=early, calendar=known_meetings(config.meetings(ccy), day))
    appended = regress.truncate(regress.fixture_outputs(ccy), day)
    poisoned = regress.truncate(regress.fixture_outputs(ccy, poison_logs(ccy, logs, day),
                                                        dirty_vintages(vintages, day, seed=len(ccy) + day.day),
                                                        calendar=surprise(ccy, day)), day)
    for got in (appended, poisoned):
        for stage in regress.STAGES:
            pd.testing.assert_frame_equal(got[stage], want[stage], check_dtype=False, obj=f"{ccy} {stage}")
    # The stages really ran: a z exists at D, and the rule read D's own nowcast.
    signal = want["signal"].set_index(["session", "k"])
    assert signal.loc[(day, 2), "z"] == signal.loc[(day, 2), "z"]
    assert want["model"].set_index("session").loc[day, "macro_as_of"] == day
