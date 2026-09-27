"""Appending future data must leave everything at date D unchanged.

The publication boundary is the easiest place to leak: on session t the market
knew fixings through t - 1 only, and a settle is only as final as the vintage
that had arrived. Each test hands the panel extra data it should not be able to
see, poisons it, and checks nothing moves.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from policypath import config, panel
from policypath.backtest import engine, instrument
from policypath.calendars import known_meetings
from policypath.macro.nowcast import nowcast
from policypath.macro.vintage import VintagePanel
from policypath.model import path as model
from policypath.signal import gap as gap_signal

DATA = Path(__file__).parent / "data"
SESSIONS = ["2022-06-13", "2023-06-13", "2024-09-17", "2026-09-21"]


@pytest.fixture(scope="module")
def meetings():
    return config.meetings("USD")


def strip(day):
    s = pd.read_csv(DATA / f"zq_strip_{pd.Timestamp(day):%Y-%m-%d}.csv")
    return pd.Series(s["implied_rate"].to_numpy(), index=pd.PeriodIndex(s["month"], freq="M"))


def solve(day, fixings, meetings):
    spec = config.currency("USD")["path"]
    return panel.solve_session(day, strip(day), meetings, fixings, spec["n_meetings"],
                               spec["min_forward_days"], spec["min_regime_days"])


@pytest.mark.parametrize("day", SESSIONS)
def test_fixings_published_after_the_session_are_invisible(day, fixings, meetings):
    day = pd.Timestamp(day)
    seen = fixings[fixings["published"] <= day]
    poisoned = fixings.copy()
    poisoned.loc[poisoned["published"] > day, "value"] = 99.0  # including day's own fixing

    want = solve(day, seen, meetings)
    got = solve(day, poisoned, meetings)

    pd.testing.assert_frame_equal(got.meetings, want.meetings)
    assert got.summary == want.summary
    assert (poisoned.loc[poisoned["date"] == day, "value"] == 99.0).all(), \
        "the session's own fixing must be among the poisoned rows"


def test_a_settle_revised_after_the_next_business_day_is_not_used():
    day = pd.Timestamp("2024-09-17")
    vintages = pd.DataFrame({
        "date": [day] * 3,
        "contract": ["ZQV4"] * 3,
        "value": [95.20, 95.21, 99.99],
        "published": pd.to_datetime(["2024-09-17 16:01", "2024-09-17 19:40", "2024-09-25 09:00"]),
        "retrieved": pd.Timestamp("2026-09-24"),
    })
    used = panel.settles_on(vintages, day)
    assert used["value"].item() == 95.21


def test_the_calendar_on_each_day_of_march_2020(meetings):
    """Emergency cuts count from their announcement; the meeting they replaced until it was called off."""
    def pillars(day):
        return set(known_meetings(meetings, day)["effective_date"].dt.strftime("%Y-%m-%d"))
    march = {"2020-03-04", "2020-03-16", "2020-03-19"}
    assert pillars("2020-03-02") & march == {"2020-03-19"}
    assert pillars("2020-03-03") & march == {"2020-03-04", "2020-03-19"}
    assert pillars("2020-03-13") & march == {"2020-03-04", "2020-03-19"}
    assert pillars("2020-03-16") & march == {"2020-03-04", "2020-03-16"}


def test_an_unscheduled_meeting_announced_later_is_invisible(fixings, meetings):
    day = pd.Timestamp("2024-09-17")
    surprise = pd.DataFrame({"announcement_date": [pd.Timestamp("2024-10-10")],
                             "effective_date": [pd.Timestamp("2024-10-11")],
                             "scheduled": [False], "cancelled": [pd.NaT]})
    poisoned = pd.concat([meetings, surprise], ignore_index=True)

    want = solve(day, fixings, meetings)
    got = solve(day, fixings, poisoned)
    pd.testing.assert_frame_equal(got.meetings, want.meetings)
    assert got.summary == want.summary
    # ...and the same meeting, announced by the session, is a pillar.
    known = solve(day, fixings, poisoned.assign(announcement_date=poisoned["announcement_date"].where(
        poisoned["scheduled"], day)))
    assert pd.Timestamp("2024-10-11") in set(known.meetings["effective_date"])


# ---- the whole pipeline: market path, nowcast, r*, model path, gap, z, P&L ----------

# Every committed strip session the ALFRED fixtures reach (they start in 2018).
PIPELINE_SESSIONS = ["2022-06-01", "2022-06-13", "2023-06-13", "2024-09-17", "2026-09-21"]
MACRO = config.currency("USD")["macro"]
MACRO_SPEC = {**MACRO, "activity": {**MACRO["activity"], "moments_start": "2018-04-01"},
              "inflation": {**MACRO["inflation"], "bridge_window": 24}}
# The spread over one fixing, not a median of 20: one leaked fixing cannot move a median.
RULE = {**config.currency("USD")["rule"], "spread_fixings": 1}
# Five sessions make a short history: a window that spans them, and a z from the second on.
SIGNAL = {**config.currency("USD")["signal"], "window": "1600D", "min_periods": 1}


def pipeline(days, fixings, target, sep, vintages, meetings):
    """Every stage, end to end, on the fixture sessions `days`, from the inputs given."""
    spec = config.currency("USD")["path"]
    summaries, frames, rows = [], [], []
    for day in map(pd.Timestamp, days):
        s = panel.solve_session(day, strip(day), meetings, fixings, spec["n_meetings"],
                                spec["min_forward_days"], spec["min_regime_days"])
        summaries.append({"session": day, "error": None, **s.summary})
        frames.append(s.meetings.assign(session=day))
        rows.append(nowcast(day, panel=vintages, spec=MACRO_SPEC))
    sessions, market = pd.DataFrame(summaries), pd.concat(frames, ignore_index=True)
    model_summary, paths = model.build(sessions, market, pd.DataFrame(rows), sep, target, fixings, RULE)
    signal = gap_signal.build(paths, SIGNAL)
    z = signal[signal["k"] == 2].set_index("session")["z"]
    pnl = engine.run(z, instrument.held_rate_change(sessions, market, 2), lag=0)
    return {"model_summary": model_summary.set_index("session"), "paths": paths.set_index(["session", "k"]),
            "signal": signal.set_index(["session", "k"]), "pnl": pnl}


def through(result, day):
    return {name: frame.loc[frame.index.get_level_values(0) <= day] for name, frame in result.items()}


@pytest.mark.parametrize("day", ["2023-06-13", "2024-09-17"])
def test_the_whole_pipeline_at_d_ignores_everything_after_d(day, fixings, target, sep, raw, meetings):
    """Truncate every input at D and run end to end; then append everything after D, then poison it.

    The output for every session up to D must not move. This is what covers the
    SEP step-fill (a later dot must not reach back) and hold-flat conditioning
    (a later print must not reach the path at D).
    """
    day = pd.Timestamp(day)
    projections = MACRO["projections"]
    by_published = lambda f: f[f["published"] <= day]
    poison = lambda f: f.assign(value=f["value"].mask(f["published"] > day, 99.0))
    full = VintagePanel(raw, projections)
    # Later vintages scaled at random, row by row: a constant would flatten every
    # growth rate after D and break the fits the later sessions run.
    later = raw["realtime_start"] > day
    noise = np.random.default_rng(0).uniform(1.5, 3.0, len(raw))
    dirty = VintagePanel(raw.assign(value=raw["value"].where(~later, raw["value"] * noise)), projections)
    early = [d for d in PIPELINE_SESSIONS if pd.Timestamp(d) <= day]
    assert len(early) >= 3 and len(early) < len(PIPELINE_SESSIONS)

    want = pipeline(early, by_published(fixings), by_published(target), by_published(sep),
                    full.truncate(day), known_meetings(meetings, day))
    appended = through(pipeline(PIPELINE_SESSIONS, fixings, target, sep, full, meetings), day)
    poisoned = through(pipeline(PIPELINE_SESSIONS, poison(fixings), poison(target), poison(sep), dirty,
                                meetings), day)
    for got in (appended, poisoned):
        for name in want:
            pd.testing.assert_frame_equal(got[name], want[name], check_dtype=False, obj=name)
    # the stages really ran: a z exists at D, and the rule's inputs are D's
    assert want["signal"].loc[(day, 2), "z"] == want["signal"].loc[(day, 2), "z"]
    assert want["model_summary"].loc[day, "macro_as_of"] == day
