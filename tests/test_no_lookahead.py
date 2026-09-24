"""Appending future data must leave the path at date D unchanged.

The publication boundary is the easiest place to leak: on session t the market
knew fixings through t - 1 only, and a settle is only as final as the vintage
that had arrived. Each test hands the panel extra data it should not be able to
see, poisons it, and checks nothing moves.
"""

from pathlib import Path
import pandas as pd
import pytest
from policypath import config, panel

DATA = Path(__file__).parent / "data"
SESSIONS = ["2022-06-13", "2023-06-13", "2024-09-17", "2026-09-21"]


@pytest.fixture(scope="module")
def fixings():
    e = pd.read_csv(DATA / "effr.csv", parse_dates=["date", "published"])
    return e.rename(columns={"effr": "value"})[["date", "value", "published"]]


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
