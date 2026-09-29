"""The ELB state and the regime state machine, on synthetic policy-rate paths labelled by hand."""

import pandas as pd
import pytest
from policypath import regimes

SESSIONS = pd.bdate_range("2010-01-04", "2014-12-31")
# (effective session, change in percent)
MOVES = [("2010-09-01", 0.25), ("2011-07-01", 0.25), ("2011-10-03", 0.25), ("2012-06-01", -0.25),
         ("2013-02-01", 0.25), ("2013-04-01", -0.25), ("2013-06-03", 0.25)]
ELB_WINDOWS = [("2010-01-04", "2010-06-30"), ("2014-09-01", "2014-09-30")]
# The first session of each state, by hand from the rules in the module docstring.
LABELS = [
    ("2010-01-04", "elb"),
    ("2010-07-01", "hold_after_cuts"),     # off the floor, no move yet
    ("2010-09-01", "early_hiking"),        # the first hike; the 10-month pause to 2011-07 stays early
    ("2011-09-01", "late_hiking"),         # 12 months in, a hike (2011-07-01) in the last six months
    ("2012-04-03", "hold_after_hikes"),    # six months after the last hike, 2011-10-03
    ("2012-06-01", "cutting"),
    ("2012-12-03", "hold_after_cuts"),     # six months after the cut (2012-12-01 is a Saturday)
    ("2013-02-01", "early_hiking"),        # a new cycle: the first hike since the cut
    ("2013-04-01", "cutting"),             # a cut ends it
    ("2013-06-03", "early_hiking"),        # a hike two months after a cut starts another cycle
    ("2014-06-03", "hold_after_hikes"),    # 12 months in and no hike for six: straight past late hiking
    ("2014-09-01", "elb"),                 # the ELB state overrides the moves
    ("2014-10-01", "hold_after_hikes"),
]


def path():
    step = pd.Series(0.0, index=SESSIONS)
    for day, change in MOVES:
        step[pd.Timestamp(day)] = change
    rate = 0.5 + step.cumsum()
    elb = pd.Series(False, index=SESSIONS)
    for a, b in ELB_WINDOWS:
        elb[a:b] = True
    return rate, elb


def expected():
    starts = pd.Series([s for _, s in LABELS], index=pd.DatetimeIndex([d for d, _ in LABELS]))
    return starts.reindex(SESSIONS, method="ffill")


def test_every_session_gets_its_hand_label():
    rate, elb = path()
    got = regimes.regimes(rate, elb)
    wrong = got[got != expected()]
    assert wrong.empty, wrong.groupby(wrong).head(1)


@pytest.mark.parametrize("cut", ["2011-08-31", "2012-04-02", "2013-05-31", "2014-06-02"])
def test_a_session_regime_uses_only_moves_up_to_it(cut):
    rate, elb = path()
    full = regimes.regimes(rate, elb)
    short = regimes.regimes(rate[:cut], elb[:cut])
    pd.testing.assert_series_equal(short, full[:cut])


def test_a_move_below_the_tolerance_is_not_a_move():
    rate, elb = path()
    wiggle = rate + pd.Series(1e-12, index=SESSIONS).where(SESSIONS.day % 2 == 0, 0.0)
    assert len(regimes.moves(wiggle)) == len(MOVES)
    pd.testing.assert_series_equal(regimes.regimes(wiggle, elb), regimes.regimes(rate, elb))


def test_elb_state_needs_the_rate_on_its_floor_and_the_rule_below_it():
    model = pd.DataFrame({
        "session": pd.bdate_range("2020-01-06", periods=5),
        "r0": [0.125, 0.125, 0.375, 0.1, 0.25],
        "elb": [0.125, 0.125, 0.125, 0.1, 0.1],      # the floor is a schedule: it moved on the fourth day
        "at_elb": [True, False, True, True, True],
    })
    got = regimes.elb_state(model)
    assert got.tolist() == [True, False, False, True, False]
    assert got.index.equals(pd.DatetimeIndex(model["session"]))


def test_states_reads_a_model_panel():
    rate, elb = path()
    # The floor is the rate itself in the ELB windows and zero elsewhere; the rule is below it throughout.
    model = pd.DataFrame({"session": SESSIONS, "r0": rate.to_numpy(), "elb": rate.where(elb, 0.0).to_numpy(),
                          "at_elb": True})
    pd.testing.assert_series_equal(regimes.states(model), regimes.regimes(rate.rename_axis("session"), elb),
                                   check_freq=False)


def test_episodes_are_runs_in_order():
    state = pd.Series(["elb", "elb", "cutting", "elb"], index=pd.bdate_range("2020-01-06", periods=4))
    got = regimes.episodes(state)
    assert got["state"].tolist() == ["elb", "cutting", "elb"]
    assert got["sessions"].tolist() == [2, 1, 1]
    assert got["first"].tolist() == [pd.Timestamp("2020-01-06"), pd.Timestamp("2020-01-08"), pd.Timestamp("2020-01-09")]
    assert got["last"].iloc[0] == pd.Timestamp("2020-01-07")


def test_lift_off_ends_the_elb_state_on_its_own_session():
    sessions = pd.bdate_range("2015-06-01", "2016-06-30")
    r0 = pd.Series(0.125, index=sessions).where(sessions < "2015-12-17", 0.375)
    model = pd.DataFrame({"session": sessions, "r0": r0.to_numpy(), "elb": 0.125, "at_elb": True})
    got = regimes.states(model)
    assert got[:"2015-12-16"].eq("elb").all()
    assert got["2015-12-17":].eq("early_hiking").all()


def test_a_hike_in_the_last_sessions_starts_a_cycle_there():
    sessions = pd.bdate_range("2025-06-02", "2026-09-21")
    rate = pd.Series(4.0, index=sessions).where(sessions < "2025-12-11", 3.75).where(sessions < "2026-09-17", 4.0)
    got = regimes.regimes(rate, pd.Series(False, index=sessions))
    assert got["2025-12-11":"2026-06-10"].eq("cutting").all() and got["2026-06-11":"2026-09-16"].eq("hold_after_cuts").all()
    assert got["2026-09-17":].tolist() == ["early_hiking"] * 3


def test_a_missing_rate_does_not_hide_the_move_across_it():
    rate, elb = path()
    gappy = rate.copy()
    gappy.iloc[gappy.index.get_loc(pd.Timestamp(MOVES[2][0])) - 1] = float("nan")
    assert len(regimes.moves(gappy)) == len(MOVES)
    pd.testing.assert_series_equal(regimes.regimes(gappy, elb), regimes.regimes(rate, elb))


def test_a_missing_elb_state_is_not_the_elb_state():
    rate, elb = path()
    unknown = pd.Series(None, index=SESSIONS, dtype=object)
    assert not regimes.regimes(rate, unknown).eq("elb").any()
    model = pd.DataFrame({"session": SESSIONS[:3], "r0": 0.1, "elb": 0.1, "at_elb": pd.Series([True, None, False],
                                                                                              dtype=object)})
    assert regimes.elb_state(model).tolist() == [True, False, False]
