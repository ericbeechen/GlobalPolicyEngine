"""The model path: r* from the SEP, the balanced-approach rule, the ELB floor, hold-flat.

Coefficients come from config and nothing is estimated, so most of this is
algebra checked by hand. The dated checks read the committed SEP and target
fixtures (tests/data/sep.csv, effr.csv).
"""

import numpy as np
import pandas as pd
import pytest
from policypath import config
from policypath.model.path import model_path, operating_spread, policy_rate
from policypath.model.reaction import floor_at_elb, inertial_path, notional, per_meeting_inertia
from policypath.model.rstar import rstar

T = pd.Timestamp
RULE = config.currency("USD")["rule"]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise RuntimeError("the model touched the network")
    monkeypatch.setattr("socket.socket.connect", refuse)


# ---- r* ------------------------------------------------------------------------

def test_an_sep_counts_from_its_release_day_and_not_before(sep):
    # 2024-09-18: the longer-run median rose 2.8 -> 2.9, released with the statement.
    assert rstar(sep, "2024-09-17", RULE)["longer_run"] == 2.8
    on = rstar(sep, "2024-09-18", RULE)
    assert on["longer_run"] == 2.9 and on["sep_date"] == T("2024-09-18")
    assert on["rstar"] == pytest.approx(0.9)


def test_before_the_first_dot_rstar_is_taylors_two_percent(sep):
    before = rstar(sep, "2012-01-24", RULE)
    assert before["rstar"] == RULE["rstar"]["before_first"] == 2.0
    assert pd.isna(before["sep_date"])
    assert rstar(sep, "2012-01-25", RULE)["sep_date"] == T("2012-01-25")


def test_the_cancelled_march_2020_sep_leaves_december_2019_in_force(sep):
    for day in ["2020-03-18", "2020-06-09"]:
        assert rstar(sep, day, RULE)["sep_date"] == T("2019-12-11")
    assert rstar(sep, "2020-06-10", RULE)["sep_date"] == T("2020-06-10")


def test_a_later_sep_does_not_reach_back(sep):
    poisoned = sep.assign(value=sep["value"].mask(sep["published"] > T("2023-06-13"), 99.0))
    assert rstar(poisoned, "2023-06-13", RULE) == rstar(sep, "2023-06-13", RULE)


# ---- the rule ------------------------------------------------------------------

def test_the_balanced_approach_rule_by_hand():
    # r* 0.5, core PCE 3%, u 0.5pp above u*: 0.5 + 3 + 0.5 x 1 + 2 x (-0.5) = 3.0
    assert notional(3.0, 0.5, 0.5, RULE) == pytest.approx(3.0)
    # a tight labour market adds: u 1pp below u* is +2pp
    assert notional(2.0, -1.0, 0.5, RULE) == pytest.approx(4.5)


def test_coefficients_come_from_config():
    taylor93 = {**RULE, "coefficients": {**RULE["coefficients"], "unemployment_gap": 1.0}}
    assert notional(3.0, 0.5, 0.5, taylor93) == pytest.approx(3.5)
    assert notional(3.0, 0.5, 0.5, taylor93) != notional(3.0, 0.5, 0.5, RULE)


def test_inertia_closes_fifteen_percent_of_the_distance_each_quarter():
    rho = per_meeting_inertia(RULE)
    assert rho == pytest.approx(0.85 ** 0.5)
    path = inertial_path(0.125, 3.0, 8, RULE)
    gap_left = 3.0 - path
    # two meetings to the quarter: after each pair, 0.85 of the distance remains
    assert gap_left[1::2] == pytest.approx(2.875 * 0.85 ** np.arange(1, 5))
    assert np.all(np.diff(path) > 0) and path[-1] < 3.0


def test_the_floor_is_the_elb_and_only_binds_below_it():
    assert floor_at_elb(-4.2, RULE) == RULE["elb"] == 0.125
    assert floor_at_elb(0.3, RULE) == 0.3


# ---- the path on a date ----------------------------------------------------------

def macro(day, infl, gap):
    return {"as_of": T(day), "published": T(day), RULE["inflation"]: infl, RULE["gap"]: gap}


MEETINGS = pd.to_datetime(["2021-11-04", "2021-12-16", "2022-01-27", "2022-03-17",
                           "2022-05-05", "2022-06-16", "2022-07-28", "2022-09-22"])


def test_the_range_in_force_is_known_on_the_day_it_applies(target):
    # The September 2024 cut was announced on the 18th, in force from the 19th.
    assert policy_rate(target, "2024-09-18")[0] == 5.375
    assert policy_rate(target, "2024-09-19")[0] == 4.875


def test_the_operating_spread_uses_only_fixings_published_by_the_session(fixings, target):
    day = T("2024-09-20")
    s = operating_spread(fixings, target, day, 20)
    seen = fixings[fixings["published"] <= day].tail(20).set_index("date")["value"]
    mid = target.set_index("date")["value"].reindex(seen.index)
    assert s == pytest.approx(float((seen - mid).median()))
    assert fixings[fixings["published"] <= day]["date"].max() == T("2024-09-19")


def test_at_the_lower_bound_the_model_sits_on_the_floor_while_the_rule_says_negative(sep, target, fixings):
    # Autumn 2021 inflation with 2020's slack: the rule says well below zero.
    day = "2021-01-04"
    summary, path = model_path(day, MEETINGS, macro(day, 1.4, 3.0), sep, target, fixings, RULE)
    assert summary["notional"] < -2 and summary["at_elb"]
    assert summary["goal"] == RULE["elb"]
    assert path["model_mid"].to_numpy() == pytest.approx(np.full(8, RULE["elb"]))
    assert path["model"].to_numpy() == pytest.approx(RULE["elb"] + summary["spread_bp"] / 100)


def test_hold_flat_the_same_macro_picture_gives_the_same_goal_at_every_meeting(sep, target, fixings):
    day = "2021-11-01"
    summary, path = model_path(day, MEETINGS, macro(day, 4.1, 0.2), sep, target, fixings, RULE)
    assert summary["goal"] == pytest.approx(notional(4.1, 0.2, summary["rstar"], RULE))
    want = inertial_path(summary["r0"], summary["goal"], 8, RULE) + summary["spread_bp"] / 100
    assert path["model"].to_numpy() == pytest.approx(want)
    assert path["effective_date"].tolist() == list(MEETINGS)


def test_a_nowcast_from_after_the_session_is_refused(sep, target, fixings):
    with pytest.raises(ValueError, match="after the session"):
        model_path("2021-11-01", MEETINGS, macro("2021-11-02", 4.1, 0.2), sep, target, fixings, RULE)
