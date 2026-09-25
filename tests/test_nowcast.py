"""The macro nowcast on dated vintages: each input reads what was known on the day.

Values come from the committed ALFRED fixtures (tests/data/alfred), so they are
the real-time numbers, not today's revised ones.
"""

import pandas as pd
import pytest
from policypath import config
from policypath.macro.gap import unemployment_gap
from policypath.macro.vintage import VintagePanel

T = pd.Timestamp
MACRO = config.currency("USD")["macro"]
GAP = MACRO["gap"]


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
