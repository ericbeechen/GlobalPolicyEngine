"""The data layer's no-look-ahead guarantee, and dated facts it has to get right.

`VintagePanel.as_of(D)` must return what a person could have seen at the end
of D: nothing published later, and nothing that betrays a later revision.
"""

import inspect
import re
import pandas as pd
import pytest
from policypath import config
from policypath.macro.vintage import VintagePanel

T = pd.Timestamp
PROJECTIONS = config.currency("USD")["macro"]["projections"]
# Just before a benchmark, around the pandemic, and inside the shutdown's ragged edge.
CUTOFFS = ["2020-04-02", "2023-06-30", "2025-02-06", "2025-12-17"]
# `raw` and `panel` (every committed ALFRED vintage) come from conftest.py.


def days_up_to(cutoff, span=400, step=6):
    cutoff = T(cutoff)
    return pd.date_range(cutoff - pd.Timedelta(days=span), cutoff, freq=f"{step}D").append(pd.DatetimeIndex([cutoff]))


# ---- construction and API invariants ------------------------------------------

def test_construction_refuses_a_series_without_realtime_intervals(panel):
    plain = panel.as_of("2024-01-20")   # series, date, value, published: what a convenience source returns
    with pytest.raises(ValueError, match="real-time intervals"):
        VintagePanel(plain)


def test_construction_refuses_overlapping_vintages(raw):
    dup = pd.concat([raw, raw[raw["realtime_end"].isna()].head(1)])
    with pytest.raises(ValueError, match="overlap"):
        VintagePanel(dup, PROJECTIONS)


def test_a_projection_must_be_declared(raw):
    with pytest.raises(ValueError, match="NROU"):
        VintagePanel(raw)


def test_as_of_returns_a_copy(panel):
    got = panel.as_of("2024-01-20")
    got["value"] = -1.0
    assert (panel.as_of("2024-01-20")["value"] != -1.0).all()


def test_there_is_no_latest_values_method():
    public = [n for n in dir(VintagePanel) if not n.startswith("_")]
    assert not [n for n in public if re.search("latest|current|recent|last|now", n)]


@pytest.mark.parametrize("method", ["as_of", "series", "first_release", "release_dates"])
def test_every_way_out_requires_an_as_of(method):
    params = inspect.signature(getattr(VintagePanel, method)).parameters
    assert "as_of" in params and params["as_of"].default is inspect.Parameter.empty


# ---- no look-ahead: test_no_lookahead.py in embryo ---------------------------

@pytest.mark.parametrize("cutoff", CUTOFFS)
def test_later_vintages_do_not_change_what_was_known(panel, cutoff):
    early = panel.truncate(cutoff)
    for day in days_up_to(cutoff):
        pd.testing.assert_frame_equal(panel.as_of(day), early.as_of(day))
        pd.testing.assert_frame_equal(panel.first_release(day), early.first_release(day))


@pytest.mark.parametrize("cutoff", CUTOFFS)
def test_poisoned_later_vintages_are_invisible(raw, cutoff):
    later = raw["realtime_start"] > T(cutoff)
    assert later.any()
    clean, dirty = VintagePanel(raw, PROJECTIONS), VintagePanel(raw.assign(value=raw["value"].mask(later, 99.0)), PROJECTIONS)
    for day in days_up_to(cutoff):
        pd.testing.assert_frame_equal(dirty.as_of(day), clean.as_of(day))


# ---- dated spot checks -------------------------------------------------------

def test_on_the_first_of_march_2020_nobody_knew_about_march(panel):
    payrolls = panel.series("PAYEMS", "2020-03-01").dropna()
    assert payrolls.index[-1] == T("2020-01-01")
    assert payrolls.diff().iloc[-1] == pytest.approx(225)          # January's first print
    assert panel.series("UNRATE", "2020-03-01").dropna().iloc[-1] == pytest.approx(3.6)


def test_the_march_2020_collapse_appears_on_its_release_day(panel):
    before, on = (panel.series("PAYEMS", d).dropna() for d in ["2020-04-02", "2020-04-03"])
    assert before.index[-1] == T("2020-02-01")
    assert on.index[-1] == T("2020-03-01") and on.diff().iloc[-1] == pytest.approx(-701)
    assert panel.series("UNRATE", "2020-05-07").dropna().iloc[-1] == pytest.approx(4.4)
    assert panel.series("UNRATE", "2020-05-08").dropna().iloc[-1] == pytest.approx(14.7)


def test_january_20th_2024_has_december_cpi_but_only_november_pce(panel):
    last = lambda s: panel.series(s, "2024-01-20").dropna().index[-1]
    for s in ["PAYEMS", "UNRATE", "CPILFESL", "INDPRO", "RSAFS"]:
        assert last(s) == T("2023-12-01"), s
    assert last("PCEPILFE") == T("2023-11-01")
    assert last("NEWORDER") == T("2023-11-01")


def test_the_shutdown_leaves_a_hole_published_as_missing(panel):
    assert T("2025-10-01") not in panel.series("UNRATE", "2025-12-15").index     # not yet published
    u = panel.series("UNRATE", "2025-12-16")
    assert pd.isna(u[T("2025-10-01")]) and u.dropna().index[-1] == T("2025-11-01")
    last = lambda s: panel.series(s, "2025-12-19").dropna().index[-1]
    assert last("CPILFESL") == T("2025-11-01") and last("PCEPILFE") == T("2025-09-01")


# ---- known revisions: proves vintages are doing something --------------------

def test_the_2024_payroll_benchmark(panel):
    march = lambda day: panel.series("PAYEMS", day)[T("2024-03-01")]
    first = panel.first_release("2026-09-24", "PAYEMS").set_index("date").loc[T("2024-03-01")]
    assert first["value"] == pytest.approx(158_133) and first["published"] == T("2024-04-05")
    # The preliminary -818k (21 Aug 2024) was an announcement, not data.
    assert march("2024-08-22") == march("2024-08-20") == pytest.approx(158_106)
    # The final benchmark: BLS reported -589k seasonally adjusted.
    assert march("2025-02-07") - march("2025-02-06") == pytest.approx(-589)


def test_seasonally_adjusted_cpi_is_revised_and_unadjusted_is_not(panel):
    june = lambda s, d: panel.series(s, d)[T("2023-06-01")]
    # New seasonal factors land on their own vintage, four days before January's CPI.
    assert june("CPILFESL", "2024-02-08") == pytest.approx(308.309)
    assert june("CPILFESL", "2024-02-09") == pytest.approx(308.245)
    assert june("CPILFENS", "2024-02-08") == june("CPILFENS", "2026-09-24") == pytest.approx(308.910)
    nsa = panel.first_release("2026-09-24", "CPILFENS").set_index("date")["value"]
    now = panel.series("CPILFENS", "2026-09-24").reindex(nsa.index)
    assert (nsa - now).abs().max() == 0
