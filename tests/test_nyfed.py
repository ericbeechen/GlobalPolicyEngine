"""The HLW real-time r*: one value per vintage, found by its headers, dated as it was published.

The workbook here is synthetic, in the NY Fed's two layouts: the 2017 model's
(output gap first, four economies) and the 2023 model's (r* third, three
economies), with the 2020Q2-2022Q3 gap between them. A reader that took the
first "US" column would get the output gap in one layout and trend growth in
the other, so the values under each group differ. Also here: the registry
tables that name the sources.
"""

import datetime
import io
import openpyxl
import pandas as pd
import pytest
from policypath import config
from policypath.sources import boe, nyfed, registry

T = pd.Timestamp
R_STAR = "Natural Rate (r*)"
LAYOUT_2017 = (["Output Gap", "Trend Growth (g), Annualized", "Other Determinants (z)", R_STAR],
               ["US", "Canada", "Euro Area", "UK"])
LAYOUT_2023 = (["Trend Growth (g), Annualized", "Other Determinants (z)", R_STAR, "Output Gap"],
               ["US", "Canada", "Euro Area"])
# Vintage -> (layout, US r* in its last quarter). 2020Q1's banner names the wrong quarter, as 2019Q2's does.
VINTAGES = {"2019Q4": (LAYOUT_2017, 0.49), "2020Q1": (LAYOUT_2017, 0.53), "2022Q4": (LAYOUT_2023, 0.70)}
# HLW's on-schedule release days, from the NY Fed page's archived copies (2018 from August, 2019, 2020, 2025, 2026).
HLW_SCHEDULE = {"2018Q2": "2018-08-31", "2018Q3": "2018-12-03", "2018Q4": "2019-03-04", "2019Q1": "2019-06-03",
                "2019Q2": "2019-09-03", "2019Q3": "2019-11-29", "2019Q4": "2020-03-02", "2020Q1": "2020-05-29",
                "2020Q2": "2020-08-31", "2024Q4": "2025-03-03", "2025Q1": "2025-06-02", "2025Q2": "2025-09-02",
                "2026Q1": "2026-06-01", "2026Q2": "2026-08-28"}


def vintage_rows(vintage, layout, rstar, banner=None):
    groups, countries = layout
    width = len(countries) + 1                                    # a blank column after each group
    group_row, country_row = [None, None], ["Date", None]
    for g in groups:
        group_row += [g] + [None] * (width - 1)
        country_row += countries + [None]
    rows = [["This spreadsheet contains updated estimates"], [f"Final data point: {banner or vintage}"], [None],
            [None], group_row, country_row]
    last = pd.Period(vintage, freq="Q")
    for q in pd.period_range(last - 2, last, freq="Q"):
        row = [q.start_time.to_pydatetime(), None]
        for g in groups:
            us = rstar if (g == R_STAR and q == last) else (-9.9 if g == "Output Gap" else 3.3)
            row += [us] + ["NA"] * (len(countries) - 1) + [None]
        rows.append(row)
    return rows


def workbook(sheets):
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for name, rows in sheets.items():
        ws = wb.create_sheet(name)
        for row in rows:
            ws.append(row)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def real_time_file(**overrides):
    sheets = {v: vintage_rows(v, layout, r, banner="2019Q4" if v == "2020Q1" else None)
              for v, (layout, r) in VINTAGES.items()}
    return workbook({**sheets, **overrides, "info": [[None, "FEDERAL RESERVE BANK of NEW YORK"]]})


def test_each_vintage_gives_its_last_quarter_under_the_r_star_group_in_either_layout():
    df = nyfed.parse_vintages(real_time_file(), R_STAR, "US")
    assert df["date"].tolist() == [T("2019-10-01"), T("2020-01-01"), T("2022-10-01")]
    assert df["value"].tolist() == [0.49, 0.53, 0.70]


def test_a_sheet_whose_last_row_is_not_its_quarter_or_has_no_number_there_fails():
    moved = vintage_rows("2019Q3", LAYOUT_2017, 0.61)              # a 2019Q3 history under a 2019Q4 name
    with pytest.raises(RuntimeError, match="not 2019Q4"):
        nyfed.parse_vintages(real_time_file(**{"2019Q4": moved}), R_STAR, "US")
    with pytest.raises(RuntimeError, match="is 'NA'"):
        nyfed.parse_vintages(real_time_file(), R_STAR, "Canada")
    with pytest.raises(RuntimeError, match="no 'UK' column"):   # the 2023 model has no UK
        nyfed.parse_vintages(workbook({"2022Q4": vintage_rows("2022Q4", LAYOUT_2023, 0.70)}), R_STAR, "UK")


def test_a_vintage_is_published_65_days_after_its_quarter_or_on_a_later_release_day():
    dates = pd.DatetimeIndex(["2019-10-01", "2022-10-01", "2025-07-01", "2026-04-01"])
    assert list(nyfed.published(dates, 65)) == [T("2020-03-05"),   # 2020 is a leap year: 31 + 29 + 5
                                                T("2023-05-19"),   # the estimates resumed with 2022Q4
                                                T("2025-12-29"),   # the 2025 shutdown delayed GDP
                                                T("2026-09-03")]
    assert nyfed.published(dates[1:2], 200)[0] == T("2023-07-19")  # a later rule wins over a release day


def test_the_configured_lag_is_never_before_a_release_on_the_hlw_schedules():
    """HLW's own schedule, not LW's above it on the page: LW follows US GDP and runs up to four days earlier."""
    lags = [lag for c in config.enabled() for s, lag in
            config.currency(c)["sources"].get("estimates", {}).get("nyfed", {}).items() if s in nyfed.SERIES]
    quarters = pd.PeriodIndex([pd.Period(q, freq="Q") for q in HLW_SCHEDULE])
    released = pd.DatetimeIndex([T(d) for d in HLW_SCHEDULE.values()])
    assert lags and max((released - quarters.end_time.normalize()).days) == 65     # 2019Q2, after Labor Day
    for lag in [*lags, nyfed.Hlw().lag_days]:
        assert (nyfed.published(quarters.start_time, lag) >= released).all(), lag


def test_the_source_serves_the_vintages_inside_the_range_asked_for(monkeypatch):
    class Response:
        content = real_time_file()
    monkeypatch.setattr(nyfed, "_get", lambda url: Response)
    monkeypatch.setitem(nyfed.SERIES, "TEST_RSTAR", ("test.xlsx", R_STAR, "US"))
    df = nyfed.Hlw().observations("TEST_RSTAR", "2020-01-01", "2026-12-31")
    assert df["date"].tolist() == [T("2020-01-01"), T("2022-10-01")]
    assert df["published"].tolist() == [T("2020-06-04"), T("2023-05-19")]


def test_the_registry_names_each_source_and_refuses_an_unknown_one():
    hlw = registry.lookup(registry.ESTIMATES, "nyfed")(65)
    assert isinstance(hlw, nyfed.Hlw) and hlw.name == "nyfed" and hlw.lag_days == 65
    assert isinstance(registry.lookup(registry.CURVES, "boe")(), boe.BoeCurve)
    assert registry.lookup(registry.DAILY, "boe")(0).lag_bdays == 0
    with pytest.raises(KeyError, match="known: \\['nyfed'\\]"):
        registry.lookup(registry.ESTIMATES, "bis")
