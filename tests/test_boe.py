"""The Bank of England curve workbooks: which spot sheet is read, and how the archives are fetched.

Every workbook here is synthetic, built with openpyxl in the layout the Bank
uses: banner rows, a "years:" maturity row, a blank row, then one row per day
dated in column A, with blank holiday rows. The monthly case pins the parse the
OIS path has always had; the half-yearly case is the gilt curve, whose 0.5y
point is often blank. Nothing touches the network: `boe._get` is replaced.
"""

import datetime
import io
import zipfile
import openpyxl
import pandas as pd
import pytest
from policypath.sources import boe

T = pd.Timestamp
MONTHLY = [m / 12 for m in range(1, 5)]
HALF_YEARLY = [0.5, 1, 1.5, 2]


def spot_sheet(title, years, days):
    """A spot sheet as the Bank lays it out; `days` maps a date to its rates (None for a blank cell)."""
    rows = [[None, title], ["Maturity"], ["months:", *[round(y * 12) for y in years]], ["years:", *years], [None]]
    rows += [[datetime.datetime.fromisoformat(d), *rates] for d, rates in days.items()]
    return rows


def workbook(sheets):
    """An xlsx in memory, {sheet name: rows}."""
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for name, rows in sheets.items():
        ws = wb.create_sheet(name)
        for row in rows:
            ws.append(row)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def forwards(days):
    """The same days on a forward sheet: every rate 9.9, so a parse that read one would show it."""
    return {d: [None if r is None else 9.9 for r in rates] for d, rates in days.items()}


def later_layout(short_days, long_days):
    """The 2016-on layout: forward and spot sheets at both ends, short end first, forwards before spots."""
    return workbook({
        "info": [["Bank of England"]],
        "1. fwds, short end": spot_sheet("forward, short end", MONTHLY, forwards(short_days)),
        "2. fwd curve": spot_sheet("forward curve", HALF_YEARLY, forwards(long_days)),
        "3. spot, short end": spot_sheet("spot, short end", MONTHLY, short_days),
        "4. spot curve": spot_sheet("spot curve", HALF_YEARLY, long_days),
    })


SHORT = {"2016-01-01": [None] * 4,                                  # New Year's Day: a blank row
         "2016-01-04": [0.45, 0.46, 0.47, 0.48],
         "2016-01-05": [0.44, "NA", 0.46, 0.47]}                    # a cell the fit left as text
LONG = {"2016-01-01": [None] * 4,
        "2016-01-04": [0.55, 0.60, 0.65, 0.70],
        "2016-01-05": [None, 0.61, 0.66, 0.71]}                     # no 0.5y rate that day


def rows(df):
    return [(d.strftime("%Y-%m-%d"), t, v) for d, t, v in zip(df["date"], df["tenor"], df["value"])]


def test_the_monthly_sheet_parses_as_it_always_has():
    """Long, date then tenor in months; blank and text cells dropped, never NaN; forwards and the long end ignored."""
    df = boe.parse_curve_workbook(later_layout(SHORT, LONG))
    assert list(df.columns) == ["date", "tenor", "value"]
    assert df["tenor"].dtype == "int64" and df["value"].dtype == "float64"
    assert rows(df) == [("2016-01-04", 1, 0.45), ("2016-01-04", 2, 0.46), ("2016-01-04", 3, 0.47),
                        ("2016-01-04", 4, 0.48), ("2016-01-05", 1, 0.44), ("2016-01-05", 3, 0.46),
                        ("2016-01-05", 4, 0.47)]


def test_the_early_layout_has_one_spot_sheet_and_it_is_monthly():
    data = workbook({"info": [["Bank of England"]], "1. fwd curve": spot_sheet("forward", MONTHLY, forwards(SHORT)),
                     "2. spot curve": spot_sheet("spot", MONTHLY, SHORT)})
    assert rows(boe.parse_curve_workbook(data)) == rows(boe.parse_curve_workbook(later_layout(SHORT, LONG)))
    with pytest.raises(RuntimeError, match="no half-yearly spot sheet"):
        boe.parse_curve_workbook(data, "half-yearly")


def test_the_half_yearly_sheet_is_the_long_end_and_a_missing_half_year_is_just_absent():
    df = boe.parse_curve_workbook(later_layout(SHORT, LONG), "half-yearly")
    assert rows(df) == [("2016-01-04", 6, 0.55), ("2016-01-04", 12, 0.60), ("2016-01-04", 18, 0.65),
                        ("2016-01-04", 24, 0.70), ("2016-01-05", 12, 0.61), ("2016-01-05", 18, 0.66),
                        ("2016-01-05", 24, 0.71)]


def test_a_workbooks_last_year_comes_from_its_name():
    assert boe.last_year("GLC Nominal daily data_2005 to 2015.xlsx") == 2015
    assert boe.last_year("GLC Nominal daily data_2025 to present.xlsx") is None
    assert boe.last_year("OIS daily data current month.xlsx") is None


def archive(members):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)
    return buf.getvalue()


@pytest.fixture
def archives(monkeypatch):
    """The three archives the two curves are read from, served from memory, with every download counted.

    Workbooks that must never be opened are not workbooks: parsing one would raise.
    """
    current = {"2016-01-05": [0.43, 0.44, 0.45, 0.46]}              # refitted since the yearly file
    files = {
        "oisddata.zip": archive({"OIS daily data_2009 to 2015.xlsx": b"not a workbook",
                                 "OIS daily data_2016 to 2024.xlsx": later_layout(SHORT, LONG)}),
        "glcnominalddata.zip": archive({"GLC Nominal daily data_1979 to 1984.xlsx": b"not a workbook",
                                        "GLC Nominal daily data_2016 to 2024.xlsx": later_layout(SHORT, LONG)}),
        boe.LATEST: archive({"OIS daily data current month.xlsx": later_layout(current, {}),
                             "GLC Nominal daily data current month.xlsx": later_layout({}, {"2016-01-06": [0.5] * 4}),
                             "GLC Inflation daily data current month.xlsx": b"not a workbook"}),
    }
    calls = []

    class Response:
        def __init__(self, content):
            self.content = content

    def get(url, **params):
        calls.append(url.removeprefix(boe.CURVES))
        return Response(files[url.removeprefix(boe.CURVES)])
    monkeypatch.setattr(boe, "_get", get)
    return calls


def test_one_curve_source_downloads_each_archive_once_and_later_files_win_a_date(archives):
    source = boe.BoeCurve()
    ois = source.fetch("OIS_SPOT", "2016-01-01", "2026-12-31")
    glc = source.fetch("GLC_SPOT", "2016-01-01", "2026-12-31")
    source.fetch("GLC_SPOT", "2016-01-05", "2016-01-06")           # a refetch window: nothing downloaded again
    assert sorted(archives) == sorted(["oisddata.zip", "glcnominalddata.zip", boe.LATEST])
    assert ois.loc[(ois["date"] == T("2016-01-05")) & (ois["tenor"] == 1), "value"].item() == 0.43
    assert glc["tenor"].min() == 6 and T("2016-01-06") in set(glc["date"])
    published = ois.groupby("date")["published"].first()
    assert list(published.items()) == [(T("2016-01-04"), T("2016-01-05")), (T("2016-01-05"), T("2016-01-06"))]


def test_a_workbook_that_ends_before_the_start_is_never_opened(archives):
    """The 1979-84 file here is not a workbook: asking for a date it holds is what opens it."""
    boe.BoeCurve().fetch("GLC_SPOT", "2016-01-01", "2016-12-31")
    with pytest.raises(zipfile.BadZipFile):
        boe.BoeCurve().fetch("GLC_SPOT", "1984-06-01", "2016-12-31")
