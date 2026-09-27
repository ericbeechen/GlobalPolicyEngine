"""Bank of England data: daily series from the Interactive Statistical Database, and the fitted OIS curves.

Two sources, both under the cache namespace ``boe``:

- `BoeSeries`: one IADB series a day (SONIA ``IUDSOIA``, Bank Rate ``IUDBEDR``)
  over the IADB's CSV export. ``published`` is `lag_bdays` London business
  days after the date: SONIA for day d is published at 9:00 the next business
  day; the Bank Rate in force on d was announced by noon on d at the latest.
- `BoeCurve`: the Bank's daily fitted UK OIS spot curve, short end: continuously
  compounded, at monthly maturities from 1 to 60 months. Published as Excel
  workbooks inside zip archives (one per span of years, plus the current
  month), and aimed to be out by noon the next business day, so ``published``
  = the next London business day. Keyed by (date, tenor in months).

The workbooks change layout over the archive: 2009-15 has no separate short-end
sheet (its "spot curve" is the monthly one), later ones do, and each carries
banner, formula and "Refresh" rows above the data. Sheets are picked by their
maturity row, not their name, and only rows dated in column A are data.
"""

import datetime
import io
import zipfile
import numpy as np
import openpyxl
import pandas as pd
import requests
from policypath.calendars import UK_BDAY
from policypath.sources.base import Source

IADB = "https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp"
CURVES = "https://www.bankofengland.co.uk/-/media/boe/files/statistics/yield-curves/"
# Per curve: the archives holding it, and the prefix its workbooks' names share.
CURVE_FILES = {"OIS_SPOT": (["oisddata.zip", "latest-yield-curve-data.zip"], "OIS daily data")}
HEADERS = {"User-Agent": "Mozilla/5.0 (policypath research)"}


def _get(url, **params):
    r = requests.get(url, params=params, headers=HEADERS, timeout=120)
    r.raise_for_status()
    return r


def published_after(dates, lag_bdays):
    """`lag_bdays` London business days after each date, never before the date itself."""
    days = pd.DatetimeIndex(dates).to_numpy().astype("datetime64[D]")
    lagged = np.busday_offset(days, lag_bdays, roll="backward", busdaycal=UK_BDAY.calendar)
    return pd.to_datetime(np.maximum(lagged, days))


def iadb(series, start, end):
    """One IADB series as date, value; days the database leaves blank are dropped."""
    r = _get(IADB, **{"csv.x": "yes", "Datefrom": pd.Timestamp(start).strftime("%d/%b/%Y"),
                      "Dateto": pd.Timestamp(end).strftime("%d/%b/%Y"), "SeriesCodes": series,
                      "CSVF": "TN", "UsingCodes": "Y", "VPD": "Y", "VFD": "N"})
    df = pd.read_csv(io.StringIO(r.text))
    if list(df.columns) != ["DATE", series]:
        raise RuntimeError(f"IADB {series}: unexpected columns {list(df.columns)}")
    df = df.dropna(subset=[series])
    return pd.DataFrame({"date": pd.to_datetime(df["DATE"], format="%d %b %Y"),
                         "value": df[series].astype(float)}).reset_index(drop=True)


class BoeSeries(Source):
    """A daily IADB series, keyed by date."""

    name = "boe"
    refetch_days = 7

    def __init__(self, lag_bdays=1):
        self.lag_bdays = lag_bdays

    def fetch(self, series, start, end):
        df = iadb(series, start, end)
        return df.assign(published=published_after(df["date"], self.lag_bdays))


def _maturities(ws):
    """The row of maturities in years, and the column each starts in, if this sheet is monthly."""
    for row in ws.iter_rows(min_row=1, max_row=6, values_only=True):
        if row and isinstance(row[0], str) and row[0].strip().lower().startswith("years"):
            years = [v for v in row[1:] if isinstance(v, (int, float))]
            if years and abs(years[0] - 1 / 12) < 1e-6:
                return years
    return None


def parse_curve_workbook(data):
    """The monthly spot sheet of one Bank of England OIS workbook, long: date, tenor, value."""
    wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    frames = []
    for name in wb.sheetnames:
        if "spot" not in name.lower():
            continue
        ws = wb[name]
        years = _maturities(ws)
        if years is None:          # the long-end sheet: half-yearly maturities
            continue
        tenors = [round(y * 12) for y in years]
        rows = [r for r in ws.iter_rows(min_row=1, values_only=True)
                if r and isinstance(r[0], datetime.datetime)]
        wide = pd.DataFrame([r[1:len(tenors) + 1] for r in rows], columns=tenors,
                            index=pd.DatetimeIndex([r[0] for r in rows], name="date"))
        long = wide.apply(pd.to_numeric, errors="coerce").stack().rename("value").reset_index()
        frames.append(long.rename(columns={"level_1": "tenor"}).dropna(subset=["value"]))  # holidays: blank rows
        break
    if not frames:
        raise RuntimeError("no monthly spot sheet in the workbook")
    return frames[0].astype({"tenor": int})


class BoeCurve(Source):
    """The Bank's fitted OIS spot curve, short end, keyed by date and tenor in months."""

    name = "boe"
    keys = ("date", "tenor")
    refetch_days = 7        # the current-month workbook can be refitted after first release

    def __init__(self, lag_bdays=1):
        self.lag_bdays = lag_bdays
        self._parsed = {}

    def _all(self, series):
        """Every workbook in the series' archives, parsed once per instance; later files win a date."""
        if series not in self._parsed:
            archives, prefix = CURVE_FILES[series]
            frames = []
            for archive in archives:
                with zipfile.ZipFile(io.BytesIO(_get(CURVES + archive).content)) as z:
                    for member in sorted(z.namelist()):
                        if member.startswith(prefix) and member.endswith(".xlsx"):
                            frames.append(parse_curve_workbook(z.read(member)))
            both = pd.concat(frames, ignore_index=True)
            self._parsed[series] = both.drop_duplicates(["date", "tenor"], keep="last")
        return self._parsed[series]

    def fetch(self, series, start, end):
        df = self._all(series)
        df = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]
        df = df.sort_values(["date", "tenor"]).reset_index(drop=True)
        return df.assign(published=published_after(df["date"], self.lag_bdays))
