"""Office for National Statistics data as it was published: revisions triangles, release days.

There is no ALFRED for the UK that is fit for this: FRED's OECD copies of UK
series arrive 22-25 days after the ONS releases them, start in 2011-13 and
stopped updating in 2025. The ONS itself publishes what is needed.

- **Revisions triangles** (UNEM04 unemployment rate, CLA03 claimant count, the
  PAYE RTI triangle): one row per monthly release, holding every estimate as
  that release published it. Each row becomes a vintage, in the same shape as
  ALFRED's (``date``, ``value``, ``realtime_start``, ``realtime_end``), so
  `macro.vintage.VintagePanel` reads them unchanged.
- **CPI** is not revised, so the current series with each month's release day
  is exactly what was known when (checked: the October 2015 vintage equals the
  current one for every month 2005-15).
- **Release days** come from the ONS release calendar, which starts in February
  2016 (CPI) and April 2016 (labour market). A triangle row carries only its
  month. Before the calendar starts, a release is dated the 26th of its month,
  the latest any has come out since 2016, rolled to the next business day:
  late by up to about ten days, never early. `LAG_APPROXIMATED_BEFORE` records
  where that applies, for the data-quality tags in config.

Triangle conventions: UNEM04 labels a three-month window by its end month (the
ONS time-series pages label it by the middle month: mixing the two is a
one-month look-ahead). The LFS was suspended in October 2023-January 2024 and
the official series (MGSX) not updated; the triangle's rows for those months
hold later estimates, not what was released, so they are dropped and the
September 2023 vintage stays current until February 2024.
"""

import io
import re
import numpy as np
import openpyxl
import pandas as pd
import requests
from policypath.calendars import UK_BDAY
from policypath.sources.base import VINTAGE_COLUMNS

ONS = "https://www.ons.gov.uk"
RELEASES = "https://api.beta.ons.gov.uk/v1/search/releases"
HEADERS = {"User-Agent": "Mozilla/5.0 (policypath research)"}
LATEST_RELEASE_DAY = 26

# Which release each series comes out in, matched on the calendar's titles.
RELEASE_TITLES = {
    "lms": r"^(?:UK labour market|UK Labour Market|Labour market overview|Labour market statistics time series)",
    "cpi": r"^(?:Consumer price inflation, UK|UK consumer price inflation)",
}
# Each series by its name in the cache: where it lives, and how to read it.
SERIES = {
    "MGSX": {"kind": "triangle", "release": "lms", "sheet": "triangle",
             "page": "/employmentandlabourmarket/peoplenotinwork/unemployment/datasets/"
                     "unemploymentraterevisionstriangleunem04",
             "drop_releases": ["2023-10", "2023-11", "2023-12", "2024-01"]},
    "CLAIMANTS": {"kind": "triangle", "release": "lms", "sheet": "triangle",
                  "page": "/employmentandlabourmarket/peoplenotinwork/outofworkbenefits/datasets/"
                          "claimantcountrevisionstrianglecla02"},
    "PAYE": {"kind": "rti", "release": "lms", "sheet": "Payrolled employees",
             "page": "/employmentandlabourmarket/peopleinwork/earningsandworkinghours/datasets/"
                     "earningsandemploymentfrompayasyouearnrealtimeinformationrevisiontriangle"},
    "D7G7": {"kind": "unrevised", "release": "cpi",
             "uri": "/economy/inflationandpriceindices/timeseries/d7g7/mm23"},
}


def _get(url, **params):
    r = requests.get(url, params=params, headers=HEADERS, timeout=120)
    r.raise_for_status()
    return r


def release_days(kind):
    """The first release day in each month for one release (``lms``, ``cpi``), by month, from the ONS calendar."""
    rows, offset = [], 0
    while True:
        batch = _get(RELEASES, limit=1000, offset=offset).json().get("releases") or []
        if not batch:
            break
        rows += [x["description"] for x in batch]
        offset += len(batch)
    df = pd.DataFrame(rows)
    df = df[df["title"].str.match(RELEASE_TITLES[kind]) & ~df["cancelled"].astype(bool)]
    day = pd.to_datetime(df["release_date"]).dt.tz_convert("Europe/London").dt.tz_localize(None).dt.normalize()
    return day.groupby(day.dt.to_period("M")).min()


def published_in(months, days):
    """The release day for releases in `months`: the calendar's where it has one, else the 26th rolled forward."""
    out = []
    for m in pd.PeriodIndex(months, freq="M"):
        if m in days.index:
            out.append(days[m])
        else:
            out.append(UK_BDAY.rollforward(pd.Timestamp(m.year, m.month, LATEST_RELEASE_DAY)))
    return pd.DatetimeIndex(out)


def lag_approximated_before(days):
    """The first month the calendar dates exactly: releases before it are dated by rule."""
    return days.index.min().start_time


def _current_file(page):
    """The dataset page's current .xlsx: under ``/current/``, or else the first (latest) edition listed."""
    html = _get(ONS + page).text
    links = (re.findall(r'href="(/file\?uri=[^"]*/current/[^"]*\.xlsx)"', html)
             or re.findall(r'href="(/file\?uri=[^"]*\.xlsx)"', html))
    if not links:
        raise RuntimeError(f"no current .xlsx on {page}")
    return _get(ONS + links[0]).content


def parse_triangle(data, sheet):
    """An ONS revisions triangle (UNEM04, CLA03): releases by month down, reference periods across."""
    ws = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)[sheet]
    refs, rows = None, []
    for r in ws.iter_rows(values_only=True):
        label = r[1] if len(r) > 1 else None
        if isinstance(label, str) and label.startswith("Relating to Period"):
            refs = [c for c in r[2:]]
        elif refs is not None and hasattr(label, "year"):
            rows.append((pd.Timestamp(label).to_period("M"), r[2:2 + len(refs)]))
    keep = [i for i, c in enumerate(refs) if hasattr(c, "year")]
    dates = pd.DatetimeIndex([refs[i] for i in keep])
    wide = pd.DataFrame([[v[i] if i < len(v) else None for i in keep] for _, v in rows],
                        index=pd.PeriodIndex([m for m, _ in rows], name="release"), columns=dates)
    return wide.apply(pd.to_numeric, errors="coerce")


def parse_rti(data, sheet):
    """The PAYE RTI triangle: releases by month name down column A, reference months across row 4."""
    ws = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)[sheet]
    refs, rows = None, []
    for r in ws.iter_rows(values_only=True):
        if r and r[0] == "Relating to period":
            refs = pd.to_datetime(pd.Series(r[1:]).dropna().astype(str), format="mixed")
        elif refs is not None and isinstance(r[0], str) and r[0] != "Initial estimate":
            try:
                month = pd.Period(pd.to_datetime(r[0], format="mixed"), "M")
            except (ValueError, TypeError):
                continue
            rows.append((month, r[1:1 + len(refs)]))
    wide = pd.DataFrame([list(v) for _, v in rows], index=pd.PeriodIndex([m for m, _ in rows], name="release"),
                        columns=pd.DatetimeIndex(refs))
    return wide.apply(pd.to_numeric, errors="coerce")


def vintages_from_triangle(wide, days):
    """ALFRED-shaped vintages from a triangle: each release row a vintage, repeated values merged.

    A value's interval runs from the release that first printed it to the day
    before the release that changed it. A blank cell in a later release is not
    a revision: the ONS stops repeating old periods, it does not withdraw them.
    """
    starts = published_in(wide.index, days)
    long = wide.set_axis(starts).stack().rename("value").reset_index()
    long.columns = ["realtime_start", "date", "value"]
    long = long.dropna(subset=["value"]).sort_values(["date", "realtime_start"])
    changed = long["value"].ne(long.groupby("date")["value"].shift())
    long = long[changed].copy()
    nxt = long.groupby("date")["realtime_start"].shift(-1)
    long["realtime_end"] = nxt - pd.Timedelta(days=1)
    return long[VINTAGE_COLUMNS].reset_index(drop=True)


def unrevised(uri, days):
    """A monthly series the ONS never revises, each month known from its release the month after."""
    text = _get(ONS + "/generator", format="csv", uri=uri).text
    df = pd.read_csv(io.StringIO(text), header=None, names=["period", "value"])
    monthly = df[df["period"].str.match(r"^\d{4} [A-Z]{3}$", na=False)]
    date = pd.to_datetime(monthly["period"], format="%Y %b")
    starts = published_in(date.dt.to_period("M") + 1, days)
    return pd.DataFrame({"date": date.to_numpy(), "value": monthly["value"].astype(float).to_numpy(),
                         "realtime_start": starts, "realtime_end": pd.NaT})[VINTAGE_COLUMNS]


def vintages(series, start=None, days=None):
    """Every vintage of one ONS series, ALFRED-shaped, from reference date `start` on.

    `days` are the release days by month (`release_days`); fetched if not given.
    """
    spec = SERIES[series]
    days = release_days(spec["release"]) if days is None else days
    if spec["kind"] == "unrevised":
        out = unrevised(spec["uri"], days)
    else:
        parse = parse_triangle if spec["kind"] == "triangle" else parse_rti
        wide = parse(_current_file(spec["page"]), spec["sheet"])
        drop = pd.PeriodIndex(spec.get("drop_releases", []), freq="M")
        out = vintages_from_triangle(wide[~wide.index.isin(drop)], days)
    if start is not None:
        out = out[out["date"] >= pd.Timestamp(start)]
    return out.astype({c: "datetime64[ns]" for c in ["date", "realtime_start", "realtime_end"]}
                      ).reset_index(drop=True)
