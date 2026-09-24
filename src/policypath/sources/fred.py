"""FRED and ALFRED observations over the St. Louis Fed REST API.

Needs a free API key in the ``FRED_API_KEY`` environment variable (keep it in
``.env``, which is gitignored, and run with ``uv run --env-file .env ...``).

Every observation keeps two dates, like ``rates.py``: ``date`` (the day the
value belongs to) and ``published`` (when it could first have been known).
``published`` comes from a business-day lag rule; ``first_seen`` fetches
ALFRED's own ``realtime_start`` so the rule can be checked against it.
"""

import os
import numpy as np
import pandas as pd
import requests
from policypath.calendars import US_BDAY
from policypath.sources.base import Source, require_published

BASE = "https://api.stlouisfed.org/fred/"
PAGE = 100_000  
MAX_VINTAGES = 2000


def _key():
    key = os.environ.get("FRED_API_KEY")
    if not key:
        raise RuntimeError("FRED_API_KEY is not set; see the module docstring")
    return key


def _get(endpoint, rows_field, limit, **params):
    """Every row of one query, following `offset` pagination.

    Errors report FRED's own message, never the URL: it would carry the API key.
    """
    params = {"api_key": _key(), "file_type": "json", "limit": limit, **params}
    rows, offset = [], 0
    while True:
        r = requests.get(BASE + endpoint, params={**params, "offset": offset}, timeout=60)
        if not r.ok:
            try:
                msg = r.json()["error_message"]
            except ValueError:
                msg = r.reason
            raise RuntimeError(f"FRED {endpoint} {params.get('series_id')}: {r.status_code} {msg}")
        j = r.json()
        rows += j[rows_field]
        offset += len(j[rows_field])
        if offset >= j["count"] or not j[rows_field]:
            return rows


def _fetch(series_id, start=None, end=None, **extra):
    params = {"series_id": series_id, **extra}
    if start is not None:
        params["observation_start"] = pd.Timestamp(start).strftime("%Y-%m-%d")
    if end is not None:
        params["observation_end"] = pd.Timestamp(end).strftime("%Y-%m-%d")
    return _get("series/observations", "observations", PAGE, **params)


def _frame(rows, cols):
    df = pd.DataFrame(rows, columns=["date", "value", *cols])
    df = df[df["value"] != "."]  # FRED's marker for "no fixing that day"
    df = df.assign(date=pd.to_datetime(df["date"]), value=df["value"].astype(float))
    for c in cols:
        df[c] = pd.to_datetime(df[c])
    return df.reset_index(drop=True)


def observations(series_id, start=None, end=None, lag_bdays=1):
    """Current-vintage observations with ``published`` = ``date`` + `lag_bdays` Fed business days.

    Columns: date, value, published. Days FRED reports as "." are dropped, not NaN.
    """
    df = _frame(_fetch(series_id, start, end), [])
    days = df["date"].to_numpy().astype("datetime64[D]")
    df["published"] = pd.to_datetime(np.busday_offset(days, lag_bdays, roll="backward",
                                                      busdaycal=US_BDAY.calendar))
    return require_published(df)


def first_seen(series_id, start=None, end=None):
    """ALFRED's first-vintage value for each date and the day FRED first showed it.

    Columns: date, value, realtime_start. Later revisions are ignored. FRED caps
    one request at 2000 vintage dates, so the real-time axis is walked in chunks;
    a value already present before a chunk starts shows up in an earlier chunk too,
    so the earliest ``realtime_start`` across chunks is the true first sighting.
    """
    vintages = _get("series/vintagedates", "vintage_dates", 10_000, series_id=series_id)
    frames = []
    for i in range(0, len(vintages), MAX_VINTAGES):
        chunk = vintages[i:i + MAX_VINTAGES]
        rows = _fetch(series_id, start, end, realtime_start=chunk[0], realtime_end=chunk[-1])
        frames.append(_frame(rows, ["realtime_start"]))
    df = pd.concat(frames, ignore_index=True)
    return (df.sort_values("realtime_start").drop_duplicates("date", keep="first")
            .sort_values("date").reset_index(drop=True))


def check_publication_lag(series_id, start=None, end=None, lag_bdays=1):
    """Rows where the lag rule and ALFRED's first-seen date disagree.

    ALFRED's history starts when FRED began keeping vintages of the series, so
    dates before that show a later ``realtime_start`` and are expected to disagree.
    """
    rule = observations(series_id, start, end, lag_bdays)
    seen = first_seen(series_id, start, end)
    both = rule.merge(seen[["date", "realtime_start"]], on="date", how="inner")
    return both[both["published"] != both["realtime_start"]].reset_index(drop=True)


def effr(start=None, end=None):
    """Realized effective fed funds rate, in percent. The NY Fed publishes it the next business day."""
    return observations("EFFR", start, end, lag_bdays=1)


class Fred(Source):
    """Current-vintage FRED series, cached as observations keyed by date.

    EFFR, SOFR and the target-range bounds are all published the next business
    day. A week is fetched again on every update: SOFR can be revised on the
    day it is published, and the refetch costs one small request per series.
    """

    name = "fred"
    refetch_days = 7

    def __init__(self, lag_bdays=1):
        self.lag_bdays = lag_bdays

    def fetch(self, series, start, end):
        return observations(series, start, end, self.lag_bdays)
