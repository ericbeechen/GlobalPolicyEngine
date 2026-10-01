"""Short-rate futures (ZQ, SR1, SR3) from Databento GLBX.MDP3: batch files on disk, and the historical API past them.

The archive holds one folder per batch job, one file per day
(`sources/archive.py` has the layout). Reads can be limited to the jobs
holding one futures root, so a day with SR1 files but no ZQ job is not
mistaken for a day ZQ did not trade.

Every observation keeps two dates: ``trade_date`` (the session the value
belongs to, Databento's ``ts_ref``) and ``published`` (when it reached us,
``ts_recv``).

CME sends a preliminary settle around 16:00 ET and the final that evening,
or on the Sunday for a Friday session. `Settlements` caches every distinct
record as its own vintage, so a point-in-time read on the trade date sees the
preliminary and anything later sees the final.

Past the last day the archive covers, `Settlements` can ask Databento's
historical API for the same dataset, schema and parent symbol the batch jobs
asked for. The records are the same ones with the same ``ts_recv``, so they go
through the same filters into the same vintage log, and a batch job bought
later for those days adds nothing. Each request is priced first
(`metadata.get_cost`, free) and refused if it would take the update past its
budget.
"""

import os
import databento as db
import pandas as pd
from policypath.sources import archive
from policypath.sources.archive import DATABENTO_DIR
from policypath.sources.base import Source

FINAL = 1
ACTUAL = 2
LOCAL_TZ = "America/New_York"
# Definitions are read this far back from a chunk's start, so a final settle
# arriving in the first file of a chunk still finds its (expired) contract.
DEFINITION_LOOKBACK = pd.Timedelta(days=7)
DAY = pd.Timedelta(days=1)

# What every batch job in the archive asked for (config/databento_jobs.json); the API is asked the same.
DATASET = "GLBX.MDP3"
LIVE_KEY = "DATABENTO_API_KEY"
# The most one update may spend on the API, in USD, by Databento's quote for each request. On
# 2026-09-30 a day of ZQ, SR1 and SR3 quoted about $0.01: statistics $0.002, definitions $0.008.
LIVE_BUDGET_USD = 1.0
EMPTY = ["date", "contract", "value", "published"]


class LiveBudgetExceeded(RuntimeError):
    """A request's quote would take this update past its budget. Nothing was requested."""


def _read(schema, start, end, root, series=None):
    paths = archive.files(schema, series, start, end, root)
    frames = [db.DBNStore.from_file(p).to_df().reset_index() for p in paths]
    if not frames:
        raise FileNotFoundError(f"no {schema} files{f' for {series}' if series else ''} "
                                f"in {root} between {start} and {end}")
    return pd.concat(frames, ignore_index=True)


def _definitions(d):
    """Outright futures contracts from raw definition records, one row per instrument_id (latest wins)."""
    d = d[d["instrument_class"] == "F"]
    d = d.sort_values("ts_recv").drop_duplicates("instrument_id", keep="last")
    cols = ["instrument_id", "raw_symbol", "asset", "expiration", "activation",
            "min_price_increment", "unit_of_measure_qty", "currency"]
    return d[cols].sort_values(["asset", "expiration"]).reset_index(drop=True)


def read_definitions(start=None, end=None, root=DATABENTO_DIR, series=None):
    """Outright futures contracts, one row per instrument_id (latest definition wins).

    Calendar spreads (instrument_class 'S', e.g. 'SR3U4-SR3M6') are dropped.
    With `series`, only the jobs holding that root are read.
    """
    return _definitions(_read("definition", start, end, root, series))


def _statistics(s):
    """Raw statistics records with the stat type as a readable name, and the two dates named for what they are."""
    s["stat"] = s["stat_type"].map(lambda t: db.StatType(t).name)
    s = s.rename(columns={"ts_ref": "trade_date", "ts_recv": "published"})
    return s[["published", "trade_date", "instrument_id", "symbol", "stat", "stat_type",
              "price", "quantity", "stat_flags", "update_action"]]


def read_statistics(start=None, end=None, root=DATABENTO_DIR, series=None):
    """All statistics records, with the stat type as a readable name."""
    return _statistics(_read("statistics", start, end, root, series))


def _settlements(s, defs, series=None):
    """The settlement records among statistics `s` that a live outright contract in `defs` could carry."""
    s = s[(s["stat_type"] == db.StatType.SETTLEMENT_PRICE.value) & s["trade_date"].notna()]
    s = s.assign(is_final=(s["stat_flags"] & FINAL) != 0)

    if series is not None:
        defs = defs[defs["asset"] == series]
    s = s.merge(defs[["instrument_id", "asset", "expiration"]], on="instrument_id")  # drops spreads
    # CME keeps sending a settle for a day or two after expiry; it is not a live contract.
    s = s[s["trade_date"].dt.normalize() <= s["expiration"].dt.normalize()]
    # Nor is a record received before its session opened a settle of that session. The
    # archive's one case: ZQX4, first traded 2011-12-01, carries a FINAL-flagged 99.135 received
    # the evening before -- ZQV4's 11-30 settle as a reference price -- then settled at 98.99.
    received = s["published"].dt.tz_convert(LOCAL_TZ).dt.tz_localize(None).dt.normalize()
    s = s[received >= s["trade_date"].dt.tz_localize(None).dt.normalize()]
    # The archive holds a single settle on Good Friday 2022, ZQJ7 at 0.0 -- not a price.
    return s[s["price"] > 0]


def settlement_records(start=None, end=None, root=DATABENTO_DIR, definitions_from=None, series=None):
    """Every settlement record for an outright contract, preliminary and final.

    Only records a live contract could carry survive: calendar spreads are
    dropped by the join to definitions, and so are settles stamped after the
    contract's expiry or received before its session, and non-positive prices.
    With `series`, only that root.
    """
    s = read_statistics(start, end, root, series)
    defs = read_definitions(start if definitions_from is None else definitions_from, end, root, series)
    return _settlements(s, defs, series)


def settlements(start=None, end=None, prefer_final=True, root=DATABENTO_DIR, series=None):
    """Daily settlement prices of outright contracts, one row per (trade_date, contract).
    ``implied_rate`` is 100 - price, in percent.
    """
    s = settlement_records(start, end, root, series=series)
    order = ["is_final", "published"] if prefer_final else ["published"]
    s = s.sort_values(order).drop_duplicates(["trade_date", "instrument_id"], keep="last")
    s = s.assign(implied_rate=100.0 - s["price"])
    cols = ["trade_date", "published", "asset", "symbol", "instrument_id", "expiration",
            "price", "implied_rate", "stat_flags", "is_final"]
    return s[cols].sort_values(["trade_date", "asset", "expiration"]).reset_index(drop=True)


def _observations(s):
    """Settlement records as the cache keeps them: one row per record, keyed by trade date and contract."""
    return pd.DataFrame({
        "date": s["trade_date"].dt.tz_localize(None).dt.normalize(),
        "contract": s["symbol"],
        "expiration": s["expiration"].dt.tz_convert(LOCAL_TZ).dt.tz_localize(None).dt.normalize(),
        "value": s["price"].astype(float),
        "is_final": s["is_final"],
        "published": s["published"].dt.tz_convert(LOCAL_TZ).dt.tz_localize(None),
    }).reset_index(drop=True)


def live_client():
    """A Databento historical client if ``DATABENTO_API_KEY`` is set, else None (the archive only)."""
    return db.Historical() if os.environ.get(LIVE_KEY) else None


def _utc(day):
    return pd.Timestamp(day).tz_localize("UTC")


class Settlements(Source):
    """Settlement vintages for one futures root (the `series`), from the archive and, with a `client`, the API past it.

    Keyed by (date, contract): ``date`` is the trade date, ``contract`` the CME
    symbol, ``value`` the settle price, ``published`` the local (New York)
    time the record reached Databento.

    Days are UTC days of receipt, as the batch files are split, so a fetched
    day holds what reached Databento that day whatever session it settles.
    A batch job is complete for every day it covers, so a fetched range
    inside `archive_ranges` is covered to its end. A day past the archive is
    covered once Databento has all of it and calls it ``available``. Until
    then it is fetched again on every update, and whatever reached Databento
    since (a final after a preliminary, Friday's final on the Sunday) is a new
    vintage.

    `spent` is what this instance's API requests were quoted, in USD; a
    request that would take it past `budget` raises `LiveBudgetExceeded`
    before it is made. What was fetched before it stays cached.
    """

    name = "databento"
    keys = ("date", "contract")
    chunk = "MS"
    dated = True  # every record carries CME's publication, as Databento received it

    def __init__(self, root=DATABENTO_DIR, client=None, budget=LIVE_BUDGET_USD):
        self.root = root
        self.client = client
        self.budget = budget
        self.spent = 0.0
        self._available = None
        self._covered = {}

    def archive_ranges(self, series):
        """The day ranges the archive's statistics jobs cover for `series`: [(first, last), ...]."""
        ranges = archive.coverage("statistics", series, self.root)
        if not ranges:
            raise FileNotFoundError(f"no statistics job for {series} under {self.root}; "
                                    "run scripts/file_databento.py after downloading one")
        return ranges

    def ranges(self, series, end):
        """What an update should cover for `series` through `end`, one `update` call per range.

        The archive's ranges, then, with a client, every day after the last
        of them through `end`. A gap between archive jobs is never filled from
        the API: it stays missing, as it was.
        """
        ranges = self.archive_ranges(series)
        last, end = ranges[-1][1], pd.Timestamp(end)
        if self.client is not None and end > last:
            ranges = [*ranges, (last + DAY, end)]
        return ranges

    def fetch(self, series, start, end):
        if self.client is not None:
            last = self.archive_ranges(series)[-1][1]
            if start <= last < end:
                raise ValueError(f"{series} {start.date()}..{end.date()} runs past the archive's last day, "
                                 f"{last.date()}: ask for each of `ranges` separately")
            if start > last:
                return self._fetch_live(series, start, end)
        self._covered[(start, end)] = end
        if not archive.files("statistics", series, start, end, self.root):
            return pd.DataFrame(columns=EMPTY)
        s = settlement_records(start, end, self.root, definitions_from=start - DEFINITION_LOOKBACK,
                               series=series)
        return _observations(s)

    def covered_through(self, obs, start, end):
        return self._covered.pop((start, end))

    # ---- the historical API, past the archive --------------------------------

    def available_end(self):
        """The moment up to which Databento has both schemas this reads, in UTC. Asked once per instance."""
        if self._available is None:
            schemas = self.client.metadata.get_dataset_range(DATASET)["schema"]
            self._available = min(pd.Timestamp(schemas[s]["end"]) for s in ("statistics", "definition"))
        return self._available

    def _get(self, schema, symbols, stype_in, start, end):
        """One API request over [start, end) as a frame, refused before it is made if its quote breaks the budget."""
        query = dict(dataset=DATASET, schema=schema, symbols=symbols, stype_in=stype_in, start=start, end=end)
        quote = self.client.metadata.get_cost(**query)
        if self.spent + quote > self.budget:
            raise LiveBudgetExceeded(
                f"{schema} {symbols[0] if len(symbols) == 1 else f'{len(symbols)} instruments'} "
                f"{start:%Y-%m-%d %H:%M}..{end:%Y-%m-%d %H:%M} UTC quotes ${quote:.4f}, which would take this "
                f"update to ${self.spent + quote:.4f}, past its ${self.budget:.2f} budget. What was fetched "
                f"before it is cached: run again to go on, or raise rates.LIVE_BUDGET_USD.")
        frame = self.client.timeseries.get_range(**query).to_df().reset_index()
        self.spent += quote
        return frame

    def _complete_through(self, start, end):
        """The last day in [start, end] with every day from `start` whole at Databento and ``available``, or None."""
        whole = (self.available_end() - DAY).tz_localize(None).normalize()
        stop = min(end, whole)
        if stop < start:
            return None
        rows = self.client.metadata.get_dataset_condition(DATASET, start_date=f"{start:%Y-%m-%d}",
                                                          end_date=f"{stop:%Y-%m-%d}")
        condition = {pd.Timestamp(r["date"]): r["condition"] for r in rows}
        through = None
        for day in pd.date_range(start, stop, freq="D"):
            if condition.get(day) != "available":
                break
            through = day
        return through

    def _fetch_live(self, series, start, end):
        stop = min(_utc(end + DAY), self.available_end())   # the API's end is exclusive
        self._covered[(start, end)] = self._complete_through(start, end)
        if stop <= _utc(start):
            return pd.DataFrame(columns=EMPTY)
        parent = [f"{series}.FUT"]
        stats = self._get("statistics", parent, "parent", _utc(start), stop)
        if stats.empty:
            return pd.DataFrame(columns=EMPTY)
        defs = self._get("definition", parent, "parent", _utc(start), stop)
        # A contract settling in the window with no definition in it expired just before the window
        # (its final arrives after its last day). Only those are looked up further back, by id: the
        # archive reads a whole week of every definition instead, which over the API is most of the cost.
        settled = stats.loc[stats["stat_type"] == db.StatType.SETTLEMENT_PRICE.value, "instrument_id"]
        known = set() if defs.empty else set(defs["instrument_id"])
        unknown = sorted({int(i) for i in settled} - known)
        if unknown:
            older = self._get("definition", unknown, "instrument_id", _utc(start - DEFINITION_LOOKBACK), _utc(start))
            defs = older if defs.empty else pd.concat([older, defs], ignore_index=True)
        if defs.empty:
            return pd.DataFrame(columns=EMPTY)
        return _observations(_settlements(_statistics(stats), _definitions(defs), series))
