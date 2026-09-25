"""Short-rate futures (ZQ, SR1, SR3) from Databento GLBX.MDP3 batch files on disk.

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
"""

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


def _read(schema, start, end, root, series=None):
    paths = archive.files(schema, series, start, end, root)
    frames = [db.DBNStore.from_file(p).to_df().reset_index() for p in paths]
    if not frames:
        raise FileNotFoundError(f"no {schema} files{f' for {series}' if series else ''} "
                                f"in {root} between {start} and {end}")
    return pd.concat(frames, ignore_index=True)


def read_definitions(start=None, end=None, root=DATABENTO_DIR, series=None):
    """Outright futures contracts, one row per instrument_id (latest definition wins).

    Calendar spreads (instrument_class 'S', e.g. 'SR3U4-SR3M6') are dropped.
    With `series`, only the jobs holding that root are read.
    """
    d = _read("definition", start, end, root, series)
    d = d[d["instrument_class"] == "F"]
    d = d.sort_values("ts_recv").drop_duplicates("instrument_id", keep="last")
    cols = ["instrument_id", "raw_symbol", "asset", "expiration", "activation",
            "min_price_increment", "unit_of_measure_qty", "currency"]
    return d[cols].sort_values(["asset", "expiration"]).reset_index(drop=True)


def read_statistics(start=None, end=None, root=DATABENTO_DIR, series=None):
    """All statistics records, with the stat type as a readable name."""
    s = _read("statistics", start, end, root, series)
    s["stat"] = s["stat_type"].map(lambda t: db.StatType(t).name)
    s = s.rename(columns={"ts_ref": "trade_date", "ts_recv": "published"})
    return s[["published", "trade_date", "instrument_id", "symbol", "stat", "stat_type",
              "price", "quantity", "stat_flags", "update_action"]]


def settlement_records(start=None, end=None, root=DATABENTO_DIR, definitions_from=None, series=None):
    """Every settlement record for an outright contract, preliminary and final.

    Only records a live contract could carry survive: calendar spreads are
    dropped by the join to definitions, and so are settles stamped after the
    contract's expiry or received before its session, and non-positive prices.
    With `series`, only that root.
    """
    s = read_statistics(start, end, root, series)
    s = s[(s["stat_type"] == db.StatType.SETTLEMENT_PRICE.value) & s["trade_date"].notna()]
    s = s.assign(is_final=(s["stat_flags"] & FINAL) != 0)

    defs = read_definitions(start if definitions_from is None else definitions_from, end, root, series)
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


class Settlements(Source):
    """Settlement vintages for one futures root (the `series`), from the archive on disk.

    Keyed by (date, contract): ``date`` is the trade date, ``contract`` the CME
    symbol, ``value`` the settle price, ``published`` the local (New York)
    time the record reached Databento. A batch job is complete for every day
    it covers, so a fetched range inside `archive_ranges` is covered to its end.
    """

    name = "databento"
    keys = ("date", "contract")
    chunk = "MS"
    dated = True  # every record carries CME's publication, as Databento received it

    def __init__(self, root=DATABENTO_DIR):
        self.root = root

    def archive_ranges(self, series):
        """The day ranges the archive's statistics jobs cover for `series`: [(first, last), ...]."""
        ranges = archive.coverage("statistics", series, self.root)
        if not ranges:
            raise FileNotFoundError(f"no statistics job for {series} under {self.root}; "
                                    "run scripts/file_databento.py after downloading one")
        return ranges

    def fetch(self, series, start, end):
        if not archive.files("statistics", series, start, end, self.root):
            return pd.DataFrame(columns=["date", "contract", "value", "published"])
        s = settlement_records(start, end, self.root, definitions_from=start - DEFINITION_LOOKBACK,
                               series=series)
        return pd.DataFrame({
            "date": s["trade_date"].dt.tz_localize(None).dt.normalize(),
            "contract": s["symbol"],
            "expiration": s["expiration"].dt.tz_convert(LOCAL_TZ).dt.tz_localize(None).dt.normalize(),
            "value": s["price"].astype(float),
            "is_final": s["is_final"],
            "published": s["published"].dt.tz_convert(LOCAL_TZ).dt.tz_localize(None),
        }).reset_index(drop=True)

    def covered_through(self, obs, start, end):
        return end
