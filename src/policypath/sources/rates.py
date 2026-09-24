"""Short-rate futures (ZQ, SR3) from Databento GLBX.MDP3 batch files on disk.

Expected layout, one file per day as delivered by a Databento batch job:
    databento/definitions/glbx-mdp3-YYYYMMDD.definition.dbn.zst
    databento/statistics/glbx-mdp3-YYYYMMDD.statistics.dbn.zst

Every observation keeps two dates: ``trade_date`` (the session the value
belongs to, Databento's ``ts_ref``) and ``published`` (when it reached us,
``ts_recv``).

CME sends a preliminary settle around 16:00 ET and the final that evening,
or on the Sunday for a Friday session. `Settlements` caches every distinct
record as its own vintage, so a point-in-time read on the trade date sees the
preliminary and anything later sees the final.
"""

from pathlib import Path
import databento as db
import pandas as pd
from policypath.sources.base import Source

DATABENTO_DIR = Path(__file__).resolve().parents[3] / "databento"
FINAL = 1
ACTUAL = 2
LOCAL_TZ = "America/New_York"
# Definitions are read this far back from a chunk's start, so a final settle
# arriving in the first file of a chunk still finds its (expired) contract.
DEFINITION_LOOKBACK = pd.Timedelta(days=7)


def _files(schema, start=None, end=None, root=DATABENTO_DIR):
    """Daily files for one schema, filtered to [start, end] by the date in the name."""
    start = pd.Timestamp(start) if start is not None else pd.Timestamp.min
    end = pd.Timestamp(end) if end is not None else pd.Timestamp.max
    folder = "definitions" if schema == "definition" else schema
    out = []
    for path in sorted((root / folder).glob(f"glbx-mdp3-*.{schema}.dbn.zst")):
        day = pd.Timestamp(path.name.split("-")[2].split(".")[0])
        if start <= day <= end:
            out.append(path)
    return out


def _read(schema, start, end, root):
    frames = [db.DBNStore.from_file(p).to_df().reset_index() for p in _files(schema, start, end, root)]
    if not frames:
        raise FileNotFoundError(f"no {schema} files in {root} between {start} and {end}")
    return pd.concat(frames, ignore_index=True)


def read_definitions(start=None, end=None, root=DATABENTO_DIR):
    """Outright futures contracts, one row per instrument_id (latest definition wins).

    Calendar spreads (instrument_class 'S', e.g. 'SR3U4-SR3M6') are dropped.
    """
    d = _read("definition", start, end, root)
    d = d[d["instrument_class"] == "F"]
    d = d.sort_values("ts_recv").drop_duplicates("instrument_id", keep="last")
    cols = ["instrument_id", "raw_symbol", "asset", "expiration", "activation",
            "min_price_increment", "unit_of_measure_qty", "currency"]
    return d[cols].sort_values(["asset", "expiration"]).reset_index(drop=True)


def read_statistics(start=None, end=None, root=DATABENTO_DIR):
    """All statistics records, with the stat type as a readable name."""
    s = _read("statistics", start, end, root)
    s["stat"] = s["stat_type"].map(lambda t: db.StatType(t).name)
    s = s.rename(columns={"ts_ref": "trade_date", "ts_recv": "published"})
    return s[["published", "trade_date", "instrument_id", "symbol", "stat", "stat_type",
              "price", "quantity", "stat_flags", "update_action"]]


def settlement_records(start=None, end=None, root=DATABENTO_DIR, definitions_from=None):
    """Every settlement record for an outright contract, preliminary and final.

    Only records a live contract could carry survive: calendar spreads are
    dropped by the join to definitions, and so are settles stamped after the
    contract's expiry and non-positive prices.
    """
    s = read_statistics(start, end, root)
    s = s[(s["stat_type"] == db.StatType.SETTLEMENT_PRICE.value) & s["trade_date"].notna()]
    s = s.assign(is_final=(s["stat_flags"] & FINAL) != 0)

    defs = read_definitions(start if definitions_from is None else definitions_from, end, root)
    s = s.merge(defs[["instrument_id", "asset", "expiration"]], on="instrument_id")  # drops spreads
    # CME keeps sending a settle for a day or two after expiry; it is not a live contract.
    s = s[s["trade_date"].dt.normalize() <= s["expiration"].dt.normalize()]
    # The archive holds a single settle on Good Friday 2022, ZQJ7 at 0.0 -- not a price.
    return s[s["price"] > 0]


def settlements(start=None, end=None, prefer_final=True, root=DATABENTO_DIR):
    """Daily settlement prices of outright contracts, one row per (trade_date, contract).
    ``implied_rate`` is 100 - price, in percent.
    """
    s = settlement_records(start, end, root)
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
    time the record reached Databento. The archive is complete for every day
    it has a file for, so a fetched range is covered to its end.
    """

    name = "databento"
    keys = ("date", "contract")
    chunk = "MS"

    def __init__(self, root=DATABENTO_DIR):
        self.root = root

    def archive_days(self):
        days = [pd.Timestamp(p.name.split("-")[2].split(".")[0]) for p in _files("statistics", root=self.root)]
        if not days:
            raise FileNotFoundError(f"no statistics files under {self.root}")
        return min(days), max(days)

    def fetch(self, series, start, end):
        if not _files("statistics", start, end, self.root):
            return pd.DataFrame(columns=["date", "contract", "value", "published"])
        s = settlement_records(start, end, self.root, definitions_from=start - DEFINITION_LOOKBACK)
        s = s[s["asset"] == series]
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
