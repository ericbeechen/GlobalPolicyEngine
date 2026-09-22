"""Short-rate futures (ZQ, SR3) from Databento GLBX.MDP3 batch files on disk.

Expected layout, one file per day as delivered by a Databento batch job:
    databento/definitions/glbx-mdp3-YYYYMMDD.definition.dbn.zst
    databento/statistics/glbx-mdp3-YYYYMMDD.statistics.dbn.zst

Every observation keeps two dates: ``trade_date`` (the session the value
belongs to, Databento's ``ts_ref``) and ``published`` (when it reached us,
``ts_recv``).
"""

from pathlib import Path
import databento as db
import pandas as pd

DATABENTO_DIR = Path(__file__).resolve().parents[3] / "databento"
# CME settlement flags (stat_flags on SETTLEMENT_PRICE records).
FINAL = 1   # bit 0: final rather than preliminary
ACTUAL = 2  # bit 1: actual rather than theoretical


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


def settlements(start=None, end=None, final_only=True, root=DATABENTO_DIR):
    """Daily settlement prices of outright contracts, one row per (trade_date, contract).

    If a settle was published more than once, the last publication is kept.
    ``implied_rate`` is 100 - price, in percent.
    """
    s = read_statistics(start, end, root)
    s = s[s["stat_type"] == db.StatType.SETTLEMENT_PRICE.value]
    if final_only:
        s = s[(s["stat_flags"] & FINAL) != 0]
    s = s.sort_values("published").drop_duplicates(["trade_date", "instrument_id"], keep="last")

    defs = read_definitions(start, end, root)
    s = s.merge(defs[["instrument_id", "asset", "expiration"]], on="instrument_id")  # drops spreads
    s["implied_rate"] = 100.0 - s["price"]
    cols = ["trade_date", "published", "asset", "symbol", "instrument_id", "expiration",
            "price", "implied_rate", "stat_flags"]
    return s[cols].sort_values(["trade_date", "asset", "expiration"]).reset_index(drop=True)
