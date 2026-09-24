"""Parquet cache under ``data/cache/``, keyed by (source, series, currency).

Each key is one append-only log: rows are added, never rewritten or removed.
A row is (date, value, published, retrieved). Re-fetching an unchanged value
adds nothing. A changed value for a date already in the log is a revision: it
is stored with ``published`` = the day we retrieved it, since that is the
earliest it could have been known here. Reads are point-in-time through `as_of`.

Only `sources/` writes here. Everything downstream reads through `read`.
"""

from pathlib import Path
import pandas as pd
from policypath.sources.base import require_published

CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "cache"
COLS = ["date", "value", "published", "retrieved"]


def _path(source, series, currency, root):
    return root / source / currency / f"{series}.parquet"


def _log(path):
    return pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=COLS)


def append(df, source, series, currency, root=CACHE_DIR):
    """Add new or revised observations to the log. Returns the number of rows added."""
    require_published(df)
    path = _path(source, series, currency, root)
    log = _log(path)
    now = pd.Timestamp.now("UTC").tz_localize(None)

    new = df[["date", "value", "published"]].assign(retrieved=now)
    new = new.merge(log[["date", "value"]].drop_duplicates(), on=["date", "value"],
                    how="left", indicator=True)
    new = new[new["_merge"] == "left_only"].drop(columns="_merge")
    revised = new["date"].isin(log["date"])
    new.loc[revised, "published"] = new.loc[revised, "published"].clip(lower=now.normalize())
    if new.empty:
        return 0

    path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.concat([log, new], ignore_index=True) if len(log) else new
    out[COLS].to_parquet(path, index=False)
    return len(new)


def read(source, series, currency, as_of, root=CACHE_DIR):
    """The series as it was knowable at the end of `as_of`: one value per date, latest vintage.

    Columns: date, value, published. Raises if nothing is cached for the key.
    """
    path = _path(source, series, currency, root)
    if not path.exists():
        raise FileNotFoundError(f"nothing cached for {source}/{currency}/{series} at {path}")
    log = _log(path)
    cutoff = pd.Timestamp(as_of).normalize() + pd.Timedelta(days=1)
    known = log[log["published"] < cutoff]
    known = known.sort_values(["published", "retrieved"]).drop_duplicates("date", keep="last")
    return known[["date", "value", "published"]].sort_values("date").reset_index(drop=True)
