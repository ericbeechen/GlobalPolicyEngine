"""Parquet cache under ``data/cache/``, keyed by (source, series, currency).

Each key is a vintage log. A row is one value of one observation as it stood
from ``published`` on. A value that changes is a new row (a revision), not an
edit of the old one, so `read` can answer "what was knowable at `as_of`" while
the latest vintage still wins for any later date. That is how the cache
overwrites a revised value without forgetting what it replaced: a preliminary
CME settle and its final both stay in the log, and a read after the final was
published sees only the final.

A source that cannot date a revision (it reports the original publication date
for a changed value, as FRED's current-vintage API does) gets the revision
stamped with the day we retrieved it, since that is the earliest we can vouch
for it.

``manifest.json`` records, per key, the reference-date ranges already covered
and the columns that identify an observation. The ranges are what make
`Source.update` incremental and idempotent.

Only `sources/` writes here. Everything downstream reads through `read` or `log`.
"""

import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from policypath.sources.base import VINTAGE_COLUMNS, require_published

CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "cache"
MANIFEST = "manifest.json"
UPDATES = "updates.json"      # {currency: when its last full update finished}
DAY = pd.Timedelta(days=1)


def _path(source, series, currency, root):
    return root / source / currency / f"{series}.parquet"


def _key(source, series, currency):
    return f"{source}/{currency}/{series}"


def _now():
    return pd.Timestamp.now("America/New_York").tz_localize(None)


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------

def _manifest(root):
    path = root / MANIFEST
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _save_manifest(manifest, root):
    root.mkdir(parents=True, exist_ok=True)
    tmp = root / (MANIFEST + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    # On Windows the swap fails while anything (an indexer, antivirus, an editor)
    # has the manifest open for a moment; wait that out rather than abort a long update.
    for attempt in range(10):
        try:
            tmp.replace(root / MANIFEST)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.5)


def merge_ranges(ranges):
    """Sort (lo, hi) date ranges and merge any that overlap or touch."""
    out = []
    for lo, hi in sorted((pd.Timestamp(a), pd.Timestamp(b)) for a, b in ranges):
        if out and lo <= out[-1][1] + DAY:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def fetched(source, series, currency, root=CACHE_DIR):
    """Reference-date ranges already covered for this key, merged and sorted."""
    entry = _manifest(root).get(_key(source, series, currency), {})
    return merge_ranges(entry.get("ranges", []))


def last_covered(source, series, currency, root=CACHE_DIR):
    ranges = fetched(source, series, currency, root)
    return ranges[-1][1] if ranges else None


def mark_updated(currency, root=CACHE_DIR):
    """Record that a full update of `currency`'s sources has just finished (`update_data.py` calls it)."""
    path = root / UPDATES
    stamps = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    stamps[currency] = _now().isoformat(timespec="seconds")
    root.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stamps, indent=2, sort_keys=True), encoding="utf-8")


def last_updated(currency, root=CACHE_DIR):
    """When a full update of `currency` last finished (New York time), or None if never.

    Not a key's own fetch time: a source that re-reads nothing once a day is
    covered (the futures archive) leaves its key's time alone on a day with no
    new settle, while one that re-reads its last week (the Bank's curves) moves
    it on every run.
    """
    path = root / UPDATES
    stamp = json.loads(path.read_text(encoding="utf-8")).get(currency) if path.exists() else None
    return pd.Timestamp(stamp) if stamp else None


def missing(source, series, currency, start, end, root=CACHE_DIR):
    """Sub-ranges of [start, end] not yet covered for this key."""
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    gaps, cursor = [], lo
    for a, b in fetched(source, series, currency, root):
        if b < cursor or a > hi:
            continue
        if a > cursor:
            gaps.append((cursor, a - DAY))
        cursor = max(cursor, b + DAY)
    if cursor <= hi:
        gaps.append((cursor, hi))
    return gaps


def mark_fetched(source, series, currency, start, end, root=CACHE_DIR):
    manifest = _manifest(root)
    entry = manifest.setdefault(_key(source, series, currency), {})
    ranges = merge_ranges([*entry.get("ranges", []), (start, end)])
    entry["ranges"] = [[a.strftime("%Y-%m-%d"), b.strftime("%Y-%m-%d")] for a, b in ranges]
    entry["updated"] = _now().isoformat(timespec="seconds")
    _save_manifest(manifest, root)


def drop(source, series, currency, root=CACHE_DIR):
    """Forget one key: its log and its manifest entry. Returns True if anything was there.

    For a series whose publication rule changed. `append` keys revisions on the
    value, so re-fetching the same values under a new `published` adds nothing;
    the log has to be rebuilt from an empty key.
    """
    path = _path(source, series, currency, root)
    manifest = _manifest(root)
    found = path.exists() or _key(source, series, currency) in manifest
    if path.exists():
        path.unlink()
    if manifest.pop(_key(source, series, currency), None) is not None:
        _save_manifest(manifest, root)
    return found


def _keys(source, series, currency, root):
    entry = _manifest(root).get(_key(source, series, currency), {})
    return entry.get("keys", ["date"])


# --------------------------------------------------------------------------
# vintage log
# --------------------------------------------------------------------------

def append(df, source, series, currency, keys=("date",), root=CACHE_DIR, dated=False):
    """Add new observations and revisions to the log. Returns the number of rows added.

    A row is added when its value differs from the vintage before it for the
    same observation, so re-appending what is already cached adds nothing.

    `dated` says the source stamps every record with when it was really
    published (Databento's receive time), so a changed value that sorts before
    the latest cached vintage is an older vintage arriving late -- a backfill
    next to a range already cached -- not an undated revision, and keeps its date.
    """
    keys = list(keys)
    require_published(df, keys)
    if df.empty:
        return 0
    path = _path(source, series, currency, root)
    old = pd.read_parquet(path) if path.exists() else None
    now = _now()

    new = df.assign(retrieved=now, _new=True)
    both = new if old is None else pd.concat([old.assign(_new=False), new], ignore_index=True)
    # Old rows sort before new rows published at the same moment, so an
    # unchanged re-fetch lands right behind the row it duplicates.
    both = both.sort_values([*keys, "published", "_new"], kind="stable")
    changed = both["value"].ne(both.groupby(keys, sort=False)["value"].shift())
    add = both[both["_new"] & changed].drop(columns="_new")
    if add.empty:
        return 0

    if old is not None and not dated:
        # A changed value claiming to be no newer than what it replaces is an
        # undated revision: it is knowable from when we saw it, not before.
        seen = old.groupby(keys)["published"].max().rename("_seen")
        add = add.join(seen, on=keys)
        undated = add["_seen"].notna() & (add["published"] <= add["_seen"])
        add.loc[undated, "published"] = add.loc[undated, "published"].clip(lower=now.normalize())
        add = add.drop(columns="_seen")

    path.parent.mkdir(parents=True, exist_ok=True)
    out = add if old is None else pd.concat([old, add], ignore_index=True)
    out.to_parquet(path, index=False)

    manifest = _manifest(root)
    manifest.setdefault(_key(source, series, currency), {})["keys"] = keys
    _save_manifest(manifest, root)
    return len(add)


def log(source, series, currency, root=CACHE_DIR):
    """Every vintage recorded for the key, for callers that need point-in-time by row.

    Raises if nothing is cached. Most callers want `read`.
    """
    path = _path(source, series, currency, root)
    if not path.exists():
        raise FileNotFoundError(f"nothing cached for {_key(source, series, currency)} at {path}; "
                                "run scripts/update_data.py")
    return pd.read_parquet(path)


def view(vintages, as_of, keys=("date",)):
    """The latest vintage of each observation published by the end of `as_of`."""
    keys = list(keys)
    cutoff = pd.Timestamp(as_of).normalize() + DAY
    known = vintages[vintages["published"] < cutoff]
    known = known.sort_values(["published", "retrieved"]).drop_duplicates(keys, keep="last")
    return known.drop(columns="retrieved").sort_values(keys).reset_index(drop=True)


class Presorted:
    """A vintage log sorted once, for reads at many dates in a row.

    `view(as_of)` returns exactly what ``view(log, as_of)`` does, and
    `latest(as_of)` the latest vintage of each date published by the end of
    `as_of`, indexed by date (what `model.path.known` returns). The panel reads
    the same few logs on every one of thousands of sessions; sorting each log
    once instead of on every read is most of the difference. Both sorts are
    pandas' multi-key sort, which is stable (``np.lexsort``), so filtering the
    sorted log keeps the order that sorting the filtered log would give, and the
    rows published by a date are a prefix of the log sorted by publication.
    """

    def __init__(self, vintages, keys=("date",)):
        self.keys = list(keys)
        self._by_published = vintages.sort_values(["published", "retrieved"]) if "retrieved" in vintages else None
        if self._by_published is not None:
            self._published = self._by_published["published"].to_numpy()
        self._by_date = vintages.sort_values(["date", "published"])
        self._date_published = self._by_date["published"].to_numpy()

    def view(self, as_of):
        cutoff = np.datetime64(pd.Timestamp(as_of).normalize() + DAY)
        known = self._by_published.iloc[:np.searchsorted(self._published, cutoff, side="left")]
        known = known.drop_duplicates(self.keys, keep="last")
        return known.drop(columns="retrieved").sort_values(self.keys).reset_index(drop=True)

    def latest(self, as_of):
        cutoff = np.datetime64(pd.Timestamp(as_of).normalize() + DAY)
        f = self._by_date[self._date_published < cutoff]
        return f.drop_duplicates("date", keep="last").set_index("date")


def read(source, series, currency, as_of, root=CACHE_DIR):
    """The series as it was knowable at the end of `as_of`: one row per observation."""
    return view(log(source, series, currency, root), as_of, _keys(source, series, currency, root))


def metadata(source, series, currency, root=CACHE_DIR):
    """What the manifest records for one key: covered ranges, or vintage counts and series metadata."""
    entry = _manifest(root).get(_key(source, series, currency))
    if entry is None:
        raise KeyError(f"nothing in the manifest for {_key(source, series, currency)}")
    return entry


# --------------------------------------------------------------------------
# real-time vintages (ALFRED)
# --------------------------------------------------------------------------

# Two pulls' values of one vintage closer than this, relative, are one value printed two ways.
SAME_VALUE = 1e-12


def write_vintages(df, source, series, currency, meta=None, root=CACHE_DIR):
    """Replace one series' vintage file with a fresh full pull. Returns the rows added.

    For sources that keep their own vintage archive (ALFRED), a pull is the
    whole history again. It may add vintages and close an interval that was
    open, but it must not drop, change or re-date a vintage already cached:
    that raises, and the file is left as it was.

    "Change" means by more than `SAME_VALUE` relative. ALFRED re-prints old
    vintages from time to time (NROU's ``5.693902493`` came back in 2026 as
    ``5.6939024929999995``): the same number, one ulp apart once parsed. Such a
    vintage keeps the bits already cached, so outputs frozen on the cache stay
    bit-identical, and only a real revision fails.
    """
    absent = [c for c in VINTAGE_COLUMNS if c not in df.columns]
    if absent:
        raise ValueError(f"vintages are missing columns {absent}")
    df = df[VINTAGE_COLUMNS].sort_values(["date", "realtime_start"]).reset_index(drop=True)
    path = _path(source, series, currency, root)
    added = len(df)
    if path.exists():
        old = pd.read_parquet(path)
        both = old.merge(df, on=["date", "realtime_start"], how="left", suffixes=("", "_new"), indicator=True)
        lost = both["_merge"] == "left_only"
        same = (np.isclose(both["value"], both["value_new"], rtol=SAME_VALUE, atol=0.0)
                | (both["value"].isna() & both["value_new"].isna()))
        reended = both["realtime_end"].notna() & (both["realtime_end"] != both["realtime_end_new"])
        bad = lost | (~lost & (~same | reended))
        if bad.any():
            raise ValueError(f"{_key(source, series, currency)}: the new pull drops or changes "
                             f"{bad.sum()} cached vintages; not written")
        added = len(df) - len(old)
        cached = old.set_index(["date", "realtime_start"])["value"]
        cached = cached.reindex(pd.MultiIndex.from_frame(df[["date", "realtime_start"]])).to_numpy()
        df["value"] = np.where(pd.notna(cached), cached, df["value"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)

    manifest = _manifest(root)
    manifest[_key(source, series, currency)] = {
        **(meta or {}),
        "vintages": int(df["realtime_start"].nunique()),
        "last_vintage": df["realtime_start"].max().strftime("%Y-%m-%d"),
        "updated": _now().isoformat(timespec="seconds"),
    }
    _save_manifest(manifest, root)
    return added


def vintages(source, series, currency, root=CACHE_DIR):
    """Every vintage cached for one series, with its real-time interval. Most callers want a VintagePanel."""
    path = _path(source, series, currency, root)
    if not path.exists():
        raise FileNotFoundError(f"no vintages cached for {_key(source, series, currency)} at {path}; "
                                "run scripts/update_data.py --macro-only")
    return pd.read_parquet(path)
