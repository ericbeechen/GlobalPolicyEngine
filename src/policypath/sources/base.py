"""The contract every source in `sources/` meets.

A source fetches raw data (over the network, or off an archive on disk) and
returns *observations*: one row per reference date and key, with the value and
the moment it became public. That holds for market data as much as for macro
data. CME publishes settlements after the close, so a settle dated Tuesday is
first knowable Tuesday evening and final by Wednesday morning, and
``published`` says so.

`Source.update` is the only way data gets into the cache. It asks the cache
manifest which ranges are still missing, fetches just those, and records what
it covered, so running it twice does no work the second time.
"""

from abc import ABC, abstractmethod
import pandas as pd


# The columns of a real-time vintage: the value, and the days it was the current estimate.
VINTAGE_COLUMNS = ["date", "value", "realtime_start", "realtime_end"]


def require_published(df, keys=("date",)):
    """No series leaves `sources/` indexed by reference date alone.

    `df` needs the `keys` columns, a ``value`` and a ``published`` column with no
    gaps, and nothing can be published before the day it describes.
    """
    missing = [c for c in [*keys, "value", "published"] if c not in df.columns]
    if missing:
        raise ValueError(f"observations are missing columns {missing}")
    if df["published"].isna().any():
        raise ValueError("observations must carry a publication date for every row")
    if (df["published"] < df["date"]).any():
        raise ValueError("an observation cannot be published before its reference date")
    return df


class Source(ABC):
    """One place observations come from.

    Subclasses set `name` (the cache namespace) and `keys` (the columns that
    identify one observation, ``date`` first), and implement `fetch`.
    """

    name: str
    keys: tuple = ("date",)
    # Days before the last covered date to fetch again on every update, so a
    # value revised shortly after publication is picked up as a new vintage.
    refetch_days: int = 0
    # Split long gaps into pieces this long (a pandas frequency), so a first
    # build neither holds the whole history in memory nor loses it all to one
    # failure: each finished piece is cached and marked before the next starts.
    chunk: str | None = None

    @abstractmethod
    def fetch(self, series, start, end):
        """Observations of `series` with reference dates in [start, end]."""

    def covered_through(self, obs, start, end):
        """The last date `fetch(start, end)` can be trusted to have fully covered.

        By default the last date it returned: anything later may simply not be
        published yet, so it stays missing and is asked for again next time.
        """
        return obs["date"].max() if len(obs) else None

    def observations(self, series, start, end):
        obs = self.fetch(series, start, end)
        return require_published(obs, self.keys)

    def update(self, series, currency, start, end, cache, on_chunk=None):
        """Fill the cache for `series` over [start, end]. Returns rows added.

        `on_chunk(lo, hi, added)` is called after each fetched piece, if given.
        """
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        gaps = cache.missing(self.name, series, currency, start, end)
        last = cache.last_covered(self.name, series, currency)
        if self.refetch_days and last is not None and last >= start:
            gaps.append((max(start, last - pd.Timedelta(days=self.refetch_days)), last))
        added = 0
        for lo, hi in _split(cache.merge_ranges(gaps), self.chunk):
            obs = self.observations(series, lo, hi)
            n = cache.append(obs, self.name, series, currency, keys=self.keys)
            through = self.covered_through(obs, lo, hi)
            if through is not None:
                cache.mark_fetched(self.name, series, currency, lo, min(through, hi))
            if on_chunk is not None:
                on_chunk(lo, hi, n)
            added += n
        return added


def _split(ranges, freq):
    """Cut each (lo, hi) range at `freq` boundaries; unchanged if `freq` is None."""
    if freq is None:
        return ranges
    out = []
    for lo, hi in ranges:
        cuts = [lo, *pd.date_range(lo, hi, freq=freq).drop(lo, errors="ignore"), hi + pd.Timedelta(days=1)]
        out += [(a, b - pd.Timedelta(days=1)) for a, b in zip(cuts[:-1], cuts[1:])]
    return out
