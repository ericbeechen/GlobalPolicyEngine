"""Real-time macro data: every vintage of every series, and what was knowable on a date.

A `VintagePanel` holds ALFRED's rows as they come: one per series, reference
date and real-time period, where ``realtime_start`` and ``realtime_end`` bound
the days on which ``value`` was the current estimate (``realtime_end`` is NaT
while it still is). Every method takes an ``as_of``. There is deliberately no
method for the latest values: that is ``as_of(today)``.
"""

import numpy as np
import pandas as pd
from policypath import config
from policypath.sources import cache

COLUMNS = ["series", "date", "value", "realtime_start", "realtime_end"]


class VintagePanel:
    """Long frame of vintages; `as_of` is the only way values come out.

    `projections` names series published ahead of the dates they describe
    (CBO's natural rate runs ten years out); every other series must be
    published on or after its reference date.
    """

    def __init__(self, frame, projections=()):
        missing = [c for c in COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(f"a VintagePanel needs real-time intervals; missing {missing}")
        df = frame[COLUMNS].copy()
        for c in ("date", "realtime_start", "realtime_end"):
            df[c] = pd.to_datetime(df[c])
        if df["realtime_start"].isna().any():
            raise ValueError("every row needs a realtime_start")
        if (df["realtime_end"] < df["realtime_start"]).any():
            raise ValueError("a row cannot stop being current before it starts")
        early = (df["realtime_start"] < df["date"]) & ~df["series"].isin(projections)
        if early.any():
            bad = df.loc[early, "series"].unique().tolist()
            raise ValueError(f"published before the date it describes: {bad}; list projections explicitly")
        df = df.sort_values(["series", "date", "realtime_start"]).reset_index(drop=True)
        # Within one observation, each vintage must be closed before the next opens.
        later = df.duplicated(["series", "date"])
        prev_end = df.groupby(["series", "date"])["realtime_end"].shift()
        overlap = later & (prev_end.isna() | (prev_end >= df["realtime_start"]))
        if overlap.any():
            raise ValueError(f"{overlap.sum()} vintages overlap the one before them")
        self._frame = df
        self._projections = tuple(projections)
        # Each series' rows, in the frame's order: a read of one series masks its rows, not every
        # series' (the nowcast reads a dozen series on each of thousands of days).
        self._by_series = {name: rows for name, rows in df.groupby("series", sort=True)}
        # The same rows as arrays, for `series` and `published`, which run thousands of times.
        self._arrays = {name: (rows["date"].to_numpy(), rows["value"].to_numpy(),
                               rows["realtime_start"].to_numpy(), rows["realtime_end"].to_numpy())
                        for name, rows in self._by_series.items()}
        # The days a series' current rows change: a row starts being current on its realtime_start
        # and stops the day after its realtime_end. Between two of these, `as_of` returns the same rows.
        self._changes = {}
        for name, (_, _, start, end) in self._arrays.items():
            ends = end[~np.isnat(end)] + np.timedelta64(1, "D")
            self._changes[name] = np.unique(np.concatenate([start, ends]))

    def _live(self, name, as_of):
        """The rows of `name` current at the end of `as_of`, as a boolean mask over its arrays."""
        _, _, start, end = self._arrays[name]
        day = np.datetime64(pd.Timestamp(as_of).normalize())
        return (start <= day) & (np.isnat(end) | (end >= day))

    @classmethod
    def from_cache(cls, ccy, root=cache.CACHE_DIR):
        """Every series the currency's macro block lists, from the cache. No network."""
        spec = config.currency(ccy)["macro"]
        frames = [cache.vintages(spec["source"], s, ccy, root).assign(series=s)
                  for s in [*spec["series"], *spec["validation"]]]
        return cls(pd.concat(frames, ignore_index=True), spec["projections"])

    @property
    def names(self):
        return sorted(self._frame["series"].unique())

    def _select(self, series):
        if series is None:
            return self._frame
        names = sorted({series} if isinstance(series, str) else set(series))
        parts = [self._by_series[n] for n in names if n in self._by_series]
        if len(parts) == 1:
            return parts[0]
        return pd.concat(parts) if parts else self._frame.iloc[:0]

    def as_of(self, as_of, series=None):
        """Every observation as it stood at the end of `as_of`: series, date, value, published.

        ``published`` is when the value shown became current. ``realtime_end`` is
        not returned: it says when the value would next be revised, which nobody
        knew on `as_of`.
        """
        day = pd.Timestamp(as_of).normalize()
        df = self._select(series)
        live = (df["realtime_start"] <= day) & (df["realtime_end"].isna() | (df["realtime_end"] >= day))
        out = df.loc[live, ["series", "date", "value", "realtime_start"]]
        return out.rename(columns={"realtime_start": "published"}).reset_index(drop=True)

    def series(self, name, as_of):
        """One series as it stood at the end of `as_of`, indexed by reference date.

        The same as ``as_of(as_of, name).set_index("date")["value"]``, from the arrays.
        """
        if name not in self._arrays:
            return self.as_of(as_of, name).set_index("date")["value"].rename(name)
        dates, values, _, _ = self._arrays[name]
        live = self._live(name, as_of)
        return pd.Series(values[live], index=pd.DatetimeIndex(dates[live], name="date"), name=name)

    def version(self, as_of, series):
        """A key that is equal for two dates exactly when `series` stood the same at the end of both.

        How many days each series' current rows changed on, up to `as_of`. A result
        computed from these series alone, on two dates with the same key, is the
        same result (`nowcast.build` reuses the day before's on such a day).
        """
        day = np.datetime64(pd.Timestamp(as_of).normalize())
        return tuple(int(np.searchsorted(self._changes[n], day, side="right")) if n in self._changes else -1
                     for n in ([series] if isinstance(series, str) else series))

    def published(self, as_of, series):
        """The latest publication among `series` as they stood at the end of `as_of`.

        The same as ``as_of(as_of, series)["published"].max()``, from the arrays.
        """
        latest = []
        for name in ([series] if isinstance(series, str) else series):
            if name in self._arrays:
                started = self._arrays[name][2][self._live(name, as_of)]
                if len(started):
                    latest.append(started.max())
        return pd.Timestamp(max(latest)) if latest else pd.NaT

    def first_release(self, as_of, series=None):
        """The first print of every observation released by the end of `as_of`.

        A true first print only for dates after the series' first ALFRED vintage
        (reports/vintages_USD.md); older dates show the archive's earliest value.
        """
        day = pd.Timestamp(as_of).normalize()
        df = self._select(series)
        df = df[df["realtime_start"] <= day].drop_duplicates(["series", "date"], keep="first")
        return (df[["series", "date", "value", "realtime_start"]]
                .rename(columns={"realtime_start": "published"}).reset_index(drop=True))

    def release_dates(self, as_of, series=None):
        """When each observation released by the end of `as_of` first appeared."""
        return self.first_release(as_of, series)[["series", "date", "published"]]

    def truncate(self, day):
        """The panel as a pull at the end of `day` would have returned it.

        Rows published after `day` are dropped and intervals still open on
        `day` are reopened. For tests: a result at any date up to `day` must be
        the same on this panel as on the full one.
        """
        day = pd.Timestamp(day).normalize()
        df = self._frame[self._frame["realtime_start"] <= day].copy()
        df.loc[df["realtime_end"] > day, "realtime_end"] = pd.NaT
        return VintagePanel(df, self._projections)
