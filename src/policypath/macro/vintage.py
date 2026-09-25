"""Real-time macro data: every vintage of every series, and what was knowable on a date.

A `VintagePanel` holds ALFRED's rows as they come: one per series, reference
date and real-time period, where ``realtime_start`` and ``realtime_end`` bound
the days on which ``value`` was the current estimate (``realtime_end`` is NaT
while it still is). Every method takes an ``as_of``. There is deliberately no
method for the latest values: that is ``as_of(today)``.
"""

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

    @classmethod
    def from_cache(cls, ccy):
        """Every series the currency's macro block lists, from the cache. No network."""
        spec = config.currency(ccy)["macro"]
        frames = [cache.vintages(spec["source"], s, ccy).assign(series=s)
                  for s in [*spec["series"], *spec["validation"]]]
        return cls(pd.concat(frames, ignore_index=True), spec["projections"])

    @property
    def names(self):
        return sorted(self._frame["series"].unique())

    def _select(self, series):
        df = self._frame
        if series is None:
            return df
        return df[df["series"].isin([series] if isinstance(series, str) else series)]

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
        """One series as it stood at the end of `as_of`, indexed by reference date."""
        return self.as_of(as_of, name).set_index("date")["value"].rename(name)

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
