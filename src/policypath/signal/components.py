"""The three things a sleeve can trade on: the level, the slope and the cross-country differential.

Each component is one number a session, in bp, with the z it is traded on,
the trailing mean and sd the z came from, and the de-meaned gap
``dev_bp = value_bp - window_mean_bp`` (= z x sd). The de-meaned gap is what the
signal expects to close: the move back to its trailing mean, where a
hysteresis exit at z = 0 sits. The raw gap carries the constant offsets (r*,
u*, the operating spread) that the z removes, and its sign disagrees with the
z's on about a third of the USD sessions with |z| >= 1, so it is only ever the
quoted number (`strategy/carry.py` measures the edge on ``dev_bp``).

- **level**: the gap at the backtest horizon (the fourth meeting), as the
  signal panel has it.
- **slope**: the gap at the further meeting of ``signal.slope`` less the gap at
  the nearer (8th minus 1st). On the current panels its z correlates 0.85 (USD)
  and 0.94 (GBP) with the level's, so a curve sleeve on it would mostly re-trade
  the level. The traded slope is orthogonalised: the slope gap less the part the
  level gap explains over the trailing window, ``slope_bp - beta_t x level_bp``,
  with beta_t the OLS slope of one on the other over the signal's trailing
  window strictly before t (two calendar years, 250 sessions first). That
  needs a year of history before its first beta and a year of betas before its
  first z, so the slope's z starts about a year after the level's. The raw
  8-minus-1 gap and its z are kept alongside (a robustness row).
- **differential**: one currency's gap minus another's at the horizon, on the
  sessions both have (`signal.cross.differential`).

In words, for the note: "the slope component is the gap at the eighth meeting
minus the gap at the first, less the part the level gap explains over the
trailing two years."
"""

import pandas as pd
from policypath.signal import cross
from policypath.signal.gap import zscore

COLUMNS = ["value_bp", "z", "window_mean_bp", "window_sd_bp", "dev_bp"]


def _component(value, z, mean, sd):
    out = pd.DataFrame({"value_bp": value, "z": z, "window_mean_bp": mean, "window_sd_bp": sd})
    out["dev_bp"] = out["value_bp"] - out["window_mean_bp"]
    out.index.name = "session"
    return out


def level(signal, k):
    """The gap at meeting horizon `k`, by session, from `signal.gap.build`'s frame: `COLUMNS`."""
    s = signal[signal["k"] == k].set_index("session").sort_index()
    return _component(s["gap_bp"], s["z"], s["window_mean_bp"], s["window_sd_bp"])


def beta(y, x, spec):
    """The OLS slope of `y` on `x` (with an intercept) over the trailing ``spec["window"]`` strictly before each session."""
    def roll(s):
        return s.rolling(spec["window"], closed="left", min_periods=spec["min_periods"])
    return roll(y).cov(x) / roll(x).var()


def slope(signal, pair, level_k, spec, orthogonal=True):
    """The slope gap, gap(k = pair[1]) - gap(k = pair[0]), by session: `COLUMNS` plus raw_bp, raw_z and beta.

    With `orthogonal` the traded value is the slope gap less ``beta`` x the
    level gap at `level_k` (`beta`); without it, the raw slope gap. `spec` is
    the currency's ``signal`` block. ``raw_bp`` and ``raw_z`` are the raw slope
    gap and its z either way.
    """
    gap = signal.pivot(index="session", columns="k", values="gap_bp").sort_index()
    raw, lvl = gap[pair[1]] - gap[pair[0]], gap[level_k]
    b = beta(raw, lvl, spec)
    traded = raw - b * lvl if orthogonal else raw
    out = _component(traded, *zscore(traded, spec))
    return out.assign(raw_bp=raw, raw_z=zscore(raw, spec)[0], beta=b)


def differential(first, second, k_first, k_second, spec):
    """The first currency's gap minus the second's at the horizon, on the sessions both have: `COLUMNS`.

    `k_first` and `k_second` are the two currencies' horizons, which must be
    equal (a differential of different meetings is not a cross-country view).
    """
    if k_first != k_second:
        raise ValueError(f"the pair's horizons differ ({k_first} and {k_second}): the differential needs one")
    d = cross.differential(first, second, spec)
    d = d[d["k"] == k_first].set_index("session").sort_index()
    return _component(d["diff_bp"], d["z"], d["window_mean_bp"], d["window_sd_bp"])
