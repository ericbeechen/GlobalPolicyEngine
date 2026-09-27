"""Market path minus model path, per meeting horizon, in bp and as a trailing z.

The z is what gets traded, the bp are what gets talked about, so both are kept.
Each horizon k is standardized against its own trailing distribution: the last
two calendar years of that k's gap, not counting today. Two years because the
market's pricing errors are regime-dependent, and a longer window mixes the ELB
years with hiking and cutting cycles whose gaps are not comparable. Today is
left out of its own window so a z uses only what was known before it.

At the lower bound the gap barely moves, and a 1-2bp sd would turn a tick of
noise into a large z. So the sd is floored at ``sd_floor_bp``. See
notes/decisions.md (2026-09-27).
"""

import pandas as pd


def gaps(paths):
    """Session x k frame of market - model in bp, from the long frame `model.path.build` returns."""
    wide = paths.assign(gap_bp=(paths["market"] - paths["model"]) * 100.0)
    return wide.pivot(index="session", columns="k", values="gap_bp").sort_index()


def zscore(gap, spec):
    """Each column's z against its trailing ``window`` of earlier sessions.

    Returns (z, mean, sd), each shaped like `gap`. NaN until ``min_periods``
    earlier gaps exist.
    """
    roll = gap.rolling(spec["window"], closed="left", min_periods=spec["min_periods"])
    mean, sd = roll.mean(), roll.std().clip(lower=spec["sd_floor_bp"])
    return (gap - mean) / sd, mean, sd


def build(paths, spec):
    """Long frame, one row per session and k: gap_bp, z, and the window's mean and sd."""
    gap = gaps(paths)
    z, mean, sd = zscore(gap, spec)
    parts = {"gap_bp": gap, "z": z, "window_mean_bp": mean, "window_sd_bp": sd}
    out = pd.concat({name: f.stack(future_stack=True) for name, f in parts.items()}, axis=1)
    return out.reset_index()
