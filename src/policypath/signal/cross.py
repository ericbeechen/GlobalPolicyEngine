"""The first cross-country signal: one currency's gap minus another's, per meeting horizon.

Each gap is market minus model in its own currency. Their difference is
positive when the first market prices more tightening against its own rule
than the second does against its. Only sessions both currencies have are
used, so neither leg is stale. A session where either gap is missing (a
reference path not yet published) keeps its row with no difference and no z,
as a missing gap does in one currency. The differential is z-scored exactly
as a single gap is (`signal.gap.zscore`). A trade expression (DV01-matched)
is not built here.
"""

import pandas as pd
from policypath.signal.gap import zscore


def differential(first, second, spec):
    """Long frame of session, k, diff_bp and z, from two `signal.gap.build` frames (first minus second)."""
    a = first.pivot(index="session", columns="k", values="gap_bp")
    b = second.pivot(index="session", columns="k", values="gap_bp")
    both = a.index.intersection(b.index)
    diff = (a.loc[both] - b.loc[both]).sort_index()
    z, mean, sd = zscore(diff, spec)
    parts = {"diff_bp": diff, "z": z, "window_mean_bp": mean, "window_sd_bp": sd}
    out = pd.concat({name: f.stack(future_stack=True) for name, f in parts.items()}, axis=1)
    return out.reset_index()
