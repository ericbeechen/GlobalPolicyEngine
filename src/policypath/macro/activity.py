"""Real activity over the ragged edge: quarter-to-date growth per series, and a robust composite.

A diagnostic, not a rule input: the rule's activity measure is the unemployment
gap. The z-scores' moments come from the same vintage as the growth they
score, over complete quarters before the target one; see notes/macro.md.
"""

import numpy as np
import pandas as pd
from policypath.macro.inflation import monthly

QUARTER = pd.DateOffset(months=3)
MAD_TO_SD = 1.4826   # a normal's s.d. per unit of median absolute deviation


def quarter_growth(x, quarter):
    """Annualized growth of `quarter`'s average over the months it has so far, against the whole previous quarter."""
    x = monthly(x)
    cur = x[(x.index >= quarter) & (x.index < quarter + QUARTER)].dropna()
    prev = x[(x.index >= quarter - QUARTER) & (x.index < quarter)].dropna()
    if cur.empty or len(prev) < 3:
        return np.nan, len(cur)
    return np.log(cur.mean() / prev.mean()) * 400, len(cur)


def history(x, before, start):
    """Complete-quarter annualized growth over [start, before): the window the moments come from."""
    q = monthly(x).resample("QS").agg(["mean", "count"])
    q = q["mean"].where(q["count"] == 3)
    g = np.log(q / q.shift(1)) * 400
    return g[(g.index >= start) & (g.index < before)].dropna()


def inputs(panel, as_of, spec):
    """Each activity series as of `as_of`, deflated where the config says so."""
    out = {}
    for name in spec["series"]:
        x = monthly(panel.series(name, as_of))
        deflator = spec.get("deflate", {}).get(name)
        if deflator:
            d = monthly(panel.series(deflator, as_of))
            # A hole inside the deflator (October 2025 CPI) is log-interpolated.
            x = x / np.exp(np.log(d).interpolate(limit_area="inside")) * 100
        out[name] = x
    return out


def score(data, quarter, start):
    """Each series' quarter-to-date growth as a robust z, and their mean."""
    out, zs = {"activity_quarter": quarter}, []
    for name, x in data.items():
        g, n = quarter_growth(x, quarter)
        h = history(x, quarter, start)
        med = h.median()
        z = (g - med) / ((h - med).abs().median() * MAD_TO_SD)
        out |= {f"{name}_growth": g, f"{name}_months": n, f"{name}_z": z}
        if not np.isnan(z):
            zs.append(z)
    out |= {"activity_z": np.mean(zs) if zs else np.nan, "activity_n": len(zs)}
    return out


def activity(panel, as_of, spec):
    """Quarter-to-date activity, scored against the series' own history up to that quarter.

    The quarter is the lead series' latest (the first listed prints first); if
    fewer than `min_series` series reach it yet, the previous quarter instead.
    """
    data = inputs(panel, as_of, spec)
    lead = data[spec["series"][0]].dropna()
    if lead.empty:
        raise ValueError(f"no {spec['series'][0]} published by {pd.Timestamp(as_of).date()}")
    quarter = lead.index[-1].to_period("Q").start_time
    out = score(data, quarter, spec["moments_start"])
    if out["activity_n"] < spec["min_series"]:
        out = score(data, quarter - QUARTER, spec["moments_start"])
    return out
