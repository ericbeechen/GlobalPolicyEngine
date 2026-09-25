"""The unemployment gap: u - u* in percentage points, from one vintage.

Not an output gap. Potential GDP is quarterly, revised hard and rewritten in
hindsight; see notes/macro.md.
"""

import pandas as pd


def unemployment_gap(panel, as_of, spec):
    """u - u* for the latest month with an unemployment rate. Positive means slack.

    u* is the natural rate for that month's quarter, from the same vintage.
    Raises rather than carrying u* forward if the vintage has no value there.
    """
    u = panel.series(spec["unemployment"], as_of).dropna()
    if u.empty:
        raise ValueError(f"no {spec['unemployment']} published by {pd.Timestamp(as_of).date()}")
    month = u.index[-1]
    quarter = month.to_period("Q").start_time
    star = panel.series(spec["natural_rate"], as_of)
    if pd.isna(star.get(quarter)):
        raise ValueError(f"{spec['natural_rate']} has no value for {quarter.date()} as of {pd.Timestamp(as_of).date()}")
    return {"gap_month": month, "u": u.iloc[-1], "u_star": star[quarter], "u_gap": u.iloc[-1] - star[quarter]}
