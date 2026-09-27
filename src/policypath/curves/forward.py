"""The policy path from a fitted spot OIS curve: the forward rate between consecutive meetings.

A spot curve at fixed maturities is the log discount factor at a few points,
log P(t) = -s(t) t, with s continuously compounded (checked on the Bank of
England curves: forwards from s t match the published instantaneous forwards
to 0.1bp; annual compounding would miss by up to 14bp). The expected average
overnight rate between two dates is then the forward over that window,
[s(t2) t2 - s(t1) t1] / (t2 - t1), exact from two spot rates. No instantaneous
forward is interpolated, and no curve is fitted here.

Meeting dates fall between the published maturities, so s t is interpolated
linearly between them: that is exact at every node and assumes the forward
flat between two nodes, which a meeting window then averages over. Below the
first node the curve says nothing about where inside the month a move lands,
so a first meeting before it pins today-until-then to the last fixing, as
ZQ's short first regime is pinned (`curves/policy_path.py`).
"""

import numpy as np
import pandas as pd

YEAR_DAYS = 365.0   # SONIA accrues Act/365


def forward_path(spot, as_of, effective_dates, end, last_fixing=None, min_regime_days=0, pin_always=False):
    """Piecewise-constant overnight rate between meeting effective dates, from one day's spot curve.

    `spot` is the curve on `as_of`: percent, continuously compounded, indexed
    by maturity in months. `effective_dates` are the meetings ahead, after
    `as_of`; `end` closes the last regime (the meeting after the last, or a
    date past it). The first regime is pinned to `last_fixing` when the first
    meeting comes before the curve's first maturity, or less than
    `min_regime_days` away, or always with `pin_always`: no meeting falls
    inside it, so its rate is the rate in force, and reading it off the curve
    instead smears the first meeting's step across the node interval it lands in.

    Returns the rate in force from `as_of` and from each effective date, like
    `implied_path`. ``attrs`` carries ``pinned``, ``residual_bp`` (NaN: a curve
    is not overdetermined) and ``nodes``.
    """
    as_of = pd.Timestamp(as_of)
    spot = spot.dropna().sort_index()
    if spot.empty:
        raise ValueError(f"no spot rates on {as_of.date()}")
    tenors = spot.index.to_numpy(dtype=float) / 12.0
    log_df = np.concatenate([[0.0], spot.to_numpy() / 100.0 * tenors])   # -log P, at 0 and each node
    nodes = np.concatenate([[0.0], tenors])

    dates = pd.DatetimeIndex([*effective_dates, end])
    if (dates <= as_of).any() or not dates.is_monotonic_increasing:
        raise ValueError(f"meeting dates must be after {as_of.date()} and increasing")
    t = (dates - as_of).days.to_numpy() / YEAR_DAYS
    if t[-1] > nodes[-1]:
        raise ValueError(f"the curve on {as_of.date()} ends at {spot.index[-1]} months, "
                         f"short of {dates[-1].date()}")
    at = np.interp(t, nodes, log_df)

    pinned = last_fixing is not None and (pin_always or t[0] < tenors[0]
                                          or (dates[0] - as_of).days < min_regime_days)
    if pinned:
        at[0] = last_fixing / 100.0 * t[0]
    starts = np.concatenate([[0.0], t[:-1]])
    before = np.concatenate([[0.0], at[:-1]])
    rates = (at - before) / (t - starts) * 100.0

    path = pd.Series(rates, index=pd.DatetimeIndex([as_of, *dates[:-1]]), name="rate")
    path.attrs = {"pinned": bool(pinned), "residual_bp": np.nan, "nodes": len(spot)}
    return path
