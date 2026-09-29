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

`log_discount` and `window_rates` are the two steps every forward off such a
curve takes, shared with the expression layer's forward legs
(`strategy/instruments.py`), so a traded forward and the path's rate at the
same meeting are one computation.
"""

import numpy as np
import pandas as pd


def log_discount(spot):
    """A day's spot curve as -log P at its nodes, with 0 at maturity 0: (nodes in years, -log P at each).

    `spot` is percent, continuously compounded, by maturity in months, without
    NaN. -log P = s t, read between nodes with `np.interp` (linear in t: the
    forward is flat between two nodes).
    """
    tenors = spot.index.to_numpy(dtype=float) / 12.0
    return np.concatenate([[0.0], tenors]), np.concatenate([[0.0], spot.to_numpy() / 100.0 * tenors])


def window_rates(t, at):
    """The average rate, percent, over each window between consecutive maturities `t` (years), from -log P `at` there.

    [s(t2) t2 - s(t1) t1] / (t2 - t1), continuously compounded: exact from the two log discount factors.
    """
    return (at[1:] - at[:-1]) / (t[1:] - t[:-1]) * 100.0


def forward_path(spot, as_of, effective_dates, end, year_days, last_fixing=None, min_regime_days=0,
                 pin_always=False):
    """Piecewise-constant overnight rate between meeting effective dates, from one day's spot curve.

    `spot` is the curve on `as_of`: percent, continuously compounded, indexed
    by maturity in months. `effective_dates` are the meetings ahead, after
    `as_of`; `end` closes the last regime (the meeting after the last, or a
    date past it). `year_days` turns days into the curve's years (365: the
    Bank's curve is on SONIA's Act/365). The first regime is pinned to `last_fixing` when the first
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
    nodes, log_df = log_discount(spot)
    tenors = nodes[1:]

    dates = pd.DatetimeIndex([*effective_dates, end])
    if (dates <= as_of).any() or not dates.is_monotonic_increasing:
        raise ValueError(f"meeting dates must be after {as_of.date()} and increasing")
    t = (dates - as_of).days.to_numpy() / float(year_days)
    if t[-1] > nodes[-1]:
        raise ValueError(f"the curve on {as_of.date()} ends at {spot.index[-1]} months, "
                         f"short of {dates[-1].date()}")
    at = np.interp(t, nodes, log_df)

    pinned = last_fixing is not None and (pin_always or t[0] < tenors[0]
                                          or (dates[0] - as_of).days < min_regime_days)
    if pinned:
        at[0] = last_fixing / 100.0 * t[0]
    rates = window_rates(np.concatenate([[0.0], t]), np.concatenate([[0.0], at]))

    path = pd.Series(rates, index=pd.DatetimeIndex([as_of, *dates[:-1]]), name="rate")
    path.attrs = {"pinned": bool(pinned), "residual_bp": np.nan, "nodes": len(spot)}
    return path
