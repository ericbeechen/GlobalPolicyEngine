"""The Fed's balanced-approach rule, in its inertial form, imposed rather than estimated.

From the Monetary Policy Report:

    notional   R* = r* + pi + a (pi - pi*) + b (u* - u)
    inertial   R_k = rho R_(k-1) + (1 - rho) max(elb, R*)

with a = 0.5 and b = 2 (balanced approach) and rho = 0.85 a quarter. Every
coefficient comes from the config's ``rule`` block, so an estimated rule is a
different block, not a rewrite. The MPR's rho is quarterly and the FOMC meets
twice a quarter, so each meeting moves ``1 - inertia ** (1 / meetings_per_quarter)``
of the way.

The lower bound, the meeting frequency and the inflation target can change over
time (the MPC met monthly until 2015; the Bank saw its floor at 0.5% until
August 2016; the ECB's "below, but close to, 2%" became a symmetric 2% in July
2021), so each may be given in config as a dated schedule, resolved for a date by `on`.

Nothing here reads data. `model/path.py` hands in the macro picture of one date.
"""

import numpy as np
import pandas as pd


def on(value, as_of):
    """A config value as it stood on `as_of`: a number, or a schedule of ``{from: date, value: x}``."""
    if not isinstance(value, list):
        return value
    live = [s for s in value if pd.Timestamp(s["from"]) <= pd.Timestamp(as_of)]
    if not live:
        raise ValueError(f"the schedule {value} has no entry in force on {pd.Timestamp(as_of).date()}")
    return max(live, key=lambda s: pd.Timestamp(s["from"]))["value"]


def notional(inflation, u_gap, rstar, spec, as_of=None):
    """The rule's unconstrained rate, percent. `u_gap` is u - u*, positive = slack.

    Kept unfloored so what the rule asks for at the lower bound is still on record.
    """
    c = spec["coefficients"]
    return (rstar + inflation + c["inflation_gap"] * (inflation - on(spec["inflation_target"], as_of))
            + c["unemployment_gap"] * -u_gap)


def lower_bound(spec, as_of):
    """The floor in force on `as_of`: per currency, and possibly dated, so it lives in config."""
    return on(spec["elb"], as_of)


def floor_at_elb(rate, elb):
    """The effective lower bound, as a floor the rule itself does not know about.

    Through 2011-15 and 2020-21 the notional rate is several points negative
    while the Fed sat at 0-0.25. The model path does not go below the bottom
    of what the central bank will set, so the floor is applied to R* before the
    inertial step; with R_0 at or above it, the whole path stays at or above it too.
    """
    return max(rate, elb)


def per_meeting_inertia(spec, as_of=None):
    return spec["inertia"] ** (1.0 / on(spec["meetings_per_quarter"], as_of))


def inertial_path(r0, target, n, spec, as_of=None):
    """The rate after each of the next `n` meetings, from `r0` in force today toward `target`.

    `target` is one rate for every meeting (hold-flat), or one per meeting
    (converge-to-target conditioning moves the goal as the macro picture does).
    """
    rho = per_meeting_inertia(spec, as_of)
    goals = [target] * n if np.ndim(target) == 0 else list(target)
    if len(goals) != n:
        raise ValueError(f"{len(goals)} targets for {n} meetings")
    out, r = [], r0
    for goal in goals:
        r = rho * r + (1.0 - rho) * goal
        out.append(r)
    return np.array(out)
