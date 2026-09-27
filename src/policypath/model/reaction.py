"""The Fed's balanced-approach rule, in its inertial form, imposed rather than estimated.

From the Monetary Policy Report:

    notional   R* = r* + pi + a (pi - pi*) + b (u* - u)
    inertial   R_k = rho R_(k-1) + (1 - rho) max(elb, R*)

with a = 0.5 and b = 2 (balanced approach) and rho = 0.85 a quarter. Every
coefficient comes from the config's ``rule`` block, so an estimated rule is a
different block, not a rewrite. The MPR's rho is quarterly and the FOMC meets
twice a quarter, so each meeting moves ``1 - inertia ** (1 / meetings_per_quarter)``
of the way.

Nothing here reads data. `model/path.py` hands in the macro picture of one date.
"""

import numpy as np


def notional(inflation, u_gap, rstar, spec):
    """The rule's unconstrained rate, percent. `u_gap` is u - u*, positive = slack.

    Kept unfloored so what the rule asks for at the lower bound is still on record.
    """
    c = spec["coefficients"]
    return (rstar + inflation + c["inflation_gap"] * (inflation - spec["inflation_target"])
            + c["unemployment_gap"] * -u_gap)


def floor_at_elb(rate, spec):
    """The effective lower bound, as a floor the rule itself does not know about.

    Through 2011-15 and 2020-21 the notional rate is several points negative
    while the Fed sat at 0-0.25. The model path does not go below the bottom
    of what the Fed will set, so the floor is applied to R* before the inertial
    step; with R_0 at or above it, the whole path stays at or above it too.
    """
    return max(rate, spec["elb"])


def per_meeting_inertia(spec):
    return spec["inertia"] ** (1.0 / spec["meetings_per_quarter"])


def inertial_path(r0, target, n, spec):
    """The rate after each of the next `n` meetings, from `r0` in force today toward `target`."""
    rho = per_meeting_inertia(spec)
    out, r = [], r0
    for _ in range(n):
        r = rho * r + (1.0 - rho) * target
        out.append(r)
    return np.array(out)
