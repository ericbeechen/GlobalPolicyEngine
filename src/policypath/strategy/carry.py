"""Carry, roll and the rate move: every formula that splits a leg's P&L, the checks on the split, and the breakeven.

Draft for the author to own: every formula and sign is written out below so
it can be re-derived without the code.

**Units and signs.** A unit position is +1 of book-currency DV01 (USD per bp),
positive = receive: long the price, gaining when the rate falls. P&L is in bp
of that DV01 (dollars per unit). Rates are in percent, so a rate difference
times 100 is bp. A leg outside the book currency holds 1/S_p of its own DV01
from p and converts at fx = S_t / S_p (every component).

**Attribution** (`attribution`) of a unit receive held from session p into t,
in the instrument h selected at p, delta calendar days apart, with
``held_p = y_p(tau_p(h))``, ``held_t = y_t(tau_t(h))`` and
``rolled = y_p(tau_t(h))`` (p's curve at the maturity h has at t)::

    carry = +100 x (c x delta/365 - r_p x delta/basis) / A_p x fx     funded (par) legs; else 0
    roll  = -(rolled - held_p) x 100 x fx
    rate  = -(held_t - rolled) x 100 x fx

c = y_p(T), the coupon a par leg struck at p pays, accruing Act/365; r_p the
overnight rate in force at p, the funding, on its own day count (USD 360, GBP
365), standing in for GC or gilt repo; A_p the par bond's modified duration,
which turns a coupon in percent of notional into bp of yield. Receiving on an
upward curve earns roll (the leg rolls down to a lower yield) and, where
y > r, carry. Receiving on an inverted curve loses both: right about the
rate and still bleeding. Paying on an inverted curve earns both.

A futures month and a forward are not funded: they cost nothing to hold, so
their carry is 0 and the whole move is roll plus rate.

**Three checks** (`identity`, `revaluation`, and the convergence tests):

1. The bookkeeping identity: ``[-(held_t - held_p) x 100 + carry / fx] x fx``
   equals carry + roll + rate to 1e-9bp. The two sides telescope, so this is
   nearly true by construction: it catches a relative sign error between roll
   and rate and a maturity mix-up, and **does not prove the signs**.
2. An independent full revaluation: the leg's P&L from its own valuation, not
   from the rate-space pieces (`revaluation`). ZQ: contracts x (P_t - P_p) x
   point value, contracts = q / DV01 per contract, point value = DV01 x 100 per
   1.00 of price, prices as settled. Forward: a receiver struck at K = F_p (worth
   0 at p) is worth N x alpha x (K - F_t) x DF_t(E_k+1) at t, N set by the
   leg's DV01 at p (alpha x DF_p x 1bp), at S_t. That is the spec's FRA-style
   value, linear in the continuously compounded forward F; the exact OIS
   receiver's DV01 in F is alpha x DF(E_k), larger by e^(alpha F), about 0.5%
   (a six-week window at 4.4%), which neither side of the check sees. Par: a
   bond struck at p with coupon c, priced at y_t(T - delta/365)
   (`instruments.bond_price`), plus the coupon accrued less the funding, on
   notional 1e4 / A_p, at S_t. What is left over is convexity and the drift of
   the leg's DV01 over the period, which the linear marks do not have, bounded
   by ``bound``. A flipped carry sign leaves about twice the carry, far
   outside it, and so does an attribution fx that disagrees with the
   valuation's spot (about 2 x dS/S x the P&L). The check is blind to
   what both sides share: an FX quote read upside down at its source moves the
   valuation's spot with the linear pieces, and a leg traded the wrong way
   round is negated on both. The quote's orientation is pinned by
   ``expression.fx.plausible`` (`instruments.build` refuses a spot outside it)
   and a test on the synthetic cache; the direction, by check 3.
3. Convergence (``tests/test_carry.py``, and the end-to-end check in the
   build), which pins the direction: move the market to the model and every
   sleeve with z > 0 gains; a fixed-window leg gains exactly the de-meaned
   gap. The GBP outright with the linear rule and fx = 1 reproduces the week
   6 backtest's P&L to 1e-9.

Convexity is not in the linear P&L: legs are DV01-linear marks re-struck at
each close. On a DV01-neutral 2s10s the design review estimated it at about
3.6bp a year per unit DV01. Measured, as the full revaluation's residual
summed by year, receiving the 2s10s (short the 10y's convexity) gives up
2.6bp a year in USD and 2.7bp in GBP, 2011-26, most in 2022 (4.8 and 7.0bp):
the linear P&L flatters a receiver by that much.

**The breakeven** (`carry_roll_ahead`, `edge`, `closure`, `bleed`). Over h =
``carry.horizon_days`` (91) the instrument selected today, held h days with the
curve frozen, earns CR_h = carry + roll (``roll = -(y(tau - h) - y(tau)) x 100``,
carry as above over h), signed for the side. That is for a fixed instrument,
not the sleeve's re-selection. The signal's edge is side x the de-meaned gap
(|z| x sd), the bp it expects to close. If over h the rate closes a fraction
phi of the edge and otherwise follows the frozen curve, the trade earns

    E_h = phi x edge + (1 - phi) x CR_h

(exact for a fixed-window instrument: at full convergence the roll cancels,
so nothing is counted twice). "Right but bleeding" is CR_h < 0 < edge; the
break-even closure is phi* = -CR_h / (edge - CR_h). phi is estimated
expanding-window per sleeve (`closure`). A signal does not pay for its bleed
when E_h < 0 (for CR_h < 0 < edge, when phi* > phi). In words: "Over a quarter
the gap has historically closed by a fraction phi; the rest of the time the
curve stands still and the trade earns its carry and roll. If that expected
quarter is negative, the signal is right and still not worth trading."
"""

import numpy as np
import pandas as pd
from policypath.strategy.instruments import YEAR, bond_price, par_duration

PARTS = ["carry", "roll", "rate"]
TOL_BP = 1e-9          # rounding: the identity's tolerance, and the slack on the revaluation's bound


def _carry(coupon, funding, basis, duration, days):
    """Coupon accrued less funding over `days`, in bp of yield: 100 x (c x days/365 - r x days/basis) / A."""
    return 100.0 * (coupon * days / YEAR - funding * days / basis) / duration


def _held_carry(leg):
    """The held leg's carry from p to t in its own currency, bp: `_carry` for a funded leg, else 0."""
    f = leg.frame
    if not leg.funded:
        return pd.Series(0.0, index=f.index)
    return _carry(f["coupon"], f["funding_p"], f["basis"], f["duration_p"], f["delta"])


def attribution(leg):
    """carry, roll, rate and total, bp, per session: a unit receive of `leg` held from the previous session."""
    f = leg.frame
    fx = f["fx"]
    carry = _held_carry(leg) * fx
    roll = -(f["rolled"] - f["held_p"]) * 100.0 * fx
    rate = -(f["held_t"] - f["rolled"]) * 100.0 * fx
    return pd.DataFrame({"carry": carry, "roll": roll, "rate": rate, "total": carry + roll + rate})


def identity(leg, parts):
    """Check 1: the linear total, -(held_t - held_p) x 100 plus carry, at fx, less carry + roll + rate, bp."""
    f = leg.frame
    linear = (-(f["held_t"] - f["held_p"]) * 100.0 + _held_carry(leg)) * f["fx"]
    return linear - parts[PARTS].sum(axis=1, skipna=False)


def _curvature(c, y, tau, f, step=0.01):
    """d2P/dy2 of `bond_price` at y (percent), per percent squared, by a central difference."""
    return (bond_price(c, y + step, tau, f) - 2.0 * bond_price(c, y, tau, f) + bond_price(c, y - step, tau, f)) / step**2


def revaluation(leg):
    """Check 2: (full, bound), bp per session, for a unit receive of `leg` from the previous session.

    `full` is the P&L from the instrument's own valuation (module docstring).
    `bound` is what the linear attribution can miss by: 0 for a futures month
    (its price is linear in its rate); for a forward, the move in F times the
    move in its discount factor, |dF| x 100 x |x| e^|x| with x = log(DF_t/DF_p);
    for a par leg the drift of its DV01 as its maturity shortens,
    1e4 |dy| |1 - A(c, tau_t)/A_p|, plus convexity, 1/2 x N x max P'' x dy^2
    (P'' the larger at the two ends of the move, 1% added for the difference
    quotient). `full - total` must lie within `bound` (+ `TOL_BP`). Every
    component converts at S_t / S_p, as the valuation of 1/S_p of DV01 does at
    S_t, so there is no FX cross term to allow for.
    """
    f = leg.frame
    if leg.kind == "futures_month":
        contracts = 1.0 / (f["spot_p"] * f["dv01"].shift(1))
        full = contracts * (f["price_t"] - f["price_p"]) * f["dv01"].shift(1) * 100.0 * f["spot"]
        return full, pd.Series(0.0, index=f.index)
    if leg.kind == "curve_forward":
        notional = 1.0 / (f["spot_p"] * f["alpha"] * f["df_p"] * 1e-4)
        full = notional * f["alpha"] * (f["held_p"] - f["held_t"]) / 100.0 * f["df_t"] * f["spot"]
        x = np.log(f["df_t"] / f["df_p"])
        return full, (f["held_p"] - f["held_t"]).abs() * 100.0 * x.abs() * np.exp(x.abs()) * f["fx"]
    notional = 1e4 / (f["duration_p"] * f["spot_p"])
    accrued = (f["coupon"] * f["delta"] / YEAR - f["funding_p"] * f["delta"] / f["basis"]) / 100.0
    price = bond_price(f["coupon"], f["held_t"], f["tau"], leg.f)
    full = notional * (price - 1.0 + accrued) * f["spot"]
    dy = f["held_t"] - f["coupon"]
    drift = 100.0 * dy.abs() * (1.0 - par_duration(f["coupon"], f["tau"], leg.f) / f["duration_p"]).abs() * f["fx"]
    curve = np.maximum(_curvature(f["coupon"], f["coupon"], f["tau"], leg.f),
                       _curvature(f["coupon"], f["held_t"], f["tau"], leg.f)) * 1.01
    convexity = 0.5 * notional * f["spot"] * curve * dy**2
    return full, drift + convexity


def carry_roll_ahead(leg, h):
    """(carry, roll), bp, of a unit receive of the instrument selected at each session, held `h` days, curve frozen.

    roll = -(y(tau - h) - y(tau)) x 100; carry as `attribution` over h (funded
    legs), at today's coupon, funding and duration. No FX move: the unit is
    book DV01 at today's spot.
    """
    f = leg.frame
    if leg.funded:
        carry = _carry(f["rate"], f["funding"], f["basis"], f["duration"], float(h))
    else:
        carry = pd.Series(0.0, index=f.index)
    return carry, -(f["ahead"] - f["rate"]) * 100.0


def edge(side, dev_bp):
    """The bp a signal expects to close: side x the de-meaned gap (|z| x sd for the component's own gap)."""
    return side * dev_bp


def closure(value_bp, dev_bp, h, min_pairs):
    """phi_h per session: how much of its de-meaned gap a component has closed over `h` days, expanding-window.

    OLS without intercept of -(value_{t+h} - value_t) on dev_t over every pair
    whose end is known by the session (t + h is the first session on or after
    t plus `h` calendar days), clipped to [0, 1]; NaN until `min_pairs` pairs.
    """
    days = value_bp.index
    end = days.searchsorted(days + pd.Timedelta(days=h))
    inside = end < len(days)
    x = dev_bp.to_numpy()[inside]
    y = -(value_bp.to_numpy()[end[inside]] - value_bp.to_numpy()[inside])
    ok = ~(np.isnan(x) | np.isnan(y))
    known = days[end[inside]][ok]
    xy, xx = np.cumsum(x[ok] * y[ok]), np.cumsum(x[ok] ** 2)
    n = known.searchsorted(days, side="right")
    phi = np.full(len(days), np.nan)
    enough = n >= min_pairs
    phi[enough] = xy[n[enough] - 1] / xx[n[enough] - 1]
    return pd.Series(np.clip(phi, 0.0, 1.0), index=days, name="phi_h")


def bleed(edge_bp, cr_bp, phi):
    """Does the expected quarter pay for the bleed? Per session: bleeding, phi_star, expected_bp, pays.

    bleeding = CR_h < 0 < edge. phi_star = -CR_h / (edge - CR_h) where CR_h < 0
    (inf where no closure pays, edge <= CR_h), 0 otherwise. expected_bp = E_h =
    phi x edge + (1 - phi) x CR_h; pays = E_h >= 0 (missing until phi is known).
    """
    cr, e = cr_bp.to_numpy(dtype=float), edge_bp.to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        star = np.where(cr < 0, np.where(e > cr, -cr / (e - cr), np.inf), 0.0)
    star[np.isnan(cr) | np.isnan(e)] = np.nan
    expected = phi * edge_bp + (1.0 - phi) * cr_bp
    pays = (expected >= 0).astype("boolean").mask(expected.isna())
    return pd.DataFrame({"bleeding": (cr_bp < 0) & (edge_bp > 0), "phi_star": star, "expected_bp": expected,
                         "pays": pays}, index=edge_bp.index)
