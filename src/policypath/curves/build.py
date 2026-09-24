"""Discount curves from QuantLib rate helpers.

`build_curve` is ported from the credit repo (systematic_rel_val/curve_fitting.py)
with the two parametric branches unchanged:

- ``piecewise_log_linear``: bootstraps through every helper, so the curve
  reprices its inputs exactly. What the SOFR curve uses.
- ``nss``: a Svensson fit, `FittedBondDiscountCurve`. QuantLib fits it to bond
  helpers only, so it is here for Treasury and credit curves; handed futures
  helpers it raises rather than failing somewhere inside SWIG.
"""

import pandas as pd
import QuantLib as ql
from policypath.curves.helpers import futures_helpers, ql_date, sofr_index

DAY_COUNTS = {"Actual360": ql.Actual360(), "Actual365Fixed": ql.Actual365Fixed()}


def build_curve(calc_date, helpers, day_count, fitting_method):
    if fitting_method == "piecewise_log_linear":
        curve = ql.PiecewiseLogLinearDiscount(calc_date, helpers, day_count)
    elif fitting_method == "nss":
        if not all(isinstance(h, ql.BondHelper) for h in helpers):
            raise TypeError("nss is fitted to bond helpers only (FittedBondDiscountCurve)")
        tolerance = 1e-8
        max_iterations = 5000
        curve = ql.FittedBondDiscountCurve(
            calc_date,
            helpers,
            day_count,
            ql.SvenssonFitting(),
            tolerance,
            max_iterations,
        )
    else:
        raise ValueError(f"unknown fitting method {fitting_method!r}")
    curve.enableExtrapolation()
    return curve


def sofr_curve(as_of, settles, fixings, spec, shape="imm_quarter", min_days_left=0):
    """The SOFR discount curve on `as_of`, bootstrapped from the futures strip.

    `settles` are one session's settles for the SOFR futures root, `fixings`
    SOFR with publication dates, `spec` the currency's ``sofr`` config block.
    Returns (curve, helpers, contracts); keep `helpers` alive as long as the curve.
    """
    as_of = pd.Timestamp(as_of)
    sofr_index(fixings, as_of)
    horizon = as_of + pd.DateOffset(years=spec["curve_years"])
    helpers, contracts = futures_helpers(settles, shape, as_of, horizon, min_days_left)
    curve = build_curve(ql_date(as_of), helpers, DAY_COUNTS[spec["day_count"]],
                        "piecewise_log_linear")
    curve.nodes()  # bootstrap now, so a bad input fails here rather than at first use
    return curve, helpers, contracts
