"""QuantLib rate helpers for the SOFR curve, from cached settles and fixings.

QuantLib keeps the evaluation date and index fixings in global singletons.
`sofr_index` resets both for one as-of date, so curves built for different
dates never see each other's history, and no fixing published after `as_of`
reaches a helper.
"""

import pandas as pd
import QuantLib as ql
from policypath.curves.futures import reference_period

FREQUENCY = {"month": ql.Monthly, "imm_quarter": ql.Quarterly}
SOFR_CALENDAR = ql.UnitedStates(ql.UnitedStates.SOFR)


def ql_date(ts):
    ts = pd.Timestamp(ts)
    return ql.Date(ts.day, ts.month, ts.year)


def sofr_business_days(start, end):
    """SOFR fixing dates in [start, end), on QuantLib's SOFR calendar."""
    days = pd.date_range(start, pd.Timestamp(end) - pd.Timedelta(days=1), freq="D")
    return pd.DatetimeIndex([d for d in days if SOFR_CALENDAR.isBusinessDay(ql_date(d))])


def sofr_index(fixings, as_of):
    """``ql.Sofr()`` holding every fixing published by `as_of` and nothing else.

    Also sets QuantLib's evaluation date to `as_of`. `fixings` has ``date``,
    ``value`` (percent) and ``published``.
    """
    as_of = pd.Timestamp(as_of)
    ql.Settings.instance().evaluationDate = ql_date(as_of)
    index = ql.Sofr()
    index.clearFixings()
    known = fixings[fixings["published"] < as_of.normalize() + pd.Timedelta(days=1)]
    dates = [ql_date(d) for d in known["date"]]
    valid = [index.isValidFixingDate(d) for d in dates]
    index.addFixings([d for d, ok in zip(dates, valid) if ok],
                     [v / 100.0 for v, ok in zip(known["value"], valid) if ok])
    return index


def futures_helpers(settles, shape, as_of, horizon=None, min_days_left=0):
    """One `SofrFutureRateHelper` per contract still accruing after `as_of`.

    `settles` has ``contract``, ``expiration`` and ``value`` (price). Contracts
    whose reference period ends within `min_days_left` days of `as_of` are left
    out, for the same reason the policy-path solver drops a nearly expired front
    contract; so are those ending after `horizon`. Returns (helpers, contracts),
    the second a frame of what each helper is: contract, start, end, price.
    """
    as_of = pd.Timestamp(as_of)
    rows = []
    for contract, expiration, price in settles[["contract", "expiration", "value"]].itertuples(index=False):
        start, end = reference_period(shape, expiration)
        if (end - as_of).days <= min_days_left or (horizon is not None and end > horizon):
            continue
        rows.append({"contract": contract, "start": start, "end": end, "price": float(price)})
    contracts = pd.DataFrame(rows).sort_values("end").reset_index(drop=True)
    helpers = [ql.SofrFutureRateHelper(r.price, r.start.month, r.start.year, FREQUENCY[shape])
               for r in contracts.itertuples()]
    return helpers, contracts
