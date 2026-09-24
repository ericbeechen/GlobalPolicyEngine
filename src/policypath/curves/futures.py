"""What a short-rate future settles to, written out independently of QuantLib.

Two contract shapes cover ZQ, SR1 and SR3:

- ``month``: the arithmetic average of the overnight rate over every calendar
  day of the contract month (ZQ on EFFR, SR1 on SOFR).
- ``imm_quarter``: SOFR compounded daily over the IMM quarter, from the third
  Wednesday of the contract month to the third Wednesday three months later,
  Actual/360 (SR3). CME names the contract by the month the quarter starts
  and expires it the business day before the quarter ends.

`curves/helpers.py` builds the QuantLib versions; `tests/test_curves.py`
checks the two agree, which is what catches a convention error in either.
"""

import numpy as np
import pandas as pd


def third_wednesday(year, month):
    first = pd.Timestamp(year, month, 1)
    return first + pd.Timedelta(days=(2 - first.weekday()) % 7 + 14)


def reference_period(shape, expiration):
    """[start, end) of the contract's reference period, from its expiration date."""
    expiration = pd.Timestamp(expiration)
    if shape == "month":
        month = expiration.to_period("M")
        return month.start_time, month.end_time.normalize() + pd.Timedelta(days=1)
    if shape == "imm_quarter":
        end_month = expiration.to_period("M")
        start_month = end_month - 3
        return (third_wednesday(start_month.year, start_month.month),
                third_wednesday(end_month.year, end_month.month))
    raise ValueError(f"unknown contract shape {shape!r}")


def settlement_rate(shape, start, end, daily, fixing_dates=None):
    """The rate a contract settles to, in percent, given the overnight rate on every day.

    `daily` is the rate on each calendar day in [start, end). For compounding,
    `fixing_dates` are the business days whose fixings apply: each accrues for
    the calendar days until the next one, as SOFR does over a weekend, clipped
    to the period. The first must fall on or before `start` -- when the period
    opens on a holiday (SR3M4 on Juneteenth 2024), the preceding business day's
    fixing covers it, per CME's rule for non-business days.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    daily = daily.loc[start:end - pd.Timedelta(days=1)]
    if len(daily) != (end - start).days:
        raise ValueError(f"need a rate for every day in {start.date()}..{end.date()}")
    if shape == "month":
        return float(daily.mean())
    if shape == "imm_quarter":
        dates = pd.DatetimeIndex(sorted(fixing_dates))
        dates = dates[dates < end]
        opening = dates[dates <= start]
        if len(opening) == 0:
            raise ValueError(f"no fixing on or before the period start {start.date()}")
        dates = dates[dates >= opening[-1]]
        lo = dates.where(dates > start, start)  # the opening fixing accrues from `start`
        hi = lo[1:].append(pd.DatetimeIndex([end]))
        accrual = (hi - lo).days.to_numpy(dtype=float)
        growth = np.prod(1.0 + daily.loc[lo].to_numpy() / 100.0 * accrual / 360.0)
        return float((growth - 1.0) * 360.0 / (end - start).days * 100.0)
    raise ValueError(f"unknown contract shape {shape!r}")
