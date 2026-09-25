"""SOFR futures against the ZQ path: the contract identity and the basis solve.

The cross-check in `curves/basis.py` reads the SOFR - EFFR basis out of SR1
(or SR3) settles given the ZQ path. Two things have to hold for that to mean
anything: SR1 must settle the way `curves/futures.py` says -- checked against
realized SOFR, as `test_policy_path.py` checks ZQ against realized EFFR -- and
the solve must hand back a basis that was put in by construction.
"""

from pathlib import Path
import pandas as pd
import pytest
from policypath.calendars import US_BDAY
from policypath.curves.basis import implied_basis, next_period
from policypath.curves.futures import reference_period, settlement_rate
from policypath.curves.helpers import sofr_business_days

DATA = Path(__file__).parent / "data"
BP = 0.01  # in percent
DAY = pd.Timedelta(days=1)


@pytest.fixture(scope="module")
def sofr():
    return pd.read_csv(DATA / "sofr.csv", parse_dates=["date", "published"])


def test_expired_sr1_contracts_settle_to_realized_sofr(sofr):
    """100 - settle == SOFR averaged over every calendar day of the contract month.

    SR1's definition, as the ZQ identity is ZQ's. Weekends and holidays carry
    the last fixing. As for ZQ, the archive holds the expiry session's settle,
    which has not seen the fixings from its own day to the month end, and SOFR
    jumps at month ends (by 10-20bp in 2018-19). So the tolerance is one tick
    plus how far those unseen fixings moved from the last one published.
    """
    settles = pd.read_csv(DATA / "sr1_expiry_settles.csv")
    settles["month"] = settles["month"].apply(pd.Period)
    daily = sofr.set_index("date")["value"].sort_index()
    daily = daily.reindex(pd.date_range(daily.index[0], daily.index[-1], freq="D")).ffill()
    full = daily.groupby(daily.index.to_period("M")).count()
    avg = daily.groupby(daily.index.to_period("M")).mean()[full == full.index.days_in_month]
    settles = settles[settles["month"].isin(avg.index)]
    assert len(settles) > 90, "fixture too thin to be a meaningful check"

    months = settles["month"].to_numpy()
    err = pd.Series(100.0 - settles["price"].to_numpy() - avg.loc[months].to_numpy(), index=months)
    allowed = 0.3 * BP + unseen_move(daily, months, settles["expiry"])
    excess = err.abs() - allowed
    assert excess.max() < 0, (f"worst month {excess.idxmax()}: {err[excess.idxmax()] / BP:+.2f}bp, "
                              f"allowed {allowed[excess.idxmax()] / BP:.2f}bp")
    calm = allowed <= 0.3 * BP + 1e-12
    assert calm.sum() > 15 and err[calm].abs().max() < 0.3 * BP  # 24 of 100: SOFR rarely sits still


def unseen_move(daily, months, expiries):
    """Per contract, |unseen fixing - last published fixing| summed over the days from expiry to month end, / days in month."""
    out = []
    for month, expiry in zip(months, pd.to_datetime(expiries)):
        last_seen = daily[:expiry - DAY].iloc[-1]
        out.append((daily[expiry:month.end_time.normalize()] - last_seen).abs().sum() / month.days_in_month)
    return pd.Series(out, index=months)


def test_sr1_reference_period_is_the_contract_month():
    # SR1U4 expires on the last business day of September 2024.
    assert reference_period("month", "2024-09-30") == (pd.Timestamp("2024-09-01"), pd.Timestamp("2024-10-01"))


# --------------------------------------------------------------------------
# the basis solve, on a market built with a known basis
# --------------------------------------------------------------------------

SESSION = pd.Timestamp("2024-06-12")
RATE_NOW, CUT, AFTER = 5.33, pd.Timestamp("2024-09-19"), 5.08
FIXED_SOFR = 5.31
BASIS = -0.03  # SOFR - EFFR, percent


def synthetic_sofr():
    """SOFR fixed at FIXED_SOFR on every business day through the session, published the next Fed day."""
    days = sofr_business_days("2024-01-02", SESSION + DAY)
    return pd.DataFrame({"date": days, "value": FIXED_SOFR, "published": days.map(lambda d: d + US_BDAY)})


def zq_path():
    """The ZQ step path as the panel stores it: first unknown day, then each effective date."""
    return pd.Series([RATE_NOW, AFTER], index=pd.DatetimeIndex([SESSION, CUT]))


def market_rate(shape, start, end, fixings):
    """What the contract settles to if SOFR is the fixings where known, the ZQ path + BASIS after."""
    days = pd.date_range(start, end - DAY)
    known = fixings[fixings["published"] < SESSION + DAY].set_index("date")["value"]
    first_future = sofr_business_days(known.index[-1] + DAY, known.index[-1] + pd.Timedelta(days=10))[0]
    forward = pd.Series(RATE_NOW, index=days).where(days < CUT, AFTER) + BASIS
    fixed = known.reindex(pd.date_range(known.index[0], days[-1])).ffill().reindex(days)
    daily = fixed.where(days < first_future, forward)
    lookback = start - pd.Timedelta(days=7)
    fixing_dates = known.loc[lookback:].index.append(sofr_business_days(max(lookback, first_future), end))
    return settlement_rate(shape, start, end, daily, fixing_dates)


@pytest.mark.parametrize("shape, expirations", [
    ("month", ["2024-06-28", "2024-07-31", "2024-08-30", "2024-09-30", "2024-10-31"]),
    ("imm_quarter", ["2024-09-17", "2024-12-17"]),  # SR3M4, SR3U4
])
def test_basis_put_into_the_settles_comes_back_out(shape, expirations):
    fixings = synthetic_sofr()
    through = pd.Timestamp("2025-01-01")
    rows = []
    for exp in expirations:
        start, end = reference_period(shape, exp)
        rows.append({"contract": exp, "expiration": pd.Timestamp(exp),
                     "value": 100.0 - market_rate(shape, start, end, fixings)})
    got = implied_basis(SESSION, zq_path(), through, pd.DataFrame(rows), fixings, shape)

    assert len(got) == len(expirations)
    # The front month is half fixed at a SOFR that is not path + basis; the solve
    # must use those days as known and still find the basis on the rest.
    assert got["basis_bp"].to_numpy() == pytest.approx(BASIS * 100.0, abs=1e-6)
    ahead = next_period(got.assign(session=SESSION))
    assert ahead.loc[SESSION, "unknown_frac"] == 1.0
    assert ahead.loc[SESSION, "start"] > SESSION
