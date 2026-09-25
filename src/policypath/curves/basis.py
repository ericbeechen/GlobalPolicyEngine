"""SOFR futures against the ZQ path: the SOFR - EFFR basis the two markets imply together.

For each session, take the EFFR path solved from ZQ and, for every SOFR
contract whose reference period lies inside its horizon, find the constant
spread b that reprices the settle when SOFR is

    realized SOFR on days already fixed, ZQ-implied EFFR + b on the rest,

combined over the reference period as the contract settles: averaged over the
calendar month for SR1, compounded over the IMM quarter for SR3. b is then the
market-implied SOFR - EFFR basis over the period's unknown days. If the ZQ path
is right, b is smooth, sits close to the basis that is later realized, and
moves where funding pressure is known to move it. If it jumps around at random,
one of the two extractions is wrong.

SR1 is the like-for-like check: it averages SOFR over the month exactly as ZQ
averages EFFR, so its basis is read month by month. Futures/forward convexity is
ignored: inside the ~13-month ZQ horizon it is well under a basis point.
"""

import numpy as np
import pandas as pd
from policypath.calendars import known_meetings
from policypath.curves.futures import reference_period, settlement_rate
from policypath.curves.helpers import sofr_business_days
from policypath.panel import settles_on

DAY = pd.Timedelta(days=1)


def path_daily(first_unknown, rate_now, steps, end):
    """The ZQ step path on every day from `first_unknown` to `end` (exclusive).

    `steps` maps effective date -> rate in force from that date.
    """
    days = pd.date_range(first_unknown, end - DAY, freq="D")
    rate = pd.Series(rate_now, index=days)
    for eff, r in sorted(steps.items()):
        rate[rate.index >= eff] = r
    return rate


def implied_basis(session, path, through, settles, sofr, shape):
    """The implied basis for each contract whose reference period ends by `through`.

    `path` is the ZQ path on the session (a Series of rates indexed by the day
    each takes effect, first entry the first unknown day), `settles` the
    session's SOFR futures settles (contract, expiration, value), `shape` what
    they settle to (``month`` for SR1, ``imm_quarter`` for SR3), `sofr` SOFR
    fixings (date, value, published). Only fixings published by the session are used.
    """
    session = pd.Timestamp(session)
    known = sofr[sofr["published"] < session.normalize() + DAY].set_index("date")["value"].sort_index()
    last = known.index[-1]
    first_future = sofr_business_days(last + DAY, last + pd.Timedelta(days=10))[0]
    effr = path_daily(path.index[0], path.iloc[0], path.iloc[1:].to_dict(), through)
    rows = []
    for contract, expiration, price in settles[["contract", "expiration", "value"]].itertuples(index=False):
        start, end = reference_period(shape, expiration)
        if end <= session or end > through or start < known.index[0]:
            continue
        # A week back, so a period opening on a holiday finds the fixing covering it.
        lookback = start - pd.Timedelta(days=7)
        fixing_dates = known.loc[lookback:first_future - DAY].index.append(
            sofr_business_days(max(lookback, first_future), end))
        days = pd.date_range(start, end - DAY, freq="D")
        forward = effr.reindex(days).bfill().to_numpy()
        fixed = known.reindex(pd.date_range(known.index[0], first_future - DAY)).ffill()
        fixed = fixed.reindex(days).to_numpy()
        unknown = days >= first_future

        def rate(b):
            daily = pd.Series(np.where(unknown, forward + b, fixed), index=days)
            return settlement_rate(shape, start, end, daily, fixing_dates)

        target = 100.0 - price
        b = 0.0
        for _ in range(4):  # Newton; the map is linear in b for a month, all but linear for a quarter
            f0, f1 = rate(b), rate(b + 0.01)
            b -= (f0 - target) / ((f1 - f0) / 0.01)
        rows.append({"contract": contract, "start": start, "end": end,
                     "unknown_start": max(start, first_future),
                     "unknown_frac": unknown.mean(), "basis_bp": b * 100.0})
    return pd.DataFrame(rows)


def realized_basis(start, end, sofr, effr):
    """SOFR - EFFR averaged over the calendar days of [start, end), in bp, ex post."""
    days = pd.date_range(start, end - DAY, freq="D")
    s = sofr.set_index("date")["value"].sort_index().reindex(days, method="ffill")
    e = effr.set_index("date")["value"].sort_index().reindex(days, method="ffill")
    return float((s - e).mean() * 100.0)


def trailing_basis(sofr, effr, as_of, n=20):
    """Mean SOFR - EFFR over the last `n` fixings published by `as_of`, in bp: the ex-ante yardstick."""
    cutoff = pd.Timestamp(as_of).normalize() + DAY
    s = sofr[sofr["published"] < cutoff].set_index("date")["value"]
    e = effr[effr["published"] < cutoff].set_index("date")["value"]
    return float((s - e).dropna().tail(n).mean() * 100.0)


def next_period(basis):
    """Per session, the first contract whose reference period lies wholly after the session's fixings.

    A period already partly fixed has few unknown days, and its implied basis
    carries price noise times the inverse of that fraction, so the summary uses
    the first period that is all forward.
    """
    ahead = basis[basis["unknown_frac"] == 1.0]
    return ahead.sort_values("start").groupby("session").head(1).set_index("session").sort_index()


def cover_end(meetings, session_meetings):
    """The last day the stored path is complete for: the day before the next
    meeting after the ones the panel kept, or the end of the horizon month."""
    last_kept = session_meetings["effective_date"].max()
    later = meetings.loc[meetings["effective_date"] > last_kept, "effective_date"]
    horizon_end = (last_kept.to_period("M") + 1).end_time.normalize() + DAY
    return min(later.min(), horizon_end) if len(later) else horizon_end


def build(sessions, meetings_panel, meetings, futures_log, sofr, effr, shape):
    """Implied basis for every solved session. One row per session and SOFR contract.

    `futures_log` is the cached settle log of the SOFR futures root, `shape`
    what it settles to.

    ``realized_bp`` is the SOFR - EFFR basis that later printed over the same
    unknown days, where they have all printed: an ex-post yardstick, never an input.
    """
    ok = sessions[sessions["error"].isna()].set_index("session")
    by_session = dict(tuple(meetings_panel.groupby("session")))
    by_date = dict(tuple(futures_log.groupby("date")))
    frames = []
    for day, row in ok.iterrows():
        if day not in by_date:
            continue
        m = by_session[day]
        path = pd.Series([row["rate_now"], *m["rate"]],
                         index=pd.DatetimeIndex([row["first_unknown"], *m["effective_date"]]))
        through = cover_end(known_meetings(meetings, day), m)
        b = implied_basis(day, path, through, settles_on(by_date[day], day), sofr, shape)
        frames.append(b.assign(session=day))
    out = pd.concat(frames, ignore_index=True)
    printed = min(sofr["date"].max(), effr["date"].max())
    out["realized_bp"] = [realized_basis(a, e, sofr, effr) if e - DAY <= printed else np.nan
                          for a, e in zip(out["unknown_start"], out["end"])]
    return out
