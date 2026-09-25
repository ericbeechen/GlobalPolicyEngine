"""The implied policy path on every session, solved from the cache.

For session t the inputs are

- the settles of t as they stood by the end of the next business day. CME's
  final settle arrives that evening, or on the Sunday for a Friday session, so
  this is the final; the preliminary is what a read at t's close would see.
- the fixings published by the end of t, i.e. through t - 1. The market
  settling on t did not know t's own fixing, so using it would put look-ahead
  into the front contract on every date.

Each session's ``published`` is when the last settle it used arrived. The path
is knowable from then, not from t's close.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from policypath import config
from policypath.calendars import US_BDAY, known_daily, known_meetings
from policypath.curves.policy_path import implied_path
from policypath.sources import cache

DAY = pd.Timedelta(days=1)


class ShortStrip(ValueError):
    """The strip, or the meeting calendar, does not reach the horizon."""


@dataclass(frozen=True)
class Session:
    """One solved session: the path at each upcoming meeting, and how it was got."""
    summary: dict
    meetings: pd.DataFrame


def settles_on(vintages, session, bday=US_BDAY):
    """Each contract's settle for `session`, as it stood by the end of the next business day."""
    cutoff = pd.Timestamp(session) + bday + DAY
    day = vintages[(vintages["date"] == session) & (vintages["published"] < cutoff)]
    return day.sort_values(["published", "retrieved"]).drop_duplicates("contract", keep="last")


def monthly_strip(settles):
    """Implied average rate by contract month, for monthly-average contracts (ZQ, SR1)."""
    month = settles["expiration"].dt.to_period("M")
    return pd.Series(100.0 - settles["value"].to_numpy(),
                     index=pd.PeriodIndex(month, name="month")).sort_index()


def solve_session(session, strip, meetings, fixings, n_meetings, min_forward_days=0,
                  min_regime_days=0):
    """The implied path over the next `n_meetings` meetings after `session`.

    `strip` is the session's monthly strip, `fixings` the overnight fixings
    with their publication dates, `meetings` the whole calendar: only the
    meetings known on the session are pillars. Raises `ShortStrip` if the strip
    or the calendar stops short, ValueError if the solve is underdetermined.
    """
    session = pd.Timestamp(session)
    meetings = known_meetings(meetings, session)
    upcoming = (meetings[meetings["effective_date"] > session]
                .sort_values("effective_date").head(n_meetings).reset_index(drop=True))
    if len(upcoming) < n_meetings:
        raise ShortStrip(f"the meeting calendar has only {len(upcoming)} meetings after {session.date()}")
    # One month past the last meeting's, so its new rate is pinned by a whole contract.
    horizon = upcoming["effective_date"].iloc[-1].to_period("M") + 1
    front = session.to_period("M")
    strip = strip[(strip.index >= front) & (strip.index <= horizon)]
    if strip.empty or strip.index.max() < horizon:
        last = strip.index.max() if len(strip) else None
        raise ShortStrip(f"strip ends {last}, short of {horizon} needed for {n_meetings} meetings")

    realized = known_daily(fixings, session)
    path = implied_path(strip, meetings["effective_date"], realized,
                        min_forward_days, min_regime_days)
    steps = path.diff() * 100.0
    first_unknown = path.index[0]

    rows = upcoming.assign(
        k=np.arange(1, n_meetings + 1),
        rate=path.reindex(upcoming["effective_date"]).to_numpy(),
        step_bp=steps.reindex(upcoming["effective_date"]).to_numpy(),
    )
    rows["cum_bp"] = (rows["rate"] - path.iloc[0]) * 100.0
    front_left = (front.end_time.normalize() - max(first_unknown, front.start_time)).days + 1
    summary = {
        "n_contracts": len(strip),
        "front_month": str(strip.index[0]),
        "last_month": str(strip.index[-1]),
        "front_is_current": strip.index[0] == front,
        "first_unknown": first_unknown,
        "front_days_left": front_left if strip.index[0] == front else np.nan,
        "last_fixing": realized.iloc[-1],
        "rate_now": path.iloc[0],
        "front_dropped": path.attrs["front_dropped"],
        "pinned": path.attrs["pinned"],
        "residual_bp": path.attrs["residual_bp"],
    }
    return Session(summary, rows)


def build(ccy, start=None, end=None, **overrides):
    """Solve every session in the cache. Returns (sessions, meetings), both long frames.

    `sessions` has one row per trade date, solved or not; a failed solve keeps
    its row with the reason in ``error``. `meetings` has one row per session
    and upcoming meeting. `overrides` replace keys of the config's ``path`` block.
    """
    cfg = config.currency(ccy)
    spec = {**cfg["path"], **overrides}
    cal = config.meetings(ccy)
    vintages = cache.log("databento", cfg["policy_futures"], ccy)
    fixings_log = cache.log(cfg["overnight"]["source"], cfg["overnight"]["series"], ccy)

    dates = pd.DatetimeIndex(sorted(vintages["date"].unique()))
    if start is not None:
        dates = dates[dates >= pd.Timestamp(start)]
    if end is not None:
        dates = dates[dates <= pd.Timestamp(end)]
    vintages = vintages[vintages["date"].isin(dates)]
    by_date = dict(tuple(vintages.groupby("date")))

    summaries, frames = [], []
    for day in dates:
        settles = settles_on(by_date[day], day)
        row = {"session": day, "published": settles["published"].max(), "error": None}
        try:
            fixings = cache.view(fixings_log, day)
            result = solve_session(day, monthly_strip(settles), cal, fixings, spec["n_meetings"],
                                   spec["min_forward_days"], spec["min_regime_days"])
        except (ValueError, np.linalg.LinAlgError) as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        else:
            row.update(result.summary)
            frames.append(result.meetings.assign(session=day))
        summaries.append(row)

    sessions = pd.DataFrame(summaries)
    meetings = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    cols = ["session", "k", "announcement_date", "effective_date", "scheduled",
            "rate", "step_bp", "cum_bp"]
    return sessions, meetings[cols] if len(meetings) else meetings
