"""The market-implied policy path: one interface, one backend per market mechanic.

Monthly-average futures and a fitted forward curve are different mechanics,
not different parameters, so each is its own extractor:

- `FuturesStripExtractor`: contracts that settle on the arithmetic average of
  the overnight rate over a calendar month (ZQ). Solved jointly by least
  squares with realized days substituted (`curves/policy_path.py`).
- `ForwardCurveExtractor`: a fitted spot OIS curve at fixed maturities (the
  Bank of England's). The rate between meetings is the forward over that
  window, exact from two spot rates (`curves/forward.py`).

Both return a `PolicyPath`: the rate in force now and after each of the next
n meetings, by effective date, in the market's overnight rate. The config's
``market.extractor`` picks one; nothing downstream can tell which produced a path.

On session t the inputs are the market data of t as it stood by the end of
the next business day (CME's final settle; the Bank's curve, published by noon
the next business day), and the fixings published by the end of t. A path's
``published`` is when the last input it used arrived.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from policypath import config
from policypath.calendars import BDAYS, US_BDAY, known_daily, known_meetings
from policypath.curves.forward import forward_path
from policypath.curves.policy_path import implied_path
from policypath.sources import cache

DAY = pd.Timedelta(days=1)


class ShortStrip(ValueError):
    """The market data, or the meeting calendar, does not reach the horizon."""


@dataclass(frozen=True)
class PolicyPath:
    """One solved session: the path at each upcoming meeting, and how it was got."""
    summary: dict
    meetings: pd.DataFrame


# Kept under its old name: tests and scripts written against the ZQ panel use it.
Session = PolicyPath


def upcoming_meetings(meetings, session, n):
    """The next `n` meetings after `session`, from the calendar as it stood on the session."""
    known = known_meetings(meetings, session)
    ahead = known[known["effective_date"] > pd.Timestamp(session)].sort_values("effective_date")
    ahead = ahead.reset_index(drop=True)
    if len(ahead) < n:
        raise ShortStrip(f"the meeting calendar has only {len(ahead)} meetings after {pd.Timestamp(session).date()}")
    return ahead


def _rows(upcoming, path):
    """The per-meeting frame every extractor returns: k, the calendar's columns, rate, step_bp, cum_bp."""
    steps = path.diff() * 100.0
    rows = upcoming.assign(
        k=np.arange(1, len(upcoming) + 1),
        rate=path.reindex(upcoming["effective_date"]).to_numpy(),
        step_bp=steps.reindex(upcoming["effective_date"]).to_numpy(),
    )
    rows["cum_bp"] = (rows["rate"] - path.iloc[0]) * 100.0
    return rows


def _as_published(rows, keys, session, bday):
    """Each key's latest vintage published by the end of the business day after `session`."""
    cutoff = pd.Timestamp(session) + bday + DAY
    rows = rows[rows["published"] < cutoff]
    return rows.sort_values(["published", "retrieved"]).drop_duplicates(keys, keep="last")


# ---- monthly-average futures (ZQ) ------------------------------------------------

def settles_on(vintages, session, bday=US_BDAY):
    """Each contract's settle for `session`, as it stood by the end of the next business day."""
    return _as_published(vintages[vintages["date"] == session], "contract", session, bday)


def monthly_strip(settles):
    """Implied average rate by contract month, for monthly-average contracts (ZQ, SR1)."""
    month = settles["expiration"].dt.to_period("M")
    return pd.Series(100.0 - settles["value"].to_numpy(),
                     index=pd.PeriodIndex(month, name="month")).sort_index()


def solve_session(session, strip, meetings, fixings, n_meetings, min_forward_days=0,
                  min_regime_days=0, bday=US_BDAY):
    """The implied path over the next `n_meetings` meetings after `session`, from a monthly strip.

    `strip` is the session's monthly strip, `fixings` the overnight fixings
    with their publication dates, `meetings` the whole calendar: only the
    meetings known on the session are pillars. Raises `ShortStrip` if the strip
    or the calendar stops short, ValueError if the solve is underdetermined.
    """
    session = pd.Timestamp(session)
    known = known_meetings(meetings, session)
    upcoming = upcoming_meetings(meetings, session, n_meetings).head(n_meetings)
    # One month past the last meeting's, so its new rate is pinned by a whole contract.
    horizon = upcoming["effective_date"].iloc[-1].to_period("M") + 1
    front = session.to_period("M")
    strip = strip[(strip.index >= front) & (strip.index <= horizon)]
    if strip.empty or strip.index.max() < horizon:
        last = strip.index.max() if len(strip) else None
        raise ShortStrip(f"strip ends {last}, short of {horizon} needed for {n_meetings} meetings")

    realized = known_daily(fixings, session, bday)
    path = implied_path(strip, known["effective_date"], realized, min_forward_days, min_regime_days)
    first_unknown = path.index[0]
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
    return PolicyPath(summary, _rows(upcoming, path))


class FuturesStripExtractor:
    """Monthly-average futures on the overnight rate, from the cache (config: ``policy_futures``)."""

    def __init__(self, ccy, cfg, root=cache.CACHE_DIR):
        self.spec, self.bday = cfg["path"], BDAYS[cfg["calendar"]]
        self.calendar = config.meetings(ccy)
        vintages = cache.log("databento", cfg["policy_futures"], ccy, root)
        self.fixings = cache.log(cfg["overnight"]["source"], cfg["overnight"]["series"], ccy, root)
        self.dates = pd.DatetimeIndex(sorted(vintages["date"].unique()))
        self.by_date = dict(tuple(vintages.groupby("date")))

    def sessions(self):
        return self.dates

    def published(self, day):
        return settles_on(self.by_date[day], day, self.bday)["published"].max()

    def session(self, day):
        settles = settles_on(self.by_date[day], day, self.bday)
        return solve_session(day, monthly_strip(settles), self.calendar, cache.view(self.fixings, day),
                             self.spec["n_meetings"], self.spec["min_forward_days"],
                             self.spec["min_regime_days"], self.bday)


# ---- a fitted spot OIS curve (the Bank of England's) -------------------------------

def rate_in_force(fixings, policy, session, bday=US_BDAY):
    """The overnight rate in force on `session`: the last fixing, moved by any policy change since.

    On session t the last published fixing is t - 1's. Where a decision takes
    effect on the day it is announced (Bank Rate, at noon), a decision on t
    itself is in force at t's close but in no fixing yet, so the policy rate's
    change from the fixing's date to t is added. `policy` is the policy rate's
    log (``date``, ``value``, ``published``), or None to take the fixing as it is.
    Returns (rate, last fixing).
    """
    realized = known_daily(fixings, session, bday)
    last = realized.iloc[-1]
    if policy is None:
        return last, last
    p = policy[policy["published"] < pd.Timestamp(session).normalize() + DAY]
    p = p.sort_values(["date", "published"]).drop_duplicates("date", keep="last").set_index("date")["value"]
    now, then = p[p.index <= session].iloc[-1], p[p.index <= realized.index[-1]].iloc[-1]
    return last + (now - then), last


def solve_curve(session, spot, meetings, fixings, n_meetings, min_regime_days=0, bday=US_BDAY,
                tail_days=45, policy=None, pin_always=False):
    """The path over the next `n_meetings` meetings after `session`, from that day's spot curve.

    The last regime runs to the meeting after the n-th if the calendar has
    it, else `tail_days` past the n-th: either way, a whole window. A pinned
    first regime is at `rate_in_force`.
    """
    session = pd.Timestamp(session)
    ahead = upcoming_meetings(meetings, session, n_meetings)
    upcoming = ahead.head(n_meetings)
    last = upcoming["effective_date"].iloc[-1]
    end = ahead["effective_date"].iloc[n_meetings] if len(ahead) > n_meetings else last + pd.Timedelta(days=tail_days)
    in_force, last_fixing = rate_in_force(fixings, policy, session, bday)
    try:
        path = forward_path(spot, session, upcoming["effective_date"], end, in_force, min_regime_days, pin_always)
    except ValueError as exc:
        raise ShortStrip(str(exc)) from exc
    summary = {
        "n_nodes": path.attrs["nodes"],
        "last_tenor": int(spot.dropna().index.max()),
        "first_unknown": session,
        "last_fixing": last_fixing,
        "rate_in_force": in_force,
        "rate_now": path.iloc[0],
        "pinned": path.attrs["pinned"],
        "residual_bp": path.attrs["residual_bp"],
    }
    return PolicyPath(summary, _rows(upcoming, path))


class ForwardCurveExtractor:
    """A spot OIS curve by maturity in months, from the cache (config: ``market.curve``)."""

    def __init__(self, ccy, cfg, root=cache.CACHE_DIR):
        self.spec, self.bday = cfg["path"], BDAYS[cfg["calendar"]]
        self.calendar = config.meetings(ccy)
        curve = cfg["market"]["curve"]
        rows = cache.log(curve["source"], curve["series"], ccy, root)
        self.fixings = cache.log(cfg["overnight"]["source"], cfg["overnight"]["series"], ccy, root)
        policy = cfg["market"].get("policy_rate")
        self.policy = None if policy is None else cache.log(policy["source"], policy["series"], ccy, root)
        self.pin_always = cfg["market"].get("pin_first_regime", False)
        self.dates = pd.DatetimeIndex(sorted(rows["date"].unique()))
        self.by_date = dict(tuple(rows.groupby("date")))

    def sessions(self):
        return self.dates

    def _curve(self, day):
        return _as_published(self.by_date[day], "tenor", day, self.bday)

    def published(self, day):
        return self._curve(day)["published"].max()

    def session(self, day):
        rows = self._curve(day)
        spot = rows.set_index("tenor")["value"].sort_index()
        return solve_curve(day, spot, self.calendar, cache.view(self.fixings, day), self.spec["n_meetings"],
                           self.spec["min_regime_days"], self.bday, policy=self.policy,
                           pin_always=self.pin_always)


EXTRACTORS = {"futures_strip": FuturesStripExtractor, "forward_curve": ForwardCurveExtractor}


def extractor(ccy, root=cache.CACHE_DIR):
    """The backend the config names for `ccy`, loaded from the cache."""
    cfg = config.currency(ccy)
    return EXTRACTORS[cfg["market"]["extractor"]](ccy, cfg, root)


def path(date, ccy, root=cache.CACHE_DIR):
    """The market-implied path on the last session on or before `date`, for any configured currency.

    ``summary["session"]`` says which session that was.
    """
    ext = extractor(ccy, root)
    days = ext.sessions()
    days = days[days <= pd.Timestamp(date)]
    if days.empty:
        raise ValueError(f"no {ccy} session on or before {pd.Timestamp(date).date()}")
    result = ext.session(days[-1])
    return PolicyPath({"session": days[-1], "published": ext.published(days[-1]), **result.summary},
                      result.meetings)
