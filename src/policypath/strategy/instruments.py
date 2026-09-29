"""The instruments the sleeves trade, marked on every session: one leg type per config ``instrument:`` name.

A leg is built once for a sleeve's list of sessions, from wide frames of
each series' prints by date, never session by session from the cache. Its
frame has two halves, one row per session t:

- what a trade at t's close puts on (the instrument **selected** at t): its id,
  its rate on t, its rate ``carry.horizon_days`` further down the day's curve
  (for carry and roll ahead, `strategy/carry.py`), its native DV01, and for
  funded legs its duration and funding rate;
- what earns the P&L credited to t: the instrument **held** from the previous
  sleeve session p into t, which is the one selected at p. Its rate on p
  (``held_p``), on t at its maturity then (``held_t``), and on p's curve at
  that same maturity (``rolled``), the days between (``delta``), the FX, and
  what the full revaluation in `strategy/carry.py` needs.

Every carry, roll and funding formula is in `strategy/carry.py`; this module
supplies rates, maturities and prices, and the instrument's own valuation
(`par_duration`, `bond_price`).

**Marks.** A leg's curve on day d is the print dated d as it stood at the end
of the ``lag``-th business day after d: `market._as_published`, done for every
date at once (`prints`). The lag is the series' own under ``sources.daily``,
or for a curve or futures root ``market.final_after_bdays`` (the Bank's curves
and CME's final settles both arrive the next business day). A session with no
print dated on it (Columbus Day: CME open, the bond market shut) carries the
last print before it, whole, not node by node: its rate change is 0, carry
still accrues, and ``stale`` counts it. A mark is never ``cache.read(as_of=t)``:
that returns the print dated t - 1 and books the move between decision and
execution.

**Ids** are unique in time. A futures leg is its contract's expiration month
(ZQF1 is both January 2011 and January 2021), a forward its window, a par leg
its tenor.

Types (config ``expression.<e>.instrument``):

- ``futures_month``: the monthly-average future (ZQ) whose month is the first
  calendar month starting on or after the k-th meeting's effective date, so it
  settles on that meeting's regime (and the next one's, where a meeting falls
  inside the month: ``regimes`` has the day weights). Rate = 100 - price. Its
  maturity is the days to the month's midpoint; the day's strip is linear in it
  between months and flat beyond. DV01 per contract = notional x accrual days /
  year days x 1bp, from ``contracts``: never typed.
- ``curve_forward``: the forward over [E_k, E_k+1) off the day's fitted spot
  curve, with -log P linear between nodes (`curves.forward.log_discount`, the
  path's own helper, so for k >= 2 it is the path panel's rate at k exactly).
  E_k+1 is the next meeting in the panel; past the last, the calendar's next
  known meeting, else E_k + ``path.tail_days``, as the path closes its last
  regime. A rolled window (a day's roll, or the carry horizon's) that starts
  below the curve's first node reads the rate in force there, as
  ``pin_first_regime`` does. DV01 per unit notional = accrual x DF(E_k+1) x 1bp.
- ``par_yield``: constant-maturity par yields (Treasury CMT), linear in maturity
  between tenors. A leg struck at p has maturity T - delta/365 at t. Duration
  A = [1 - (1 + y/f)^(-fT)] / y (T at y = 0), DV01 = A x 1bp.
- ``par_from_spot``: par yields built from a fitted spot curve (the Bank's
  gilt curve): DF(t) = exp(-s(t) t) with s t linear between the day's nodes
  and 0 at t = 0, so a missing 0.5y node is interpolated, never a crash and
  never a method switch; c(T) = f (1 - DF(T)) / sum DF(i/f) on the half-year
  grid; then as ``par_yield``, linear between grid points.

A leg outside the book currency converts at spot S (book currency per unit of
the leg's): a unit leg (+1 book DV01) holds 1/S_p of its own DV01 from p, so
its P&L converts at ``fx = S_t / S_p``. Book-currency legs have S = 1. A spot
outside ``expression.fx.plausible`` is refused (`plausible_spot`): the full
revaluation values the leg at the same S, so it cannot see a quote read
upside down, and a range can.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from policypath.calendars import BDAYS, known_meetings
from policypath.curves.forward import log_discount, window_rates
from policypath.sources import cache

DAY = pd.Timedelta(days=1)
YEAR = 365.0          # a par leg's maturity runs down, and its coupon accrues, Act/365
FUNDED = ("par_yield", "par_from_spot")


@dataclass
class Leg:
    """One leg's marks on a sleeve's sessions.

    `frame` is indexed by session (columns in the module docstring); `kind` is
    the instrument type; `regimes` (outright legs) is session x k, the share of
    the selected instrument's days in each meeting's regime; `f` the coupons a
    year of a par leg; `contract` the futures contract's spec.
    """
    kind: str
    ccy: str
    frame: pd.DataFrame
    regimes: pd.DataFrame | None = None
    f: int | None = None
    contract: dict | None = None

    @property
    def funded(self):
        return self.kind in FUNDED


# ---- marks -------------------------------------------------------------------------

def prints(rows, keys, lag, bday):
    """Each date's print as it stood at the end of the `lag`-th business day after it, for every date at once.

    Row for row what `market._as_published` keeps for one date, keyed by
    (date, *keys): the latest vintage published by then.
    """
    dates = pd.DatetimeIndex(rows["date"].unique())
    cutoff = pd.Series([d + lag * bday + DAY for d in dates], index=dates)
    rows = rows[rows["published"] < rows["date"].map(cutoff)]
    return rows.sort_values(["published", "retrieved"]).drop_duplicates(["date", *keys], keep="last")


def asof(frame, days):
    """Each session's print: the whole row of `frame` dated on or before it. Returns (rows, stale).

    `stale` is True where the row is dated before the session, or there is none.
    """
    days = pd.DatetimeIndex(days)
    at = frame.index.searchsorted(days, side="right") - 1
    rows = frame.iloc[np.maximum(at, 0)].set_axis(days)
    rows.iloc[np.flatnonzero(at < 0)] = np.nan
    stale = pd.Series((at < 0) | (frame.index[np.maximum(at, 0)] != days), index=days, name="stale")
    return rows, stale


class Marks:
    """One currency's prints by date, from the cache: each series read, filtered and pivoted once, then shared."""

    def __init__(self, ccy, cfg, root=cache.CACHE_DIR):
        self.ccy, self.cfg, self.root = ccy, cfg, root
        self.bday = BDAYS[cfg["calendar"]]
        self._memo = {}

    def lag(self, source, series):
        """Business days from a print's date to its final: the series' own daily lag, else the market's."""
        daily = self.cfg["sources"]["daily"].get(source, {})
        return daily[series] if series in daily else self.cfg["market"]["final_after_bdays"]

    def _once(self, key, make):
        if key not in self._memo:
            self._memo[key] = make()
        return self._memo[key]

    def _rows(self, source, series, keys):
        rows = cache.log(source, series, self.ccy, self.root)
        return prints(rows, keys, self.lag(source, series), self.bday)

    def daily(self, source, series):
        """One daily series by date."""
        def make():
            return self._rows(source, series, []).set_index("date")["value"].sort_index()
        return self._once(("daily", source, series), make)

    def curve(self, source, series):
        """A fitted curve, date x tenor (months)."""
        def make():
            return self._rows(source, series, ["tenor"]).pivot(index="date", columns="tenor", values="value").sort_index()
        return self._once(("curve", source, series), make)

    def futures(self, source, series):
        """Settle prices, date x contract month (a monthly Period: the expiration's month, unique in time)."""
        def make():
            rows = self._rows(source, series, ["contract"])
            rows = rows.assign(month=rows["expiration"].dt.to_period("M"))
            return rows.pivot(index="date", columns="month", values="value").sort_index()
        return self._once(("futures", source, series), make)

    def tenors(self, source, series):
        """Par yields, date x tenor (years), from a ``{tenor: series}`` map."""
        def make():
            return pd.concat({float(t): self.daily(source, s) for t, s in series.items()}, axis=1).sort_index()
        return self._once(("tenors", source, tuple(series.items())), make)

    def par(self, source, series, f, longest):
        """Par yields on the 1/f-year grid to `longest` years, date x maturity (years), from a spot curve."""
        return self._once(("par", source, series, f, longest),
                          lambda: par_curve(self.curve(source, series), f, longest))


# ---- valuation ---------------------------------------------------------------------

def par_duration(y, T, f):
    """Modified duration of a par bond, years: [1 - (1 + y/f)^(-fT)] / y, with y in percent; T where |y| < 1e-10.

    Also the annuity a bond at yield y pays its coupon on, so its DV01 per unit
    notional is A x 1bp. Works elementwise on arrays.
    """
    y = np.asarray(y, dtype=float) / 100.0
    small = np.abs(y) < 1e-10
    safe = np.where(small, 1.0, y)
    return np.where(small, T, (1.0 - (1.0 + safe / f) ** (-f * np.asarray(T, dtype=float))) / safe)


def bond_price(c, y, tau, f):
    """Clean price per 1 of face of a bond with coupon c (percent, paid f times a year), at yield y, tau years left.

    c x A(y, tau) + (1 + y/f)^(-f tau), A as `par_duration`: equal to 1 at y = c
    for any tau (a bond struck at par has no accrued interest to track), and its
    slope in y there is -A(c, tau).
    """
    c = np.asarray(c, dtype=float)
    return c / 100.0 * par_duration(y, tau, f) + (1.0 + np.asarray(y, dtype=float) / 100.0 / f) ** (-f * tau)


def par_curve(spot, f, longest):
    """Par yields, percent, at maturities 1/f, 2/f, .. `longest` years, from each row's spot curve (months).

    c(T) = f (1 - DF(T)) / sum DF(i/f), DF from `curves.forward.log_discount`
    (s t linear between the row's nodes, 0 at 0). A maturity past a row's last
    node is NaN.
    """
    grid = np.arange(1, int(round(longest * f)) + 1) / f
    out = np.full((len(spot), len(grid)), np.nan)
    tenors = spot.columns
    for i, row in enumerate(spot.to_numpy(dtype=float)):
        ok = ~np.isnan(row)
        if not ok.any():
            continue
        nodes, log_df = log_discount(pd.Series(row[ok], index=tenors[ok]))
        df = np.exp(-np.interp(grid, nodes, log_df))
        df[grid > nodes[-1]] = np.nan
        out[i] = f * (1.0 - df) / np.cumsum(df) * 100.0
    return pd.DataFrame(out, index=spot.index, columns=grid)


# ---- the legs, from each session's print --------------------------------------------

def _interp(x, xs, ys):
    """`ys` at `x`, linear in `xs` between points and flat beyond, over the non-NaN points; NaN if none."""
    ok = ~np.isnan(ys)
    return np.interp(x, xs[ok], ys[ok]) if ok.any() else np.nan


def _held(frame, days):
    """The held half's bookkeeping: previous session's id and rate, and the calendar days since it."""
    frame["delta"] = np.r_[np.nan, np.diff(days.to_numpy()) / np.timedelta64(1, "D")]
    frame["held_id"] = frame["id"].shift(1)
    frame["held_p"] = frame["rate"].shift(1)
    return frame


def contract_dv01(contract):
    """DV01 of one futures contract, book currency per bp: notional x accrual days / year days x 1bp."""
    return contract["notional"] * contract["accrual"]["days"] / contract["accrual"]["year_days"] * 1e-4


def tick_values(contract):
    """Book currency per tick, front month and others: tick / 1bp x DV01 (CME quotes them rounded)."""
    return {k: t / 0.01 * contract_dv01(contract) for k, t in contract["tick"].items()}


def contract_month(effective):
    """The first calendar month starting on or after each effective date (a monthly Period)."""
    month = effective.dt.to_period("M")
    return month.where(effective.dt.day == 1, month + 1)


def month_regimes(days, months, effective):
    """Share of each month's days in each meeting's regime: session x k, from session x k effective dates.

    Regime k runs from E_k to E_k+1 (the last one without end).
    """
    start = months.dt.start_time.to_numpy()
    end = (months + 1).dt.start_time.to_numpy()
    E = effective.to_numpy(dtype="datetime64[ns]")
    after = np.concatenate([E[:, 1:], np.full((len(E), 1), np.datetime64("2262-01-01"))], axis=1)
    lo = np.maximum(E, start[:, None].astype("datetime64[ns]"))
    hi = np.minimum(after, end[:, None].astype("datetime64[ns]"))
    days_in = (hi - lo) / np.timedelta64(1, "D")
    total = (end - start) / np.timedelta64(1, "D")
    return pd.DataFrame(np.clip(days_in, 0, None) / total[:, None], index=days, columns=effective.columns)


def futures_month(days, prices, effective, k, h, dv01):
    """A futures-month leg on sessions `days`: the frame (module docstring) and its regime weights.

    `prices` is session x contract month settle prices (each session's print),
    `effective` session x k meeting effective dates, `h` the carry horizon in
    days, `dv01` one contract's (`contract_dv01`).
    """
    prices = prices.sort_index(axis=1)
    months = prices.columns
    mid = (months.start_time + pd.to_timedelta(months.days_in_month / 2.0, unit="D")).to_numpy()
    P = prices.to_numpy(dtype=float)
    Y = 100.0 - P
    selected = contract_month(effective[k])
    col = months.get_indexer(pd.PeriodIndex(selected))
    to_mid = (mid[None, :] - days.to_numpy()[:, None]) / np.timedelta64(1, "D")   # days to each month's midpoint
    n = len(days)
    rate, ahead, rolled, held_t, price_p, price_t = (np.full(n, np.nan) for _ in range(6))
    for i in range(n):
        c = col[i]
        if c >= 0:
            rate[i] = Y[i, c]
            ahead[i] = _interp(to_mid[i, c] - h, to_mid[i], Y[i])
        if i and col[i - 1] >= 0:
            m = col[i - 1]
            held_t[i], price_p[i], price_t[i] = Y[i, m], P[i - 1, m], P[i, m]
            rolled[i] = _interp(to_mid[i, m], to_mid[i - 1], Y[i - 1])
    ids = np.where(col >= 0, selected.astype(str).to_numpy(), None)
    frame = pd.DataFrame({"id": ids, "rate": rate, "ahead": ahead, "dv01": dv01}, index=days)
    frame = _held(frame, days).assign(rolled=rolled, held_t=held_t, price_p=price_p, price_t=price_t)
    return frame, month_regimes(days, selected, effective)


def _curve_reader(row, in_force):
    """-log P at maturity tau (years) off one day's spot curve.

    With ``pin`` a maturity below the curve's first node reads the rate in
    force instead (as ``pin_first_regime`` does): the curve says nothing about
    where inside its first node interval a move lands.
    """
    ok = ~np.isnan(row.to_numpy(dtype=float))
    if not ok.any():
        return lambda tau, pin=False: np.nan
    nodes, log_df = log_discount(row[ok])

    def at(tau, pin=False):
        return in_force / 100.0 * tau if pin and tau < nodes[1] else np.interp(tau, nodes, log_df)
    return at


def _forward(at, start_days, end_days, year_days, pin=False):
    """The forward, percent, between two maturities given in days, from a `_curve_reader`."""
    t = np.array([start_days, end_days]) / float(year_days)
    return window_rates(t, np.array([at(t[0], pin), at(t[1], pin)]))[0]


def curve_forward(days, curve, start, end, h, year_days, in_force):
    """A forward leg on sessions `days`: window [start, end) selected on each session, marked off `curve`.

    `curve` is session x tenor (months) spot rates, each session's print;
    `start`, `end` the window selected on each session; `in_force` the rate in
    force on each session (percent). The window's own marks read the curve as
    the path does, so for k >= 2 ``rate`` is the path's rate at k; a rolled
    window (``rolled``, ``ahead``) reads the rate in force below the first node.
    Returns the frame (module docstring).
    """
    n, curve = len(days), curve.sort_index(axis=1)
    readers = [_curve_reader(curve.iloc[i], in_force.iloc[i]) for i in range(n)]
    a = ((start - days.to_series()).dt.days).to_numpy()
    b = ((end - days.to_series()).dt.days).to_numpy()
    cols = {c: np.full(n, np.nan) for c in ["rate", "ahead", "dv01", "rolled", "held_t", "df_p", "df_t", "alpha"]}
    for i in range(n):
        at = readers[i]
        cols["rate"][i] = _forward(at, a[i], b[i], year_days)
        cols["ahead"][i] = _forward(at, a[i] - h, b[i] - h, year_days, pin=True)
        cols["dv01"][i] = (b[i] - a[i]) / float(year_days) * np.exp(-at(b[i] / float(year_days))) * 1e-4
        if i:
            step = (days[i] - days[i - 1]).days
            ta, tb = a[i - 1] - step, b[i - 1] - step       # the window held from p, in days from t
            cols["rolled"][i] = _forward(readers[i - 1], ta, tb, year_days, pin=True)
            cols["held_t"][i] = _forward(at, ta, tb, year_days)
            cols["df_p"][i] = np.exp(-readers[i - 1](b[i - 1] / float(year_days)))
            cols["df_t"][i] = np.exp(-at(tb / float(year_days)))
            cols["alpha"][i] = (b[i - 1] - a[i - 1]) / float(year_days)
    ids = [f"{s:%Y-%m-%d}/{e:%Y-%m-%d}" for s, e in zip(start, end)]
    frame = pd.DataFrame({"id": ids, "rate": cols["rate"], "ahead": cols["ahead"], "dv01": cols["dv01"]}, index=days)
    return _held(frame, days).assign(**{c: cols[c] for c in ["rolled", "held_t", "df_p", "df_t", "alpha"]})


def par_leg(days, curve, tenor, f, h, funding, basis):
    """A par leg of `tenor` years on sessions `days`, struck at each close and marked off `curve`.

    `curve` is session x maturity (years) par yields in percent, each
    session's print; `funding` the overnight rate in force (percent) and
    `basis` its day count. Returns the frame (module docstring).
    """
    curve = curve.sort_index(axis=1)
    mats, Y = curve.columns.to_numpy(dtype=float), curve.to_numpy(dtype=float)
    n = len(days)
    rate, ahead, rolled, held_t = (np.full(n, np.nan) for _ in range(4))
    step = np.r_[np.nan, np.diff(days.to_numpy()) / np.timedelta64(1, "D")]
    for i in range(n):
        rate[i] = _interp(tenor, mats, Y[i])
        ahead[i] = _interp(tenor - h / YEAR, mats, Y[i])
        if i:
            tau = tenor - step[i] / YEAR
            rolled[i], held_t[i] = _interp(tau, mats, Y[i - 1]), _interp(tau, mats, Y[i])
    duration = par_duration(rate, tenor, f)
    frame = pd.DataFrame({"id": f"{tenor:g}y", "rate": rate, "ahead": ahead, "dv01": duration * 1e-4,
                          "duration": duration, "funding": funding.to_numpy(dtype=float), "basis": float(basis)},
                         index=days)
    frame = _held(frame, days)
    return frame.assign(rolled=rolled, held_t=held_t, tau=tenor - frame["delta"] / YEAR, coupon=frame["held_p"],
                        duration_p=frame["duration"].shift(1), funding_p=frame["funding"].shift(1))


def with_fx(frame, spot):
    """`frame` with the spot rate on each session (book currency per unit), its value at p, and fx = S_t / S_p."""
    spot = np.asarray(spot, dtype=float)
    spot_p = np.r_[np.nan, spot[:-1]]
    return frame.assign(spot=spot, spot_p=spot_p, fx=spot / spot_p)


def plausible_spot(spot, days, fx):
    """Raise if any session's spot (book currency per unit) is outside ``fx["plausible"]``: an inverted quote."""
    lo, hi = fx["plausible"]
    bad = np.flatnonzero((spot < lo) | (spot > hi))
    if len(bad):
        i = bad[0]
        raise ValueError(f"{fx['series']} gives a spot of {spot[i]:.4f} on {days[i]:%Y-%m-%d}, outside "
                         f"expression.fx.plausible [{lo}, {hi}] on {len(bad)} sessions: is the quote upside down "
                         "(book_per_unit)?")


def native(leg, q):
    """Book-currency DV01 `q` per session in the selected instrument's own units: contracts, or notional in its currency."""
    return q / (leg.frame["spot"] * leg.frame["dv01"])


# ---- from the cache and the panels --------------------------------------------------

@dataclass(frozen=True)
class Inputs:
    """What every leg type reads: the currency's config and marks, the sleeve's sessions, the panels, the horizon."""
    cfg: dict
    marks: Marks
    days: pd.DatetimeIndex
    h: int
    sessions: pd.DataFrame
    meetings: pd.DataFrame
    calendar: pd.DataFrame | None

    def funding(self):
        """The overnight rate in force on each session: ``rate_in_force`` where the panel has it, else ``rate_now``."""
        col = "rate_in_force" if "rate_in_force" in self.sessions else "rate_now"
        return self.sessions.set_index("session")[col].reindex(self.days)

    def effective(self):
        """Session x k effective dates, from the meetings panel."""
        return self.meetings.pivot(index="session", columns="k", values="effective_date").reindex(self.days)


def _window_end(effective, k, cfg, calendar):
    """E_k+1 on each session: the panel's next meeting, or past the last the calendar's next, else + tail_days."""
    if k < effective.columns.max():
        return effective[k + 1]
    last = effective[k]
    out = []
    for day, e in last.items():
        known = known_meetings(calendar, day)["effective_date"]
        after = known[known > e]
        out.append(after.min() if len(after) else e + pd.Timedelta(days=cfg["path"]["tail_days"]))
    return pd.Series(out, index=last.index)


def _futures_leg(inp, x, k, tenor):
    contract = inp.cfg["contracts"][x["contract"]]
    prices, stale = asof(inp.marks.futures(inp.cfg["market"]["futures"]["source"], x["contract"]), inp.days)
    frame, regimes = futures_month(inp.days, prices, inp.effective(), k, inp.h, contract_dv01(contract))
    return frame, stale, {"regimes": regimes, "contract": contract}


def _forward_leg(inp, x, k, tenor):
    spec = inp.cfg["market"]["curve"]
    curve, stale = asof(inp.marks.curve(spec["source"], spec["series"]), inp.days)
    effective = inp.effective()
    end = _window_end(effective, k, inp.cfg, inp.calendar)
    frame = curve_forward(inp.days, curve, effective[k], end, inp.h, spec["year_days"], inp.funding())
    regimes = pd.DataFrame(0.0, index=inp.days, columns=effective.columns)
    regimes[k] = 1.0
    return frame, stale, {"regimes": regimes}


def _par(inp, x, tenor, curve):
    curve, stale = asof(curve, inp.days)
    frame = par_leg(inp.days, curve, tenor, x["coupons_per_year"], inp.h, inp.funding(),
                    inp.cfg["overnight"]["day_count"])
    return frame, stale, {"f": x["coupons_per_year"]}


def _par_yield_leg(inp, x, k, tenor):
    return _par(inp, x, tenor, inp.marks.tenors(x["source"], x["series"]))


def _par_from_spot_leg(inp, x, k, tenor):
    return _par(inp, x, tenor, inp.marks.par(x["source"], x["series"], x["coupons_per_year"], max([*x["legs"], tenor])))


# By the config's ``expression.<e>.instrument``: f(inputs, expression block, k, tenor) -> (frame, stale, extras).
LEGS = {"futures_month": _futures_leg, "curve_forward": _forward_leg, "par_yield": _par_yield_leg,
        "par_from_spot": _par_from_spot_leg}


def build(cfg, expression, marks, days, h, sessions, meetings, calendar=None, k=None, tenor=None):
    """One leg of ``expression.<expression>`` in currency block `cfg`, on sleeve sessions `days`: a `Leg`.

    `marks` is the currency's `Marks`; `sessions` and `meetings` its panels;
    `calendar` its meeting calendar (read only for a forward past the panel's
    last meeting); `k` the meeting horizon of an outright leg, `tenor` the years
    of a par leg; `h` the carry horizon in days. A currency with an
    ``expression.fx`` converts at it (the config requires one outside the book
    currency, and none in it), inside its ``plausible`` range. The frame
    carries ``stale`` (the session has no print of its own) and ``fx_stale``.
    """
    x = cfg["expression"][expression]
    inp = Inputs(cfg, marks, pd.DatetimeIndex(days), h, sessions, meetings, calendar)
    frame, stale, extras = LEGS[x["instrument"]](inp, x, k, tenor)
    fx = cfg["expression"].get("fx")
    if fx is None:
        spot, fx_stale = np.ones(len(inp.days)), np.zeros(len(inp.days), dtype=bool)
    else:
        quote, fx_stale = asof(marks.daily(fx["source"], fx["series"]).to_frame(), inp.days)
        quote = quote.iloc[:, 0].to_numpy()
        spot, fx_stale = (quote if fx["book_per_unit"] else 1.0 / quote), fx_stale.to_numpy()
        plausible_spot(spot, inp.days, fx)
    frame = with_fx(frame, spot).assign(stale=stale.to_numpy(), fx_stale=fx_stale)
    return Leg(x["instrument"], marks.ccy, frame, **extras)
