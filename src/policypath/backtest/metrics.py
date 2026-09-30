"""Performance statistics, one definition each, shared by every report from week 8 on.

**Sharpe** (`sharpe`): the mean over the sd of daily P&L over the evaluation
sessions, flat days included, times the square root of the calendar's
observed sessions a year (`per_year`), not a typed 252: the book's union
calendar and each currency's own have different counts. A flat day is a day
the strategy chose to be flat, so it stays in; which sessions are evaluated at
all is the ELB treatment's to say (`strategy/positions.py`, `samples`).

**Its standard error** (`sharpe_se`): sqrt((1 + SR^2 / 2) / years), the iid
large-sample value (Lo 2002), with years = the evaluation sessions over the
sessions a year. Daily P&L of a position held for weeks is not iid, and the
sample is a few cycles, so this is the least the uncertainty can be: every
Sharpe the reports quote carries it.

**The information coefficient** (`forward`, `ic`, week 9's robustness grid):
a signal-level number, free of sizing, costs and the book. ``fwd_t(h) =
sum_(j=1..h) x_(t+lag+j)``: what a unit position decided at t's close earns
over the h sessions it is then held, x the rate-change component of unit P&L
(spec section 6). The IC is the mean product of the standardised ranks of
z_t and fwd_t (Spearman's rho), and its t divides by a Newey-West standard
error with Bartlett weights to lag h, because consecutive forward windows
share h - 1 sessions. That lag covers the overlap and nothing past it: a
persistent z keeps the rank products autocorrelated beyond h (on the week 9
grid 0.12 at lag 21 for the GBP outright, 0.33 for the USD outright), so the
t overstates the precision. `ic_offsets` is spec section 6's cross-check: the
IC on every h-th session only, whose forward windows do not overlap, once for
each of the h start offsets, summarised by their mean and range.

**The long-run variance** (`long_run_var`): the Newey-West estimate with
Bartlett weights 1 - k / (L + 1), the IC's and the robustness grid's paired
SE's (`strategy/robustness.py`).

**The rest of the performance numbers** (week 10, `evaluate`), all over a
result's counted sessions (``kept``) and net of costs unless named gross:

- the **hit rate**, daily (`hit_rate`: the share of sessions with a position
  that made money) and per trade (`trades`: a trade is a run of one side of
  the position on after each close, from the close that puts it on to the
  close that takes it off; it earns the P&L of the sessions it is held into
  and pays the costs of the closes it is on after, and of the close that takes
  it off. On a flip, that close's whole cost goes to the new trade);
- the **maximum drawdown** (`max_drawdown`): the largest fall of cumulative
  net P&L from its running peak, the peak never below the start (0), in the
  P&L's own units;
- **turnover** (`turnover`): DV01 traded a year over the mean gross DV01 held on
  the sessions with a position, so a round trip is 2 turns however many legs;
- **time in market** (`time_in_market`): the share of sessions with a position.

**A result** (`Result`, `result`) is anything with a daily frame and its
calendar's sessions a year: a week 8 sleeve run and a week 9 book
(`report.costs.Run`, `report.portfolio.Book`) both are. Its frame needs gross,
cost and kept; traded and gross_dv01 give turnover and time in market, and a
signed ``held`` (the position held into each session) the trades.

**Without a window** (`without`, `ex_2022`, `ex_2022_23`): the statistics on
the P&L with the window's sessions left out, the positions not re-run, so the
question answered is "what did the rest of the sample earn", not "what would
a strategy that never saw 2022 have done". The windows are
``evaluation.exclude``'s, named in config: ex_2022 is calendar 2022, and
ex_2022_23 Dec 2021 to Aug 2023, the cycle from the first priced hike to the
last delivered one. One violent, well-telegraphed cycle can carry a rates
backtest, so both are a function call on any result.

**The level factor** (`factor`): OLS with an intercept of a sleeve's unit P&L
on the outright unit P&L of its currencies, over the sessions all have: the
betas and R^2 `report/attribution.py` splits a sleeve's P&L with, into what
its level exposure earned and the rest. Full-sample, so ex post: an
attribution, not a hedge anyone could have run.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd

YEAR_DAYS = 365.25


def per_year(sessions):
    """A calendar's observed sessions a year: its count over the calendar years its first to last session span."""
    sessions = pd.DatetimeIndex(sessions)
    if len(sessions) < 2:
        return np.nan
    return len(sessions) / (((sessions[-1] - sessions[0]).days + 1) / YEAR_DAYS)


def sharpe(pnl, periods):
    """Annualised Sharpe of daily P&L `pnl` (the evaluation sessions only, flat days as 0): NaN if its sd is 0."""
    x = np.asarray(pnl, dtype=float)
    if len(x) < 2:
        return np.nan
    sd = x.std(ddof=1)
    return float(x.mean() / sd * np.sqrt(periods)) if sd > 0 else np.nan


def sharpe_se(sr, years):
    """Standard error of an annualised Sharpe `sr` over `years` of daily data: sqrt((1 + SR^2 / 2) / years)."""
    return float(np.sqrt((1.0 + sr ** 2 / 2.0) / years)) if years > 0 else np.nan


def forward(x, h, lag):
    """fwd_t = x_(t+lag+1) + ... + x_(t+lag+h), by session of `x`: NaN where any of them is missing."""
    return x.rolling(h, min_periods=h).sum().shift(-(lag + h))


def long_run_var(x, lags):
    """The Newey-West long-run variance of `x` (an array): its autocovariances to `lags`, Bartlett weights, over n."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    d = x - x.mean()
    lagged = sum((1.0 - k / (lags + 1.0)) * (d[k:] @ d[:-k]) / n for k in range(1, min(lags, n - 1) + 1))
    return d @ d / n + 2.0 * lagged


def rank_products(z, fwd):
    """The products of the standardised ranks of `z` and `fwd` over the sessions both have, by session: their mean
    is the IC. Empty where fewer than 3 sessions or either has no spread."""
    both = z.notna() & fwd.notna()
    if both.sum() < 3:
        return pd.Series(dtype=float)
    a, b = z[both].rank(), fwd[both].rank()
    sa, sb = a.std(ddof=0), b.std(ddof=0)
    if not (sa > 0 and sb > 0):
        return pd.Series(dtype=float)
    return ((a - a.mean()) / sa) * ((b - b.mean()) / sb)


def ic(z, fwd, lags):
    """(IC, t, n): the rank IC of `z` against `fwd` over the sessions both have (module docstring), its Newey-West
    t with Bartlett weights to `lags`, and the sessions it is over. NaN where fewer than 3 sessions or no spread."""
    n = int((z.notna() & fwd.notna()).sum())
    p = rank_products(z, fwd).to_numpy()
    if not len(p):
        return np.nan, np.nan, n
    mean = float(p.mean())
    var = long_run_var(p, lags)
    return mean, (mean / np.sqrt(var / n) if var > 0 else np.nan), n


def ic_offsets(z, fwd, h):
    """The non-overlapping cross-check of IC(h): the rank IC on every `h`-th session of the index of `z`, from each
    of the h start offsets. Returns (their mean, lowest, highest, the fewest sessions one is over); NaN if none."""
    got = [ic(z.iloc[o::h], fwd.iloc[o::h], 0) for o in range(h)]
    v = np.array([x for x, _, _ in got if pd.notna(x)])
    if not len(v):
        return np.nan, np.nan, np.nan, 0
    return float(v.mean()), float(v.min()), float(v.max()), min(n for _, _, n in got)


# ---- week 10: the rest of the performance numbers ----------------------------------

@dataclass
class Result:
    """A daily frame (gross, cost, kept; traded, gross_dv01 and held if there) and its calendar's sessions a year."""
    daily: pd.DataFrame
    periods: float


def result(x):
    """`x` as a `Result`: a week 9 book (its run), a week 8 run, or a `Result` already."""
    x = getattr(x, "run", x)
    return x if isinstance(x, Result) else Result(x.daily, x.periods)


def hit_rate(net, on):
    """The share of the sessions in `on` (bool, by session) whose `net` P&L is above 0; NaN if there are none."""
    x = net[on.reindex(net.index, fill_value=False)]
    return float((x > 0).mean()) if len(x) else np.nan


def trades(gross, cost, held):
    """Each trade's net P&L, indexed by the session of the close that put it on (module docstring).

    `held` is the signed position held into each session, so the position on
    after the close of t is the one held at t + 1. Gross P&L on t belongs to
    the trade on after the close before; a close's cost to the trade on after
    it, or, where it goes flat, to the one it took off. A trade still on at
    the end is marked there.
    """
    side = np.sign(held.fillna(0.0))
    on = side.shift(-1, fill_value=0.0)
    new = (on != 0) & (on != on.shift(1, fill_value=0.0))
    tid = new.cumsum().where(on != 0)
    pnl_id = tid.shift(1)
    cost_id = tid.where(on != 0, tid.shift(1))
    ids = np.arange(1, int(new.sum()) + 1)
    net = gross.groupby(pnl_id).sum().reindex(ids, fill_value=0.0) \
        - cost.groupby(cost_id).sum().reindex(ids, fill_value=0.0)
    return pd.Series(net.to_numpy(), index=on.index[new.to_numpy()], name="trade")


def max_drawdown(net):
    """The largest fall of cumulative `net` from its running peak, the peak never below 0; in `net`'s units."""
    e = np.asarray(net, dtype=float).cumsum()
    if not len(e):
        return np.nan
    return float((np.maximum.accumulate(np.maximum(e, 0.0)) - e).max())


def turnover(traded, gross_dv01, years):
    """DV01 traded a year over the mean gross DV01 on the sessions with a position; NaN with no position."""
    held = gross_dv01[gross_dv01 > 0]
    return float(traded.sum() / years / held.mean()) if len(held) and years > 0 else np.nan


def time_in_market(gross_dv01):
    """The share of sessions with a position."""
    return float((gross_dv01 > 0).mean()) if len(gross_dv01) else np.nan


def evaluate(x, drop=None, capital=None):
    """Every performance number of a result `x` over its counted sessions, less the `drop` window ([first, last]).

    Sharpe gross and net with their SEs, net and gross P&L a year, vol a year
    and worst drawdown (percent of `capital` where given, else the P&L's
    units), the hit rates, the trades, turns a year and time in market. The
    positions are not re-run without the window: its sessions are only left
    out, and so are the trades put on inside it.
    """
    r = result(x)
    d = r.daily[r.daily["kept"]]
    if drop is not None:
        lo, hi = pd.Timestamp(drop[0]), pd.Timestamp(drop[1])
        d = d[(d.index < lo) | (d.index > hi)]
    n, per = len(d), r.periods
    years = n / per if per > 0 else np.nan
    net = d["gross"] - d["cost"]
    unit = 100.0 / capital if capital else 1.0
    gsr, nsr = sharpe(d["gross"], per), sharpe(net, per)
    dv01 = d["gross_dv01"] if "gross_dv01" in d else None
    on = dv01 > 0 if dv01 is not None else net != 0
    out = {"sessions": n, "years": years, "first": d.index.min() if n else pd.NaT,
           "last": d.index.max() if n else pd.NaT,
           "gross_sr": gsr, "gross_se": sharpe_se(gsr, years) if pd.notna(gsr) else np.nan,
           "net_sr": nsr, "net_se": sharpe_se(nsr, years) if pd.notna(nsr) else np.nan,
           "net_year": net.sum() / years * unit if n else np.nan,
           "gross_year": d["gross"].sum() / years * unit if n else np.nan,
           "vol": float(net.std(ddof=1) * np.sqrt(per) * unit) if n > 1 else np.nan,
           "max_dd": max_drawdown(net) * unit, "hit_daily": hit_rate(net, on),
           "turns_year": turnover(d["traded"], dv01, years) if dv01 is not None and "traded" in d else np.nan,
           "time_in_market": time_in_market(dv01) if dv01 is not None else np.nan}
    if "held" in r.daily:
        k = r.daily[r.daily["kept"]]
        t = trades(k["gross"], k["cost"], k["held"])
        if drop is not None:
            t = t[(t.index < lo) | (t.index > hi)]
        out |= {"trades": len(t), "hit_trade": float((t > 0).mean()) if len(t) else np.nan}
    return out


def window(name, book=None):
    """The ``evaluation.exclude`` window `name` of the book config (the configured one by default): (first, last)."""
    if book is None:
        from policypath import config
        book = config.strategy()
    lo, hi = book["evaluation"]["exclude"][name]
    return pd.Timestamp(lo), pd.Timestamp(hi)


def without(x, name, book=None, capital=None):
    """`evaluate` on result `x` with the ``evaluation.exclude`` window `name` left out, the positions not re-run."""
    return evaluate(x, window(name, book), capital)


def ex_2022(x, book=None, capital=None):
    """The result `x` without calendar 2022 (``evaluation.exclude.ex_2022``): `evaluate`'s numbers."""
    return without(x, "ex_2022", book, capital)


def ex_2022_23(x, book=None, capital=None):
    """The result `x` without the 2022-23 cycle (``evaluation.exclude.ex_2022_23``): `evaluate`'s numbers."""
    return without(x, "ex_2022_23", book, capital)


def ic_table(z, x, horizons, lag, mask):
    """IC(h) of `z` against the next h sessions of `x` after the execution `lag`, for each h: one dict a horizon.

    `mask` (bool by session) says which sessions' z count. Each row has the
    IC, its Newey-West t (Bartlett weights to h), the sessions, and the
    non-overlapping cross-check (`ic_offsets`: mean, lowest and highest over
    the h start offsets).
    """
    zm = z.where(mask.reindex(z.index, fill_value=False))
    out = []
    for h in horizons:
        fwd = forward(x.reindex(z.index), h, lag)
        v, t, n = ic(zm, fwd, h)
        mean, lo, hi, fewest = ic_offsets(zm, fwd, h)
        out.append({"h": h, "ic": v, "t": t, "sessions": n, "offsets_mean": mean, "offsets_min": lo,
                    "offsets_max": hi, "offsets_sessions": fewest})
    return out


def factor(y, xs):
    """OLS with an intercept of `y` on the columns of `xs` over the sessions all have: ({column: beta}, R^2, n)."""
    both = pd.concat([y.rename("_y"), xs], axis=1).dropna()
    n = len(both)
    if n < xs.shape[1] + 2:
        return {c: np.nan for c in xs.columns}, np.nan, n
    a = np.column_stack([np.ones(n), both[list(xs.columns)].to_numpy(dtype=float)])
    b = both["_y"].to_numpy(dtype=float)
    coef, *_ = np.linalg.lstsq(a, b, rcond=None)
    resid = b - a @ coef
    tss = ((b - b.mean()) ** 2).sum()
    return dict(zip(xs.columns, coef[1:].tolist())), (float(1.0 - resid @ resid / tss) if tss > 0 else np.nan), n
