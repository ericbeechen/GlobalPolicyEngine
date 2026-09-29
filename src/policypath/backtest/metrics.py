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

Week 10 adds hit rates, drawdown, turnover and time in market.
"""

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
