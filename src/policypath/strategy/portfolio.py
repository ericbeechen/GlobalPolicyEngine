"""The book: every sleeve sized together, on one calendar, from one covariance, under one vol target. Week 9.

The arithmetic of one book, close by close; `report/portfolio.py` runs it on
week 8's sleeves (their rules, ELB treatment, legs, marks and costs unchanged)
and reports it. Numbers are from the build of 2026-09-29 (reports/portfolio.md).

**The book calendar** (`to_book`, `credits`, `asof`) is the sessions of
``book.currency`` (USD). A sleeve's P&L on its own sessions is credited to the
next book session on or after it: a sterling session on a US holiday (80 since
2012) is summed into the next US session, and a US session on a UK holiday
(73) gets nothing from the sterling sleeves, a stale session for them.
Positions are decided only at book closes; a sleeve on another calendar picks
a decision up at its next own session and executes it ``lag`` of its sessions
later, as week 8's sleeves do. A sleeve session before the first book session
is dropped: there is no book session to credit it to.

**What is sized** is each sleeve's side, g, the week 8 hysteresis side under
the ELB treatment, as of the book close. Only the sleeves with a side are
sized (`targets`), by one of three constructions (`CONSTRUCTIONS`):

- ``inverse_vol`` (the headline, proposed): w = g / sd. Each sleeve at the
  same risk alone; the correlations reach it only through the vol target.
- ``erc``: every sleeve's share of the ex-ante variance equal, the signs fixed
  by g (`erc`). With one sleeve it is inverse-vol.
- ``mean_variance``: w = Sigma^-1 mu, mu = g x min(|z|, ``portfolio.z_cap``)
  x sd, the plan's "z as the expected return" (`mean_variance`). It is the
  one that can put a sleeve on the other side from its own signal: it does on
  8% of its active sleeve-sessions (14% unshrunk).

**The covariance** each reads is `risk.ewma_cov` of the sleeves' unit P&L on
the book calendar (+1 book DV01 on the first leg), shrunk, with each sd
floored as week 8's (`floored`: at 0.5 x its trailing two-year median) and
the matrix rescaled to the floored sds, its correlations kept: D R D. So the
floor shrinks a quiet sleeve in every construction, not only inverse-vol.

**Size** (`targets`). The weights are scaled so the ex-ante vol,
sqrt(w' Sigma w x 252), is ``book.vol_target`` x capital, 5% of $100m (252 as
week 8's vol-scaled sleeves). The target is the whole book's, so the sleeves
with a side carry all of it however few they are: 1.6 on average in 2014, 4.5
in 2022. Then the gross DV01 cap: every leg's |q| summed, at most
``risk.max_gross_dv01_per_capital`` x capital, 0.004 per bp ($400k per bp), so
one 25bp policy step against every leg at once loses 10% of capital, the
drawdown trigger; over it, every position is scaled down pro rata. It binds on
29% of the headline's sessions with a position, most of 2014 (92%) and
2017-18 (71%, 78%), when the USD front end's vol was lowest and vol-targeting
alone would have run up to $1.5m per bp gross.

**The band** (`band`): a sleeve trades to its target on a new side (from 0,
to 0, or across it) and otherwise only when the target is more than
``risk.no_trade_band`` (10%) from the position decided before, week 8's rule
on each leg's DV01. The cap outranks the band: where the positions kept
inside their bands would break it, every sleeve trades to its target.

**The drawdown overlay** (`overlay`): half size (``portfolio.drawdown.scale``)
from the close after the drawdown, (peak - equity) / capital, exceeds 10%;
full again once it is back under 5%. The state at a close reads the drawdown
at the close before, not its own: the equity credited to a session is marked
on prints final a business day later, the rule sigma and the covariance keep
(spec section 5.2 writes dd_t; this is dd_(t-1)). The scale multiplies the
targets decided at the close, so it acts with the execution lag. It fired
once on the headline, on 2015-04-06, and never released. It is reported
beside the book, never inside a robustness cell or a cost curve.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from policypath.strategy import risk

ERC_TOL = 1e-10           # the largest relative change in any weight over a sweep, at convergence
ERC_SWEEPS = 10_000
OVERLAY_ROUNDS = 100      # passes of the drawdown overlay's fixed point before it gives up


# ---- the book calendar -------------------------------------------------------------

def credited_to(index, days):
    """Position on the book calendar `days` each sleeve session is credited to: the next book session on or after it.

    -1 for a session before the first book session or after the last: there is
    no book session to credit it to.
    """
    index = pd.DatetimeIndex(index)
    at = days.searchsorted(index)
    return np.where((index >= days[0]) & (at < len(days)), at, -1)


def to_book(x, days):
    """A sleeve-session series or frame of flows (P&L, costs, DV01 traded) summed into the book calendar `days`.

    A book session no sleeve session is credited to gets 0 (a stale session for
    the sleeve); one that receives two (a sleeve session on a book holiday, and
    the next) gets their sum.
    """
    at = credited_to(x.index, days)
    keep = at >= 0
    summed = x[keep].groupby(at[keep]).sum()
    return summed.reindex(range(len(days)), fill_value=0.0).set_axis(days)


def credits(index, days):
    """Sleeve sessions credited to each book session: 0 is stale, 2 or more a catch-up."""
    return to_book(pd.Series(1, index=pd.DatetimeIndex(index)), days).astype(int)


def asof(x, days):
    """A level (a side, a z, a position) at each of `days`: its value at the last session of `x` on or before it.

    NaN before `x`'s first session, and wherever that session's value is NaN (a
    missing z stays missing, never the last one carried).
    """
    at = x.index.searchsorted(days, side="right") - 1
    v = x.to_numpy(dtype=float)
    return pd.Series(np.where(at >= 0, v[np.maximum(at, 0)], np.nan), index=days, name=x.name)


def unit_pnl(units, days):
    """Book session x sleeve unit P&L, bp: each sleeve's unit total (`units`, {name: series by its sessions}) summed
    into the book calendar; NaN before a sleeve's first credited session, 0 on a stale one after it."""
    out = {}
    for name, x in units.items():
        x = x.dropna()
        at = credited_to(x.index, days)
        col = to_book(x, days)
        out[name] = col.where(days >= days[at[at >= 0][0]]) if (at >= 0).any() else col * np.nan
    return pd.DataFrame(out, index=days)


# ---- the covariance the constructions read -----------------------------------------

def floored(cov, shrink, floor):
    """(sd, matrices): each sleeve's sd from the diagonal, floored as `risk.sigma_floor`, and the covariance
    rescaled to it (D R D, the correlations kept), session x sleeve x sleeve. `shrink` picks the shrunk estimate."""
    m = cov.shrunk if shrink else cov.s
    raw = np.sqrt(np.diagonal(m, axis1=1, axis2=2))
    sd = np.column_stack([risk.sigma_floor(pd.Series(raw[:, i], index=cov.index), floor).to_numpy()
                          for i in range(raw.shape[1])])
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = sd / raw
    return sd, m * ratio[:, :, None] * ratio[:, None, :]


# ---- the constructions (one session, the active sleeves) ---------------------------

def inverse_vol(g, sd, **_):
    """w = g / sd: each active sleeve at the same risk alone, the correlations ignored."""
    return g / sd


def erc(g, cov, v0=None, tol=ERC_TOL, **_):
    """Equal risk contribution with the signs fixed by `g`: w = g x v, v > 0 with v_i (G cov G v)_i equal for all i.

    Cyclical coordinate descent on M = G cov G (Griveau-Billion, Richard and
    Roncalli 2013): each v_i in turn is the positive root of
    M_ii v_i^2 + c_i v_i - b = 0, c_i = sum_(j != i) M_ij v_j, b = 1/n, the exact
    minimiser along v_i of v'Mv / 2 - b sum log v_i, whose minimum is the ERC
    point. Swept until no v_i moves by more than `tol` of itself. One sleeve is
    v = 1 / sd: inverse-vol. `v0` starts it (the previous session's v).
    """
    n = len(g)
    m = cov * np.outer(g, g)
    d = np.diag(m).copy()
    b = 1.0 / n
    v = 1.0 / np.sqrt(d) if v0 is None or len(v0) != n else np.asarray(v0, dtype=float).copy()
    for _ in range(ERC_SWEEPS):
        worst = 0.0
        for i in range(n):
            c = m[i] @ v - d[i] * v[i]
            new = (-c + np.sqrt(c * c + 4.0 * d[i] * b)) / (2.0 * d[i])
            worst = max(worst, abs(new - v[i]) / new)
            v[i] = new
        if worst < tol:
            return g * v
    raise RuntimeError(f"ERC did not converge in {ERC_SWEEPS} sweeps")


def mean_variance(g, cov, sd, z, z_cap, **_):
    """w = cov^-1 mu, mu = g x min(|z|, z_cap) x sd: the z-scored signal as the expected return (the plan's).

    With cov = D R D (D the sds), w = D^-1 R^-1 (mu / sd) = D^-1 R^-1 (g min(|z|,
    z_cap)): with no correlation it is inverse-vol weighted by the capped |z|,
    and every correlation moves weight toward hedging. It can put a sleeve on
    the other side from its own signal, where the others' positions say so.
    """
    return np.linalg.solve(cov, g * np.minimum(np.abs(z), z_cap) * sd)


CONSTRUCTIONS = {"inverse_vol": inverse_vol, "erc": erc, "mean_variance": mean_variance}


def contributions(w, cov):
    """Each sleeve's share of the ex-ante variance, w_i (cov w)_i / w'cov w (they sum to 1)."""
    rc = w * (cov @ w)
    return rc / rc.sum()


def gross(q, legs):
    """Gross DV01, book currency per bp: every leg's |q| (a curve or cross sleeve has two legs of |q| each)."""
    return float(np.abs(q) @ legs)


def vol_target(w, cov, periods, target):
    """k x w with sqrt(k^2 w'cov w x periods) = target: the ex-ante vol a year at `target` (book currency)."""
    return w * (target / np.sqrt((w @ cov @ w) * periods))


@dataclass
class Targets:
    """What a construction wants at each book close, before the drawdown overlay and the band.

    ``q`` (session x sleeve): first-leg book DV01, vol-targeted and capped.
    ``exante``: its ex-ante vol a year (book currency); ``uncapped``: the gross
    DV01 before the cap; ``capped``: the cap bound. ``share``: each sleeve's
    share of the ex-ante variance; ``flipped``: an active sleeve whose weight is
    on the other side from its signal (mean-variance only can do that).
    """
    q: pd.DataFrame
    exante: pd.Series
    uncapped: pd.Series
    capped: pd.Series
    share: pd.DataFrame
    flipped: pd.DataFrame


def targets(construction, g, z, sd, cov, legs, periods, target, limit, first, z_cap):
    """The construction's `Targets` at every book session from position `first` (the book's first decision).

    `g`, `z`, `sd` are session x sleeve arrays (the side, the z, the floored sd),
    `cov` session x sleeve x sleeve (rescaled to `sd`), `legs` the legs a
    sleeve has, `limit` the gross DV01 cap (book currency, None for none). Only
    active sleeves (g != 0) are sized.
    """
    days, names = g.index, list(g.columns)
    t, n = g.shape
    gv, zv = g.to_numpy(dtype=float), z.to_numpy(dtype=float)
    legs = np.asarray(legs, dtype=float)
    q, share = np.zeros((t, n)), np.zeros((t, n))
    exante, uncapped, capped = np.zeros(t), np.zeros(t), np.zeros(t, dtype=bool)
    make = CONSTRUCTIONS[construction]
    v0 = None
    for i in range(first, t):
        a = gv[i] != 0
        if not a.any():
            v0 = None
            continue
        c = cov[i][np.ix_(a, a)]
        if not (np.isfinite(c).all() and np.linalg.eigvalsh(c)[0] > 0):
            raise RuntimeError(f"the covariance of {[x for x, y in zip(names, a) if y]} on {days[i]:%Y-%m-%d} "
                               "is missing or not positive definite")
        w = make(g=gv[i, a], sd=sd[i, a], cov=c, z=zv[i, a], z_cap=z_cap, v0=v0)
        v0 = w * gv[i, a] if construction == "erc" else None
        k = vol_target(w, c, periods, target)
        uncapped[i] = gross(k, legs[a])
        if limit is not None and uncapped[i] > limit:
            k, capped[i] = k * (limit / uncapped[i]), True
        q[i, a] = k
        exante[i] = np.sqrt(k @ c @ k * periods)
        share[i, a] = contributions(k, c)
    frame = lambda x: pd.DataFrame(x, index=days, columns=names)
    return Targets(frame(q), pd.Series(exante, index=days), pd.Series(uncapped, index=days),
                   pd.Series(capped, index=days), frame(share), frame((np.sign(q) * gv) < 0))


def band(target, width, legs, limit=None):
    """The positions decided at each book close: each sleeve's target, traded only where it must be.

    A sleeve trades to its target on a new side (from 0, to 0, or across it)
    and otherwise only when the target is more than `width` from the position
    decided before (``risk.no_trade_band``, each leg's DV01, as week 8's).
    Where keeping positions inside their band would leave the gross DV01 over
    `limit`, every sleeve trades to its target: the cap holds on every close.
    """
    tv = target.to_numpy(dtype=float)
    legs = np.asarray(legs, dtype=float)
    out = np.zeros_like(tv)
    held = np.zeros(tv.shape[1])
    with np.errstate(divide="ignore", invalid="ignore"):
        for i, want in enumerate(tv):
            keep = (want != 0) & (np.sign(want) == np.sign(held)) & (np.abs(want / held - 1.0) <= width)
            new = np.where(keep, held, want)
            if limit is not None and gross(new, legs) > limit * (1.0 + 1e-12):
                new = want
            out[i] = held = new
    return pd.DataFrame(out, index=target.index, columns=target.columns)


# ---- the drawdown overlay ----------------------------------------------------------

def drawdown(equity, capital):
    """(peak - equity) / capital at each session, the peak the highest equity to date and never below 0 (the start)."""
    e = equity.to_numpy(dtype=float)
    peak = np.maximum.accumulate(np.maximum(e, 0.0))
    return pd.Series((peak - e) / capital, index=equity.index, name="drawdown")


def overlay(equity, capital, trigger, release, scale):
    """The drawdown control's scale decided at each book close, and the sessions it triggered on.

    The state reads the drawdown at the close before (the equity credited to
    the session is marked on prints final a business day later, as sigma and
    the covariance): on when it is above `trigger`, off when it is back below
    `release`, `scale` while on. The scale multiplies the targets decided at
    the close, so it acts with the execution lag.
    """
    dd = drawdown(equity, capital).shift(1, fill_value=0.0).to_numpy()
    out = np.ones(len(dd))
    on, starts = False, []
    for i, x in enumerate(dd):
        if not on and x > trigger:
            on = True
            starts.append(equity.index[i])
        elif on and x < release:
            on = False
        out[i] = scale if on else 1.0
    return pd.Series(out, index=equity.index, name="scale"), starts
