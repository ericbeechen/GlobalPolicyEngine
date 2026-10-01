"""The covariance scored as a forecast of the P&L it sizes, and what the gap between realized and ex-ante vol is made of.

Paleologo's evaluation of a risk model (The Elements of Quantitative
Investing, 2024, ch. 5) on the book's sleeves. `report/portfolio.py` runs it
and reports it beside the book; nothing here feeds a position.

**A candidate** (`candidates`) is a covariance path, session x sleeve x
sleeve, each session's matrix from the unit P&L strictly before it, built the
way the book builds its own (`estimate`: `risk.ewma_cov`, shrunk or not,
rescaled to the floored sds by `portfolio.floored` or not). The configured
one is the book's (``risk.ewma_lambda``, ``risk.nonsynchronous_lag``,
shrunk, floored at ``risk.sigma_floor``); each other moves one of those off
it: lambda (`LAMBDAS`), the shrinkage, the lag term, the floor.

**The horizon** is the book's. A position decided at the close of d is held
into d + ``ahead`` (the execution lag + 1), so a forecast is scored there:
the matrix d sized on against the unit P&L x_t, t = d + ahead (`at_horizon`).
A test portfolio w gives the forecast h = w' Sigma_d w and the realized r =
(w' x_t)^2. Every w is rescaled so that the configured candidate's h is 1,
which leaves QLIKE and the bias statistic as they were and puts MSE in units
of the configured forecast, every session weighted alike.

**The sample** is the sessions from the book's first decision + ahead on
which every sleeve has a unit P&L and is credited exactly one session of its
own (not stale, not a catch-up: `portfolio.credits`), and every candidate has
a matrix at d.

**The losses** (`score`), for each candidate against the configured one:

- QLIKE, r/h - log(r/h) - 1: zero where r = h, and with MSE one of the two
  losses Patton (2011) shows rank forecasts correctly through a noisy
  realized variance. It is undefined where r = 0 (an unchanged settle), so
  differences use r/h + log h, which differs from it by a term the forecast
  does not enter;
- MSE, (r - h)^2, as a ratio to the configured candidate's: robust too, but
  carried by the largest days;
- the bias statistic, b = sqrt(mean(r / h)): realized over ex-ante vol for a
  book that holds w at the forecast's risk, the report's own ratio.

A difference in mean loss carries its Diebold-Mariano t: the mean of the
per-session differences over its Newey-West SE, Bartlett weights to
`DM_LAGS` (vol clusters, so the losses are autocorrelated).

**The test portfolios** (`fixed`, `realized_alpha`):

- each sleeve alone: w = e_i, the sleeves' losses averaged per session;
- random portfolios (Procedure 5.1): `DRAWS` a session, w = u / sd_d, u ~
  N(0, I), sd_d the configured floored sd: in the book's risk units, so no
  one sleeve's vol dominates. Drawn from `SEED`, the same draws for every
  candidate, so the report is reproducible;
- the headline's targets: w = q_d, the construction's targets decided at d
  (before the band and the execution on the sleeves' own calendars);
- realized alpha (Procedure 5.3, the precision matrix mean-variance reads):
  consecutive `ALPHA_BLOCK`-session blocks of the sample, alpha = the block's
  mean unit P&L, w_c = Sigma_c^-1 alpha with each candidate's own matrix at
  the block's first decision, r = w_c' S w_c with S the block's sample
  covariance (ddof 1), h = w_c' Sigma_c w_c. For Gaussian P&L a block's sample
  mean and sample covariance are independent, so under a correct Sigma, E[r] =
  h; a precision matrix that leans on directions whose risk it understates
  scores b above 1. It reads the future by design: a test portfolio, not a
  strategy. The blocks do not overlap, so its t has no lags.

**What realized over ex-ante is made of** (`benchmark`). A vol-targeted book
sees b > 1 even when its covariance is right on average: sizing on a noisy
estimate puts on more where the estimate is low (E[1/h] > 1/E[h]), and fat
tails make the estimate noisier. The benchmark measures both on P&L that has
no vol dynamics at all. The unit P&L is whitened by the configured 1-step
matrix (z_t = L_t^-1 x_t, L_t L_t' = Sigma_t), the pooled shocks rescaled to
an identity covariance, and synthetic P&L drawn `REPS` times, iid: once
Gaussian, once by resampling the shocks' rows (their tails and their
co-movement kept, their clustering removed). Each draw goes through the same
estimate and the same test portfolio, on the same sample, and b is pooled
over the draws. So realized over ex-ante factors as:

    b_gaussian (estimation noise) x b_bootstrap / b_gaussian (the tails)
      x b / b_bootstrap (the vol dynamics the EWMA does not track)

and, for the book, x the report's ratio over the frictionless b (the band,
the cap, the calendar and the costs). The benchmark is ex post: it pools the
whole sample's shocks.
"""

import numpy as np
import pandas as pd
from policypath.backtest import metrics
from policypath.strategy import portfolio, risk

CONFIGURED = "configured"
LAMBDAS = (0.94, 0.99)        # the lambdas scored beside the configured one: a 16- and a 100-session mean age
DRAWS = 200                   # random portfolios a session (Procedure 5.1)
ALPHA_BLOCK = 21              # sessions in a realized-alpha block (Procedure 5.3): a month
REPS = 20                     # synthetic draws of each benchmark
SEED = 20260930
DM_LAGS = 21                  # sessions: the Newey-West lags of a daily family's Diebold-Mariano t


# ---- the candidates ----------------------------------------------------------------

def estimate(x, lam, lags, shrink=True, floor=None):
    """The covariance path of unit P&L `x` (session x sleeve), session x sleeve x sleeve, as the book builds it:
    `risk.ewma_cov` at `lam` with `lags` lag terms, shrunk or not, and rescaled to the sds floored at `floor` x their
    trailing median (`portfolio.floored`) unless `floor` is None."""
    cov = risk.ewma_cov(x, lam, lags)
    if floor is None:
        return cov.shrunk if shrink else cov.s
    return portfolio.floored(cov, shrink, floor)[1]


def candidates(x, lam, lags, floor, lambdas=LAMBDAS):
    """{name: covariance path}: the configured estimate first, then each with one choice moved off it."""
    out = {CONFIGURED: estimate(x, lam, lags, True, floor)}
    out |= {f"lambda {a:g}": estimate(x, a, lags, True, floor) for a in lambdas if a != lam}
    out["unshrunk"] = estimate(x, lam, lags, False, floor)
    if lags:
        out["no lag term"] = estimate(x, lam, 0, True, floor)
    if floor is not None:
        out["no floor"] = estimate(x, lam, lags, True, None)
    return out


def at_horizon(m, ahead):
    """Array `m` (session first) shifted so that row t holds row t - `ahead`: what was decided `ahead` sessions back."""
    out = np.full_like(m, np.nan, dtype=float)
    if ahead < len(m):
        out[ahead:] = m[:len(m) - ahead]
    return out


def sample(x, mats, clean, first):
    """The sessions scored (module docstring): at or after position `first` (the book's first decision + ahead),
    every sleeve's unit P&L there and `clean` (bool by session), every matrix of `mats` (at the horizon) finite."""
    v = np.asarray(x, dtype=float)
    ok = np.isfinite(v).all(axis=1) & np.asarray(clean, dtype=bool)
    for m in mats.values():
        ok &= np.isfinite(m).all(axis=(1, 2))
    ok[:first] = False
    return ok


# ---- the test portfolios -----------------------------------------------------------

def sleeves_alone(t, n):
    """Each sleeve alone: session x n x n, w = e_i."""
    return np.broadcast_to(np.eye(n), (t, n, n))


def random_weights(sd, draws=DRAWS, seed=SEED):
    """Procedure 5.1's portfolios in risk units: session x draws x sleeve, u / sd, u ~ N(0, I) from `seed`."""
    t, n = sd.shape
    u = np.random.default_rng(seed).standard_normal((t, draws, n))
    with np.errstate(invalid="ignore", divide="ignore"):
        return u / sd[:, None, :]


def quad(w, m):
    """w' m w for every session and portfolio: w session x k x n, m session x n x n."""
    return np.einsum("tki,tij,tkj->tk", w, m, w)


def fixed(x, mats, w, rows):
    """(r, h) for the fixed portfolios `w` (session x k x sleeve, at the horizon) on the sessions `rows`.

    `r` is (w' x_t)^2 and `h` {candidate: w' Sigma w}, session x k, each w rescaled so the configured h is 1.
    """
    xv, ww = np.asarray(x, dtype=float)[rows], np.asarray(w, dtype=float)[rows]
    base = quad(ww, mats[CONFIGURED][rows])
    ww = ww / np.sqrt(base)[..., None]
    r = np.einsum("tki,ti->tk", ww, xv) ** 2
    return r, {c: quad(ww, m[rows]) for c, m in mats.items()}


def realized_alpha(x, mats, rows, block=ALPHA_BLOCK):
    """(r, h, starts) for Procedure 5.3: per candidate, session x 1 arrays over the blocks of `block` consecutive
    sessions of `rows`, and the position of each block's first session (module docstring)."""
    at = np.flatnonzero(rows)
    xv = np.asarray(x, dtype=float)
    starts, r, h = [], {c: [] for c in mats}, {c: [] for c in mats}
    for i in range(0, len(at) - block + 1, block):
        days = at[i:i + block]
        xs = xv[days]
        alpha, s = xs.mean(axis=0), np.cov(xs, rowvar=False, ddof=1)
        for c, m in mats.items():
            w = np.linalg.solve(m[days[0]], alpha)
            w = w / np.sqrt(w @ m[days[0]] @ w)
            r[c].append(w @ s @ w)
            h[c].append(1.0)
        starts.append(days[0])
    return ({c: np.array(v)[:, None] for c, v in r.items()}, {c: np.array(v)[:, None] for c, v in h.items()},
            np.array(starts, dtype=int))


# ---- the losses --------------------------------------------------------------------

def bias(r, h):
    """The bias statistic, sqrt(mean(r / h)) over every entry: realized over ex-ante vol at the forecast's risk."""
    return float(np.sqrt(np.mean(np.asarray(r) / np.asarray(h))))


def qlike(r, h):
    """QLIKE in its forecast-ranking form, r/h + log h, by entry: it differs from r/h - log(r/h) - 1 by log r - 1,
    which the forecast does not enter, and is defined at r = 0."""
    return r / h + np.log(h)


def mse(r, h):
    """(r - h)^2 by entry."""
    return (r - h) ** 2


def dm(d, lags):
    """The Diebold-Mariano t of the per-session loss differences `d`: their mean over its Newey-West SE."""
    d = np.asarray(d, dtype=float)
    var = metrics.long_run_var(d, lags)
    return float(d.mean() / np.sqrt(var / len(d))) if var > 0 and len(d) > 1 else np.nan


def score(r, h, lags):
    """Every candidate of `h` against the configured one, from (r, h) session x k (r per candidate where the
    candidates' portfolios differ: a dict like `h`).

    {candidate: {b, qlike (the mean difference in QLIKE per entry), qlike_t, mse_ratio, mse_t}}, the
    configured candidate's differences 0; plus ``sessions`` and ``draws``.
    """
    rr = r if isinstance(r, dict) else {c: r for c in h}
    base_q = qlike(rr[CONFIGURED], h[CONFIGURED]).mean(axis=1)
    base_m = mse(rr[CONFIGURED], h[CONFIGURED]).mean(axis=1)
    out = {}
    for c in h:
        q, m = qlike(rr[c], h[c]).mean(axis=1), mse(rr[c], h[c]).mean(axis=1)
        same = c == CONFIGURED
        out[c] = {"b": bias(rr[c], h[c]), "qlike": float((q - base_q).mean()),
                  "qlike_t": np.nan if same else dm(q - base_q, lags),
                  "mse_ratio": float(m.mean() / base_m.mean()), "mse_t": np.nan if same else dm(m - base_m, lags)}
    first = next(iter(h.values()))
    return {"sessions": int(first.shape[0]), "draws": int(first.shape[1]), "candidates": out}


def kurtosis(r, h):
    """The kurtosis of the standardised P&L z = sqrt(r / h) by column: mean z^4 / (mean z^2)^2 (3 if Gaussian)."""
    z2 = np.asarray(r) / np.asarray(h)
    return z2.mean(axis=0) ** -2 * (z2 ** 2).mean(axis=0)


# ---- the benchmark -----------------------------------------------------------------

def shocks(x, m, rows):
    """The unit P&L at `rows` whitened by the 1-step matrices `m` (z_t = L_t^-1 x_t) and rescaled so the pooled
    shocks' second moment is the identity: n x sleeve."""
    xv = np.asarray(x, dtype=float)[rows]
    low = np.linalg.cholesky(m[rows])
    z = np.linalg.solve(low, xv[..., None])[..., 0]
    c = np.linalg.cholesky(z.T @ z / len(z))
    return np.linalg.solve(c, z.T).T


def synthetic(pool, days, names, reps=REPS, seed=SEED):
    """Yield (kind, frame) `reps` times for each kind: iid P&L on `days` with an identity covariance, ``gaussian``
    N(0, I) and ``bootstrap`` rows of `pool` (`shocks`) drawn with replacement."""
    rng = np.random.default_rng(seed)
    t, n = len(days), pool.shape[1]
    for _ in range(reps):
        yield "gaussian", pd.DataFrame(rng.standard_normal((t, n)), index=days, columns=names)
        yield "bootstrap", pd.DataFrame(pool[rng.integers(0, len(pool), t)], index=days, columns=names)


def benchmark(x, cfg, pool_rows, rows, make, lam, lags, floor, ahead, reps=REPS, seed=SEED):
    """b with no vol dynamics (module docstring): {kind: {family: b by column}}.

    `x` is the unit P&L (a frame), `cfg` the configured matrices (1-step, not at the horizon), `pool_rows` the
    sessions whose shocks are pooled and `rows` {family: the sessions scored}. `make(m)` gives {family: weights
    session x k x sleeve, decided at each session} from a synthetic covariance path `m`; each synthetic draw is
    estimated as the configured candidate is, and its families' b is pooled over the draws.
    """
    pool = shocks(x, cfg, pool_rows)
    sums, counts = {}, {}
    for kind, y in synthetic(pool, x.index, list(x.columns), reps, seed):
        m = estimate(y, lam, lags, True, floor)
        mh = at_horizon(m, ahead)
        for fam, w in make(m).items():
            r, h = fixed(y.to_numpy(), {CONFIGURED: mh}, at_horizon(w, ahead), rows[fam])
            key = (kind, fam)
            sums[key] = sums.get(key, 0.0) + (r / h[CONFIGURED]).sum(axis=0)
            counts[key] = counts.get(key, 0) + len(r)
    out = {}
    for (kind, fam), s in sums.items():
        out.setdefault(kind, {})[fam] = np.sqrt(s / counts[(kind, fam)])
    return out
