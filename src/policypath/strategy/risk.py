"""How much a sleeve's P&L moves: the EWMA volatility the sizing reads, and its floor. Shared by weeks 8 and 9.

**EWMA sigma** (`ewma_sigma`). The zero-mean exponentially weighted sd of a
daily P&L series, from observations strictly before each session:
``sigma_t^2 = sum_j w_j x_(t-j)^2`` over j >= 1, ``w_j`` proportional to
``lambda^(j-1)`` and normalised over the history there is (``risk.ewma_lambda``,
0.97: the weights' mean age is 1 / (1 - lambda) = 33 sessions, their half-life
23). Zero-mean because a sleeve's daily mean is a few percent of its sd, and
estimating it adds noise, not information. Strictly before t because the P&L
credited to t is marked on prints dated t, which are final only a business day
later: a decision at t's close cannot have seen it. NaN until 1 / (1 - lambda)
observations (34), one mean age of history.

**The floor** (`sigma_floor`): ``max(sigma_t, risk.sigma_floor x the median of
sigma over the trailing two years)``, 0.5 x. A sleeve's EWMA sigma spans a
wide range: the USD outright's runs from 3.5 to 241bp a year (1st to 99th
percentile, a factor of 68; yearly medians 5bp in 2013-14, 111bp in 2022), the
GBP outright's a factor of 35, the curve and cross sleeves' about 5. Sized by
sigma alone a sleeve puts on its largest position exactly where its recent vol
is lowest: on leaving the floor, when the vol is about to return. The floor
caps the position at twice what the sleeve's typical vol of the last two years
would give; it binds on 7% of the outright sleeves' sessions and on none of
the others'. The median is of the sigmas known on or before the session, so
the floor is as real-time as sigma.

**The sleeves' covariance** (`ewma_cov`, week 9's book). The same weights
across sleeves, zero mean, from the rows strictly before each session on
which every sleeve has an observation: ``S0 = sum w x x'``. NaN until
`min_periods` complete rows.

*The non-synchronous term.* London closes before New York, so a move after
the London close reaches the sterling legs a day later, as a lag-1
cross-covariance. The term added is the Newey-West one at one lag
(``risk.nonsynchronous_lag``): ``S = S0 + (S1 + S1') / 2``, ``S1 = sum w
x_t x_(t-1)'`` over the pairs there are. The weight 1/2 (Bartlett's) departs
from spec section 5.2, whose unweighted ``S0 + S1 + S1'`` is the truncated
kernel, which is not positive semi-definite: on the book's sleeves it was
not positive definite on 905 of the 3,670 sessions from the book's first
decision, with a variance below zero on 5. The cross sleeve is the largest
contributor (its legs are marked at the two closes, and its lag-1
autocorrelation averages -0.14, -0.48 at its lowest): without it the failures
fall to 209, not to none. Every variance below zero is the GBP outright's, a
sleeve marked at one close, whose lag-1 autocorrelation falls to -0.56: its
own P&L's, not the closes' timing. So the term's diagonal is a one-lag
long-run variance correction for every sleeve, S_ii = S0_ii (1 + rho_i), not
only a non-synchronous cross-term; it moves a sleeve's inverse-vol size by
1 / sqrt(1 + rho_i), up to 1.5x for the GBP outright
(`report/portfolio.lag_effect`). With the weight S is positive definite on
all 3,670. (With EWMA
weights even that is not guaranteed: the older day of a pair weighs lambda
times the newer, a deficit of about (1 - lambda) / 2, 1.5%, of S0 in the worst
direction. `portfolio.targets` checks every matrix it sizes on.) The term
raises the mean correlation of the two 2s10s sleeves from 0.42 to 0.52, and
of the two outrights from 0.32 to 0.35.

*The shrinkage*, toward the diagonal: ``pi_ij = sum w (x_i x_j - S0_ij)^2``
(= sum w (x_i x_j)^2 - S0_ij^2, the weights summing to 1), ``n_eff = 1 /
sum w^2`` ((1 + lambda) / (1 - lambda) = 65.7 in a long history), ``delta =
clip(sum_(i != j) pi_ij / n_eff / sum_(i != j) S_ij^2, 0, 1)``, ``Sigma = (1 -
delta) S + delta diag(S)``. A Ledoit-Wolf-type intensity with Schafer and
Strimmer's target D (the diagonal, unequal variances): pi / n_eff estimates
the sampling variance of each weighted cross moment, and delta is the share of
the off-diagonal's size that is noise. It counts S0's noise, not the lag
term's. Five sleeves is a small matrix, but an effective sample of 66
sessions is a short one: on the sleeves delta averages 0.20 (0.03 to 0.97),
and it halves the correlation matrix's median condition number (11.3 to 5.7;
the maximum 85 to 22). The diagonal is not shrunk, so a sleeve's sd is the
same either way.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd

FLOOR_WINDOW = "730D"     # the trailing window the floor's median is over: two calendar years, as the signal's z


def min_periods(lam):
    """Observations before an EWMA sigma exists: 1 / (1 - lambda), the weights' mean age, rounded up."""
    return int(np.ceil(1.0 / (1.0 - lam) - 1e-9))


def ewma_sigma(x, lam):
    """Zero-mean EWMA sd of `x` (by session), from observations strictly before each session: same units as `x`.

    Weights lambda^(age - 1), age in sessions, normalised over the observations
    there are: a missing observation is skipped, not counted as 0, and the
    session still ages the ones before it. NaN until `min_periods` observations.
    """
    x = pd.Series(x, dtype=float)
    var = (x ** 2).ewm(alpha=1.0 - lam, adjust=True, ignore_na=False, min_periods=min_periods(lam)).mean()
    return np.sqrt(var.shift(1)).rename("sigma")


def sigma_floor(sigma, floor, window=FLOOR_WINDOW):
    """`sigma` floored at `floor` x its median over the trailing `window`, the session's own sigma included.

    `sigma` is by session (`ewma_sigma`: each value already strictly before its
    session), so the floor uses nothing a session did not know. NaN where sigma is.
    """
    median = sigma.rolling(window, min_periods=1).median()
    return np.maximum(sigma, floor * median).where(sigma.notna()).rename("sigma")


@dataclass(frozen=True)
class Cov:
    """The sleeves' covariance at each session, from observations strictly before it (`ewma_cov`).

    Arrays are session x sleeve x sleeve, bp^2 a session per unit DV01 squared:
    ``s0`` the EWMA second moment, ``s`` with the non-synchronous term, ``shrunk``
    toward the diagonal by ``delta``. ``n_eff`` is 1 / sum w^2 and
    ``observations`` the complete rows before the session. NaN until
    `min_periods` complete rows.
    """
    index: pd.DatetimeIndex
    names: list
    s0: np.ndarray
    s: np.ndarray
    shrunk: np.ndarray
    delta: np.ndarray
    n_eff: np.ndarray
    observations: np.ndarray


def _ewm(products, lam):
    """EWMA mean of each column of `products` (session x k) from rows strictly before each session, as `ewma_sigma`."""
    frame = pd.DataFrame(products)
    return frame.ewm(alpha=1.0 - lam, adjust=True, ignore_na=False).mean().shift(1).to_numpy(copy=True)


def ewma_cov(x, lam, lags=1):
    """The zero-mean EWMA covariance of `x` (session x sleeve), strictly before each session, and its shrinkage.

    A row with any sleeve missing is not an observation (it still ages the ones
    before it), so every estimate is a weighted sum of outer products of complete
    rows. ``s = s0 + sum_l (1 - l / (lags + 1)) (s_l + s_l')``, ``s_l`` the EWMA
    of x_t x_(t-l)' over the pairs there are; the shrinkage is the module
    docstring's.
    """
    v = x.to_numpy(dtype=float).copy()
    complete = ~np.isnan(v).any(axis=1)
    v[~complete] = np.nan
    t, n = v.shape
    outer = v[:, :, None] * v[:, None, :]
    s0 = _ewm(outer.reshape(t, n * n), lam).reshape(t, n, n)
    fourth = _ewm((outer ** 2).reshape(t, n * n), lam).reshape(t, n, n)
    s = s0.copy()
    for lag in range(1, lags + 1):
        before = np.full_like(v, np.nan)
        before[lag:] = v[:-lag]
        sl = _ewm((v[:, :, None] * before[:, None, :]).reshape(t, n * n), lam).reshape(t, n, n)
        s = s + (1.0 - lag / (lags + 1.0)) * (sl + sl.transpose(0, 2, 1))
    w1, w2, count = np.zeros(t), np.zeros(t), np.zeros(t, dtype=int)
    a = b = 0.0
    c = 0
    for i in range(t):
        w1[i], w2[i], count[i] = a, b, c
        a, b = lam * a + complete[i], lam * lam * b + complete[i]
        c += int(complete[i])
    with np.errstate(divide="ignore", invalid="ignore"):
        n_eff = np.where(w2 > 0, w1 ** 2 / w2, np.nan)
        off = ~np.eye(n, dtype=bool)
        spread = (fourth - s0 ** 2)[:, off].sum(axis=1)
        size = (s[:, off] ** 2).sum(axis=1)
        delta = np.clip(np.where(size > 0, spread / n_eff / size, 0.0), 0.0, 1.0)
    diagonal = np.eye(n)[None] * np.diagonal(s, axis1=1, axis2=2)[:, :, None]
    shrunk = (1.0 - delta)[:, None, None] * s + delta[:, None, None] * diagonal
    early = count < min_periods(lam)
    for m in (s0, s, shrunk):
        m[early] = np.nan
    delta[early], n_eff[early] = np.nan, np.nan
    return Cov(pd.DatetimeIndex(x.index), list(x.columns), s0, s, shrunk, delta, n_eff, count)


def correlation(m):
    """The correlation matrix of covariance `m` (the last two axes)."""
    sd = np.sqrt(np.diagonal(m, axis1=-2, axis2=-1))
    return m / sd[..., :, None] / sd[..., None, :]


def condition(m):
    """The condition number of `m`'s correlation matrix: largest over smallest eigenvalue (scale-free)."""
    e = np.linalg.eigvalsh(correlation(m))
    return e[..., -1] / e[..., 0]
