"""A Nelson-Siegel-Svensson fit of one day's spot curve: the robustness alternative to log-linear interpolation.

The baseline reads a meeting window's forward off the published nodes with
s t linear between them (`curves/forward.py`). This replaces the nodes with a
smooth parametric curve fitted to them:

    s(t) = b0 + b1 L(t/tau1) + b2 [L(t/tau1) - e^(-t/tau1)] + b3 [L(t/tau2) - e^(-t/tau2)],
    L(x) = (1 - e^(-x)) / x

continuously compounded, t in years. Given the decay times it is linear in
the betas, so the fit is profile least squares: every (tau1, tau2) on a grid
gets its betas by OLS, and the pair with the smallest squared error wins.
tau1 is on `TAU_GRID`, 25 log-spaced points from one month to five years (the
OIS short end's own span); tau2 on the same grid with tau2 >= 1.5 tau1
(`MIN_RATIO`): closer than that, the two curvature loadings are nearly
collinear. The grid steps by 60^(1/24) = 1.186, so the closest pair it allows
is three steps apart, tau2 / tau1 = 1.67: that is where the ``ratio`` edge
sits. 253 pairs. A grid needs no starting values and has no local minima
between its points, but an optimum on its edge says the grid chose it, not
the data, so `fit` names the edge (`EDGES`) and the caller counts them. The
node RMSE says how far the smooth curve sits from the Bank's own. The grid's
resolution matters too: on 63 GBP days a 97-point grid on the same range
moves the day's largest meeting rate by 0.53bp on average (max 3.6), against
1.27bp (max 6.3) for NSS against log-linear (notes/DECISIONS.md, V6).

Every pair's least-squares projection depends only on the maturities, so it is
built once per set of them (`_projections`), and a day's fit is one batched
matrix product over the 253 pairs.
"""

from functools import lru_cache
import numpy as np
import pandas as pd

TAU_GRID = np.geomspace(1.0 / 12.0, 5.0, 25)   # years
MIN_RATIO = 1.5                                 # tau2 >= 1.5 tau1
EDGES = ("tau1_min", "tau2_max", "ratio")


def pairs():
    """Every (i, j) grid index pair with TAU_GRID[j] >= MIN_RATIO x TAU_GRID[i], as an array of shape (P, 2)."""
    i, j = np.meshgrid(np.arange(len(TAU_GRID)), np.arange(len(TAU_GRID)), indexing="ij")
    ok = TAU_GRID[j] >= MIN_RATIO * TAU_GRID[i] * (1 - 1e-12)
    return np.column_stack([i[ok], j[ok]])


def loadings(t, tau1, tau2):
    """The four NSS loadings at maturities `t` (years): shape (..., len(t), 4) for broadcastable taus."""
    t = np.asarray(t, dtype=float)
    x1, x2 = t / np.asarray(tau1)[..., None], t / np.asarray(tau2)[..., None]
    l1, l2 = (1 - np.exp(-x1)) / x1, (1 - np.exp(-x2)) / x2
    return np.stack([np.ones_like(x1), l1, l1 - np.exp(-x1), l2 - np.exp(-x2)], axis=-1)


@lru_cache(maxsize=64)
def _projections(t):
    """For maturities `t` (a tuple, years): every pair's design (P, n, 4) and its pseudo-inverse (P, 4, n)."""
    p = pairs()
    design = loadings(np.array(t), TAU_GRID[p[:, 0]], TAU_GRID[p[:, 1]])
    return design, np.linalg.pinv(design)


def _edge(i, j):
    """Which edges of the (tau1, tau2) grid the pair (i, j) is on, joined by '+', or '' inside it."""
    first_j = np.flatnonzero(TAU_GRID >= MIN_RATIO * TAU_GRID[i] * (1 - 1e-12))[0]
    on = [i == 0, j == len(TAU_GRID) - 1, j == first_j]
    return "+".join(name for name, hit in zip(EDGES, on) if hit)


def fit(t, y):
    """The best NSS fit to spot rates `y` (percent) at maturities `t` (years), over the grid.

    Returns a dict: ``beta`` (4 betas, percent), ``tau1`` and ``tau2`` (years),
    ``rmse_bp`` (root mean squared error at the nodes, bp) and ``edge`` (the
    grid edges the optimum sits on, '' if none).
    """
    t, y = np.asarray(t, dtype=float), np.asarray(y, dtype=float)
    design, pinv = _projections(tuple(t))
    beta = pinv @ y                                    # (P, 4)
    sse = (((design @ beta[..., None])[..., 0] - y) ** 2).sum(axis=1)
    best = int(np.argmin(sse))
    i, j = pairs()[best]
    return {"beta": beta[best], "tau1": TAU_GRID[i], "tau2": TAU_GRID[j],
            "rmse_bp": float(np.sqrt(sse[best] / len(y))) * 100.0, "edge": _edge(i, j)}


def spot(fitted, t):
    """The fitted curve at maturities `t` (years), percent."""
    return loadings(t, fitted["tau1"], fitted["tau2"]) @ fitted["beta"]


def daily(fitted, first, last, year_days):
    """The fitted curve at every whole day from maturity `first` to `last` (months), and at both, by months.

    What `curves/forward.py` reads a meeting window off: its s t is linear
    between nodes, so with a node on every day each window's forward is the
    fitted curve's own, and the first and last maturities are the published curve's.
    """
    days = np.arange(np.ceil(first * year_days / 12.0), np.floor(last * year_days / 12.0) + 1)
    months = np.union1d([first, last], days * 12.0 / year_days)
    months = months[(months >= first) & (months <= last)]
    return pd.Series(spot(fitted, months / 12.0), index=months)
