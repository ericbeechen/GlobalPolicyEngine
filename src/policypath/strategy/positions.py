"""The position a sleeve decides at each close: the rule on its z, the ELB treatment, and the sizing.

Everything here is decided at a session's close from what was known there.
`strategy/expression.py` (`run`) executes it ``lag`` sessions later and
credits the P&L; `strategy/costs.py` charges what it trades.

**Rules.** ``linear``: s = z (week 6's sizing). ``hysteresis(z, enter,
exit)``: from flat, take side sign(z) where |z| >= ``enter``; a held side is
closed where z x side <= ``exit`` and, the same session, the other side is
taken if |z| >= ``enter``; a missing z closes the position (NaN is flat, never
"hold what you have"). The book's pair is (``positions.enter``,
``positions.exit``) = (1.0, 0.0): enter at a one-sd gap, leave where the gap
is back at its trailing mean, the move the edge measures
(`strategy/carry.py`). It was fixed in config before the grid of pairs was
run (notes/DECISIONS.md, K5); the grid is reported around it, never
searched.

**The carry filter** (``positions.carry_filter``, a diagnostic, off): an entry
is skipped on a session whose expected quarter does not pay for its bleed
(`carry.bleed`, E_h < 0, from `expression.ahead`); before phi_h exists there is
nothing to filter on and the entry stands. It acts on entries, so only under
hysteresis.

**The ELB state** is `regimes.elb_state`: the policy rate on its floor and the
rule's notional below it, so the model path is flat at the floor and has no
view. It is the model's state, not the data's: it moves with r* and the
coefficients. A cross sleeve is in it when either currency is. Treatments
(``evaluation.elb.chosen``, ``flat`` proposed, K6):

- ``flat``: no position in the state. z is masked there, so the hysteresis
  starts flat again on leaving it; the close into the state and the re-open out
  of it are traded and charged. Statistics leave out the sessions the state
  has the sleeve flat and not trading: those where both the position held into
  the session and the one put on at its close were decided in the state. The
  last pre-state P&L and both boundary trades stay in.
- ``exclude``: the ``hold`` run, with every position decided in the state
  left out: its P&L (on the sessions it is held into) and the cost of putting
  it on (at the close it is traded). A session counts where either survives,
  so the first trade out of the state, decided outside it, is charged on a
  session whose P&L is left out. No boundary trades, because the position
  never closes.
- ``hold``: positions straight through; statistics over every session, and the
  P&L of the positions decided in the state on its own line.

**Sizing** (`size`): ``unit`` is q = s, one book DV01 a unit of s.
``vol_scaled`` is q = s x sigma_target / sigma_t, sigma_t the sleeve's unit P&L
sd, EWMA, strictly before t and floored (`strategy/risk.py`), and
sigma_target = ``book.vol_target`` x ``book.capital`` a year (5% of $100m), a
day: each sleeve alone at the book's risk. The multiplier sigma_target / sigma
is re-set on a new side and otherwise only when its target moves more than
``risk.no_trade_band`` (10%) from the one held, so vol noise does not trade
every day (K7, K8). Sharpe and the breakeven cost do not depend on
sigma_target; the P&L's units do.
"""

import numpy as np
import pandas as pd
from policypath import regimes
from policypath.backtest.engine import SESSIONS_PER_YEAR

RULES = ("linear", "hysteresis")
TREATMENTS = ("flat", "exclude", "hold")
SIZING = ("vol_scaled", "unit")


def linear(z):
    """s = z at each close; a missing z is flat (0)."""
    return z.fillna(0.0).rename("side")


def hysteresis(z, enter, exit, allow=None):
    """The side, -1, 0 or +1, at each close under the (enter, exit) pair (module docstring).

    `allow` (bool by session, missing = True) gates entries only: an entry,
    including the other side after a same-session exit, needs it True.
    """
    v = z.to_numpy(dtype=float)
    ok = np.ones(len(v), dtype=bool) if allow is None else allow.reindex(z.index).fillna(True).to_numpy(dtype=bool)
    out = np.zeros(len(v))
    side = 0.0
    for i, x in enumerate(v.tolist()):
        if side and (x != x or x * side <= exit):
            side = 0.0
        if not side and abs(x) >= enter and ok[i]:
            side = 1.0 if x > 0 else -1.0
        out[i] = side
    return pd.Series(out, index=z.index, name="side")


def pays(ahead):
    """The carry filter's gate by session from `expression.ahead`: False where the expected quarter is below 0."""
    return ahead["pays"].astype("boolean").fillna(True).astype(bool)


def elb(sleeve, models):
    """In the ELB state on each of the sleeve's sessions: either of its currencies' `regimes.elb_state`.

    `models` is {ccy: model panel}. A session a currency's panel lacks is not in its state.
    """
    days = sleeve.sessions
    state = np.zeros(len(days), dtype=bool)
    for ccy in sleeve.ccys:
        state |= regimes.elb_state(models[ccy]).reindex(days, fill_value=False).to_numpy(dtype=bool)
    return pd.Series(state, index=days, name="elb")


def decide(z, rule, enter, exit, state, treatment, allow=None):
    """The rule's s at each close under an ELB `treatment`: z is masked in the `state` under ``flat`` only."""
    if treatment not in TREATMENTS:
        raise ValueError(f"ELB treatment {treatment!r} is not one of {TREATMENTS}")
    if treatment == "flat":
        z = z.where(~state.reindex(z.index, fill_value=False))
    if rule == "linear":
        return linear(z)
    if rule == "hysteresis":
        return hysteresis(z, enter, exit, allow)
    raise ValueError(f"rule {rule!r} is not one of {RULES}")


def samples(state, treatment, lag):
    """(kept, pnl, cost, elb_line) by session: which sessions a treatment's statistics are over, and what counts in them.

    A position decided at d is held into d + lag + 1, so the held position's
    decision is in the state where the state, shifted lag + 1, is; the one put
    on at the close, where it is shifted lag. ``pnl`` and ``cost`` say whether
    a kept session's P&L (the held position's) and its close's cost (the one
    put on) count: both, except under ``exclude``, which drops the P&L of a
    position decided in the state and the cost of putting one on.
    ``elb_line`` is the held position's decision in the state.
    """
    held = state.shift(lag + 1, fill_value=False).astype(bool)
    put = state.shift(lag, fill_value=False).astype(bool)
    every = pd.Series(True, index=state.index)
    kept = {"hold": every, "exclude": ~(held & put), "flat": ~(held & put)}[treatment]
    pnl, cost = (~held, ~put) if treatment == "exclude" else (every, every)
    return kept.rename("kept"), pnl.rename("pnl"), cost.rename("cost"), held.rename("elb_line")


def target_sigma(book):
    """sigma_target: the book's vol target a year in book currency, a session (``SESSIONS_PER_YEAR``)."""
    return book["book"]["vol_target"] * book["book"]["capital"] / np.sqrt(SESSIONS_PER_YEAR)


def vol_scaled(s, sigma, target, band):
    """q = s x target / sigma, the multiplier re-set on a new side, else only outside `band` of the one held.

    `sigma` is by session, already lagged and floored; where it is missing the
    sleeve is flat. A new side is a change of sign of s (from 0, or across 0).
    """
    goal = (target / sigma.reindex(s.index)).to_numpy(dtype=float)
    v = s.to_numpy(dtype=float)
    out = np.zeros(len(v))
    m, prev = np.nan, 0.0
    for i, (x, g) in enumerate(zip(v.tolist(), goal.tolist())):
        if x == 0 or g != g:
            m, prev = np.nan, 0.0
            continue
        sign = 1.0 if x > 0 else -1.0
        if sign != prev or m != m or abs(g / m - 1.0) > band:
            m = g
        out[i], prev = x * m, sign
    return pd.Series(out, index=s.index, name="decided")


def size(s, mode, sigma=None, target=None, band=0.0):
    """The decided book DV01 on the sleeve's first leg: ``unit`` (q = s) or ``vol_scaled`` (`vol_scaled`)."""
    if mode == "unit":
        return s.rename("decided")
    if mode == "vol_scaled":
        return vol_scaled(s, sigma, target, band)
    raise ValueError(f"sizing {mode!r} is not one of {SIZING}")
