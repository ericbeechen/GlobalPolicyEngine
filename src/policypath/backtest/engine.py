"""A signal-agnostic P&L engine: a position series in, a P&L series out.

Crude by design: one instrument, DV01 held fixed, no costs, no sizing beyond
the position it is given. Positions are in DV01 units and positive means
*receive* (long the rate's price), so P&L is -dv01 x position x the rate's change.
Anything that yields a position per session can be run through it: the policy
gap z now, the credit z later.
"""

import numpy as np
import pandas as pd

SESSIONS_PER_YEAR = 252


def run(position, change, dv01=1.0, lag=0):
    """P&L per session.

    `position` is decided at each session's close; `change` is, per session,
    the move since the previous session in the instrument held into it. The
    position earning `change` on session t is the one decided ``lag + 1``
    sessions earlier, so ``lag=0`` trades at the close the signal was read on.
    Returns a frame with position (held), change, pnl and equity, on `change`'s index.
    """
    held = position.reindex(change.index.union(position.index)).shift(lag + 1).reindex(change.index)
    pnl = (-dv01 * held * change).fillna(0.0)
    return pd.DataFrame({"position": held, "change": change, "pnl": pnl, "equity": pnl.cumsum()})


def stats(result, periods=SESSIONS_PER_YEAR):
    """Headline numbers for one `run` result, over the sessions it held a position."""
    live = result[result["position"].notna()]
    pnl = live["pnl"]
    equity = pnl.cumsum()
    vol = pnl.std() * np.sqrt(periods)
    return {
        "first": live.index.min(), "last": live.index.max(), "sessions": len(live),
        "total": pnl.sum(), "per_year": pnl.mean() * periods, "vol": vol,
        "sharpe": pnl.mean() * periods / vol if vol > 0 else np.nan,
        "hit_rate": (pnl[pnl != 0] > 0).mean(),
        "max_drawdown": (equity - equity.cummax()).min(),
        "mean_abs_position": live["position"].abs().mean(),
    }
