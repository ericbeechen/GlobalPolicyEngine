"""Who closes the gap: when the market and the rule disagree, which of the two moves.

For every session t of a currency with a z, outside the ELB state (the rule has
no view there), take the meeting that is ``backtest.horizon``-th at t, the one
the outrights trade, and the gap's deviation from its trailing mean at t: the
edge the z trades (|z| x sd, DECISIONS C11), which leaves out the constant
offsets the r*, u* and spread choices put in the bp gap. Over the next h
sessions, measured as the IC measures them, from the session after the close
(t + 1) to h sessions after that, three changes are taken, in bp:

- the market's rate for that same meeting;
- the rule's rate for it;
- the policy rate in force, the rule's starting point.

Each is regressed on the edge with a constant (`credit.ols`): Newey-West,
Bartlett weights to lag h, and the non-overlapping check of every h-th
session from each start offset (`credit.nonoverlapping`), at the IC's
horizons (``evaluation.ic_horizons``). Holding the meeting keeps a meeting
passing from booking the path's slope to the next meeting as a move, as the
credit bridge holds it (D6). A session whose meeting is no longer on the path
h sessions later is left out, every fit is on the sessions all three changes
have, and a session left out keeps its place, so a lag still pairs sessions
that far apart.

**Reading the slopes.** The gap is market minus rule, so a negative slope on
the market, or a positive one on the rule, is that side moving to close it.
The gap's own slope is the market's less the rule's, an identity, and the
share of the edge closed (`closed`) is minus it. `who` words which side did the
closing: a rule's share of it at `MOSTLY` or more is mostly the rule, at
1 - `MOSTLY` or less mostly the market, both in between, and neither when the
gap does not close. The policy rate's slope says how far the central bank's
own rate moved: the rule is inertial from the rate in force, so a move the
market priced and the central bank then made lifts the rule's whole path.

**The bins** are the edge's deciles on each fit's sessions, with the mean edge
and the mean of each change in each: what the explorer draws.
"""

import numpy as np
import pandas as pd
from policypath import config, credit, panel, regimes

BP = 100.0
BINS = 10
MOSTLY = 2 / 3        # the rule's share of the closing from which the rule did it (and up to 1 - MOSTLY, the market)
CHANGES = ("market", "rule", "rate")


def same_meeting(paths, k, at, start, end):
    """(market, rule) change in bp from sessions `start` to `end` for the meeting that was `k`-th at `at`.

    `at`, `start` and `end` are arrays of sessions (NaT allowed); `paths` is the
    paths panel (session, k, effective_date, market, model, in percent). NaN
    where a session is missing or the meeting is not on its path.
    """
    rates = paths.set_index(["session", "effective_date"])[["market", "model"]]
    meeting = paths[paths["k"] == k].set_index("session")["effective_date"].reindex(at).to_numpy()

    def on(days):
        return rates.reindex(pd.MultiIndex.from_arrays([pd.DatetimeIndex(days), meeting])).to_numpy()

    change = (on(end) - on(start)) * BP
    return change[:, 0], change[:, 1]


def later(sessions, j):
    """By position: the session `j` positions later, NaT past the last."""
    out = np.full(len(sessions), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    i = np.arange(len(sessions)) + j
    ok = i < len(sessions)
    out[ok] = sessions.to_numpy()[i[ok]]
    return out


def frame(paths, signal, model, k, h):
    """By every session of the paths: the edge at t, the ELB state, the three changes over h sessions and the gap's.

    The changes run from session t + 1 to t + 1 + h of the currency's own
    sessions. ``keep`` marks the fit's sessions: a z, outside the ELB state,
    and every change there.
    """
    sessions = pd.DatetimeIndex(sorted(paths["session"].unique()))
    sig = signal[signal["k"] == k].set_index("session").reindex(sessions)
    start, end = later(sessions, 1), later(sessions, 1 + h)
    market, rule = same_meeting(paths, k, sessions.to_numpy(), start, end)
    r0 = model.set_index("session")["r0"].reindex(sessions)
    f = pd.DataFrame({"edge": (sig["gap_bp"] - sig["window_mean_bp"]).where(sig["z"].notna()).to_numpy(),
                      "market": market, "rule": rule,
                      "rate": ((r0.shift(-(1 + h)) - r0.shift(-1)) * BP).to_numpy(),
                      "elb": regimes.elb_state(model).reindex(sessions, fill_value=False).to_numpy()},
                     index=sessions)
    f["gap"] = f["market"] - f["rule"]
    f["keep"] = f["edge"].notna() & ~f["elb"] & f[list(CHANGES)].notna().all(axis=1)
    return f


def who(rule_share, closed):
    """Which side closed the gap, in words (module docstring)."""
    if not closed > 0:
        return "neither"
    if rule_share >= MOSTLY:
        return "the rule"
    if rule_share <= 1 - MOSTLY:
        return "the market"
    return "both"


def bins(f, n=BINS):
    """The edge's `n` quantile bins over `f`'s rows: the mean edge, the mean of each change, and the sessions."""
    if len(f) < n:
        return []
    cut = pd.qcut(f["edge"], n, labels=False, duplicates="drop")
    g = f.groupby(cut)
    out = g[["edge", *CHANGES, "gap"]].mean()
    out["sessions"] = g.size()
    return out.reset_index(drop=True).to_dict("records")


def fit(f, h):
    """The slopes of each change on the edge over `f`'s kept sessions (NW lag `h`), the share closed, and the bins."""
    keep = f["keep"].to_numpy()
    X = np.column_stack([np.ones(len(f)), f["edge"].to_numpy(dtype=float)])
    kept = f[keep]
    out = {"h": h, "sessions": int(keep.sum()),
           "first": kept.index.min() if len(kept) else None, "last": kept.index.max() if len(kept) else None}
    for name in (*CHANGES, "gap"):
        if not keep.any():
            out[name] = {"slope": np.nan, "se": np.nan, "t": np.nan, "r2": np.nan, "n": 0,
                         "nonoverlap": {"mean": np.nan, "min": np.nan, "max": np.nan, "n": 0}}
            continue
        y = np.where(keep, f[name].to_numpy(dtype=float), np.nan)
        r = credit.ols(y, X, h)
        no = credit.nonoverlapping(y, X, h)
        out[name] = {"slope": r["beta"][1], "se": r["se"][1], "t": r["t"][1], "r2": r["r2"], "n": r["n"],
                     "nonoverlap": {"mean": no["mean"], "min": no["min"], "max": no["max"], "n": no["n"]}}
    closed = out["rule"]["slope"] - out["market"]["slope"]
    share = out["rule"]["slope"] / closed if closed > 0 else np.nan
    out |= {"closed": closed, "rule_share": share, "who": who(share, closed), "bins": bins(kept)}
    return out


def run(ccy, load=panel.load, cfg=None, horizons=None):
    """One currency's fits at each horizon: {k, horizons: {h: `fit`}}. `load(ccy, name)` reads a panel."""
    cfg = config.currency(ccy) if cfg is None else cfg
    horizons = config.strategy()["evaluation"]["ic_horizons"] if horizons is None else horizons
    paths, signal, model = (load(ccy, name) for name in ("paths", "signal", "model"))
    k = cfg["backtest"]["horizon"]
    return {"k": k, "horizons": {h: fit(frame(paths, signal, model, k, h), h) for h in horizons}}
