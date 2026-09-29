"""Where a currency is in its policy cycle on each session, in real time: the ELB state and the cycle's regime.

Shared by the ELB treatment (week 8), the regime tables (week 10) and the
credit bridge (week 12), so all three condition on one definition.

**The ELB state** (`elb_state`): the policy rate on its floor *and* the rule's
notional below it. With both on the floor the model path is flat at the floor
and has no view, so a gap there is not a disagreement with the rule. The state
is the model's, not the data's: it moves with r* and the coefficients. A rate
on the floor with the rule above it (USD 2014-15: the rule wanted lift-off
before the Fed gave it) is not in the state. The state has no minimum
duration, so it flickers where the rule hovers at the floor: GBP 2020-21 goes
ELB, cutting (18 sessions), ELB, hold after cuts (45), ELB (18), hold after
cuts, four ELB entries against USD's two. A treatment that trades at each
entry and exit pays for every flicker.

**The regime** (`regimes`), from the policy rate's moves up to and including
the session, so a session's regime was knowable on it. In order of
precedence:

- ``elb``: the ELB state, whatever the moves say;
- ``early_hiking``: under `EARLY_MONTHS` (12) since the cycle's first hike, the
  first hike after the last cut, and no cut since;
- ``late_hiking``: `EARLY_MONTHS` or more into the cycle, no cut, and a hike in
  the last `RECENT_MONTHS` (6);
- ``hold_after_hikes``: the last move a hike, `RECENT_MONTHS` or more ago;
- ``cutting``: the last move a cut, less than `RECENT_MONTHS` ago;
- ``hold_after_cuts``: otherwise (the last move a cut, 6 months or more ago, or
  no move yet in the history).

Twelve months splits a cycle where the commentary does: a year of hikes is
confirmation, the second year is where "one too many" starts. Six months is
two or three meetings without a move, the usual reading of a pause. Early
hiking takes precedence over a pause inside the first year (USD 2016, a year
between the first and second hikes, is early hiking throughout), and a hike
takes precedence over a cut less than six months before it (a new cycle
starts at its first hike). A history that starts before the first move it
sees calls the sessions before it ``hold_after_cuts``; both currencies' panels
start on the floor after the 2008-09 cuts, so that is also what happened.
Months are calendar months (`pandas.DateOffset`). See notes/DECISIONS.md (D8).
"""

import pandas as pd

AT_FLOOR = 1e-9      # percent: the policy rate is on its floor within this, and a move is larger than it
EARLY_MONTHS = 12    # the first year of a hiking cycle is early; later is late
RECENT_MONTHS = 6    # a move this recent is the cycle's current direction; none for this long is a hold

ELB = "elb"
STATES = (ELB, "early_hiking", "late_hiking", "hold_after_hikes", "cutting", "hold_after_cuts")
HIKING = ("early_hiking", "late_hiking")


def elb_state(model):
    """By session: the policy rate on its floor (``r0 - elb`` within `AT_FLOOR`) and the rule below it (``at_elb``).

    `model` is the model panel (`model.path.build`'s summaries): session, r0,
    elb, at_elb. Returns a bool Series indexed by session; a missing
    ``at_elb`` is not in the state.
    """
    on_floor = model["r0"].to_numpy() - model["elb"].to_numpy() <= AT_FLOOR
    below = model["at_elb"].fillna(False).astype(bool).to_numpy()
    return pd.Series(on_floor & below, index=pd.DatetimeIndex(model["session"]), name="elb_state")


def moves(rate):
    """The policy rate's changes, in percent, indexed by the first session on the new rate. `rate` is by session.

    A missing rate carries the one before it, so a gap in the history does
    not hide the move across it.
    """
    change = rate.sort_index().ffill().diff()
    return change[change.abs() > AT_FLOOR]


def _last(when, sessions):
    """By session: the latest date in `when` on or before it (NaT before the first)."""
    marked = pd.Series(pd.NaT, index=sessions, dtype="datetime64[ns]")
    marked[when] = when
    return marked.ffill()


def regimes(rate, elb):
    """The regime on every session of `rate` (the policy rate in force, percent, by session): one of `STATES`.

    `elb` is the ELB state by session (`elb_state`), aligned to `rate`; a
    session missing from it, or NaN in it, is not in the state. Uses only
    moves on or before each session.
    """
    rate = rate.sort_index()
    sessions = rate.index
    step = moves(rate)
    hikes, cuts = step.index[step > 0], step.index[step < 0]
    last_hike, last_cut = _last(hikes, sessions), _last(cuts, sessions)
    # A cycle's first hike: a hike with no hike since the last cut before it.
    before = _last(hikes, sessions).shift(1).reindex(hikes)
    cut_before = last_cut.reindex(hikes)
    first = hikes[before.isna().to_numpy() | (cut_before > before).to_numpy()]
    cycle = _last(first, sessions)

    day = sessions.to_series(index=sessions)
    hiking = last_hike.notna() & (last_cut.isna() | (last_hike > last_cut))
    early = hiking & (day < cycle + pd.DateOffset(months=EARLY_MONTHS))
    late = hiking & ~early & (day < last_hike + pd.DateOffset(months=RECENT_MONTHS))
    cutting = ~hiking & last_cut.notna() & (day < last_cut + pd.DateOffset(months=RECENT_MONTHS))
    state = pd.Series("hold_after_cuts", index=sessions, name="regime")
    state[hiking] = "hold_after_hikes"
    state[late] = "late_hiking"
    state[early] = "early_hiking"
    state[cutting] = "cutting"
    state[elb.reindex(sessions).fillna(False).astype(bool)] = ELB
    return state


def states(model):
    """`regimes` on a model panel: the policy rate in force (``r0``) and `elb_state`, by session."""
    return regimes(pd.Series(model["r0"].to_numpy(), index=pd.DatetimeIndex(model["session"])), elb_state(model))


def episodes(state):
    """Runs of one regime in session order: state, first, last (sessions) and sessions (count), one row a run."""
    state = state.sort_index()
    run = (state != state.shift()).cumsum()
    sessions = state.index.to_series(index=state.index)
    grouped = pd.DataFrame({"state": state, "session": sessions, "run": run}).groupby("run")
    out = grouped.agg(state=("state", "first"), first=("session", "min"), last=("session", "max"),
                      sessions=("session", "size"))
    return out.reset_index(drop=True)
