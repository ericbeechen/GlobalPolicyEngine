"""The traded instrument: the implied rate after one meeting.

The position is always at the k-th meeting ahead, so which meeting that is
rolls forward as meetings pass. The change credited to a session is the move in
the meeting held *into* it, the one that was k-th at the previous session.
Measuring the k-th rate on each day instead would book the jump from one
meeting to the next as P&L.
"""

import numpy as np
import pandas as pd


def held_rate_change(sessions, meetings, k):
    """Per session, the change in pp in the rate after the meeting that was k-th at the previous session.

    `sessions` and `meetings` are the panel's frames. A meeting that has taken
    effect by the next session is priced there at that session's current rate
    (``rate_now``): its decision is then the rate in force.
    """
    ok = sessions[sessions["error"].isna()].set_index("session").sort_index()
    rate = meetings.set_index(["session", "effective_date"])["rate"]
    held = meetings[meetings["k"] == k].set_index("session")["effective_date"].reindex(ok.index)
    days = ok.index
    out = pd.Series(np.nan, index=days[1:], name="change")
    for prev, day in zip(days[:-1], days[1:]):
        meeting = held[prev]
        if pd.isna(meeting):
            continue
        before = rate.get((prev, meeting))
        after = ok.at[day, "rate_now"] if meeting <= day else rate.get((day, meeting), np.nan)
        out[day] = after - before
    return out
