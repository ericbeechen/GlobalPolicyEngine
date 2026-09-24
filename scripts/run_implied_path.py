"""Implied FOMC path from one day's ZQ settlements.

    uv run python scripts/run_implied_path.py 2022-06-01
"""
import sys
from pathlib import Path
import pandas as pd
from policypath.calendars import label_path, next_meetings
from policypath.curves.policy_path import implied_path
from policypath.sources import rates

N_MEETINGS = 8

day = pd.Timestamp(sys.argv[1] if len(sys.argv) > 1 else "2022-06-01")

s = rates.settlements(day - pd.Timedelta(days=7), day + pd.Timedelta(days=4))
s = s[s["asset"] == "ZQ"]
sessions = s["trade_date"].dt.date
if not (sessions <= day.date()).any():
    sys.exit(f"No final ZQ settlements on or in the week before {day.date()}")
session = sessions[sessions <= day.date()].max()
if session != day.date():
    print(f"No ZQ session on {day.date()}; using last session {session}")
zq = s[sessions == session]

meetings = pd.read_csv(Path(__file__).parents[1] / "config/meetings/fomc.csv",
                       parse_dates=["announcement_date", "effective_date"])
upcoming = next_meetings(day, meetings, N_MEETINGS)
if len(upcoming) < N_MEETINGS:
    sys.exit(f"fomc.csv has only {len(upcoming)} meetings after {day.date()}")
last_effective = upcoming["effective_date"].iloc[-1]
# One month past the last meeting's, so its new rate is pinned by a whole contract.
horizon = last_effective.to_period("M") + 1

month = zq["expiration"].dt.tz_localize(None).dt.to_period("M")
implied_avg = pd.Series(zq["implied_rate"].values, index=month).sort_index()
if implied_avg.index.max() < horizon:
    sys.exit(f"ZQ strip on {session} ends {implied_avg.index.max()}, "
             f"short of {horizon} needed for {N_MEETINGS} meetings")
implied_avg = implied_avg[implied_avg.index <= horizon]

path = implied_path(implied_avg, meetings["effective_date"])
labelled = label_path(path, meetings)
labelled = labelled[labelled["effective"] <= last_effective]

print(implied_avg.to_string())
print()
print(labelled.to_string(index=False, na_rep="-",
                         formatters={"announced": lambda d: "-" if pd.isna(d) else f"{d:%Y-%m-%d}",
                                     "effective": lambda d: f"{d:%Y-%m-%d}",
                                     "rate": "{:.4f}".format,
                                     "move_bp": lambda v: "-" if pd.isna(v) else f"{v:+.1f}"}))
