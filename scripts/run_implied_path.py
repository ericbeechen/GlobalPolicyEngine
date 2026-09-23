"""Implied FOMC path from one day's ZQ settlements.

    uv run python scripts/run_implied_path.py 2022-06-01
"""
import sys
from pathlib import Path
import pandas as pd
from policypath.curves.policy_path import implied_path
from policypath.sources import rates

HORIZON_MONTHS = 12

day = pd.Timestamp(sys.argv[1] if len(sys.argv) > 1 else "2022-06-01")

# Read a week back so weekends and holidays fall back to the last session on or
# before `day`. Final settles arrive a day or two later, so read a few days past it.
s = rates.settlements(day - pd.Timedelta(days=7), day + pd.Timedelta(days=4))
s = s[s["asset"] == "ZQ"]
sessions = s["trade_date"].dt.date
if not (sessions <= day.date()).any():
    sys.exit(f"No final ZQ settlements on or in the week before {day.date()}")
session = sessions[sessions <= day.date()].max()
if session != day.date():
    print(f"No ZQ session on {day.date()}; using last session {session}")
zq = s[sessions == session]

# ZQ expires on the last business day of its contract month.
month = zq["expiration"].dt.tz_localize(None).dt.to_period("M")
implied_avg = pd.Series(zq["implied_rate"].values, index=month).sort_index()
implied_avg = implied_avg[implied_avg.index <= day.to_period("M") + HORIZON_MONTHS]

# Keep meetings earlier in the front month too: that contract's average includes them.
meetings = pd.read_csv(Path(__file__).parents[1] / "config/meetings/fomc.csv",
                       parse_dates=["announcement_date", "effective_date"])

print(implied_avg.to_string())
print(implied_path(implied_avg, meetings["effective_date"]).to_string())
