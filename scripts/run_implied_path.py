"""Implied FOMC path from one day's ZQ settlements.

    uv run python scripts/run_implied_path.py 2022-06-01
"""
import sys
from pathlib import Path
import pandas as pd
from policypath.calendars import label_path
from policypath.curves.policy_path import implied_path
from policypath.sources import rates

HORIZON_MONTHS = 12

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

month = zq["expiration"].dt.tz_localize(None).dt.to_period("M")
implied_avg = pd.Series(zq["implied_rate"].values, index=month).sort_index()
implied_avg = implied_avg[implied_avg.index <= day.to_period("M") + HORIZON_MONTHS]

meetings = pd.read_csv(Path(__file__).parents[1] / "config/meetings/fomc.csv",
                       parse_dates=["announcement_date", "effective_date"])

path = implied_path(implied_avg, meetings["effective_date"])
labelled = label_path(path, meetings)

print(implied_avg.to_string())
print()
print(labelled.to_string(index=False, na_rep="-",
                         formatters={"announced": lambda d: "-" if pd.isna(d) else f"{d:%Y-%m-%d}",
                                     "effective": lambda d: f"{d:%Y-%m-%d}",
                                     "rate": "{:.4f}".format,
                                     "move_bp": lambda v: "-" if pd.isna(v) else f"{v:+.1f}"}))
