"""Implied FOMC path from one day's ZQ settlements, solved exactly as the panel does.

Reads the cache (run scripts/update_data.py first). Falls back to the last
session on or before the date given.

    uv run python scripts/run_implied_path.py 2022-06-01
"""

import sys
import pandas as pd
from policypath import config, panel
from policypath.calendars import label_path
from policypath.sources import cache

CCY = "USD"
cfg = config.currency(CCY)
spec = cfg["path"]
day = pd.Timestamp(sys.argv[1] if len(sys.argv) > 1 else "2022-06-01")

vintages = cache.log("databento", cfg["policy_futures"], CCY)
sessions = vintages.loc[vintages["date"] <= day, "date"]
if sessions.empty:
    sys.exit(f"No ZQ settlements on or before {day.date()}")
session = sessions.max()
if session != day:
    print(f"No ZQ session on {day.date()}; using last session {session.date()}")

settles = panel.settles_on(vintages[vintages["date"] == session], session)
fixings = cache.read(cfg["overnight"]["source"], cfg["overnight"]["series"], CCY, session)
strip = panel.monthly_strip(settles)
result = panel.solve_session(session, strip, config.meetings(CCY), fixings, spec["n_meetings"],
                             spec["min_forward_days"], spec["min_regime_days"])

s = result.summary
print(strip[strip.index <= s["last_month"]].to_string())
print(f"\nknown through {s['first_unknown'] - pd.Timedelta(days=1):%Y-%m-%d} "
      f"(last fixing {s['last_fixing']:.2f}); first regime "
      f"{'pinned to it' if s['pinned'] else 'solved'}; residual {s['residual_bp']:.2f}bp\n")
m = result.meetings
path = pd.Series([s["rate_now"], *m["rate"]], index=pd.DatetimeIndex([s["first_unknown"], *m["effective_date"]]))
print(label_path(path, config.meetings(CCY)).to_string(
    index=False, na_rep="-",
    formatters={"announced": lambda d: "-" if pd.isna(d) else f"{d:%Y-%m-%d}",
                "effective": lambda d: f"{d:%Y-%m-%d}",
                "rate": "{:.4f}".format,
                "move_bp": lambda v: "-" if pd.isna(v) else f"{v:+.1f}"}))
