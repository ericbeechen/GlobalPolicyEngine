"""The market path and the model path side by side for one session, solved exactly as the panel does.

Reads the cache (run scripts/update_data.py first). Falls back to the last ZQ
session on or before the date given. Before the nowcast's first date (a vintaged
u* exists from 2011-03-04) there is a market path and no model path, and it says so.

    uv run python scripts/run_model_path.py 2024-09-17
"""

import sys
import pandas as pd
from policypath import config, panel
from policypath.calendars import US_BDAY
from policypath.macro import nowcast
from policypath.model import path
from policypath.sources import cache

CCY = "USD"
cfg = config.currency(CCY)
spec, rule = cfg["path"], cfg["rule"]
day = pd.Timestamp(sys.argv[1] if len(sys.argv) > 1 else "2024-09-17")

vintages = cache.log("databento", cfg["policy_futures"], CCY)
sessions = vintages.loc[vintages["date"] <= day, "date"]
if sessions.empty:
    sys.exit(f"No ZQ settlements on or before {day.date()}")
session = sessions.max()
if session != day:
    print(f"No ZQ session on {day.date()}; using last session {session.date()}")

inputs = path.inputs(CCY)
settles = panel.settles_on(vintages[vintages["date"] == session], session)
market = panel.solve_session(session, panel.monthly_strip(settles), config.meetings(CCY),
                             cache.view(inputs["fixings"], session), spec["n_meetings"],
                             spec["min_forward_days"], spec["min_regime_days"])
m = market.meetings
print(f"{session:%Y-%m-%d}: EFFR now {market.summary['rate_now']:.3f} (last fixing {market.summary['last_fixing']:.2f})")

# As the panel does: a session on a Fed holiday (Columbus, Veterans Day) reads the business day before.
macro_day = US_BDAY.rollback(session)
if macro_day < pd.Timestamp(cfg["macro"]["start"]):
    print(f"\nNo model path before {cfg['macro']['start']}: the nowcast needs a vintaged u* (CBO's NROU, "
          "first vintage 2011-02-02) and average hourly earnings (2011-03-04).\n")
    print(m[["k", "announcement_date", "effective_date", "rate"]].rename(columns={"rate": "market"})
          .to_string(index=False, formatters={"market": "{:.3f}".format}))
    sys.exit(0)

macro = nowcast.nowcast(macro_day, CCY)
summary, model = path.model_path(session, m["effective_date"], macro, inputs["sep"], inputs["target"],
                                 inputs["fixings"], rule)
sep = "none yet: Taylor's 2%" if pd.isna(summary["sep_date"]) else f"SEP {summary['sep_date']:%Y-%m-%d}"
print(f"core PCE {summary['inflation']:.2f}% ({macro['infl_month']:%Y-%m}), "
      f"u - u* {summary['u_gap']:+.2f}pp ({macro['gap_month']:%Y-%m}), r* {summary['rstar']:.2f} ({sep})")
print(f"rule's notional rate {summary['notional']:.2f}%"
      + (f", below the ELB: floored at {rule['elb']}" if summary["at_elb"] else "")
      + f"; midpoint in force {summary['r0']:.3f}; EFFR - midpoint {summary['spread_bp']:+.1f}bp\n")
out = m[["k", "announcement_date", "effective_date"]].assign(
    market=m["rate"].to_numpy(), model=model["model"].to_numpy())
out["gap_bp"] = (out["market"] - out["model"]) * 100
print(out.to_string(index=False, formatters={"market": "{:.3f}".format, "model": "{:.3f}".format,
                                            "gap_bp": "{:+.1f}".format}))
