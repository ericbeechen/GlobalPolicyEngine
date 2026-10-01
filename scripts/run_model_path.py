"""The market path and the model path side by side for one session, for any configured currency.

Reads the cache (run scripts/update_data.py first). The market path comes from
`market.path`, the same call for every currency; the nowcast is computed live
for the session. Falls back to the last session on or before the date given.
Before the nowcast's first date there is a market path and no model path, and
it says so.

    uv run python scripts/run_model_path.py 2024-09-17
    uv run python scripts/run_model_path.py 2024-08-02 --ccy GBP
"""

import argparse
import pandas as pd
from policypath import config, market
from policypath.calendars import BDAYS
from policypath.macro import nowcast
from policypath.model import path

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("date", nargs="?", default="2024-09-17")
parser.add_argument("--ccy", default=config.enabled()[0], help="default: the first enabled currency")
args = parser.parse_args()
cfg = config.currency(args.ccy)
rule, w = cfg["rule"], cfg["report"]["labels"]

p = market.path(args.date, args.ccy)
session, m = p.summary["session"], p.meetings
if session != pd.Timestamp(args.date):
    print(f"No {args.ccy} session on {pd.Timestamp(args.date).date()}; using last session {session.date()}")
print(f"{session:%Y-%m-%d}: {w['rate']} now {p.summary['rate_now']:.3f} (last fixing {p.summary['last_fixing']:.4f})")

# As the panel does: a session on a holiday of the macro calendar reads the business day before.
macro_day = BDAYS[cfg["calendar"]].rollback(session)
if macro_day < pd.Timestamp(cfg["macro"]["start"]):
    print(f"\nNo model path before {cfg['macro']['start']}, the nowcast's first date "
          f"({w['nowcast_start']}).\n")
    print(m[["k", "announcement_date", "effective_date", "rate"]].rename(columns={"rate": "market"})
          .to_string(index=False, formatters={"market": "{:.3f}".format}))
    raise SystemExit(0)

inputs = path.inputs(args.ccy)
macro = nowcast.nowcast(macro_day, args.ccy)
summary, model = path.model_path(session, m["effective_date"], macro, inputs["sep"], inputs["target"],
                                 inputs["fixings"], rule, inputs["projected"])
src = "constant" if pd.isna(summary["sep_date"]) else f"{rule['rstar']['label']} {summary['sep_date']:%Y-%m-%d}"
if pd.isna(summary["sep_date"]) and "before_first" in rule["rstar"]:
    src = f"none yet: {rule['rstar']['before_first_label']}"
print(f"{w['inflation']} {summary['inflation']:.2f}% ({macro['infl_month']:%Y-%m}), "
      f"u - u* {summary['u_gap']:+.2f}pp ({macro['gap_month']:%Y-%m}), r* {summary['rstar']:.2f} ({src})")
print(f"rule's notional rate {summary['notional']:.2f}%"
      + (f", below the ELB: floored at {summary['elb']}" if summary["at_elb"] else "")
      + f"; {w['policy_rate'].lower()} in force {summary['r0']:.3f}; "
        f"{w['rate']} - {w['policy_rate'].lower()} {summary['spread_bp']:+.1f}bp\n")
out = m[["k", "announcement_date", "effective_date"]].assign(
    market=m["rate"].to_numpy(), model=model["model"].to_numpy())
out["gap_bp"] = (out["market"] - out["model"]) * 100
print(out.to_string(index=False, formatters={"market": "{:.3f}".format, "model": "{:.3f}".format,
                                            "gap_bp": "{:+.1f}".format}))
