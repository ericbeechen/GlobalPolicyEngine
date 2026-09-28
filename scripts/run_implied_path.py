"""The market-implied path from one session's data, solved exactly as the panel solves it, for any configured currency.

Reads the cache (run scripts/update_data.py first). Falls back to the last
session on or before the date given. Prints what the backend the config names
solved from (a futures strip's contracts, or a curve's nodes), then the path.

    uv run python scripts/run_implied_path.py 2022-06-01
    uv run python scripts/run_implied_path.py 2024-08-02 --ccy GBP
"""

import argparse
import pandas as pd
from policypath import config, market
from policypath.calendars import label_path

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("date", nargs="?", default="2022-06-01")
parser.add_argument("--ccy", default=config.enabled()[0], help="default: the first enabled currency")
args = parser.parse_args()
cfg = config.currency(args.ccy)
day = pd.Timestamp(args.date)

ext = market.extractor(args.ccy)
sessions = ext.sessions()[ext.sessions() <= day]
if sessions.empty:
    raise SystemExit(f"No {args.ccy} session on or before {day.date()}")
session = sessions[-1]
if session != day:
    print(f"No {args.ccy} session on {day.date()}; using last session {session.date()}")

result = ext.session(session)
s = result.summary
if isinstance(ext, market.FuturesStripExtractor):
    strip = market.monthly_strip(market.settles_on(ext.by_date[session], session, ext.bday, ext.after))
    print(strip[strip.index <= s["last_month"]].to_string())
else:
    print(f"{cfg['market']['label']}: {s['n_nodes']} nodes to {s['last_tenor']} months")
print(f"\nknown through {s['first_unknown'] - pd.Timedelta(days=1):%Y-%m-%d} "
      f"(last fixing {s['last_fixing']:.2f}); first regime "
      f"{'pinned to it' if s['pinned'] else 'solved'}; residual {s['residual_bp']:.2f}bp\n")
m = result.meetings
path = pd.Series([s["rate_now"], *m["rate"]], index=pd.DatetimeIndex([s["first_unknown"], *m["effective_date"]]))
print(label_path(path, config.meetings(args.ccy)).to_string(
    index=False, na_rep="-",
    formatters={"announced": lambda d: "-" if pd.isna(d) else f"{d:%Y-%m-%d}",
                "effective": lambda d: f"{d:%Y-%m-%d}",
                "rate": "{:.4f}".format,
                "move_bp": lambda v: "-" if pd.isna(v) else f"{v:+.1f}"}))
