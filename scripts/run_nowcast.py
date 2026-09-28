"""The macro nowcast as it could have been read at the end of one date, for any configured currency.

Reads the cache only, so it needs no .env (run scripts/update_data.py
--macro-only first). Prints every field the nowcast returned, grouped as the
nowcast builds them: inflation, the unemployment gap, activity where the config
asks for it, then the latest month each input series had reached.

    uv run python scripts/run_nowcast.py 2024-01-20
    uv run python scripts/run_nowcast.py 2024-01-20 --ccy GBP
"""

import argparse
import pandas as pd
from policypath import config
from policypath.macro.nowcast import nowcast
from policypath.macro.vintage import VintagePanel

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("date", nargs="?", default="2024-01-20")
parser.add_argument("--ccy", default=config.enabled()[0], help="default: the first enabled currency")
args = parser.parse_args()
spec = config.currency(args.ccy)["macro"]
day = pd.Timestamp(args.date)
panel = VintagePanel.from_cache(args.ccy)
n = nowcast(day, args.ccy, panel)


def show(v):
    if isinstance(v, pd.Timestamp):
        return f"{v:%Y-%m-%d}"
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def group(field):
    if field.startswith(("infl_", "target_", "bridge_", "wages_")) or field == "n_bridged":
        return "inflation"
    return "gap" if field in ("gap_month", "u", "u_star", "u_gap") else "activity"


print(f"as_of {show(n['as_of'])}   published {show(n['published'])}")
for name in ["inflation", "gap", "activity"]:
    fields = {k: v for k, v in n.items() if k not in ("as_of", "published") and group(k) == name}
    if fields:
        print(f"{name:<11} " + "  ".join(f"{k} {show(v)}" for k, v in fields.items()))
months = {s: f"{panel.series(s, day).dropna().index[-1]:%Y-%m}" for s in spec["series"] if s not in spec["projections"]}
print("last month  " + "  ".join(f"{s} {m}" for s, m in months.items()))
