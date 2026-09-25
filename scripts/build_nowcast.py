"""Nowcast every Fed business day from the macro start to today, and report.

Reads only the cache (run scripts/update_data.py --macro-only first). Writes
the panel to data/panel/<ccy>_macro.parquet, reports/nowcast_<ccy>.md and its
figures, and prints the headline.

    uv run python scripts/build_nowcast.py
"""

import argparse
from pathlib import Path
import pandas as pd
from policypath import config
from policypath.macro import nowcast
from policypath.macro.vintage import VintagePanel
from policypath.report import nowcast as report

ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--ccy", default="USD")
parser.add_argument("--end", default=None, help="last as_of (default: today)")
args = parser.parse_args()
spec = config.currency(args.ccy)["macro"]
end = pd.Timestamp(args.end) if args.end else pd.Timestamp.today().normalize()

panel = VintagePanel.from_cache(args.ccy)
frame = nowcast.build(args.ccy, spec["start"], end, panel)
out = ROOT / "data" / "panel"
out.mkdir(parents=True, exist_ok=True)
frame.to_parquet(out / f"{args.ccy}_macro.parquet", index=False)

headline, tables, gap = report.checks(frame, panel, spec, end)
reports = ROOT / "reports"
(reports / f"nowcast_{args.ccy}.md").write_text(report.markdown(args.ccy, headline, tables))
drawn = report.write_figures(gap, reports / "figures", args.ccy)

print(f"{headline['fed_business_days']} days, {headline['first']:%Y-%m-%d} .. {headline['last']:%Y-%m-%d}")
print(f"n_bridged: {headline['n_bridged']}   activity_n: {headline['activity_n']}")
for title in list(tables)[:2] + list(tables)[-1:]:
    print(f"\n{title}\n{tables[title].round(3).to_string(index=False)}")
print(f"\nwrote {out / f'{args.ccy}_macro.parquet'}, {reports / f'nowcast_{args.ccy}.md'} and {len(drawn)} figures")
