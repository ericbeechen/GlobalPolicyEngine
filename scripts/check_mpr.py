"""Check the GBP path against the Bank of England's own MPR conditioning paths, and report.

Reads data/panel/GBP_*.parquet (run scripts/build_panel.py --ccy GBP first),
SONIA from the cache and the committed ground truth in tests/data/boe/.
Writes reports/mpr_check_GBP.md and prints the summary.

    uv run python scripts/check_mpr.py
"""

import argparse
from pathlib import Path
from policypath import config, panel
from policypath.calendars import BDAYS
from policypath.report import conditioning
from policypath.sources import cache

ROOT = Path(__file__).resolve().parents[1]
TRUTH = ROOT / "tests" / "data" / "boe" / "conditioning_paths.csv"
EXAMPLES = ["August 2019", "February 2023", "August 2024"]

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--ccy", default="GBP")
args = parser.parse_args()
cfg = config.currency(args.ccy)

sessions, meetings = panel.load(args.ccy, "sessions"), panel.load(args.ccy, "meetings")
fixings = cache.log(cfg["overnight"]["source"], cfg["overnight"]["series"], args.ccy)
summary, detail = conditioning.checks(conditioning.truth(TRUTH), sessions, meetings, fixings,
                                      BDAYS[cfg["calendar"]])
out = ROOT / "reports" / f"mpr_check_{args.ccy}.md"
out.write_text(conditioning.markdown(args.ccy, summary, detail, EXAMPLES))
print(summary.round(2).to_string(index=False))
print(f"\nall quarters: mean {detail['diff_bp'].mean():+.2f}bp, mean abs {detail['diff_bp'].abs().mean():.2f}bp")
print(f"wrote {out}")
