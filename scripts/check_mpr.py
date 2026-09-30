"""Check the path against the central bank's own published conditioning paths, and report.

For every enabled currency whose config has ``validation.conditioning`` (GBP:
the Bank of England's MPR conditioning paths). Reads data/panel/<ccy>_*.parquet
(run scripts/build_panel.py --ccy <ccy> first), the overnight fixings from the
cache and the committed ground truth the config names. Writes
reports/mpr_check_<ccy>.md and prints the summary.

    uv run python scripts/check_mpr.py
    uv run python scripts/check_mpr.py --ccy GBP
"""

import argparse
from pathlib import Path
from policypath import config, panel
from policypath.calendars import BDAYS
from policypath.report import conditioning
from policypath.sources import cache

ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--ccy", nargs="+", default=None, help="default: every enabled currency with a conditioning check")
args = parser.parse_args()
currencies = args.ccy or [c for c in config.enabled() if "conditioning" in config.currency(c).get("validation", {})]
if not currencies:
    raise SystemExit("no enabled currency has validation.conditioning in config/currencies.yml")

for ccy in currencies:
    cfg = config.currency(ccy)
    check = cfg.get("validation", {}).get("conditioning")
    if check is None:
        raise SystemExit(f"{ccy} has no validation.conditioning in config/currencies.yml")
    sessions, meetings = panel.load(ccy, "sessions"), panel.load(ccy, "meetings")
    fixings = cache.log(cfg["overnight"]["source"], cfg["overnight"]["series"], ccy)
    summary, detail = conditioning.checks(conditioning.truth(ROOT / check["truth"]), sessions, meetings, fixings,
                                          BDAYS[cfg["calendar"]])
    out = ROOT / "reports" / f"mpr_check_{ccy}.md"
    out.write_text(conditioning.markdown(ccy, summary, detail, check["examples"]), encoding="utf-8")
    print(summary.round(2).to_string(index=False))
    diff = detail["diff_bp"]
    print(f"\n{ccy} all quarters: mean {diff.mean():+.2f}bp, mean abs {diff.abs().mean():.2f}bp")
    print(f"wrote {out}")
