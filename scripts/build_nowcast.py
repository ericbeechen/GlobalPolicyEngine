"""Nowcast every business day from the macro start to today, and report.

Reads only the cache (run scripts/update_data.py --macro-only first). Writes
the panel to data/panel/<ccy>_macro.parquet, reports/nowcast_<ccy>.md and its
figures, and prints the headline. Which report is written is the config's
``macro.report``: the US one (the PCE bridge, activity against GDPNow) or the
UK one (the LFS against the claimant count and PAYE).

    uv run python scripts/build_nowcast.py
    uv run python scripts/build_nowcast.py --ccy GBP
"""

import argparse
from pathlib import Path
import pandas as pd
from policypath import config
from policypath.macro import nowcast
from policypath.macro.vintage import VintagePanel
from policypath.report import labour
from policypath.report import nowcast as us

ROOT = Path(__file__).resolve().parents[1]
REPORTS = {"nowcast": us, "labour": labour}

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

for line in REPORTS[spec.get("report", "nowcast")].write(args.ccy, frame, panel, spec, end, ROOT / "reports"):
    print(line)
print(f"wrote {out / f'{args.ccy}_macro.parquet'}")
