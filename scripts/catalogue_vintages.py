"""Write reports/vintages_<ccy>.md: what each ALFRED series covers in real time.

Reads only the cache (run scripts/update_data.py --macro-only first).

    uv run python scripts/catalogue_vintages.py
"""

import argparse
from pathlib import Path
import pandas as pd
from policypath.report import vintages

ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--ccy", default="USD")
parser.add_argument("--as-of", default=None, help="catalogue the cache as it stood on this date")
args = parser.parse_args()
as_of = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp.today().normalize()

tables = vintages.catalogue(args.ccy, as_of)
out = ROOT / "reports" / f"vintages_{args.ccy}.md"
out.write_text(vintages.markdown(args.ccy, tables, as_of))
print(tables["Series"].drop(columns=["title", "units"]).to_string(index=False))
print(f"\n{len(tables['Revision calendar'])} large revisions since {vintages.SINCE[:4]}; wrote {out}")