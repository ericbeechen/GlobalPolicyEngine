"""The weekly brief, one command from cache to a dated one-page PDF in reports/.

Reads data/panel/ (run build_panel.py, build_nowcast.py and build_model.py for
each currency in config/brief.yml first). Writes reports/brief_<date>.pdf and
prints what it says. `--date` defaults to today; a past date gives the brief
that could have been sent then.

    uv run python scripts/build_brief.py
    uv run python scripts/build_brief.py --date 2026-09-27
"""

import argparse
from pathlib import Path
import pandas as pd
from policypath.report import brief

ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--date", default=None, help="the brief's date (default: today)")
parser.add_argument("--preview", default=None, help="also render the page to this PNG")
args = parser.parse_args()
date = pd.Timestamp(args.date) if args.date else pd.Timestamp.today().normalize()

settings = brief.settings()
horizons = settings["horizons"]
path, said = brief.write(date, ROOT / "reports", brief=settings, preview=args.preview)
for ccy, snap in said["reads"].items():
    m = snap["meetings"].set_index("k")
    print(f"{ccy} {snap['session']:%Y-%m-%d}: gap " + ", ".join(f"k{h} {m.loc[h, 'gap_bp']:+.0f}bp (z {m.loc[h, 'z']:+.1f})"
                                                       for h in horizons))
d = said["differential"]
print(f"differential {said['differential_session']:%Y-%m-%d}: " + ", ".join(
    f"k{h} {d.loc[h, 'diff_bp']:+.0f}bp (z {d.loc[h, 'z']:+.1f})" for h in horizons))
print("\n".join(said["changed"]))
print(said["wrong"])
print(f"wrote {path}")
