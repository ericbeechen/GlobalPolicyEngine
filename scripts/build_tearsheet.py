"""The tear sheet, one command from the panels and the cache to a dated one-page PDF in reports/tearsheets/.

Reads data/panel/ (run build_panel.py, build_nowcast.py and build_model.py for
every currency the book trades first) and the cache, as build_portfolio.py
does, and runs the strategy chain in memory: the sleeves, the headline
book, the carry benchmark and the book under each ELB treatment. Writes
reports/tearsheets/tearsheet_<last book session>.pdf, reports/metrics.md,
reports/results/metrics.json and reports/figures/attribution_{light,dark}.png,
and prints the headline. `--preview` also renders the page to a PNG.

    uv run python scripts/build_tearsheet.py
    uv run python scripts/build_tearsheet.py --preview tearsheet.png
"""

import argparse
from pathlib import Path
import time
from policypath.report import attribution, tearsheet

ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--preview", default=None, help="also render the page to this PNG")
args = parser.parse_args()

start = time.perf_counter()
built = attribution.build(log=lambda m: print(f"{m} ({time.perf_counter() - start:.1f}s)"))
n = built.numbers
sr = attribution.sr
print(f"\n{'':<16} {'gross':>13} {'net':>13} {'ex-2022':>13} {'ex-2022-23':>13}")
for r in n["performance"]:
    print(f"{r['name']:<16} {sr(r['gross_sr'], r['gross_se']):>13} {sr(r['net_sr'], r['net_se']):>13} "
          f"{sr(r['ex_2022']['net_sr'], r['ex_2022']['net_se']):>13} "
          f"{sr(r['ex_2022_23']['net_sr'], r['ex_2022_23']['net_se']):>13}")
lv = n["level"]
print(f"\nlevel factor: {lv['variance_share']:.0%} of gross P&L variance ({'dominates' if lv['dominates'] else 'does not dominate'})")
print("ELB: " + ", ".join(f"{t} {sr(e['net_sr'], e['net_se'])}" for t, e in n["elb"].items()))
print(f"carry benchmark: corr {n['carry']['corr']:+.2f} ({n['carry']['words']})")
paths = attribution.write(built, ROOT)
paths["tearsheet"] = tearsheet.write(built, ROOT / "reports" / "tearsheets", preview=args.preview)
print("\nwrote " + ", ".join(str(p.relative_to(ROOT)) for p in paths.values()) +
      f" ({time.perf_counter() - start:.1f}s)")
