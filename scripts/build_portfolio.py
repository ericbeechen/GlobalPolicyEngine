"""The book: every sleeve sized together under one covariance and one vol target, net of costs.

Reads data/panel/ (run build_panel.py, build_nowcast.py and build_model.py for
every currency the book trades first) and the cache, as build_strategy.py
does. Writes reports/portfolio.md, reports/figures/book_equity_{light,dark}.png
and reports/results/portfolio.json, and prints the headline. The sleeves' legs
and unit P&L are built once; the covariance once per set of sleeves; each
construction's targets once, and every book (shrinkage, the drawdown control's
rounds, the RV-only book, the cost curve) reuses them.

    uv run python scripts/build_portfolio.py
"""

import argparse
from pathlib import Path
import time
from policypath.report import portfolio as report

ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.parse_args()

start = time.perf_counter()
built = report.build()
print(f"{len(built.books)} books on {len(built.setup.days):,} book sessions ({time.perf_counter() - start:.1f}s)")
print(f"\n{'book':<40} {'gross':>13} {'net':>13} {'a year':>8} {'vol':>6} {'worst dd':>9} {'turns':>6} {'c*':>18}")
for ch, s in built.summaries.items():
    name = report._name(ch) + ("" if ch.shrink else ", unshrunk") + (", drawdown control" if ch.overlay else "")
    print(f"{name:<40} {report.sr(s['gross_sr'], s['gross_se']):>13} {report.sr(s['net_sr'], s['net_se']):>13} "
          f"{report.pc(s['return_pct']):>8} {s['vol_pct']:>5.1f}% {s['max_dd_pct']:>8.1f}% {s['turns_year']:>6.1f} "
          f"{report.cstar(s['cstar_bp']):>18}")
paths = report.write(built, ROOT)
print("\nwrote " + ", ".join(str(p.relative_to(ROOT)) for p in paths.values()) +
      f" ({time.perf_counter() - start:.1f}s)")
