"""The expression layer: every sleeve's instruments, carry and roll, the checks on its P&L, and the breakeven.

Reads data/panel/ (run build_panel.py, build_nowcast.py and build_model.py for
every currency the book trades first) and the cache. Writes
data/panel/book_expression.parquet (one row per sleeve and session),
reports/expression.md, reports/figures/carry_{light,dark}.png,
reports/results/expression.json and reports/results/signals.csv, and prints the
headline. Linear rule, no costs. Prints the checks first, and writes nothing if
one fails.

    uv run python scripts/build_expression.py
"""

import argparse
from pathlib import Path
import time
from policypath.report import expression as report

ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.parse_args()

start = time.perf_counter()
built = report.build()
c = built.checks

print(f"{len(built.sleeves)} sleeves, {len(built.panel):,} sleeve sessions, {len(built.episodes):,} episodes "
      f"({time.perf_counter() - start:.1f}s)")
print(f"identity: max {c['identity_max_bp'].max():.1e}bp; full revaluation: max residual "
      f"{c['full_residual_max_bp'].max():.3f}bp, {int(c['outside_bound'].sum() // 2)} sessions outside the bound")
for name, ccy, n, diff in built.reproduced:
    print(f"{name} vs data/panel/{ccy}_backtest.parquet: {n:,} sessions, max difference {diff:.1e}bp")
paths, stats, filt = report.write(built, ROOT)
print(f"\n{'sleeve':<14} {'signals':>8} {'bleeding':>9} {'no pay':>7} {'right but lost':>15} "
      f"{'next q bleeding':>16} {'others':>7}")
for name, v in stats.items():
    s = v["signals"]
    print(f"{name:<14} {s['sessions']:>8,} {report.pct(s['bleeding']):>9} {report.pct(s['no_pay']):>7} "
          f"{report.pct(s['right_but_lost']):>15} {report.bp(s['next_bleeding_bp']):>16} "
          f"{report.bp(s['next_rest_bp']):>7}")
print(f"\ncarry filter (diagnostic): skips {int(filt['skipped'].sum())} of {int(filt['entries'].sum())} entries, "
      f"{filt['effect_bp'].sum():+.0f}bp over their next quarters")
print("\nnow:")
for name, r in report.latest(built.panel).iterrows():
    print(f"  {name:<14} {r['session']:%Y-%m-%d} z {r['z']:+.2f} {report.SIDE[float(r['side'])]:<7} "
          f"{r['instrument']:<32} edge {r['edge_bp']:+6.1f}bp CR_h {r['cr_h_bp']:+6.1f}bp  {report.verdict(r)}")
print("\nwrote " + ", ".join(str(p.relative_to(ROOT)) for p in paths.values()))
