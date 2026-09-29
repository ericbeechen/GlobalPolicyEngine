"""Costs, turnover and hysteresis: every sleeve under the book's rules, net of costs, and how fragile that is.

Reads data/panel/ (run build_panel.py, build_nowcast.py and build_model.py for
every currency the book trades first) and the cache, as build_expression.py
does. Writes reports/costs.md, reports/figures/cost_sensitivity_{light,dark}.png,
reports/figures/hysteresis_grid_{light,dark}.png and reports/results/costs.json,
and prints the headline. The sleeves' legs and unit P&L are built once and
every run (rules, ELB treatments, sizings, the grid of hysteresis pairs, the
cost levels) reuses them.

    uv run python scripts/build_strategy.py
"""

import argparse
from pathlib import Path
import time
from policypath.report import costs as report

ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.parse_args()

start = time.perf_counter()
built = report.build()
runs = sum(len(v) for v in built.grid.values()) * (len(built.sleeves) + 1)
print(f"{len(built.sleeves)} sleeves, {runs:,} grid runs ({time.perf_counter() - start:.1f}s)")
spec = report.chosen(built.book)
print(f"\n{spec.words()}, at the configured costs (SE):")
print(f"{'':<16} {'gross':>13} {'net':>13} {'c*':>18} {'mix':>7} {'active':>7} {'turns/yr':>9}")
for name, run in built.chosen["vol_scaled"].items():
    s = report.stats(run)
    print(f"{name:<16} {report.sr(s['gross_sr'], s['gross_se']):>13} {report.sr(s['net_sr'], s['net_se']):>13} "
          f"{report.cstar(s['cstar_bp']):>18} {s['mix_bp']:>6.2f}bp {s['active']:>7,} {s['turns_year']:>9.1f}")
g = report.surface(built.grid["vol_scaled"], built.book, report.SUM)
print(f"\ngrid ({g['cells']} cells), the sum: chosen {report.sr(g['net_sr'], g['net_se'])}, neighbours "
      f"{report.sr(g['near_min'])} .. {report.sr(g['near_max'])}, grid {report.sr(g['grid_min'])} .. "
      f"{report.sr(g['grid_max'])}, {report.pct(g['share_positive'])} of cells > 0, "
      f"{'stable' if g['stable'] else 'not stable'}")
paths = report.write(built, ROOT)
print(f"\nwrote " + ", ".join(str(p.relative_to(ROOT)) for p in paths.values()) +
      f" ({time.perf_counter() - start:.1f}s)")
