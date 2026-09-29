"""The robustness grid: the headline book with one choice at a time moved off the chosen specification.

Reads data/panel/ (run build_panel.py, build_nowcast.py and build_model.py for
every currency the book trades first) and the cache, as build_portfolio.py
does. Each row's signal is rebuilt through the chain's own entry points (the
model path, the gap and its z, and the market path for a curve-fitting row)
and kept in data/panel/robustness/ under a fingerprint of what it read, so a
rerun rebuilds only the rows whose inputs changed. A first build rebuilds them
all, about 100s; --rows splits it. The legs, unit P&L and covariance are built
once and every row reuses them. Run build_portfolio.py first: the chosen
specification on its own sample must be reports/results/portfolio.json's
headline, or the build stops. Writes reports/robustness.md,
reports/figures/robustness_{light,dark}.png and reports/results/robustness.json,
and prints the grid.

    uv run python scripts/build_robustness.py
    uv run python scripts/build_robustness.py --rows estimated converge   # rebuild these rows' signals, no report
"""

import argparse
from pathlib import Path
import time
from policypath.report import robustness as report
from policypath.strategy import expression, robustness

ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--rows", nargs="+", metavar="KEY",
                    help="only rebuild (and store) these rows' signals (config/strategy.yml robustness keys), no report")
args = parser.parse_args()

start = time.perf_counter()
world = expression.World()
grid = robustness.rows(world.book)
store = robustness.Store()
if args.rows:
    unknown = sorted(set(args.rows) - {r.key for r in grid})
    if unknown:
        parser.error(f"no robustness row {unknown}; the rows are {[r.key for r in grid[1:]]}")
    robustness.signals(world, grid, store, keys=set(args.rows), log=print)
    print(f"stored {len(args.rows)} rows' signals in {store.root.relative_to(ROOT)} ({time.perf_counter() - start:.1f}s)")
    raise SystemExit
built = report.build(world, store=store, log=print)
gone = store.prune()
s = report.summary(built, report.published_headline(ROOT, world.book))     # stops unless portfolio.json agrees
print(f"{len(built.rows) - 1} rows over {s['sample']['book']['sessions']:,} common book sessions "
      f"({time.perf_counter() - start:.1f}s; {gone} stale stored files removed)")
print(f"\n{'row':<48} {'net':>13} {'change':>7} {'paired':>7} {'gross':>7} {'own':>13}")
for r in built.rows:
    k = r.key
    print(f"{r.words():<48} {report.sr(*s['net'][k]['book'][:2]):>13} {report._delta(s, k):>+7.2f} "
          f"{s['paired'][k]['book']:>7.2f} {report.sr(s['gross'][k]['book'][0]):>7} "
          f"{report.sr(*s['own'][k]['book'][:2]):>13}")
paths = report.write(built, ROOT, s)
print("\nwrote " + ", ".join(str(p.relative_to(ROOT)) for p in paths.values()) +
      f" ({time.perf_counter() - start:.1f}s)")
