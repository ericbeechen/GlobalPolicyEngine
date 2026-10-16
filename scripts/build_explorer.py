"""The explorer: the pipeline, a session time machine, who closes the gap and the decision log, in one page.

Reads config/explorer.yml, reports/numbers.md, reports/results/closure.json and
metrics.json, notes/DECISIONS.md and data/panel/ (every enabled currency's
paths, signal and model), and runs the book's setup in memory for each
sleeve's side (`report/portfolio.py`'s `setup`, from the panels and the cache
as build_portfolio.py reads them). Writes reports/explorer.html, one
self-contained file to open in a browser. Run it after build_closure.py and
build_numbers.py.

    uv run python scripts/build_explorer.py
"""

import argparse
import json
from pathlib import Path
import time
from policypath import config, panel
from policypath.report import explorer

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reports" / "results"

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.parse_args()

start = time.perf_counter()
sleeves = [s["name"] for s in config.strategy()["sleeves"]]
sides = explorer.book_sides()[sleeves]
print(f"the book's sides decided ({time.perf_counter() - start:.0f}s)")
first = json.loads((RESULTS / "metrics.json").read_text(encoding="utf-8"))["first_counted"]
machines = {ccy: explorer.machine(panel.load(ccy, "paths"), panel.load(ccy, "signal"), panel.load(ccy, "model"),
                                  config.currency(ccy), sides, first) for ccy in config.enabled()}
closure = json.loads((RESULTS / "closure.json").read_text(encoding="utf-8"))
data = explorer.payload(ROOT, machines, closure, sleeves)
path = explorer.write(ROOT, data)
stages, rows = data["stages"], data["decisions"]["rows"]
print(f"{len(stages)} stages, {sum(len(s['numbers']) for s in stages)} numbers, {len(rows)} decisions, "
      + ", ".join(f"{c} {len(m['d']):,} sessions" for c, m in machines.items()))
print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size / 1e6:.1f} MB, {time.perf_counter() - start:.0f}s)")
