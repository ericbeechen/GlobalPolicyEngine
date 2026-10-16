"""Who closes the gap: when the market and the rule disagree, which of the two moves, for every enabled currency.

Reads data/panel/ (paths, signal, model: run build_model.py for every currency
first) and writes reports/closure.md and reports/results/closure.json.

    uv run python scripts/build_closure.py
"""

import argparse
import json
from pathlib import Path
from policypath import closure, config, panel
from policypath.report import closure as report

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
RESULTS = REPORTS / "results" / "closure.json"

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.parse_args()

ccys = config.enabled()
cfgs = {ccy: config.currency(ccy) for ccy in ccys}
results = {ccy: closure.run(ccy, cfg=cfgs[ccy]) for ccy in ccys}
spans = {}
for ccy in ccys:
    sessions = panel.load(ccy, "paths")["session"]
    spans[ccy] = (sessions.min(), sessions.max())

(REPORTS / "closure.md").write_text(report.markdown(results, cfgs, spans), encoding="utf-8")
RESULTS.parent.mkdir(parents=True, exist_ok=True)
RESULTS.write_text(json.dumps(report.results_json(results, cfgs), indent=1, ensure_ascii=False) + "\n",
                   encoding="utf-8")
for ccy, r in results.items():
    f = r["horizons"][max(r["horizons"])]
    print(f"{ccy}, {f['h']} sessions: market {report.slope_t(f['market'])}, rule {report.slope_t(f['rule'])}, "
          f"policy rate {report.slope_t(f['rate'])}; closed {f['closed']:.2f}, {f['who']} "
          f"({report.share(f['rule_share'])} the rule's), {f['sessions']:,} sessions")
print(f"wrote reports/closure.md and {RESULTS.relative_to(ROOT)}")
