"""The credit bridge (week 12): three tests of the policy-path gap against credit spreads, per currency with a credit block.

Reads data/panel/ (signal, paths, model; the cross sleeves' other currencies'
signal) and the cache, and writes only its own outputs:
reports/credit_<ccy>.md, reports/figures/credit_<ccy>_{light,dark}.png and
reports/results/credit.json (one block per currency; a run on some currencies
replaces their blocks and keeps the others'). Then checks that every
{credit.<ccy>.headline.<key>} placeholder in notes/credit_section.md names a
key in credit.json, and fails if one does not.

    uv run python scripts/build_credit.py [--ccy USD]
"""

import argparse
import json
from pathlib import Path
import re
from policypath import config, credit, panel
from policypath.report import credit as report
from policypath.strategy.instruments import Marks

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
RESULTS = REPORTS / "results" / "credit.json"
NOTE = ROOT / "notes" / "credit_section.md"

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--ccy", nargs="*", default=None, help="default: every enabled currency with a credit block")
args = parser.parse_args()
bridged = [c for c in config.enabled() if "credit" in config.currency(c)]
unknown = sorted(set(args.ccy or []) - set(bridged))
if unknown:
    parser.error(f"{unknown}: not an enabled currency with a credit block ({bridged})")
ccys = args.ccy or bridged
book = config.strategy()

results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
results = {c: block for c, block in results.items() if c in bridged}
for ccy in ccys:
    cfg = config.currency(ccy)
    inputs = credit.Inputs(ccy, config.currency, book, panel.load, Marks(ccy, cfg).daily)
    found, frames = credit.run(inputs)
    (REPORTS / f"credit_{ccy}.md").write_text(report.markdown(ccy, found, cfg))
    drawn = report.write_figures(frames, found, cfg, REPORTS / "figures", ccy)
    results[ccy] = report.jsonable({"headline": report.headline(found, cfg), **found})
    h = results[ccy]["headline"]
    print(f"{ccy}: {h['weeks']} weeks, {h['sessions']} sessions with a z; staleness shows in {h['stale_spreads']}")
    print(f"  test 1: R² of the market's part alone {h[f't1_{found['primary']}_r2_market_pct']}%")
    print(f"  test 2: slope {h['t2_slope']}bp per z, NW t {h['t2_t']}, mean non-overlapping t "
          f"{h['t2_nonoverlap_t']}: {h['t2_verdict']} the pre-registered rule. {h['t2_kernel_sentence']} "
          f"With no link the registered t reaches ±1.96 in {h['t2_placebo_reject_pct']}% of placebo samples; "
          f"the headline's one-sided placebo p is {h['t2_placebo_p']}.")
    print(f"  test 3: {h['t3_sentence']} Late minus early {h['t3_diff']} (t {h['t3_diff_t']}, mde {h['t3_mde']})")
    print(f"  wrote reports/credit_{ccy}.md and {len(drawn)} figures")

RESULTS.parent.mkdir(parents=True, exist_ok=True)
RESULTS.write_text(json.dumps(dict(sorted(results.items())), indent=1, ensure_ascii=False) + "\n")
print(f"wrote {RESULTS.relative_to(ROOT)}")

if NOTE.exists():
    keys = re.findall(r"\{credit\.(\w+)\.headline\.(\w+)\}", NOTE.read_text())
    missing = sorted({f"{c}.{k}" for c, k in keys if results.get(c, {}).get("headline", {}).get(k) is None})
    if missing:
        raise SystemExit(f"{NOTE.relative_to(ROOT)}: {len(missing)} placeholders name no key in credit.json: {missing}")
    print(f"{NOTE.relative_to(ROOT)}: {len(keys)} placeholders, every one names a key in credit.json")
