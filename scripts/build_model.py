"""The model path next to the market path on every session, the gap signal, the crude backtest, and the ship.

Reads data/panel/ (run scripts/build_panel.py and scripts/build_nowcast.py
first) and the cache. Writes data/panel/<ccy>_model.parquet (the rule on each
session), <ccy>_paths.parquet (market and model per meeting), <ccy>_signal.parquet
(gap and z) and <ccy>_backtest.parquet, reports/model_<ccy>.md, the figures,
and the dated one-page PDF.

    uv run python scripts/build_model.py
"""

import argparse
from pathlib import Path
import pandas as pd
from policypath import config, panel
from policypath.model import path
from policypath.report import charts, onepager
from policypath.report import model as report
from policypath.signal import gap

ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--ccy", default=config.enabled()[0], help="default: the first enabled currency")
parser.add_argument("--author", default=None, help="name for the one-pager's byline")
parser.add_argument("--preview", default=None, help="also render the one-pager to this PNG")
parser.add_argument("--onepager", action="store_true",
                    help="also write the week 4 one-page note (config report.onepager); the weekly brief is "
                         "scripts/build_brief.py")
args = parser.parse_args()
cfg = config.currency(args.ccy)
if args.onepager and not cfg["report"]["onepager"]:
    parser.error(f"{args.ccy} has no one-pager: its prose (report/onepager.py) is written for the currency "
                 "whose config sets report.onepager: true")

sessions, meetings, macro = (panel.load(args.ccy, name) for name in ["sessions", "meetings", "macro"])
inputs = path.inputs(args.ccy)
summaries, paths = path.build(sessions, meetings, macro, inputs["sep"], inputs["target"], inputs["fixings"],
                              cfg["rule"])
signal = gap.build(paths, cfg["signal"])
headline, tables, result = report.checks(sessions, summaries, paths, signal, meetings, cfg)
today = report.snapshot(paths["session"].max(), paths, summaries, sessions, signal)

out = ROOT / "data" / "panel"
summaries.to_parquet(out / f"{args.ccy}_model.parquet", index=False)
paths.to_parquet(out / f"{args.ccy}_paths.parquet", index=False)
signal.to_parquet(out / f"{args.ccy}_signal.parquet", index=False)
result.reset_index(names="session").to_parquet(out / f"{args.ccy}_backtest.parquet", index=False)

reports = ROOT / "reports"
(reports / f"model_{args.ccy}.md").write_text(report.markdown(args.ccy, headline, tables, today, cfg))
effr = inputs["fixings"]
k = cfg["backtest"]["horizon"]
moments, labels = [tuple(m) for m in cfg["report"]["chart_moments"]], cfg["report"]["labels"]
drawn = charts.write_all(today, effr, signal, result, headline["backtest"], k, reports / "figures", args.ccy,
                         labels, moments)
page = (onepager.write(today, effr, signal, result, headline, reports, args.ccy, author=args.author,
                       preview=args.preview, moments=moments, labels=labels) if args.onepager else None)

b = headline["backtest"]
print(f"{headline['model_sessions']} sessions with a model path, "
      f"{headline['first_model']:%Y-%m-%d} .. {headline['last']:%Y-%m-%d}; {headline['at_elb']} on the floor")
print(f"backtest k={k}: {b['total']:+.0f}bp x z, Sharpe {b['sharpe']:.2f} (no costs)")
m = today["meetings"]
print(f"\n{today['session']:%Y-%m-%d}\n" + m[["k", "effective_date", "market", "model", "gap_bp", "z"]]
      .round({"market": 3, "model": 3, "gap_bp": 1, "z": 2}).to_string(index=False))
print(f"\nwrote {out}/{args.ccy}_{{model,paths,signal,backtest}}.parquet, "
      f"{reports / f'model_{args.ccy}.md'}, {len(drawn)} figures and {page}")
