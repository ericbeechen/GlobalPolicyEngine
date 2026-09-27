"""Solve the implied path on every session in the cache, check it, report.

Reads only the cache (run scripts/update_data.py first). Writes the panel to
data/panel/, the coverage report and the path figure to reports/, and prints
the coverage headline, solver failures included. Where the config has a
``sofr`` block (USD) it also runs the SOFR futures cross-check.

    uv run python scripts/build_panel.py
    uv run python scripts/build_panel.py --ccy GBP
"""

import argparse
from pathlib import Path
from policypath import config, panel
from policypath.calendars import BDAYS
from policypath.curves import basis
from policypath.report import coverage, crosscheck, figures
from policypath.sources import cache

ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--ccy", default="USD")
args = parser.parse_args()
cfg = config.currency(args.ccy)
spec = cfg["path"]

sessions, meetings = panel.build(args.ccy, start=spec["start"])
out = ROOT / "data" / "panel"
out.mkdir(parents=True, exist_ok=True)
sessions.to_parquet(out / f"{args.ccy}_sessions.parquet", index=False)
meetings.to_parquet(out / f"{args.ccy}_meetings.parquet", index=False)

start, end = sessions["session"].min(), sessions["session"].max()
run_checks, render = coverage.REPORTS[cfg["market"]["extractor"]]
headline, tables = run_checks(sessions, meetings, start, end, spec["n_meetings"], BDAYS[cfg["calendar"]])
reports = ROOT / "reports"
reports.mkdir(exist_ok=True)
(reports / f"coverage_{args.ccy}.md").write_text(render(args.ccy, headline, tables, start, end))

as_of = sessions["session"].max()
effr = cache.read(cfg["overnight"]["source"], cfg["overnight"]["series"], args.ccy, as_of)
nq = check = by_year = None
if "sofr" in cfg:
    # The cross-check: the SOFR - EFFR basis implied by SOFR futures against the ZQ path.
    check = cfg["sofr"]["crosscheck"]
    shape = cfg["contract_shapes"][check]
    sofr = cache.read(cfg["sofr"]["source"], cfg["sofr"]["series"], args.ccy, as_of)
    futures = cache.log("databento", check, args.ccy)
    futures = futures[futures["date"] >= cfg.get("first_sessions", {}).get(check, futures["date"].min())]
    implied = basis.build(sessions, meetings, config.meetings(args.ccy), futures, sofr, effr, shape)
    implied.to_parquet(out / f"{args.ccy}_sofr_basis.parquet", index=False)
    by_year, jumps, nq = crosscheck.summary(implied, sofr, effr)
    (reports / f"sofr_check_{args.ccy}.md").write_text(
        crosscheck.markdown(args.ccy, check, shape, by_year, jumps, nq))
drawn = figures.write_all(sessions, meetings, effr[effr["date"] >= start], nq, check,
                          reports / "figures", args.ccy,
                          {**cfg["market"], "policy": cfg["report"]["labels"]["policy"]})

for key, value in headline.items():
    print(f"{key:>20}: {value:.2f}" if isinstance(value, float) else f"{key:>20}: {value}")
for title, df in tables.items():
    print(f"{len(df):>6}  {title}")
failures = tables["Solver failures"]
if len(failures):
    print("\nsolver failures:")
    print(failures.to_string(index=False))
if by_year is not None:
    print(f"\n{check} cross-check, first contract wholly ahead of each session:")
    print(by_year.round(2).to_string(index=False))
print(f"\nwrote {out}, {reports} and {len(drawn)} figures")
