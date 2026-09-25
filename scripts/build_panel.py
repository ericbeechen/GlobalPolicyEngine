"""Solve the implied path on every session in the cache, check it against SOFR futures, report.

Reads only the cache (run scripts/update_data.py first). Writes the panel and
the SOFR cross-check (SR1, per config) to data/panel/, the coverage and cross-check reports and
the README figures to reports/, and prints the coverage headline, solver
failures included.

    uv run python scripts/build_panel.py
"""

import argparse
from pathlib import Path
from policypath import config, panel
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
headline, tables = coverage.checks(sessions, meetings, start, end, spec["n_meetings"])
reports = ROOT / "reports"
reports.mkdir(exist_ok=True)
(reports / f"coverage_{args.ccy}.md").write_text(coverage.markdown(args.ccy, headline, tables, start, end))

# The cross-check: the SOFR - EFFR basis implied by SOFR futures against the ZQ path.
check = cfg["sofr"]["crosscheck"]
shape = cfg["contract_shapes"][check]
as_of = sessions["session"].max()
sofr = cache.read(cfg["sofr"]["source"], cfg["sofr"]["series"], args.ccy, as_of)
effr = cache.read(cfg["overnight"]["source"], cfg["overnight"]["series"], args.ccy, as_of)
futures = cache.log("databento", check, args.ccy)
futures = futures[futures["date"] >= cfg.get("first_sessions", {}).get(check, futures["date"].min())]
implied = basis.build(sessions, meetings, config.meetings(args.ccy), futures, sofr, effr, shape)
implied.to_parquet(out / f"{args.ccy}_sofr_basis.parquet", index=False)
by_year, jumps, nq = crosscheck.summary(implied, sofr, effr)
(reports / f"sofr_check_{args.ccy}.md").write_text(
    crosscheck.markdown(args.ccy, check, shape, by_year, jumps, nq))
drawn = figures.write_all(sessions, meetings, effr[effr["date"] >= start], nq, check,
                          reports / "figures", args.ccy)

for key, value in headline.items():
    print(f"{key:>20}: {value:.2f}" if isinstance(value, float) else f"{key:>20}: {value}")
for title, df in tables.items():
    print(f"{len(df):>6}  {title}")
failures = tables["Solver failures"]
if len(failures):
    print("\nsolver failures:")
    print(failures.to_string(index=False))
print(f"\n{check} cross-check, first contract wholly ahead of each session:")
print(by_year.round(2).to_string(index=False))
print(f"\nwrote {out}, {reports} and {len(drawn)} figures")
