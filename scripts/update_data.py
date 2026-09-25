"""Bring the cache under data/cache/ up to date, from FRED, the Databento archive and ALFRED.

Incremental and idempotent: the manifest records what is already covered, so a
second run fetches only the last week of each FRED series again (to catch
revisions) and reads no archive files. ALFRED series are pulled whole each
time and checked against what is cached. Deleting data/ and running this
rebuilds everything.

    uv run --env-file .env python scripts/update_data.py
    uv run --env-file .env python scripts/update_data.py --macro-only
"""

import argparse
import time
import pandas as pd
from policypath import config
from policypath.sources import cache, fred, rates

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--ccy", default="USD")
parser.add_argument("--macro-only", action="store_true", help="pull ALFRED only; no archive needed")
args = parser.parse_args()
cfg = config.currency(args.ccy)
today = pd.Timestamp.today().normalize()


def report(source, series):
    ranges = cache.fetched(source, series, args.ccy)
    span = f"{ranges[0][0].date()} .. {ranges[-1][1].date()}" if ranges else "nothing"
    return f"{len(ranges)} range(s), {span}"


t0 = time.perf_counter()
if not args.macro_only:
    source = fred.Fred()
    for series in cfg["fred_series"]:
        added = source.update(series, args.ccy, cfg["history_start"], today, cache)
        print(f"fred/{series:<9} +{added:>6} rows   covered {report('fred', series)}")

    source = rates.Settlements()
    first, last = source.archive_days()
    for root in cfg["futures_roots"]:
        def progress(lo, hi, n, root=root):
            print(f"  {root} {lo.date()} .. {hi.date()}: +{n}", flush=True)
        added = source.update(root, args.ccy, first, last, cache, on_chunk=progress)
        print(f"databento/{root:<4} +{added:>6} rows   covered {report('databento', root)}")

macro = cfg["macro"]
for series in [*macro["series"], *macro["validation"]]:
    meta = {**fred.series_info(series), "first_vintage": f"{fred.vintage_dates(series).iloc[0]:%Y-%m-%d}"}
    added = cache.write_vintages(fred.vintages(series, macro["history_start"]), macro["source"],
                                 series, args.ccy, meta=meta)
    print(f"alfred/{series:<14} +{added:>6} rows")

print(f"done in {time.perf_counter() - t0:.1f}s")
