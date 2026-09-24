"""Bring the cache under data/cache/ up to date, from FRED and the Databento archive.

Incremental and idempotent: the manifest records what is already covered, so a
second run fetches only the last week of each FRED series again (to catch
revisions) and reads no archive files. Deleting data/ and running this rebuilds
everything.

    uv run --env-file .env python scripts/update_data.py
"""

import argparse
import time
import pandas as pd
from policypath import config
from policypath.sources import cache, fred, rates

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--ccy", default="USD")
args = parser.parse_args()
cfg = config.currency(args.ccy)
today = pd.Timestamp.today().normalize()


def report(source, series):
    ranges = cache.fetched(source, series, args.ccy)
    span = f"{ranges[0][0].date()} .. {ranges[-1][1].date()}" if ranges else "nothing"
    return f"{len(ranges)} range(s), {span}"


t0 = time.perf_counter()
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

print(f"done in {time.perf_counter() - t0:.1f}s")
