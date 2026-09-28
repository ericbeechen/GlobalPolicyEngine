"""Bring the cache under data/cache/ up to date for one currency, from every source its config lists.

Incremental and idempotent: the manifest records what is already covered, so a
second run fetches only the last week of each daily series again (to catch
revisions) and reads no archive files. Macro vintages (ALFRED, the ONS) are
pulled whole each time and checked against what is cached. Deleting data/ and
running this rebuilds everything.

What is pulled is the config's ``sources`` block (daily series with their
publication lags, curves, futures roots) and its ``macro`` block; which code
pulls it is sources/registry.py. A new currency is never an edit here.

    uv run --env-file .env python scripts/update_data.py
    uv run --env-file .env python scripts/update_data.py --macro-only
    uv run --env-file .env python scripts/update_data.py --refetch DFEDTARL DFEDTARU
    uv run python scripts/update_data.py --ccy GBP
"""

import argparse
import time
import pandas as pd
from policypath import config
from policypath.sources import cache, registry

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--ccy", default=config.enabled()[0], help="default: the first enabled currency")
parser.add_argument("--macro-only", action="store_true", help="pull the macro vintages only; no archive needed")
parser.add_argument("--refetch", nargs="+", default=[], metavar="SERIES",
                    help="drop these series from the cache first and pull them whole again")
args = parser.parse_args()
cfg = config.currency(args.ccy)
sources = cfg["sources"]
today = pd.Timestamp.today().normalize()


def report(source, series):
    ranges = cache.fetched(source, series, args.ccy)
    span = f"{ranges[0][0].date()} .. {ranges[-1][1].date()}" if ranges else "nothing"
    return f"{len(ranges)} range(s), {span}"


def refetch(name, series):
    if series in args.refetch and cache.drop(name, series, args.ccy):
        print(f"{name}/{series:<9} dropped")


t0 = time.perf_counter()
if not args.macro_only:
    for name, lags in sources.get("daily", {}).items():
        make = registry.lookup(registry.DAILY, name)
        for series, lag in lags.items():
            refetch(name, series)
            added = make(lag).update(series, args.ccy, cfg["history_start"], today, cache)
            print(f"{name}/{series:<9} +{added:>6} rows   covered {report(name, series)}")

    for name, curves in sources.get("curves", {}).items():
        source = registry.lookup(registry.CURVES, name)()   # one download of the archives for every curve
        for series in curves:
            refetch(name, series)
            added = source.update(series, args.ccy, cfg["history_start"], today, cache)
            print(f"{name}/{series:<9} +{added:>6} rows   covered {report(name, series)}")

    for name, roots in sources.get("futures", {}).items():
        source = registry.lookup(registry.FUTURES, name)()
        for root in roots:
            def progress(lo, hi, n, root=root):
                print(f"  {root} {lo.date()} .. {hi.date()}: +{n}", flush=True)
            # Each root only over the days a job holding it covers, so a gap between jobs stays missing.
            added = sum(source.update(root, args.ccy, first, last, cache, on_chunk=progress)
                        for first, last in source.archive_ranges(root))
            print(f"{name}/{root:<4} +{added:>6} rows   covered {report(name, root)}")

macro = cfg["macro"]
pull = registry.lookup(registry.MACRO, macro["source"])
for series in [*macro["series"], *macro["validation"]]:
    vintages, meta = pull(series, macro["history_start"])
    added = cache.write_vintages(vintages, macro["source"], series, args.ccy, meta=meta)
    print(f"{macro['source']}/{series:<14} +{added:>6} rows")

print(f"done in {time.perf_counter() - t0:.1f}s")
