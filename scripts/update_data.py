"""Bring the cache under data/cache/ up to date for one currency, from every source its config lists.

Incremental and idempotent: the manifest records what is already covered, so a
second run fetches only the last week of each daily series again (to catch
revisions) and reads no archive files. Macro vintages (ALFRED, the ONS) are
pulled whole each time and checked against what is cached. Deleting data/ and
running this rebuilds everything.

What is pulled is the config's ``sources`` block (daily series with their
publication lags, curves, futures roots, quarterly estimates) and its ``macro``
block; which code pulls it is sources/registry.py. A new currency is never an
edit here. ``--only`` pulls just the series it names and nothing else, not even
the macro vintages: for adding a series without refreshing the rest.

Every currency needs the FRED key in .env: GBP's too, for its FX rate (DEXUSUK).

Futures come from the Databento archive on disk. With DATABENTO_API_KEY in .env
too, the days after the archive's last are pulled over Databento's historical
API, each request priced first and the whole update capped at
rates.LIVE_BUDGET_USD (a day of all three roots is about a cent).
``--archive-only`` leaves the API alone.

    uv run --env-file .env python scripts/update_data.py
    uv run --env-file .env python scripts/update_data.py --archive-only
    uv run --env-file .env python scripts/update_data.py --macro-only
    uv run --env-file .env python scripts/update_data.py --refetch DFEDTARL DFEDTARU
    uv run --env-file .env python scripts/update_data.py --ccy GBP
    uv run --env-file .env python scripts/update_data.py --ccy GBP --only DEXUSUK GLC_SPOT
"""

import argparse
import time
import pandas as pd
from policypath import config
from policypath.sources import cache, registry

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--ccy", default=config.enabled()[0], help="default: the first enabled currency")
which = parser.add_mutually_exclusive_group()
which.add_argument("--macro-only", action="store_true", help="pull the macro vintages only; no archive needed")
which.add_argument("--only", nargs="+", default=[], metavar="SERIES",
                   help="pull only these series from the sources block; no macro vintages")
parser.add_argument("--refetch", nargs="+", default=[], metavar="SERIES",
                    help="drop these series from the cache first and pull them whole again")
parser.add_argument("--archive-only", action="store_true",
                    help="futures from the archive on disk only, even with an API key set")
args = parser.parse_args()
cfg = config.currency(args.ccy)
sources = cfg["sources"]
today = pd.Timestamp.today().normalize()
KINDS = ("daily", "estimates", "curves", "futures")
listed = {s for kind in KINDS for group in sources.get(kind, {}).values() for s in group}
if set(args.only) - listed:
    parser.error(f"--only {sorted(set(args.only) - listed)}: not under sources in the {args.ccy} block")


def wanted(series):
    return not args.only or series in args.only


def report(source, series):
    ranges = cache.fetched(source, series, args.ccy)
    span = f"{ranges[0][0].date()} .. {ranges[-1][1].date()}" if ranges else "nothing"
    return f"{len(ranges)} range(s), {span}"


def refetch(name, series):
    if series in args.refetch and cache.drop(name, series, args.ccy):
        print(f"{name}/{series:<9} dropped")


t0 = time.perf_counter()
if not args.macro_only:
    # Series with a publication lag each: business days for a daily series, calendar days for an estimate.
    for kind, table in (("daily", registry.DAILY), ("estimates", registry.ESTIMATES)):
        for name, lags in sources.get(kind, {}).items():
            make = registry.lookup(table, name)
            for series, lag in lags.items():
                if not wanted(series):
                    continue
                refetch(name, series)
                added = make(lag).update(series, args.ccy, cfg["history_start"], today, cache)
                print(f"{name}/{series:<9} +{added:>6} rows   covered {report(name, series)}")

    for name, curves in sources.get("curves", {}).items():
        source = registry.lookup(registry.CURVES, name)()   # one download of each archive for every curve
        for series in filter(wanted, curves):
            refetch(name, series)
            added = source.update(series, args.ccy, cfg["history_start"], today, cache)
            print(f"{name}/{series:<9} +{added:>6} rows   covered {report(name, series)}")

    for name, roots in sources.get("futures", {}).items():
        source = registry.lookup(registry.FUTURES, name)(not args.archive_only)
        for root in filter(wanted, roots):
            def progress(lo, hi, n, root=root):
                print(f"  {root} {lo.date()} .. {hi.date()}: +{n}", flush=True)
            # Each root over the days a job holding it covers, so a gap between jobs stays missing,
            # then any days after the last job the source can reach live.
            added = sum(source.update(root, args.ccy, first, last, cache, on_chunk=progress)
                        for first, last in source.ranges(root, today))
            cost = f"   ${source.spent:.4f} quoted so far" if source.spent else ""
            print(f"{name}/{root:<4} +{added:>6} rows   covered {report(name, root)}{cost}")

macro = cfg["macro"]
pull = registry.lookup(registry.MACRO, macro["source"])
for series in [] if args.only else [*macro["series"], *macro["validation"]]:
    vintages, meta = pull(series, macro["history_start"])
    added = cache.write_vintages(vintages, macro["source"], series, args.ccy, meta=meta)
    print(f"{macro['source']}/{series:<14} +{added:>6} rows")

print(f"done in {time.perf_counter() - t0:.1f}s")
