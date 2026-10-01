"""Bring the cache under data/cache/ up to date for every enabled currency (or those named), from every source its config lists.

Incremental and idempotent: the manifest records what is already covered, so a
second run fetches only the last week of each daily series again (to catch
revisions) and reads no archive files. Macro vintages (ALFRED, the ONS) are
pulled whole each time and checked against what is cached. Deleting data/ and
running this rebuilds everything.

What is pulled is the config's ``sources`` block (daily series with their
publication lags, curves, futures roots, quarterly estimates, series kept with
every real-time vintage) and its ``macro`` block; which code pulls it is
sources/registry.py. A new currency is never an
edit here. ``--only`` pulls just the series it names and nothing else, not even
the macro vintages: for adding a series without refreshing the rest.

Every currency needs the FRED key in .env: GBP's too, for its FX rate (DEXUSUK).

Futures come from the Databento archive on disk. With DATABENTO_API_KEY in .env
too, the days after the archive's last are pulled over Databento's historical
API, each request priced first and the whole update capped at
rates.LIVE_BUDGET_USD (a day of all three roots is about a cent).
``--archive-only`` leaves the API alone.

    uv run --env-file .env python scripts/update_data.py                 # every enabled currency
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
parser.add_argument("--ccy", nargs="+", default=config.enabled(), help="default: every enabled currency")
which = parser.add_mutually_exclusive_group()
which.add_argument("--macro-only", action="store_true", help="pull the macro vintages only; no archive needed")
which.add_argument("--only", nargs="+", default=[], metavar="SERIES",
                   help="pull only these series from the sources block; no macro vintages")
parser.add_argument("--refetch", nargs="+", default=[], metavar="SERIES",
                    help="drop these series from the cache first and pull them whole again")
parser.add_argument("--archive-only", action="store_true",
                    help="futures from the archive on disk only, even with an API key set")
args = parser.parse_args()
today = pd.Timestamp.today().normalize()
KINDS = ("daily", "estimates", "curves", "futures", "vintages")


def listed(ccy):
    sources = config.currency(ccy)["sources"]
    return {s for kind in KINDS for group in sources.get(kind, {}).values() for s in group}


unknown = set(args.only) - set().union(*map(listed, args.ccy))
if unknown:
    parser.error(f"--only {sorted(unknown)}: not under sources in the {', '.join(args.ccy)} block")


def wanted(series):
    return not args.only or series in args.only


def update(ccy):
    cfg = config.currency(ccy)
    sources = cfg["sources"]

    def report(source, series):
        ranges = cache.fetched(source, series, ccy)
        span = f"{ranges[0][0].date()} .. {ranges[-1][1].date()}" if ranges else "nothing"
        return f"{len(ranges)} range(s), {span}"

    def refetch(name, series):
        if series in args.refetch and cache.drop(name, series, ccy):
            print(f"{name}/{series:<9} dropped")

    print(f"--- {ccy}")
    if not args.macro_only:
        # Series with a publication lag each: business days for a daily series, calendar days for an estimate.
        for kind, table in (("daily", registry.DAILY), ("estimates", registry.ESTIMATES)):
            for name, lags in sources.get(kind, {}).items():
                make = registry.lookup(table, name)
                for series, lag in lags.items():
                    if not wanted(series):
                        continue
                    refetch(name, series)
                    added = make(lag).update(series, ccy, cfg["history_start"], today, cache)
                    print(f"{name}/{series:<9} +{added:>6} rows   covered {report(name, series)}")

        for name, curves in sources.get("curves", {}).items():
            source = registry.lookup(registry.CURVES, name)()   # one download of each archive for every curve
            for series in filter(wanted, curves):
                refetch(name, series)
                added = source.update(series, ccy, cfg["history_start"], today, cache)
                print(f"{name}/{series:<9} +{added:>6} rows   covered {report(name, series)}")

        for name, roots in sources.get("futures", {}).items():
            source = registry.lookup(registry.FUTURES, name)(not args.archive_only)
            for root in filter(wanted, roots):
                def progress(lo, hi, n, root=root):
                    print(f"  {root} {lo.date()} .. {hi.date()}: +{n}", flush=True)
                # Each root over the days a job holding it covers, so a gap between jobs stays missing,
                # then any days after the last job the source can reach live.
                added = sum(source.update(root, ccy, first, last, cache, on_chunk=progress)
                            for first, last in source.ranges(root, today))
                cost = f"   ${source.spent:.4f} quoted so far" if source.spent else ""
                print(f"{name}/{root:<4} +{added:>6} rows   covered {report(name, root)}{cost}")

    # Series kept with every real-time vintage outside the nowcast, pulled whole like the macro ones (with them under --macro-only).
    for name, series_list in sources.get("vintages", {}).items():
        pull = registry.lookup(registry.MACRO, name)
        for series in filter(wanted, series_list):
            vintages, meta = pull(series, cfg["history_start"])
            added = cache.write_vintages(vintages, name, series, ccy, meta=meta)
            print(f"{name}/{series:<14} +{added:>6} rows")

    macro = cfg["macro"]
    pull = registry.lookup(registry.MACRO, macro["source"])
    for series in [] if args.only else [*macro["series"], *macro["validation"]]:
        vintages, meta = pull(series, macro["history_start"])
        added = cache.write_vintages(vintages, macro["source"], series, ccy, meta=meta)
        print(f"{macro['source']}/{series:<14} +{added:>6} rows")


t0 = time.perf_counter()
for ccy in args.ccy:
    update(ccy)
    if not (args.only or args.macro_only):   # a partial pull leaves the market data where it was
        cache.mark_updated(ccy)
print(f"done in {time.perf_counter() - t0:.1f}s")
