"""Bring the cache under data/cache/ up to date, from FRED, the Databento archive, the Bank of England and ALFRED.

Incremental and idempotent: the manifest records what is already covered, so a
second run fetches only the last week of each FRED series again (to catch
revisions) and reads no archive files. ALFRED series are pulled whole each
time and checked against what is cached. Deleting data/ and running this
rebuilds everything.

    uv run --env-file .env python scripts/update_data.py
    uv run --env-file .env python scripts/update_data.py --macro-only
    uv run --env-file .env python scripts/update_data.py --refetch DFEDTARL DFEDTARU
"""

import argparse
import time
import pandas as pd
from policypath import config
from policypath.sources import boe, cache, fred, ons, rates

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--ccy", default="USD")
parser.add_argument("--macro-only", action="store_true", help="pull ALFRED only; no archive needed")
parser.add_argument("--refetch", nargs="+", default=[], metavar="SERIES",
                    help="drop these FRED series from the cache first and pull them whole again")
args = parser.parse_args()
cfg = config.currency(args.ccy)
today = pd.Timestamp.today().normalize()


def report(source, series):
    ranges = cache.fetched(source, series, args.ccy)
    span = f"{ranges[0][0].date()} .. {ranges[-1][1].date()}" if ranges else "nothing"
    return f"{len(ranges)} range(s), {span}"


def daily(name, series_list, lags, make):
    """One current-vintage daily source: each series with its publication lag from config."""
    for series in series_list:
        if series in args.refetch and cache.drop(name, series, args.ccy):
            print(f"{name}/{series:<9} dropped")
        source = make(lags.get(series, 1))
        added = source.update(series, args.ccy, cfg["history_start"], today, cache)
        print(f"{name}/{series:<9} +{added:>6} rows   covered {report(name, series)}")


t0 = time.perf_counter()
if not args.macro_only:
    # Each block is there only for the currencies whose config lists it.
    daily("fred", cfg.get("fred_series", []), cfg.get("fred_lags", {}), lambda lag: fred.Fred(lag_bdays=lag))
    daily("boe", cfg.get("boe_series", []), cfg.get("boe_lags", {}), lambda lag: boe.BoeSeries(lag_bdays=lag))
    curve = boe.BoeCurve()                     # one download of the archives for every curve series
    for series in cfg.get("boe_curves", []):
        if series in args.refetch and cache.drop("boe", series, args.ccy):
            print(f"boe/{series:<9} dropped")
        added = curve.update(series, args.ccy, cfg["history_start"], today, cache)
        print(f"boe/{series:<9} +{added:>6} rows   covered {report('boe', series)}")

    source = rates.Settlements()
    for root in cfg.get("futures_roots", []):
        def progress(lo, hi, n, root=root):
            print(f"  {root} {lo.date()} .. {hi.date()}: +{n}", flush=True)
        # Each root only over the days a job holding it covers, so a gap between jobs stays missing.
        added = sum(source.update(root, args.ccy, first, last, cache, on_chunk=progress)
                    for first, last in source.archive_ranges(root))
        print(f"databento/{root:<4} +{added:>6} rows   covered {report('databento', root)}")

def alfred(series, start):
    meta = {**fred.series_info(series), "first_vintage": f"{fred.vintage_dates(series).iloc[0]:%Y-%m-%d}"}
    return fred.vintages(series, start), meta


release_days = {}


def ons_vintages(series, start):
    kind = ons.SERIES[series]["release"]
    if kind not in release_days:
        release_days[kind] = ons.release_days(kind)
    v = ons.vintages(series, start, release_days[kind])
    meta = {"first_vintage": f"{v['realtime_start'].min():%Y-%m-%d}",
            "release_days_by_rule_before": f"{ons.lag_approximated_before(release_days[kind]):%Y-%m-%d}"}
    return v, meta


# Where a currency's macro vintages come from, by the config's ``macro.source``.
MACRO_SOURCES = {"alfred": alfred, "ons": ons_vintages}
macro = cfg.get("macro", {"series": [], "validation": [], "source": None})
for series in [*macro["series"], *macro["validation"]]:
    vintages, meta = MACRO_SOURCES[macro["source"]](series, macro["history_start"])
    added = cache.write_vintages(vintages, macro["source"], series, args.ccy, meta=meta)
    print(f"{macro['source']}/{series:<14} +{added:>6} rows")

print(f"done in {time.perf_counter() - t0:.1f}s")
