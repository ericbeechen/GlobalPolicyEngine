"""Bring the cache up to date and rebuild every report from it, in the README's order: `uv run all`.

First the cache, for every enabled currency (needs the FRED key in .env and,
for the futures, the Databento archive), then a check that no currency's market
data is behind another's. Then the panels, nowcasts and models per currency,
the strategy layer, the brief, the tear sheet and last the numbers sheet. Stops
at the first step that fails.

The check: every currency's last full update must have finished on the same
day. One that did not (GBP from three days ago beside USD from today) would
give reports cut at different dates, so the run stops rather than build them.
The sources' own lags are fine: updated together, the Bank's curve can end a
session or two before CME's settles, and that is the data, not a stale cache.

    uv run all
    uv run all --cache-only              # build from the cache as it is (the check still runs)
    uv run all --from build_portfolio    # resume at a step
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from policypath import config
from policypath.sources import cache

ROOT = Path(__file__).resolve().parents[2]


def env_file():
    """The environment plus .env's KEY=VALUE lines, as `uv run --env-file .env` would give."""
    env = dict(os.environ)
    path = ROOT / ".env"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.strip().partition("=")
            if sep and not key.startswith("#"):
                env.setdefault(key.strip(), value.strip().strip("'\""))
    return env


def steps(update):
    ccys = config.enabled()
    out = [["update_data", "--ccy", *ccys]] if update else []
    for ccy in ccys:
        out += [["build_panel", "--ccy", ccy], ["build_nowcast", "--ccy", ccy], ["build_model", "--ccy", ccy]]
    out += [["catalogue_vintages"], ["check_mpr"],
            ["build_expression"], ["build_credit"], ["build_brief"], ["build_strategy"],
            ["build_portfolio"], ["build_robustness"], ["build_tearsheet"], ["build_numbers"]]
    return out


def market_series(cfg):
    """(source, series) of the market data a currency's path is solved from: its futures or its curve."""
    market = cfg["market"]
    spec = market.get("futures") or market.get("curve")
    return spec["source"], spec["series"]


def freshness(ccys, root=cache.CACHE_DIR):
    """[(currency, source/series, last session its market data covers, when its last full update finished)]."""
    rows = []
    for ccy in ccys:
        source, series = market_series(config.currency(ccy))
        rows.append((ccy, f"{source}/{series}", cache.last_covered(source, series, ccy, root),
                     cache.last_updated(ccy, root)))
    return rows


def in_step(rows):
    """None if every currency's last full update finished on the same day, else the message that stops the run."""
    days = {ccy: (pulled.date() if pulled is not None else None) for ccy, _, _, pulled in rows}
    if len(set(days.values())) == 1 and None not in days.values():
        return None
    said = ", ".join(f"{ccy} {d if d else 'never'}" for ccy, d in days.items())
    return (f"the cache is out of step: last full update {said}. Reports built now would be cut at "
            "different dates. Run `uv run all` (it updates every currency first), or "
            "`uv run --env-file .env python scripts/update_data.py`")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache-only", action="store_true", help="build from the cache as it is, without pulling")
    parser.add_argument("--from", dest="start", default=None, metavar="SCRIPT",
                        help="skip the steps before the first run of this script")
    args = parser.parse_args()

    todo = steps(not args.cache_only)
    if args.start:
        names = [s[0] for s in todo]
        if args.start not in names:
            parser.error(f"--from {args.start}: not one of {sorted(set(names))}")
        todo = todo[names.index(args.start):]

    env = env_file()
    t0 = time.perf_counter()
    checked = False
    for i, (script, *flags) in enumerate(todo, 1):
        if script != "update_data" and not checked:
            rows = freshness(config.enabled())
            for ccy, key, last, pulled in rows:
                print(f"{ccy} {key}: through {last.date() if last is not None else 'nothing'}, "
                      f"updated {pulled if pulled is not None else 'never'}")
            stop = in_step(rows)
            if stop:
                sys.exit(stop)
            checked = True
        label = " ".join([script, *flags])
        print(f"\n=== [{i}/{len(todo)}] {label}", flush=True)
        t = time.perf_counter()
        result = subprocess.run([sys.executable, str(ROOT / "scripts" / f"{script}.py"), *flags], cwd=ROOT, env=env)
        if result.returncode:
            sys.exit(f"\n{label} failed (exit {result.returncode}); rerun with --from {script} once fixed")
        print(f"=== {label}: {time.perf_counter() - t:.0f}s", flush=True)
    print(f"\nall {len(todo)} steps done in {time.perf_counter() - t0:.0f}s")
    if not args.cache_only:
        print("the cache changed: snapshot it with scripts/backup_cache.py --to <folder>")
