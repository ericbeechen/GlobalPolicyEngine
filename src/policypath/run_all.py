"""Rebuild every report from the cache, in the README's order: `uv run all`.

Panels, nowcasts and models per enabled currency, then the strategy layer, the
brief, the tear sheet and last the numbers sheet. Stops at the first step that
fails. `--update` first brings the cache up to date (needs the FRED key in .env
and, for USD, the Databento archive), then reminds you to back it up.

    uv run all
    uv run all --update
    uv run all --from build_portfolio   # resume at a step
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from policypath import config

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
    out = []
    if update:
        out += [["update_data", "--ccy", ccy] for ccy in ccys]
    for ccy in ccys:
        out += [["build_panel", "--ccy", ccy], ["build_nowcast", "--ccy", ccy], ["build_model", "--ccy", ccy]]
    out += [["catalogue_vintages"], ["check_mpr"],
            ["build_expression"], ["build_credit"], ["build_brief"], ["build_strategy"],
            ["build_portfolio"], ["build_robustness"], ["build_tearsheet"], ["build_numbers"]]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--update", action="store_true", help="pull new data into the cache first")
    parser.add_argument("--from", dest="start", default=None, metavar="SCRIPT",
                        help="skip the steps before the first run of this script")
    args = parser.parse_args()

    todo = steps(args.update)
    if args.start:
        names = [s[0] for s in todo]
        if args.start not in names:
            parser.error(f"--from {args.start}: not one of {sorted(set(names))}")
        todo = todo[names.index(args.start):]

    env = env_file()
    t0 = time.perf_counter()
    for i, (script, *flags) in enumerate(todo, 1):
        label = " ".join([script, *flags])
        print(f"\n=== [{i}/{len(todo)}] {label}", flush=True)
        t = time.perf_counter()
        result = subprocess.run([sys.executable, str(ROOT / "scripts" / f"{script}.py"), *flags], cwd=ROOT, env=env)
        if result.returncode:
            sys.exit(f"\n{label} failed (exit {result.returncode}); rerun with --from {script} once fixed")
        print(f"=== {label}: {time.perf_counter() - t:.0f}s", flush=True)
    print(f"\nall {len(todo)} steps done in {time.perf_counter() - t0:.0f}s")
    if args.update:
        print("the cache changed: snapshot it with scripts/backup_cache.py --to <folder>")
