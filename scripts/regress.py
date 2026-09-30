"""Freeze every stage's output, or check it is bit-identical to what was frozen.

Snapshot before a refactor, check after every change. The full tier reads the
real cache (every session: run update_data.py first) and keeps its reference
in data/reference/<ccy>/, gitignored. With --fixtures it runs on the committed
tests/data instead and writes tests/data/reference/<ccy>/, which
tests/test_regression.py checks on every pytest run. Exits non-zero on any difference:
bit for bit on the full tier (frozen on this machine), to `regress.PLATFORM_ULPS` on
the fixtures (frozen on either).

    uv run python scripts/regress.py freeze            # every enabled currency, full sample
    uv run python scripts/regress.py check --ccy GBP
    uv run python scripts/regress.py freeze --fixtures
"""

import argparse
import sys
from policypath import config, regress

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("action", choices=["freeze", "check"])
parser.add_argument("--ccy", nargs="+", default=None, help="default: every enabled currency")
parser.add_argument("--fixtures", action="store_true", help="the committed fixtures, not the full cache")
args = parser.parse_args()

failed = False
for ccy in args.ccy or config.enabled():
    ref = (regress.FIXTURE_REF if args.fixtures else regress.FULL_DIR) / ccy
    end = None
    if args.action == "check" and not args.fixtures:
        want, meta = regress.reference(ref)
        end = meta["end"]
    if args.fixtures:
        frames, seconds = regress.timed(regress.fixture_outputs, ccy)
    else:
        frames, seconds = regress.timed(regress.outputs, ccy, end=end)
    if args.action == "freeze":
        info = regress.freeze(frames, ref, tier="fixtures" if args.fixtures else "full", seconds=round(seconds, 1))
        print(f"{ccy}: froze {sum(info['rows'].values())} rows through {info['end']} "
              f"at {info['commit']} in {seconds:.1f}s -> {ref}")
        continue
    if args.fixtures:
        want, meta = regress.reference(ref)
    diffs = regress.compare(regress.truncate(frames, meta["end"]), want,
                            regress.PLATFORM_ULPS if args.fixtures else 0)
    failed |= bool(diffs)
    status = "IDENTICAL" if not diffs else f"{len(diffs)} stage(s) differ"
    print(f"{ccy}: {status} to the reference frozen at {meta['commit']} through {meta['end']} ({seconds:.1f}s)")
    for stage, what in diffs.items():
        print(f"  {stage:>9}: {what}")
sys.exit(1 if failed else 0)
