"""The numbers sheet: every number the note quotes, from the generated reports to reports/numbers.md.

Reads reports/results/*.json and the markdown reports beside them; needs no
data. Run it after any build script. It stops if a report no longer prints a
number its JSON holds.

    uv run python scripts/build_numbers.py
"""

import argparse
from pathlib import Path
from policypath.report import numbers

ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.parse_args()
print(f"wrote {numbers.write(ROOT).relative_to(ROOT)}")
