"""File fresh Databento batch downloads into one folder per job, then check every job.

Download a job's files (daily .dbn.zst plus manifest, metadata and condition
JSON) straight into databento/definitions/ or databento/statistics/; name
collisions with other jobs, "(2)" and all, do not matter. This moves each file
to <schema folder>/<job_id>/ under its original name, matched by the sha256 in
its job's manifest, and prints what every filed job covers. It then rewrites
config/databento_jobs.json, the committed record of what each job requested,
since the archive itself is gitignored. Safe to rerun.

    uv run python scripts/file_databento.py
    uv run python scripts/file_databento.py --hashes   # also re-hash every filed file
"""

import argparse
import json
from pathlib import Path
from policypath.sources import archive

RECORD = Path(__file__).resolve().parents[1] / "config" / "databento_jobs.json"

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--hashes", action="store_true", help="verify sha256 of every filed file (slow)")
args = parser.parse_args()

moved, left = archive.file_downloads()
for job_id, n in sorted(moved.items()):
    print(f"filed {n:>5} files into {job_id}")
for path in left:
    print(f"left in place (no manifest matches it, or its target exists): {path}")

report = archive.verify(hashes=args.hashes)
print(report.to_string(index=False))
RECORD.write_text(json.dumps(archive.record(), indent=2) + "\n", encoding="utf-8")
print(f"wrote {RECORD}")
if report["missing"].any():
    raise SystemExit("some jobs are missing files their manifest lists; download them again")
