"""Snapshot data/cache/ to a dated zip somewhere off this disk.

The cache stops being rebuildable once it has seen a revision the source no
longer serves: FRED's current-vintage API and the BoE's refitted current-month
curve replace old values upstream, and the cache dates those revisions by when
it retrieved them. From then on the cache is the only record of what was known
on each date, so snapshot it after every update. Restore by unzipping into
data/cache/.

    uv run python scripts/backup_cache.py --to D:/backups/policypath
"""

import argparse
import datetime
from pathlib import Path
import shutil
from policypath.sources.cache import CACHE_DIR

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--to", required=True, type=Path, help="folder to write cache_<date>.zip into")
args = parser.parse_args()

if not (CACHE_DIR / "manifest.json").exists():
    raise SystemExit(f"no cache at {CACHE_DIR}")
args.to.mkdir(parents=True, exist_ok=True)
base = args.to / f"cache_{datetime.date.today():%Y-%m-%d}"
path = shutil.make_archive(str(base), "zip", root_dir=CACHE_DIR)
print(f"wrote {path} ({Path(path).stat().st_size / 1e6:.1f} MB)")
