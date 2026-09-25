"""The Databento batch archive on disk: which job holds which daily files.

A batch job is one schema for a set of parent symbols (``ZQ.FUT``) over
[start, end), delivered as one file per day plus ``manifest.json``,
``metadata.json`` and ``condition.json``. Every job names its files the same
way, ``glbx-mdp3-YYYYMMDD.<schema>.dbn.zst``, so two jobs covering the same day
collide in one download folder -- a browser saves the second as
``... .dbn (2).zst`` and the manifests as ``manifest (2).json``. Each job is
therefore kept in its own folder, under its Databento job id:

    databento/definitions/<job_id>/manifest.json, metadata.json, condition.json,
                                   glbx-mdp3-YYYYMMDD.definition.dbn.zst ...
    databento/statistics/<job_id>/...

`file_downloads` sorts a fresh download into that layout. It matches every
file to its job by the sha256 in the job's manifest, never by its name, so a
file renamed on download still lands in the right job, and one that was
truncated matches nothing and is left where it is.

A job covers its roots on every day of [start, end), whether or not a root
traded that day: SR1 listed in 2018, so a job asking for it from 2010 has no
files before then, and that is the whole truth about those days.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import pandas as pd

DATABENTO_DIR = Path(__file__).resolve().parents[3] / "databento"
FOLDERS = {"definition": "definitions", "statistics": "statistics"}
DAY = pd.Timedelta(days=1)


@dataclass(frozen=True)
class Job:
    """One batch job: its schema, the futures roots it holds, and the days it covers."""
    job_id: str
    schema: str
    roots: tuple
    first: pd.Timestamp   # first day covered
    last: pd.Timestamp    # last day covered, inclusive
    folder: Path

    def files(self, start=None, end=None):
        """Daily files of this job with the date in their name in [start, end]."""
        start = pd.Timestamp(start) if start is not None else pd.Timestamp.min
        end = pd.Timestamp(end) if end is not None else pd.Timestamp.max
        out = []
        for path in sorted(self.folder.glob(f"glbx-mdp3-*.{self.schema}.dbn.zst")):
            if start <= file_day(path) <= end:
                out.append(path)
        return out


def file_day(path):
    """The day in a daily file's name: glbx-mdp3-20240917.statistics.dbn.zst -> 2024-09-17."""
    return pd.Timestamp(Path(path).name.split("-")[2].split(".")[0])


def _day(ns):
    return pd.Timestamp(ns, unit="ns").normalize()


def jobs(schema, root=DATABENTO_DIR):
    """Every job filed under the schema's folder, sorted by the first day it covers."""
    out = []
    for meta_path in (root / FOLDERS[schema]).glob("*/metadata.json"):
        query = json.loads(meta_path.read_text())["query"]
        if query["schema"] != schema:
            raise ValueError(f"{meta_path} is a {query['schema']} job filed under {FOLDERS[schema]}/")
        out.append(Job(
            job_id=meta_path.parent.name,
            schema=schema,
            roots=tuple(s.removesuffix(".FUT") for s in query["symbols"]),
            first=_day(query["start"]),
            last=_day(query["end"]) - DAY,  # the query's end is exclusive
            folder=meta_path.parent,
        ))
    return sorted(out, key=lambda j: (j.first, j.job_id))


def files(schema, series=None, start=None, end=None, root=DATABENTO_DIR):
    """Daily files for one schema in [start, end], from the jobs holding `series` (all jobs if None)."""
    held = [j for j in jobs(schema, root) if series is None or series in j.roots]
    return sorted((p for j in held for p in j.files(start, end)), key=lambda p: (file_day(p), str(p)))


def coverage(schema, series, root=DATABENTO_DIR):
    """The day ranges the jobs holding `series` cover, merged: [(first, last), ...]."""
    out = []
    for j in jobs(schema, root):
        if series not in j.roots:
            continue
        if out and j.first <= out[-1][1] + DAY:
            out[-1] = (out[-1][0], max(out[-1][1], j.last))
        else:
            out.append((j.first, j.last))
    return out


# --------------------------------------------------------------------------
# filing a download
# --------------------------------------------------------------------------

def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def file_downloads(root=DATABENTO_DIR):
    """Move loose batch downloads into one folder per job.

    Looks at the files directly under each schema folder. The loose manifests
    say which job each file belongs to, by hash. Returns (moved, left): moved
    counts files per job id, left lists the files that matched no manifest,
    or whose target already exists, so were not touched.
    """
    moved, left = {}, []
    for folder in FOLDERS.values():
        base = root / folder
        if not base.exists():
            continue
        loose = sorted(p for p in base.iterdir() if p.is_file())
        manifests = {p: json.loads(p.read_text()) for p in loose
                     if p.name.startswith("manifest") and p.suffix == ".json"}
        by_hash = {f["hash"].removeprefix("sha256:"): (m["job_id"], f["filename"])
                   for m in manifests.values() for f in m["files"]}
        for path in loose:
            if path in manifests:
                job_id, name = manifests[path]["job_id"], "manifest.json"
            elif (hit := by_hash.get(_sha256(path))) is not None:
                job_id, name = hit
            else:
                left.append(path)
                continue
            target = base / job_id / name
            if target.exists():
                left.append(path)
                continue
            target.parent.mkdir(exist_ok=True)
            path.replace(target)
            moved[job_id] = moved.get(job_id, 0) + 1
    return moved, left


def verify(root=DATABENTO_DIR, hashes=False):
    """Check every filed job against its manifest. One row per job.

    ``missing`` counts files the manifest lists that are absent or the wrong
    size; with `hashes`, the wrong sha256 too (slow: it reads every file).
    """
    rows = []
    for schema in FOLDERS:
        for j in jobs(schema, root):
            manifest = json.loads((j.folder / "manifest.json").read_text())
            bad = 0
            for f in manifest["files"]:
                p = j.folder / f["filename"]
                ok = p.exists() and p.stat().st_size == f["size"]
                if ok and hashes:
                    ok = _sha256(p) == f["hash"].removeprefix("sha256:")
                bad += not ok
            rows.append({"schema": schema, "job_id": j.job_id, "roots": ",".join(j.roots),
                         "first": j.first.date(), "last": j.last.date(),
                         "daily_files": len(j.files()), "missing": bad})
    return pd.DataFrame(rows)
