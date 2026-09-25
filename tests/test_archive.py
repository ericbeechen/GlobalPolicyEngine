"""Filing Databento batch downloads by job, and reading coverage per root.

Two jobs for the same days name their daily files identically, so a browser
saves one set as "... (2).zst". These build a download like that on disk,
with stand-in file contents, and check each file lands with its own job.
"""

import hashlib
import json
import pandas as pd
from policypath.sources import archive


def _ns(day):
    return int(pd.Timestamp(day).value)


def write_job(folder, job_id, symbols, start, end, days, tag, suffix=""):
    """One downloaded statistics job: a daily file per day, its manifest and metadata."""
    files = []
    for day in days:
        name = f"glbx-mdp3-{pd.Timestamp(day):%Y%m%d}.statistics.dbn.zst"
        body = f"{tag} {day}".encode()
        on_disk = name.replace(".dbn.zst", f".dbn{suffix}.zst")
        (folder / on_disk).write_bytes(body)
        files.append({"filename": name, "size": len(body), "hash": "sha256:" + hashlib.sha256(body).hexdigest()})
    meta = {"query": {"schema": "statistics", "symbols": symbols, "start": _ns(start), "end": _ns(end)}}
    body = json.dumps(meta).encode()
    (folder / f"metadata{suffix}.json").write_bytes(body)
    files.append({"filename": "metadata.json", "size": len(body), "hash": "sha256:" + hashlib.sha256(body).hexdigest()})
    (folder / f"manifest{suffix}.json").write_text(json.dumps({"job_id": job_id, "files": files}))


def test_colliding_downloads_are_filed_by_hash_not_name(tmp_path):
    folder = tmp_path / "statistics"
    folder.mkdir()
    write_job(folder, "JOB-ZQ", ["ZQ.FUT"], "2020-12-28", "2020-12-31", ["2020-12-28", "2020-12-29", "2020-12-30"], "zq")
    # Downloaded second: its colliding names got " (2)".
    write_job(folder, "JOB-SR1", ["SR1.FUT"], "2020-12-28", "2021-01-02",
              ["2020-12-28", "2020-12-29", "2020-12-30", "2020-12-31"], "sr1", suffix=" (2)")

    moved, left = archive.file_downloads(tmp_path)

    assert moved == {"JOB-ZQ": 5, "JOB-SR1": 6} and left == []
    assert not any(p.is_file() for p in folder.iterdir())
    assert (folder / "JOB-SR1" / "glbx-mdp3-20201229.statistics.dbn.zst").read_bytes() == b"sr1 2020-12-29"
    assert (folder / "JOB-ZQ" / "glbx-mdp3-20201229.statistics.dbn.zst").read_bytes() == b"zq 2020-12-29"
    assert archive.verify(tmp_path)["missing"].sum() == 0
    assert archive.file_downloads(tmp_path) == ({}, [])  # rerunning does nothing


def test_a_corrupt_file_matches_no_job_and_is_left_alone(tmp_path):
    folder = tmp_path / "statistics"
    folder.mkdir()
    write_job(folder, "JOB-ZQ", ["ZQ.FUT"], "2020-12-28", "2020-12-30", ["2020-12-28", "2020-12-29"], "zq")
    bad = folder / "glbx-mdp3-20201229.statistics.dbn.zst"
    bad.write_bytes(b"truncated")

    moved, left = archive.file_downloads(tmp_path)

    assert left == [bad] and bad.exists()
    assert archive.verify(tmp_path)["missing"].item() == 1


def test_coverage_is_per_root_and_keeps_gaps_between_jobs(tmp_path):
    folder = tmp_path / "statistics"
    folder.mkdir()
    write_job(folder, "OLD", ["ZQ.FUT"], "2010-06-06", "2020-12-31", ["2010-06-07"], "a")
    write_job(folder, "NEW", ["ZQ.FUT", "SR3.FUT"], "2020-12-31", "2026-09-22", ["2020-12-31"], "b", " (2)")
    write_job(folder, "SR1", ["SR1.FUT"], "2010-06-06", "2026-09-25", ["2018-05-07"], "c", " (3)")
    write_job(folder, "LATE", ["ZQ.FUT"], "2026-10-01", "2026-10-03", ["2026-10-01"], "d", " (4)")
    archive.file_downloads(tmp_path)

    T = pd.Timestamp
    # The query end is exclusive: a job to 2020-12-31 covers through the 30th.
    assert archive.coverage("statistics", "ZQ", tmp_path) == [(T("2010-06-06"), T("2026-09-21")),
                                                              (T("2026-10-01"), T("2026-10-02"))]
    assert archive.coverage("statistics", "SR3", tmp_path) == [(T("2020-12-31"), T("2026-09-21"))]
    assert archive.coverage("statistics", "SR1", tmp_path) == [(T("2010-06-06"), T("2026-09-24"))]
    # SR1's files never stand in for ZQ days, and vice versa.
    assert [p.parent.name for p in archive.files("statistics", "ZQ", root=tmp_path)] == ["OLD", "NEW", "LATE"]
    assert [p.parent.name for p in archive.files("statistics", "SR1", root=tmp_path)] == ["SR1"]
