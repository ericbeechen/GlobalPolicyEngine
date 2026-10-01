"""`uv run all`: it updates every currency first, and stops on a cache whose currencies were updated on different days.

The steps are checked as a list (nothing runs). The freshness check reads a
manifest and update stamps written to a temporary folder, so it needs no cache.
"""

import json
import pandas as pd
from policypath import config, run_all
from policypath.sources import cache


def test_the_update_comes_first_and_covers_every_currency():
    todo = run_all.steps(update=True)
    assert todo[0] == ["update_data", "--ccy", *config.enabled()]
    assert ["update_data"] not in [s[:1] for s in todo[1:]]
    for ccy in config.enabled():
        assert ["build_panel", "--ccy", ccy] in todo


def test_cache_only_pulls_nothing():
    assert all(s[0] != "update_data" for s in run_all.steps(update=False))


def manifest(tmp_path, updated):
    """A cache whose market series cover to 2026-09-25 and whose currencies last updated at `updated` {ccy: time}."""
    entries = {}
    for ccy in config.enabled():
        source, series = run_all.market_series(config.currency(ccy))
        entries[f"{source}/{ccy}/{series}"] = {"keys": ["date"], "ranges": [["2009-01-01", "2026-09-25"]],
                                               "updated": "2026-09-01T00:00:00"}
    (tmp_path / cache.MANIFEST).write_text(json.dumps(entries), encoding="utf-8")
    (tmp_path / cache.UPDATES).write_text(json.dumps(updated), encoding="utf-8")
    return tmp_path


def test_updated_the_same_day_is_in_step(tmp_path):
    ccys = config.enabled()
    root = manifest(tmp_path, {c: f"2026-10-01T{9 + i:02d}:00:00" for i, c in enumerate(ccys)})
    rows = run_all.freshness(ccys, root)
    assert [r[2] for r in rows] == [pd.Timestamp("2026-09-25")] * len(ccys)
    assert run_all.in_step(rows) is None


def test_a_currency_updated_days_earlier_stops_the_run(tmp_path):
    first, *rest = config.enabled()
    root = manifest(tmp_path, {first: "2026-10-01T14:00:00", **{c: "2026-09-28T16:00:00" for c in rest}})
    stop = run_all.in_step(run_all.freshness(config.enabled(), root))
    assert stop.startswith("the cache is out of step")
    assert f"{first} 2026-10-01" in stop and all(f"{c} 2026-09-28" in stop for c in rest)


def test_a_currency_never_updated_stops_the_run(tmp_path):
    first, *rest = config.enabled()
    root = manifest(tmp_path, {first: "2026-10-01T14:00:00"})
    stop = run_all.in_step(run_all.freshness(config.enabled(), root))
    assert stop and all(f"{c} never" in stop for c in rest)


def test_mark_updated_keeps_the_other_currencies(tmp_path, monkeypatch):
    first, *rest = config.enabled()
    monkeypatch.setattr(cache, "_now", lambda: pd.Timestamp("2026-10-01 14:00"))
    for ccy in config.enabled():
        cache.mark_updated(ccy, tmp_path)
    monkeypatch.setattr(cache, "_now", lambda: pd.Timestamp("2026-10-02 09:00"))
    cache.mark_updated(first, tmp_path)
    assert cache.last_updated(first, tmp_path) == pd.Timestamp("2026-10-02 09:00")
    assert all(cache.last_updated(c, tmp_path) == pd.Timestamp("2026-10-01 14:00") for c in rest)
