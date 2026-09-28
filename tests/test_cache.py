"""The cache is a vintage log: re-appending is free, revisions never leak backwards."""

from functools import partial
from types import SimpleNamespace
import pandas as pd
import pytest
from policypath.sources import cache
from policypath.sources.base import Source

T = pd.Timestamp


def obs(rows, keys=("date",)):
    cols = [*keys, "value", "published"]
    df = pd.DataFrame(rows, columns=cols)
    for c in ("date", "published"):
        df[c] = pd.to_datetime(df[c])
    return df


def test_appending_the_same_rows_twice_adds_nothing(tmp_path):
    df = obs([("2024-03-01", 5.33, "2024-03-04"), ("2024-03-04", 5.33, "2024-03-05")])
    assert cache.append(df, "fred", "EFFR", "USD", root=tmp_path) == 2
    assert cache.append(df, "fred", "EFFR", "USD", root=tmp_path) == 0
    assert len(cache.log("fred", "EFFR", "USD", root=tmp_path)) == 2


def test_preliminary_and_final_settles_are_both_kept(tmp_path):
    """A read at the close sees the preliminary; a read the next day sees the final."""
    keys = ("date", "contract")
    df = obs([("2024-03-01", "ZQJ4", 94.680, "2024-03-01 15:01"),
              ("2024-03-01", "ZQJ4", 94.675, "2024-03-01 19:40"),
              ("2024-03-01", "ZQJ4", 94.675, "2024-03-03 08:05")], keys)  # an unchanged resend
    assert cache.append(df, "databento", "ZQ", "USD", keys=keys, root=tmp_path) == 2

    vintages = cache.log("databento", "ZQ", "USD", root=tmp_path)
    assert cache.view(vintages[vintages["published"] < T("2024-03-01 16:00")], "2024-03-01",
                      keys)["value"].item() == 94.680
    assert cache.read("databento", "ZQ", "USD", "2024-03-01", root=tmp_path)["value"].item() == 94.675


def test_a_view_at_the_end_of_a_day_sees_that_day_and_not_the_next_morning():
    """`view` is the first of two filters on every fixing the market path reads (`calendars.known_daily` is
    the second), so a one-day slip in either alone moves no output: each is pinned by its own test."""
    log = obs([("2024-03-01", 5.31, "2024-03-01 23:59"), ("2024-03-04", 5.32, "2024-03-05 00:00"),
               ("2024-03-05", 5.33, "2024-03-05 08:00")]).assign(retrieved=T("2024-03-06"))
    assert cache.view(log, "2024-03-04")["date"].tolist() == [T("2024-03-01")]
    assert cache.view(log, "2024-03-05")["date"].tolist() == [T("2024-03-01"), T("2024-03-04"), T("2024-03-05")]


def test_an_undated_revision_is_known_only_from_when_we_saw_it(tmp_path):
    """FRED reports a revised value with the original publication date."""
    cache.append(obs([("2024-03-01", 5.33, "2024-03-04")]), "fred", "SOFR", "USD", root=tmp_path)
    cache.append(obs([("2024-03-01", 5.31, "2024-03-04")]), "fred", "SOFR", "USD", root=tmp_path)

    assert cache.read("fred", "SOFR", "USD", "2024-03-04", root=tmp_path)["value"].item() == 5.33
    today = pd.Timestamp.now().normalize()
    assert cache.read("fred", "SOFR", "USD", today, root=tmp_path)["value"].item() == 5.31


def test_a_dated_vintage_backfilled_late_keeps_its_date(tmp_path):
    """The seam between two archive jobs: a session's final was cached first, from the
    later job, and its preliminary arrives with the earlier one. It is an older
    vintage, knowable at the close, not a revision to stamp with today."""
    keys = ("date", "contract")
    final = obs([("2020-12-30", "ZQF1", 99.915, "2020-12-30 19:13")], keys)
    prelim = obs([("2020-12-30", "ZQF1", 99.910, "2020-12-30 15:00")], keys)
    cache.append(final, "databento", "ZQ", "USD", keys=keys, root=tmp_path, dated=True)
    assert cache.append(prelim, "databento", "ZQ", "USD", keys=keys, root=tmp_path, dated=True) == 1

    log = cache.log("databento", "ZQ", "USD", root=tmp_path)
    assert sorted(log["published"]) == [T("2020-12-30 15:00"), T("2020-12-30 19:13")]
    at_close = cache.view(log[log["published"] < T("2020-12-30 16:00")], "2020-12-30", keys)
    assert at_close["value"].item() == 99.910
    assert cache.read("databento", "ZQ", "USD", "2020-12-30", root=tmp_path)["value"].item() == 99.915


def test_publication_date_is_required(tmp_path):
    df = obs([("2024-03-01", 5.33, None)])
    with pytest.raises(ValueError, match="publication date"):
        cache.append(df, "fred", "EFFR", "USD", root=tmp_path)


def test_manifest_reports_only_what_is_missing(tmp_path):
    cache.mark_fetched("fred", "EFFR", "USD", T("2024-01-01"), T("2024-01-31"), root=tmp_path)
    cache.mark_fetched("fred", "EFFR", "USD", T("2024-03-01"), T("2024-03-31"), root=tmp_path)
    assert cache.missing("fred", "EFFR", "USD", "2024-01-01", "2024-04-15", root=tmp_path) == [
        (T("2024-02-01"), T("2024-02-29")), (T("2024-04-01"), T("2024-04-15"))]
    cache.mark_fetched("fred", "EFFR", "USD", T("2024-02-01"), T("2024-02-29"), root=tmp_path)
    assert cache.fetched("fred", "EFFR", "USD", root=tmp_path) == [(T("2024-01-01"), T("2024-03-31"))]


class Counting(Source):
    """A source that serves business days from a fixed table and counts requests."""
    name = "fake"

    def __init__(self):
        self.calls = []

    def fetch(self, series, start, end):
        self.calls.append((start, end))
        dates = pd.bdate_range(start, min(end, T("2024-03-29")))
        return pd.DataFrame({"date": dates, "value": 1.0, "published": dates + pd.offsets.BDay()})


def test_update_is_incremental_and_rerunning_adds_nothing(tmp_path):
    rooted = SimpleNamespace(
        merge_ranges=cache.merge_ranges,
        **{name: partial(getattr(cache, name), root=tmp_path)
           for name in ["missing", "last_covered", "append", "mark_fetched"]})

    src = Counting()
    assert src.update("X", "USD", "2024-03-01", "2024-03-31", rooted) == 21
    assert len(src.calls) == 1
    # Covered through the last date served, not the date asked for: the 30th and
    # 31st may simply not be published yet, so only they are asked for again.
    assert cache.last_covered("fake", "X", "USD", root=tmp_path) == T("2024-03-29")

    assert src.update("X", "USD", "2024-03-01", "2024-03-31", rooted) == 0
    assert src.calls[1:] == [(T("2024-03-30"), T("2024-03-31"))]


def test_dropping_a_key_forgets_its_log_and_its_coverage(tmp_path):
    df = obs([("2024-03-01", 5.33, "2024-03-04")])
    cache.append(df, "fred", "DFEDTARU", "USD", root=tmp_path)
    cache.mark_fetched("fred", "DFEDTARU", "USD", T("2024-03-01"), T("2024-03-01"), root=tmp_path)
    cache.append(df, "fred", "EFFR", "USD", root=tmp_path)
    assert cache.drop("fred", "DFEDTARU", "USD", root=tmp_path)
    with pytest.raises(FileNotFoundError):
        cache.log("fred", "DFEDTARU", "USD", root=tmp_path)
    assert cache.fetched("fred", "DFEDTARU", "USD", root=tmp_path) == []
    assert len(cache.log("fred", "EFFR", "USD", root=tmp_path)) == 1     # nothing else touched
    assert not cache.drop("fred", "DFEDTARU", "USD", root=tmp_path)


def test_a_same_day_series_is_never_published_before_its_date(monkeypatch):
    """With no lag, a weekend's target range is public on the weekend day, not the Friday before."""
    from policypath.sources import fred
    rows = [{"date": d, "value": "5.50"} for d in ["2024-09-13", "2024-09-14", "2024-09-15", "2024-09-16"]]
    monkeypatch.setattr(fred, "_fetch", lambda *a, **k: rows)
    same_day = fred.observations("DFEDTARU", lag_bdays=0)
    assert (same_day["published"] == same_day["date"]).all()
    next_day = fred.observations("DFEDTARU", lag_bdays=1)
    assert next_day["published"].tolist() == list(pd.to_datetime(["2024-09-16"] * 3 + ["2024-09-17"]))


def vintage(rows):
    return pd.DataFrame(rows, columns=["date", "value", "realtime_start", "realtime_end"]).astype(
        {"date": "datetime64[ns]", "realtime_start": "datetime64[ns]", "realtime_end": "datetime64[ns]"})


def test_a_vintage_reprinted_to_the_last_bit_keeps_its_cached_value(tmp_path):
    """ALFRED re-prints old vintages at full float precision: not a revision, and the cached bits stay."""
    cache.write_vintages(vintage([("1990-07-01", 5.693902493, "2020-01-28", None)]), "alfred", "NROU", "USD",
                         root=tmp_path)
    again = vintage([("1990-07-01", 5.6939024929999995, "2020-01-28", "2020-08-02"),
                     ("1990-07-01", 5.70, "2020-08-03", None)])
    assert cache.write_vintages(again, "alfred", "NROU", "USD", root=tmp_path) == 1
    got = cache.vintages("alfred", "NROU", "USD", root=tmp_path)
    assert got["value"].tolist() == [5.693902493, 5.70]
    assert got["realtime_end"].iloc[0] == T("2020-08-02")


def test_a_real_revision_of_a_cached_vintage_still_refuses(tmp_path):
    cache.write_vintages(vintage([("1990-07-01", 5.693902493, "2020-01-28", None)]), "alfred", "NROU", "USD",
                         root=tmp_path)
    with pytest.raises(ValueError, match="changes 1 cached vintages"):
        cache.write_vintages(vintage([("1990-07-01", 5.6939025, "2020-01-28", None)]), "alfred", "NROU", "USD",
                             root=tmp_path)
    assert cache.vintages("alfred", "NROU", "USD", root=tmp_path)["value"].item() == 5.693902493

