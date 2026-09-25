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
