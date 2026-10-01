"""Settlements past the archive, over Databento's historical API, offline.

`Fake` stands in for `databento.Historical`: it serves a fixed set of
statistics and definition records by receive time, as the API does, prices
each request by its rows, and records every request. The archive is one
statistics job's metadata, ending 2026-09-21, with no daily files.
"""

from functools import partial
import json
from types import SimpleNamespace
import databento as db
import pandas as pd
import pytest
from policypath.sources import cache, rates, registry

T = pd.Timestamp
SETTLE, OPEN_INTEREST = db.StatType.SETTLEMENT_PRICE.value, db.StatType.OPEN_INTEREST.value
U6, V6, SPREAD = 102, 103, 900   # ZQU6 expires 2026-09-30, ZQV6 2026-10-30; ZQU6-ZQV6 is a calendar spread


def utc(s):
    return T(s, tz="UTC")


def stat(recv, trade, iid, symbol, price, final, stat_type=SETTLE):
    return {"ts_recv": utc(recv), "ts_ref": utc(trade), "instrument_id": iid, "symbol": symbol, "stat_type": stat_type,
            "price": price, "quantity": 0, "stat_flags": rates.FINAL if final else 0, "update_action": 1}


STATS = pd.DataFrame([
    stat("2026-09-22 20:00", "2026-09-22", U6, "ZQU6", 95.800, False),      # preliminary, 16:00 New York
    stat("2026-09-23 01:00", "2026-09-22", U6, "ZQU6", 95.805, True),       # its final that evening
    stat("2026-09-23 01:00", "2026-09-22", V6, "ZQV6", 96.000, True),
    stat("2026-09-23 01:00", "2026-09-22", SPREAD, "ZQU6-ZQV6", 0.195, True),              # a spread: dropped
    stat("2026-09-23 01:00", "2026-09-22", U6, "ZQU6", 250000.0, False, OPEN_INTEREST),   # not a settle: dropped
    stat("2026-09-30 20:00", "2026-09-30", U6, "ZQU6", 95.790, False),      # ZQU6's last day
    stat("2026-10-01 00:30", "2026-09-30", U6, "ZQU6", 95.7925, True),      # its final, the day after it expired
    stat("2026-10-01 00:30", "2026-09-30", V6, "ZQV6", 96.010, True),
    stat("2026-10-01 20:00", "2026-10-01", U6, "ZQU6", 95.790, False),      # after expiry: dropped
])


def definitions():
    rows = []
    for day in pd.date_range("2026-09-10", "2026-10-05"):
        live = [(U6, "ZQU6", "F", "2026-09-30 21:00"), (V6, "ZQV6", "F", "2026-10-30 21:00"),
                (SPREAD, "ZQU6-ZQV6", "S", "2026-09-30 21:00")]
        for iid, sym, cls, exp in live:
            if utc(exp).normalize() >= utc(day):   # an expired instrument is no longer defined
                rows.append({"ts_recv": utc(day) + pd.Timedelta(seconds=5), "instrument_id": iid, "raw_symbol": sym,
                             "asset": "ZQ", "instrument_class": cls, "expiration": utc(exp),
                             "activation": utc("2025-09-01"), "min_price_increment": 0.0025,
                             "unit_of_measure_qty": 5_000_000.0, "currency": "USD"})
    return pd.DataFrame(rows)


class Store:
    def __init__(self, frame):
        self.frame = frame

    def to_df(self):
        return self.frame.set_index("ts_recv")


class Fake:
    """The parts of `databento.Historical` the source uses. Quotes $0.0001 a row."""

    def __init__(self, available, conditions=None):
        self.available, self.conditions = available, conditions or {}
        self.frames = {"statistics": STATS, "definition": definitions()}
        self.requests, self.quotes = [], []
        self.metadata = SimpleNamespace(get_cost=self.get_cost, get_dataset_range=self.get_dataset_range,
                                        get_dataset_condition=self.get_dataset_condition)
        self.timeseries = SimpleNamespace(get_range=self.get_range)

    def _rows(self, dataset, schema, symbols, stype_in, start, end):
        assert dataset == rates.DATASET and start < end <= utc(self.available)
        f = self.frames[schema]
        f = f[(f["ts_recv"] >= start) & (f["ts_recv"] < end)]
        if stype_in == "instrument_id":
            return f[f["instrument_id"].isin(symbols)]
        assert stype_in == "parent" and symbols == ["ZQ.FUT"]
        return f

    def get_cost(self, **query):
        self.quotes.append(query["schema"])
        return 0.0001 * len(self._rows(**query))

    def get_range(self, **query):
        self.requests.append((query["schema"], query["stype_in"], list(query["symbols"]),
                              query["start"].tz_localize(None), query["end"].tz_localize(None)))
        return Store(self._rows(**query).reset_index(drop=True))

    def get_dataset_range(self, dataset):
        end = f"{T(self.available):%Y-%m-%dT%H:%M:%S}.000000000Z"
        return {"start": "2010-06-06T00:00:00.000000000Z", "end": end,
                "schema": {s: {"start": "2010-06-06T00:00:00.000000000Z", "end": end} for s in self.frames}}

    def get_dataset_condition(self, dataset, start_date, end_date):
        return [{"date": f"{d:%Y-%m-%d}", "condition": self.conditions.get(f"{d:%Y-%m-%d}", "available")}
                for d in pd.date_range(start_date, end_date)]


@pytest.fixture
def root(tmp_path):
    """An archive whose one ZQ statistics job covers 2026-09-01 to 2026-09-21, and an empty cache beside it."""
    job = tmp_path / "archive" / "statistics" / "JOB-ZQ"
    job.mkdir(parents=True)
    query = {"schema": "statistics", "symbols": ["ZQ.FUT"], "start": T("2026-09-01").value, "end": T("2026-09-22").value}
    (job / "metadata.json").write_text(json.dumps({"query": query}), encoding="utf-8")
    return tmp_path


def rooted(root):
    return SimpleNamespace(merge_ranges=cache.merge_ranges,
                           **{n: partial(getattr(cache, n), root=root / "cache")
                              for n in ["missing", "last_covered", "append", "mark_fetched"]})


def update(src, root, end):
    return sum(src.update("ZQ", "USD", lo, hi, rooted(root)) for lo, hi in src.ranges("ZQ", end))


def logged(root):
    log = cache.log("databento", "ZQ", "USD", root / "cache")
    return log.sort_values(["published", "contract"])[["date", "contract", "value", "is_final", "published"]]


def test_ranges_reach_past_the_archive_only_with_a_client(root):
    archive_only = [(T("2026-09-01"), T("2026-09-21"))]
    assert rates.Settlements(root / "archive").ranges("ZQ", "2026-09-30") == archive_only
    live = rates.Settlements(root / "archive", client=Fake("2026-09-30 13:00"))
    assert live.ranges("ZQ", "2026-09-30") == [*archive_only, (T("2026-09-22"), T("2026-09-30"))]
    assert live.ranges("ZQ", "2026-09-21") == archive_only


def test_live_days_go_through_the_archive_filters_into_the_same_log(root):
    client = Fake("2026-09-30 13:00")
    src = rates.Settlements(root / "archive", client=client)

    assert update(src, root, "2026-09-30") == 3

    # The spread, the open interest and ZQU6's settles after 13:00 on the 30th are not there.
    assert logged(root).values.tolist() == [
        [T("2026-09-22"), "ZQU6", 95.800, False, T("2026-09-22 16:00")],
        [T("2026-09-22"), "ZQU6", 95.805, True, T("2026-09-22 21:00")],
        [T("2026-09-22"), "ZQV6", 96.000, True, T("2026-09-22 21:00")],
    ]
    # Covered through the last day Databento has whole, not the 30th it has only part of.
    assert cache.fetched("databento", "ZQ", "USD", root / "cache") == [(T("2026-09-01"), T("2026-09-29"))]
    # Every contract settling in the window is defined in it, so nothing is looked up further back.
    assert [(s, stype) for s, stype, *_ in client.requests] == [("statistics", "parent"), ("definition", "parent")]
    assert client.requests[0][3:] == (T("2026-09-22"), T("2026-09-30 13:00"))
    assert src.spent == pytest.approx(0.0001 * (5 + 9 * 3))   # 5 statistics records, 9 days of 3 definitions


def test_a_day_not_whole_or_not_available_is_asked_again_and_only_what_arrived_since_is_added(root):
    first = Fake("2026-09-30 13:00", conditions={"2026-09-25": "degraded"})
    assert update(rates.Settlements(root / "archive", client=first), root, "2026-09-30") == 3
    assert cache.last_covered("databento", "ZQ", "USD", root / "cache") == T("2026-09-24")

    later = Fake("2026-10-02 12:00")
    assert update(rates.Settlements(root / "archive", client=later), root, "2026-10-02") == 3

    assert logged(root).tail(3).values.tolist() == [
        [T("2026-09-30"), "ZQU6", 95.790, False, T("2026-09-30 16:00")],
        [T("2026-09-30"), "ZQU6", 95.7925, True, T("2026-09-30 20:30")],
        [T("2026-09-30"), "ZQV6", 96.010, True, T("2026-09-30 20:30")],
    ]
    assert cache.last_covered("databento", "ZQ", "USD", root / "cache") == T("2026-10-01")
    # Asked again from the 25th, in the month pieces an update is split into. ZQU6's final
    # arrived on the 1st, after its last day, so only it is looked up in the week before.
    assert [(r[0], r[1], r[3]) for r in later.requests] == [
        ("statistics", "parent", T("2026-09-25")), ("definition", "parent", T("2026-09-25")),
        ("statistics", "parent", T("2026-10-01")), ("definition", "parent", T("2026-10-01")),
        ("definition", "instrument_id", T("2026-09-24"))]
    assert later.requests[-1][2] == [U6]


def test_a_request_over_budget_is_refused_before_it_is_made(root):
    client = Fake("2026-09-30 13:00")
    src = rates.Settlements(root / "archive", client=client, budget=0.0001)

    with pytest.raises(rates.LiveBudgetExceeded, match="past its \\$0.00 budget"):
        update(src, root, "2026-09-30")

    assert client.quotes == ["statistics"] and client.requests == [] and src.spent == 0.0
    assert cache.fetched("databento", "ZQ", "USD", root / "cache") == [(T("2026-09-01"), T("2026-09-21"))]


def test_a_fetch_across_the_archive_end_is_refused(root):
    src = rates.Settlements(root / "archive", client=Fake("2026-09-30 13:00"))
    with pytest.raises(ValueError, match="runs past the archive's last day, 2026-09-21"):
        src.fetch("ZQ", T("2026-09-15"), T("2026-09-25"))


def test_nothing_is_asked_for_before_databento_has_it(root):
    client = Fake("2026-09-21 23:00")
    src = rates.Settlements(root / "archive", client=client)
    assert update(src, root, "2026-09-30") == 0
    assert client.requests == [] and src.spent == 0.0
    assert cache.last_covered("databento", "ZQ", "USD", root / "cache") == T("2026-09-21")


def test_the_registry_goes_live_only_with_a_key_and_never_archive_only(monkeypatch):
    monkeypatch.delenv(rates.LIVE_KEY, raising=False)
    assert registry.lookup(registry.FUTURES, "databento")(True).client is None
    monkeypatch.setenv(rates.LIVE_KEY, "db-" + "x" * 29)
    assert registry.lookup(registry.FUTURES, "databento")(False).client is None
    assert isinstance(registry.lookup(registry.FUTURES, "databento")(True).client, db.Historical)
