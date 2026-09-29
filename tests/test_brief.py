"""The brief: the footer reads only the blocks the brief reads, the trades table is point in time and carries
each sleeve's round trip, and "what changed" says so when there is nothing newer.
"""

import pandas as pd
from policypath import config
from policypath.report import brief

SETTINGS = brief.settings()
BLOCKS = SETTINGS["footer_blocks"]


def test_tags_are_scoped_to_top_level_blocks():
    cfg = {"market": {"tags": {"a": {"lag": "one"}}, "curve": {"tags": {"b": {"quality": "two"}}}},
           "credit": {"tags": {"c": {"quality": "three"}}}, "tags": {"d": {"modelling": "four"}}}
    assert [t for _, t in brief.tags(cfg)] == ["one", "two", "three", "four"]
    assert brief.tags(cfg, ["market"]) == [("a", "one"), ("b", "two")]
    assert brief.tags(cfg, []) == []


def test_the_footer_blocks_are_blocks_every_brief_currency_has():
    for ccy in SETTINGS["currencies"]:
        missing = [b for b in BLOCKS if b not in config.currency(ccy)]
        assert not missing, (ccy, missing)


def test_the_footer_leaves_out_what_the_brief_does_not_read():
    scoped = {line.split(": ", 1)[1] for line in brief.footer(SETTINGS["currencies"], BLOCKS)}
    for ccy in SETTINGS["currencies"]:
        cfg = config.currency(ccy)
        inside = {t for _, t in brief.tags(cfg, BLOCKS)}
        outside = {t for _, t in brief.tags(cfg, [b for b in cfg if b not in BLOCKS])} - inside
        checks = {t for n, t in brief.tags(cfg, BLOCKS) if n in brief.checks_only(cfg, BLOCKS)}
        assert not (outside | checks) & scoped
        assert inside - checks <= scoped
        assert {t for _, t in brief.tags(cfg, ["expression"])} <= scoped     # the trades table's instruments
    left_out = {t for c in SETTINGS["currencies"] for _, t in brief.tags(config.currency(c), ["credit", "sources"])}
    assert left_out and not left_out & scoped                             # the credit bridge and HLW r*


def test_the_trades_table_reads_each_sleeve_on_or_before_the_date():
    frame = pd.DataFrame({
        "sleeve": ["a", "a", "b"], "session": pd.to_datetime(["2024-01-02", "2024-01-09", "2024-01-03"]),
        "z": [0.5, 2.0, -1.4], "side": [1.0, 1.0, -1.0], "instrument": ["x", "x", "y vs z"],
        "edge_bp": [3.0, 30.0, 12.0], "cr_h_bp": [2.0, 5.0, -4.0], "expected_bp": [2.5, 9.0, float("nan")],
        "pays": pd.array([True, True, pd.NA])})
    rows = brief.trades(frame, "2024-01-05", enter=1.0)
    assert [r[0] for r in rows] == ["a", "b"]
    assert rows[0][2] == "+0.5*" and rows[0][8] == "earns carry and roll"      # below entry: starred
    assert rows[1][3] == "pay" and rows[1][7] == "n/a" and rows[1][8] == "bleeds"
    costed = brief.trades(frame, "2024-01-05", enter=1.0, round_trips={"a": 1.048, "b": 2.0})
    assert [r[8] for r in costed] == ["1.05bp", "2.00bp"] and costed[1][9] == "bleeds"   # before the verdict


def test_what_changed_says_so_when_there_is_no_newer_session():
    snap = {"session": pd.Timestamp("2026-09-21")}
    assert brief.changed("AAA", snap, snap, 4, {"report": {"labels": {}}}) == (
        "AAA: no session since 21 Sep, so nothing to compare yet.")
