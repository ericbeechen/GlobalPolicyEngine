"""The brief: the footer reads only the blocks the brief reads, the trades table is point in time and carries
each sleeve's round trip, and "what changed" says so when there is nothing newer, and never calls a bridged
month printed.
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


def _read(session, market, model, gap_bp, z, inflation, macro):
    """A read as `brief.read` returns it, with only the fields "what changed" uses."""
    meetings = pd.DataFrame({"k": [4], "market": [market], "model": [model], "gap_bp": [gap_bp], "z": [z],
                             "effective_date": [pd.Timestamp("2027-04-29")],
                             "announcement_date": [pd.Timestamp("2027-04-28")]})
    rule = {"r0": 3.88, "inflation": inflation, "u_gap": -0.29, "rstar": 3.0}
    return {"session": pd.Timestamp(session), "meetings": meetings, "rule": rule, "macro": pd.Series(macro)}


def test_what_changed_calls_a_bridged_month_bridged_and_a_print_printed():
    cfg = {"report": {"labels": {"inflation": "Core PCE", "inflation_bridge": "core CPI", "policy": "fed funds",
                                 "unemployment_period": "for {month:%B}"}},
           "rule": {"coefficients": {"inflation_gap": 0.5, "unemployment_gap": 2.0}, "rstar": {}}}
    october = pd.Timestamp("2026-10-01")
    before = _read("2026-11-06", 3.94, 4.00, -6.0, 1.4, 3.12, {"infl_month": pd.Timestamp("2026-09-01"), "n_bridged": 0})
    head = ("AAA, 6 Nov to 13 Nov: the gap at the fourth meeting went from -6bp to +24bp (z +1.4 to +2.6). "
            "The market moved +35bp, and the rule +4bp because ")
    for n, why in [(1, "core PCE for October, bridged from core CPI, is 3.2% (from 3.1%)."),
                   (2, "core PCE for October, bridged from core CPI, is 3.2% (from 3.1%)."),
                   (0, "core PCE for October printed 3.2% (from 3.1%).")]:
        now = _read("2026-11-13", 4.29, 4.04, 24.0, 2.6, 3.23, {"infl_month": october, "n_bridged": n})
        assert brief.changed("AAA", now, before, 4, cfg) == head + why
    # The week the month prints, last week's figure for it was the bridge's, not an earlier print.
    bridge = _read("2026-11-06", 3.94, 4.00, -6.0, 1.4, 3.23, {"infl_month": october, "n_bridged": 1})
    now = _read("2026-11-13", 4.29, 4.04, 24.0, 2.6, 3.31, {"infl_month": october, "n_bridged": 0})
    assert brief.changed("AAA", now, bridge, 4, cfg) == head + "core PCE for October printed 3.3% (the bridge had 3.2%)."
    # A published 12-month rate (macro.inflation.rate) has no bridge column and no label for one.
    labels = {k: v for k, v in cfg["report"]["labels"].items() if k != "inflation_bridge"}
    now = _read("2026-11-13", 4.29, 4.04, 24.0, 2.6, 3.23, {"infl_month": october})
    assert brief.changed("AAA", now, before, 4, {**cfg, "report": {"labels": labels}}) == (
        head + "core PCE for October printed 3.2% (from 3.1%).")
