"""The expression report's builder: episodes, the next quarter, the groups, the names, and the outputs end to end.

Small frames for the pieces; the synthetic two-currency world of
`test_expression.py` for the build, written to a temporary tree. The world is
two and a half years long, so its closure needs fewer pairs than the book's
250 before the breakeven statistics have rows to count. The build refuses to
write when a check fails; the verdict on the breakeven reads the excluded
windows too.
"""

from dataclasses import replace
import json
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from policypath import config
from policypath.backtest import policy
from policypath.report import expression as report
from test_expression import PAIRS, world

T = pd.Timestamp
GBP = config.currency("GBP")


def test_episodes_follow_the_hysteresis_pair():
    days = pd.bdate_range("2024-01-01", periods=9)
    z = pd.Series([0.5, 1.2, 0.4, -0.1, -1.5, 2.0, np.nan, -1.0, -0.2], index=days)
    e = report.episodes(z, enter=1.0, exit=0.0)
    assert list(e["entry"]) == [days[1], days[4], days[5], days[7]]
    assert list(e["side"]) == [1, -1, 1, -1]
    # out when z x side <= 0, flipped on the same session, out on a missing z; the last still open
    assert list(e["exit"][:3]) == [days[3], days[5], days[6]] and pd.isna(e["exit"].iloc[3])


def test_the_next_quarter_starts_after_execution_and_needs_the_whole_window():
    days = pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-05", "2024-01-10", "2024-01-11"])
    parts = pd.DataFrame({"carry": [0.0, 1, 2, 4, 8], "total": [100.0, 1, 2, 4, 8]}, index=days)
    sleeve = SimpleNamespace(sessions=days, lag=1)
    side = pd.Series([1.0, -1.0, 1.0, 1.0, 1.0], index=days)
    out = report.next_quarter(sleeve, parts, side, h=4)
    # decided 1 Jan, executed 2 Jan, credited on the sessions after it to 6 Jan: 5 Jan only
    assert out.loc[days[0], "next_total"] == 2.0 and out.loc[days[0], "next_carry"] == 2.0
    assert out.loc[days[1], "next_total"] == 0.0           # executed 5 Jan, nothing credited by 9 Jan; paid side
    assert out.loc[days[2]:, "next_total"].isna().all()   # the window runs past the last session
    paid = report.next_quarter(sleeve, parts, -side, h=4)
    assert paid.loc[days[0], "next_total"] == -2.0         # a pay loses what a receive makes


def test_a_quarter_overlaps_a_window_from_its_execution_to_h_days_on():
    frame = pd.DataFrame({"executed": pd.to_datetime(["2021-09-30", "2021-10-05", "2022-06-01", "2023-01-02"])})
    got = report.overlaps(frame, T("2022-01-01"), T("2022-12-31"), 91)
    assert got.tolist() == [False, True, True, False]
    assert report.window_label(T("2022-01-01"), T("2022-12-31")) == "2022"
    assert report.window_label(T("2021-12-01"), T("2023-08-31")) == "Dec 2021 to Aug 2023"


def means(bleeding, rest, no_pay, pays):
    return {"next_bleeding_bp": bleeding, "next_rest_bp": rest, "next_no_pay_bp": no_pay, "next_pays_bp": pays}


def test_the_verdict_says_when_leaving_a_window_out_changes_it():
    full = means(7.0, 1.0, 11.0, 3.0)                     # neither flag did worse
    same = {"w": {"label": "2019", "pooled": means(5.0, 1.0, 4.0, 3.0)}}
    flipped = {"w": {"label": "2022", "pooled": means(0.8, 1.5, -0.7, 1.5)}}
    assert report.judgement(full, {}) == report.VERDICTS[0] == report.judgement(full, same)
    head, tail = report.judgement(full, flipped)
    assert head == "The breakeven's verdict turns on 2022"
    assert tail.startswith("In the full sample neither flag did worse; without 2022 both flags did worse.")

def test_groups_and_verdicts_are_one_classifier():
    """The last row earns carry and roll with a negative edge, so its E_h < 0: it earns, it does not bleed."""
    rows = pd.DataFrame({"cr_h_bp": [3.0, -2.0, -2.0, -2.0, 2.0], "pays": pd.array([True, True, False, pd.NA, False])})
    assert list(report.group(rows).fillna("none")) == ["earns", "bleeds_pays", "no_pay", "none", "earns"]
    verdicts = [report.verdict(r) for _, r in rows.iterrows()]
    assert verdicts == ["earns carry and roll", "bleeds, pays", "bleeds, does not pay", "bleeds", "earns carry and roll"]
    known = report.group(rows).notna()
    assert (report.group(rows)[known].map(report.GROUPS) == pd.Series(verdicts)[known]).all()


def test_instrument_names_are_words():
    assert report._name("futures_month", "ZZ", "2027-04") == "ZZ Apr 2027"
    assert report._name("curve_forward", "fwd", "2027-03-18/2027-04-29") == "fwd 18 Mar-29 Apr 2027"
    assert report._name("curve_forward", "fwd", "2011-12-08/2012-01-12") == "fwd 8 Dec 2011-12 Jan 2012"
    assert report._name("par_yield", "bond", "2y") == "2y bond"


def test_the_example_is_right_on_the_rate_and_lost_the_most_to_carry_and_roll():
    eps = pd.DataFrame({"next_rate": [5.0, 5.0, -1.0, 2.0], "next_total": [-1.0, -3.0, -9.0, 1.0],
                        "next_carry": [-6.0, -4.0, -8.0, 0.0], "next_roll": [0.0, -4.0, 0.0, -1.0]})
    assert report.example(eps).name == 1
    assert report.example(eps[eps["next_rate"] < 0]) is None


def test_a_range_needs_two_sleeves():
    stats = {"a": {"signals": {"x": 0.1}}, "b": {"signals": {"x": np.nan}}, "pooled": {"signals": {"x": 0.3}}}
    assert report._range(stats, "x") == ""
    stats["b"]["signals"]["x"] = 0.5
    assert report._range(stats, "x") == ", from 10% (a) to 50% (b)"


def test_the_contract_is_compared_with_its_exchanges_published_quotes():
    ct = {"name": "XX", "key": "XX", "ccy": "USD", "exchange": "EX", "notional": 5e6, "days": 30, "year_days": 360,
          "dv01": 5e6 * 30 / 360 * 1e-4, "dv01_quoted": 41.67,
          "ticks": {"front": {"tick": 0.0025, "value": 10.4167, "quoted": 10.4175},
                    "other": {"tick": 0.005, "value": 20.8333, "quoted": 20.835}}}
    assert "EX quotes 41.67 (`contracts.XX.quoted`), the derived value rounded to the cent" in report._contract_lines(ct)
    assert "its rounded DV01 times the tick" in report._contract_lines(ct)
    wrong = report._contract_lines({**ct, "dv01_quoted": 41.5})
    assert "-0.1667 from the derived value" in wrong and "times the tick" not in wrong


def test_the_results_keep_tiny_residuals():
    assert report._plain({"a": 7.105e-15, "b": 10.51234567, "c": float("nan")}) == {"a": 7.105e-15, "b": 10.5123,
                                                                                    "c": None}


def test_latest_is_point_in_time():
    frame = pd.DataFrame({"sleeve": ["a", "a", "b", "b"], "z": [1.0, 2.0, np.nan, -1.0],
                          "session": pd.to_datetime(["2024-01-01", "2024-01-08", "2024-01-02", "2024-01-03"])})
    now = report.latest(frame, "2024-01-05")
    assert list(now.index) == ["a", "b"] and now.loc["a", "z"] == 1.0 and now.loc["b", "z"] == -1.0


@pytest.fixture(scope="module")
def written(tmp_path_factory):
    root = tmp_path_factory.mktemp("repo")
    w = world(root / "cache")
    w.book = {**w.book, "carry": {**w.book["carry"], "closure_min_pairs": PAIRS}}
    w._panels["GBP"]["backtest"] = policy.run(w.panel("GBP", "signal"), w.panel("GBP", "sessions"),
                                              w.panel("GBP", "meetings"), GBP["backtest"]["horizon"],
                                              GBP["backtest"]).reset_index(names="session")
    built = report.build(w)
    paths, stats, filt = report.write(built, root, panel_root=root / "panel")
    return built, paths, stats, filt


def test_the_build_writes_every_output(written):
    built, paths, stats, _ = written
    for p in paths.values():
        assert p.exists() and p.stat().st_size > 0
    md = paths["report"].read_text()
    for head in ["## The answer", "## The breakeven, per sleeve", "## Now", "## Instruments", "## The three checks",
                 "### Convexity", "## P&L by component", "## The components"]:
        assert head in md
    assert "Do not edit by hand" in md and "no costs" in md and "nan" not in md
    assert "A correct signal can still be a losing trade" in md and "The breakeven" in md
    for w in built.book["evaluation"]["exclude"].values():
        label = report.window_label(T(w[0]), T(w[1]))
        assert f"Leaving out every quarter that overlaps {label}" in md
    assert "descriptions, not tests" in md
    numbers = json.loads(paths["json"].read_text())
    assert "generated" not in numbers and set(numbers["without"]) == set(built.book["evaluation"]["exclude"])
    assert set(numbers["now"]) == set(built.sleeves)
    assert numbers["checks"]["outside_bound"] == 0 and numbers["checks"]["identity_max_bp"] < 1e-9
    assert numbers["checks"]["reproduced"][0]["max_bp"] < 1e-9
    panel = pd.read_parquet(paths["panel"])
    assert len(panel) == sum(len(s.sessions) for s in built.sleeves.values())


def test_the_numbers_agree_with_the_panel(written):
    built, _, stats, filt = written
    enter = built.book["positions"]["enter"]
    for name in built.sleeves:
        rows = report.sample(built.panel[built.panel["sleeve"] == name], enter)
        s = stats[name]["signals"]
        assert s["sessions"] == len(rows)
        if name.endswith("outright"):
            assert len(rows) > 50                     # the statistics are over real rows, not an empty sample
        if not len(rows):
            continue
        assert sum(s["shares"].values()) == pytest.approx(1.0)
        assert s["bleeding"] == pytest.approx(((rows["cr_h_bp"] < 0) & (rows["edge_bp"] > 0)).mean())
    assert (filt["effect_bp"] == filt["kept_sum_bp"] - filt["all_sum_bp"]).all()


def test_the_build_refuses_to_write_when_a_check_fails(written, tmp_path):
    built = written[0]
    broken = replace(built, checks=built.checks.assign(outside_bound=built.checks["outside_bound"] + 1))
    with pytest.raises(ValueError, match="outside the revaluation bound"):
        report.write(broken, tmp_path, panel_root=tmp_path)
    off = replace(built, reproduced=[(n, c, k, 1e-6) for n, c, k, _ in built.reproduced])
    with pytest.raises(ValueError, match="reproduces"):
        report.write(off, tmp_path, panel_root=tmp_path)
    assert not list(tmp_path.iterdir())                  # nothing written


def test_an_outrights_closure_splits_into_the_market_and_the_rule(written):
    built = written[0]
    h = built.book["carry"]["horizon_days"]
    assert set(built.splits) == {n for n, s in built.sleeves.items() if s.kind == "outright"}
    for name, sp in built.splits.items():
        c = built.sleeves[name].component
        end = c.index.searchsorted(c.index + pd.Timedelta(days=h))
        ok = end < len(c)
        x, v = c["dev_bp"].to_numpy()[ok], c["value_bp"].to_numpy()
        y = -(v[end[ok]] - v[ok])
        live = ~(np.isnan(x) | np.isnan(y))
        assert sp["phi"] == pytest.approx(x[live] @ y[live] / (x[live] @ x[live]), rel=1e-9)   # the gap's own closure
        assert sp["market"] + sp["rule"] == pytest.approx(sp["phi"]) and sp["rule"] != 0


def test_the_positions_table_receives_the_first_leg_on_a_positive_side(written):
    built = written[0]
    s = built.sleeves["USD 2s10s"]
    rows = report.positions(s, pd.Series({"side": 1.0}, name=s.sessions[-1]))
    assert [(r["leg"], r["side"], r["dv01"][0]) for r in rows] == [("USD 2y", "receive", "+"), ("USD 10y", "pay", "-")]


def test_signals_csv_is_one_row_per_episode_and_derived(written):
    built, paths, _, _ = written
    csv = pd.read_csv(paths["signals"], parse_dates=["entry", "exit"])
    assert len(csv) == len(built.episodes)
    assert set(csv["side"]) <= {"receive", "pay"}
    assert not {"price_t", "price_p", "held_t", "held_p", "rate"} & set(csv.columns)   # no marks, derived only
