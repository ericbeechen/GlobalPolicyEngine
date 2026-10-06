"""The costs report's runs end to end: no look-ahead, the expression seam, the sum's calendar, the grid's words.

The synthetic two-currency world of `test_expression.py`, with a model panel
per currency that puts it in the ELB state for a while before D (GBP
flickering in and out), a closure that needs fewer pairs than the book's (the
world is two and a half years long), and a grid trimmed to four cells so the
build is quick. Checks: poisoning every z, mark, overnight rate and ELB input
after D changes no position, cost or P&L credited through D, under every
rule, treatment and sizing and with the carry filter on; a run's gross P&L is
`expression.run`'s; the sum credits a session to the next book session and
adds only what each run's treatment keeps; a sleeve is evaluated from the
first session a position on its first z is held; the grid says "stable" only
when its test passes; and the build writes every output.
"""

import copy
import json
import re
import numpy as np
import pandas as pd
import pytest
from policypath import config
from policypath.backtest import metrics
from policypath.report import costs as report
from policypath.strategy import costs, expression, positions
from test_expression import CUT, PAIRS, world

T = pd.Timestamp
BOOK = config.strategy()
SPELLS = {"USD": [("2021-01-04", "2021-06-30"), ("2022-09-01", "2022-10-14")],
          "GBP": [("2022-06-01", "2022-06-20"), ("2022-07-11", "2022-08-01"), ("2023-03-01", "2023-04-03")]}


def book():
    b = copy.deepcopy(BOOK)
    b["carry"]["closure_min_pairs"] = PAIRS
    b["positions"]["grid"] = {"enter": [1.0, 1.5], "exit": [0.0, 0.5]}
    return b


def model_of(days, spells, poison_after=None):
    """A model panel on `days`: in the ELB state (rate and rule on a 0.1 floor) inside `spells`, above it outside."""
    at = pd.Series(False, index=days)
    for a, b in spells:
        at[a:b] = True
    out = pd.DataFrame({"session": days, "r0": np.where(at, 0.1, 0.6), "elb": 0.1, "at_elb": at.to_numpy()})
    if poison_after is not None:
        late = out["session"] > poison_after
        out.loc[late, "r0"] = 0.1
        out.loc[late, "at_elb"] = np.arange(late.sum()) % 3 == 0          # a flickering state after D
    return out


def world_of(root, poison_after=None):
    w = world(root, poison_after)
    w.book = book()
    for ccy, spells in SPELLS.items():
        w._panels[ccy]["model"] = model_of(pd.DatetimeIndex(w.panel(ccy, "signal")["session"].unique()), spells,
                                           poison_after)
    return w


@pytest.fixture(scope="module")
def clean(tmp_path_factory):
    w = world_of(tmp_path_factory.mktemp("clean"))
    sleeves = expression.build(w)
    return w, sleeves, {n: report.inputs(s, w, w.book) for n, s in sleeves.items()}


SPECS = [dict(), dict(sizing="unit", rule="linear", treatment="hold", enter=1.5, exit=0.5),
         dict(treatment="exclude", carry_filter=True, restrike_months=1)]


def test_nothing_credited_through_d_sees_later_z_marks_or_elb_state(clean, tmp_path):
    w, sleeves, inp = clean
    pw = world_of(tmp_path, poison_after=CUT)
    poisoned = {n: report.inputs(s, pw, pw.book) for n, s in expression.build(pw).items()}
    traded_before = 0
    for kw in SPECS:
        spec = report.chosen(w.book, **kw)
        for name in sleeves:
            a = report.simulate(inp[name], spec, w.book).daily
            b = report.simulate(poisoned[name], spec, pw.book).daily
            pd.testing.assert_frame_equal(a.loc[:CUT], b.loc[:CUT], obj=f"{name} {spec}")
            assert not a.loc[CUT:].iloc[1:].equals(b.loc[CUT:].iloc[1:])             # the poison is there to see
            traded_before += int((a.loc[:CUT, "traded"] > 0).sum())
            if kw == {} and name.endswith("outright"):
                assert (a.loc[:CUT, "held"] != 0).sum() > 50 and a.loc[:CUT, "cost"].sum() > 0
    assert traded_before > 200
    gates = [i.allow for i in inp.values()]
    assert any((~g.loc[:CUT]).any() for g in gates)                                # the carry gate bites before D
    assert inp["USD outright"].state.loc["2022-09-01":"2022-10-14"].all()


def test_a_runs_gross_pnl_is_the_expression_seam(clean):
    w, sleeves, inp = clean
    for name, s in sleeves.items():
        run = report.simulate(inp[name], report.chosen(w.book, "unit", rule="linear", treatment="hold"), w.book)
        _, sessions = expression.run(s, expression.linear(s))
        assert np.array_equal(run.daily["gross"].to_numpy(), sessions["pnl"].to_numpy())
        hyst = report.simulate(inp[name], report.chosen(w.book, "unit"), w.book).daily
        unit = inp[name].u.groupby("session")["total"].sum(min_count=1).reindex(s.sessions)
        assert np.allclose(hyst["gross"], (hyst["side"] * unit).fillna(0.0), atol=1e-12)


def test_flat_keeps_no_position_in_the_state_and_its_sessions_are_the_treatments(clean):
    w, sleeves, inp = clean
    i = inp["USD outright"]
    flat = report.simulate(i, report.chosen(w.book), w.book).daily
    held_in_state = i.state.shift(i.sleeve.lag + 1, fill_value=False)
    assert (flat.loc[held_in_state.to_numpy(), "held"] == 0).all()
    kept, *_ = positions.samples(i.state, "flat", i.sleeve.lag)
    assert flat["kept"].equals(kept & (flat.index >= i.start))


def run_of(days, gross, kept=None, start=None):
    days = pd.DatetimeIndex(days)
    daily = pd.DataFrame({c: 0.0 for c in report.SUMMED}, index=days).assign(
        gross=gross, kept=True if kept is None else kept, elb_line=False)
    return report.Run("r", None, daily, days[0] if start is None else T(start), metrics.per_year(days))


def test_the_sum_credits_a_session_to_the_next_book_session():
    book_days = pd.DatetimeIndex(["2024-05-24", "2024-05-28", "2024-05-29"])       # a US holiday on the 27th
    us = run_of(book_days, [1.0, 2.0, 4.0])
    uk = run_of(["2024-05-24", "2024-05-27", "2024-05-28"], [10.0, 20.0, 40.0], kept=[False, True, True],
                start="2024-05-27")
    total = report.combine([us, uk], book_days)
    assert total.daily["gross"].tolist() == [1.0, 62.0, 4.0]                      # the 27th is paid on the 28th
    assert total.start == T("2024-05-27") and total.daily["kept"].tolist() == [False, True, True]


def test_the_sum_adds_only_what_each_runs_treatment_keeps(clean):
    w, _, inp = clean
    days = report._book_days(w, w.book)
    spec = report.chosen(w.book, treatment="exclude")
    runs = {n: report.simulate(i, spec, w.book) for n, i in inp.items()}
    dropped = [r.daily.loc[~r.daily["kept"] & (r.daily.index >= r.start)] for r in runs.values()]
    assert not any((d["gross"] != 0).any() for d in dropped)               # exclude zeroes what it leaves out
    total = report.combine(list(runs.values()), days)
    kept = sum(r.daily["gross"].where(r.daily["kept"], 0.0).sum() for r in runs.values())
    late = sum(r.daily.loc[r.daily.index > days[-1], "gross"].sum() for r in runs.values())
    assert total.daily["gross"].sum() == pytest.approx(kept - late)
    us = run_of(["2024-05-24", "2024-05-28"], [1.0, 2.0])
    uk = run_of(["2024-05-24", "2024-05-28"], [10.0, 20.0], kept=[True, False])       # in the state on the 28th
    assert report.combine([us, uk], us.daily.index).daily["gross"].tolist() == [11.0, 2.0]


def test_a_sleeve_is_evaluated_from_the_first_session_a_position_on_its_first_z_is_held(clean):
    w, sleeves, inp = clean
    for name, s in sleeves.items():
        at = s.sessions.get_loc(s.component["z"].first_valid_index())
        assert inp[name].start == s.sessions[at + s.lag + 1]
        d = report.simulate(inp[name], report.chosen(w.book, "unit", rule="linear", treatment="hold"), w.book).daily
        first = d.index.get_loc(d.index[d["kept"]][0])
        assert d["held"].iloc[first] != 0 and d["held"].iloc[first - 1] == 0


def test_neighbours_are_the_adjacent_cells_that_are_rules():
    assert len(report.neighbours(BOOK, (1.0, 0.0))) == 8
    assert sorted(report.neighbours(BOOK, (0.5, -0.5))) == [(0.5, -0.25), (0.75, -0.5), (0.75, -0.25)]
    assert sorted(report.neighbours(BOOK, (0.5, 0.25))) == [(0.5, 0.0), (0.75, 0.0), (0.75, 0.25), (0.75, 0.5)]
    assert all(x < e for e, x in report.cells(BOOK)) and len(report.cells(BOOK)) == 33


def fake_grid(near_spread, se):
    here = (BOOK["positions"]["enter"], BOOK["positions"]["exit"])
    halves = {"split": T("2020-01-01"), "first_sr": 0.1, "first_se": 0.5, "second_sr": 0.2, "second_se": 0.5}
    grid = {}
    for cell in report.cells(BOOK):
        near = cell == here or cell in report.neighbours(BOOK, here)
        v = 0.3 + (near_spread if cell == (1.25, 0.25) else 0.0) if near else -1.0
        grid[cell] = {"x": {"net_sr": v, "net_se": se, "gross_sr": v + 0.1, "halves": halves}}
    return grid


def test_stable_needs_the_neighbour_range_under_one_se():
    assert report.surface(fake_grid(0.2, 0.3), BOOK, "x")["stable"]
    s = report.surface(fake_grid(0.4, 0.3), BOOK, "x")
    assert not s["stable"] and s["near_range"] == pytest.approx(0.4) and s["share_positive"] == pytest.approx(9 / 33)


@pytest.fixture(scope="module")
def written(clean, tmp_path_factory):
    """The build, its report and numbers, and one theme of each figure (`write` draws both, as the script runs it)."""
    w, _, _ = clean
    root = tmp_path_factory.mktemp("repo")
    built = report.build(w)
    paths = {"report": root / "costs.md", "json": root / "costs.json", "cost": root / "cost.png",
             "grid": root / "grid.png"}
    paths["report"].write_text(report.markdown(built))
    paths["json"].write_text(json.dumps(report.results(built)))
    report.cost_figure(built, paths["cost"])
    report.grid_figure(built, paths["grid"], "dark")
    return built, paths


def test_the_build_writes_every_output(written):
    built, paths = written
    for p in paths.values():
        assert p.exists() and p.stat().st_size > 0
    md = paths["report"].read_text()
    for head in ["## The answer", "## What each leg costs", "## The cost sensitivity", "## Turnover",
                 "## The hysteresis grid", "## Rules and ELB treatments", "## By year", "## The carry filter"]:
        assert head in md
    assert "Do not edit by hand" in md and not re.search(r"\bnan\b", md) and "Equal-risk sum" in md
    assert "not a measured bid-offer" in md and "fee inside c" in md
    s = report.surface(built.grid["vol_scaled"], built.book, report.SUM)
    assert ("grid is stable by" in md) == s["stable"]
    numbers = json.loads(paths["json"].read_text())
    assert set(numbers["chosen"]["vol_scaled"]) == {*built.sleeves, report.SUM}
    assert len(numbers["grid"]["vol_scaled"]) == len(report.cells(built.book))
    assert numbers["spec"]["treatment"] == built.book["evaluation"]["elb"]["chosen"]


def test_every_sharpe_in_the_tables_carries_its_se(written):
    built, paths = written
    md = paths["report"].read_text()
    rules = md[md.index("## Rules and ELB treatments"):md.index("### The ELB state")]
    cells = [c.strip() for line in rules.splitlines() if line.startswith("| ") and "---" not in line
             for c in line.split("|")[3:-1]]
    sharpes = [c for c in cells if c[:1] in "+-" and "." in c]
    assert sharpes and all("(" in c for c in sharpes)


def test_a_futures_legs_configured_cost_is_on_its_uniform_curve_at_the_mix(written):
    built, _ = written
    run = built.chosen["vol_scaled"]["USD outright"]
    st = report.stats(run)
    assert st["mix_bp"] == pytest.approx(built.inputs["USD outright"].costs[0].round_trip_bp)
    assert report.curve(run, [st["mix_bp"]])[0] == pytest.approx(st["net_sr"], abs=1e-9)
    assert report.curve(run, [0.0], "assumed")[0] == pytest.approx(st["net_sr"], abs=1e-9)    # only observed legs


def test_a_sleeves_round_trip_from_config_is_its_legs(clean):
    w, _, inp = clean
    for spec in w.book["sleeves"]:
        legs = inp[spec["name"]].costs
        assert costs.round_trip(spec) == pytest.approx(sum(c.round_trip_bp for c in legs))
        assert len(legs) == (1 if spec["kind"] == "outright" else 2)
