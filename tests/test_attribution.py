"""Attribution, evaluation and the tear sheet end to end on the synthetic two-currency world (`test_portfolio.py`).

Checks: the split adds up (carry + roll + rate is each sleeve's gross in the
book, an outright is all level, the component groups' variance shares add to
one); the regime table partitions the counted sessions; the ELB split puts no
P&L on positions decided in the state where a treatment has none; the carry
benchmark trades only the sign of carry and roll; ex-2022 is `metrics.evaluate`
without the window; the costs by cause are the cost charged, and turns x mean
gross DV01 x the one-way cost is the cost a year; the information identity
solves for the rate needed; carry against rate by sleeve adds up to the book;
the level gap is the panel's means; the report says nothing is nan; the tear
sheet fits one page.
"""

import json
import re
import numpy as np
import pandas as pd
import pytest
from policypath import regimes
from policypath.backtest import metrics
from policypath.report import attribution, tearsheet
from test_portfolio import book_world


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    return attribution.build(book_world(tmp_path_factory.mktemp("world")))


def test_the_split_adds_up(built):
    st = built.portfolio.setup
    attribution.check(built.headline, built.parts)
    for n, f in built.parts.items():
        if st.inputs[n].sleeve.kind == "outright":
            np.testing.assert_allclose(f["level"], f["gross"], atol=1e-9)
    shares = [r["variance_share"] for r in built.numbers["components"] if r["group"] != "book"]
    assert sum(shares) == pytest.approx(1.0)
    book = next(r for r in built.numbers["components"] if r["group"] == "book")
    assert book["carry_roll"] + book["rate"] == pytest.approx(book["gross"])


def test_the_level_verdict_follows_the_variance_share(built):
    lv = built.numbers["level"]
    assert lv["dominates"] == (lv["variance_share"] >= attribution.LEVEL_DOMINATES)
    words = attribution.level_words(built.numbers)
    assert ("The level dominates" in words) == lv["dominates"]


def test_the_regime_table_partitions_the_counted_sessions(built):
    rows = {r["regime"]: r for r in built.numbers["regimes"]["book"]}
    single = [r for g, r in rows.items() if len(g) == 1]
    kept = int(built.headline.run.daily["kept"].sum())
    assert sum(r["sessions"] for r in single) <= kept
    for pair in (("early_hiking", "late_hiking"), ("hold_after_hikes", "hold_after_cuts")):
        assert rows[pair]["sessions"] == rows[(pair[0],)]["sessions"] + rows[(pair[1],)]["sessions"]
    for v in built.numbers["regimes"]["sleeves"].values():
        assert sum(r["sessions"] for r in v["rows"]) <= kept


def test_the_elb_split_has_no_state_pnl_where_the_treatment_holds_none(built):
    e = built.numbers["elb"]
    assert set(e) == {"flat", "exclude", "hold"}
    assert e["flat"]["state_sessions"] == 0 and e["exclude"]["in_state"] == pytest.approx(0.0, abs=1e-9)
    assert e["flat"]["closes"] is not None and e["hold"]["closes"] is None
    for x in e.values():
        assert x["in_state"] + x["outside"] == pytest.approx(x["net_year"] * x["years"], rel=1e-9, abs=1e-9)


def test_the_carry_benchmark_trades_the_sign_of_carry_and_roll(built):
    cs, _ = built.carry
    for i in cs.inputs.values():
        z = i.sleeve.component["z"].dropna()
        assert set(np.unique(z)) <= {-1.0, 1.0}
    assert built.numbers["carry"]["words"] in [w for _, w in attribution.CARRY_WORDS]


def test_ex_2022_is_evaluate_without_the_window(built):
    book = built.portfolio.setup.book
    cap = book["book"]["capital"]
    p = built.numbers["performance"][0]
    assert p["ex_2022"] == metrics.evaluate(built.headline, metrics.window("ex_2022", book), cap)
    assert p["ex_2022_23"]["sessions"] <= p["ex_2022"]["sessions"] <= p["sessions"]


def test_the_costs_by_cause_are_the_cost_charged_and_the_turnover_identity_holds(built):
    st, n = built.portfolio.setup, built.numbers
    attribution.check_costs(st, built.headline)
    capital = st.book["book"]["capital"]
    rows = {r["name"]: r for r in n["costs"]}
    for r in rows.values():
        assert sum(r["cost"].values()) == pytest.approx(r["cost_year"])
        assert sum(r["turns"].values()) == pytest.approx(r["turns_year"])
        assert r["turns_year"] * r["mean_dv01"] * r["one_way_bp"] / capital * 100 == pytest.approx(r["cost_year"])
    book = n["performance"][0]
    assert rows["book"]["turns_year"] == pytest.approx(book["turns_year"])
    comp = next(r for r in n["components"] if r["group"] == "book")
    assert rows["book"]["cost_year"] == pytest.approx(comp["cost"])


def test_the_information_identity_solves_for_the_rate_needed():
    rng = np.random.default_rng(0)
    q, x = rng.normal(1.0, 2.0, 500), rng.normal(-0.1, 1.0, 500)
    rho, _ = attribution.information(q, x, 0.0)
    assert rho == pytest.approx(np.corrcoef(q, x)[0, 1])
    for need in (-40.0, 0.0, 75.0):
        _, rho_star = attribution.information(q, x, need)
        assert len(q) * (rho_star * q.std() * x.std() + q.mean() * x.mean()) == pytest.approx(need)
    _, at_realised = attribution.information(q, x, float(q @ x))
    assert at_realised == pytest.approx(rho)
    assert np.isnan(attribution.information(np.ones(5), x[:5], 1.0)[1])


def test_carry_by_sleeve_adds_up_to_the_book_and_reads_the_signs(built):
    cs, comp = built.numbers["carry_sleeves"], {r["group"]: r for r in built.numbers["components"]}
    for key in ("carry_roll", "rate", "cost"):
        assert cs["book"][key] == pytest.approx(comp["book"][key])
    for r in cs["sleeves"]:
        assert r["need"] == pytest.approx(r["cost"] - r["carry_roll"])
        assert attribution.reading(r).startswith("right" if r["rate"] > 0 else "wrong")
        assert attribution.reading(r).endswith("collecting carry" if r["carry_roll"] > 0 else "bleeding")
        assert pd.isna(r["bleeding"]) or 0.0 <= r["bleeding"] <= 1.0
    for s, (entries, skipped) in built.numbers["filter"].items():
        assert 0 <= skipped <= entries


def test_the_level_gap_rows_are_the_panel_means(built):
    w, n = built.world, built.numbers
    for c, v in n["level_gap"].items():
        gap = w.panel(c, "signal").pivot(index="session", columns="k", values="gap_bp")
        state = regimes.states(w.panel(c, "model")).reindex(gap.index).map(attribution.COLLAPSED)
        assert sum(r["sessions"] for r in v["regimes"]) == int(state.notna().sum())
        for r in v["regimes"]:
            for k in v["k"]:
                assert r["mean"][k] == pytest.approx(gap.loc[state == r["regime"], k].mean(), nan_ok=True)
        off = gap.loc[state != "ELB", v["k"][1]]
        for y, m in v["years"].items():
            assert m == pytest.approx(off[off.index.year == y].mean())


def test_the_report_and_json_say_no_nan_and_the_tear_sheet_fits_one_page(built, tmp_path):
    md = attribution.markdown(built)
    for head in ["## The answer", "## Performance", "## Where the costs go", "## By component", "### The level factor",
                 "## Carry against rate", "### By sleeve", "## By regime", "## The ELB", "## The IC",
                 "## The level the z removes", "## The current signal"]:
        assert head in md
    assert "Do not edit by hand" in md and not re.search(r"\bnan\b", md)
    numbers = json.loads(json.dumps(attribution.results(built)))
    assert set(numbers["elb"]) == {"flat", "exclude", "hold"}
    assert len(numbers["current"]) == len(built.parts)
    path = tearsheet.write(built, tmp_path, generated="2026-09-29")
    assert path.name == f"tearsheet_{built.numbers['days'][1]:%Y-%m-%d}.pdf" and path.stat().st_size > 0
    assert len(re.findall(r"[.!?](?:\s|$)", tearsheet.framework(built))) == 3
    for theme in ("light", "dark"):
        attribution.figure(built, tmp_path / f"a_{theme}.png", theme)
