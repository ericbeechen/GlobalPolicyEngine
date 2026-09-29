"""The portfolio report end to end on the synthetic two-currency world: the build's checks, its outputs, its words.

The world of `test_expression.py` with week 8's ELB spells (`test_report_costs.py`),
a book that decides after 60 sessions of unit P&L, and a drawdown control
tight enough to fire. Checks: the build passes its own invariants and writes
every output; every Sharpe in the books table carries its SE; "anecdotal" is
said exactly when the drawdown control fired twice or fewer; shrinkage cannot
move inverse-vol's relative weights; the RV-only book holds no outright; the
headline's curve crosses zero at its closed-form breakeven, and the curves'
zero-cost point is the gross Sharpe; the book counts only sessions a sleeve's
ELB treatment counts.
"""

import json
import re
import numpy as np
import pytest
from policypath.report import costs as sleeves
from policypath.report import portfolio as report
from policypath.strategy import portfolio, positions
from test_portfolio import book_world


@pytest.fixture(scope="module")
def written(tmp_path_factory):
    built = report.build(book_world(tmp_path_factory.mktemp("world")))
    root = tmp_path_factory.mktemp("repo")
    paths = {"report": root / "portfolio.md", "json": root / "portfolio.json", "figure": root / "book.png"}
    paths["report"].write_text(report.markdown(built))
    paths["json"].write_text(json.dumps(report.results(built)))
    report.figure(built, paths["figure"], "dark")
    return built, paths


def test_the_build_writes_every_output(written):
    built, paths = written
    for p in paths.values():
        assert p.exists() and p.stat().st_size > 0
    md = paths["report"].read_text()
    for head in ["## The answer", "## How the book is built", "## The books side by side", "## Cost sensitivity",
                 "## The covariance", "### Shrinkage on and off", "### The non-synchronous term",
                 "## The drawdown control", "## Turnover", "## Realized and ex-ante vol", "## Net DV01 by currency",
                 "## What carries it", "## The book calendar", "## Checks"]:
        assert head in md
    assert "Do not edit by hand" in md and not re.search(r"\bnan\b", md)
    assert "the headline, proposed" in md and "Schafer and Strimmer's target D" in md
    numbers = json.loads(paths["json"].read_text())
    assert numbers["spec"]["construction"] == built.setup.book["portfolio"]["headline"]
    assert len(numbers["books"]) == len(built.books) and "inverse_vol_shrunk" in numbers["books"]


def test_every_sharpe_in_the_books_table_carries_its_se(written):
    _, paths = written
    md = paths["report"].read_text()
    part = md[md.index("## The books side by side"):md.index("## Cost sensitivity")]
    cells = [c.strip() for line in part.splitlines() if line.startswith("| ") and "---" not in line
             for c in line.split("|")[2:4]]
    assert cells[2:] and all("(" in c for c in cells[2:])


def test_anecdotal_is_said_when_the_overlay_fired_twice_or_fewer(written):
    built, paths = written
    md = paths["report"].read_text()
    h = report.headline(built.setup.book)
    n = len(built.summaries[report.Choice(h.construction, overlay=True)]["episodes"])
    assert n >= 1                                                  # the tight trigger fires in the world
    assert ("anecdotal" in md) == (n <= 2)


def test_shrinkage_moves_inverse_vols_scale_not_its_relative_weights(written):
    built, _ = written
    a = built.books[report.Choice("inverse_vol")].targets.q
    b = built.books[report.Choice("inverse_vol", shrink=False)].targets.q
    on = (a != 0).sum(axis=1) > 1
    ra, rb = a[on].div(a[on].abs().sum(axis=1), axis=0), b[on].div(b[on].abs().sum(axis=1), axis=0)
    capped = built.books[report.Choice("inverse_vol")].targets.capped[on] | \
        built.books[report.Choice("inverse_vol", shrink=False)].targets.capped[on]
    assert np.allclose(ra[~capped], rb[~capped], atol=1e-12)


def test_the_rv_only_book_holds_no_outright(written):
    built, _ = written
    rv = built.books[report.Choice(report.headline(built.setup.book).construction, sleeves=built.rv)]
    assert set(rv.decided.columns) == set(built.rv)
    assert all(built.setup.inputs[n].sleeve.kind != "outright" for n in rv.decided.columns)


def test_the_headline_curve_starts_at_its_gross_and_crosses_zero_at_its_breakeven(written):
    built, _ = written
    for ch, b in built.books.items():
        s = built.summaries[ch]
        assert sleeves.curve(b.run, [0.0])[0] == pytest.approx(s["gross_sr"], abs=1e-9)
        if s["cstar_bp"] is not None and np.isfinite(s["cstar_bp"]):
            assert abs(sleeves.curve(b.run, [s["cstar_bp"]])[0]) < 1e-9


def test_the_book_counts_only_sessions_a_sleeves_treatment_counts(written):
    built, _ = written
    b = built.books[report.headline(built.setup.book)]
    d = b.run.daily
    assert not d.loc[d.index < b.run.start, "kept"].any()
    left = d.loc[~d["kept"] & (d.index >= b.run.start), report.FLOWS].abs().to_numpy().sum()
    assert left == 0.0
    st = built.setup
    want = np.zeros(len(st.days), dtype=bool)
    for i in st.inputs.values():
        kept, *_ = positions.samples(i.state, st.spec.treatment, i.sleeve.lag)
        want |= portfolio.to_book((kept & (i.sleeve.sessions >= b.run.start)).astype(int), st.days).to_numpy() > 0
    assert np.array_equal(d["kept"].to_numpy(), want & (st.days >= b.run.start))
