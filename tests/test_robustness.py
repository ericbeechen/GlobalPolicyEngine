"""The robustness grid: a row changes the signal and nothing else, and its cells are over the sessions every row counts.

On the synthetic two-currency world (`test_portfolio.book_world`: week 8's ELB
spells, a book that trades early) the grid carries six rows: one that changes
nothing; one that moves the sterling signal as a re-fitted curve would (NSS-like:
the market path moves by under a basis point); one whose dollar z window is
longer, so its z starts later; one whose sterling model spends longer in the
ELB state; one with estimated coefficients whose first estimates fall on
different days in the two currencies; and one that trades the raw slope (a
book signal row). Checks: under the NSS-like row the unit P&L, the legs and
the covariance are the baseline's bit for bit, every row's book is sized on
the baseline's covariance, and the dollar sleeves' cells do not move; a book
signal row reaches the curve sleeves and a currency row only its currency's;
the common sample is the intersection of the rows' eligible sessions, and
every cell is over it; the row that changes nothing reproduces every baseline
cell exactly; a row cannot build a leg; side changes are over the sessions
both have a z; the IC is on the rate component, outside every row's ELB
state; the estimated window starts at the latest first estimate a book or
sleeve trades; a copied grid reports the same; the headline is checked
against the published one. On the committed fixtures: the rebuild with no
override is the chain's own output, a row's rebuild equals
`regress.fixture_outputs` under the same override (a rule and a curve fit),
and an r* override replaces the whole block. And the pieces: eligibility, the
IC, its forward sum and its non-overlapping check, the paired SE and its
Newey-West version, the signal moves, the first estimate, the store's
fingerprint.
"""

import copy
import json
import pickle
import numpy as np
import pandas as pd
import pytest
from policypath import config, fixtures, regress
from policypath.backtest import metrics
from policypath.report import robustness as report
from policypath.signal import gap
from policypath.strategy import expression, robustness
from policypath.strategy.robustness import BASELINE
from test_portfolio import book_world
from test_report_costs import SPELLS, model_of

T = pd.Timestamp
EXTRA_SPELL = ("2022-01-03", "2022-03-31")     # the sterling model's extra time in the ELB state under "lower"
CHOICES = [
    {"name": "nothing", "chosen": "as built",
     "rows": [{"key": "same", "label": "the same", "every": {"signal": {"window": "730D", "min_periods": 250}}}]},
    {"name": "curve fit", "chosen": "log-linear",
     "rows": [{"key": "refit", "label": "refit", "currencies": {"GBP": {"market": {"curve": {"method": "nss"}}}}}]},
    {"name": "z window", "chosen": "730D",
     "rows": [{"key": "longer", "label": "800D",
               "currencies": {"USD": {"signal": {"window": "800D", "min_periods": 274}}}}]},
    {"name": "r*", "chosen": "-1.6",
     "rows": [{"key": "lower", "label": "-2.6", "currencies": {"GBP": {"rule": {"rstar": {"constant": -2.6}}}}}]},
    {"name": "coefficients", "chosen": "imposed",
     "rows": [{"key": "est", "label": "estimated",
               "every": {"rule": {"estimate": {"prior_quarters": 8, "drop_cuts_to_floor": True}}}}]},
    {"name": "slope", "chosen": "orthogonal",
     "rows": [{"key": "raw", "label": "raw", "book": {"signal": {"slope_orthogonal": False}}}]},
]
FIRST = {"USD": 100, "GBP": 200}               # the session each currency's estimated coefficients first rest on data


def rows_signals(w):
    """Each row's (model, signal) per currency it changes, made by hand as the chain would move them."""
    panels = {c: {n: w.panel(c, n) for n in ("model", "signal", "paths")} for c in ("USD", "GBP")}
    same = {c: (p["model"], p["signal"].copy()) for c, p in panels.items()}
    rng = np.random.default_rng(5)
    paths = panels["GBP"]["paths"]
    moved = paths.assign(market=paths["market"] + rng.normal(0, 0.004, len(paths)))
    refit = {"GBP": robustness.rebuild("GBP", config.currency("GBP"), {**panels["GBP"], "paths": moved},
                                       model=False)}
    longer_cfg = regress.merge(config.currency("USD"), CHOICES[2]["rows"][0]["currencies"]["USD"])
    longer = {"USD": robustness.rebuild("USD", longer_cfg, panels["USD"], model=False)}
    model = panels["GBP"]["model"]
    lower = {"GBP": (model_of(pd.DatetimeIndex(model["session"]), [*SPELLS["GBP"], EXTRA_SPELL]).assign(rstar=-2.6),
                     panels["GBP"]["signal"])}
    est = {}
    for c, p in panels.items():
        at = np.arange(len(p["model"])) - FIRST[c]
        est[c] = (p["model"].assign(coef_quarters=np.where(at >= 0, 1 + at // 63, 0)), p["signal"])
    return {"same": same, "refit": refit, "longer": longer, "lower": lower, "est": est, "raw": {}}


@pytest.fixture(scope="module")
def grid(tmp_path_factory):
    w = book_world(tmp_path_factory.mktemp("world"))
    w.book["robustness"] = {"ic_horizon": 21, "choices": copy.deepcopy(CHOICES)}
    for c, r in (("USD", 1.1), ("GBP", -1.6)):
        w._panels[c]["model"] = w.panel(c, "model").assign(rstar=r)
    sized, run_book = [], report.books.run_book
    with pytest.MonkeyPatch.context() as mp:            # the Risk every row's book is sized on
        mp.setattr(report.books, "run_book", lambda st, rk, *a, **k: sized.append(rk) or run_book(st, rk, *a, **k))
        g = report.build(w, signals=rows_signals(w))
    return w, g, report.summary(g), sized


# ---- a row changes the signal and nothing else ---------------------------------------

def test_under_an_nss_like_signal_change_the_unit_pnl_legs_and_covariance_are_the_baselines(grid):
    _, g, s, _ = grid
    st0, st = g.setups[BASELINE.key], g.setups["refit"]
    assert st.units.equals(st0.units)
    for n, i in st0.inputs.items():
        j = st.inputs[n]
        assert j.u is i.u and j.sigma is i.sigma and j.costs is i.costs
        assert all(a[2] is b[2] for a, b in zip(i.sleeve.legs, j.sleeve.legs))
    z0, z1 = st0.inputs["GBP outright"].sleeve.component["z"], st.inputs["GBP outright"].sleeve.component["z"]
    assert not z0.equals(z1) and z0.corr(z1) > 0.9                 # the signal moved, a little
    assert s["net"]["refit"]["USD outright"] == s["net"][BASELINE.key]["USD outright"]
    assert s["net"]["refit"]["USD 2s10s"] == s["net"][BASELINE.key]["USD 2s10s"]
    assert not g.books["refit"].decided.equals(g.books[BASELINE.key].decided)       # and the book traded on it


def test_every_rows_book_is_sized_on_the_baselines_covariance(grid):
    _, g, _, sized = grid
    assert len(sized) == len(g.rows) and all(rk is g.risk for rk in sized)


def test_a_book_signal_row_reaches_every_sleeve_and_a_currency_row_only_its_currencys(grid):
    _, g, _, _ = grid
    st0 = g.setups[BASELINE.key]
    for key, reached in (("raw", lambda i: True), ("lower", lambda i: "GBP" in i.sleeve.ccys)):
        st = g.setups[key]
        for n, i in st0.inputs.items():
            assert (st.inputs[n] is not i) == reached(i)                  # rebuilt on the row's view, or the baseline's
    for n, i in st0.inputs.items():                                     # and under the raw slope only a curve's z moves
        moved = not g.setups["raw"].inputs[n].sleeve.component["z"].equals(i.sleeve.component["z"])
        assert moved == (i.sleeve.kind == "curve")


def test_a_row_cannot_build_a_leg_of_its_own(grid):
    w, g, _, _ = grid
    leg = next(iter(g.setups[BASELINE.key].inputs.values())).sleeve.legs[0][2]
    view = robustness.View(w, {}, build=False)
    with pytest.raises(RuntimeError, match="has not built"):
        view.leg("USD", "outright", leg.frame.index, k=4)
    legs = {("USD", "outright", (("k", 4),)): [(leg.frame.index, leg)]}
    assert robustness.View(w, legs).leg("USD", "outright", leg.frame.index, k=4) is leg
    with pytest.raises(RuntimeError, match="has not built"):
        robustness.View(w, legs).leg("USD", "outright", leg.frame.index[1:], k=4)


# ---- the samples and the cells -------------------------------------------------------

def test_the_common_sample_is_the_intersection_and_every_cell_is_over_it(grid):
    _, g, s, _ = grid
    for n, m in g.common.items():
        want = np.logical_and.reduce([g.eligible[r.key][n].to_numpy() for r in g.rows])
        assert np.array_equal(m.to_numpy(), want)
        assert s["net"][BASELINE.key][n][2] == int(m.sum())
    base = g.eligible[BASELINE.key]
    first = lambda e: e[e].index.min()
    assert first(g.eligible["longer"]["USD outright"]) > first(base["USD outright"])        # its z starts later
    spell = slice(*EXTRA_SPELL)
    inside = lambda e: e[spell].iloc[2:]          # flat keeps the P&L into the state and the close into it (lag 1)
    assert inside(base["GBP outright"]).all() and not inside(g.eligible["lower"]["GBP outright"]).any()
    assert not inside(g.common["GBP outright"]).any()
    assert (base["USD outright"] & ~g.common["USD outright"]).any()        # the later z costs the baseline sessions
    assert not (g.common["book"] & ~base["book"]).any()                    # the book counts a session any sleeve has
    x = g.books["lower"].run.daily
    assert s["net"]["lower"]["book"] == robustness.cell(x["gross"] - x["cost"], g.common["book"], g.books["lower"].run.periods)


def test_a_row_that_changes_nothing_reproduces_every_baseline_cell_exactly(grid):
    _, g, s, _ = grid
    b = BASELINE.key
    for part in ("net", "gross", "own", "own_gross", "ic"):
        assert s[part]["same"] == s[part][b]
    assert s["paired"]["same"]["book"] == 0.0
    assert g.books["same"].run.daily.equals(g.books[b].run.daily)
    assert all(g.runs["same"][n].daily.equals(r.daily) for n, r in g.runs[b].items())


def test_side_changes_are_over_the_sessions_both_have_a_z_and_again_over_the_common_sample(grid):
    _, g, _, _ = grid
    st0, st = g.setups[BASELINE.key], g.setups["longer"]
    n = "USD outright"
    z0, z1 = (x.inputs[n].sleeve.component["z"] for x in (st0, st))
    assert (z0.notna() & z1.isna()).any()                                  # the longer window's z starts later
    both = z0.notna() & z1.notna()
    differ = st0.side[n] != st.side[n]
    assert g.sides["longer"][n] == (float(differ[both].mean()), float(differ[both & g.common[n]].mean()))
    assert g.sides["longer"]["GBP outright"] == (0.0, 0.0)


def test_the_ic_is_on_the_rate_component_outside_every_rows_elb_state(grid):
    _, g, s, _ = grid
    for n, i in g.setups[BASELINE.key].inputs.items():
        m = g.ic_common[n]
        assert not (m & i.sleeve.component["z"].isna()).any()
        for r in g.rows:
            assert not (m & g.setups[r.key].inputs[n].state.reindex(m.index, fill_value=False)).any()
    lower = g.setups["lower"].inputs["GBP outright"]
    state = lower.state.reindex(lower.sleeve.sessions, fill_value=False)
    assert (state & lower.sleeve.component["z"].notna()).any()             # the state has a z: the sample drops it
    i = g.setups[BASELINE.key].inputs["GBP outright"]
    by = i.u.groupby("session")
    rate, total = (by[c].sum(min_count=1).reindex(i.sleeve.sessions) for c in ("rate", "total"))
    want = robustness.ic(i.sleeve.component["z"], rate, 21, i.sleeve.lag, g.ic_common["GBP outright"])
    assert s["ic"][BASELINE.key]["GBP outright"] == want
    assert want != robustness.ic(i.sleeve.component["z"], total, 21, i.sleeve.lag, g.ic_common["GBP outright"])
    assert s["ic_offsets"][BASELINE.key]["GBP outright"] == robustness.ic_offsets(
        i.sleeve.component["z"], rate, 21, i.sleeve.lag, g.ic_common["GBP outright"])


def test_the_estimated_window_starts_at_the_latest_first_estimate_a_book_or_sleeve_trades(grid):
    w, g, s, _ = grid
    first = {c: T(w.panel(c, "model")["session"].iloc[k]) for c, k in FIRST.items()}
    assert g.first["est"] == first and g.first[BASELINE.key] == {"USD": None, "GBP": None}
    e = s["estimated"]
    assert e["keys"] == ["est"] and first["USD"] < first["GBP"]
    for n, i in g.setups[BASELINE.key].inputs.items():
        assert e["start"][n] == max(first[c] for c in i.sleeve.ccys)
    assert e["start"]["book"] == first["GBP"]
    for n, start in e["start"].items():
        assert e["cells"][BASELINE.key][n][2] == int((g.common[n] & (g.common[n].index >= start)).sum())


def test_a_copied_grid_reports_the_same(grid):
    _, g, s, _ = grid
    twin = pickle.loads(pickle.dumps(g))
    assert twin.rows[0] is not BASELINE and twin.rows[0] == BASELINE
    assert report.markdown(twin, report.summary(twin)) == report.markdown(g, s)


def test_the_headline_is_checked_against_the_published_book(grid):
    _, g, s, _ = grid
    own = s["own"][BASELINE.key]["book"]
    ok = {"net_sr": float(f"{own[0]:.6g}"), "sessions": own[2]}
    assert "portfolio.json has" in report._own_words(g, report.summary(g, ok))
    assert "portfolio.json" not in report._own_words(g, s)
    for bad in ({**ok, "sessions": own[2] + 1}, {**ok, "net_sr": ok["net_sr"] + 1e-4}):
        with pytest.raises(RuntimeError, match="build_portfolio"):
            report.summary(g, bad)


def test_the_elb_state_moves_with_the_rows_model(grid):
    _, g, _, _ = grid
    lower, base = (g.setups[k].inputs["GBP outright"].state[slice(*EXTRA_SPELL)] for k in ("lower", BASELINE.key))
    assert lower.all() and not base.any()
    assert g.elb["lower"]["GBP"] > g.elb[BASELINE.key]["GBP"]


def test_the_report_is_generated_whole(grid, tmp_path):
    _, g, s, _ = grid
    paths = report.write(g, tmp_path, s)
    md = paths["report"].read_text()
    for head in ["## The answer", "## One line per choice", "## The grid: net Sharpe over the common sample",
                 "## Gross Sharpe", "## Each row on its own sample", "## IC(21)", "## What each row does",
                 "### The traded side", "## Checks"]:
        assert head in md
    assert "One choice at a time" in md and "interactions between choices are not run" in md
    assert "SE(SR) there is" in md and "Do not edit by hand" in md and "nan" not in md.lower().split()
    assert "The note on r\\*" in md                                  # the lower row is a constant offset
    assert "Newey-West" in md and "The non-overlapping check" in md and "inf" not in md.split()
    numbers = json.loads(paths["json"].read_text())
    assert set(numbers["rows"]) == {r.key for r in g.rows} and numbers["spec"]["design"] == "one at a time"
    assert all(paths[f"figure_{t}"].stat().st_size > 0 for t in ("light", "dark"))


# ---- the rebuild is the chain's own --------------------------------------------------

def fixture_base(ccy, root):
    """The fixture chain's panels as `data/panel/` holds them (through parquet), a cache laid out under `root`, and
    the currency block the fixtures run on."""
    cfg = regress.merge(config.currency(ccy), fixtures.spec(ccy).get("overrides") or {})
    fixtures.write(ccy, root)
    return {n: regress._roundtrip(f) for n, f in regress.fixture_outputs(ccy).items()}, cfg


@pytest.mark.parametrize("ccy, over", [
    ("USD", {"rule": {"estimate": {"prior_quarters": 8, "drop_cuts_to_floor": True}}}),
    ("USD", {"rule": {"rstar": {"constant": 1.1}}}),
    ("GBP", {"market": {"curve": {"method": "nss"}}}),
    ("GBP", {"signal": {"window": "365D", "min_periods": 125}}),
])
def test_a_rows_rebuild_is_what_the_chain_gives_under_the_same_override(ccy, over, tmp_path):
    base, cfg = fixture_base(ccy, tmp_path)
    assert robustness.rebuilt_equals(ccy, cfg, base, tmp_path) is None
    got = dict(zip(("model", "signal"), robustness.variant(ccy, cfg, over, base, tmp_path)))
    want = regress.fixture_outputs(ccy, overrides=over)
    assert regress.compare(got, {n: regress._roundtrip(want[n]) for n in got}) == {}


def test_an_rstar_override_replaces_the_whole_block(tmp_path):
    usd = config.currency("USD")
    assert {"source", "series"} <= set(usd["rule"]["rstar"])
    assert robustness.block("USD", usd, {"rule": {"rstar": {"constant": 1.1}}})["rule"]["rstar"] == {"constant": 1.1}
    for row in robustness.rows(config.strategy()):
        for ccy, over in row.currencies.items():
            if "rstar" in over.get("rule", {}):
                assert robustness.block(ccy, config.currency(ccy), over)["rule"]["rstar"] == over["rule"]["rstar"]
    base, cfg = fixture_base("USD", tmp_path)
    model, _ = robustness.variant("USD", cfg, {"rule": {"rstar": {"constant": 1.1}}}, base, tmp_path)
    assert (model["rstar"] == 1.1).all() and model["sep_date"].isna().all()
    with pytest.raises(config.ConfigError, match="market.curve.method"):
        robustness.block("GBP", config.currency("GBP"), {"market": {"curve": {"method": "NSS"}}})


def test_the_rows_of_the_book_are_the_configs_with_the_baseline_first():
    book = config.strategy()
    grid = robustness.rows(book)
    assert grid[0] is BASELINE and not BASELINE.currencies and not BASELINE.book
    assert [r.key for r in grid[1:]] == [r["key"] for c in book["robustness"]["choices"] for r in c["rows"]]
    every = next(r for c in book["robustness"]["choices"] for r in c["rows"] if "every" in r)
    row = next(r for r in grid if r.key == every["key"])
    assert set(row.currencies) == set(robustness.book_currencies(book))


# ---- the pieces ----------------------------------------------------------------------

def test_a_session_is_eligible_where_the_treatment_counts_it_and_a_z_stands_behind_it():
    days = pd.bdate_range("2024-01-01", periods=8)
    z = pd.Series([np.nan, np.nan, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], index=days)
    state = pd.Series([False] * 5 + [True] * 3, index=days)
    got = robustness.eligible(z, state, "flat", 1)
    # a z behind the position put on (lag 1) from the fourth session; flat's state drops a session whose held and
    # put-on positions were both decided in it (the eighth: decided at the sixth and seventh)
    assert got.tolist() == [False, False, False, True, True, True, True, False]
    assert robustness.common([got, pd.Series(True, index=days)]).equals(got)
    with pytest.raises(ValueError):
        robustness.common([got, got.iloc[1:]])


def test_the_forward_sum_starts_after_the_lag_and_the_ic_is_the_rank_correlation():
    days = pd.bdate_range("2024-01-01", periods=40)
    x = pd.Series(np.arange(40.0), index=days)
    fwd = metrics.forward(x, 3, 1)
    assert fwd.iloc[0] == 2 + 3 + 4 and fwd.iloc[35] == 37 + 38 + 39 and fwd.iloc[36:].isna().all()
    rng = np.random.default_rng(1)
    z = pd.Series(rng.normal(size=40), index=days)
    y = pd.Series(np.exp(z) + rng.normal(0, 0.5, 40), index=days)
    ic, t, n = metrics.ic(z, y, 3)
    assert n == 40 and ic == pytest.approx(z.rank().corr(y.rank()), abs=1e-12) and t > 0
    assert metrics.ic(z, -z, 3)[0] == pytest.approx(-1.0)
    assert np.isnan(metrics.ic(z.iloc[:2], y.iloc[:2], 3)[0])
    each = [z.iloc[o::3].rank().corr(y.iloc[o::3].rank()) for o in range(3)]
    mean, lo, hi, fewest = metrics.ic_offsets(z, y, 3)
    assert (mean, lo, hi) == pytest.approx((np.mean(each), min(each), max(each))) and fewest == 13
    p = metrics.rank_products(z, y)
    assert p.mean() == pytest.approx(ic) and len(p) == 40


def test_the_paired_se_is_the_differences_sd_over_the_baselines_per_root_year():
    days = pd.bdate_range("2024-01-01", periods=500)
    rng = np.random.default_rng(2)
    x0 = pd.Series(rng.normal(size=500), index=days)
    x = x0 + pd.Series(rng.normal(0, 0.1, 500), index=days)
    mask = pd.Series(True, index=days)
    assert robustness.paired_se(x0, x0, mask, 250) == 0.0
    want = (x - x0).std() / x0.std() * np.sqrt(250 / 500)
    assert robustness.paired_se(x, x0, mask, 250) == pytest.approx(want)
    e = rng.normal(0, 0.1, 500)
    x = x0 + pd.Series(e + np.r_[0.0, e[:-1]] + np.r_[0.0, 0.0, e[:-2]], index=days)      # an MA(2) difference
    d = ((x - x0) / x0.std()).to_numpy()
    d = d - d.mean()
    lrv = d @ d / 500 + 2 * sum((1 - k / 4) * (d[k:] @ d[:-k]) / 500 for k in (1, 2, 3))
    assert metrics.long_run_var((x - x0) / x0.std(), 3) == pytest.approx(lrv)
    assert robustness.paired_se(x, x0, mask, 250, lags=3) == pytest.approx(np.sqrt(lrv * 250 / 500))
    assert robustness.paired_se(x, x0, mask, 250, lags=3) > 1.3 * robustness.paired_se(x, x0, mask, 250)


def level_signal(days, values, spec):
    """A signal panel at k = 4 with the chain's own z (`signal.gap.zscore`)."""
    z, mean, sd = gap.zscore(pd.DataFrame({4: values}, index=days), spec)
    return pd.DataFrame({"session": days, "k": 4, "gap_bp": values, "z": z[4].to_numpy(),
                         "window_mean_bp": mean[4].to_numpy(), "window_sd_bp": sd[4].to_numpy()})


def test_a_constant_rstar_shifts_the_gap_off_the_floor_and_leaves_the_z_where_no_floor_is_in_its_window():
    days = pd.bdate_range("2023-01-02", periods=300)
    spec = {"window": "60D", "min_periods": 20, "sd_floor_bp": 0.5}
    floored = pd.Series(False, index=days)
    floored.iloc[100:130] = True
    base = np.random.default_rng(3).normal(0, 10, 300)
    row = base + np.where(floored, 0.0, 10.0)                  # r* 1pp lower: +10bp where the goal is off the floor
    model = pd.DataFrame({"session": days, "rstar": -1.6, "at_elb": floored.to_numpy()})
    on = pd.Series(np.arange(300) >= 50, index=days)
    m = robustness.moves(level_signal(days, base, spec), level_signal(days, row, spec), 4, model,
                         model.assign(rstar=-2.6), on, spec["window"])
    assert m["mean_change_bp"] == pytest.approx(10 * 270 / 300) and m["latest_change_bp"] == pytest.approx(10)
    assert m["latest_row_bp"] == pytest.approx(m["latest_base_bp"] + 10) and m["bp_per_pp"] == pytest.approx(-10)
    assert m["rstar_offset_mean"] == pytest.approx(-1.0) and m["floored_base"] == m["floored_row"] == 0.1
    z = level_signal(days, base, spec)["z"].notna().to_numpy()
    assert m["traded_sessions"] == int((on.to_numpy() & z).sum())
    assert m["floored_base_traded"] == pytest.approx(30 / m["traded_sessions"])
    near = floored.astype(float).rolling("60D", closed="both").max().astype(bool).to_numpy()
    assert m["floor_free"] == int((on.to_numpy() & z & ~near).sum()) > 0
    assert m["floor_free_max_dz"] < 1e-9 and m["corr_z"] < 1 - 1e-6       # the z moves, but not away from the floor


def test_the_first_estimate_is_the_first_session_resting_on_a_quarter():
    days = pd.bdate_range("2024-01-01", periods=5)
    model = pd.DataFrame({"session": days, "coef_quarters": [0, 0, 1, 1, 2]})
    assert robustness.first_estimate(model) == days[2]
    assert robustness.first_estimate(model.assign(coef_quarters=0)) is None
    assert robustness.first_estimate(model.drop(columns="coef_quarters")) is None


def test_the_store_keeps_a_row_under_a_fingerprint_of_what_it_read(tmp_path):
    panels, cache_root = tmp_path / "panel", tmp_path / "cache"
    (cache_root / "fred" / "USD").mkdir(parents=True)
    panels.mkdir()
    for name in robustness.INPUTS:
        pd.DataFrame({"x": [1.0]}).to_parquet(panels / f"USD_{name}.parquet")
    pd.DataFrame({"v": [1.0]}).to_parquet(cache_root / "fred" / "USD" / "S.parquet")
    cfg = {"rule": {"a": 1}}
    key = robustness.fingerprint("USD", cfg, panels, cache_root)
    assert robustness.fingerprint("USD", cfg, panels, cache_root) == key
    assert robustness.fingerprint("USD", {"rule": {"a": 2}}, panels, cache_root) != key
    pd.DataFrame({"x": [2.0]}).to_parquet(panels / "USD_macro.parquet")
    assert robustness.fingerprint("USD", cfg, panels, cache_root) != key
    store = robustness.Store(tmp_path / "store")
    frames = (pd.DataFrame({"session": [T("2024-01-02")], "r0": [1.0]}), pd.DataFrame({"z": [0.5]}))
    got = store.put("USD", key, frames)
    assert all(a.equals(b) for a, b in zip(got, frames)) and store.get("USD", "other") is None
    store.mark("USD", "check")
    fresh = robustness.Store(tmp_path / "store")
    assert fresh.get("USD", key) is not None and fresh.prune() == 1           # the check was not asked for
    assert robustness.Store(tmp_path / "store").prune() == 2                  # nor, now, the row
