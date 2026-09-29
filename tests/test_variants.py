"""The model-side robustness variants: each is a config override that reaches the stage it changes, and none moves the baseline.

A variant is only worth reporting if the override actually gets through: a key
the chain never reads leaves the baseline in place and a "robust" row that is
the baseline twice. So the plumbing tests run the whole chain on the committed
fixtures (`regress.fixture_outputs`) with and without each override. The
estimated rule and converge-to-target conditioning are checked on synthetic
inputs, where the right answer is known.
"""

import numpy as np
import pandas as pd
import pytest
from policypath import config, fixtures, regimes, regress
from policypath.curves import nss
from policypath.model import estimate
from policypath.model.path import converge_goals, model_path
from policypath.model.reaction import inertial_path, notional
from policypath.model.rstar import rstar

T = pd.Timestamp
USD_RULE = config.currency("USD")["rule"]
HLW = {"source": "nyfed", "series": "HLW_RSTAR", "label": "HLW", "real": True, "before_first": 2.0,
       "before_first_label": "Taylor's 2%"}
ESTIMATE = {"prior_quarters": 8, "drop_cuts_to_floor": True}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise RuntimeError("the variants touched the network")
    monkeypatch.setattr("socket.socket.connect", refuse)


def hlw_log():
    """Three synthetic vintages: the first published after the first fixture session, so it has none."""
    return pd.DataFrame({"date": pd.to_datetime(["2022-01-01", "2023-01-01", "2024-04-01"]),
                         "value": [0.4, 0.9, 0.7],
                         "published": pd.to_datetime(["2022-06-05", "2023-06-04", "2024-09-03"]),
                         "retrieved": fixtures.RETRIEVED})


# ---- plumbing: an override reaches the stage it changes -------------------------------

def test_an_hlw_override_reads_the_nyfed_log_as_a_real_rate():
    logs = {**fixtures.logs("USD"), ("nyfed", "HLW_RSTAR"): hlw_log()}
    out = regress.fixture_outputs("USD", logs=logs, overrides={"rule": {"rstar": HLW}})
    model = out["model"].set_index("session")
    want = {"2022-06-01": 2.0, "2022-06-13": 0.4, "2023-06-13": 0.9, "2024-09-17": 0.7, "2026-09-21": 0.7}
    assert model["rstar"].to_dict() == {T(d): v for d, v in want.items()}
    assert pd.isna(model.loc["2022-06-01", "sep_date"]) and model.loc["2023-06-13", "sep_date"] == T("2023-01-01")


def test_an_rstar_override_replaces_the_block_and_one_the_config_cannot_serve_raises():
    """Merged into GBP's ``{constant: -1.6}``, HLW would leave the constant in force and change nothing."""
    assert regress.merge(config.currency("GBP"), {"rule": {"rstar": HLW}})["rule"]["rstar"] == HLW
    assert regress.merge(config.currency("USD"), {"rule": {"rstar": {"constant": 1.1}}})["rule"]["rstar"] == {
        "constant": 1.1}
    with pytest.raises(config.ConfigError, match="nyfed/HLW_RSTAR"):
        regress.fixture_outputs("GBP", overrides={"rule": {"rstar": HLW}})
    with pytest.raises(config.ConfigError, match="market.curve.method"):
        regress.fixture_outputs("GBP", overrides={"market": {"curve": {"method": "NSS"}}})


def test_real_true_skips_the_target_subtraction():
    log = hlw_log()
    nominal = {**USD_RULE, "rstar": {k: v for k, v in HLW.items() if k != "real"}}
    assert rstar(log, "2023-06-13", {**USD_RULE, "rstar": HLW})["rstar"] == 0.9
    assert rstar(log, "2023-06-13", nominal)["rstar"] == pytest.approx(0.9 - USD_RULE["inflation_target"])
    assert rstar(log, "2022-06-04", {**USD_RULE, "rstar": HLW})["rstar"] == HLW["before_first"]


def test_an_nss_override_changes_the_gbp_paths_not_the_marks_and_naming_the_default_changes_nothing():
    frozen = regress.reference(regress.FIXTURE_REF / "GBP")[0]
    default = regress.fixture_outputs("GBP", overrides={"market": {"curve": {"method": "log_linear"}}})
    assert regress.compare(default, frozen) == {}
    fitted = regress.fixture_outputs("GBP", overrides={"market": {"curve": {"method": "nss"}}})
    diffs = regress.compare(fitted, frozen)
    assert {"sessions", "meetings", "paths", "signal", "backtest"} <= set(diffs) and "macro" not in diffs
    marks = ["k", "session", "change"]
    assert regress.compare({"b": fitted["backtest"][marks]}, {"b": frozen["backtest"][marks]}) == {}, \
        "the P&L is marked on the Bank's curve: only the positions move"
    moved = (fitted["meetings"]["rate"] - frozen["meetings"]["rate"]).abs() * 100
    assert 0 < moved.max() < 10, "a smooth fit to the same nodes moves a meeting's rate by basis points, not more"
    s = fitted["sessions"]
    assert (s["nss_rmse_bp"] < 1).all() and all(set(e.split("+")) <= set(nss.EDGES) for e in s["nss_edge"] if e)
    assert (s["n_nodes"] == frozen["sessions"]["n_nodes"]).all()


def test_an_estimate_override_with_no_usable_quarter_is_the_imposed_rule():
    """The fixture sessions have no quarter whose previous quarter is in the panel: every estimate is the prior."""
    frozen = regress.reference(regress.FIXTURE_REF / "USD")[0]
    out = regress.fixture_outputs("USD", overrides={"rule": {"estimate": ESTIMATE}})
    assert (out["model"]["coef_quarters"] == 0).all()
    assert (out["model"]["coef_inflation_gap"] == USD_RULE["coefficients"]["inflation_gap"]).all()
    assert set(regress.compare(out, frozen)) == {"model"}, "the path is the imposed rule's, bit for bit"
    new = [c for c in out["model"].columns if c not in frozen["model"].columns]
    assert new == ["coef_inflation_gap", "coef_unemployment_gap", "ols_inflation_gap", "ols_unemployment_gap",
                   "coef_quarters"]
    assert regress.compare({"model": out["model"].drop(columns=new)}, {"model": frozen["model"]}) == {}


# ---- NSS ----------------------------------------------------------------------------

def test_nss_recovers_a_curve_it_can_represent_and_names_a_grid_edge():
    t = np.arange(1, 61) / 12.0
    i, j = 5, 15                       # inside the grid: the first tau2 allowed with tau1 = TAU_GRID[5] is TAU_GRID[8]
    truth = {"beta": np.array([3.0, -1.0, 2.0, -1.5]), "tau1": nss.TAU_GRID[i], "tau2": nss.TAU_GRID[j]}
    got = nss.fit(t, nss.spot(truth, t))
    assert got["tau1"] == truth["tau1"] and got["tau2"] == truth["tau2"]
    assert got["beta"] == pytest.approx(truth["beta"], abs=1e-8) and got["rmse_bp"] < 1e-8
    assert got["edge"] == ""
    edge = {**truth, "tau1": nss.TAU_GRID[0], "tau2": nss.TAU_GRID[-1]}
    assert nss.fit(t, nss.spot(edge, t))["edge"] == "tau1_min+tau2_max"


def test_the_daily_nodes_span_the_published_maturities_and_are_the_fitted_curve():
    fitted = {"beta": np.array([3.0, -1.0, 2.0, -1.5]), "tau1": 0.5, "tau2": 2.0}
    curve = nss.daily(fitted, 1, 60, 365)
    assert curve.index[0] == 1 and curve.index[-1] == 60 and curve.index.is_monotonic_increasing
    assert np.diff(curve.index).max() <= 12 / 365 + 1e-12
    assert curve.to_numpy() == pytest.approx(nss.spot(fitted, curve.index.to_numpy() / 12))


# ---- converge-to-target conditioning ----------------------------------------------------

AS_OF = T("2024-01-10")
DATES = pd.DatetimeIndex(["2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
                          "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18"])


def flat_log(value, start="2023-01-02", end="2024-01-10", lag=0):
    days = pd.bdate_range(start, end)
    return pd.DataFrame({"date": days, "value": value, "published": days + pd.Timedelta(days=lag)})


def converging(**changes):
    return {**USD_RULE, "rstar": {"constant": 0.5}, "conditioning": {"converge": {"half_life_quarters": 4}},
            **changes}


def run(spec, infl=3.5, gap=-0.4):
    macro = {"as_of": AS_OF, "published": AS_OF, spec["inflation"]: infl, spec["gap"]: gap}
    return model_path(AS_OF, DATES, macro, None, flat_log(5.375), flat_log(5.33, lag=1), spec)


def test_inflation_and_the_gap_halve_their_distance_every_half_life():
    spec = converging()
    goals = converge_goals(AS_OF, DATES, 3.5, -0.4, 0.5, spec)
    q = (DATES - AS_OF).days.to_numpy() / 91.3125
    pi = 2.0 + 1.5 * 0.5 ** (q / 4)
    assert goals == pytest.approx(np.maximum(notional(pi, -0.4 * 0.5 ** (q / 4), 0.5, spec), spec["elb"]))
    summary, path = run(spec)
    assert summary["goal"] == pytest.approx(notional(3.5, -0.4, 0.5, spec)), "the summary stays today's picture"
    assert path["model_mid"].to_numpy() == pytest.approx(inertial_path(summary["r0"], goals, 8, spec))


def test_at_target_converge_is_hold_flat():
    spec = converging()
    hold = {k: v for k, v in spec.items() if k != "conditioning"}
    assert run(spec, 2.0, 0.0)[1]["model"].to_numpy() == pytest.approx(run(hold, 2.0, 0.0)[1]["model"].to_numpy())


def test_a_schedule_change_between_as_of_and_a_meeting_does_not_move_the_path():
    later = str(DATES[2].date())
    dated = converging(inflation_target=[{"from": "2000-01-01", "value": 2.0}, {"from": later, "value": 3.0}],
                       elb=[{"from": "2000-01-01", "value": 0.125}, {"from": later, "value": 4.0}],
                       meetings_per_quarter=[{"from": "2000-01-01", "value": 2}, {"from": later, "value": 3}])
    constant = converging(inflation_target=2.0, elb=0.125, meetings_per_quarter=2)
    assert np.array_equal(run(dated)[1]["model"].to_numpy(), run(constant)[1]["model"].to_numpy())


def test_under_converge_at_elb_means_the_goal_is_on_the_floor_at_every_meeting():
    """Rate on the floor, today's rule below it: hold-flat has no view, converge lifts the path as the gap closes."""
    def on_floor(spec, gap):
        macro = {"as_of": AS_OF, "published": AS_OF, spec["inflation"]: 1.0, spec["gap"]: gap}
        return model_path(AS_OF, DATES, macro, None, flat_log(0.125), flat_log(0.08, lag=1), spec)

    hold = {k: v for k, v in converging().items() if k != "conditioning"}
    summary, path = on_floor(hold, 0.6)
    assert summary["notional"] < summary["elb"] and summary["at_elb"]
    assert path["model_mid"].to_numpy() == pytest.approx(0.125)
    summary, path = on_floor(converging(), 0.6)
    assert summary["notional"] < summary["elb"] and not summary["at_elb"] and path["model_mid"].iloc[-1] > 0.2
    summary, path = on_floor(converging(), 3.0)
    assert summary["at_elb"] and path["model_mid"].to_numpy() == pytest.approx(0.125)


def test_an_array_target_of_one_rate_is_the_scalar_path_bit_for_bit():
    assert np.array_equal(inertial_path(0.125, 3.0, 8, USD_RULE), inertial_path(0.125, np.full(8, 3.0), 8, USD_RULE))
    with pytest.raises(ValueError, match="targets for"):
        inertial_path(0.125, np.full(7, 3.0), 8, USD_RULE)


# ---- the estimated rule --------------------------------------------------------------------

TRUE = np.array([1.2, 0.8])
RHO, R_STAR, PI_STAR = 0.85, 0.5, 2.0


def synthetic_panel(quarters=60, seed=7, elb_quarters=(), cut_quarters=()):
    """Three sessions a quarter whose policy rate follows the partial-adjustment rule with `TRUE` exactly.

    Quarters in `elb_quarters` sit on the floor with the rule below it, their rate nowhere near the rule's.
    Quarters in `cut_quarters` are an emergency cut: the rate goes to the floor with the rule above it.
    """
    rng = np.random.default_rng(seed)
    starts = pd.period_range("2005Q1", periods=quarters, freq="Q")
    rows, r = [], 3.0
    for n, q in enumerate(starts):
        pi, gap = 2.0 + rng.normal(0, 1.0), rng.normal(0, 1.0)
        r = RHO * r + (1 - RHO) * (R_STAR + pi + TRUE[0] * (pi - PI_STAR) - TRUE[1] * gap)
        floor = n in elb_quarters
        for day in [q.start_time + pd.Timedelta(days=5), q.start_time + pd.Timedelta(days=40), q.end_time.normalize()]:
            rows.append({"session": day, "inflation": pi, "u_gap": gap, "rstar": R_STAR,
                         "r0": 0.125 if floor or n in cut_quarters else r, "elb": 0.125, "at_elb": floor})
    return pd.DataFrame(rows)


SPEC = {**USD_RULE, "inertia": RHO, "inflation_target": PI_STAR, "estimate": ESTIMATE}


def last_row(coefs):
    return coefs.iloc[-1]


def test_ols_recovers_the_coefficients_and_the_shrunk_estimate_is_the_ridge_toward_the_imposed_ones():
    model = synthetic_panel()
    coefs = estimate.coefficients(model, SPEC)
    last = last_row(coefs)
    assert [last["ols_inflation_gap"], last["ols_unemployment_gap"]] == pytest.approx(TRUE, abs=1e-10)
    ends = estimate.quarter_ends(model, SPEC)
    used = ends[ends["usable"] & (ends["quarter"] < model["session"].iloc[-1].to_period("Q"))]
    X, y = used[["x_inflation_gap", "x_unemployment_gap"]].to_numpy(), used["y"].to_numpy()
    K = 8 * np.diag((X ** 2).mean(axis=0))
    beta0 = np.array([USD_RULE["coefficients"]["inflation_gap"], USD_RULE["coefficients"]["unemployment_gap"]])
    want = np.linalg.solve(X.T @ X + K, X.T @ y + K @ beta0)
    assert [last["coef_inflation_gap"], last["coef_unemployment_gap"]] == pytest.approx(want, abs=1e-12)
    assert last["coef_quarters"] == len(used) == 58   # the first quarter has no previous one, the last is today's


def test_the_estimate_starts_at_the_prior_and_moves_toward_the_data():
    coefs = estimate.coefficients(synthetic_panel(), SPEC)
    first = coefs.iloc[0]
    assert first["coef_quarters"] == 0 and pd.isna(first["ols_inflation_gap"])
    assert [first["coef_inflation_gap"], first["coef_unemployment_gap"]] == [0.5, 2.0]
    by_quarter = coefs.groupby("coef_quarters")[["coef_inflation_gap", "coef_unemployment_gap"]].first()
    distance = np.abs(by_quarter.to_numpy() - TRUE).sum(axis=1)
    # With 58 quarters against a prior of 8 the prior keeps about 8 / 66 of the distance.
    assert distance[-1] < distance[29] < distance[8] < distance[0] and distance[-1] < 0.15 * distance[0]


def test_with_no_prior_the_shrunk_estimate_is_ols_once_identified_and_never_nan():
    coefs = estimate.coefficients(synthetic_panel(), {**SPEC, "estimate": {**ESTIMATE, "prior_quarters": 0}})
    identified = coefs[coefs["coef_quarters"] >= 2]
    assert identified["coef_inflation_gap"].to_numpy() == pytest.approx(identified["ols_inflation_gap"].to_numpy())
    assert np.isfinite(coefs[["coef_inflation_gap", "coef_unemployment_gap"]].to_numpy()).all()


def test_a_singular_system_moves_the_estimate_off_the_prior_only_where_the_sample_identifies_it():
    beta0 = [0.5, 2.0]
    # No prior and one quarter: the estimate nearest the prior that fits the quarter exactly.
    x, y = np.array([0.5, -0.3]), 1.0
    shrunk, ols = estimate.fit(x[None, :], np.array([y]), beta0, 0)
    assert np.isnan(ols).all() and x @ shrunk == pytest.approx(y)
    assert shrunk - beta0 == pytest.approx(x * (y - x @ beta0) / (x @ x))
    # A regressor that is zero on every row (inflation exactly on target): the data and the prior's weight are
    # both zero there, so its coefficient keeps the prior and the other is the one-coefficient ridge.
    shrunk, _ = estimate.fit(np.array([[0.0, 0.6]]), np.array([-1.0]), beta0, 8)
    assert shrunk == pytest.approx([0.5, (0.6 * -1.0 + 8 * 0.36 * 2.0) / (0.36 + 8 * 0.36)])


def test_quarters_in_the_elb_state_and_the_quarter_after_them_are_left_out():
    model = synthetic_panel(elb_quarters=(20, 21, 22))
    ends = estimate.quarter_ends(model, SPEC)
    assert list(np.flatnonzero(~ends["usable"])) == [0, 20, 21, 22, 23]
    last = last_row(estimate.coefficients(model, SPEC))
    assert [last["ols_inflation_gap"], last["ols_unemployment_gap"]] == pytest.approx(TRUE, abs=1e-10)
    assert (regimes.elb_state(model).to_numpy() == model["at_elb"].to_numpy()).all()


def test_a_cut_to_the_floor_with_the_rule_above_it_is_censored_and_left_out_with_the_quarter_after():
    """An emergency cut: the inversion gives a goal far below the floor, which the rule cannot have."""
    model = synthetic_panel(cut_quarters=(30,))
    assert not regimes.elb_state(model).any()
    ends = estimate.quarter_ends(model, SPEC)
    assert ends.loc[30, "implied"] < -5 and list(np.flatnonzero(ends["cut"])) == [30, 31]
    assert list(np.flatnonzero(~ends["usable"])) == [0, 30, 31]
    last = last_row(estimate.coefficients(model, SPEC))
    assert [last["ols_inflation_gap"], last["ols_unemployment_gap"]] == pytest.approx(TRUE, abs=1e-10)
    kept = {**SPEC, "estimate": {**ESTIMATE, "drop_cuts_to_floor": False}}
    assert list(np.flatnonzero(~estimate.quarter_ends(model, kept)["usable"])) == [0]
    last = last_row(estimate.coefficients(model, kept))
    assert [last["ols_inflation_gap"], last["ols_unemployment_gap"]] != pytest.approx(TRUE, abs=1e-6), \
        "kept, the quarter pulls the fit off the truth"


@pytest.mark.parametrize("cut", ["2010-02-10", "2012-06-30", "2014-11-14"])
def test_no_estimate_on_or_before_a_date_moves_when_everything_after_it_is_poisoned(cut):
    model = synthetic_panel()
    later = model["session"] > T(cut)
    poisoned = model.assign(**{c: model[c].mask(later, 99.0) for c in ["inflation", "u_gap", "rstar", "r0"]})
    upto = model["session"] <= T(cut)
    want = estimate.coefficients(model, SPEC)[upto.to_numpy()].reset_index(drop=True)
    pd.testing.assert_frame_equal(estimate.coefficients(poisoned, SPEC)[upto.to_numpy()].reset_index(drop=True), want)
    pd.testing.assert_frame_equal(estimate.coefficients(model[upto], SPEC), want)
