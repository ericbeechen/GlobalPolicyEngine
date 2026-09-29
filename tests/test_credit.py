"""The credit bridge on synthetic data: OLS and Newey-West by hand, the weekly sampling, and each test's mechanics."""

import numpy as np
import pandas as pd
import pytest
from policypath import config, credit
from policypath.signal import components, gap


# ---- OLS and Newey-West, against numbers worked by hand ---------------------------------

def test_the_mean_and_its_newey_west_variance_by_hand():
    # y = 1..5 on a constant: beta = 3, u = -2..2, sum u^2 = 10, sum u_t u_(t-1) = 4, sum u_t u_(t-2) = -1.
    y, X = np.arange(1.0, 6.0), np.ones(5)
    assert credit.ols(y, X, 0)["beta"][0] == pytest.approx(3.0)
    assert credit.ols(y, X, 0)["se"][0] ** 2 == pytest.approx(10 / 25)
    assert credit.ols(y, X, 1)["se"][0] ** 2 == pytest.approx((10 + 2 * 0.5 * 4) / 25)          # 14/25
    assert credit.ols(y, X, 2)["se"][0] ** 2 == pytest.approx((10 + 2 * (2 / 3) * 4 + 2 * (1 / 3) * -1) / 25)
    assert credit.ols(y, X, 2, uniform=True)["se"][0] ** 2 == pytest.approx((10 + 2 * 4 + 2 * -1) / 25)   # 16/25


def test_equal_weights_can_give_a_negative_variance_and_then_no_se():
    # y = 1, -1, 1, -1 on a constant: sum u^2 = 4, sum u_t u_(t-1) = -3; Bartlett 4 - 3 = 1, equal weights 4 - 6 < 0.
    y, X = np.array([1.0, -1.0, 1.0, -1.0]), np.ones(4)
    assert credit.ols(y, X, 1)["se"][0] ** 2 == pytest.approx(1 / 16)
    assert np.isnan(credit.ols(y, X, 1, uniform=True)["se"][0])


def test_a_slope_its_r2_and_its_white_and_newey_west_variance_by_hand():
    # x centred, so X'X = diag(5, 10): b = sum xy / sum x^2 = 8/10, a = mean y = 3,
    # u = [-0.4, 0.8, -1, 1.2, -0.6], R^2 = 1 - 3.6/10. Scores x u = [0.8, -0.8, 0, 1.2, -1.2]:
    # sum (xu)^2 = 4.16, sum (xu)_t (xu)_(t-1) = -2.08, so var b = 4.16/100 (White), (4.16 - 2.08)/100 (lag 1).
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
    X = np.column_stack([np.ones(5), x])
    white, nw = credit.ols(y, X, 0), credit.ols(y, X, 1)
    assert white["beta"] == pytest.approx([3.0, 0.8])
    assert white["r2"] == pytest.approx(0.64)
    assert white["se"][1] ** 2 == pytest.approx(0.0416)
    assert nw["se"][1] ** 2 == pytest.approx(0.0208)
    assert nw["t"][1] == pytest.approx(0.8 / np.sqrt(0.0208))


def test_a_missing_row_keeps_its_place_in_time():
    # Row 2 left out: the mean of the other four is 3, and the lag-1 pairs across the gap count 0.
    fit = credit.ols(np.array([1.0, 2.0, np.nan, 4.0, 5.0]), np.ones(5), 1)
    assert fit["n"] == 4
    assert fit["se"][0] ** 2 == pytest.approx((10 + 2 * 0.5 * 4) / 16)


def test_newey_west_matches_the_sum_written_out():
    rng = np.random.default_rng(0)
    X = np.column_stack([np.ones(40), rng.normal(size=(40, 2))])
    y = X @ [0.5, 1.0, -2.0] + rng.normal(size=40)
    fit, lags = credit.ols(y, X, 3), 3
    u = y - X @ fit["beta"]
    g = X * u[:, None]
    meat = sum(np.outer(g[t], g[t]) for t in range(40))
    for j in range(1, lags + 1):
        w = 1 - j / (lags + 1)
        meat = meat + w * sum(np.outer(g[t], g[t - j]) + np.outer(g[t - j], g[t]) for t in range(j, 40))
    bread = np.linalg.inv(X.T @ X)
    assert fit["cov"] == pytest.approx(bread @ meat @ bread)


def test_collinear_regressors_give_nan_not_a_crash():
    fit = credit.ols(np.arange(5.0), np.column_stack([np.ones(5), np.ones(5)]), 0)
    assert np.isnan(fit["beta"]).all() and np.isnan(fit["r2"])


# ---- sampling --------------------------------------------------------------------

def test_weekly_takes_the_last_print_on_or_before_each_wednesday_within_the_week():
    days = pd.bdate_range("2024-01-01", "2024-02-02").drop(pd.Timestamp("2024-01-17"))  # a holiday Wednesday
    days = days[(days < "2024-01-22") | (days > "2024-01-31")]                              # and a week with no print
    values = pd.Series(np.arange(len(days), dtype=float), index=days)
    ends = credit.week_ends("2024-01-01", "2024-02-02", "Wednesday")
    got = credit.weekly(values, ends)
    assert list(ends.strftime("%m-%d")) == ["01-03", "01-10", "01-17", "01-24", "01-31"]
    assert got["2024-01-10"] == values["2024-01-10"]
    assert got["2024-01-17"] == values["2024-01-16"]     # the Tuesday
    assert got["2024-01-24"] == values["2024-01-19"]     # the Friday before, five days back
    assert np.isnan(got["2024-01-31"])                   # nothing in the seven days to it


def test_a_print_seven_days_before_a_week_end_belongs_to_the_week_before():
    values = pd.Series([1.0], index=pd.to_datetime(["2024-01-03"]))
    got = credit.weekly(values, pd.to_datetime(["2024-01-03", "2024-01-09", "2024-01-10"]))
    assert got.iloc[0] == got.iloc[1] == 1.0 and np.isnan(got.iloc[2])


def test_a_week_reads_nothing_after_its_end():
    days = pd.bdate_range("2024-01-01", "2024-03-01")
    values = pd.Series(np.arange(len(days), dtype=float), index=days)
    ends = credit.week_ends(days[0], days[-1], "Wednesday")
    poisoned = values.where(values.index <= "2024-02-07", 1e9)
    pd.testing.assert_series_equal(credit.weekly(values, ends)[:"2024-02-07"], credit.weekly(poisoned, ends)[:"2024-02-07"])


def test_on_sessions_carries_the_last_print_and_counts_stale_sessions():
    prints = pd.Series([1.0, 2.0], index=pd.to_datetime(["2024-01-03", "2024-01-05"]))
    sessions = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    got, stale = credit.on_sessions(prints, sessions)
    assert np.isnan(got.iloc[0]) and got.iloc[1:].tolist() == [1.0, 1.0, 2.0]
    assert stale == 1       # the 4th; the 2nd is before the first print, not stale


def test_a_spread_is_its_legs_signed_by_role_in_bp_where_all_printed():
    a = pd.Series([5.0, 5.5, 6.0], index=pd.bdate_range("2024-01-01", periods=3))
    b = pd.Series([4.0, 4.1], index=a.index[1:])
    series = {"A": a, "B": b}
    got = credit.spread(lambda source, name: series[name], {"long": {"source": "x", "series": "A"},
                                                            "minus": {"source": "x", "series": "B"}})
    assert got.tolist() == pytest.approx([150.0, 190.0])
    assert set(config.SPREAD_SIGNS) == {"long", "plus", "short", "minus"}


def test_the_held_change_follows_the_meeting_not_its_rank():
    s0, s1 = pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-10")
    e = pd.to_datetime(["2024-01-05", "2024-02-01", "2024-03-01"])
    # At s0 the second meeting is 2024-02-01; by s1 the first has passed and it is first.
    paths = pd.DataFrame({"session": [s0, s0, s0, s1, s1], "k": [1, 2, 3, 1, 2],
                          "effective_date": [e[0], e[1], e[2], e[1], e[2]],
                          "market": [1.0, 1.2, 1.5, 1.3, 1.7], "model": [1.0, 1.1, 1.2, 1.05, 1.25]})
    market, model = credit.held(paths, 2, pd.DatetimeIndex([s0]), pd.DatetimeIndex([s1]))
    assert market[0] == pytest.approx(10.0) and model[0] == pytest.approx(-5.0)


# ---- test 2 mechanics ------------------------------------------------------------

def test_ahead_starts_the_session_after_the_signal():
    s = pd.Series([0.0, 1.0, 3.0, 6.0, 10.0])
    assert credit.ahead(s, 2).tolist()[:2] == [6.0 - 1.0, 10.0 - 3.0]
    assert credit.ahead(s, 2).iloc[2:].isna().all()


def test_spans_marks_every_change_that_touches_the_window():
    sessions = pd.bdate_range("2020-02-03", periods=20)
    got = credit.spans(sessions, 2, ["2020-02-12", "2020-02-13"])
    # t's change runs from t + 1 to t + 3 sessions: from the 7th (it ends on the 12th) to the 12th (it starts on
    # the 13th). The 13th's starts on the 14th, after the window.
    assert list(sessions[got.to_numpy()].strftime("%m-%d")) == ["02-07", "02-10", "02-11", "02-12"]


def test_a_week_touches_a_window_its_change_overlaps():
    ends = credit.week_ends("2020-01-01", "2020-03-31", "Wednesday")
    got = credit.week_spans(ends, ("2020-02-12", "2020-02-19"))
    # The week to the 12th ends on the window's first day; the week to the 26th starts on its last.
    assert list(ends[got.to_numpy()].strftime("%m-%d")) == ["02-12", "02-19", "02-26"]


def cell(slope, t, mean):
    return {"slope": slope, "t": t, "nonoverlap": {"mean": mean}}


def test_d1s_rule_needs_a_negative_slope_a_t_past_the_line_and_a_negative_non_overlapping_mean():
    assert credit.supported(cell(-1.0, -2.0, -0.1))
    assert credit.supported(cell(-1.0, -credit.Z95, -0.1))
    assert not credit.supported(cell(-1.0, -2.0, 0.1))
    assert not credit.supported(cell(-1.0, -1.9, -0.1))
    assert not credit.supported(cell(1.0, -2.0, -0.1))


def test_the_predictive_fit_is_the_registered_regression_at_newey_west_lag_h():
    daily, _ = synthetic_world().daily_frame()
    for h in credit.HORIZONS:
        got = credit.predictive(daily, "quality", h, {})
        y = credit.ahead(daily["quality"], h).where(daily["z"].notna()).to_numpy()
        for regressor in ("z", "gap_bp"):
            fit = credit.ols(y, np.column_stack([np.ones(len(daily)), daily[regressor].to_numpy()]), h)
            assert got[regressor]["all"]["slope"] == pytest.approx(fit["beta"][1])
            assert got[regressor]["all"]["t"] == pytest.approx(fit["t"][1])


def test_the_staleness_control_is_the_yields_change_over_the_five_sessions_to_t_plus_one():
    daily, _ = synthetic_world().daily_frame()
    X = credit._ahead_design(daily, "z", True)
    yld = daily["yield"].to_numpy()
    assert X.shape[1] == 3 and credit._ahead_design(daily, "z", False).shape[1] == 2
    assert X[300:310, 2] == pytest.approx(yld[301:311] - yld[296:306])
    h = min(credit.HORIZONS)
    y = credit.ahead(daily["quality"], h).where(daily["z"].notna()).to_numpy()
    got = credit.predictive(daily, "quality", h, {}, stale=True)["z"]["all"]
    assert got["t"] == pytest.approx(credit.ols(y, X, h)["t"][1])


def test_the_samples_are_every_z_without_each_window_and_without_elb_sessions():
    daily, _ = synthetic_world().daily_frame()
    window = ("2013-01-01", "2013-02-28")
    got = credit.samples(daily, 21, {"w": window})
    assert got["all"].equals(daily["z"].notna())
    assert got["ex_elb"].sum() == (daily["z"].notna() & ~daily["elb"]).sum() < got["all"].sum()
    assert (got["all"] & ~got["ex_w"]).equals(got["all"] & credit.spans(daily.index, 21, window))


def test_the_non_overlapping_check_fits_each_offset_on_its_own_rows():
    rng = np.random.default_rng(1)
    n, h = 200, 5
    x = rng.normal(size=n)
    y = -0.5 * x + rng.normal(size=n)
    X = np.column_stack([np.ones(n), x])
    got = credit.nonoverlapping(y, X, h)
    slopes = [np.polyfit(x[o::h], y[o::h], 1)[0] for o in range(h)]
    assert got["offsets"] == h and got["n"] == n / h
    assert got["mean"] == pytest.approx(np.mean(slopes))
    assert (got["min"], got["max"]) == pytest.approx((min(slopes), max(slopes)))
    assert got["share_negative"] == np.mean(np.array(slopes) < 0)
    assert got["share_clearing"] == np.mean([credit.ols(y[o::h], X[o::h])["t"][1] <= -credit.Z95 for o in range(h)])
    assert credit.nonoverlapping(-y, X, h)["share_clearing"] == 0.0


# ---- staleness and test 3 mechanics -----------------------------------------------------

def test_staleness_shows_in_a_spread_that_catches_up_with_last_weeks_yield():
    rng = np.random.default_rng(2)
    dyield = pd.Series(rng.normal(size=600))
    fresh = pd.Series(rng.normal(size=600))
    stale = fresh + 0.5 * dyield.shift(1).fillna(0.0)
    assert not credit.staleness(fresh, dyield)["shows"]
    assert credit.staleness(stale, dyield)["shows"]


def weeks_with_slopes(slopes, n=26, noise=0.01, seed=3):
    """A weekly frame: one run of `n` weeks per (state, slope), Δspread = slope x Δlevel + noise."""
    rng = np.random.default_rng(seed)
    ends = pd.date_range("2015-01-07", periods=n * len(slopes), freq="W-WED")
    states = np.repeat([s for s, _ in slopes], n)
    level = rng.normal(size=len(ends))
    dy = np.repeat([b for _, b in slopes], n) * level + noise * rng.normal(size=len(ends))
    return pd.DataFrame({"quality": dy, "level": level, "state": states, "elb": states == "elb",
                         "start": ends - pd.Timedelta(days=7)}, index=ends)


def test_the_pooled_regression_recovers_each_regimes_slope_and_the_late_minus_early():
    slopes = [("elb", 9.0), ("early_hiking", -1.0), ("late_hiking", 1.0), ("cutting", 0.3)]
    w = weeks_with_slopes(slopes)
    by_state, diff = credit.pooled(w, "quality", "level")
    assert set(by_state) == {"early_hiking", "late_hiking", "cutting"}      # the ELB weeks are out
    assert by_state["early_hiking"]["slope"] == pytest.approx(-1.0, abs=0.01)
    assert by_state["cutting"]["weeks"] == 26
    assert diff["diff"] == pytest.approx(2.0, abs=0.02)
    assert diff["mde"] == pytest.approx((credit.Z95 + credit.Z80) * diff["se"])


def test_episodes_and_the_sign_sentence_over_two_cycles():
    slopes = [("early_hiking", -1.0), ("late_hiking", 1.0), ("cutting", 0.0),
              ("early_hiking", -0.5), ("late_hiking", -0.2)]
    w = weeks_with_slopes(slopes)
    state = w["state"].set_axis(w["start"])
    table = credit.episodes(w, state, "quality", "level")
    assert table["state"].tolist() == [s for s, _ in slopes]
    assert table["slope"].to_numpy() == pytest.approx([b for _, b in slopes], abs=0.01)
    cyc = credit.cycles(table)
    assert cyc["differs"].tolist() == [True, False] and cyc["as_expected"].tolist() == [True, False]
    assert cyc["early_first"].iloc[1] == table["first"].iloc[3]
    assert credit.sign_sentence(cyc) == ("The sign differs in one of the two cycles and not in the other: "
                                         "early negative and late positive, as the hypothesis has it.")
    assert credit.sign_sentence(cyc.iloc[[1]]) == "The sign does not differ in the one cycle."
    assert credit.sign_sentence(cyc.iloc[[1, 1]]) == "The sign does not differ in either of the two cycles."
    assert credit.sign_sentence(cyc.iloc[[0, 0]]) == ("The sign differs in each of the two cycles: "
                                                      "early negative and late positive, as the hypothesis has it.")
    assert credit.sign_sentence(cyc.iloc[[0, 1, 1]]).startswith(
        "The sign differs in one of the three cycles and not in the other two:")
    flipped = cyc.iloc[[0, 0]].assign(as_expected=[True, False])
    assert credit.sign_sentence(flipped).endswith("the hypothesis's way in one of them.")
    assert "no complete hiking cycle" in credit.sign_sentence(cyc.iloc[:0])


def test_an_early_episode_with_no_late_one_after_it_is_not_a_cycle():
    slopes = [("early_hiking", -1.0), ("late_hiking", 1.0), ("cutting", 0.0), ("early_hiking", -0.5)]
    w = weeks_with_slopes(slopes)
    cyc = credit.cycles(credit.episodes(w, w["state"].set_axis(w["start"]), "quality", "level"))
    assert len(cyc) == 1 and cyc["early_slope"].iloc[0] == pytest.approx(-1.0, abs=0.01)


def test_an_episode_shorter_than_a_quarter_gets_no_slope_of_its_own():
    w = weeks_with_slopes([("early_hiking", -1.0), ("late_hiking", 1.0)], n=credit.MIN_WEEKS - 1)
    assert credit.episodes(w, w["state"].set_axis(w["start"]), "quality", "level").empty
    assert credit.pooled(w, "quality", "level")[0] == {}


def test_an_episode_of_exactly_a_quarter_gets_one():
    w = weeks_with_slopes([("early_hiking", -1.0), ("late_hiking", 1.0)], n=credit.MIN_WEEKS)
    assert len(credit.episodes(w, w["state"].set_axis(w["start"]), "quality", "level")) == 2
    assert set(credit.pooled(w, "quality", "level")[0]) == {"early_hiking", "late_hiking"}


# ---- end to end, on a small synthetic world ---------------------------------------------

SIGNAL = {"window": "730D", "min_periods": 250, "sd_floor_bp": 5}


def made_up(block, sessions, spread, rng):
    """A walk for every series `block` names, but the primary's legs add up to `spread` (bp); cross-checks print last 300."""
    primary = block["spreads"][block["primary"]]
    legs = [*(ref for s in [*block["spreads"].values(), block["control"]] for ref in s.values()),
            *block["crosscheck"].values(), block["staleness"]]
    out = {n: pd.Series(3.0 + np.cumsum(rng.normal(0, 0.05, len(sessions))), index=sessions)
           for n in sorted({ref["series"] for ref in legs})}
    others = [role for role in primary if role != "long"]
    out |= {primary[role]["series"]: pd.Series(1.0, index=sessions) for role in others}
    out[primary["long"]["series"]] = pd.Series(spread / 100 - sum(config.SPREAD_SIGNS[r] for r in others), index=sessions)
    out |= {ref["series"]: out[ref["series"]][-300:] for ref in block["crosscheck"].values()}
    return out


def synthetic_world(lead=-0.03, seed=4):
    """Two currencies' panels and one's credit series. The spread's daily change is `lead` x yesterday's gap (bp)."""
    return world(lead, seed)[0]


def world(lead=-0.03, seed=4, block=None, r0=None):
    """`synthetic_world` and its currency block: (Inputs, cfg).

    `block` is a credit block to run instead of the small one here, on series
    `made_up` for it (the primary with the lead). `r0(sessions)` is the policy
    rate, percent, instead of two hikes off the floor.
    """
    rng = np.random.default_rng(seed)
    sessions = pd.bdate_range("2010-01-04", periods=1400)
    meetings = pd.bdate_range("2009-12-01", periods=80, freq="30B")
    level = np.cumsum(rng.normal(0, 0.01, len(sessions)))
    rows = []
    for i, s in enumerate(sessions):
        ahead = meetings[meetings > s][:8]
        for k, e in enumerate(ahead, start=1):
            rows.append((s, k, e, 1.0 + level[i] + 0.02 * k + 0.002 * rng.normal(), 1.0 + 0.01 * k))
    paths = pd.DataFrame(rows, columns=["session", "k", "effective_date", "market", "model"])
    other = paths.assign(market=paths["market"] + 0.001 * rng.normal(size=len(paths)))
    signals = {"AAA": gap.build(paths, SIGNAL), "BBB": gap.build(other, SIGNAL)}
    rate = (np.where(sessions < "2012-01-02", 0.1, np.where(sessions < "2013-06-03", 0.35, 0.6)) if r0 is None
            else r0(sessions))
    model = pd.DataFrame({"session": sessions, "r0": rate, "elb": 0.1, "at_elb": sessions < "2011-01-03"})
    panels = {("AAA", "signal"): signals["AAA"], ("BBB", "signal"): signals["BBB"], ("AAA", "paths"): paths,
              ("AAA", "model"): model}

    gap4 = signals["AAA"].query("k == 4").set_index("session")["gap_bp"]
    quality = 150 + np.cumsum(np.r_[0.0, lead * gap4.to_numpy()[:-1]] + rng.normal(0, 0.3, len(sessions)))
    ten = 3.0 + np.cumsum(rng.normal(0, 0.05, len(sessions)))
    series = {"BAA": pd.Series(quality / 100 + 1.0, index=sessions), "AAA_Y": pd.Series(1.0, index=sessions),
              "TEN": pd.Series(ten, index=sessions), "THIRTY": pd.Series(ten + 0.4, index=sessions),
              "IG": pd.Series(1.0 + 0.001 * rng.normal(size=300).cumsum(), index=sessions[-300:])}
    ref = {n: {"source": "x", "series": n} for n in series}
    if block is None:
        block = {"primary": "quality", "spreads": {"quality": {"long": ref["BAA"], "short": ref["AAA_Y"]}},
                 "control": {"long": ref["THIRTY"], "short": ref["TEN"]}, "crosscheck": {"ig": ref["IG"]},
                 "staleness": ref["TEN"], "week_ends": "Wednesday", "exclude": {"window": ["2013-01-01", "2013-02-28"]},
                 "labels": {"quality": "Q", "ig": "IG", "control": "C", "staleness": "T"}}
    else:
        series = made_up(block, sessions, quality, rng)
    cfg = {"backtest": {"horizon": 4}, "signal": SIGNAL, "credit": block, "report": {"labels": {"market": "M"}}}
    book = {"signal": {"slope": [1, 8], "slope_orthogonal": True},
            "sleeves": [{"name": "AAA - BBB 2y", "kind": "cross", "pair": ["AAA", "BBB"], "tenor": 2}]}
    inputs = credit.Inputs("AAA", lambda c: cfg, book, lambda c, name: panels[(c, name)],
                           lambda source, name: series[name])
    return inputs, cfg


def test_the_bridge_runs_end_to_end_and_finds_a_lead_that_is_there():
    inputs = synthetic_world(lead=-0.03)
    results, frames = credit.run(inputs)
    assert results["spreads"] == ["quality", "ig"]
    assert results["regressors"] == ["market", "model", "slope", "differential"]
    head = results["predictive"]["quality"][max(credit.HORIZONS)]["z"]["all"]
    assert head["slope"] < 0 and results["supported"]
    weeks = frames["weeks"]
    assert (weeks.index.dayofweek == 2).all()
    assert (weeks["start"] < weeks.index).all() and (weeks.index - weeks["start"] < pd.Timedelta(days=14)).all()
    assert weeks["level"].equals(weeks["market"] + weeks["model"])
    own = set(config.CREDIT_COLUMNS) | set(results["regressors"])
    assert set(weeks.columns) - set(results["spreads"]) <= own
    assert set(frames["daily"].columns) - set(results["spreads"]) <= own
    assert results["sample"]["sessions"] == int(frames["daily"]["z"].notna().sum())
    ex = results["predictive"]["quality"][max(credit.HORIZONS)]["z"]["ex_window"]
    assert ex["n"] < head["n"]
    mask = credit.week_spans(weeks.index, ("2013-01-01", "2013-02-28"))
    described = credit.staleness(weeks["quality"].where(~mask), weeks["yield"].where(~mask))
    assert results["staleness_ex"]["quality"]["window"]["ar1"] == pytest.approx(described["ar1"])
    assert results["staleness_ex"]["quality"]["window"]["n"] < results["staleness"]["quality"]["n"]


def test_the_weekly_level_is_the_held_gaps_change_and_the_rest_the_signals_own_changes():
    inputs = synthetic_world()
    rng = np.random.default_rng(7)
    # A rule that moves, so the model's half of the level is not zero on the weeks no meeting passes.
    inputs.paths = inputs.paths.assign(model=inputs.paths["model"] + 0.05 * rng.normal(size=len(inputs.paths)))
    inputs.level = components.level(gap.build(inputs.paths, SIGNAL), inputs.k)
    weeks = inputs.weeks()
    start, end = pd.DatetimeIndex(weeks["start"]), inputs.sessions[credit.on_or_before(inputs.sessions, weeks.index)]
    meeting = inputs.paths[inputs.paths["k"] == inputs.k].set_index("session")["effective_date"]
    held = meeting.reindex(start).to_numpy() == meeting.reindex(end).to_numpy()
    assert held.sum() > 50 and (~held).sum() > 10
    gap_change = inputs.level["value_bp"].reindex(end).to_numpy() - inputs.level["value_bp"].reindex(start).to_numpy()
    assert weeks["level"].to_numpy()[held] == pytest.approx(gap_change[held])
    def own(values):
        return values.reindex(end).to_numpy() - values.reindex(start).to_numpy()
    market = inputs.paths[inputs.paths["k"] == inputs.k].set_index("session")["market"] * 100
    assert weeks["market"].to_numpy()[held] == pytest.approx(own(market)[held])
    assert weeks["slope"].to_numpy() == pytest.approx(own(inputs.slope["value_bp"]), nan_ok=True)
    diff = next(iter(inputs.differentials.values()))["value_bp"]
    assert weeks["differential"].to_numpy() == pytest.approx(own(diff), nan_ok=True)


def test_a_week_takes_its_regime_and_elb_state_from_the_session_it_starts_on():
    inputs = synthetic_world()
    days = inputs.sessions
    inputs.state = pd.Series(days.strftime("%Y-%m-%d"), index=days)
    weeks = inputs.weeks()
    assert (weeks["state"] == weeks["start"].dt.strftime("%Y-%m-%d")).all()
    starts = days.isin(pd.DatetimeIndex(weeks["start"]))
    inputs.elb = pd.Series(starts, index=days)
    assert inputs.weeks()["elb"].all()
    inputs.elb = pd.Series(~starts, index=days)
    assert not inputs.weeks()["elb"].any()


def test_the_contemporaneous_fit_without_a_window_is_the_fit_on_the_other_weeks():
    inputs = synthetic_world()
    weeks, parts = inputs.weeks(), inputs.regressor_names()
    mask = credit.week_spans(weeks.index, ("2013-01-01", "2013-02-28"))
    full = credit.contemporaneous(weeks, "quality", parts)["components"]
    got = credit.contemporaneous(weeks, "quality", parts, drop=mask)["components"]
    usable = weeks[["quality", *parts, "control"]].notna().all(axis=1)
    kept = weeks[usable & ~mask]
    hand = credit.ols(kept["quality"], np.column_stack([np.ones(len(kept)), kept[parts]]))
    assert got["n"] == full["n"] - (usable & mask).sum() < full["n"]
    assert [got["coef"][c] for c in ["const", *parts]] == pytest.approx(hand["beta"])


def test_the_placebo_rotates_the_spreads_changes_against_z():
    daily, _ = synthetic_world(lead=0.0, seed=5).daily_frame()
    h = max(credit.HORIZONS)
    got = credit.placebo(daily, "quality", h)
    assert got["t"] == pytest.approx(credit.predictive(daily, "quality", h, {})["z"]["all"]["t"])
    part = daily[daily["z"].first_valid_index():]
    n = len(part) - 1
    assert got["shifts"] == len(range(credit.PLACEBO_MIN, n - credit.PLACEBO_MIN + 1, credit.PLACEBO_STEP))
    def by_hand(k):
        """Rotate the daily changes by k, rebuild the spread from them, refit the headline cell: its t."""
        level = pd.Series(np.r_[0.0, np.cumsum(np.roll(np.diff(part["quality"].to_numpy()), k))], index=part.index)
        y = credit.ahead(level, h).where(part["z"].notna()).to_numpy()
        return credit.ols(y, np.column_stack([np.ones(len(part)), part["z"].to_numpy()]), h)["t"][1]
    k = credit.PLACEBO_MIN + 7
    one, other = by_hand(k), by_hand(n - k)
    alone = credit.placebo(daily, "quality", h, least=k, step=n)
    assert alone["shifts"] == 1 and alone["critical"] == pytest.approx(one)
    assert alone["reject"] == float(abs(one) >= credit.Z95) and alone["p"] == float(one <= got["t"])
    pair = credit.placebo(daily, "quality", h, least=k, step=n - 2 * k)
    assert pair["shifts"] == 2 and pair["critical"] == pytest.approx(np.quantile([one, other], 0.025))


def test_a_lead_that_is_there_is_rare_under_the_placebo():
    daily, _ = synthetic_world(lead=-0.03).daily_frame()
    assert credit.placebo(daily, "quality", max(credit.HORIZONS))["p"] < 0.05


def test_no_lead_is_no_lead():
    results, _ = credit.run(synthetic_world(lead=0.0, seed=5))
    assert not results["supported"]


def test_without_each_year_leaves_out_every_change_that_touches_the_year():
    daily, _ = synthetic_world().daily_frame()
    h = max(credit.HORIZONS)
    got = credit.without_each_year(daily, "quality", h)
    assert list(got) == [str(y) for y in range(daily["z"].first_valid_index().year, daily.index[-1].year + 1)]
    everything = credit.predictive(daily, "quality", h, {})["z"]["all"]
    usable = daily["z"].notna() & credit.ahead(daily["quality"], h).notna()
    for year in got:
        touched = credit.spans(daily.index, h, (f"{year}-01-01", f"{year}-12-31")) & usable
        assert got[year]["n"] == everything["n"] - touched.sum()
        assert got[year]["supported"] == credit.supported(got[year])


def test_the_checks_after_the_run_are_the_headline_refitted_four_ways():
    daily, _ = synthetic_world().daily_frame()
    h = max(credit.HORIZONS)
    floor = float(daily["sd_bp"].median())
    daily = daily.assign(sd_bp=daily["sd_bp"].clip(lower=floor))       # as the signal floors it: exactly at the floor
    got = credit.after_the_run(daily, "quality", h, floor)
    y = credit.ahead(daily["quality"], h).where(daily["z"].notna()).to_numpy()
    X = np.column_stack([np.ones(len(daily)), daily["z"].to_numpy()])
    head = credit.predictive(daily, "quality", h, {})["z"]["all"]
    assert got["bartlett_2h"]["t"] == pytest.approx(credit.ols(y, X, 2 * h)["t"][1])
    assert got["uniform"]["t"] == pytest.approx(credit.ols(y, X, h - 1, uniform=True)["t"][1])
    assert got["bartlett_2h"]["slope"] == got["uniform"]["slope"] == pytest.approx(head["slope"])
    clipped = credit.predictive(daily.assign(z=daily["z"].clip(-credit.Z_CLIP, credit.Z_CLIP)), "quality", h, {})
    assert got["z_clipped"]["slope"] == pytest.approx(clipped["z"]["all"]["slope"])
    at_floor = (daily["sd_bp"] == floor) & np.isfinite(y) & daily["z"].notna()
    assert got["above_floor"]["n"] == head["n"] - at_floor.sum() > 0 and at_floor.sum() > 100
