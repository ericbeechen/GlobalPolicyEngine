"""The book's arithmetic on hand-made numbers, and the whole book on the synthetic two-currency world.

Checks: the EWMA covariance is the lambda-weighted mean of the outer products
of the complete rows before the session, with n_eff = 1 / sum w^2; the lag
term is the Bartlett-weighted lag-1 moment; the shrinkage intensity is the
formula on a hand example; ERC's risk contributions are equal and one sleeve
is inverse-vol; mean-variance is |z|-weighted inverse-vol when the covariance
is diagonal, and inverse-vol when every |z| is at the cap; the vol target
hits its target and the cap holds; the band trades a new side and a move
outside it, and gives way to the cap; the drawdown state machine reads the
close before and releases under its release level; the book calendar sums a
sleeve session into the next book session and counts stale ones. On the
synthetic world: nothing up to D (the covariance, to the first close after D,
the targets, the decided positions, the overlay's scale, the P&L and costs
credited) moves when every mark, z and overnight rate after D is poisoned;
the sleeves, built like the real ones, have a full-rank covariance, which a
cross built from the outright legs would not; the book's covariance has the
configured lambda and non-synchronous term; a book is counted from the first session its first decision
is held; the overlay's scale multiplies the targets of the close it is
decided at; a resize is the trade executed lag sessions after another
sleeve's side moved; the summary's vol, worst drawdown and half-size share
are the net P&L's on the counted sessions; mean-variance without
correlations is |z|-weighted inverse-vol. The numbers the words read: the
paired SE of a Sharpe difference (Jobson-Korkie with Memmel's correction,
scale-free), block vol, an exclude window left out, "never released". The
floored matrix has the floored sds and the same correlations, and the
unweighted lag term's failures are counted right on a hand example.
"""

from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from policypath.backtest import metrics
from policypath.report import portfolio as report
from policypath.strategy import portfolio, positions, risk
from test_expression import CUT
from test_report_costs import world_of

T = pd.Timestamp
DAYS = pd.bdate_range("2024-01-01", periods=12)


# ---- the covariance ----------------------------------------------------------------

def weights(complete, lam, t):
    """The normalised weights of the complete rows before position t: lambda^(t - 1 - j), zero for others."""
    w = np.array([lam ** (t - 1 - j) if complete[j] else 0.0 for j in range(t)])
    return w / w.sum()


def rows(seed=1, n=12, k=3):
    x = np.random.default_rng(seed).normal(0.0, [1.0, 2.0, 0.5][:k], (n, k))
    x[:, 1] += 0.6 * x[:, 0]
    x[4, 2] = np.nan                                  # one sleeve missing: the row is not an observation
    return pd.DataFrame(x, index=pd.bdate_range("2024-01-01", periods=n), columns=list("abc")[:k])


def test_the_ewma_covariance_is_the_weighted_outer_products_of_complete_rows_before_t():
    lam = 0.8
    x = rows()
    cov = risk.ewma_cov(x, lam, lags=0)
    v = x.to_numpy()
    complete = ~np.isnan(v).any(axis=1)
    first = risk.min_periods(lam)                     # 5 complete rows before an estimate exists
    assert np.isnan(cov.s0[:first + 1]).all()        # rows 0-5 hold 5 complete rows only from position 6 on
    for t in (first + 1, 8, 11):
        w = weights(complete, lam, t)
        ok = w > 0
        want = (w[ok, None, None] * v[:t][ok][:, :, None] * v[:t][ok][:, None, :]).sum(axis=0)
        assert np.allclose(cov.s0[t], want, rtol=1e-12)
        assert cov.n_eff[t] == pytest.approx(1.0 / (w ** 2).sum(), rel=1e-12)
        assert cov.observations[t] == complete[:t].sum()
    np.testing.assert_array_equal(cov.s, cov.s0)       # no lag term asked for


def test_n_eff_tends_to_one_plus_lambda_over_one_minus_lambda():
    x = pd.DataFrame(np.random.default_rng(2).normal(size=(600, 2)), index=pd.bdate_range("2020-01-01", periods=600))
    cov = risk.ewma_cov(x, 0.97)
    n = 599
    closed = ((1 - 0.97 ** n) / 0.03) ** 2 / ((1 - 0.97 ** (2 * n)) / (1 - 0.97 ** 2))
    assert cov.n_eff[-1] == pytest.approx(closed, rel=1e-10)
    assert cov.n_eff[-1] == pytest.approx(1.97 / 0.03, rel=1e-6)


def test_the_lag_term_is_the_bartlett_weighted_lag_one_moment():
    lam = 0.8
    x = rows(3)
    v = x.to_numpy()
    complete = ~np.isnan(v).any(axis=1)
    for lags in (1, 2):
        cov = risk.ewma_cov(x, lam, lags)
        t = 10
        want = cov.s0[t].copy()
        for lag in range(1, lags + 1):
            pairs = np.array([j >= lag and complete[j] and complete[j - lag] for j in range(t)])
            w = np.array([lam ** (t - 1 - j) if pairs[j] else 0.0 for j in range(t)])
            w = w / w.sum()                            # normalised over the pairs there are
            s1 = sum(w[j] * np.outer(v[j], v[j - lag]) for j in range(t) if pairs[j])
            want = want + (1 - lag / (lags + 1)) * (s1 + s1.T)
        assert np.allclose(cov.s[t], want, rtol=1e-12)
    assert np.allclose(risk.ewma_cov(x, lam, 1).s[10] - risk.ewma_cov(x, lam, 0).s[10],
                       (risk.ewma_cov(x, lam, 1).s[10] - risk.ewma_cov(x, lam, 1).s0[10]))


def test_the_shrinkage_intensity_is_the_formula_on_a_hand_example():
    lam = 0.8
    x = rows(4)
    cov = risk.ewma_cov(x, lam, 1)
    v = x.to_numpy()
    complete = ~np.isnan(v).any(axis=1)
    t = 11
    w = weights(complete, lam, t)
    ok = w > 0
    s0 = cov.s0[t]
    prod = v[:t][ok][:, :, None] * v[:t][ok][:, None, :]
    pi = (w[ok, None, None] * (prod - s0) ** 2).sum(axis=0)      # sum w (x_i x_j - S0_ij)^2
    n_eff = 1.0 / (w ** 2).sum()
    off = ~np.eye(3, dtype=bool)
    s = cov.s[t]
    delta = min(max(pi[off].sum() / n_eff / (s[off] ** 2).sum(), 0.0), 1.0)
    assert cov.delta[t] == pytest.approx(delta, rel=1e-10)
    assert np.allclose(cov.shrunk[t], (1 - delta) * s + delta * np.diag(np.diag(s)), rtol=1e-12)
    assert np.allclose(np.diag(cov.shrunk[t]), np.diag(s))       # the diagonal is not shrunk
    live = ~np.isnan(cov.delta)
    assert ((cov.delta[live] >= 0) & (cov.delta[live] <= 1)).all()


def test_condition_is_scale_free():
    m = np.array([[4.0, 1.0], [1.0, 1.0]])
    d = np.diag([10.0, 0.1])
    assert risk.condition(d @ m @ d) == pytest.approx(risk.condition(m))
    assert risk.condition(np.eye(3)) == pytest.approx(1.0)


# ---- the constructions -------------------------------------------------------------

def a_cov(n=4, seed=5):
    a = np.random.default_rng(seed).normal(size=(n, n))
    return a @ a.T + n * np.eye(n)


def test_erc_equalises_risk_contributions_with_the_signs_of_g():
    cov = a_cov()
    g = np.array([1.0, -1.0, 1.0, -1.0])
    w = portfolio.erc(g, cov)
    rc = w * (cov @ w)
    assert np.allclose(rc, rc.mean(), rtol=1e-9) and (np.sign(w) == g).all()
    share = portfolio.contributions(w, cov)
    assert share.sum() == pytest.approx(1.0) and np.allclose(share, 0.25, atol=1e-9)
    warm = portfolio.erc(g, cov, v0=np.abs(w) * 3.0)                                 # a warm start finds the same point
    assert np.allclose(warm / np.abs(warm).sum(), w / np.abs(w).sum(), rtol=1e-9)


def test_erc_with_one_sleeve_is_inverse_vol():
    cov = np.array([[4.0]])
    assert portfolio.erc(np.array([-1.0]), cov) == pytest.approx(portfolio.inverse_vol(np.array([-1.0]), np.array([2.0])))


def test_mean_variance_is_z_weighted_inverse_vol_when_sigma_is_diagonal():
    sd = np.array([1.0, 2.0, 4.0])
    g, z = np.array([1.0, -1.0, 1.0]), np.array([1.5, -4.0, 2.0])
    cov = np.diag(sd ** 2)
    w = portfolio.mean_variance(g, cov, sd, z, 3.0)
    assert np.allclose(w, g * np.minimum(np.abs(z), 3.0) / sd)
    capped = portfolio.mean_variance(g, cov, sd, np.array([3.5, -3.0, 5.0]), 3.0)   # every |z| at the cap
    iv = portfolio.inverse_vol(g, sd)
    assert np.allclose(portfolio.vol_target(capped, cov, 252, 1e6), portfolio.vol_target(iv, cov, 252, 1e6))
    full = a_cov(3)                                       # and in general, D^-1 R^-1 (mu / sd)
    sd = np.sqrt(np.diag(full))
    r = full / np.outer(sd, sd)
    mu = g * np.minimum(np.abs(z), 3.0) * sd
    assert np.allclose(portfolio.mean_variance(g, full, sd, z, 3.0), np.linalg.solve(r, mu / sd) / sd)


def test_the_vol_target_hits_its_target_on_a_known_sigma():
    cov = a_cov(3)
    q = portfolio.vol_target(np.array([1.0, -2.0, 0.5]), cov, 252, 5e6)
    assert np.sqrt(q @ cov @ q * 252) == pytest.approx(5e6, rel=1e-12)


def frames(g, z=None):
    days = pd.bdate_range("2024-01-01", periods=len(g))
    g = pd.DataFrame(g, index=days, columns=["a", "b"], dtype=float)
    return g, g * 2.0 if z is None else pd.DataFrame(z, index=days, columns=["a", "b"])


def test_targets_size_only_active_sleeves_at_the_target_and_under_the_cap():
    g, z = frames([[0, 0], [1, 0], [1, -1], [1, -1], [0, 0]])
    sd = np.tile([2.0, 1.0], (5, 1))
    cov = np.tile(np.array([[4.0, 0.6], [0.6, 1.0]]), (5, 1, 1))
    legs = [1, 2]
    free = portfolio.targets("inverse_vol", g, z, sd, cov, legs, 252, 5e6, None, 1, 3.0)
    assert (free.q.iloc[0] == 0).all() and (free.q.iloc[4] == 0).all()             # before `first`, and flat
    assert free.q.iloc[1, 1] == 0 and free.q.iloc[1, 0] > 0 and free.q.iloc[2, 1] < 0
    assert np.allclose(free.exante.iloc[1:4], 5e6)
    ratio = free.q.iloc[2, 0] / -free.q.iloc[2, 1]
    assert ratio == pytest.approx(0.5)                                             # g / sd
    limit = 0.8 * free.uncapped.iloc[2]
    capped = portfolio.targets("inverse_vol", g, z, sd, cov, legs, 252, 5e6, limit, 1, 3.0)
    assert capped.capped.iloc[2] and portfolio.gross(capped.q.iloc[2].to_numpy(), legs) == pytest.approx(limit)
    assert capped.exante.iloc[2] == pytest.approx(0.8 * 5e6)
    bad = np.tile(np.array([[1.0, 2.0], [2.0, 1.0]]), (5, 1, 1))                    # not positive definite
    with pytest.raises(RuntimeError, match="not positive definite"):
        portfolio.targets("erc", g, z, sd, bad, legs, 252, 5e6, None, 1, 3.0)


def test_the_band_trades_a_new_side_and_a_move_outside_it_and_gives_way_to_the_cap():
    days = pd.bdate_range("2024-01-01", periods=7)
    target = pd.DataFrame({"a": [0.0, 100, 105, 115, -50, -52, 0], "b": [10.0, 10, 10, 10, 10, 10, 10]}, index=days)
    got = portfolio.band(target, 0.10, [1, 1])
    assert got["a"].tolist() == [0, 100, 100, 115, -50, -50, 0]
    assert (got["b"] == 10).all()
    down = pd.DataFrame({"a": [100.0, 95.0], "b": [10.0, 10.0]}, index=days[:2])     # a 5% move, inside the band
    assert portfolio.band(down, 0.10, [1, 1])["a"].tolist() == [100, 100]
    tight = portfolio.band(down, 0.10, [1, 1], limit=106.0)                         # 100 + 10 kept would break it
    assert tight["a"].tolist() == [100, 95]


# ---- the drawdown overlay ----------------------------------------------------------

def test_the_drawdown_is_from_the_peak_in_capital_units():
    e = pd.Series([0.0, -5.0, 10.0, 2.0, 12.0], index=DAYS[:5])
    assert portfolio.drawdown(e, 100.0).tolist() == [0.0, 0.05, 0.0, 0.08, 0.0]


def test_the_drawdown_state_machine_reads_the_close_before_and_releases_under_its_level():
    #                  dd:  0    .08   .11   .12   .07   .06   .04   .11   .10
    e = pd.Series([0.0, -8.0, -11.0, -12.0, -7.0, -6.0, -4.0, -11.0, -10.0], index=DAYS[:9])
    scale, starts = portfolio.overlay(e, 100.0, 0.10, 0.05, 0.5)
    assert scale.tolist() == [1, 1, 1, 0.5, 0.5, 0.5, 0.5, 1, 0.5]     # .07 and .06 hold it; .04 releases it
    assert starts == [DAYS[3], DAYS[8]]              # dd 0.11 at the close of DAYS[2] acts at DAYS[3]'s
    at_trigger = pd.Series([0.0, -10.0, -10.0], index=DAYS[:3])
    assert portfolio.overlay(at_trigger, 100.0, 0.10, 0.05, 0.5)[0].tolist() == [1, 1, 1]      # over, not at
    _, twice = portfolio.overlay(pd.Series([0, -11, -2, -13, -1.0], index=DAYS[:5]), 100.0, 0.10, 0.05, 0.5)
    assert twice == [DAYS[2], DAYS[4]]


# ---- the book calendar -------------------------------------------------------------

def test_a_sleeve_session_is_summed_into_the_next_book_session():
    book = pd.DatetimeIndex(["2024-05-24", "2024-05-28", "2024-05-29", "2024-05-30"])     # a US holiday, the 27th
    sleeve = pd.Series([1.0, 2.0, 4.0, 8.0, 16.0],
                       index=pd.DatetimeIndex(["2024-05-23", "2024-05-24", "2024-05-27", "2024-05-28", "2024-05-31"]))
    assert portfolio.to_book(sleeve, book).tolist() == [2.0, 12.0, 0.0, 0.0]        # before and after: dropped
    assert portfolio.credits(sleeve.index, book).tolist() == [1, 2, 0, 0]
    units = portfolio.unit_pnl({"x": sleeve}, book)["x"]
    assert units.tolist() == [2.0, 12.0, 0.0, 0.0]                                    # 0 on a stale session
    late = portfolio.unit_pnl({"x": sleeve.iloc[3:]}, book)["x"]
    assert late.isna().tolist() == [True, False, False, False]


def test_a_level_is_read_as_of_the_book_close_and_a_missing_one_stays_missing():
    z = pd.Series([1.0, np.nan, 2.0], index=pd.DatetimeIndex(["2024-05-23", "2024-05-24", "2024-05-28"]))
    book = pd.DatetimeIndex(["2024-05-22", "2024-05-24", "2024-05-27", "2024-05-29"])
    got = portfolio.asof(z, book)
    assert got.isna().tolist() == [True, True, True, False] and got.iloc[3] == 2.0


# ---- the whole book on the synthetic world -----------------------------------------

def book_world(root, poison_after=None):
    """The synthetic world, with a book that trades early and a drawdown control that fires before D."""
    w = world_of(root, poison_after)
    w.book["portfolio"]["min_history"] = 60
    w.book["portfolio"]["drawdown"] = {"trigger": 0.01, "release": 0.005, "scale": 0.5}
    return w


@pytest.fixture(scope="module")
def worlds(tmp_path_factory):
    clean = report.setup(book_world(tmp_path_factory.mktemp("clean")))
    poisoned = report.setup(book_world(tmp_path_factory.mktemp("poisoned"), poison_after=CUT))
    return clean, poisoned


def test_nothing_up_to_d_sees_anything_after_it(worlds):
    clean, poisoned = worlds
    names = list(clean.inputs)
    a, b = report.risk_of(clean, names), report.risk_of(poisoned, names)
    upto = clean.days <= CUT
    read = np.r_[True, upto[:-1]]                       # a close's covariance reads the P&L strictly before it
    for m in ("s0", "s", "shrunk"):
        np.testing.assert_array_equal(getattr(a.cov, m)[read], getattr(b.cov, m)[read])
    assert not np.array_equal(a.cov.s[~upto][5:], b.cov.s[~upto][5:], equal_nan=True)      # the poison is there
    traded = 0
    for c in ("inverse_vol", "erc", "mean_variance"):
        for overlay in (False, True):
            ch = report.Choice(c, overlay=overlay)
            x, y = report.run_book(clean, a, ch), report.run_book(poisoned, b, ch)
            pd.testing.assert_frame_equal(x.targets.q.loc[:CUT], y.targets.q.loc[:CUT])
            pd.testing.assert_frame_equal(x.decided.loc[:CUT], y.decided.loc[:CUT])
            pd.testing.assert_series_equal(x.scale.loc[:CUT], y.scale.loc[:CUT])
            pd.testing.assert_frame_equal(x.run.daily.loc[:CUT], y.run.daily.loc[:CUT], obj=ch.words())
            traded += int((x.run.daily.loc[:CUT, "traded"] > 0).sum())
            if overlay:
                assert any(e <= CUT for e in x.episodes)                              # the overlay acts before D
    assert traded > 500


def test_the_sleeves_covariance_is_full_rank_and_an_outright_cross_would_not_be(worlds):
    clean, _ = worlds
    x = clean.units.dropna()
    e = np.linalg.eigvalsh(np.corrcoef(x.to_numpy().T))
    assert e[0] > 1e-3 * e[-1]                                    # five sleeves, five directions
    outright = x.assign(cross=x["GBP outright"] - x["USD outright"])     # a cross built from the outright legs
    e = np.linalg.eigvalsh(np.corrcoef(outright.to_numpy().T))
    assert e[0] < 1e-10 * e[-1]


def test_the_books_covariance_has_the_configured_lambda_and_non_synchronous_term(worlds):
    clean, _ = worlds
    names, r = list(clean.inputs), clean.book["risk"]
    got = report.risk_of(clean, names).cov
    want = risk.ewma_cov(clean.units[names], r["ewma_lambda"], r["nonsynchronous_lag"])
    for m in ("s0", "s", "shrunk"):
        np.testing.assert_array_equal(getattr(got, m), getattr(want, m))
    assert r["nonsynchronous_lag"] >= 1 and np.nanmax(np.abs(got.s - got.s0)) > 0     # the lag term is there


def test_the_book_is_its_sleeves_executed_on_their_own_calendars(worlds):
    clean, _ = worlds
    rk = report.risk_of(clean, list(clean.inputs))
    b = report.run_book(clean, rk, report.Choice("inverse_vol"))
    for name, i in clean.inputs.items():
        s = i.sleeve
        q = portfolio.asof(b.decided[name], s.sessions).fillna(0.0)
        held = q.shift(s.lag + 1).fillna(0.0)                     # decided lag + 1 sleeve sessions before
        unit = i.u.groupby("session")["total"].sum(min_count=1).reindex(s.sessions)
        credited = portfolio.to_book((held * unit).fillna(0.0), clean.days)
        assert np.allclose(b.parts[name]["gross"], credited, atol=1e-6)
    legs = [len(i.sleeve.legs) for i in clean.inputs.values()]
    assert (b.decided.abs().to_numpy() @ legs).max() <= report.limit(clean.book) * (1 + 1e-9)


def test_a_book_is_evaluated_from_the_first_session_its_first_decision_is_held(worlds):
    clean, _ = worlds
    rk = report.risk_of(clean, list(clean.inputs))
    b = report.run_book(clean, rk, report.Choice("inverse_vol"))
    lag = max(i.sleeve.lag for i in clean.inputs.values())
    held = rk.first + lag + 1                                     # decided at the close, executed lag sessions on
    assert b.run.start == clean.days[held]
    d = b.run.daily
    counted = np.zeros(len(clean.days), dtype=bool)
    for i in clean.inputs.values():
        k, *_ = positions.samples(i.state, clean.spec.treatment, i.sleeve.lag)
        counted |= portfolio.to_book(k.astype(int), clean.days).to_numpy() > 0
    assert counted[rk.first:held].any()                           # the treatment alone would count them
    assert not d["kept"].iloc[:held].any() and (d["gross_dv01"].iloc[:held] == 0).all()


def test_the_overlays_scale_multiplies_the_targets_of_the_close_it_is_decided_at(worlds):
    clean, _ = worlds
    rk = report.risk_of(clean, list(clean.inputs))
    ch = report.Choice("inverse_vol", overlay=True)
    tg = report.target_of(clean, rk, ch)
    k = next(i for i in range(rk.first + 20, len(clean.days)) if (tg.q.iloc[i - 1] != 0).any() and (tg.q.iloc[i] != 0).any())
    ones = pd.Series(1.0, index=clean.days)
    half = ones.where(np.arange(len(clean.days)) < k, 0.5)
    a, b = report.simulate(clean, rk, ch, tg, ones), report.simulate(clean, rk, ch, tg, half)
    pd.testing.assert_frame_equal(b.decided.iloc[:k], a.decided.iloc[:k])      # nothing before the close moves
    assert np.allclose(b.decided.iloc[k], 0.5 * tg.q.iloc[k])                  # halving is outside the band: it trades


def test_a_resize_is_traded_lag_sessions_after_another_sleeves_side_moved(worlds):
    clean, _ = worlds
    rk = report.risk_of(clean, list(clean.inputs))
    b = report.run_book(clean, rk, report.Choice("inverse_vol"))
    resized = 0
    for name, i in clean.inputs.items():
        s = i.sleeve
        daily, *_ = report._sleeve(clean, name, b.decided, b.run.start)
        others = pd.DataFrame({n: portfolio.asof(clean.g[n], s.sessions) for n in b.decided if n != name}).fillna(0.0)
        moved = (others.diff().fillna(0.0) != 0).any(axis=1).to_numpy()       # at the close another side moved
        own = portfolio.asof(b.decided[name], s.sessions).fillna(0.0).diff().fillna(0.0).to_numpy() != 0
        at = np.flatnonzero(daily["resize"].to_numpy() > 0)
        assert (at >= s.lag).all() and moved[at - s.lag].all() and own[at - s.lag].all()   # decided there, lag before
        later = np.r_[np.zeros(s.lag, dtype=bool), moved[:-s.lag]]
        assert (daily["rescale"].to_numpy()[later] == 0).all()             # such a trade is never a rescale
        resized += len(at)
    assert resized > 10


def test_the_summary_reads_the_net_pnl_on_the_counted_sessions(worlds):
    clean, _ = worlds
    rk = report.risk_of(clean, list(clean.inputs))
    b = report.run_book(clean, rk, report.Choice("inverse_vol"))
    cap = clean.book["book"]["capital"]
    s = report.summary(b, cap)
    x, d = report.net(b), report.kept(b)
    per = b.run.periods
    assert s["vol_pct"] == pytest.approx(x.std() * np.sqrt(per) / cap * 100, rel=1e-12)
    assert abs(d["gross"].std() / x.std() - 1) > 1e-3                          # the costs move it: the test can tell
    on = d["gross_dv01"] > 0
    assert s["active_vol_pct"] == pytest.approx(x[on].std() * np.sqrt(per) / cap * 100, rel=1e-12)
    assert s["active_vol_block_pct"] == pytest.approx(report.block_vol(x[on], per) / cap * 100, rel=1e-12)
    dd = portfolio.drawdown(x.cumsum(), cap).max() * 100                       # the worst drawdown is net's too
    assert s["max_dd_pct"] == pytest.approx(dd, rel=1e-12)
    assert abs(portfolio.drawdown(d["gross"].cumsum(), cap).max() * 100 / dd - 1) > 1e-3
    o = report.run_book(clean, rk, report.Choice("inverse_vol", overlay=True))
    half = o.scale.reindex(report.kept(o).index) < 1                          # the share of the counted sessions
    assert report.summary(o, cap)["halved"] == pytest.approx(half.mean(), rel=1e-12)
    assert abs((o.scale < 1).mean() - half.mean()) > 1e-3


def test_mean_variance_without_correlations_is_z_weighted_inverse_vol_on_the_book(worlds):
    clean, _ = worlds
    rk = report.risk_of(clean, list(clean.inputs))
    tg = report.target_of(clean, rk, report.DIAGONAL)
    sd, _ = rk.floored[True]
    cap = clean.book["portfolio"]["z_cap"]
    w = clean.g.to_numpy() * np.minimum(np.abs(clean.z.to_numpy()), cap) / sd
    on = ((clean.g != 0).sum(axis=1) > 1).to_numpy() & (np.arange(len(clean.days)) >= rk.first)
    q = tg.q.to_numpy()[on]
    w = np.where(clean.g.to_numpy()[on] != 0, w[on], 0.0)
    assert on.sum() > 50
    assert np.allclose(q / np.abs(q).sum(axis=1, keepdims=True), w / np.abs(w).sum(axis=1, keepdims=True), atol=1e-12)
    assert not tg.flipped.to_numpy().any()                                      # with no correlation, no hedge


# ---- the numbers the words read ----------------------------------------------------

def a_book(net, kept=None, periods=252.0):
    """A stand-in `Book` with daily net P&L `net` (all cost-free) and its counted sessions."""
    days = pd.bdate_range("2020-01-01", periods=len(net))
    kept = np.ones(len(net), dtype=bool) if kept is None else kept
    daily = pd.DataFrame({"gross": net, "cost": 0.0, "kept": kept}, index=days)
    return SimpleNamespace(run=SimpleNamespace(daily=daily, periods=periods))


def test_the_paired_se_is_jobson_korkie_with_memmels_correction():
    rng = np.random.default_rng(7)
    x = rng.normal(0.02, 1.0, 2000)
    y = 0.9 * x + 0.3 * rng.normal(-0.01, 1.0, 2000)
    d = report.difference(a_book(x), a_book(y))
    s1, s2, rho, n = x.mean() / x.std(ddof=1), y.mean() / y.std(ddof=1), np.corrcoef(x, y)[0, 1], 2000
    want = np.sqrt((2 * (1 - rho) + (s1 ** 2 + s2 ** 2 - 2 * s1 * s2 * rho ** 2) / 2) / n * 252)
    assert d["diff"] == pytest.approx((s1 - s2) * np.sqrt(252), rel=1e-12)
    assert d["se"] == pytest.approx(want, rel=1e-12) and d["rho"] == pytest.approx(rho)
    same = report.difference(a_book(x), a_book(2.0 * x))                     # scale-free: a book at double size
    assert abs(same["diff"]) < 1e-12 and same["se"] < 1e-6
    single = np.sqrt(252 / n)                                                 # far under a single Sharpe's SE
    assert d["se"] < 0.5 * single
    free = report.difference(a_book(x), a_book(rng.normal(0.0, 1.0, 2000)))  # independent: both SEs add
    assert free["se"] == pytest.approx(np.sqrt(2) * single, rel=0.05)
    k = np.arange(2000) < 1500                                                # over the sessions both count
    part = report.difference(a_book(x, k), a_book(y))
    assert part["sessions"] == 1500
    assert part["diff"] == pytest.approx(report.difference(a_book(x[:1500]), a_book(y[:1500]))["diff"], rel=1e-12)


def test_block_vol_is_the_sd_of_non_overlapping_sums():
    x = pd.Series([1.0, -1.0, 2.0, 0.0, 3.0, 1.0, 5.0])                       # blocks of 2: 0, 2, 4; the 5 dropped
    assert report.block_vol(x, 252, 2) == pytest.approx(np.std([0.0, 2.0, 4.0], ddof=1) * np.sqrt(126))
    assert np.isnan(report.block_vol(x.iloc[:3], 252, 2))                     # one block: no sd


def test_a_window_is_left_out_of_the_book_s_counted_pnl(worlds):
    clean, _ = worlds
    rk = report.risk_of(clean, list(clean.inputs))
    b = report.run_book(clean, rk, report.Choice("inverse_vol"))
    cap = clean.book["book"]["capital"]
    windows = {"ex_2022": ["2022-01-01", "2022-12-31"], "ex_1999": ["1999-01-01", "1999-12-31"]}
    [w] = report.without(b, windows, cap)                                     # a window with no session is skipped
    x = report.net(b)
    inside = (x.index >= T("2022-01-01")) & (x.index <= T("2022-12-31"))
    assert w["words"] == "2022" and w["sessions"] == inside.sum()
    assert w["inside_pct"] + w["outside_pct"] == pytest.approx(x.sum() / cap * 100)
    assert w["net_sr"] == pytest.approx(metrics.sharpe(x[~inside], b.run.periods), rel=1e-12)
    late = x.iloc[len(x) // 2:]
    keep = ~inside[len(x) // 2:]
    assert w["second_sessions"] == keep.sum()
    assert w["second_sr"] == pytest.approx(metrics.sharpe(late[keep], b.run.periods), rel=1e-12)
    assert sum(w["sleeves_inside_pct"].values()) == pytest.approx(w["inside_pct"])
    assert report.window_words(T("2021-12-01"), T("2023-08-31")) == "Dec 2021 to Aug 2023"


def test_never_released_needs_the_scale_below_one_from_the_first_trigger_on():
    days = pd.bdate_range("2024-01-01", periods=6)
    one = SimpleNamespace(episodes=[days[2]], scale=pd.Series([1, 1, 0.5, 0.5, 0.5, 0.5], index=days))
    twice = SimpleNamespace(episodes=[days[1], days[4]], scale=pd.Series([1, 0.5, 0.5, 1, 0.5, 0.5], index=days))
    assert report.never_released(one)
    assert not report.never_released(twice)                                   # released, then triggered again
    assert not report.never_released(SimpleNamespace(episodes=[], scale=pd.Series(1.0, index=days)))


# ---- the covariance's floor and the lag term's numbers -----------------------------

def test_the_floored_matrix_has_the_floored_sds_and_the_same_correlations():
    days = pd.bdate_range("2020-01-01", periods=400)
    rng = np.random.default_rng(11)
    x = rng.normal(0.0, 1.0, (400, 2))
    x[:, 1] = 0.5 * x[:, 0] + rng.normal(0.0, 1.0, 400)
    x[300:, 0] *= 0.05                                                         # the first sleeve goes quiet
    cov = risk.ewma_cov(pd.DataFrame(x, index=days, columns=["a", "b"]), 0.9, 1)
    for shrink in (True, False):
        m0 = cov.shrunk if shrink else cov.s
        sd, m = portfolio.floored(cov, shrink, 0.5)
        live = ~np.isnan(sd).any(axis=1)
        raw = np.sqrt(np.diagonal(m0, axis1=1, axis2=2))
        assert (sd[live] > raw[live] * (1 + 1e-9)).any()                     # the floor binds
        assert np.allclose(np.sqrt(np.diagonal(m[live], axis1=1, axis2=2)), sd[live], rtol=1e-12)
        assert np.allclose(risk.correlation(m[live]), risk.correlation(m0[live]), rtol=1e-12)


def test_lag_effect_counts_the_unweighted_forms_failures_on_a_hand_example():
    days = pd.bdate_range("2024-01-01", periods=3)
    s0 = np.array([np.eye(2), np.eye(2), [[1.0, 0.9], [0.9, 1.0]]])
    half = np.array([np.diag([-0.6, 0.1]), np.diag([-0.3, 0.0]), [[0.0, 0.08], [0.08, 0.0]]])   # (S1 + S1') / 2
    s = s0 + half
    cov = risk.Cov(days, ["a", "b"], s0, s, s, np.zeros(3), np.full(3, 10.0), np.full(3, 50))
    e = report.lag_effect(report.Risk(["a", "b"], cov, 0, {}), 0)
    # unweighted s0 + s1 + s1': diag(-0.2, 1.2) (a variance below zero), diag(0.4, 1), off-diagonal 1.06 (not PD)
    assert (e["sessions"], e["not_pd"], e["negative_variance"], e["bartlett_not_pd"]) == (3, 2, 1, 0)
    assert e["negative_by_sleeve"] == {"a": 1, "b": 0} and e["not_pd_without"] == {"a": 0, "b": 1}
    assert e["rho"]["a"] == pytest.approx(-0.3) and e["rho_min"]["a"] == pytest.approx(-0.6)
    assert e["size_max"]["a"] == pytest.approx(np.sqrt(1 / 0.4)) and e["size"]["b"] == pytest.approx(
        np.mean([np.sqrt(1 / 1.1), 1.0, 1.0]))
