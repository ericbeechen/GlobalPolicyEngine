"""Carry, roll and the rate move: the twelve sign cases worked by hand, and the three checks on synthetic legs.

Twelve ways to get a sign backwards (3 expressions x 2 currencies x 2
directions), each pinned to numbers that can be checked with a calculator.
Then the identity, the independent full revaluation (inside its bound, 0 for
a futures month; outside it when the carry sign is flipped or the
attribution's FX ratio disagrees with the valuation's spot; and blind, as it
must be, to a quote inverted at its source), and convergence: move the market
to the model and every sleeve with z > 0 gains, a fixed-window leg exactly the
de-meaned gap. Last, the breakeven's signs on a hand-worked inverted curve,
receiving and paying.
"""

from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from policypath import config
from policypath.strategy import carry, expression, instruments

T = pd.Timestamp
ZQ = config.currency("USD")["contracts"]["ZQ"]
FX = (1.25, 1.30)          # USD per GBP at p and t: fx = 1.04


def leg(kind, ccy, frame, spot=None, regimes=None, f=None, contract=None):
    spot = np.ones(len(frame)) if spot is None else np.asarray(spot, dtype=float)
    return instruments.Leg(kind, ccy, instruments.with_fx(frame, spot).assign(stale=False, fx_stale=False),
                           regimes, f, contract)


def zq_leg(days, strips, effective):
    """`strips`: one {month: price} per session; `effective`: session x k effective dates (k = 1 traded)."""
    prices = pd.DataFrame(strips, index=days)
    prices.columns = pd.PeriodIndex(prices.columns, freq="M")
    frame, regimes = instruments.futures_month(days, prices.sort_index(axis=1), effective, 1, 91,
                                               instruments.contract_dv01(ZQ))
    return leg("futures_month", "USD", frame, regimes=regimes, contract=ZQ)


def forward_leg(days, curves, start, end, in_force, spot=None):
    """`curves`: one {tenor in months: spot %} per session; the window [start, end) on every session."""
    curve = pd.DataFrame(curves, index=days)
    frame = instruments.curve_forward(days, curve, pd.Series(start, index=days), pd.Series(end, index=days), 91,
                                      365, pd.Series(in_force, index=days, dtype=float))
    return leg("curve_forward", "GBP", frame, spot)


def par_leg(days, curves, tenor, funding, basis, spot=None, ccy="USD"):
    """`curves`: one {maturity in years: par yield %} per session."""
    curve = pd.DataFrame(curves, index=days)
    frame = instruments.par_leg(days, curve, tenor, 2, 91, pd.Series(funding, index=days, dtype=float), basis)
    return leg("par_yield", ccy, frame, spot, f=2)


def sleeve(legs, z=None, lag=0):
    days = legs[0][2].frame.index
    z = pd.Series(0.0, index=days) if z is None else z
    component = pd.DataFrame({"value_bp": 0.0, "z": z, "window_mean_bp": 0.0, "window_sd_bp": 5.0, "dev_bp": 0.0},
                             index=days)
    return expression.Sleeve("test", "test", (), component, legs, lag, component["dev_bp"])


def credited(legs, direction):
    """carry, roll, rate and pnl credited to the last session, for a position of `direction` held from the one before."""
    s = sleeve(legs)
    _, sessions = expression.run(s, pd.Series(float(direction), index=s.sessions))
    return sessions.iloc[-1][["carry", "roll", "rate", "pnl"]].to_numpy(dtype=float)


# ---- the twelve sign cases ---------------------------------------------------------

P, S = T("2026-01-05"), T("2026-01-06")      # one day apart
WEEK = pd.DatetimeIndex([T("2025-01-06"), T("2025-01-13")])
A_4_2, A_5_10 = 1.9038643, 7.7945811          # par duration (1 - (1 + y/2)^(-2T)) / y at 4%/2y and 5%/10y
A_46_2, A_4_10 = 1.8900847, 8.1757167         # at 4.6%/2y and 4%/10y


def usd_outright():
    """Receive the June 2026 ZQ on an upward strip (May 96.00, Jun 95.90, Jul 95.80); it rallies 5bp.

    Mid-May to mid-June is 30.5 days, so a day's roll down the strip is 10bp / 30.5
    = +0.328bp; the rest of the 5bp is the rate move. No carry.
    """
    days = pd.DatetimeIndex([P, S])
    before = {"2026-05": 96.00, "2026-06": 95.90, "2026-07": 95.80}
    effective = pd.DataFrame({1: T("2026-05-15"), 2: T("2026-07-29")}, index=days)
    return [("USD outright", 1, zq_leg(days, [before, {m: p + 0.05 for m, p in before.items()}], effective))]


def gbp_outright():
    """Receive the 1y1y OIS forward on an inverted curve (1y spot 5.0%, 2y 4.5%: the forward is 4.0%); -5bp.

    Rolled a day it reads 4 + 1/365 %, so roll = -100/365 bp; the whole curve falls
    5bp; both at fx = 1.30 / 1.25 = 1.04.
    """
    p, t = T("2025-01-01"), T("2025-01-02")
    before = {1: 5.0, 12: 5.0, 24: 4.5, 36: 4.3}
    after = {m: s - 0.05 for m, s in before.items()}
    return [("GBP outright", 1, forward_leg(pd.DatetimeIndex([p, t]), [before, after], p + pd.Timedelta(days=365),
                                            p + pd.Timedelta(days=730), 5.0, FX))]


def usd_curve():
    """Receive 2y, pay 10y on an upward Treasury curve, unchanged for a week, funded at 2% (Act/360).

    2y rolls 1%/yr x 7/365 = +1.918bp, 10y 0.1%/yr x 7/365 (paid: -0.192);
    carry 100 (4 x 7/365 - 2 x 7/360) / 1.9039 - 100 (5 x 7/365 - 2 x 7/360) / 7.7946.
    """
    curve = {1: 3.0, 2: 4.0, 3: 4.2, 5: 4.5, 7: 4.7, 10: 5.0}
    return [("USD 2y", 1, par_leg(WEEK, [curve, curve], 2, 2.0, 360)),
            ("USD 10y", -1, par_leg(WEEK, [curve, curve], 10, 2.0, 360))]


def gbp_curve():
    """Receive 2y, pay 10y on an inverted gilt curve funded at 5.25% (Act/365): right and bleeding.

    2y rolls up 0.4%/yr x 7/365 (-0.767bp), 10y 0.1%/yr (paid: +0.192); carry
    100 (4.6 - 5.25) 7/365 / 1.8901 - 100 (4 - 5.25) 7/365 / 8.1757; all x 1.04.
    """
    curve = {1: 5.0, 1.5: 4.8, 2: 4.6, 9.5: 4.05, 10: 4.0}
    return [("GBP 2y", 1, par_leg(WEEK, [curve, curve], 2, 5.25, 365, FX, "GBP")),
            ("GBP 10y", -1, par_leg(WEEK, [curve, curve], 10, 5.25, 365, FX, "GBP"))]


FLAT = {1: 4.0, 2: 4.0, 3: 4.0, 10: 4.0}
DAY = pd.DatetimeIndex([T("2025-01-06"), T("2025-01-07")])


def cross(first):
    """Receive `first`'s 2y, pay the other's; both flat at 4%, funded at 4%; `first`'s 2y falls 10bp.

    The GBP leg's P&L converts at S_t: 10bp x 1.04 when GBP is received. The USD
    leg's funding is Act/360, so it costs 100 (4/365 - 4/360) / 1.9039 = -0.008bp
    a day to receive (paid: +0.008).
    """
    lower = {m: y - 0.10 for m, y in FLAT.items()}
    gbp = [FLAT, lower] if first == "GBP" else [FLAT, FLAT]
    usd = [FLAT, lower] if first == "USD" else [FLAT, FLAT]
    legs = {"GBP": par_leg(DAY, gbp, 2, 4.0, 365, FX, "GBP"), "USD": par_leg(DAY, usd, 2, 4.0, 360)}
    second = "USD" if first == "GBP" else "GBP"
    return [(f"{first} 2y", 1, legs[first]), (f"{second} 2y", -1, legs[second])]


USD_CARRY = 100 * (4 / 365 - 4 / 360) / A_4_2
CASES = {
    # (carry, roll, rate) for a receive, bp
    "USD outright": (usd_outright, (0.0, 10 / 30.5, 5 - 10 / 30.5)),
    "GBP outright": (gbp_outright, (0.0, -100 / 365 * 1.04, 5 * 1.04)),
    "USD 2s10s": (usd_curve, (100 * (4 * 7 / 365 - 2 * 7 / 360) / A_4_2 - 100 * (5 * 7 / 365 - 2 * 7 / 360) / A_5_10,
                              100 * 7 / 365 - 10 * 7 / 365, 0.0)),
    "GBP 2s10s": (gbp_curve, ((100 * -0.65 * 7 / 365 / A_46_2 - 100 * -1.25 * 7 / 365 / A_4_10) * 1.04,
                              (-40 * 7 / 365 + 10 * 7 / 365) * 1.04, 0.0)),
    "GBP - USD 2y": (lambda: cross("GBP"), (-USD_CARRY, 0.0, 10 * 1.04)),
    "USD - GBP 2y": (lambda: cross("USD"), (USD_CARRY, 0.0, 10.0)),
}


@pytest.mark.parametrize("direction", [1, -1], ids=["receive", "pay"])
@pytest.mark.parametrize("case", list(CASES))
def test_the_twelve_sign_cases(case, direction):
    build, (c, r, m) = CASES[case]
    got = credited(build(), direction)
    assert got == pytest.approx(np.array([c, r, m, c + r + m]) * direction, rel=1e-6, abs=1e-12)


def test_the_worked_cases_say_what_the_plan_should_have():
    """Receiving on an upward curve earns roll and carry; on an inverted one it bleeds both; paying there earns both."""
    up, down = credited(usd_curve(), 1), credited(gbp_curve(), 1)
    assert up[0] > 0 and up[1] > 0
    assert down[0] < 0 and down[1] < 0
    assert (credited(gbp_curve(), -1)[:2] > 0).all()


def test_a_long_zq_contract_makes_its_dv01_on_a_one_bp_fall():
    dv01 = instruments.contract_dv01(ZQ)
    assert dv01 == pytest.approx(5_000_000 * 30 / 360 * 1e-4) and round(dv01, 2) == 41.67
    days = pd.DatetimeIndex([P, S])
    effective = pd.DataFrame({1: T("2026-05-15"), 2: T("2026-07-29")}, index=days)
    one = zq_leg(days, [{"2026-06": 95.90}, {"2026-06": 95.91}], effective)
    assert instruments.native(one, dv01).iloc[-1] == pytest.approx(1.0)          # a q of one DV01 is one contract
    full, _ = carry.revaluation(one)
    assert full.iloc[-1] * dv01 == pytest.approx(dv01)                         # +$41.67 for the 1bp fall
    assert credited([("USD outright", 1, one)], 1)[3] == pytest.approx(1.0)


def test_the_gbp_leg_scales_with_spot_at_t():
    legs = cross("GBP")
    gbp = carry.attribution(legs[0][2]).iloc[-1]
    assert gbp["rate"] == pytest.approx(10 * FX[1] / FX[0])


# ---- identity and full revaluation on random curves --------------------------------

N = 80
RNG = np.random.default_rng(7)
DAYS = pd.bdate_range("2024-01-02", periods=N)
SPOT = 1.27 * np.exp(np.cumsum(RNG.normal(0, 0.006, N)))


def random_par(tenor, basis, spot=None, ccy="USD"):
    base = np.array([3.0, 3.6, 3.8, 4.0, 4.2, 4.5])
    level = np.cumsum(RNG.normal(0, 0.05, N))[:, None]
    tilt = np.cumsum(RNG.normal(0, 0.02, N))[:, None] * np.linspace(-1, 1, 6)
    curves = [dict(zip([1, 2, 3, 5, 7, 10], row)) for row in base + level + tilt]
    return par_leg(DAYS, curves, tenor, np.full(N, 1.0), basis, spot, ccy)


def random_forward(spot):
    months = np.array([1, 3, 6, 12, 18, 24, 36])
    level = 4.0 + np.cumsum(RNG.normal(0, 0.05, N))
    curves = [dict(zip(months, lvl - 0.2 * months / 12)) for lvl in level]
    start, end = pd.Series(T("2024-09-18"), index=DAYS), pd.Series(T("2024-11-07"), index=DAYS)
    frame = instruments.curve_forward(DAYS, pd.DataFrame(curves, index=DAYS), start, end, 91, 365,
                                      pd.Series(level, index=DAYS))
    return leg("curve_forward", "GBP", frame, spot)


def random_zq():
    months = pd.period_range("2024-01", periods=24, freq="M")
    level = 95.0 + np.cumsum(RNG.normal(0, 0.04, N))
    strips = [dict(zip(months.astype(str), lvl - 0.02 * np.arange(24))) for lvl in level]
    effective = pd.DataFrame({1: T("2024-09-18"), 2: T("2024-11-07")}, index=DAYS)
    return zq_leg(DAYS, strips, effective)


LEGS = {"zq": random_zq(), "forward": random_forward(SPOT), "usd 2y": random_par(2, 360), "usd 10y": random_par(10, 360),
        "gbp 2y": random_par(2, 365, SPOT, "GBP"), "gbp 10y": random_par(10, 365, SPOT, "GBP")}


@pytest.mark.parametrize("name", list(LEGS))
def test_the_identity_holds_on_every_session(name):
    one = LEGS[name]
    parts = carry.attribution(one)
    residual = carry.identity(one, parts).iloc[1:]
    assert residual.notna().all() and residual.abs().max() < carry.TOL_BP


@pytest.mark.parametrize("name", list(LEGS))
def test_the_full_revaluation_is_within_its_bound(name):
    one = LEGS[name]
    full, bound = carry.revaluation(one)
    residual = (full - carry.attribution(one)["total"]).iloc[1:]
    assert (residual.abs() <= bound.iloc[1:] + carry.TOL_BP).all()


@pytest.mark.parametrize("name", ["usd 2y", "usd 10y", "gbp 2y", "gbp 10y"])
def test_a_flipped_carry_sign_fails_the_revaluation(name):
    one = LEGS[name]
    full, bound = carry.revaluation(one)
    parts = carry.attribution(one)
    flipped = parts["total"] - 2 * parts["carry"]
    outside = ((full - flipped).abs() > bound + carry.TOL_BP).iloc[1:]
    assert outside.mean() > 0.9


@pytest.mark.parametrize("name", ["forward", "gbp 2y", "gbp 10y"])
def test_an_attribution_fx_that_disagrees_with_the_valuation_fails_it(name):
    one = LEGS[name]
    full, bound = carry.revaluation(one)
    inverted = replace(one, frame=one.frame.assign(fx=one.frame["spot_p"] / one.frame["spot"]))
    outside = ((full - carry.attribution(inverted)["total"]).abs() > bound + carry.TOL_BP).iloc[1:]
    assert outside.mean() > 0.6                  # most: a session with little move and little FX change stays inside


@pytest.mark.parametrize("name", ["forward", "gbp 2y", "gbp 10y"])
def test_a_quote_inverted_at_its_source_is_invisible_to_the_revaluation(name):
    """Both sides value the leg at the same spot, so check 2 cannot see it: `instruments.plausible_spot` does."""
    one = LEGS[name]
    frame = instruments.with_fx(one.frame.drop(columns=["spot", "spot_p", "fx"]), 1.0 / one.frame["spot"].to_numpy())
    upside_down = replace(one, frame=frame)
    full, bound = carry.revaluation(upside_down)
    total = carry.attribution(upside_down)["total"]
    assert ((full - total).abs() <= bound + carry.TOL_BP).iloc[1:].all()
    assert not np.allclose(total.iloc[1:], carry.attribution(one)["total"].iloc[1:])    # yet the P&L is wrong


def test_a_futures_month_revalues_with_no_slack():
    _, bound = carry.revaluation(LEGS["zq"])
    assert (bound == 0).all()


# ---- convergence: the market moves to the model -------------------------------------

THREE = pd.bdate_range("2025-03-03", periods=3)     # z read on the first, held from the second into the third
Z0, DEV = 1.6, 12.0                                 # z > 0: the market prices DEV bp more than its trailing mean


def run_z(legs):
    """P&L credited to the last session under the linear rule, one session's execution lag."""
    s = sleeve(legs, pd.Series([Z0, 0.0, 0.0], index=THREE), lag=1)
    _, sessions = expression.run(s, expression.linear(s))
    return sessions["pnl"].iloc[-1]


def converging(before, after):
    """The market's rate on the three sessions: `before` on the first two, `after` (on the model) on the third."""
    return [before, before, after]


def test_an_outright_forward_gains_exactly_the_de_meaned_gap():
    start, end = T("2025-09-18"), T("2025-11-06")
    curves = []
    for day, fwd in zip(THREE, converging(4.0 + DEV / 100, 4.0)):
        a, b = (start - day).days / 365 * 12, (end - day).days / 365 * 12    # nodes at the window's edges
        la, lb = 0.035 * a / 12, 0.035 * a / 12 + fwd / 100 * (b - a) / 12
        curves.append({1: 3.5, a: la / (a / 12) * 100, b: lb / (b / 12) * 100, 36: 4.0})
    one = forward_leg(THREE, curves, start, end, 3.5)
    assert run_z([("GBP outright", 1, one)]) == pytest.approx(Z0 * DEV, abs=1e-9)


def test_an_outright_zq_month_gains_exactly_the_de_meaned_gap():
    effective = pd.DataFrame({1: T("2025-08-01"), 2: T("2025-10-30")}, index=THREE)
    strips = [{"2025-07": 96.0, "2025-08": 100 - y, "2025-09": 96.0} for y in converging(4.0 + DEV / 100, 4.0)]
    assert run_z([("USD outright", 1, zq_leg(THREE, strips, effective))]) == pytest.approx(Z0 * DEV, abs=1e-9)


def flat_par(ccy, basis, moves, spot=None):
    """Flat par curves at 4% with funding set so carry is 0, the front (<= 2y) and back moving by `moves` bp."""
    front, back = moves
    curves = [{m: 4.0 - (f if m <= 2 else b) / 100 for m in (1, 2, 3, 7, 10)}
              for f, b in converging((0.0, 0.0), (front, back))]
    funding = 4.0 * basis / 365
    return {t: par_leg(THREE, curves, t, funding, basis, spot, ccy) for t in (2, 10)}


@pytest.mark.parametrize("ccy, basis", [("USD", 360), ("GBP", 365)])
def test_a_curve_sleeve_gains_when_the_front_converges(ccy, basis):
    legs = flat_par(ccy, basis, (DEV, 0.2 * DEV), None if ccy == "USD" else [FX[0]] * 3)
    pnl = run_z([(f"{ccy} 2y", 1, legs[2]), (f"{ccy} 10y", -1, legs[10])])
    assert pnl > 0 and pnl == pytest.approx(Z0 * 0.8 * DEV, rel=1e-9)


def test_a_cross_sleeve_gains_when_the_first_currency_converges():
    first = flat_par("GBP", 365, (DEV, DEV), [FX[0]] * 3)[2]
    second = flat_par("USD", 360, (0.0, 0.0))[2]
    pnl = run_z([("GBP 2y", 1, first), ("USD 2y", -1, second)])
    assert pnl > 0 and pnl == pytest.approx(Z0 * DEV, rel=1e-9)


# ---- the breakeven ------------------------------------------------------------------

def test_carry_and_roll_ahead_on_a_frozen_curve():
    """91 days down an upward 1y-2y segment (1%/yr) funded at 2%: roll 100 x 91/365, carry over 91 days."""
    one = par_leg(WEEK, [{1: 3.0, 2: 4.0, 10: 5.0}] * 2, 2, 2.0, 360)
    c, r = carry.carry_roll_ahead(one, 91)
    assert r.iloc[0] == pytest.approx(100 * 91 / 365)
    assert c.iloc[0] == pytest.approx(100 * (4 * 91 / 365 - 2 * 91 / 360) / A_4_2, rel=1e-7)


@pytest.mark.parametrize("z", [1.5, -1.5], ids=["receive", "pay"])
def test_ahead_on_an_inverted_curve_receiving_bleeds_and_paying_earns(z):
    """A 2y gilt at 4.6% on an inverted curve (1.5y 4.8%), funded at 5.25%, de-meaned gap 7.5 x z / 1.5.

    91 days on the curve reads 4.6 + 0.2 x (91/365) / 0.5 = 4.6997%: a receive rolls -9.97bp and carries
    100 (4.6 - 5.25) 91/365 / 1.8901 = -8.57bp. Either side's edge is +7.5bp; a pay earns both.
    """
    curve = {1: 5.0, 1.5: 4.8, 2: 4.6, 9.5: 4.05, 10: 4.0}
    one = par_leg(WEEK, [curve, curve], 2, 5.25, 365, ccy="GBP")
    roll, carry_ = -0.2 * (91 / 365) / 0.5 * 100, 100 * (4.6 - 5.25) * 91 / 365 / A_46_2
    comp = pd.DataFrame({"value_bp": 3 * z, "z": z, "window_mean_bp": -2 * z, "window_sd_bp": 5.0, "dev_bp": 5 * z},
                        index=WEEK)
    s = expression.Sleeve("GBP 2y", "curve", ("GBP",), comp, [("GBP 2y", 1, one)], 1, comp["dev_bp"])
    a = expression.ahead(s, 91, 250).iloc[0]
    side = np.sign(z)
    assert a["side"] == side and a["edge_bp"] == pytest.approx(7.5)
    assert a["carry_h_bp"] == pytest.approx(side * carry_, rel=1e-6)
    assert a["roll_h_bp"] == pytest.approx(side * roll, rel=1e-6)
    assert a["cr_h_bp"] == pytest.approx(side * (carry_ + roll), rel=1e-6)
    assert bool(a["bleeding"]) == (z > 0)


def test_bleed_and_the_break_even_closure():
    edge = pd.Series([20.0, 20.0, 10.0, 5.0])
    cr = pd.Series([-5.0, 3.0, -10.0, -10.0])
    out = carry.bleed(edge, cr, pd.Series([0.3, 0.3, 0.3, 0.3]))
    assert out["bleeding"].tolist() == [True, False, True, True]
    assert out["phi_star"].tolist() == pytest.approx([5 / 25, 0.0, 0.5, 10 / 15])
    assert out["expected_bp"].tolist() == pytest.approx([0.3 * 20 - 0.7 * 5, 0.3 * 20 + 0.7 * 3, 3 - 7, 1.5 - 7])
    assert out["pays"].tolist() == [True, True, False, False]
    # does not pay exactly when the break-even closure is above the historical one
    assert ((out["phi_star"] > 0.3) == ~out["pays"].astype(bool)).all()


def test_closure_is_expanding_and_uses_only_pairs_already_closed():
    """Every gap is 0.6 of itself 30 days on, so each closes 40%: phi = 0.4 once 50 pairs have closed."""
    days = pd.date_range("2020-01-01", periods=200, freq="D")
    base = np.random.default_rng(1).normal(0, 10, 30)
    value = pd.Series([base[i % 30] * 0.6 ** (i // 30) for i in range(200)], index=days)
    phi = carry.closure(value, value, 30, 50)
    assert phi.iloc[:79].isna().all()                       # the 50th pair (day 49, day 79) closes on day 79
    assert phi.iloc[79:].to_numpy() == pytest.approx(0.4, abs=1e-12)
    cut = days[120]
    later = value.where(value.index <= cut, 99.0)
    assert carry.closure(later, later, 30, 50)[:cut].equals(phi[:cut])
