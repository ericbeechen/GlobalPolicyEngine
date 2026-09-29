"""The rules, the ELB treatments and the sizing, on hand-made z paths.

Checks: hysteresis enters at |z| >= enter, exits at z x side <= exit, reverses
in one session, and treats a missing z as flat; the carry gate blocks entries
and never exits; under flat the state zeroes the position and the hysteresis
starts flat on leaving it, the close into the state and the re-open out of it
are charged, and hold trades nothing at the boundary; each treatment's
sessions are the ones the spec names; vol-scaled sizing reads only the lagged
sigma and re-scales only outside the no-trade band.
"""

from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from policypath import config
from policypath.strategy import costs, positions, risk

BOOK = config.strategy()
DAYS = pd.bdate_range("2024-01-01", periods=12)


def z_of(*values):
    return pd.Series(values, index=DAYS[:len(values)], dtype=float)


def test_hysteresis_enters_holds_and_exits_at_its_thresholds():
    z = z_of(0.5, 1.0, 0.4, 0.01, 0.0, -0.99, -1.0, -0.2, 0.0)
    got = positions.hysteresis(z, 1.0, 0.0).tolist()
    assert got == [0, 1, 1, 1, 0, 0, -1, -1, 0]                 # at enter it enters; at exit it exits


def test_hysteresis_reverses_in_one_session_and_only_past_enter():
    z = z_of(1.5, -1.2, -0.5, 1.1, -0.8, 0.3)
    assert positions.hysteresis(z, 1.0, 0.0).tolist() == [1, -1, -1, 1, 0, 0]
    # an exit above zero: out of a receive at 0.25, and the reversal still needs |z| >= enter
    z = z_of(1.2, 0.3, 0.25, -1.3)
    assert positions.hysteresis(z, 1.0, 0.25).tolist() == [1, 1, 0, -1]
    # an exit below zero holds through the mean
    z = z_of(1.2, -0.3, -0.5, -1.0)
    assert positions.hysteresis(z, 1.0, -0.5).tolist() == [1, 1, 0, -1]


def test_a_missing_z_is_flat_and_the_next_entry_needs_the_threshold():
    z = z_of(1.4, np.nan, 0.8, 1.0, np.nan, np.nan, -2.0)
    assert positions.hysteresis(z, 1.0, 0.0).tolist() == [1, 0, 0, 1, 0, 0, -1]
    assert positions.linear(z).tolist() == [1.4, 0, 0.8, 1.0, 0, 0, -2.0]


def test_the_carry_gate_blocks_entries_not_exits():
    z = z_of(1.5, 1.6, 1.2, -1.4, -1.5, 0.2)
    allow = pd.Series([False, True, True, False, True, True], index=z.index)
    assert positions.hysteresis(z, 1.0, 0.0, allow).tolist() == [0, 1, 1, 0, -1, 0]
    ahead = pd.DataFrame({"pays": pd.array([True, False, pd.NA], dtype="boolean")}, index=DAYS[:3])
    assert positions.pays(ahead).tolist() == [True, False, True]     # before phi_h is known, an entry stands


def test_the_elb_state_of_a_cross_sleeve_is_either_currencys():
    days = DAYS[:6]
    model = {c: pd.DataFrame({"session": days, "r0": [0.1, 0.1, 0.1, 1.0, 0.1, 0.1], "elb": 0.1,
                              "at_elb": at}) for c, at in
             (("A", [True, True, False, True, False, False]), ("B", [False, False, False, False, True, False]))}
    sleeve = SimpleNamespace(sessions=days, ccys=("A", "B"))
    assert positions.elb(sleeve, model).tolist() == [True, True, False, False, True, False]
    one = SimpleNamespace(sessions=days, ccys=("A",))
    assert positions.elb(one, model).tolist() == [True, True, False, False, False, False]    # r0 off the floor


def test_flat_zeroes_the_state_and_starts_flat_on_leaving_it():
    z = z_of(1.5, 1.5, 1.5, 0.8, 0.8, 1.2)
    state = pd.Series([False, True, True, False, False, False], index=z.index)
    flat = positions.decide(z, "hysteresis", 1.0, 0.0, state, "flat")
    hold = positions.decide(z, "hysteresis", 1.0, 0.0, state, "hold")
    assert flat.tolist() == [1, 0, 0, 0, 0, 1]                  # 0.8 after the state is not an entry
    assert hold.tolist() == [1, 1, 1, 1, 1, 1]
    assert positions.decide(z, "hysteresis", 1.0, 0.0, state, "exclude").equals(hold)
    assert positions.decide(z, "linear", 1.0, 0.0, state, "flat").tolist() == [1.5, 0, 0, 0.8, 0.8, 1.2]
    with pytest.raises(ValueError):
        positions.decide(z, "hysteresis", 1.0, 0.0, state, "ignore")


def test_each_treatment_counts_the_sessions_it_says():
    state = pd.Series([False, False, True, True, True, False, False, False], index=DAYS[:8])
    held_in_state = [False, False, False, False, True, True, True, False]   # decided two sessions before (lag 1)
    put_in_state = [False, False, False, True, True, True, False, False]    # decided one session before
    both = [True, True, True, True, False, False, True, True]
    for name, want, pnl, cost in (("hold", [True] * 8, [True] * 8, [True] * 8),
                                  ("exclude", both, [not x for x in held_in_state], [not x for x in put_in_state]),
                                  ("flat", both, [True] * 8, [True] * 8)):
        kept, p, c, line = positions.samples(state, name, lag=1)
        assert kept.tolist() == want, name
        assert p.tolist() == pnl and c.tolist() == cost, name
        assert line.tolist() == held_in_state


def leg_frame(days, q, ids, label="L"):
    """`expression.run`'s per-leg frame, by hand: the position put on at each close, in the instrument selected."""
    ids = pd.Series(ids, index=days, dtype=object)
    return pd.DataFrame({"session": days, "leg": label, "q": q, "selected": ids.to_numpy(),
                         "held_id": ids.shift(1).to_numpy()})


def test_flat_charges_the_close_into_the_state_and_the_reopen_and_hold_charges_neither():
    days = DAYS[:8]
    z = pd.Series([1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5], index=days)
    state = pd.Series([False, False, True, True, False, False, False, False], index=days)
    sleeve = SimpleNamespace(sessions=days, lag=1)
    rate = [costs.LegCost("L", "X", "outright", 0.5, False, None)]
    charged = {}
    for name in ("flat", "hold"):
        s = positions.decide(z, "hysteresis", 1.0, 0.0, state, name)
        q = s.shift(1).fillna(0.0).to_numpy()
        charged[name] = costs.charge(costs.traded(sleeve, leg_frame(days, q, ["M"] * 8), s, rate),
                                     costs.configured(rate))
    assert charged["hold"].tolist() == [0, 0.5, 0, 0, 0, 0, 0, 0]                 # the one entry
    assert charged["flat"].tolist() == [0, 0.5, 0, 0.5, 0, 0.5, 0, 0]             # entry, close at t0 + 1, re-open
    kept, *_ = positions.samples(state, "flat", lag=1)
    assert kept[charged["flat"] > 0].all()                                        # both boundary trades count


def test_vol_scaled_sizes_by_the_target_over_sigma_with_a_no_trade_band():
    s = z_of(0, 1, 1, 1, 1, 1, -1, -1, 0, 1)
    sigma = z_of(2, 2, 2, 2.1, 2.3, 4, 4, 1, 1, 1)
    q = positions.vol_scaled(s, sigma, target=4.0, band=0.1)
    # entry at 4/2; 4/2.1 is inside the band; 4/2.3 is 13% off, re-set; 4/4 re-set; a new side re-sets; flat is 0
    want = [0, 2, 2, 2, 4 / 2.3, 1, -1, -4, 0, 4]
    assert q.tolist() == pytest.approx(want)
    assert positions.size(s, "unit").equals(s.rename("decided"))
    missing = positions.vol_scaled(s, sigma.where(sigma.index != DAYS[2]), 4.0, 0.1)
    assert missing.iloc[2] == 0 and missing.iloc[3] == pytest.approx(4 / 2.1)      # no sigma, flat; then a new entry


def test_vol_scaled_reads_only_sigma_known_at_the_decision():
    rng = np.random.default_rng(5)
    days = pd.bdate_range("2021-01-01", periods=300)
    z = pd.Series(np.cumsum(rng.normal(0, 0.3, len(days))), index=days)
    pnl = pd.Series(rng.normal(0, 1, len(days)), index=days)
    r = BOOK["risk"]
    target = positions.target_sigma(BOOK)

    def decided(x):
        sigma = risk.sigma_floor(risk.ewma_sigma(x, r["ewma_lambda"]), r["sigma_floor"])
        return positions.size(positions.hysteresis(z, 1.0, 0.0), "vol_scaled", sigma, target, r["no_trade_band"])

    clean = decided(pnl)
    cut = 200
    poisoned = pnl.copy()
    poisoned.iloc[cut:] = 50.0                        # the P&L credited on the cut session and after
    got = decided(poisoned)
    pd.testing.assert_series_equal(got.iloc[:cut + 1], clean.iloc[:cut + 1])
    assert (clean.iloc[:cut] != 0).sum() > 50
    assert not got.iloc[cut + 1:].equals(clean.iloc[cut + 1:])
