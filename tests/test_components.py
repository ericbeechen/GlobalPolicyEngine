"""The signal components: level, the orthogonalised slope, and the cross-country differential."""

import numpy as np
import pandas as pd
import pytest
from policypath.signal import components, cross
from policypath.signal.gap import zscore

SPEC = {"window": "730D", "min_periods": 250, "sd_floor_bp": 5}
DAYS = pd.bdate_range("2018-01-01", periods=900)


def signal(seed=0, loading=1.05):
    """A long signal frame whose gap at meeting k is (0.4 + 0.15 k) x a level walk plus noise: slope = 1.05 x level."""
    rng = np.random.default_rng(seed)
    walk = np.cumsum(rng.normal(0, 3, len(DAYS)))
    gap = pd.DataFrame({k: walk * (0.4 + loading / 7 * (k - 1)) + rng.normal(0, 2, len(DAYS)) for k in range(1, 9)},
                       index=DAYS)
    z, mean, sd = zscore(gap, SPEC)
    parts = {"gap_bp": gap, "z": z, "window_mean_bp": mean, "window_sd_bp": sd}
    out = pd.concat({n: f.stack(future_stack=True) for n, f in parts.items()}, axis=1)
    return out.rename_axis(["session", "k"]).reset_index()


def test_level_is_the_panels_horizon_with_its_de_meaned_gap():
    sig = signal()
    level = components.level(sig, 4)
    four = sig[sig["k"] == 4].set_index("session")
    assert level["z"].equals(four["z"]) and level["value_bp"].equals(four["gap_bp"])
    live = level.dropna()
    assert (live["dev_bp"] - live["z"] * live["window_sd_bp"]).abs().max() < 1e-9


def test_the_slope_is_orthogonalised_against_the_level():
    sig = signal()
    slope = components.slope(sig, [1, 8], 4, SPEC)
    gap = sig.pivot(index="session", columns="k", values="gap_bp")
    assert slope["raw_bp"].equals(gap[8] - gap[1])
    assert slope["beta"].dropna().median() == pytest.approx(1.05 / (0.4 + 1.05 * 3 / 7), rel=0.05)
    both = pd.concat([components.level(sig, 4)["z"], slope["z"], slope["raw_z"]], axis=1, keys=["l", "s", "r"]).dropna()
    assert both["l"].corr(both["r"]) > 0.9 > 0.5 > abs(both["l"].corr(both["s"]))
    assert slope["z"].first_valid_index() > slope["beta"].first_valid_index()
    raw = components.slope(sig, [1, 8], 4, SPEC, orthogonal=False)
    assert raw["value_bp"].equals(raw["raw_bp"]) and raw["z"].equals(raw["raw_z"])


def test_the_slopes_beta_and_z_use_only_earlier_sessions():
    sig = signal()
    cut = DAYS[700]
    poisoned = sig.copy()
    poisoned.loc[poisoned["session"] > cut, ["gap_bp", "z", "window_mean_bp", "window_sd_bp"]] = 99.0
    a, b = components.slope(sig, [1, 8], 4, SPEC), components.slope(poisoned, [1, 8], 4, SPEC)
    assert a.loc[:cut].equals(b.loc[:cut])


def test_the_slopes_beta_at_t_leaves_t_out():
    """Strictly before t: poisoning the gap from t on leaves beta at t as it was."""
    sig = signal()
    cut = DAYS[700]
    poisoned = sig.copy()
    poisoned.loc[poisoned["session"] >= cut, "gap_bp"] = 99.0
    a, b = components.slope(sig, [1, 8], 4, SPEC), components.slope(poisoned, [1, 8], 4, SPEC)
    assert a["beta"].loc[:cut].equals(b["beta"].loc[:cut])


def test_the_differential_is_the_cross_signal_at_the_horizon_and_needs_one_horizon():
    first, second = signal(1), signal(2)
    d = components.differential(first, second, 4, 4, SPEC)
    want = cross.differential(first, second, SPEC)
    want = want[want["k"] == 4].set_index("session")
    assert d["value_bp"].equals(want["diff_bp"]) and d["z"].equals(want["z"])
    with pytest.raises(ValueError, match="horizons differ"):
        components.differential(first, second, 4, 3, SPEC)
