"""The rule's two coefficients estimated on expanding windows: the robustness check on imposing them (R1).

Author-adjacent: the imposed rule is the model, and this is what the data would
have said instead, in real time. It keeps the imposed rule's form and inertia
and estimates only a (inflation gap) and b (unemployment gap).

Partial adjustment, as the inertial rule has it. At quarter ends,
R_q = rho R_(q-1) + (1 - rho) R*_q, so the notional rate the committee moved
toward over quarter q is

    R*_impl = (R_q - rho R_(q-1)) / (1 - rho)

with rho the config's ``rule.inertia`` (per quarter: 0.85), and

    y = R*_impl - r* - pi = a (pi - pi*) + b (u* - u),   X = [pi - pi*, -(u - u*)]

a regression with no intercept. R is the policy rate in force at the quarter's
last session; pi, u - u*, r* and pi* are what the model had on that session, so
every row is real time. On session D the sample is every quarter strictly
before D's own where neither q nor q - 1 is censored. A quarter end is
censored in the ELB state (`regimes.elb_state`: the rate on its floor and the
rule below it, so the rate says nothing about a or b), and, with
``rule.estimate.drop_cuts_to_floor``, where the rate reached its floor by a cut
that quarter: the committee may have wanted lower, and the inversion gives a
goal below the floor, which the rule's floored goal can never be (USD 2020Q1,
the emergency cuts, R*_impl -8.4, with the nowcast still on February's macro).
A quarter that ends on the floor with the rule above it is kept: the committee
chose to wait there, and that is reaction-function information.
1 / (1 - rho) = 6.7, so a 25bp move the rule did not call for is 1.7pp of
R*_impl: the raw fit is noisy, and it is kept as the evidence.

Shrunk toward the imposed values (ridge with a prior mean):

    beta = (X'X + K)^-1 (X'y + K beta0),   K = n0 diag(mean(X^2))

beta0 the config's ``rule.coefficients``, n0 = ``rule.estimate.prior_quarters``
(8). K is n0 quarters of an average row, so the prior weighs as much as n0
quarters of data: "it starts at the balanced approach and moves toward the
data as off-floor quarters accumulate; the prior counts as two years." With no
usable quarter the estimate is beta0. Where X'X + K is singular (a regressor
that is zero on every row of the sample, where K is zero too; or n0 = 0 and
fewer quarters than coefficients), the estimate moves off beta0 only in the
directions the sample identifies, so the path is never NaN. The raw OLS path
is reported next to it (NaN until two quarters identify it). See
notes/DECISIONS.md (V4, V9).
"""

import numpy as np
import pandas as pd
from policypath.model.reaction import on
from policypath.regimes import AT_FLOOR, elb_state


def quarter_ends(model, spec):
    """One row per calendar quarter with a model session: its last session's inputs and the regression's row.

    `model` is the model panel (`model.path.build`'s summaries). Columns:
    quarter, session, r0, r0_prev (the previous quarter's last session's, NaN
    if that quarter has none), implied (R*_impl), y, x_inflation_gap,
    x_unemployment_gap, elb (either quarter in the ELB state), cut (either
    quarter ending on its floor after a cut that quarter) and usable (``cut``
    counts only with ``rule.estimate.drop_cuts_to_floor``).
    """
    m = model.sort_values("session").reset_index(drop=True)
    m = m.assign(quarter=m["session"].dt.to_period("Q"), elb_state=elb_state(m).to_numpy())
    last = m.groupby("quarter").tail(1).set_index("quarter")
    prev = last.reindex(last.index - 1)
    rho = spec["inertia"]
    pi_star = np.array([on(spec["inflation_target"], day) for day in last["session"]])
    r0_prev = prev["r0"].to_numpy()
    implied = (last["r0"].to_numpy() - rho * r0_prev) / (1.0 - rho)
    on_floor = (last["r0"] - last["elb"] <= AT_FLOOR).to_numpy()
    cut = pd.Series(on_floor & (r0_prev > last["r0"].to_numpy() + AT_FLOOR), index=last.index)
    out = pd.DataFrame({
        "quarter": last.index, "session": last["session"].to_numpy(), "r0": last["r0"].to_numpy(),
        "r0_prev": r0_prev, "implied": implied,
        "y": implied - last["rstar"].to_numpy() - last["inflation"].to_numpy(),
        "x_inflation_gap": last["inflation"].to_numpy() - pi_star,
        "x_unemployment_gap": -last["u_gap"].to_numpy(),
        "elb": last["elb_state"].to_numpy() | prev["elb_state"].fillna(False).astype(bool).to_numpy(),
        "cut": cut.to_numpy() | cut.reindex(last.index - 1, fill_value=False).to_numpy(),
    })
    finite = np.isfinite(out[["y", "x_inflation_gap", "x_unemployment_gap"]].to_numpy()).all(axis=1)
    censored = out["elb"] | (out["cut"] & spec["estimate"]["drop_cuts_to_floor"])
    return out.assign(usable=finite & ~censored)


def fit(X, y, beta0, n0):
    """(shrunk, ols) coefficients on one sample: `X` is n x 2, `y` n, `beta0` the prior, `n0` its weight in rows.

    Shrunk is beta0 on an empty sample. Where its system is singular it is
    the solution nearest beta0 (least squares on beta - beta0): a direction
    neither the sample nor the prior identifies stays at the prior. OLS is NaN
    where X'X is singular.
    """
    beta0 = np.asarray(beta0, dtype=float)
    nan = np.full(len(beta0), np.nan)
    if len(y) == 0:
        return beta0, nan
    xtx, xty = X.T @ X, X.T @ y
    prior = n0 * np.diag((X ** 2).mean(axis=0))
    a, b = xtx + prior, xty + prior @ beta0
    full = np.linalg.matrix_rank(a) == len(beta0)
    shrunk = np.linalg.solve(a, b) if full else beta0 + np.linalg.lstsq(a, b - a @ beta0, rcond=None)[0]
    ols = np.linalg.solve(xtx, xty) if np.linalg.matrix_rank(xtx) == len(beta0) else nan
    return shrunk, ols


def coefficients(model, spec):
    """The coefficients in force on every session of `model`, estimated from the quarters before its own.

    One row per session: session, coef_inflation_gap, coef_unemployment_gap
    (shrunk: what the model path uses), ols_inflation_gap, ols_unemployment_gap
    (raw) and coef_quarters (usable quarters in the sample).
    """
    ends = quarter_ends(model, spec)
    usable = ends[ends["usable"]]
    X, y = usable[["x_inflation_gap", "x_unemployment_gap"]].to_numpy(), usable["y"].to_numpy()
    c = spec["coefficients"]
    beta0, n0 = [c["inflation_gap"], c["unemployment_gap"]], spec["estimate"]["prior_quarters"]
    quarters = model["session"].dt.to_period("Q")
    rows = {}
    for q in quarters.unique():
        before = (usable["quarter"] < q).to_numpy()
        shrunk, ols = fit(X[before], y[before], beta0, n0)
        rows[q] = [*shrunk, *ols, int(before.sum())]
    table = pd.DataFrame.from_dict(rows, orient="index", columns=[
        "coef_inflation_gap", "coef_unemployment_gap", "ols_inflation_gap", "ols_unemployment_gap", "coef_quarters"])
    per_session = table.loc[quarters.to_numpy()].reset_index(drop=True)
    return pd.concat([model["session"].reset_index(drop=True), per_session], axis=1)
