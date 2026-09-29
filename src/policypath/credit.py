"""The credit bridge: does the policy-path gap say anything about credit spreads? Three tests.

Run for every enabled currency whose config has a ``credit`` block (USD:
Moody's seasoned corporate yields and ICE BofA's option-adjusted spreads, from
FRED). Spread changes stand in for excess returns: a spread 10bp wider on a
bond of duration D is about D x 10bp lost against Treasuries, with the carry
left out. That is said wherever the numbers are.

**The spreads** (``credit.spreads``: each a signed sum of cached series,
`config.SPREAD_SIGNS`, in bp). The primary is Baa - Aaa: both legs are Moody's
seasoned long bonds, so the maturity and the Treasury benchmark cancel and what
is left is the price of the lower grade. Baa - 30y and Baa - 10y (FRED's own
BAA10Y) sit alongside; they also carry the Treasury curve's moves. ICE BofA's IG
and HY OAS start 2023-09-29 on FRED, so they are a three-year cross-check. Every
print is the one dated on its day as it stood after the series' publication lag
(`strategy.instruments.Marks`), as the trading legs read theirs.

**Two samplings.** Tests 1 and 3 use weekly changes: the last print on or
before each ``credit.week_ends`` day (Wednesday, the Fed's week) within the six
days before it, so a holiday Wednesday reads the Tuesday. Daily changes in
seasoned-bond quotes are mostly noise, and a week is still short enough for the
policy path's moves to be news. Test 2 is on the signal's sessions, because its
horizons are in sessions.

**The regressors** are the currency's own signal components (`signal.components`), in bp:

- the level's change, split in two: the market's rate for the meeting that was
  ``backtest.horizon``-th at the start of the week, and minus the model's rate
  for it. They sum to the change in that meeting's gap. Holding the meeting keeps
  a meeting passing from booking the path's slope to the next meeting as a
  repricing, as the backtest credits its P&L;
- the orthogonal slope (8th meeting's gap minus the 1st's, less the level's
  trailing-beta share) and each cross sleeve's differential that includes this
  currency: the signal's own changes, not held, so in a week a meeting passes
  they include the path's shape across it. A differential is missing on the
  other currency's holidays, and a week that starts or ends on one is left out
  of test 1;
- the Treasury 10s30s (``credit.control``), with and without.

**Tests.**

1. *Contemporaneous*: weekly Δspread on the market's part alone (its R² is how
   much of credit's weekly variation is a repricing of the policy path), on
   every component, and on every component plus the 10s30s. NW lag 4, a month.
2. *Predictive*: s(t+1+h) - s(t+1) on gap(t) at the horizon, in bp and in z,
   h = 21 and 63 sessions, NW lag h. The change starts the session after the
   signal, so none of it was known with the signal. Cross-checked on
   non-overlapping changes: every h-th session, for each of the h start
   offsets. Run on every session with a z (ELB sessions included), without
   each ``credit.exclude`` window (every change that spans it left out), and
   without ELB sessions. The sign was registered before the first run
   (notes/DECISIONS.md, D1): negative. The headline cell is also refitted
   without each calendar year in turn (D17), and under four more checks
   (D18): Bartlett at lag 2h; Hansen-Hodrick's equal weights to lag h - 1,
   which give the overlap all of its weight where Bartlett at lag h gives
   it about two thirds; z clipped at ±3; and without the sessions whose z
   divides by the floored sd. A placebo (D19) measures the registered
   test's size: the spread's daily changes rotated in time against z, so
   neither loses its persistence and the link is gone. All six were added
   after the run, so they are checks, not registered cells.
3. *Conditional*: weekly Δspread on the weekly change in the level by the
   regime at the start of the week (`regimes`, real time; ELB weeks out). A
   slope and intercept per regime in one regression (NW lag 4), the late-minus-
   early difference with its minimum detectable size at 80% power, and the
   slope in each episode (a run of one regime).

**Staleness**, checked before test 2: the first-order autocorrelation of each
spread's weekly change, and its correlation with the change in the
``credit.staleness`` yield (the 10y) a week earlier. Either outside
±1.96/√n and the yield's lagged change becomes a control in tests 1 and 2
(D2): the week before in test 1, the five sessions to t+1 in test 2. The
same statistics without each ``credit.exclude`` window are reported beside
them, as a description only: the rule reads the full sample.

**OLS and Newey-West in numpy** (`ols`). A row with a NaN is left out of every
sum but keeps its place in time, so Newey-West lag j always pairs observations
j sessions (or weeks) apart, and a gap is not closed up. Bartlett weights
1 - j/(L+1), no small-sample correction (Newey and West, 1987); the uniform
weights of Hansen and Hodrick (1980) for D18's check only.
"""

import numpy as np
import pandas as pd
from policypath import config, regimes
from policypath.signal import components

HORIZONS = (21, 63)       # test 2, sessions ahead: a month and a quarter
WEEKLY_LAGS = 4           # Newey-West lags on weekly changes: a month
STALE_LAGS = (0, 1, 2)    # weeks between the yield's change and the spread's, for the staleness check
CONTROL_SESSIONS = 5      # test 2's staleness control: the yield's change over the week to t+1
MIN_WEEKS = 13            # a regime or an episode needs a quarter of weeks for its own slope
Z_CLIP = 3.0              # D18's check: z clipped at ±3
PLACEBO_MIN = 252         # D19's placebo: the spread's changes shifted by at least a year of sessions against z,
PLACEBO_STEP = 5          # every fifth shift (a week; shifts a day apart give nearly the same fit)
Z95 = 1.959963984540054   # two-sided 5%
Z80 = 0.8416212335729143  # 80% power
BP = 100.0
WEEK = pd.Timedelta(days=7)


# ---- OLS and Newey-West ------------------------------------------------------------

def newey_west(scores, lags, uniform=False):
    """The long-run covariance of `scores` (n x k, x_t u_t): S0 + sum_j w_j (S_j + S_j').

    Bartlett weights w_j = 1 - j/(L+1), or with `uniform` w_j = 1 (Hansen-
    Hodrick), which need not be positive definite.
    """
    s = scores.T @ scores
    for j in range(1, lags + 1):
        cross = scores[j:].T @ scores[:-j]
        s = s + (1.0 if uniform else 1.0 - j / (lags + 1.0)) * (cross + cross.T)
    return s


def ols(y, X, lags=0, uniform=False):
    """OLS of `y` (n) on `X` (n x k; the caller adds the constant), Newey-West standard errors at `lags`.

    Returns {beta, se, t, cov, r2, n}; r2 is centred. A row with a NaN in y
    or X is left out of every sum but keeps its place, so lag j pairs rows j
    apart. All NaN where fewer rows than regressors, or they are collinear.
    `uniform` weights every lag fully (`newey_west`); a negative variance
    then gives a NaN se.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float).reshape(len(y), -1)
    k = X.shape[1]
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    Xo, yo = np.where(ok[:, None], X, 0.0), np.where(ok, y, 0.0)
    xtx = Xo.T @ Xo
    if ok.sum() <= k or np.linalg.matrix_rank(xtx) < k:
        nan = np.full(k, np.nan)
        return {"beta": nan, "se": nan, "t": nan, "cov": np.full((k, k), np.nan), "r2": np.nan, "n": int(ok.sum())}
    bread = np.linalg.inv(xtx)
    beta = bread @ (Xo.T @ yo)
    u = yo - Xo @ beta
    cov = bread @ newey_west(Xo * u[:, None], lags, uniform) @ bread
    var = np.diag(cov)
    se = np.sqrt(np.where(var > 0, var, np.nan))
    dev = y[ok] - y[ok].mean()
    return {"beta": beta, "se": se, "t": beta / se, "cov": cov, "r2": 1.0 - (u @ u) / (dev @ dev), "n": int(ok.sum())}


def _terms(fit, names):
    """A fit as {coef, se, t} by regressor name, with r2 and n."""
    return {"coef": dict(zip(names, fit["beta"])), "se": dict(zip(names, fit["se"])),
            "t": dict(zip(names, fit["t"])), "r2": fit["r2"], "n": fit["n"]}


def _design(frame, columns):
    return np.column_stack([np.ones(len(frame)), frame[columns].to_numpy(dtype=float)])


# ---- data --------------------------------------------------------------------------

def spread(daily, legs):
    """One spread in bp by date: `legs` ({role: {source, series}}) signed by `config.SPREAD_SIGNS`, where every leg printed.

    `daily(source, series)` returns a series by date (`strategy.instruments.Marks.daily`).
    """
    parts = [config.SPREAD_SIGNS[role] * daily(ref["source"], ref["series"]) for role, ref in legs.items()]
    return pd.concat(parts, axis=1, join="inner").sum(axis=1) * BP


def series(credit, daily):
    """(spreads, control, staleness yield), bp by date. Spreads: the primary first, then the rest, then the cross-checks."""
    names = [credit["primary"], *(n for n in credit["spreads"] if n != credit["primary"])]
    out = {n: spread(daily, credit["spreads"][n]) for n in names}
    out |= {n: spread(daily, {"long": ref}) for n, ref in credit["crosscheck"].items()}
    return out, spread(daily, credit["control"]), spread(daily, {"long": credit["staleness"]})


def week_ends(first, last, weekday):
    """Every `weekday` (a `config.WEEKDAYS` name) from `first` to `last`."""
    return pd.date_range(pd.Timestamp(first), pd.Timestamp(last), freq=f"W-{weekday[:3].upper()}")


def on_or_before(index, ends):
    """For each of `ends`: the position in sorted `index` of its last date on or before it within the week, else -1."""
    index = pd.DatetimeIndex(index)
    at = index.searchsorted(ends, side="right") - 1
    fresh = (at >= 0) & ((ends - index[np.maximum(at, 0)]) < WEEK)
    return np.where(fresh, at, -1)


def weekly(values, ends):
    """`values` (a series by date) on each of `ends`: the last print on or before it within the week (NaN if none)."""
    values = values.dropna().sort_index()
    at = on_or_before(values.index, ends)
    return pd.Series(np.where(at >= 0, values.to_numpy()[np.maximum(at, 0)], np.nan), index=ends)


def on_sessions(values, sessions):
    """`values` on each session: the last print dated on or before it. Returns (values, stale count).

    Stale: a session after the series' first print with no print of its own
    (it carries the one before). Sessions before the first print are NaN, not stale.
    """
    values = values.dropna().sort_index()
    at = values.index.searchsorted(sessions, side="right") - 1
    got = pd.Series(np.where(at >= 0, values.to_numpy()[np.maximum(at, 0)], np.nan), index=sessions)
    stale = (at >= 0) & (values.index[np.maximum(at, 0)] != sessions)
    return got, int(stale.sum())


def held(paths, k, start, end):
    """(market, model) change in bp from sessions `start` to `end`, for the meeting that was `k`-th at `start`.

    `paths` is the paths panel (session, k, effective_date, market, model, in
    percent). NaN where either session is missing or the meeting is not on
    the path at `end`.
    """
    rates = paths.set_index(["session", "effective_date"])[["market", "model"]]
    meeting = paths[paths["k"] == k].set_index("session")["effective_date"].reindex(start).to_numpy()
    def at(days):
        return rates.reindex(pd.MultiIndex.from_arrays([days, meeting])).to_numpy()
    change = (at(end) - at(start)) * BP
    return change[:, 0], change[:, 1]


class Inputs:
    """One currency's bridge inputs: its spreads, its signal components and its regimes.

    `cfg_of(ccy)` is a currency block, `book` the strategy config (its cross
    sleeves give the differentials), `load(ccy, name)` a panel and
    `daily(source, series)` a cached series by date, as published.
    """

    def __init__(self, ccy, cfg_of, book, load, daily):
        cfg = cfg_of(ccy)
        self.ccy, self.credit = ccy, cfg["credit"]
        self.k, self.sd_floor = cfg["backtest"]["horizon"], cfg["signal"]["sd_floor_bp"]
        self.spreads, self.control, self.yield_ = series(self.credit, daily)
        signal = load(ccy, "signal")
        self.paths = load(ccy, "paths")
        self.level = components.level(signal, self.k)
        self.slope = components.slope(signal, book["signal"]["slope"], self.k, cfg["signal"],
                                      book["signal"]["slope_orthogonal"])
        self.differentials = {}
        for s in book["sleeves"]:
            if s["kind"] == "cross" and ccy in s["pair"]:
                a, b = (cfg_of(c) for c in s["pair"])
                self.differentials[s["name"]] = components.differential(
                    load(s["pair"][0], "signal"), load(s["pair"][1], "signal"), a["backtest"]["horizon"],
                    b["backtest"]["horizon"], a["signal"])
        model = load(ccy, "model")
        self.elb = regimes.elb_state(model)
        self.state = regimes.states(model)
        self.sessions = self.level.index

    def regressor_names(self):
        """The component regressors, in order: market, model, slope, then one differential per cross sleeve."""
        diffs = ["differential"] if len(self.differentials) == 1 else [f"differential_{i + 1}"
                                                                        for i in range(len(self.differentials))]
        return ["market", "model", "slope", *diffs]

    def weeks(self):
        """The weekly frame: one row per week end (from the second), every change in bp, and the regime at its start.

        Columns: each spread's change, ``control`` (the 10s30s change),
        ``yield`` and ``yield_lag`` (the staleness yield's change that week and
        the week before), the regressors (`regressor_names`) and ``level``
        (market + model), and ``start``, ``state`` and ``elb`` (the session the
        week starts from, its regime and ELB state). No spread may take one of
        these names (`config.CREDIT_COLUMNS`).
        """
        sessions = self.sessions
        ends = week_ends(sessions[0] - WEEK, sessions[-1], self.credit["week_ends"])
        at = on_or_before(sessions, ends)
        day = sessions[np.maximum(at, 0)].where(at >= 0)
        start, end = day[:-1], day[1:]
        out = pd.DataFrame(index=ends[1:])
        for name, s in self.spreads.items():
            out[name] = weekly(s, ends).diff().iloc[1:].to_numpy()
        out["control"] = weekly(self.control, ends).diff().iloc[1:].to_numpy()
        out["yield"] = weekly(self.yield_, ends).diff().iloc[1:].to_numpy()
        out["yield_lag"] = out["yield"].shift(1)
        market, model = held(self.paths, self.k, start, end)
        out["market"], out["model"] = market, -model
        out["slope"] = (self.slope["value_bp"].reindex(end).to_numpy()
                        - self.slope["value_bp"].reindex(start).to_numpy())
        for name, d in zip(self.regressor_names()[3:], self.differentials.values()):
            out[name] = d["value_bp"].reindex(end).to_numpy() - d["value_bp"].reindex(start).to_numpy()
        out["level"] = out["market"] + out["model"]
        out["start"] = start
        out["state"] = self.state.reindex(start).to_numpy()
        out["elb"] = self.elb.reindex(start).fillna(False).astype(bool).to_numpy()
        return out.dropna(subset=["start"])

    def daily_frame(self):
        """By signal session: each spread (bp, the last print on or before), the level's gap_bp, z and sd_bp, the yield and ELB state.

        ``sd_bp`` is the trailing sd z divides by (floored). Also returns the
        stale-session count per spread.
        """
        out = self.level[["value_bp", "z", "window_sd_bp"]].rename(columns={"value_bp": "gap_bp",
                                                                            "window_sd_bp": "sd_bp"})
        stale = {}
        for name, s in self.spreads.items():
            out[name], stale[name] = on_sessions(s, self.sessions)
        out["yield"], stale["yield"] = on_sessions(self.yield_, self.sessions)
        out["elb"] = self.elb.reindex(self.sessions).fillna(False).astype(bool).to_numpy()
        return out, stale


# ---- staleness ---------------------------------------------------------------------

def staleness(dy, dyield, lags=STALE_LAGS):
    """Weekly change `dy`: its first-order autocorrelation, and its correlation with `dyield` j weeks earlier.

    ``shows`` when either the autocorrelation or the correlation a week back is
    outside ±1.96/√n (D2).
    """
    n = int(dy.notna().sum())
    band = Z95 / np.sqrt(n)
    ar1 = dy.corr(dy.shift(1))
    cross = {j: dy.corr(dyield.shift(j)) for j in lags}
    return {"n": n, "band": band, "ar1": ar1, "cross": cross, "shows": bool(abs(ar1) > band or abs(cross[1]) > band)}


# ---- test 1: contemporaneous -------------------------------------------------------

def contemporaneous(weeks, name, parts, stale=False, lags=WEEKLY_LAGS, drop=None):
    """Weekly Δspread `name` on the market's part alone, on every component (`parts`), and on those plus the 10s30s.

    One sample for all three (every regressor there), so the R²s compare.
    With `stale` the yield's change the week before joins the component fits
    (the market-alone R² stays a plain variance share). `drop` (a bool mask on
    `weeks`) leaves weeks out. Returns {spec: `_terms` plus first and last week}.
    """
    extra = ["yield_lag"] if stale else []
    need = [name, *parts, "control", *extra]
    keep = weeks[need].notna().all(axis=1) & (True if drop is None else ~drop)
    y = weeks[name].where(keep)
    specs = {"market": ["market"], "components": [*parts, *extra], "components_control": [*parts, "control", *extra]}
    out = {}
    for spec, cols in specs.items():
        fit = ols(y, _design(weeks, cols), lags)
        out[spec] = {**_terms(fit, ["const", *cols]), "first": weeks.index[keep].min(), "last": weeks.index[keep].max()}
    return out


# ---- test 2: predictive ------------------------------------------------------------

def ahead(values, h):
    """By session t, bp: values(t + 1 + h) - values(t + 1), sessions by position."""
    return values.shift(-(1 + h)) - values.shift(-1)


def spans(sessions, h, window):
    """By session t: True where t's change, from session t + 1 to t + 1 + h, touches `window` ([first, last] days).

    It touches the window when it starts on or before the window's last day
    and ends on or after its first.
    """
    a, b = (pd.Timestamp(d) for d in window)
    day = pd.Series(sessions, index=sessions)
    return (day.shift(-(1 + h)) >= a) & (day.shift(-1) <= b)


def week_spans(ends, window):
    """By week end: True where the week's change, from the week end seven days before to this one, touches `window`."""
    a, b = (pd.Timestamp(d) for d in window)
    return pd.Series((ends >= a) & (ends - WEEK <= b), index=ends)


def nonoverlapping(y, X, h):
    """OLS on every `h`-th row, for each of the h start offsets (White standard errors): the slope's spread over them.

    Returns {mean, min, max, mean_t, share_negative, share_clearing, offsets,
    n}, the slope being `X`'s second column; share_clearing is the share of
    offsets whose own t is at most -1.96, n the mean rows per offset.
    """
    fits = [ols(y[o::h], X[o::h], 0) for o in range(h)]
    slopes = np.array([f["beta"][1] for f in fits])
    ts = np.array([f["t"][1] for f in fits])
    return {"mean": np.nanmean(slopes), "min": np.nanmin(slopes), "max": np.nanmax(slopes), "mean_t": np.nanmean(ts),
            "share_negative": float(np.mean(slopes[np.isfinite(slopes)] < 0)),
            "share_clearing": float(np.mean(ts[np.isfinite(ts)] <= -Z95)), "offsets": h,
            "n": float(np.mean([f["n"] for f in fits]))}


def samples(frame, h, exclude):
    """The test 2 samples by session: all (every session with a z), without each window, without ELB sessions."""
    base = frame["z"].notna()
    out = {"all": base}
    out |= {f"ex_{name}": base & ~spans(frame.index, h, window) for name, window in exclude.items()}
    out["ex_elb"] = base & ~frame["elb"]
    return out


def _ahead_design(frame, regressor, stale):
    """The constant, `regressor` and, with `stale`, the yield's change over the `CONTROL_SESSIONS` to t+1."""
    cols = [frame[regressor].to_numpy(dtype=float)]
    if stale:
        cols.append((frame["yield"].shift(-1) - frame["yield"].shift(CONTROL_SESSIONS - 1)).to_numpy(dtype=float))
    return np.column_stack([np.ones(len(frame)), *cols])


def predictive(frame, name, h, exclude, stale=False):
    """s(t+1+h) - s(t+1) for spread `name` on gap(t), in bp and in z, per sample: NW lag h and the non-overlapping check.

    `frame` is `Inputs.daily_frame`. With `stale` the yield's change over the
    `CONTROL_SESSIONS` to t+1 is a control. Returns {regressor: {sample: {slope,
    se, t, r2, n, nonoverlap}}}.
    """
    y = ahead(frame[name], h)
    out = {}
    for regressor in ("z", "gap_bp"):
        X = _ahead_design(frame, regressor, stale)
        out[regressor] = {}
        for sample, keep in samples(frame, h, exclude).items():
            yk = y.where(keep).to_numpy()
            fit = ols(yk, X, h)
            out[regressor][sample] = {"slope": fit["beta"][1], "se": fit["se"][1], "t": fit["t"][1], "r2": fit["r2"],
                                      "n": fit["n"], "change_sd": np.nanstd(yk[np.isfinite(X).all(axis=1)]),
                                      "nonoverlap": nonoverlapping(yk, X, h)}
    return out


def supported(cell):
    """D1's rule on one predictive cell: slope negative, NW t at most -1.96, the non-overlapping mean slope negative."""
    return bool(cell["slope"] < 0 and cell["t"] <= -Z95 and cell["nonoverlap"]["mean"] < 0)


def calendar_years(sessions):
    """Each calendar year of `sessions` as a window to leave out: {"2014": ("2014-01-01", "2014-12-31"), ...}."""
    return {str(y): (f"{y}-01-01", f"{y}-12-31") for y in sorted(set(pd.DatetimeIndex(sessions).year))}


def without_each_year(frame, name, h, stale=False):
    """The headline cell (z, every session with a z) refitted without each calendar year, and D1's rule on each.

    Every change that touches the year is left out (`spans`), as for each
    ``credit.exclude`` window. Added after the first run (D17), so a check on
    how much the headline leans on one year, not a registered cell. Returns
    {year: cell plus ``supported``}.
    """
    years = calendar_years(frame.index[frame["z"].notna()])
    fits = predictive(frame, name, h, years, stale)["z"]
    return {y: fits[f"ex_{y}"] | {"supported": supported(fits[f"ex_{y}"])} for y in years}


def after_the_run(frame, name, h, sd_floor, stale=False):
    """The headline cell (z, every session with a z) under four checks added after the first run (D18).

    - ``bartlett_2h``: Newey-West at lag 2h instead of h;
    - ``uniform``: Hansen-Hodrick, every lag to h - 1 at full weight. The
      overlapping changes are an MA(h-1), and Bartlett at lag h gives its
      autocovariances about two thirds of their weight in all;
    - ``z_clipped``: z clipped at ±`Z_CLIP`;
    - ``above_floor``: without the sessions whose trailing sd is at the
      signal's floor (`sd_floor`, bp), where z divides by the floor.

    Checks, not registered cells: D1's cell and rule stand. Returns {check:
    {slope, se, t, n}}.
    """
    y = ahead(frame[name], h).where(frame["z"].notna()).to_numpy()
    X = _ahead_design(frame, "z", stale)
    clipped = _ahead_design(frame.assign(z=frame["z"].clip(-Z_CLIP, Z_CLIP)), "z", stale)
    above = np.where(frame["sd_bp"].to_numpy() > sd_floor, y, np.nan)
    fits = {"bartlett_2h": ols(y, X, 2 * h), "uniform": ols(y, X, h - 1, uniform=True),
            "z_clipped": ols(y, clipped, h), "above_floor": ols(above, X, h)}
    return {c: {"slope": f["beta"][1], "se": f["se"][1], "t": f["t"][1], "n": f["n"]} for c, f in fits.items()}


def ahead_of_changes(d, h):
    """By row t: d(t+2) + ... + d(t+1+h), which is `ahead` rebuilt from the daily changes d(i) = s(i) - s(i-1)."""
    return pd.Series(d).rolling(h, min_periods=h).sum().shift(-(1 + h)).to_numpy()


def placebo(frame, name, h, stale=False, least=PLACEBO_MIN, step=PLACEBO_STEP):
    """How often the headline cell's NW t rejects when the spread has no link to z: the test's size (D19).

    From the first session with a z, the daily changes of spread `name` are
    rotated in time by every `step`-th shift from `least` sessions to `least`
    short of their number, and the h-session changes rebuilt from them. Each
    series keeps its own volatility and persistence (the rotation breaks the
    spread's once, at the wrap), and no shift lines one up with the other. The
    design (z, and with `stale` the staleness control) is the headline's.
    Added after the first run, so a check, not a registered cell. Returns {t
    (the headline's), shifts, reject (the share of shifted |t| at least 1.96:
    the size where 5% is nominal), reject_negative (t at most -1.96: nominal
    2.5%), critical (the shifted t's 2.5% quantile: the one-sided line at its
    nominal size), p (the share of shifted t at or below the headline's)}.
    """
    first = int(np.flatnonzero(frame["z"].notna().to_numpy())[0])
    have = frame["z"].notna().to_numpy()[first:]
    X = _ahead_design(frame, "z", stale)[first:]
    changes = np.diff(frame[name].to_numpy(dtype=float)[first:])

    def t_of(d):
        return ols(np.where(have, ahead_of_changes(np.r_[np.nan, d], h), np.nan), X, h)["t"][1]

    t0 = t_of(changes)
    ts = np.array([t_of(np.roll(changes, k)) for k in range(least, len(changes) - least + 1, step)])
    ts = ts[np.isfinite(ts)]
    return {"t": t0, "shifts": len(ts), "reject": float(np.mean(np.abs(ts) >= Z95)),
            "reject_negative": float(np.mean(ts <= -Z95)), "critical": float(np.quantile(ts, 0.025)),
            "p": float(np.mean(ts <= t0))}


# ---- test 3: conditional -----------------------------------------------------------

def pooled(weeks, name, regressor, lags=WEEKLY_LAGS):
    """Δspread on `regressor` with a slope and intercept per regime (ELB weeks out), in one regression.

    Regimes with fewer than `MIN_WEEKS` weeks are left out. Returns
    ({state: {slope, se, t, weeks}}, late_minus_early {diff, se, t, mde}) where
    mde is the smallest difference 80% power would detect at 5%.
    """
    keep = weeks[[name, regressor]].notna().all(axis=1) & ~weeks["elb"]
    counts = weeks.loc[keep, "state"].value_counts()
    present = [s for s in regimes.STATES if s != regimes.ELB and counts.get(s, 0) >= MIN_WEEKS]
    diff = {"diff": np.nan, "se": np.nan, "t": np.nan, "mde": np.nan}
    if not present:
        return {}, diff
    columns = []
    for s in present:
        d = (weeks["state"] == s).astype(float).to_numpy()
        columns += [d, d * weeks[regressor].to_numpy(dtype=float)]
    y = weeks[name].where(keep & weeks["state"].isin(present)).to_numpy()
    fit = ols(y, np.column_stack(columns), lags)
    by_state = {s: {"slope": fit["beta"][2 * i + 1], "se": fit["se"][2 * i + 1], "t": fit["t"][2 * i + 1],
                    "weeks": int(counts[s])} for i, s in enumerate(present)}
    early, late = regimes.HIKING
    if early in present and late in present:
        c = np.zeros(len(fit["beta"]))
        c[2 * present.index(late) + 1], c[2 * present.index(early) + 1] = 1.0, -1.0
        d, se = c @ fit["beta"], np.sqrt(c @ fit["cov"] @ c)
        diff = {"diff": d, "se": se, "t": d / se, "mde": (Z95 + Z80) * se}
    return by_state, diff


def episodes(weeks, state, name, regressor, lags=WEEKLY_LAGS):
    """The slope of Δspread on `regressor` in each episode (a run of one regime, ELB out) with `MIN_WEEKS` weeks or more.

    `state` is the regime by session. One row per episode: state, first and
    last week, weeks, slope, se, t, and mde (the slope 80% power would detect).
    """
    runs = regimes.episodes(state)
    run = runs["first"].searchsorted(weeks["start"], side="right") - 1
    rows = []
    for i, ep in runs.iterrows():
        if ep["state"] == regimes.ELB:
            continue
        inside = (run == i) & weeks[[name, regressor]].notna().all(axis=1).to_numpy()
        if inside.sum() < MIN_WEEKS:
            continue
        w = weeks[inside]
        fit = ols(w[name], _design(w, [regressor]), lags)
        rows.append({"state": ep["state"], "first": w.index[0], "last": w.index[-1], "weeks": fit["n"],
                     "slope": fit["beta"][1], "se": fit["se"][1], "t": fit["t"][1], "mde": (Z95 + Z80) * fit["se"][1]})
    return pd.DataFrame(rows, columns=["state", "first", "last", "weeks", "slope", "se", "t", "mde"])


CYCLE_FIELDS = ("first", "last", "weeks", "slope", "se", "t", "mde")


def cycles(table):
    """Hiking cycles in an `episodes` table: an early-hiking episode and the late-hiking one right after it.

    One row per cycle with both: each episode's `CYCLE_FIELDS` (``early_first``,
    ``late_slope``, ...), whether the two slopes' signs differ, and whether they
    fall as D3 expects (early < 0 < late).
    """
    early, late = regimes.HIKING
    rows = []
    for i in range(len(table) - 1):
        a, b = table.iloc[i], table.iloc[i + 1]
        if a["state"] == early and b["state"] == late:
            row = {f"{part}_{f}": ep[f] for part, ep in (("early", a), ("late", b)) for f in CYCLE_FIELDS}
            rows.append(row | {"differs": bool(np.sign(a["slope"]) != np.sign(b["slope"])),
                               "as_expected": bool(a["slope"] < 0 < b["slope"])})
    return pd.DataFrame(rows, columns=[f"{part}_{f}" for part in ("early", "late") for f in CYCLE_FIELDS]
                        + ["differs", "as_expected"])


WORDS = {0: "no", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}


def sign_sentence(cyc):
    """The generated verdict on test 3: in how many hiking cycles the early and late slopes differ in sign, and which way.

    It names the cycles where the sign does not differ as well as those where
    it does, and says nothing stronger (D15). The chart's subtitle is the part
    before the colon.
    """
    n, differ, expected = len(cyc), int(cyc["differs"].sum()), int(cyc["as_expected"].sum())
    if n == 0:
        return "The sample has no complete hiking cycle, so the sign cannot be compared."
    def word(i):
        return WORDS.get(i, str(i))
    if n == 1:
        where = "in the one cycle"
    elif differ == n:
        where = f"in each of the {word(n)} cycles"
    elif differ == 0:
        where = f"in {'either' if n == 2 else 'any'} of the {word(n)} cycles"
    else:
        rest = n - differ
        where = f"in {word(differ)} of the {word(n)} cycles and not in the {'other' if rest == 1 else f'other {word(rest)}'}"
    if differ == 0:
        return f"The sign does not differ {where}."
    way = ("early negative and late positive, as the hypothesis has it" if expected == differ else
           "the other way round from the hypothesis" if expected == 0 else
           f"the hypothesis's way in {word(expected)} of them")
    return f"The sign differs {where}: {way}."


# ---- everything ----------------------------------------------------------------------

def run(inputs):
    """Every test on one currency's `Inputs`. Returns (results, frames): results nest dicts of numbers; frames feed the report."""
    credit, names = inputs.credit, list(inputs.spreads)
    parts = inputs.regressor_names()
    weeks = inputs.weeks()
    daily, stale_sessions = inputs.daily_frame()
    exclude = credit["exclude"]

    stale = {n: staleness(weeks[n], weeks["yield"]) for n in names}
    drop = {name: week_spans(weeks.index, window) for name, window in exclude.items()}
    stale_ex = {n: {w: staleness(weeks[n].where(~mask), weeks["yield"].where(~mask)) for w, mask in drop.items()}
                for n in names}
    test1 ={n: contemporaneous(weeks, n, parts, stale[n]["shows"]) for n in names}
    test1_ex = {n: {w: contemporaneous(weeks, n, parts, stale[n]["shows"], drop=mask)["components"]
                    for w, mask in drop.items()} for n in names}
    test2 = {n: {h: predictive(daily, n, h, exclude, stale[n]["shows"]) for h in HORIZONS} for n in names}
    primary, h = credit["primary"], max(HORIZONS)
    unstaled = predictive(daily, primary, h, {}, False)["z"]["all"] if stale[primary]["shows"] else None
    by_year = without_each_year(daily, primary, h, stale[primary]["shows"])
    checks = after_the_run(daily, primary, h, inputs.sd_floor, stale[primary]["shows"])
    size = placebo(daily, primary, h, stale[primary]["shows"])
    test3 = {}
    for n in names:
        test3[n] = {}
        for regressor in ("level", "market"):
            by_state, diff = pooled(weeks, n, regressor)
            table = episodes(weeks, inputs.state, n, regressor)
            cyc = cycles(table)
            test3[n][regressor] = {"pooled": by_state, "late_minus_early": diff, "episodes": table, "cycles": cyc,
                                   "sentence": sign_sentence(cyc)}
    results = {"spreads": names, "primary": primary, "regressors": parts, "k": inputs.k,
               "differentials": list(inputs.differentials), "staleness": stale, "staleness_ex": stale_ex,
               "contemporaneous": test1, "contemporaneous_ex": test1_ex, "predictive": test2,
               "headline_unstaled": unstaled, "headline_without_each_year": by_year, "headline_checks": checks,
               "headline_placebo": size, "conditional": test3,
               "stale_sessions": stale_sessions,
               "first_print": {n: s.first_valid_index() for n, s in inputs.spreads.items()},
               "sample": {"weeks": len(weeks), "first_week": weeks.index[0], "last_week": weeks.index[-1],
                          "sessions": int(daily["z"].notna().sum()), "first_session": daily["z"].first_valid_index(),
                          "last_session": daily.index[-1], "elb_sessions": int(daily.loc[daily["z"].notna(), "elb"].sum()),
                          "elb_weeks": int(weeks["elb"].sum())},
               "supported": supported(test2[primary][h]["z"]["all"])}
    return results, {"weeks": weeks, "daily": daily, "state": inputs.state}
