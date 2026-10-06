"""The sleeves end to end, on synthetic panels and a synthetic cache laid out as the real one.

Two currencies on two calendars, every series the book's five sleeves read
(ZQ settles, Treasury par yields, the OIS and gilt spot curves, the FX rate),
the crude backtest recomputed on the synthetic GBP panel. Checks: the three
checks hold on every sleeve and session, the GBP outright reproduces the
backtest, a stale mark books no rate move, and nothing credited or expected
through D moves when every mark, z and overnight rate after D is poisoned
(with the breakeven live before D). Each leg marks the cache's print dated on
the session: the ZQ month the k-th meeting selects, the par node, the FX quote
the right way up. An outright's edge is its instrument's day-weighted
de-meaned gap; the cross sleeve receives the first currency on first minus
second. And the ZQ DV01 is derived: its rounded value appears in no code token
under src/.
"""

import ast
import io
from pathlib import Path
import re
import tokenize
import numpy as np
import pandas as pd
import pytest
from policypath import config
from policypath.backtest import policy
from policypath.calendars import BDAYS
from policypath.curves.forward import forward_path
from policypath.signal.gap import zscore
from policypath.strategy import carry, expression

T = pd.Timestamp
SRC = Path(__file__).resolve().parents[1] / "src" / "policypath"
BOOK = config.strategy()
USD, GBP = config.currency("USD"), config.currency("GBP")
START = T("2021-01-04")
N = 620
EFFECTIVE = pd.date_range("2021-01-27", periods=40, freq="42D")
CUT = T("2023-02-15")                      # D for the look-ahead test: after the curve sleeves' first z
PAIRS = 20                                 # closure pairs: the world is too short for the book's 250
POISON = 99.0
FX_POISON = 2.2                            # USD per GBP: wrong, but inside expression.fx.plausible


def calendar():
    return pd.DataFrame({"announcement_date": EFFECTIVE - pd.Timedelta(days=1), "effective_date": EFFECTIVE,
                         "scheduled": True, "cancelled": pd.NaT})


def sessions_of(ccy):
    days = pd.bdate_range(START, periods=N)
    drop = ["2021-05-31", "2022-05-30"] if ccy == "USD" else ["2021-08-30", "2022-08-29"]
    return days[~days.isin(pd.to_datetime(drop))]


def meetings_of(days, rates=None):
    rows = []
    for i, day in enumerate(days):
        ahead = EFFECTIVE[EFFECTIVE > day][:8]
        for k, e in enumerate(ahead, start=1):
            rows.append({"session": day, "k": k, "announcement_date": e - pd.Timedelta(days=1), "effective_date": e,
                         "scheduled": True, "rate": np.nan if rates is None else rates[i][k - 1]})
    return pd.DataFrame(rows)


def signal_of(days, seed):
    rng = np.random.default_rng(seed)
    walk = np.cumsum(rng.normal(0, 3, len(days)))
    gap = pd.DataFrame({k: walk * (0.5 + 0.12 * k) + rng.normal(0, 3, len(days)) for k in range(1, 9)}, index=days)
    z, mean, sd = zscore(gap, USD["signal"])
    parts = {"gap_bp": gap, "z": z, "window_mean_bp": mean, "window_sd_bp": sd}
    return pd.concat({n: f.stack(future_stack=True) for n, f in parts.items()}, axis=1).rename_axis(
        ["session", "k"]).reset_index()


def vintages(dates, values, lag_bday, key=None, keys=None):
    """A vintage log: one row per date (and key), final `lag_bday` later (published on the day with None)."""
    dates = pd.DatetimeIndex(dates)
    final = {d: d if lag_bday is None else d + lag_bday + pd.Timedelta(hours=12) for d in dates.unique()}
    out = pd.DataFrame({"date": dates, "value": values, "published": dates.map(final), "retrieved": T("2026-09-28")})
    return out.assign(**{key: keys}) if key else out


def curve_log(days, curves, lag_bday):
    """A fitted curve's log from a session x tenor frame, NaN nodes left out."""
    long = curves.set_axis(days).stack().rename("value").rename_axis(["date", "tenor"]).reset_index()
    return vintages(long["date"], long["value"].to_numpy(), lag_bday, "tenor", long["tenor"].to_numpy())


def write(root, source, ccy, series, frame):
    path = Path(root) / source / ccy / f"{series}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def paths_of(signal, meetings, seed):
    """The path panel behind a synthetic signal: a wandering rule path, and the market above it by the gap."""
    rows = signal[["session", "k", "gap_bp"]].merge(meetings[["session", "k", "effective_date"]], on=["session", "k"])
    walk = pd.Series(np.cumsum(np.random.default_rng(seed).normal(0, 0.02, rows["session"].nunique())),
                     index=rows["session"].unique())
    model = 1.0 + 0.05 * rows["k"] + rows["session"].map(walk)
    return rows.assign(model=model, market=model + rows["gap_bp"] / 100.0).drop(columns="gap_bp")


def ois_curve(level):
    months = np.arange(1, 61)
    return pd.Series(level + 0.3 * months / 60 - 0.1 * (months / 60) ** 2, index=months)


def world(root, poison_after=None):
    """The synthetic world under `root`; with `poison_after`, every mark, z and overnight rate dated after it is
    POISON (the FX quote FX_POISON), and every mark before it is revised to POISON later than it was final."""
    rng = np.random.default_rng(11)
    usd_days, gbp_days = sessions_of("USD"), sessions_of("GBP")
    fed, uk = BDAYS[USD["calendar"]], BDAYS[GBP["calendar"]]
    marks = {}

    # USD: ZQ settles by expiration month, and Treasury par yields at six tenors (none on one session)
    level = 1.0 + np.cumsum(rng.normal(0, 0.03, N))
    months = [pd.period_range(d.to_period("M"), periods=30, freq="M") for d in usd_days]
    zq = pd.DataFrame({"date": np.repeat(usd_days, 30), "contract": [f"C{m}" for ms in months for m in ms],
                       "expiration": [m.end_time.normalize() for ms in months for m in ms],
                       "value": (100 - (level[:len(usd_days), None] + 0.03 * np.arange(30))).ravel(), "is_final": True})
    marks["databento", "USD", "ZQ"] = vintages(zq["date"], zq["value"].to_numpy(), fed).assign(
        contract=zq["contract"], expiration=zq["expiration"], is_final=True)
    treasury = usd_days[usd_days != T("2021-10-11")]
    for tenor, series in USD["expression"]["curve"]["series"].items():
        y = 1.5 + 0.25 * tenor + np.cumsum(rng.normal(0, 0.04, len(treasury)))
        marks["fred", "USD", series] = vintages(treasury, y, fed)

    # GBP: the OIS curve (monthly), the gilt spot curve (half-yearly, the 0.5y node missing some days), FX
    level = 0.5 + np.cumsum(rng.normal(0, 0.03, len(gbp_days)))
    curves = pd.DataFrame([ois_curve(lvl) for lvl in level], index=gbp_days)
    marks["boe", "GBP", "OIS_SPOT"] = curve_log(gbp_days, curves, uk)
    tenors = np.arange(6, 121, 6)
    glc = pd.DataFrame(level[:, None] + 0.8 + 0.02 * tenors / 12 + rng.normal(0, 0.01, (len(gbp_days), len(tenors))),
                       index=gbp_days, columns=tenors)
    glc.loc[glc.index[::7], 6] = np.nan
    marks["boe", "GBP", "GLC_SPOT"] = curve_log(gbp_days, glc, uk)
    fx_days = gbp_days[gbp_days != T("2021-11-11")]
    marks["fred", "GBP", "DEXUSUK"] = vintages(fx_days, 1.3 * np.exp(np.cumsum(rng.normal(0, 0.005, len(fx_days)))),
                                               None)

    # the panels: sessions, meetings (GBP's rates are the path's, off the OIS curve), signals
    usd_sessions = pd.DataFrame({"session": usd_days, "error": None, "rate_now": 0.9})
    in_force = 0.4 + 0.1 * np.sin(np.arange(len(gbp_days)) / 50)
    gbp_sessions = pd.DataFrame({"session": gbp_days, "error": None, "rate_now": in_force, "rate_in_force": in_force})
    rates = []
    for d, (_, c), r in zip(gbp_days, curves.iterrows(), in_force):
        ahead = EFFECTIVE[EFFECTIVE > d]
        path = forward_path(c, d, ahead[:8], ahead[8], 365, last_fixing=r, pin_always=True)
        rates.append(path.reindex(ahead[:8]).to_numpy())
    panels = {"USD": {"sessions": usd_sessions, "meetings": meetings_of(usd_days), "signal": signal_of(usd_days, 1)},
              "GBP": {"sessions": gbp_sessions, "meetings": meetings_of(gbp_days, rates), "signal": signal_of(gbp_days, 2)}}
    for seed, tables in enumerate(panels.values(), start=3):
        tables["paths"] = paths_of(tables["signal"], tables["meetings"], seed)

    if poison_after is not None:
        for key, frame in marks.items():
            late = frame["date"] > poison_after
            bad = FX_POISON if key[2] == GBP["expression"]["fx"]["series"] else POISON
            revised = frame[~late].assign(value=bad, published=frame.loc[~late, "published"] + pd.Timedelta(days=10))
            marks[key] = pd.concat([frame.assign(value=frame["value"].where(~late, bad)), revised], ignore_index=True)
        for tables in panels.values():
            s = tables["signal"]
            s.loc[s["session"] > poison_after, ["gap_bp", "z", "window_mean_bp", "window_sd_bp"]] = POISON
            o = tables["sessions"]
            o.loc[o["session"] > poison_after, [c for c in ("rate_now", "rate_in_force") if c in o]] = POISON
    for (source, ccy, series), frame in marks.items():
        write(root, source, ccy, series, frame)
    return expression.World(BOOK, root=Path(root), panels=panels, calendars={"USD": calendar(), "GBP": calendar()})


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    w = world(tmp_path_factory.mktemp("cache"))
    return w, expression.build(w)


def test_every_sleeve_is_built_on_its_sessions(built):
    w, sleeves = built
    assert list(sleeves) == [s["name"] for s in BOOK["sleeves"]]
    usd = set(w.panel("USD", "signal")["session"])
    gbp = set(w.panel("GBP", "signal")["session"])
    for s in sleeves.values():
        want = usd & gbp if s.kind == "cross" else (usd if s.ccys == ("USD",) else gbp)
        assert set(s.sessions) == want
    legs = {label: sign for label, sign, _ in sleeves["USD 2s10s"].legs}
    assert legs == {"USD 2y": 1, "USD 10y": -1}


def test_the_checks_hold_on_every_sleeve_both_ways(built):
    _, sleeves = built
    table = pd.concat([expression.checks(s) for s in sleeves.values()])
    assert (table["sessions"] > 500).all()
    assert (table["identity_max_bp"] < carry.TOL_BP).all()
    assert (table["outside_bound"] == 0).all()


def test_pnl_is_the_held_position_times_the_unit(built):
    _, sleeves = built
    s = sleeves["GBP - USD 2y"]
    legs, sessions = expression.run(s, expression.linear(s))
    held = s.component["z"].shift(s.lag + 1)
    unit = expression.unit(s).groupby("session")["total"].sum(min_count=1)
    assert np.allclose(sessions["pnl"], (held * unit).fillna(0.0), atol=1e-12)
    gbp = legs[legs["leg"] == "GBP 2y"].set_index("session")
    assert np.allclose(gbp["native"], gbp["q"] / (s.legs[0][2].frame["spot"] * s.legs[0][2].frame["dv01"]),
                       equal_nan=True)


def test_the_gbp_outright_reproduces_the_crude_backtest(built):
    w, sleeves = built
    backtest = policy.run(w.panel("GBP", "signal"), w.panel("GBP", "sessions"), w.panel("GBP", "meetings"),
                          GBP["backtest"]["horizon"], GBP["backtest"]).reset_index(names="session")
    n, worst = expression.backtest_residual(sleeves["GBP outright"], backtest)
    assert n > 300 and worst < 1e-9


def test_a_stale_mark_books_no_rate_move_and_still_carries(built):
    _, sleeves = built
    u = expression.unit(sleeves["USD 2s10s"]).set_index("session")
    day = u.loc[T("2021-10-11")]
    assert day["stale"].all() and (day["rate"] == 0).all() and (day["carry"] != 0).all()
    fx = expression.unit(sleeves["GBP outright"]).set_index("session")
    assert fx.loc[T("2021-11-11"), "fx_stale"] and fx["fx_stale"].sum() == 1


def test_nothing_credited_through_d_sees_later_marks_z_or_overnight_rates(built, tmp_path):
    _, clean = built
    poisoned = expression.build(world(tmp_path, poison_after=CUT))
    h = BOOK["carry"]["horizon_days"]
    for name, s in clean.items():
        p = poisoned[name]
        a, b = expression.run(s, expression.linear(s))[0], expression.run(p, expression.linear(p))[0]
        pd.testing.assert_frame_equal(a[a["session"] <= CUT], b[b["session"] <= CUT])
        a, b = expression.ahead(s, h, PAIRS), expression.ahead(p, h, PAIRS)
        pd.testing.assert_frame_equal(a.loc[:CUT], b.loc[:CUT])
        assert not a.loc[CUT:].iloc[1:].equals(b.loc[CUT:].iloc[1:])      # the poison is there to be seen
        # not vacuous: the curve sleeves trade before D, the others have a live breakeven there
        live = a.loc[:CUT, "z" if s.kind == "curve" else "pays"].notna().sum()
        assert live > (30 if s.kind == "curve" else 100), (name, live)


def cached(w, source, ccy, series):
    """A series' log as the synthetic cache holds it, one row per print (no revisions: the clean world)."""
    return pd.read_parquet(Path(w.root) / source / ccy / f"{series}.parquet")


def on_or_before(prints, days):
    """Each session's print: the last dated on or before it."""
    return prints.sort_index().reindex(prints.index.union(days)).ffill().reindex(days)


def test_the_gbp_spot_is_usd_per_gbp_dated_on_the_session(built):
    w, sleeves = built
    fx = GBP["expression"]["fx"]
    quote = cached(w, fx["source"], "GBP", fx["series"]).set_index("date")["value"]
    for name in ("GBP outright", "GBP - USD 2y"):
        frame = next(leg for _, _, leg in sleeves[name].legs if leg.ccy == "GBP").frame
        want = on_or_before(quote, frame.index)
        assert np.allclose(frame["spot"], want) and frame["spot"].between(1.0, 2.0).all()   # about 1.3, not 0.77
        assert np.allclose(frame["fx"].iloc[1:], (want / want.shift(1)).iloc[1:])


def test_a_par_leg_marks_the_print_dated_on_the_session(built):
    w, sleeves = built
    x = USD["expression"]["curve"]
    for tenor, (_, _, leg) in zip(x["legs"], sleeves["USD 2s10s"].legs):
        prints = cached(w, x["source"], "USD", x["series"][tenor]).set_index("date")["value"]
        assert np.allclose(leg.frame["rate"], on_or_before(prints, leg.frame.index))


def zq_month(e):
    """The first calendar month starting on or after an effective date, by hand."""
    return e.to_period("M") if e.day == 1 else e.to_period("M") + 1


def test_the_zq_leg_is_the_kth_meetings_month_at_its_settle_dated_on_the_session(built):
    w, sleeves = built
    frame = sleeves["USD outright"].legs[0][2].frame
    k = USD["backtest"]["horizon"]
    x = USD["expression"]["outright"]
    settles = cached(w, USD["market"]["futures"]["source"], "USD", x["contract"])
    settles = settles.assign(month=settles["expiration"].dt.to_period("M")).pivot(
        index="date", columns="month", values="value")
    for day in frame.index[::37]:
        month = zq_month(EFFECTIVE[EFFECTIVE > day][k - 1])
        assert frame.loc[day, "id"] == str(month)
        assert frame.loc[day, "rate"] == pytest.approx(100 - settles.loc[day, month])


def month_weights(day, k):
    """By hand from the meeting dates: the share of the k-th meeting's ZQ month in regimes k and k + 1."""
    ahead = EFFECTIVE[EFFECTIVE > day]
    month = zq_month(ahead[k - 1])
    start, end = month.start_time, (month + 1).start_time
    n, nxt = (end - start).days, ahead[k]
    if start < nxt < end:
        return {k: (nxt - start).days / n, k + 1: (end - nxt).days / n}
    return {k: 1.0}


def test_an_outrights_edge_is_the_day_weighted_de_meaned_gap_of_its_instrument(built):
    w, sleeves = built
    for ccy in ("USD", "GBP"):
        k = config.currency(ccy)["backtest"]["horizon"]
        s = sleeves[f"{ccy} outright"]
        dev = expression.dev_by_k(w.panel(ccy, "signal"))
        straddling = 0
        for day in s.dev_bp.dropna().index[::5]:
            weights = month_weights(day, k) if s.legs[0][2].kind == "futures_month" else {k: 1.0}
            straddling += len(weights) > 1
            assert s.dev_bp.loc[day] == pytest.approx(sum(v * dev.loc[day, j] for j, v in weights.items()), abs=1e-9)
        assert ccy == "GBP" or straddling > 10                    # the straddling months are exercised


def test_the_cross_sleeve_receives_the_first_currency_on_first_minus_second(built):
    w, sleeves = built
    s = sleeves["GBP - USD 2y"]
    first, second = next(x["pair"] for x in BOOK["sleeves"] if x["name"] == s.name)
    assert [(sign, leg.ccy) for _, sign, leg in s.legs] == [(1, first), (-1, second)]
    k = USD["backtest"]["horizon"]
    gap = {c: w.panel(c, "signal").query("k == @k").set_index("session")["gap_bp"] for c in (first, second)}
    assert np.allclose(s.component["value_bp"], (gap[first] - gap[second]).reindex(s.sessions), equal_nan=True)


def test_ahead_reads_phi_from_the_components_own_closure_after_min_pairs(built):
    _, sleeves = built
    s = sleeves["USD outright"]
    h = BOOK["carry"]["horizon_days"]
    for pairs in (PAIRS, 250):
        want = carry.closure(s.component["value_bp"], s.component["dev_bp"], h, pairs)
        assert np.allclose(expression.ahead(s, h, pairs)["phi_h"], want, equal_nan=True)
    assert expression.ahead(s, h, PAIRS)["phi_h"].notna().sum() > expression.ahead(s, h, 250)["phi_h"].notna().sum()


def test_the_zq_dv01_is_derived_from_the_contract_spec(built):
    _, sleeves = built
    spec = USD["contracts"][USD["expression"]["outright"]["contract"]]
    frame = sleeves["USD outright"].legs[0][2].frame
    derived = spec["notional"] * spec["accrual"]["days"] / spec["accrual"]["year_days"] * 1e-4
    assert (frame["dv01"] == derived).all()


ROUNDED_DV01 = re.compile(r"41\.6(7|66)")


def code_tokens(path):
    """Every number and string in `path`'s code, docstrings and comments skipped."""
    text = path.read_text()
    docstrings = set()
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                docstrings.add((first.lineno, first.col_offset))
    return [tok.string for tok in tokenize.generate_tokens(io.StringIO(text).readline)
            if tok.type in (tokenize.NUMBER, tokenize.STRING, tokenize.FSTRING_MIDDLE) and tok.start not in docstrings]


def test_no_code_token_types_the_zq_dv01(tmp_path):
    hits = {str(p.relative_to(SRC)): t for p in SRC.rglob("*.py") for t in code_tokens(p) if ROUNDED_DV01.search(t)}
    assert not hits
    probe = tmp_path / "probe.py"
    probe.write_text('"""DV01 is $41.67."""\n# 41.6667\nx = 41.67\n')
    assert [t for t in code_tokens(probe) if ROUNDED_DV01.search(t)] == ["41.67"]
