"""The model path next to the market path, one session at a time.

On session D the model reads only what was public by the end of D: the nowcast
as of D, the latest SEP, the target range in force on D and the fixings
published by D. Its meetings are the market path's own pillars, so the two
paths are compared meeting by meeting on one calendar.

The rule works in the central bank's own terms: its policy rate, which the
config gives as one rate (Bank Rate), a range read at its midpoint (the Fed's
target range), or a dated schedule of either where the operative rate changed
inside the sample (the ECB's MRO, then its deposit rate). The market settles on
the overnight rate, so the model path is moved onto it by the operating spread
between the two, held flat like the macro inputs.

Two robustness variants are optional keys of the rule block, and
without them the path is built exactly as before. ``conditioning: {converge:
{half_life_quarters: h}}`` lets inflation close on its target and the
unemployment gap on zero, each halving its distance every h quarters, so the
rule's goal moves meeting by meeting (`converge_goals`). ``estimate:
{prior_quarters: n0}`` replaces the imposed coefficients on each session with
ones estimated from the quarters before it (`model/estimate.py`).

A third, ``projected: {source, series, label}``, swaps the rule's path for the
central bank's own projection of its policy rate (the FOMC's SEP median, say),
as published by the session (`projected_path`). The summary is still the
rule's, so the ELB state, and with it the sessions the book counts, do not move.
"""

import numpy as np
import pandas as pd
from policypath import config
from policypath.model import estimate
from policypath.model.reaction import floor_at_elb, inertial_path, lower_bound, notional, on
from policypath.model.rstar import rstar
from policypath.sources import cache

DAY = pd.Timedelta(days=1)
QUARTER_DAYS = 365.25 / 4


def known(frame, as_of):
    """``value`` by date as published by the end of `as_of`, the latest vintage of each date.

    `frame` is a log (date, value, published), or a `cache.Presorted` of one, which gives the same answer faster.
    """
    if isinstance(frame, cache.Presorted):
        return frame.latest(as_of)
    f = frame[frame["published"] < pd.Timestamp(as_of).normalize() + DAY]
    return f.sort_values(["date", "published"]).drop_duplicates("date", keep="last").set_index("date")


def target_range(lower, upper):
    """The two bounds as one midpoint series: date, value, published."""
    both = lower[["date", "value", "published"]].merge(
        upper[["date", "value", "published"]], on="date", suffixes=("_lo", "_hi"))
    return pd.DataFrame({"date": both["date"], "value": (both["value_lo"] + both["value_hi"]) / 2,
                         "published": both[["published_lo", "published_hi"]].max(axis=1)})


def policy_rate_log(spec, ccy, root=cache.CACHE_DIR):
    """The config's ``rule.policy_rate`` as one log: date, value, published.

    One series is the rate; two are a range, read at its midpoint. A dated
    schedule ``[{from, source, series}, ...]`` takes each entry's rate from its
    ``from`` date until the next entry's.
    """
    entries = spec if isinstance(spec, list) else [spec]
    frames = []
    for i, entry in enumerate(entries):
        names = [entry["series"]] if isinstance(entry["series"], str) else entry["series"]
        logs = [cache.log(entry["source"], name, ccy, root) for name in names]
        if len(logs) == 1:
            rate = logs[0][["date", "value", "published"]]
        elif len(logs) == 2:
            rate = target_range(*logs)
        else:
            raise ValueError(f"a policy rate is one series or a range of two, not {names}")
        if "from" in entry:
            rate = rate[rate["date"] >= pd.Timestamp(entry["from"])]
        if i + 1 < len(entries):
            rate = rate[rate["date"] < pd.Timestamp(entries[i + 1]["from"])]
        frames.append(rate)
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0].reset_index(drop=True)


def inputs(ccy, root=cache.CACHE_DIR, rule=None):
    """The rule's non-macro inputs from the cache, as full vintage logs.

    ``target`` is the policy rate in force (`policy_rate_log`); ``sep`` is the
    log r* is read from (the SEP's, or HLW's under a robustness override), None
    where r* is a constant; ``projected`` the real-time vintages of the
    projected policy rate under ``rule.projected``, else None. `rule` replaces
    the config's ``rule`` block (a variant's, merged by the caller); by default
    the config's own.
    """
    cfg = config.currency(ccy)
    rule = cfg["rule"] if rule is None else rule
    rs, pr = rule["rstar"], rule.get("projected")
    return {"sep": None if "constant" in rs else cache.log(rs["source"], rs["series"], ccy, root),
            "target": policy_rate_log(rule["policy_rate"], ccy, root),
            "fixings": cache.log(cfg["overnight"]["source"], cfg["overnight"]["series"], ccy, root),
            "projected": None if pr is None else cache.vintages(pr["source"], pr["series"], ccy, root)}


def policy_rate(target, as_of, seen=None):
    """The policy rate in force on `as_of`, and the date it is for. Announced before the day starts.

    `seen` is ``known(target, as_of)`` where the caller has it already.
    """
    t = (known(target, as_of) if seen is None else seen)["value"]
    t = t[t.index <= pd.Timestamp(as_of)]
    if t.empty:
        raise ValueError(f"no target range known on {pd.Timestamp(as_of).date()}")
    return t.iloc[-1], t.index[-1]


def operating_spread(fixings, target, as_of, n, breaks=(), seen=None):
    """The overnight rate minus the policy rate, median over the last `n` fixings published by `as_of`.

    A median, since one fixing is not a level: in 2015-17 EFFR printed 5-12bp
    low on most month ends. `breaks` are dates the overnight rate's definition
    changed (SONIA's reform, 2018-04-23): fixings from before the latest break on
    or before `as_of` are not used, so the spread is estimated either side of it.
    Until the first fixing after a break is published, the estimate from before it stands.
    `seen` is ``known(target, as_of)`` where the caller has it already.
    """
    f = known(fixings, as_of)["value"]
    passed = [pd.Timestamp(b) for b in breaks if pd.Timestamp(b) <= pd.Timestamp(as_of)]
    if passed and (f.index >= max(passed)).any():
        f = f[f.index >= max(passed)]
    f = f.tail(n)
    mid = (known(target, as_of) if seen is None else seen)["value"].reindex(f.index)
    spread = (f - mid).dropna()
    if spread.empty:
        raise ValueError(f"no fixing with a known target range by {pd.Timestamp(as_of).date()}")
    return float(spread.median())


def converge_goals(as_of, dates, inflation, u_gap, r_star, spec):
    """The rule's floored goal at each meeting in `dates` under converge-to-target conditioning.

    Inflation closes on its target and the unemployment gap on zero, each
    halving its distance every ``conditioning.converge.half_life_quarters``:
    x_k = x* + (x_0 - x*) 0.5 ** (q_k / h), with q_k the quarters (91.3125 days)
    from `as_of` to meeting k. r* stays flat. Every dated schedule (the target,
    the floor) is read as it stood on `as_of`: a change the config dates after
    it was not known then, so it cannot move the path.
    """
    half_life = spec["conditioning"]["converge"]["half_life_quarters"]
    decay = 0.5 ** ((dates - as_of).days.to_numpy() / QUARTER_DAYS / half_life)
    pi_star = on(spec["inflation_target"], as_of)
    rates = notional(pi_star + (inflation - pi_star) * decay, u_gap * decay, r_star, spec, as_of)
    return np.maximum(rates, lower_bound(spec, as_of))


def projected_path(as_of, dates, r0, vintages):
    """The central bank's own projection of its policy rate at each of `dates`, as published by the end of `as_of`.

    `vintages` (date, value, realtime_start, realtime_end) holds year-end
    projections, each dated on the first day of its year, as an annual
    series is, and current from ``realtime_start`` to ``realtime_end``
    inclusive (NaT while current). The path runs linearly in time from `r0`,
    the rate in force on `as_of`, through each year's projection on 31
    December. A year counts only while one of `dates` still falls in it: once
    its last meeting is past, its year-end rate is the rate in force, not the
    projection. Past the last year projected the path is flat. All NaN if
    nothing has been published by `as_of`.
    """
    as_of = pd.Timestamp(as_of)
    dates = pd.DatetimeIndex(dates)
    live = vintages[(vintages["realtime_start"] <= as_of)
                    & (vintages["realtime_end"].isna() | (vintages["realtime_end"] >= as_of))
                    & vintages["value"].notna()]
    if live.empty:
        return np.full(len(dates), np.nan)
    ends = pd.DatetimeIndex(pd.to_datetime(live["date"].dt.year.astype(str) + "-12-31"))
    keep = (ends > as_of) & (ends.year >= dates.min().year)   # this year's only if a meeting of it is ahead
    anchors = pd.Series(live["value"].to_numpy()[keep], index=ends[keep]).sort_index()
    if anchors.empty:
        return np.full(len(dates), r0)
    days = np.array([0.0, *(anchors.index - as_of).days])
    rates = np.array([r0, *anchors.to_numpy()])
    return np.interp((dates - as_of).days.to_numpy().astype(float), days, rates)


def model_path(as_of, effective_dates, macro, sep, target, fixings, spec, projected=None):
    """The rule's path over `effective_dates` from session `as_of`, in the overnight rate's terms.

    `macro` is the nowcast as of the session (a row of ``nowcast.build``), `sep`,
    `target` and `fixings` are logs with ``published``; `spec` is the rule
    block, and `projected` the vintages ``spec["projected"]`` names (`inputs`).
    Returns (summary, frame of k, effective_date, model, model_mid).
    The summary is today's picture under either conditioning: its notional
    and goal are the rule's at today's inflation and gap. ``at_elb`` says the
    goal is on the floor at every meeting ahead, so a rate on the floor stays
    there and the path has no view: under hold-flat, today's notional below the
    floor; under converge, every meeting's goal on it (the ELB state moves with
    the conditioning as it does with r*).
    """
    as_of = pd.Timestamp(as_of)
    if pd.Timestamp(macro["as_of"]) > as_of:
        raise ValueError(f"nowcast as of {macro['as_of']:%Y-%m-%d} is after the session {as_of.date()}")
    r = rstar(sep, as_of, spec)
    # Hold-flat conditioning, unless the rule block says otherwise: inflation, the gap
    # and r* stay at today's values at every meeting ahead, so R* is one number for the
    # whole path. The path answers "if the macro picture does not change, where does
    # the rule take policy?".
    pi, gap = macro[spec["inflation"]], macro[spec["gap"]]
    unconstrained = notional(pi, gap, r["rstar"], spec, as_of)
    elb = lower_bound(spec, as_of)
    goal = floor_at_elb(unconstrained, elb)
    seen = known(target, as_of)
    r0, r0_date = policy_rate(target, as_of, seen)
    spread = operating_spread(fixings, target, as_of, spec["spread_fixings"], spec.get("spread_breaks", ()), seen)
    dates = pd.DatetimeIndex(effective_dates)
    if "conditioning" in spec:
        goals = converge_goals(as_of, dates, pi, gap, r["rstar"], spec)
        at_elb = bool((goals <= elb).all())   # the goals are floored: on the floor at every meeting
    else:
        goals, at_elb = goal, unconstrained < elb
    if "projected" in spec:
        mid = projected_path(as_of, dates, r0, projected)
    else:
        mid = inertial_path(r0, goals, len(dates), spec, as_of)
    frame = pd.DataFrame({"k": np.arange(1, len(dates) + 1), "effective_date": dates,
                          "model": mid + spread, "model_mid": mid})
    summary = {"macro_as_of": pd.Timestamp(macro["as_of"]), "inflation": pi, "u_gap": gap, **r,
               "notional": unconstrained, "at_elb": at_elb, "elb": elb, "goal": goal,
               "r0": r0, "r0_date": r0_date, "spread_bp": spread * 100.0,
               "macro_published": pd.Timestamp(macro["published"])}
    return summary, frame


def build(sessions, meetings, macro, sep, target, fixings, spec, projected=None):
    """The model path on every solved session the nowcast covers.

    `sessions` and `meetings` are the panel's frames, `macro` the nowcast panel.
    Each session takes the latest nowcast on or before it: a Fed holiday CME
    trades (Columbus, Veterans Day) takes the business day before's, when
    nothing new was published. Returns (summaries, paths): one row per session,
    and one per session and meeting with ``market`` and ``model`` side by side.

    With ``estimate`` in `spec`, the imposed rule runs first; its quarter ends
    give each session's coefficients (`estimate.coefficients`), and the rule
    runs again with them. The summaries then carry the estimates. `projected`
    is what ``rule.projected`` reads (`inputs`), where `spec` has it.
    """
    ok = sessions[sessions["error"].isna()][["session"]].sort_values("session")
    ok = ok.astype({"session": "datetime64[ns]"})
    macro = macro.sort_values("as_of").astype({"as_of": "datetime64[ns]"})
    ok = pd.merge_asof(ok, macro.rename(columns={"as_of": "macro_as_of"}), left_on="session",
                       right_on="macro_as_of", direction="backward")
    ok = ok[ok["macro_as_of"].notna()]
    by = dict(tuple(meetings.groupby("session")))
    # Every session reads the same two logs: sorted once here rather than on each read.
    target, fixings = cache.Presorted(target), cache.Presorted(fixings)
    summaries, paths = _run(ok, by, sep, target, fixings, spec, projected=projected)
    if "estimate" not in spec:
        return summaries, paths
    coefficients = estimate.coefficients(summaries, spec)
    summaries, paths = _run(ok, by, sep, target, fixings, spec, coefficients, projected)
    return summaries.merge(coefficients, on="session", how="left"), paths


def _run(ok, by, sep, target, fixings, spec, coefficients=None, projected=None):
    """`model_path` on every row of `ok`, with the imposed coefficients or each session's from `coefficients`."""
    rules = None if coefficients is None else {
        day: {**spec, "coefficients": {"inflation_gap": a, "unemployment_gap": b}}
        for day, a, b in coefficients[["session", "coef_inflation_gap", "coef_unemployment_gap"]]
        .itertuples(index=False)}
    summaries, frames = [], []
    for row in ok.to_dict("records"):
        day = row["session"]
        m = by[day].sort_values("k")
        summary, frame = model_path(day, m["effective_date"], {**row, "as_of": row["macro_as_of"]},
                                    sep, target, fixings, spec if rules is None else rules[day], projected)
        summaries.append({"session": day, **summary})
        frames.append(m[["session", "k", "announcement_date", "effective_date"]]
                      .assign(market=m["rate"].to_numpy(), model=frame["model"].to_numpy(),
                              model_mid=frame["model_mid"].to_numpy()))
    return pd.DataFrame(summaries), pd.concat(frames, ignore_index=True)
