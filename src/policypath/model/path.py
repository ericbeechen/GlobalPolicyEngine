"""The model path next to the market path, one session at a time.

On session D the model reads only what was public by the end of D: the nowcast
as of D, the latest SEP, the target range in force on D and the fixings
published by D. Its meetings are the market path's own pillars, so the two
paths are compared meeting by meeting on one calendar.

The rule works in the Fed's own terms (the midpoint of the target range). ZQ
settles on EFFR, so the model path is moved onto EFFR by the operating spread
between them, held flat like the macro inputs.
"""

import numpy as np
import pandas as pd
from policypath import config
from policypath.model.reaction import floor_at_elb, inertial_path, lower_bound, notional
from policypath.model.rstar import rstar
from policypath.sources import cache

DAY = pd.Timedelta(days=1)


def known(frame, as_of):
    """``value`` by date as published by the end of `as_of`, the latest vintage of each date."""
    f = frame[frame["published"] < pd.Timestamp(as_of).normalize() + DAY]
    return f.sort_values(["date", "published"]).drop_duplicates("date", keep="last").set_index("date")


def target_range(lower, upper):
    """The two bounds as one midpoint series: date, value, published."""
    both = lower[["date", "value", "published"]].merge(
        upper[["date", "value", "published"]], on="date", suffixes=("_lo", "_hi"))
    return pd.DataFrame({"date": both["date"], "value": (both["value_lo"] + both["value_hi"]) / 2,
                         "published": both[["published_lo", "published_hi"]].max(axis=1)})


def inputs(ccy, root=cache.CACHE_DIR):
    """The rule's non-macro inputs from the cache, as full vintage logs.

    A policy rate with one value rather than a range (Bank Rate) lists the same
    series as both bounds. ``sep`` is None where r* is a constant.
    """
    cfg = config.currency(ccy)
    rule = cfg["rule"]
    source, (lo, hi) = rule["target_range"]["source"], rule["target_range"]["series"]
    rs = rule["rstar"]
    return {"sep": None if "constant" in rs else cache.log(rs["source"], rs["series"], ccy, root),
            "target": target_range(cache.log(source, lo, ccy, root), cache.log(source, hi, ccy, root)),
            "fixings": cache.log(cfg["overnight"]["source"], cfg["overnight"]["series"], ccy, root)}


def policy_rate(target, as_of):
    """The midpoint in force on `as_of`, and the date it is for. Announced before the day starts."""
    t = known(target, as_of)["value"]
    t = t[t.index <= pd.Timestamp(as_of)]
    if t.empty:
        raise ValueError(f"no target range known on {pd.Timestamp(as_of).date()}")
    return t.iloc[-1], t.index[-1]


def operating_spread(fixings, target, as_of, n, breaks=()):
    """The overnight rate minus the policy rate, median over the last `n` fixings published by `as_of`.

    A median, since one fixing is not a level: in 2015-17 EFFR printed 5-12bp
    low on most month ends. `breaks` are dates the overnight rate's definition
    changed (SONIA's reform, 2018-04-23): fixings from before the latest break on
    or before `as_of` are not used, so the spread is estimated either side of it.
    Until the first fixing after a break is published, the estimate from before it stands.
    """
    f = known(fixings, as_of)["value"]
    passed = [pd.Timestamp(b) for b in breaks if pd.Timestamp(b) <= pd.Timestamp(as_of)]
    if passed and (f.index >= max(passed)).any():
        f = f[f.index >= max(passed)]
    f = f.tail(n)
    mid = known(target, as_of)["value"].reindex(f.index)
    spread = (f - mid).dropna()
    if spread.empty:
        raise ValueError(f"no fixing with a known target range by {pd.Timestamp(as_of).date()}")
    return float(spread.median())


def model_path(as_of, effective_dates, macro, sep, target, fixings, spec):
    """The rule's path over `effective_dates` from session `as_of`, in EFFR terms.

    `macro` is the nowcast as of the session (a row of ``nowcast.build``), `sep`,
    `target` and `fixings` are logs with ``published``; `spec` is the rule
    block. Returns (summary, frame of k, effective_date, model, model_mid).
    """
    as_of = pd.Timestamp(as_of)
    if pd.Timestamp(macro["as_of"]) > as_of:
        raise ValueError(f"nowcast as of {macro['as_of']:%Y-%m-%d} is after the session {as_of.date()}")
    r = rstar(sep, as_of, spec)
    # Hold-flat conditioning: inflation, the gap and r* stay at today's values at
    # every meeting ahead, so R* is one number for the whole path. The path answers
    # "if the macro picture does not change, where does the rule take policy?".
    pi, gap = macro[spec["inflation"]], macro[spec["gap"]]
    unconstrained = notional(pi, gap, r["rstar"], spec)
    elb = lower_bound(spec, as_of)
    goal = floor_at_elb(unconstrained, elb)
    r0, r0_date = policy_rate(target, as_of)
    spread = operating_spread(fixings, target, as_of, spec["spread_fixings"], spec.get("spread_breaks", ()))
    dates = pd.DatetimeIndex(effective_dates)
    mid = inertial_path(r0, goal, len(dates), spec, as_of)
    frame = pd.DataFrame({"k": np.arange(1, len(dates) + 1), "effective_date": dates,
                          "model": mid + spread, "model_mid": mid})
    summary = {"macro_as_of": pd.Timestamp(macro["as_of"]), "inflation": pi, "u_gap": gap, **r,
               "notional": unconstrained, "at_elb": unconstrained < elb, "elb": elb, "goal": goal,
               "r0": r0, "r0_date": r0_date, "spread_bp": spread * 100.0,
               "macro_published": pd.Timestamp(macro["published"])}
    return summary, frame


def build(sessions, meetings, macro, sep, target, fixings, spec):
    """The model path on every solved session the nowcast covers.

    `sessions` and `meetings` are the panel's frames, `macro` the nowcast panel.
    Each session takes the latest nowcast on or before it: a Fed holiday CME
    trades (Columbus, Veterans Day) takes the business day before's, when
    nothing new was published. Returns (summaries, paths): one row per session,
    and one per session and meeting with ``market`` and ``model`` side by side.
    """
    ok = sessions[sessions["error"].isna()][["session"]].sort_values("session")
    ok = ok.astype({"session": "datetime64[ns]"})
    macro = macro.sort_values("as_of").astype({"as_of": "datetime64[ns]"})
    ok = pd.merge_asof(ok, macro.rename(columns={"as_of": "macro_as_of"}), left_on="session",
                       right_on="macro_as_of", direction="backward")
    ok = ok[ok["macro_as_of"].notna()]
    by = dict(tuple(meetings.groupby("session")))
    summaries, frames = [], []
    for row in ok.to_dict("records"):
        day = row["session"]
        m = by[day].sort_values("k")
        summary, frame = model_path(day, m["effective_date"], {**row, "as_of": row["macro_as_of"]},
                                    sep, target, fixings, spec)
        summaries.append({"session": day, **summary})
        frames.append(m[["session", "k", "announcement_date", "effective_date"]]
                      .assign(market=m["rate"].to_numpy(), model=frame["model"].to_numpy(),
                              model_mid=frame["model_mid"].to_numpy()))
    return pd.DataFrame(summaries), pd.concat(frames, ignore_index=True)
