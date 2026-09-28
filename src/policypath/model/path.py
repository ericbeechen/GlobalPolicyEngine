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
"""

import numpy as np
import pandas as pd
from policypath import config
from policypath.model.reaction import floor_at_elb, inertial_path, lower_bound, notional
from policypath.model.rstar import rstar
from policypath.sources import cache

DAY = pd.Timedelta(days=1)


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


def inputs(ccy, root=cache.CACHE_DIR):
    """The rule's non-macro inputs from the cache, as full vintage logs.

    ``target`` is the policy rate in force (`policy_rate_log`); ``sep`` is None
    where r* is a constant.
    """
    cfg = config.currency(ccy)
    rule = cfg["rule"]
    rs = rule["rstar"]
    return {"sep": None if "constant" in rs else cache.log(rs["source"], rs["series"], ccy, root),
            "target": policy_rate_log(rule["policy_rate"], ccy, root),
            "fixings": cache.log(cfg["overnight"]["source"], cfg["overnight"]["series"], ccy, root)}


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


def model_path(as_of, effective_dates, macro, sep, target, fixings, spec):
    """The rule's path over `effective_dates` from session `as_of`, in the overnight rate's terms.

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
    unconstrained = notional(pi, gap, r["rstar"], spec, as_of)
    elb = lower_bound(spec, as_of)
    goal = floor_at_elb(unconstrained, elb)
    seen = known(target, as_of)
    r0, r0_date = policy_rate(target, as_of, seen)
    spread = operating_spread(fixings, target, as_of, spec["spread_fixings"], spec.get("spread_breaks", ()), seen)
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
    # Every session reads the same two logs: sorted once here rather than on each read.
    target, fixings = cache.Presorted(target), cache.Presorted(fixings)
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
