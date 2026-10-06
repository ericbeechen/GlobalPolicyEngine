"""The credit bridge's report, its one chart, and the numbers the note quotes. Generated, never hand-edited.

`scripts/build_credit.py` writes ``reports/credit_<ccy>.md``, the chart
``reports/figures/credit_<ccy>_{light,dark}.png`` and ``reports/results/credit.json``
(one block per currency). The note (``notes/credit_section.md``) quotes the
flat ``headline`` dict of that block by key, never a typed number. Its numbers
are strings formatted as the report prints them (signed where the sign is the
point), so the note and the report agree to the digit. A t that two places
would round onto the 5% line (1.96) gets three, so it cannot read as on it.
The first ``credit.exclude`` window is the one the headline keys and the chart
name; the tables show every window.

The chart carries the section in three panels: the primary spread through time
with the regime underneath (red is hiking, blue cutting, gray the ELB state, as
`figures.implied_paths` colours hikes and cuts; late hiking sits further out
on the theme's ramp than early, so darker on the light surface and lighter on
the dark, where the ramp runs outward to pale),
test 2 as its non-overlapping quarters (every 63rd session: each dot one
quarter's change against the z it started from) with the headline fit, and
test 3 as each hiking episode's slope with its 95% interval.
"""

import itertools
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from policypath import config, credit, regimes
from policypath.report.coverage import table
from policypath.report.figures import THEMES, _style, _title

STATE_WORDS = {"elb": "ELB", "early_hiking": "early hiking", "late_hiking": "late hiking",
               "hold_after_hikes": "hold after hikes", "cutting": "cutting", "hold_after_cuts": "hold after cuts"}
SAMPLE_WORDS = {"all": "every session with a z", "ex_elb": "without ELB sessions"}


# ---- json ----------------------------------------------------------------------------

def jsonable(x):
    """`x` with every number, date and frame in a form json writes: floats to 6 places, NaN to null, frames to records."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, pd.DataFrame):
        return [jsonable(r) for r in x.to_dict("records")]
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, pd.Timestamp):
        return None if pd.isna(x) else f"{x:%Y-%m-%d}"
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else round(float(x), 6)
    return x


ORDINALS = ("first", "second", "third", "fourth", "fifth")


def _whole(x):
    """`x` rounded to a whole number (an int), for counts and whole percentages."""
    return None if x is None or not np.isfinite(x) else int(round(float(x)))


def _fmt(x, places=2, sign=False):
    """`x` as the report and the note print it: `places` decimals, with its sign where `sign`."""
    return None if x is None or not np.isfinite(x) else f"{float(x):{'+' if sign else ''}.{places}f}"


def _t(x):
    """A t statistic, signed, to 2 places; to 3 where 2 would round it onto the 5% line, so +1.958 is not +1.96."""
    if x is None or not np.isfinite(x):
        return None
    return _fmt(x, 3 if f"{abs(x):.2f}" == f"{credit.Z95:.2f}" else 2, sign=True)


def _signs(slopes):
    """The signs of one phase's slopes over the hiking cycles, in words: "negative in both cycles", "negative in the first cycle and positive in the second"."""
    words = ["negative" if s < 0 else "positive" for s in slopes]
    n = len(words)
    if n == 0:
        return "not estimated"
    if len(set(words)) == 1:
        return f"{words[0]} in " + ("the one cycle" if n == 1 else "both cycles" if n == 2 else f"all {n} cycles")
    parts = [f"{w} in the {ORDINALS[i] if i < len(ORDINALS) else f'{i + 1}th'}" for i, w in enumerate(words)]
    parts[0] += " cycle"
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def _against_registered(slope):
    """Whether a test 2 slope has the sign D1 registered (negative), in words."""
    return None if not np.isfinite(slope) else "the registered sign" if slope < 0 else "the other sign"


def _month(day):
    return f"{day:%b %Y}"


def _sample_word(sample, cfg):
    if sample.startswith("ex_") and sample[3:] in cfg["credit"]["exclude"]:
        a, b = cfg["credit"]["exclude"][sample[3:]]
        return f"without {a} to {b}"
    return SAMPLE_WORDS[sample]


def _kernel_sentence(passes, checks, h):
    """Whether the headline's verdict survives the other two kernels (`credit.after_the_run`), in words."""
    u, b = checks["uniform"]["t"], checks["bartlett_2h"]["t"]
    tail = f"with Hansen-Hodrick's equal weights the t is {_t(u)}, and with Bartlett at lag {2 * h} it is {_t(b)}."
    if not passes:
        return tail[0].upper() + tail[1:]
    both = u <= -credit.Z95 and b <= -credit.Z95
    return (f"The pass does not depend on the kernel D1 registered: {tail}" if both else
            f"The pass depends on the kernel D1 registered: {tail}")


def _z_checks_words(slope, checks):
    """What clipping z and leaving out the floored sessions (`credit.after_the_run`) do to the headline `slope`, in words."""
    flatter = [c for c in ("z_clipped", "above_floor") if checks[c]["slope"] > slope]
    if not flatter:
        return "both more negative than the headline's, so the inflated z's do not carry it"
    if len(flatter) == 2:
        return "both less negative than the headline's, so the inflated z's carry part of it"
    what = {"z_clipped": "clipping z", "above_floor": "leaving out the floored sessions"}[flatter[0]]
    return f"{what} makes it less negative, so the inflated z's carry part of it"


def _cell_words(h, reg, sample, cfg):
    """One test 2 cell in words: "the gap in bp at h = 63, every session with a z"."""
    return f"{'z' if reg == 'z' else 'the gap in bp'} at h = {h}, {_sample_word(sample, cfg)}"


def headline(results, cfg):
    """The flat dict the note quotes (``{credit.<ccy>.headline.<key>}``): counts as ints, numbers as formatted strings, and generated words."""
    r, credit_cfg, labels = results, cfg["credit"], cfg["credit"]["labels"]
    p, h_long, h_short = r["primary"], max(credit.HORIZONS), min(credit.HORIZONS)
    s = r["sample"]
    out = {
        "k": r["k"], "h_short": h_short, "h_long": h_long, "weekly_lags": credit.WEEKLY_LAGS,
        "control_sessions": credit.CONTROL_SESSIONS, "min_weeks": credit.MIN_WEEKS,
        "week_ends": credit_cfg["week_ends"], "primary_label": labels[p],
        "first_week": f"{s['first_week']:%Y-%m-%d}", "last_week": f"{s['last_week']:%Y-%m-%d}", "weeks": s["weeks"],
        "first_session": f"{s['first_session']:%Y-%m-%d}", "last_session": f"{s['last_session']:%Y-%m-%d}",
        "sessions": s["sessions"], "elb_sessions": s["elb_sessions"],
        "elb_share_pct": _whole(100.0 * s["elb_sessions"] / s["sessions"]),
    }
    name, (a, b) = next(iter(credit_cfg["exclude"].items()))
    out |= {"exclude_name": name, "exclude_first": a, "exclude_last": b}
    out |= {f"{n}_first_print": None if day is None else f"{day:%Y-%m-%d}" for n, day in r["first_print"].items()}

    st = r["staleness"]
    shows = [n for n in r["spreads"] if st[n]["shows"]]
    out |= {"stale_band": _fmt(st[p]["band"], 3), "stale_spreads": ", ".join(labels[n] for n in shows) or "none",
            "stale_primary": "shows" if st[p]["shows"] else "does not show"}
    for n in r["spreads"]:
        out |= {f"stale_{n}_ar1": _fmt(st[n]["ar1"], 2, True), f"stale_{n}_lag1": _fmt(st[n]["cross"][1], 2, True),
                f"stale_{n}_lag0": _fmt(st[n]["cross"][0], 2, True)}
        out |= {f"stale_{n}_ar1_ex_{w}": _fmt(x["ar1"], 2, True) for w, x in r["staleness_ex"][n].items()}

    for n in r["spreads"]:
        c = r["contemporaneous"][n]
        out |= {f"t1_{n}_weeks": c["market"]["n"], f"t1_{n}_first": f"{c['market']['first']:%Y-%m-%d}",
                f"t1_{n}_r2_market_pct": _fmt(100 * c["market"]["r2"], 1),
                f"t1_{n}_r2_components_pct": _fmt(100 * c["components"]["r2"], 1),
                f"t1_{n}_r2_control_pct": _fmt(100 * c["components_control"]["r2"], 1),
                f"t1_{n}_market_coef": _fmt(c["market"]["coef"]["market"], 3, True),
                f"t1_{n}_market_t": _t(c["market"]["t"]["market"]),
                f"t1_{n}_control_coef": _fmt(c["components_control"]["coef"]["control"], 3, True),
                f"t1_{n}_control_t": _t(c["components_control"]["t"]["control"])}
        for w, fit in r["contemporaneous_ex"][n].items():
            out[f"t1_{n}_r2_ex_{w}_pct"] = _fmt(100 * fit["r2"], 1)

    for n in r["spreads"]:
        cells = ([(h, reg, smp) for h in credit.HORIZONS for reg in ("z", "gap_bp")
                  for smp in r["predictive"][n][h][reg]] if n == p else [(h_long, "z", "all")])
        for h, reg, smp in cells:
            x = r["predictive"][n][h][reg][smp]
            key, places = f"t2_{n}_h{h}_{reg}_{smp}", 3 if reg == "gap_bp" else 2
            out |= {f"{key}_slope": _fmt(x["slope"], places, True), f"{key}_t": _t(x["t"]), f"{key}_n": x["n"],
                    f"{key}_nonoverlap_mean": _fmt(x["nonoverlap"]["mean"], places, True),
                    f"{key}_nonoverlap_t": _t(x["nonoverlap"]["mean_t"]), f"{key}_sign": _against_registered(x["slope"])}
    head = r["predictive"][p][h_long]["z"]["all"]
    others = [(h, reg, smp, x) for h, by_reg in r["predictive"][p].items() for reg, by_sample in by_reg.items()
              for smp, x in by_sample.items() if (h, reg, smp) != (h_long, "z", "all")]
    clearing = [_cell_words(h, reg, smp, cfg) for h, reg, smp, x in others if x["t"] <= -credit.Z95]
    out |= {"t2_cells": len(others), "t2_cells_negative": sum(x["slope"] < 0 for *_, x in others),
            "t2_cells_clearing_n": len(clearing), "t2_cells_clearing": "; ".join(clearing) or "none"}
    no = head["nonoverlap"]
    out |= {"t2_slope": _fmt(head["slope"], 2, True), "t2_t": _t(head["t"]), "t2_n": head["n"],
            "t2_r2_pct": _fmt(100 * head["r2"], 1), "t2_change_sd": _fmt(head["change_sd"], 1),
            "t2_nonoverlap_mean": _fmt(no["mean"], 2, True), "t2_nonoverlap_min": _fmt(no["min"], 2, True),
            "t2_nonoverlap_max": _fmt(no["max"], 2, True), "t2_nonoverlap_t": _t(no["mean_t"]),
            "t2_nonoverlap_n": _whole(no["n"]), "t2_nonoverlap_negative_pct": _whole(100 * no["share_negative"]),
            "t2_verdict": "passes" if r["supported"] else "does not pass", "t2_threshold": _fmt(-credit.Z95, 2)}
    years = r["headline_without_each_year"]
    slopes = [c["slope"] for c in years.values()]
    failing = sum(not c["supported"] for c in years.values())
    out |= {"t2_years": len(years), "t2_years_failing": failing, "t2_years_passing": len(years) - failing,
            "t2_years_negative": sum(s < 0 for s in slopes), "t2_years_slope_min": _fmt(min(slopes), 2, True),
            "t2_years_slope_max": _fmt(max(slopes), 2, True)}
    for i, y in enumerate(sorted(years, key=lambda y: years[y]["t"], reverse=True)[:2], start=1):
        out |= {f"t2_without_year{i}": y, f"t2_without_year{i}_slope": _fmt(years[y]["slope"], 2, True),
                f"t2_without_year{i}_t": _t(years[y]["t"])}
    checks = r["headline_checks"]
    out |= {"t2_nonoverlap_offsets": no["offsets"], "t2_nonoverlap_clearing": _whole(no["share_clearing"] * no["offsets"]),
            "t2_z_clip": _whole(credit.Z_CLIP), "t2_floor_sessions": head["n"] - checks["above_floor"]["n"],
            "t2_kernel_sentence": _kernel_sentence(r["supported"], checks, h_long),
            "t2_z_checks_words": _z_checks_words(head["slope"], checks)}
    for c, x in checks.items():
        out |= {f"t2_{c}_slope": _fmt(x["slope"], 2, True), f"t2_{c}_t": _t(x["t"]), f"t2_{c}_n": x["n"],
                f"t2_{c}_line": "clears" if x["t"] <= -credit.Z95 else "does not clear"}
    pl = r["headline_placebo"]
    out |= {"t2_placebo_shifts": pl["shifts"], "t2_placebo_reject_pct": _whole(100 * pl["reject"]),
            "t2_placebo_reject_negative_pct": _whole(100 * pl["reject_negative"]),
            "t2_placebo_critical": _fmt(pl["critical"], 2, True), "t2_placebo_p": _fmt(pl["p"], 2),
            "t2_placebo_p_pct": _whole(100 * pl["p"]),
            "t2_placebo_line": "clears" if head["t"] <= pl["critical"] else "does not clear"}

    c3 = r["conditional"][p]["level"]
    out["t3_sentence"] = c3["sentence"]
    out["t3_market_sentence"] = r["conditional"][p]["market"]["sentence"]
    out["t3_weeks"] = sum(x["weeks"] for x in c3["pooled"].values())
    for state, x in c3["pooled"].items():
        out |= {f"t3_{state}_slope": _fmt(x["slope"], 3, True), f"t3_{state}_t": _t(x["t"]),
                f"t3_{state}_weeks": x["weeks"]}
    d = c3["late_minus_early"]
    out |= {"t3_diff": _fmt(d["diff"], 3, True), "t3_diff_se": _fmt(d["se"], 3), "t3_diff_t": _t(d["t"]),
            "t3_mde": _fmt(d["mde"], 3)}
    for i, cyc in enumerate(c3["cycles"].to_dict("records"), start=1):
        for part in ("early", "late"):
            key, row = f"t3_cycle{i}_{part}", {f: cyc[f"{part}_{f}"] for f in credit.CYCLE_FIELDS}
            out |= {f"{key}_label": f"{_month(row['first'])} to {_month(row['last'])}",
                    f"{key}_slope": _fmt(row["slope"], 3, True), f"{key}_t": _t(row["t"]),
                    f"{key}_weeks": int(row["weeks"]), f"{key}_mde": _fmt(row["mde"], 3)}
    out |= {"t3_cycles": len(c3["cycles"]), "t3_episodes": len(c3["episodes"]),
            "t3_early_signs": _signs(c3["cycles"]["early_slope"]), "t3_late_signs": _signs(c3["cycles"]["late_slope"])}
    return out


# ---- the report ----------------------------------------------------------------------

def _ct(coef, t, places=3):
    return "" if coef is None or not np.isfinite(coef) else f"{coef:+.{places}f} ({t:+.1f})"


def _regressor_words(results, cfg):
    words = cfg["report"]["labels"]
    k = results["k"]
    out = {"market": f"{words['market']} rate, meeting {k}", "model": f"minus the rule's rate, meeting {k}",
           "slope": "orthogonal slope", "control": cfg["credit"]["labels"]["control"],
           "yield_lag": f"{cfg['credit']['labels']['staleness']}, week before"}
    out |= {name: f"differential, meeting {k} ({sleeve} signal)"
            for name, sleeve in zip(results["regressors"][3:], results["differentials"])}
    return out


def _check_words(h):
    """What each of `credit.after_the_run`'s checks is, for h sessions ahead."""
    return {"bartlett_2h": f"Newey-West, Bartlett, lag {2 * h}", "uniform": f"Hansen-Hodrick, equal weights, lag {h - 1}",
            "z_clipped": f"z clipped at ±{credit.Z_CLIP:.0f}", "above_floor": "without the sessions whose sd is at its floor"}


def _net_legs(*spreads):
    """The signed sum of `spreads` (each {role: {source, series}}) as {(source, series): net sign}, zeros dropped."""
    out = {}
    for legs in spreads:
        for role, ref in legs.items():
            key = (ref["source"], ref["series"])
            out[key] = out.get(key, 0) + config.SPREAD_SIGNS[role]
    return {key: sign for key, sign in out.items() if sign}


def identities(results, cfg):
    """A sentence for each pair of spreads where one is the other plus the control, as their legs and their fits both show.

    Then with the control in both fits every coefficient is the same but the
    control's, which is one higher. Said only where the legs add up and the
    fits agree (the same weeks and regressors, the coefficients as stated).
    """
    block, labels = cfg["credit"], cfg["credit"]["labels"]
    out = []
    for a, b in itertools.permutations(block["spreads"], 2):
        if _net_legs(block["spreads"][a]) != _net_legs(block["spreads"][b], block["control"]):
            continue
        fa, fb = (results["contemporaneous"][n]["components_control"]["coef"] for n in (a, b))
        n = [results["contemporaneous"][s]["components_control"]["n"] for s in (a, b)]
        if n[0] == n[1] and set(fa) == set(fb) and all(np.isclose(fa[c] - fb[c], float(c == "control"), atol=1e-9)
                                                       for c in fa):
            out.append(f"{labels[a]} is {labels[b]} plus the {labels['control']}, so with the {labels['control']} as "
                       "a control their other coefficients are identical and the control's differs by exactly one.")
    return out


def tables(results, cfg):
    """Every section's table, by title."""
    r, labels = results, cfg["credit"]["labels"]
    words = _regressor_words(r, cfg)
    p = r["primary"]

    stale = pd.DataFrame([{"spread": labels[n], "weeks": r["staleness"][n]["n"],
                           "AR(1)": f"{r['staleness'][n]['ar1']:+.2f}",
                           **{f"corr, yield {j}w before": f"{v:+.2f}" for j, v in r["staleness"][n]["cross"].items()},
                           "band": f"±{r['staleness'][n]['band']:.2f}",
                           "staleness": "shows" if r["staleness"][n]["shows"] else "no",
                           **{f"AR(1), {_sample_word('ex_' + w, cfg)}": f"{x['ar1']:+.2f}"
                              for w, x in r["staleness_ex"][n].items()}} for n in r["spreads"]])

    rows = []
    for n in r["spreads"]:
        for spec, fit in r["contemporaneous"][n].items():
            row = {"spread": labels[n], "spec": spec.replace("_", " + "), "weeks": fit["n"], "R² %": f"{100 * fit['r2']:.1f}"}
            row |= {words[c]: _ct(fit["coef"].get(c), fit["t"].get(c)) for c in [*r["regressors"], "control", "yield_lag"]}
            rows.append(row)
        for w, fit in r["contemporaneous_ex"][n].items():
            if fit["n"] == r["contemporaneous"][n]["components"]["n"]:
                continue
            row = {"spread": labels[n], "spec": f"components, {_sample_word('ex_' + w, cfg)}", "weeks": fit["n"],
                   "R² %": f"{100 * fit['r2']:.1f}"}
            row |= {words[c]: _ct(fit["coef"].get(c), fit["t"].get(c)) for c in [*r["regressors"], "control", "yield_lag"]}
            rows.append(row)
    test1 = pd.DataFrame(rows)
    test1 = test1.loc[:, (test1 != "").any()]

    rows = []
    for n in r["spreads"]:
        for h, by_reg in r["predictive"][n].items():
            for reg, by_sample in by_reg.items():
                for smp, x in by_sample.items():
                    if smp != "all" and x["n"] == by_sample["all"]["n"]:
                        continue
                    no = x["nonoverlap"]
                    places = 3 if reg == "gap_bp" else 2
                    rows.append({"spread": labels[n], "h": h, "on": "z" if reg == "z" else "gap, bp",
                                 "sample": _sample_word(smp, cfg), "sessions": x["n"],
                                 "slope": f"{x['slope']:+.{places}f}", "NW t": _t(x["t"]),
                                 "non-overlapping mean [min, max]": f"{no['mean']:+.{places}f} [{no['min']:+.{places}f}, "
                                                                   f"{no['max']:+.{places}f}]",
                                 "mean t": _t(no["mean_t"]), "negative %": f"{100 * no['share_negative']:.0f}"})
    test2 = pd.DataFrame(rows)
    years = pd.DataFrame([{"year left out": y, "sessions": x["n"], "slope": f"{x['slope']:+.2f}", "NW t": _t(x["t"]),
                           "non-overlapping mean": f"{x['nonoverlap']['mean']:+.2f}",
                           "mean t": _t(x["nonoverlap"]["mean_t"]),
                           "D1's rule": "passes" if x["supported"] else "does not pass"}
                          for y, x in r["headline_without_each_year"].items()])
    h_long = max(credit.HORIZONS)
    head = r["predictive"][p][h_long]["z"]["all"]
    rows = [("registered (D1): Newey-West, Bartlett, lag " + str(h_long), head)]
    rows += [(_check_words(h_long)[c], x) for c, x in r["headline_checks"].items()]
    checks = pd.DataFrame([{"check": what, "sessions": x["n"], "slope": f"{x['slope']:+.2f}", "t": _t(x["t"]),
                            f"clears {-credit.Z95:.2f}": "yes" if x["t"] <= -credit.Z95 else "no"} for what, x in rows])

    out = {"staleness": stale, "test1": test1, "test2": test2, "test2_years": years, "test2_checks": checks}
    for reg in ("level", "market"):
        c = r["conditional"][p][reg]
        pooled = pd.DataFrame([{"regime": STATE_WORDS[s], "weeks": x["weeks"], "slope": f"{x['slope']:+.3f}",
                                "NW se": f"{x['se']:.3f}", "t": _t(x["t"])} for s, x in c["pooled"].items()])
        d = c["late_minus_early"]
        pooled.loc[len(pooled)] = {"regime": "late minus early", "weeks": "", "slope": f"{d['diff']:+.3f}",
                                   "NW se": f"{d['se']:.3f}", "t": _t(d["t"])}
        ep = c["episodes"].assign(regime=lambda e: e["state"].map(STATE_WORDS),
                                  slope=lambda e: e["slope"].map("{:+.3f}".format), se=lambda e: e["se"].map("{:.3f}".format),
                                  t=lambda e: e["t"].map(_t), mde=lambda e: e["mde"].map("{:.3f}".format))
        out[f"test3_{reg}"] = pooled
        out[f"episodes_{reg}"] = ep[["regime", "first", "last", "weeks", "slope", "se", "t", "mde"]]
    others = pd.DataFrame([{"spread": labels[n], **{f"late minus early, {reg}": _ct(
        r["conditional"][n][reg]["late_minus_early"]["diff"], r["conditional"][n][reg]["late_minus_early"]["t"])
        for reg in ("level", "market")}, "verdict (level)": r["conditional"][n]["level"]["sentence"]}
        for n in r["spreads"]])
    out["test3_spreads"] = others
    return out


def markdown(ccy, results, cfg):
    """The report for one currency."""
    r, h, labels = results, headline(results, cfg), cfg["credit"]["labels"]
    t = tables(results, cfg)
    p, words = r["primary"], _regressor_words(r, cfg)
    h_long = max(credit.HORIZONS)
    stale_note = ("Where staleness shows, the yield's change the week before is a control in the component fits "
                  f"of test 1, and its change over the {credit.CONTROL_SESSIONS} sessions to t+1 in test 2.")
    unstaled = r["headline_unstaled"]
    sleeves = r["differentials"]
    cross = (f", the cross-country differential at meeting {r['k']} (the signal of the {' and '.join(sleeves)} "
             f"sleeve{'s' if len(sleeves) > 1 else ''})" if sleeves else "")
    lines = [
        f"# The credit bridge: {ccy}",
        "",
        f"Generated by `scripts/build_credit.py`. Do not edit by hand. Methods: `src/policypath/credit.py`; "
        "decisions: notes/DECISIONS.md, \"Credit bridge\". The expected sign of test 2 was registered "
        "before the first run (D1). A sample that leaves out a window the spread never reaches is not shown.",
        "",
        f"- Spreads: {', '.join(labels[n] for n in r['spreads'])}; the primary is {labels[p]}. "
        "**Spread changes stand in for excess returns** (a spread 10bp wider on a bond of duration D is about "
        "D x 10bp lost against the benchmark, carry left out).",
        f"- Weekly: the last print on or before each {cfg['credit']['week_ends']}, {h['first_week']} to "
        f"{h['last_week']} ({h['weeks']} weeks). Sessions for test 2: {h['first_session']} to {h['last_session']}, "
        f"{h['sessions']} with a z, of which {h['elb_sessions']} ({h['elb_share_pct']}%) are in the ELB state. "
        "**ELB sessions are included** in the headline, and left out as a row.",
        f"- Regressors, bp: the {words['market']} (held from the start of the week), {words['model']}; the two sum to "
        f"the change in that meeting's gap. Then the orthogonal slope{cross}, and the {labels['control']} as a control.",
        f"- Stale marks (a session carrying the print before it): "
        + ", ".join(f"{labels.get(n, labels['staleness'] if n == 'yield' else n)} {c}" for n, c in r["stale_sessions"].items()),
        "",
        "## Staleness, before test 2",
        "",
        f"The weekly change's first-order autocorrelation, and its correlation with the {labels['staleness']}'s change "
        f"0, 1 and 2 weeks before. Staleness shows when either the AR(1) or the one-week correlation is outside the "
        f"band (±1.96/√n, D2). {stale_note} The AR(1) without each excluded window is a description only: the "
        "rule reads the full sample.",
        "",
        table(t["staleness"]),
        "",
        "## Test 1: contemporaneous",
        "",
        f"Weekly Δspread on the regressors, coefficient (NW t, lag {credit.WEEKLY_LAGS}), bp per bp. One sample for "
        f"the three specs of each spread. The R² of the market's part alone is how much of credit's weekly variation "
        f"is a repricing of the policy path: **{h[f't1_{p}_r2_market_pct']}%** for {labels[p]}.",
        "",
        table(t["test1"]),
        "",
        *(line for sentence in identities(r, cfg) for line in (sentence, "")),
        "## Test 2: predictive",
        "",
        f"s(t+1+h) - s(t+1) on the gap at meeting {r['k']} on session t, in z and in bp; NW lag h. The non-overlapping "
        "check fits every h-th session for each of the h start offsets (White se): the mean slope, its range, "
        "the mean t, and the share of offsets with a negative slope.",
        "",
        f"**Headline (D1): {labels[p]}, z, h = {h_long}, every session with a z: slope {h['t2_slope']}bp per unit "
        f"of z, NW t {h['t2_t']}, non-overlapping mean {h['t2_nonoverlap_mean']} "
        f"(mean t {h['t2_nonoverlap_t']}). It {h['t2_verdict']} the pre-registered rule** (slope negative, "
        f"NW t ≤ {h['t2_threshold']}, non-overlapping mean negative). The last condition asks only for a sign, "
        "and the mean slope over the offsets is close to the pooled slope by construction, so once the NW t passes it "
        f"adds little: {h['t2_nonoverlap_clearing']} of the {h['t2_nonoverlap_offsets']} offsets clear "
        f"{h['t2_threshold']} on their own. {h['t2_kernel_sentence']} With no link at all (D19, below) the registered t "
        f"reaches ±1.96 in {h['t2_placebo_reject_pct']}% of placebo samples, and the headline's one-sided placebo p is "
        f"{h['t2_placebo_p']}.",
        *([] if unstaled is None else ["", f"Without the staleness control: slope {unstaled['slope']:+.2f}, "
                                           f"NW t {_t(unstaled['t'])}."]),
        "",
        table(t["test2"]),
        "",
        f"**The headline without each calendar year** (every change that touches the year left out, as for each "
        f"excluded window; a check added after the run, D17, not a registered cell). The slope is negative without "
        f"{h['t2_years_negative']} of the {h['t2_years']} years ({h['t2_years_slope_min']} to "
        f"{h['t2_years_slope_max']}), and the headline fails D1's rule without {h['t2_years_failing']} of them. "
        f"It leans most on {h['t2_without_year1']} (without it {h['t2_without_year1_slope']}, "
        f"t {h['t2_without_year1_t']}) and {h['t2_without_year2']} ({h['t2_without_year2_slope']}, "
        f"t {h['t2_without_year2_t']}). Without {h['t2_years_passing']} of the {h['t2_years']} years it still passes.",
        "",
        table(t["test2_years"]),
        "",
        f"**The headline under four more checks added after the run** (D18; checks, not registered cells). The "
        f"overlapping changes are an MA({h_long - 1}): Bartlett at lag {h_long} gives their autocovariances about two "
        f"thirds of their weight, and Hansen-Hodrick's equal weights to lag {h_long - 1} give them all of it. "
        f"Clipping z at ±{h['t2_z_clip']} and leaving out the {h['t2_floor_sessions']} sessions whose z divides by "
        "the floored sd ask whether the headline rests on a few inflated z's.",
        "",
        table(t["test2_checks"]),
        "",
        f"**How often the registered test rejects with no link** (D19; a check added after the run, not a registered "
        f"cell). {labels[p]}'s daily changes from the first z on are rotated in time against z, by every "
        f"{credit.PLACEBO_STEP}th shift of at least {credit.PLACEBO_MIN} sessions ({h['t2_placebo_shifts']} shifts). "
        "Each series keeps its own volatility and persistence, and no shift lines one up with the other. The "
        f"registered NW t reaches ±1.96 in {h['t2_placebo_reject_pct']}% of the shifts, where 5% is nominal, and "
        f"{h['t2_threshold']} in {h['t2_placebo_reject_negative_pct']}%, where 2.5% is. At the placebo's own 2.5% the "
        f"line is {h['t2_placebo_critical']}, which the headline's {h['t2_t']} {h['t2_placebo_line']}. "
        f"{h['t2_placebo_p_pct']}% of the shifted t's are at or below the headline's: its one-sided placebo p is "
        f"{h['t2_placebo_p']}.",
        "",
        "## Test 3: conditional",
        "",
        f"Weekly Δ{labels[p]} on the weekly change in the level (the gap at the held meeting {r['k']}), by the regime "
        f"at the start of the week (`regimes.py`, real time; ELB weeks out). One regression with a slope and "
        f"intercept per regime (regimes with {credit.MIN_WEEKS}+ weeks), NW lag {credit.WEEKLY_LAGS}. The hypothesis "
        "(D3): early hiking negative, late hiking positive. The minimum detectable difference (mde) is the size "
        "80% power would find at 5%.",
        "",
        f"**{h['t3_sentence']}** Late minus early, pooled: {h['t3_diff']} (se {h['t3_diff_se']}, "
        f"t {h['t3_diff_t']}); the minimum detectable difference is {h['t3_mde']}.",
        "",
        table(t["test3_level"]),
        "",
        f"Per episode (a run of one regime with enough weeks), NW lag {credit.WEEKLY_LAGS}:",
        "",
        table(t["episodes_level"]),
        "",
        f"On the market's part alone ({words['market']}): {h['t3_market_sentence']}",
        "",
        table(t["test3_market"]),
        "",
        table(t["episodes_market"]),
        "",
        "Every spread:",
        "",
        table(t["test3_spreads"]),
        "",
        f"![The credit bridge](figures/credit_{ccy}_light.png)",
        "",
    ]
    return "\n".join(lines)


# ---- the chart -----------------------------------------------------------------------

def _colours(t):
    """Regime washes from the theme's diverging ramp: cutting on the blue arm, hiking on the red (late further out)."""
    d = t["diverging"]
    return {"early_hiking": d[7], "late_hiking": d[9], "cutting": d[2], "elb": t["muted"]}


def figure(frames, results, cfg, path, theme="light"):
    """Three panels: the spread through time on its regimes, test 2's non-overlapping quarters, test 3's episodes."""
    t = _style(theme)
    colours = _colours(t)
    r, labels, p = results, cfg["credit"]["labels"], results["primary"]
    h = max(credit.HORIZONS)
    fig = plt.figure(figsize=(11, 8.4))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], hspace=0.5, wspace=0.12)
    top, left, right = fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])

    daily = frames["daily"]
    level = daily[p].dropna()
    state = frames["state"]
    for run in regimes.episodes(state).itertuples():
        if run.state in colours:
            top.axvspan(run.first, run.last, color=colours[run.state], alpha=0.22 if theme == "light" else 0.35, lw=0)
    top.plot(level.index, level.to_numpy(), color=t["ink"], lw=1.2)
    top.set_ylabel("bp")
    top.set_xlim(level.index[0], level.index[-1])
    top.xaxis.set_major_locator(mdates.YearLocator(2))
    top.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    top.legend(handles=[Patch(color=colours[s], alpha=0.5, label=STATE_WORDS[s]) for s in colours],
               loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize=9, labelcolor=t["secondary"])
    _title(top, f"{labels[p]} and where the policy cycle was",
           "The spread by session; the bands are the regime in real time. Holds are unshaded.", t)

    head = r["predictive"][p][h]["z"]["all"]
    y = credit.ahead(daily[p], h)
    keep = daily["z"].notna() & y.notna()
    window = next(iter(cfg["credit"]["exclude"].values()))
    spans = credit.spans(daily.index, h, window)
    rows = np.flatnonzero(keep.to_numpy())
    rows = rows[::h]
    x, yy, hit = daily["z"].to_numpy()[rows], y.to_numpy()[rows], spans.to_numpy()[rows]
    left.axhline(0, color=t["axis"], lw=1)
    left.axvline(0, color=t["axis"], lw=1)
    left.scatter(x[~hit], yy[~hit], s=26, color=t["series"][0], edgecolor=t["surface"], lw=1.2, zorder=3)
    left.scatter(x[hit], yy[hit], s=26, color=t["series"][1], edgecolor=t["surface"], lw=1.2, zorder=3)
    fit = credit.ols(y.where(keep), np.column_stack([np.ones(len(y)), daily["z"].to_numpy()]), h)
    grid_z = np.linspace(np.nanmin(x), np.nanmax(x), 2)
    left.plot(grid_z, fit["beta"][0] + fit["beta"][1] * grid_z, color=t["ink"], lw=2)
    left.set_xlabel(f"z of the gap at meeting {r['k']} on the day")
    left.set_ylabel(f"bp, change over the next {h} sessions")
    left.grid(True, axis="both")
    left.legend(handles=[Line2D([], [], ls="", marker="o", color=t["series"][0], label="one quarter"),
                         Line2D([], [], ls="", marker="o", color=t["series"][1], label=f"spans {window[0][:7]} to {window[1][:7]}"),
                         Line2D([], [], color=t["ink"], lw=2, label="fit on every session")],
                loc="lower left", fontsize=9, labelcolor=t["secondary"])
    _title(left, "Test 2: does the gap lead the spread?",
           f"Slope {head['slope']:+.2f}bp per unit z, NW t {_t(head['t'])}; "
           f"mean non-overlapping t {_t(head['nonoverlap']['mean_t'])}", t)

    ep = r["conditional"][p]["level"]["episodes"]
    ep = ep[ep["state"].isin(regimes.HIKING)].reset_index(drop=True)
    ypos = np.arange(len(ep))[::-1]
    right.axvline(0, color=t["axis"], lw=1)
    for yv, row in zip(ypos, ep.itertuples()):
        lo, hi = row.slope - credit.Z95 * row.se, row.slope + credit.Z95 * row.se
        right.plot([lo, hi], [yv, yv], color=colours[row.state], lw=2)
        right.plot([row.slope], [yv], "o", ms=8, color=colours[row.state], mec=t["surface"], mew=2, zorder=3)
        right.annotate(f"{STATE_WORDS[row.state].capitalize()}, {_month(row.first)} to {_month(row.last)}",
                       xy=(0, yv), xycoords=("axes fraction", "data"), xytext=(0, 9), textcoords="offset points",
                       fontsize=9, color=t["secondary"])
    right.set_yticks([])
    right.set_ylim(-0.6, len(ep) - 0.2)
    right.set_xlabel(f"bp of spread per bp of the gap, weekly (95% interval, NW lag {credit.WEEKLY_LAGS})")
    right.grid(True, axis="x")
    right.grid(False, axis="y")
    _title(right, "Test 3: does the sign flip late in a cycle?", r["conditional"][p]["level"]["sentence"].split(":")[0], t)

    for ax in (top, left, right):
        ax.tick_params(axis="both", length=0)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_figures(frames, results, cfg, out_dir, ccy):
    """The chart, light and dark: ``credit_<ccy>_<theme>.png`` under `out_dir`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for theme in THEMES:
        path = out_dir / f"credit_{ccy}_{theme}.png"
        figure(frames, results, cfg, path, theme)
        written.append(path)
    return written
