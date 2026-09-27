"""The weekly brief: market path against model path per currency, the gaps, the cross-country differential, what changed.

One page, generated from `data/panel/` with nothing typed by hand: every
number comes from the panels, every sentence is a template filled from them,
and the limitations footer is built from the ``tags`` each currency's config
attaches to its series and methods. Dated by the brief date in the filename.

Point in time by construction: each currency is read on its last session on
or before the brief date, and every panel row was built from what was public
at its own session, so a brief for a past date, run today, is the brief that
could have been sent then. "What changed" is the same read a week earlier.
"""

from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from policypath import config, panel
from policypath.report import charts
from policypath.report.figures import _style
from policypath.report.model import snapshot
from policypath.report.onepager import MARGIN, ORDINAL, PAGE, _Column, _day
from policypath.signal.cross import differential
from policypath.sources import cache

BRIEF = config.CONFIG_DIR / "brief.yml"


def settings(path=BRIEF):
    with open(path) as f:
        return yaml.safe_load(f)


def load(ccy, root=panel.PANEL_DIR):
    """The tables a brief reads for one currency."""
    return {name: panel.load(ccy, name, root) for name in ["sessions", "paths", "signal", "model", "macro"]}


def read(ccy, tables, date):
    """One currency on its last session on or before `date`: the chart's snapshot, plus the macro row it used."""
    sessions = tables["model"]["session"]
    day = sessions[sessions <= pd.Timestamp(date)].max()
    if pd.isna(day):
        raise ValueError(f"no {ccy} model session on or before {pd.Timestamp(date).date()}")
    today = snapshot(day, tables["paths"], tables["model"], tables["sessions"], tables["signal"])
    macro = tables["macro"].set_index("as_of").loc[today["rule"]["macro_as_of"]]
    return {**today, "macro": macro}


def tags(cfg):
    """Every ``tags`` block anywhere in a currency's config, as (name, text) pairs."""
    found = []

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "tags" and isinstance(value, dict):
                    found.extend((name, text) for name, t in value.items() for text in t.values())
                else:
                    walk(value)
    walk(cfg)
    return found


def footer(currencies):
    """The limitations footer: each tag once, with the currencies it applies to."""
    by_text = {}
    for ccy in currencies:
        for _, text in tags(config.currency(ccy)):
            by_text.setdefault(text, []).append(ccy)
    return [f"{'/'.join(c)}: {text}" for text, c in by_text.items()]


def _row(snap, k):
    return snap["meetings"].set_index("k").loc[k]


def bp(x):
    """Signed basis points, with no minus sign on a zero."""
    return f"{0.0 if round(x) == 0 else x:+.0f}bp"


def changed(ccy, now, before, k, cfg):
    """One sentence: how the gap at the k-th meeting moved over the week, split into market and rule, and why."""
    w = cfg["report"]["labels"]
    a, b = _row(now, k), _row(before, k)
    d_gap, d_mkt, d_mod = a["gap_bp"] - b["gap_bp"], (a["market"] - b["market"]) * 100, (a["model"] - b["model"]) * 100
    text = (f"{ccy} ({before['session']:%d %b} to {now['session']:%d %b}): the gap at the {ORDINAL[k]} meeting "
            f"moved {bp(d_gap)} to {bp(a['gap_bp'])} (z {b['z']:+.1f} to {a['z']:+.1f}): the market {bp(d_mkt)}, "
            f"the rule {bp(d_mod)}")
    rn, rb = now["rule"], before["rule"]
    c = cfg["rule"]["coefficients"]
    reasons = []
    if rn["r0"] != rb["r0"]:
        reasons.append((abs(rn["r0"] - rb["r0"]) * 100, f"{w['policy']} moved to {rn['r0']:.2f}%"))
    if rn["inflation"] != rb["inflation"]:
        reasons.append((abs(rn["inflation"] - rb["inflation"]) * (1 + c["inflation_gap"]) * 100,
                        f"{w['inflation']} for {now['macro']['infl_month']:%B} printed {rn['inflation']:.1f}% "
                        f"(from {rb['inflation']:.1f}%)"))
    if rn["u_gap"] != rb["u_gap"]:
        reasons.append((abs(rn["u_gap"] - rb["u_gap"]) * c["unemployment_gap"] * 100,
                        f"unemployment for the three months to {now['macro']['gap_month']:%B} came in at "
                        f"{now['macro']['u']:.1f}%"))
    if rn["rstar"] != rb["rstar"]:
        reasons.append((abs(rn["rstar"] - rb["rstar"]) * 100, f"a new SEP moved r* to {rn['rstar']:.1f}%"))
    if abs(d_mod) >= 1 and reasons:
        text += ", the rule because " + max(reasons)[1]
    elif abs(d_mod) >= 1:
        text += ", the rule only by its inertial step toward the same goal"
    if a["effective_date"] != b["effective_date"]:
        text += f"; the horizon rolled on to the {a['announcement_date']:%B %Y} meeting"
    return text + "."


def wrong(reads, k):
    """One sentence on what would make the widest reading wrong."""
    ccy, snap = max(reads.items(), key=lambda kv: abs(_row(kv[1], k)["z"]))
    cfg = config.currency(ccy)
    w, r, row = cfg["report"]["labels"], snap["rule"], _row(snap, k)
    side = "more" if row["gap_bp"] > 0 else "less"
    return (f"The widest reading is {ccy} at the {ORDINAL[k]} meeting (z {row['z']:+.1f}, {bp(row['gap_bp'])}: "
            f"the market prices {side} tightening than the rule). It is wrong if the market is right that "
            f"{w['inflation']} ({r['inflation']:.1f}%) and unemployment will not stay where they are: the rule holds "
            f"them flat, so a market pricing their change looks mispriced to it when it is not.")


def _table(fig, col, header, rows, widths, size=8.5):
    """A plain table at the column's cursor: bold header, right-aligned numbers after the first column."""
    lines = [header, *rows]
    step = size * 1.55 / 72 / PAGE[1]
    for i, line in enumerate(lines):
        x = col.left                                    # widths are inches; the figure is placed in fractions
        for j, (cell, wd) in enumerate(zip(line, widths)):
            fig.text(x + (wd / PAGE[0] if j else 0), col.y - i * step, cell, fontsize=size, va="top",
                     ha="right" if j else "left", weight="bold" if i == 0 else "normal", color=col.t["ink"])
            x += wd / PAGE[0]
    col.y -= len(lines) * step + 0.08 / PAGE[1]
    col._check()


def write(date, out_dir, root=panel.PANEL_DIR, brief=None, preview=None, generated=None):
    """reports/brief_<date>.pdf. Returns (path, the numbers it printed)."""
    brief = brief or settings()
    date = pd.Timestamp(date)
    horizons, k = brief["horizons"], brief["traded"]
    tables = {ccy: load(ccy, root) for ccy in brief["currencies"]}
    reads = {ccy: read(ccy, t, date) for ccy, t in tables.items()}
    week = date - pd.Timedelta(days=brief["lookback_days"])
    last_week = {ccy: read(ccy, t, week) for ccy, t in tables.items()}
    first, second = brief["differential"]
    signal_spec = config.currency(first)["signal"]
    diff = differential(tables[first]["signal"], tables[second]["signal"], signal_spec)
    diff_now = diff[diff["session"] <= date]
    diff_day = diff_now["session"].max()
    diff_now = diff_now[diff_now["session"] == diff_day].set_index("k")

    t = _style("light")
    generated = pd.Timestamp(generated) if generated is not None else pd.Timestamp.today().normalize()
    fig = plt.figure(figsize=PAGE)
    col = _Column(fig, t)
    col.text("Policy paths: what markets price against the central banks' own rules", size=15, weight="bold",
             gap=0.03)
    col.text(f"Weekly brief, {_day(date)}  ·  " + "  ·  ".join(
        f"{c} as of {reads[c]['session']:%d %b}" for c in reads) + f"  ·  generated {generated:%Y-%m-%d}",
             size=8.5, color=t["secondary"], gap=0.08)
    col.text("For each currency, the market-implied path of the policy rate over the next eight meetings, "
             "against the path a balanced-approach rule would take on the macro data public that day, held flat. "
             "The gap is market minus rule; its z is against the gap's own trailing two years.", size=8.5, gap=0.1)

    # One chart per currency, side by side.
    height = 2.7
    top = col.y - 0.42 / PAGE[1]
    width = (col.width - 0.28) / len(reads)
    for i, (ccy, snap) in enumerate(reads.items()):
        cfg = config.currency(ccy)
        ax = fig.add_axes([col.left + 0.05 + i * (width + 0.28), top - height / PAGE[1], width - 0.05,
                           height / PAGE[1]])
        overnight = cache.log(cfg["overnight"]["source"], cfg["overnight"]["series"], ccy)
        # Short direct labels: two charts share the page's width.
        charts.paths_now(ax, snap, overnight, t, lead_days=90, labels={**cfg["report"]["labels"], "rule": "Rule"})
        ax.set_title(f"{ccy}: {cfg['report']['labels']['policy']}, {snap['session']:%d %b %Y}", loc="left",
                     fontsize=10, fontweight="bold", color=t["ink"], pad=6)
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
        for text in ax.texts:
            text.set_fontsize(7.5)
        ax.tick_params(labelsize=7)
        ax.yaxis.label.set_fontsize(7.5)
        ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(bymonth=[1, 7]))
    col.y = top - (height + 0.42) / PAGE[1]
    col.text("Black: the overnight rate realized. Blue: the market path. Orange: the rule's path. "
             "Labels: the gap at the fourth and eighth meetings.", size=7.5, color=t["secondary"], gap=0.1)

    # The table: gaps and z per currency, then the differential.
    header = ["", "as of", *[f"gap {h}" for h in horizons], *[f"z {h}" for h in horizons]]
    rows = []
    for ccy, snap in reads.items():
        m = snap["meetings"].set_index("k")
        rows.append([ccy, f"{snap['session']:%d %b}", *[bp(m.loc[h, "gap_bp"]) for h in horizons],
                     *[f"{m.loc[h, 'z']:+.1f}" for h in horizons]])
    rows.append([f"{first} − {second}", f"{diff_day:%d %b}", *[bp(diff_now.loc[h, "diff_bp"]) for h in horizons],
                 *[f"{diff_now.loc[h, 'z']:+.1f}" for h in horizons]])
    col.text("The gaps, by meetings ahead", size=10, weight="bold", gap=0.04)
    _table(fig, col, header, rows, [1.1, 0.7, *[0.75] * len(horizons), *[0.6] * len(horizons)])
    d4 = diff_now.loc[k]
    col.text(f"The {first} − {second} row is the first cross-country signal: {first}'s gap minus {second}'s on the "
             f"last session both have. At the {ORDINAL[k]} meeting it is {bp(d4['diff_bp'])}, "
             f"a z of {d4['z']:+.1f} against its own two years. Most of its level is the two r* choices; the z is "
             "what moves.", size=8.5, gap=0.1)

    col.text("What changed since last week", size=10, weight="bold", gap=0.03)
    for ccy in reads:
        col.text(changed(ccy, reads[ccy], last_week[ccy], k, config.currency(ccy)), size=8.5, gap=0.03)
    col.y -= 0.06 / PAGE[1]
    col.text("What would make this wrong", size=10, weight="bold", gap=0.03)
    col.text(wrong(reads, k), size=8.5, gap=0.1)
    col.text("Limitations (from the data-quality and lag tags in config)", size=9, weight="bold", gap=0.02)
    col.text("  ·  ".join(footer(brief["currencies"])), size=6.8, color=t["secondary"], gap=0.02)

    out_dir = Path(out_dir)
    path = out_dir / f"brief_{date:%Y-%m-%d}.pdf"
    fig.savefig(path)
    if preview is not None:
        fig.savefig(preview, dpi=110)
    plt.close(fig)
    numbers = {"reads": reads, "differential": diff_now, "differential_session": diff_day,
               "changed": [changed(c, reads[c], last_week[c], k, config.currency(c)) for c in reads],
               "wrong": wrong(reads, k)}
    return path, numbers

