"""The weekly brief: market path against model path per currency, the gaps, the trades, what changed.

One page, generated from `data/panel/` with nothing typed by hand: every
number comes from the panels, every sentence is a template filled from them,
and the limitations footer is built from the ``tags`` each currency's config
attaches to its series and methods. The footer reads only the blocks the
brief itself reads (``footer_blocks`` in ``config/brief.yml``): the credit
bridge's, HLW's and the robustness tags belong to the full limitations, and
the page has no room for them. The trades table is each sleeve's breakeven
from ``data/panel/book_expression.parquet`` (`report.expression`), and what a
round trip of all its legs costs at the configured costs (`strategy.costs`:
ZQ from its tick, the rest assumed). Dated by the brief date in the filename.

Point in time by construction: each currency is read on its last session on
or before the brief date, and every panel row was built from what was public
at its own session, so a brief for a past date, run today, is the brief that
could have been sent then. "What changed" is the same read a week earlier.
"""

from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from policypath import config, panel
from policypath.report import charts
from policypath.report import expression
from policypath.report.figures import _style
from policypath.report.model import snapshot
from policypath.report.onepager import ORDINAL, PAGE, _Column, _day
from policypath.signal.cross import differential
from policypath.sources import cache
from policypath.strategy import costs

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


def tags(cfg, blocks=None):
    """Every ``tags`` block in a currency's config as (name, text) pairs; under the top-level `blocks` only if given."""
    found = []

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "tags" and isinstance(value, dict):
                    found.extend((name, text) for name, t in value.items() for text in t.values())
                else:
                    walk(value)
    walk(cfg if blocks is None else {k: v for k, v in cfg.items() if k in blocks})
    return found


def checks_only(cfg, blocks):
    """The series `blocks` list under ``validation``: cross-checks, never inputs, so the brief never shows them."""
    return {s for b in blocks if isinstance(cfg.get(b), dict) and isinstance(cfg[b].get("validation"), list)
            for s in cfg[b]["validation"]}


def footer(currencies, blocks=None):
    """The limitations footer: each tag once, with the currencies it applies to.

    With `blocks`, only those blocks' tags (`tags`), less the tags of the
    series they list as validation only (`checks_only`).
    """
    by_text = {}
    for ccy in currencies:
        cfg = config.currency(ccy)
        skip = checks_only(cfg, blocks) if blocks is not None else set()
        for name, text in tags(cfg, blocks):
            if name in skip:
                continue
            by_text.setdefault(text, []).append(ccy)
    return [f"{'/'.join(c)}: {text}" for text, c in by_text.items()]


def _row(snap, k):
    return snap["meetings"].set_index("k").loc[k]


def bp(x):
    """Signed basis points, with no minus sign on a zero; n/a where there is no number."""
    if pd.isna(x):
        return "n/a"
    return f"{0.0 if round(x) == 0 else x:+.0f}bp"


def changed(ccy, now, before, k, cfg):
    """One sentence: how the gap at the k-th meeting moved over the week, split into market and rule, and why.

    If the currency has no session after last week's read (its panel ends
    there), it says so rather than report a move of zero.
    """
    w = cfg["report"]["labels"]
    if now["session"] == before["session"]:
        return f"{ccy}: no session since {now['session']:%d %b}, so nothing to compare yet."
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
                        "unemployment " + w["unemployment_period"].format(month=now["macro"]["gap_month"])
                        + f" came in at {now['macro']['u']:.1f}%"))
    if rn["rstar"] != rb["rstar"]:
        label = cfg["rule"]["rstar"].get("label", "estimate")
        reasons.append((abs(rn["rstar"] - rb["rstar"]) * 100, f"a new {label} moved r* to {rn['rstar']:.1f}%"))
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


def _table(fig, col, header, rows, widths, size=7.5, left=(0,)):
    """A plain table at the column's cursor: bold header, the `left` columns left-aligned, the rest right-aligned.

    `widths` are inches; a left-aligned column after the first starts 0.1in in, clear of the number before it.
    """
    lines = [header, *rows]
    step = size * 1.45 / 72 / PAGE[1]
    for i, line in enumerate(lines):
        x = col.left                                    # widths are inches; the figure is placed in fractions
        for j, (cell, wd) in enumerate(zip(line, widths)):
            at = (0.1 if j else 0.0) / PAGE[0] if j in left else wd / PAGE[0]
            fig.text(x + at, col.y - i * step, cell, fontsize=size, va="top",
                     ha="left" if j in left else "right", weight="bold" if i == 0 else "normal", color=col.t["ink"])
            x += wd / PAGE[0]
    col.y -= len(lines) * step + 0.08 / PAGE[1]
    col._check()


def trades(frame, date, enter, round_trips=None):
    """The trades table's rows: each sleeve on its last session on or before `date`, from the book panel `frame`.

    With `round_trips` ({sleeve: bp}), each row carries its round trip before the verdict.
    """
    rows = []
    for name, r in expression.latest(frame, date).iterrows():
        cost = [] if round_trips is None else [f"{round_trips[name]:.2f}bp"]
        rows.append([name, f"{r['session']:%d %b}", f"{r['z']:+.1f}" + ("" if abs(r["z"]) >= enter else "*"),
                     expression.SIDE[float(r["side"])], r["instrument"], bp(r["edge_bp"]), bp(r["cr_h_bp"]),
                     bp(r["expected_bp"]), *cost, expression.verdict(r)])
    return rows


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
    height = 2.2
    top = col.y - 0.30 / PAGE[1]
    width = (col.width - 0.28) / len(reads)
    for i, (ccy, snap) in enumerate(reads.items()):
        cfg = config.currency(ccy)
        ax = fig.add_axes([col.left + 0.05 + i * (width + 0.28), top - height / PAGE[1], width - 0.05,
                           height / PAGE[1]])
        overnight = cache.log(cfg["overnight"]["source"], cfg["overnight"]["series"], ccy)
        # Short direct labels: two charts share the page's width.
        charts.paths_now(ax, snap, overnight, t, {**cfg["report"]["labels"], "rule": "Rule"}, lead_days=90)
        realized = ax.lines[0]
        ax.annotate(cfg["report"]["labels"]["rate"], xy=(realized.get_xdata()[0], realized.get_ydata()[0]),
                    xytext=(0, 4), textcoords="offset points", va="bottom", color=t["secondary"])
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
    col.y = top - (height + 0.26) / PAGE[1]

    # The table: gaps and z per currency, then the differential.
    header = ["", "as of", *[f"gap {h}" for h in horizons], *[f"z {h}" for h in horizons]]
    rows = []
    for ccy, snap in reads.items():
        m = snap["meetings"].set_index("k")
        rows.append([ccy, f"{snap['session']:%d %b}", *[bp(m.loc[h, "gap_bp"]) for h in horizons],
                     *[f"{m.loc[h, 'z']:+.1f}" for h in horizons]])
    rows.append([f"{first} − {second}", f"{diff_day:%d %b}", *["" for _ in horizons],
                 *[f"{diff_now.loc[h, 'z']:+.1f}" for h in horizons]])
    col.text("The gaps, by meetings ahead (the charts label gap 4 and gap 8)", size=10, weight="bold", gap=0.04)
    _table(fig, col, header, rows, [1.1, 0.7, *[0.75] * len(horizons), *[0.6] * len(horizons)])
    d4 = diff_now.loc[k]
    col.text(f"{first} − {second} is gap minus gap on the sessions both have, z {d4['z']:+.1f} at the {ORDINAL[k]} "
             "meeting. Its basis points are not shown: the two gaps take r* by different methods, so their difference "
             "is mostly that choice, and only its z is comparable.", size=8.5, gap=0.08)

    book = config.strategy()
    h, enter = book["carry"]["horizon_days"], book["positions"]["enter"]
    proxies = list(dict.fromkeys(s["kind"] for s in book["sleeves"] if s["kind"] != "outright"))
    trade_rows = trades(expression.load(root), date, enter, {s["name"]: costs.round_trip(s) for s in book["sleeves"]})
    col.text("The trades, before costs: what holding each signal earns or costs (CR), and a round trip (RT)",
             size=10, weight="bold", gap=0.04)
    _table(fig, col, ["", "as of", "z", "side", "instrument", "edge", f"CR {h}d", f"E {h}d", "RT", "carry and roll"],
           trade_rows, [0.85, 0.45, 0.4, 0.55, 1.95, 0.5, 0.5, 0.5, 0.45, 1.15], size=7.5, left=(0, 3, 4, 9))
    col.text(f"Edge: de-meaned gap ({', '.join(proxies)}: a proxy for the legs' move). CR: carry and roll, curve "
             f"frozen. E: past closure of the edge, CR on the rest. * |z| < {enter:g}.", size=7.5,
             color=t["secondary"], gap=0.08)

    col.text("What changed since last week", size=10, weight="bold", gap=0.03)
    for ccy in reads:
        col.text(changed(ccy, reads[ccy], last_week[ccy], k, config.currency(ccy)), size=8.5, gap=0.03)
    col.y -= 0.06 / PAGE[1]
    col.text("What would make this wrong", size=10, weight="bold", gap=0.03)
    col.text(wrong(reads, k), size=8.5, gap=0.1)
    col.text("Limitations (from the data-quality and lag tags in config)", size=9, weight="bold", gap=0.02)
    col.text("  ·  ".join(footer(brief["currencies"], brief["footer_blocks"])), size=6.2, color=t["secondary"],
             gap=0.02)

    out_dir = Path(out_dir)
    path = out_dir / f"brief_{date:%Y-%m-%d}.pdf"
    fig.savefig(path)
    if preview is not None:
        fig.savefig(preview, dpi=110)
    plt.close(fig)
    numbers = {"reads": reads, "differential": diff_now, "differential_session": diff_day,
               "changed": [changed(c, reads[c], last_week[c], k, config.currency(c)) for c in reads],
               "wrong": wrong(reads, k), "trades": trade_rows}
    return path, numbers

