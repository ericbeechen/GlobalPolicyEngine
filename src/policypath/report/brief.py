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

import re
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
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
NUMBER = {4: "four", 6: "six", 8: "eight", 10: "ten", 12: "twelve"}
SHORT = {1: "1st", 2: "2nd", 3: "3rd", **{h: f"{h}th" for h in range(4, 13)}}


def settings(path=BRIEF):
    with open(path, encoding="utf-8") as f:
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


def limitations(lines):
    """The footer's lines ("USD: text", "USD/GBP: text") as (lead, body) pairs, one per set of currencies.

    The tags every currency shares come first; each tag is a sentence.
    """
    groups = {}
    for line in lines:
        ccys, text = line.split(": ", 1)
        groups.setdefault(tuple(ccys.split("/")), []).append(text)
    out = []
    for ccys in sorted(groups, key=len, reverse=True):         # stable: currencies keep the config's order
        sentences = [s if "*" in s.split()[0] else s[0].upper() + s[1:] for s in groups[ccys]]
        out.append((f"{_and(ccys)}:", " ".join(s.rstrip(".") + "." for s in sentences)))
    return out


def _and(names):
    names = list(names)
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _rich(col, parts, size, color, gap=0.0, left=None, width=None, top=None):
    """A paragraph of (text, bold) parts, wrapped word by word like `_Column.text`; a no-break space holds.

    Placed at `left`, `width` and `top` (figure fractions) where given, else at
    the column's cursor, which it then moves down. Returns its bottom.
    """
    fig, step = col.fig, size * 1.35 / 72 / PAGE[1]
    left = col.left if left is None else left
    width = 0.995 * (col.width if width is None else width)          # a hair short, for the PDF's rounding
    at = col.y if top is None else top
    fonts = {w: FontProperties(family=plt.rcParams["font.family"], size=size, weight="bold" if w else "normal")
             for w in (False, True)}
    px = fig.dpi * PAGE[0]

    def measure(text, bold):
        return col.renderer.get_text_width_height_descent(text, fonts[bold], ismath=False)[0] / px

    space = measure("a a", False) - measure("aa", False)

    def runs(words):
        """A line's words as (text, bold) runs of one weight."""
        out = []
        for word, bold in words:
            if out and out[-1][1] == bold:
                out[-1] = (out[-1][0] + " " + word, bold)
            else:
                out.append((word, bold))
        return out

    def length(words):
        r = runs(words)
        return sum(measure(text, bold) for text, bold in r) + space * (len(r) - 1)

    lines, line = [], []
    for word in [(w, bold) for text, bold in parts for w in text.split(" ") if w]:
        if line and length(line + [word]) > width:
            lines.append(line)
            line = []
        line.append(word)
    lines.append(line)
    for i, words in enumerate(lines):
        x = left
        for text, bold in runs(words):
            fig.text(x, at - i * step, text, fontsize=size, weight="bold" if bold else "normal", color=color,
                     va="top")
            x += measure(text, bold) + space
    bottom = at - len(lines) * step
    if top is None:
        col.y = bottom - gap / PAGE[1]
        col._check()
    return bottom


def _row(snap, k):
    return snap["meetings"].set_index("k").loc[k]


def bp(x):
    """Signed basis points, with no minus sign on a zero; n/a where there is no number."""
    if pd.isna(x):
        return "n/a"
    return f"{0.0 if round(x) == 0 else x:+.0f}bp"


def _dm(d):
    """A day as the brief names it: 1 Oct."""
    return f"{d.day} {d:%b}"


def _mid(label):
    """A config label inside a sentence: "Core PCE" becomes "core PCE"; an acronym ("CPI") stays as it is."""
    return label[0].lower() + label[1:] if len(label) > 1 and label[1].islower() else label


def typeset(text):
    """The page's typography: a true minus on negative numbers and between currencies ("GBP − USD")."""
    return re.sub(r"(?<![\w.])-(?=\d)", "−", text.replace(" - ", " − "))


def changed(ccy, now, before, k, cfg):
    """Two sentences: how the gap at the k-th meeting moved over the week, split into market and rule, and why.

    If the currency has no session after last week's read (its panel ends
    there), it says so rather than report a move of zero.
    """
    w = cfg["report"]["labels"]
    if now["session"] == before["session"]:
        return f"{ccy}: no session since {_dm(now['session'])}, so nothing to compare yet."
    a, b = _row(now, k), _row(before, k)
    d_mkt, d_mod = (a["market"] - b["market"]) * 100, (a["model"] - b["model"]) * 100
    text = (f"{ccy}, {_dm(before['session'])} to {_dm(now['session'])}: the gap at the {ORDINAL[k]} meeting went "
            f"from {bp(b['gap_bp'])} to {bp(a['gap_bp'])} (z {b['z']:+.1f} to {a['z']:+.1f}). The market moved "
            f"{bp(d_mkt)}")
    rn, rb = now["rule"], before["rule"]
    c = cfg["rule"]["coefficients"]
    reasons = []
    if rn["r0"] != rb["r0"]:
        reasons.append((abs(rn["r0"] - rb["r0"]) * 100, f"{w['policy']} moved to {rn['r0']:.2f}%"))
    if rn["inflation"] != rb["inflation"]:
        reasons.append((abs(rn["inflation"] - rb["inflation"]) * (1 + c["inflation_gap"]) * 100,
                        (f"{_mid(w['inflation'])} for {now['macro']['infl_month']:%B} printed "
                         f"{rn['inflation']:.1f}% (from {rb['inflation']:.1f}%)")))
    if rn["u_gap"] != rb["u_gap"]:
        reasons.append((abs(rn["u_gap"] - rb["u_gap"]) * c["unemployment_gap"] * 100,
                        "unemployment " + w["unemployment_period"].format(month=now["macro"]["gap_month"])
                        + f" came in at {now['macro']['u']:.1f}%"))
    if rn["rstar"] != rb["rstar"]:
        label = cfg["rule"]["rstar"].get("label", "estimate")
        reasons.append((abs(rn["rstar"] - rb["rstar"]) * 100, f"a new {label} moved r* to {rn['rstar']:.1f}%"))
    if round(d_mod) == 0:
        text += "; the rule held"
    elif abs(d_mod) >= 1 and reasons:
        text += f", and the rule {bp(d_mod)} because " + max(reasons)[1]
    elif abs(d_mod) >= 1:
        text += f", and the rule {bp(d_mod)}, only its inertial step toward an unchanged R*"
    else:
        text += f", and the rule {bp(d_mod)}"
    if a["effective_date"] != b["effective_date"]:
        text += f". The horizon rolled on to the {a['announcement_date']:%B %Y} meeting"
    return text + "."


def wrong(reads, k):
    """What would make the widest reading wrong, and which bp levels lean on a constant r*."""
    ccy, snap = max(reads.items(), key=lambda kv: abs(_row(kv[1], k)["z"]))
    cfg = config.currency(ccy)
    w, r, row = cfg["report"]["labels"], snap["rule"], _row(snap, k)
    side = "more" if row["gap_bp"] > 0 else "less"
    text = (f"The widest reading is {ccy} at the {ORDINAL[k]} meeting (z {row['z']:+.1f}, {bp(row['gap_bp'])}): the "
            f"market prices {side} tightening than the rule. It is wrong if the market is right that "
            f"{_mid(w['inflation'])} ({r['inflation']:.1f}%) and unemployment ({snap['macro']['u']:.1f}%) will move: "
            "the rule holds today's inputs flat, so a market that prices their change looks mispriced to it.")
    for c in reads:
        constant = config.currency(c)["rule"]["rstar"].get("constant")
        if constant is not None:
            text += (f" {c}'s r* is a constant ({constant:.1f}%), a hindsight number that moves the bp gap and mostly "
                     "leaves the z unchanged.")
    return text


def _table(fig, col, header, rows, widths, size=7.5, left=(0,)):
    """The tear sheet's plain table at the column's cursor: bold header, `left` columns left-aligned, the rest right.

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


def _grid(fig, col, columns, rows, size=7.5, groups=(), rule_before=(), bold=()):
    """A ruled table at the column's cursor: a heavy rule above and below, a light one under the header.

    `columns` are (header, inches, "left" or "right"); a header may run to two
    lines, and sits on the header row's last line. `groups` are (label, first,
    last) column spans, labelled over their headers with a rule under each.
    `rule_before` are row indices with a hairline above them; `bold` are column
    indices set in bold.
    """
    t, step, pad = col.t, size * 1.45 / 72, 0.07
    ascent = size * 0.95 / 72                                      # top of a row to its baseline
    edges = np.cumsum([col.left * PAGE[0], *[wd for _, wd, _ in columns]])
    y = col.y * PAGE[1]                                            # inches from the page's foot

    def rule(at, lw, color, a=edges[0], b=edges[-1]):
        fig.add_artist(Line2D([a / PAGE[0], b / PAGE[0]], [at / PAGE[1]] * 2, transform=fig.transFigure, lw=lw,
                              color=color, solid_capstyle="butt"))

    def cell(j, at, text, va="baseline", weight="normal", color=None):
        _, _, align = columns[j]
        x = edges[j] + (pad if j else 0) if align == "left" else edges[j + 1] - pad
        fig.text(x / PAGE[0], at / PAGE[1], typeset(text), fontsize=size, ha=align, va=va, weight=weight,
                 color=color or t["ink"], linespacing=1.2)

    rule(y, 0.9, t["ink"])
    y -= 0.04
    for label, a, b in groups:
        fig.text((edges[a] + edges[b + 1]) / 2 / PAGE[0], (y - ascent) / PAGE[1], typeset(label), fontsize=size,
                 ha="center", va="baseline", color=t["secondary"])
        rule(y - ascent - 0.04, 0.5, t["axis"], edges[a] + pad, edges[b + 1] - pad)
    if groups:
        y -= ascent + 0.07
    lines = max(h.count("\n") + 1 for h, _, _ in columns)
    y -= ascent + (lines - 1) * size * 1.2 / 72                    # a header's last line sits on this baseline
    for j, (h, _, _) in enumerate(columns):
        cell(j, y, h, weight="bold")
    y -= 0.05
    rule(y, 0.5, t["axis"])
    y -= 0.03
    for i, row in enumerate(rows):
        if i in rule_before:
            rule(y, 0.5, t["grid"])
            y -= 0.03
        for j, text in enumerate(row):
            if text is None:                                       # a spacer column
                continue
            muted = text in ("", "—")
            cell(j, y - ascent, text or "—", weight="bold" if j in bold and not muted else "normal",
                 color=t["muted"] if muted else None)
        y -= step
    rule(y, 0.9, t["ink"])
    col.y = (y - 0.08) / PAGE[1]
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
    """reports/brief_<date>.pdf. Returns (path, the numbers it printed).

    The page is laid out with text measured unhinted, as the PDF sets it: hinted,
    small text measures a few percent narrow and runs past the margin.
    """
    with plt.rc_context({"text.hinting": "no_hinting"}):
        return _write(date, out_dir, root, brief, preview, generated)


def _write(date, out_dir, root, brief, preview, generated):
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
    same = len({s["session"] for s in reads.values()}) == 1
    sessions = (f"{' and '.join(reads)} as of {_day(next(iter(reads.values()))['session'])}" if same else
                "  ·  ".join(f"{c} as of {_day(reads[c]['session'])}" for c in reads))
    eight = len(next(iter(reads.values()))["meetings"])
    col.text("Does the market price the Fed's own rule?", size=15, weight="bold", gap=0.03)
    col.text(f"Weekly brief, {_day(date)}  ·  {sessions}  ·  generated {generated:%Y-%m-%d}", size=8.5,
             color=t["secondary"], gap=0.08)
    col.text(f"The policy path each market prices over the next {NUMBER.get(eight, eight)} meetings, against the path "
             "the Fed's own balanced-approach rule takes on that currency's data public that day, today's inputs held "
             "flat. The gap is market minus rule, z-scored against its trailing two years.", size=8.5, gap=0.1)

    # One chart per currency, side by side on one time axis, under one key.
    marked = list(dict.fromkeys([k, horizons[-1]]))
    key = [Line2D([], [], color=t["ink"], lw=2), Line2D([], [], color=t["series"][0], lw=2),
           Line2D([], [], color=t["series"][1], lw=2)]
    names = ["Realized overnight rate", "Market-implied path", "Rule path, today's inputs held flat"]
    fig.legend(key, names, loc="upper left", bbox_to_anchor=(col.left, col.y), ncols=len(key), fontsize=7.5,
               labelcolor=t["secondary"], handlelength=1.8, columnspacing=1.8, borderaxespad=0, borderpad=0)
    fig.text(col.left + col.width, col.y - 0.015 / PAGE[1], f"Brackets: the gap at the "
             f"{' and '.join(ORDINAL[h] for h in marked)} meetings", fontsize=7.5, ha="right", va="top",
             color=t["secondary"])
    col.y -= 0.27 / PAGE[1]
    sep, room_left, room_right, height = 0.45, 0.32, 0.48, 1.74          # inches
    each = (col.width * PAGE[0] - sep * (len(reads) - 1)) / len(reads)
    head, top = col.y - 0.12 / PAGE[1], col.y - 0.26 / PAGE[1]
    drawn = []
    for i, (ccy, snap) in enumerate(reads.items()):
        cfg = config.currency(ccy)
        w = cfg["report"]["labels"]
        x0 = col.left + i * (each + sep) / PAGE[0]
        title = fig.text(x0, head, f"{ccy}: {w['policy']}", fontsize=10, weight="bold", va="baseline",
                         color=t["ink"])
        fig.text(x0 + title.get_window_extent(col.renderer).width / (fig.dpi * PAGE[0]) + 0.1 / PAGE[0], head,
                 f"next {len(snap['meetings'])} {w['meetings']} meetings, %; market path from {w['market']}",
                 fontsize=7.5, va="baseline", color=t["secondary"])
        ax = fig.add_axes([x0 + room_left / PAGE[0], top - height / PAGE[1],
                           (each - room_left - room_right) / PAGE[0], height / PAGE[1]])
        overnight = cache.log(cfg["overnight"]["source"], cfg["overnight"]["series"], ccy)
        charts.paths_now(ax, snap, overnight, t, w, lead_days=90, gaps=(), legend=False, ends=("Market", "Rule"),
                         size=7.5)
        realized = ax.lines[0]
        ax.annotate(w["rate"], xy=(realized.get_xdata()[0], realized.get_ydata()[0]), xytext=(0, 4),
                    textcoords="offset points", va="bottom", fontsize=7.5, color=t["secondary"])
        ax.set_ylabel("")
        ax.tick_params(labelsize=7)
        drawn.append((ax, snap))
    lo, hi = min(a.get_xlim()[0] for a, _ in drawn), max(a.get_xlim()[1] for a, _ in drawn)
    for ax, snap in drawn:
        ax.set_xlim(lo, hi)
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        charts.gap_marks(ax, snap, t, ks=marked, size=7.5)
    col.y = top - (height + 0.26) / PAGE[1]

    # The gaps and their z per currency, then the differential (its z only).
    n = len(horizons)
    col.text(f"The gaps at the {', '.join(ORDINAL[h] for h in horizons[:-1])} and {ORDINAL[horizons[-1]]} meetings "
             "ahead", size=10, weight="bold", gap=0.05)
    rows = []
    for ccy, snap in reads.items():
        m = snap["meetings"].set_index("k")
        rows.append([ccy, *[bp(m.loc[h, "gap_bp"])[:-2] for h in horizons], None,
                     *[f"{m.loc[h, 'z']:+.1f}" for h in horizons]])
    rows.append([f"{first} - {second}†", *["—"] * n, None, *[f"{diff_now.loc[h, 'z']:+.1f}" for h in horizons]])
    gaps = [("", 1.1, "left"), *[(SHORT[h], 0.62, "right") for h in horizons], ("", 0.3, "left"),
            *[(SHORT[h], 0.5, "right") for h in horizons]]
    table_top = col.y
    _grid(fig, col, gaps, rows, groups=[("Gap, market − rule, bp", 1, n), ("z", n + 2, 2 * n + 1)],
           rule_before=[len(reads)], bold=[1 + horizons.index(k), n + 2 + horizons.index(k)])
    # Beside the table: what the bold column is, and why the differential has no bp.
    as_of = "" if same and diff_day == next(iter(reads.values()))["session"] else f" (as of {_dm(diff_day)})"
    beside = col.left + (sum(wd for _, wd, _ in gaps) + 0.3) / PAGE[0]
    note = _rich(col, [(typeset(f"Bold: the {ORDINAL[k]} meeting, the one the book trades. †\u00a0The {first} gap "
                                f"minus the {second} gap on the sessions both have{as_of}, z-scored the same way. Its "
                                "bp are not shown: the two gaps take r* by different methods, so the bp difference is "
                                "mostly that choice, and only the z compares."), False)],
                 size=7, color=t["secondary"], left=beside, width=col.left + col.width - beside,
                 top=table_top - 0.03 / PAGE[1])
    col.y = min(col.y, note - 0.08 / PAGE[1]) - 0.04 / PAGE[1]

    # The trades: each sleeve on its last session, before costs, and its round trip.
    book = config.strategy()
    h, enter, exit_ = book["carry"]["horizon_days"], book["positions"]["enter"], book["positions"]["exit"]
    proxies = list(dict.fromkeys(s["kind"] for s in book["sleeves"] if s["kind"] != "outright"))
    proxy = f" (for the {_and(proxies)} sleeves, a proxy for the legs' move)" if proxies else ""
    ticked = [config.currency(c)["expression"]["outright"]["label"] for c in reads
              if config.currency(c)["costs"]["outright"].get("observed")]
    ticks = f"{_and(ticked)} at its tick, the rest assumed" if ticked else "all assumed"
    trade_rows = trades(expression.load(root), date, enter, {s["name"]: costs.round_trip(s) for s in book["sleeves"]})
    dated = len({r[1] for r in trade_rows} | {f"{s['session']:%d %b}" for s in reads.values()}) > 1
    columns = [("Sleeve", 0.95, "left"), *([("As of", 0.5, "left")] if dated else []), ("z", 0.4, "right"),
               ("Side", 0.6, "left"), ("Instrument", 1.4 if dated else 1.9, "left"), ("Edge", 0.55, "right"),
               (f"Carry and\nroll, {h}d", 0.7, "right"), (f"Expected,\n{h}d", 0.65, "right"),
               ("Round\ntrip", 0.5, "right"), ("Breakeven", 1.05, "left")]
    shown = [[name, *([when.lstrip("0")] if dated else []), z, side, instrument,
              *[c.removesuffix("bp") for c in (edge, cr, e, rt)], verdict]
             for name, when, z, side, instrument, edge, cr, e, rt, verdict in trade_rows]
    col.text("The trades: each sleeve's signal, its carry and roll, and its round trip (bp per unit DV01)", size=10,
             weight="bold", gap=0.05)
    _grid(fig, col, columns, shown)
    _rich(col, [(typeset(f"Edge: |z| × the gap's trailing sd, the bp the signal expects to close{proxy}. Carry and "
                         f"roll: earned over the next {h} days if the curve stands still, signed for the side. "
                         "Expected: the share of the gap that has historically closed, plus carry and roll on the "
                         f"rest. Round trip: all legs in and out at the configured costs ({ticks}). Breakeven: "
                         "bleeds if carry and roll runs against the side; pays if the expected quarter is "
                         f"positive. * |z|\u00a0<\u00a0{enter:g}: no entry (enter at {enter:g}, exit at {exit_:g})."),
              False)],
          size=6.6, color=t["secondary"], gap=0.1)

    col.text("What changed since last week", size=10, weight="bold", gap=0.03)
    for ccy in reads:
        col.text(typeset(changed(ccy, reads[ccy], last_week[ccy], k, config.currency(ccy))), size=8.5, gap=0.03)
    col.y -= 0.06 / PAGE[1]
    col.text("What would make this wrong", size=10, weight="bold", gap=0.03)
    col.text(typeset(wrong(reads, k)), size=8.5, gap=0.1)
    parts = [("Limitations, from the data-quality and lag tags in config.", True)]
    parts += [p for lead, body in limitations(footer(brief["currencies"], brief["footer_blocks"]))
              for p in [(lead, True), (typeset(body), False)]]
    _rich(col, parts, size=6.2, color=t["secondary"])

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

