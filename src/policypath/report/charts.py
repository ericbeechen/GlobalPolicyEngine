"""The charts the one-pages and reports embed: the path and gap, and the attribution charts.

Generated, never hand-edited. Written in light and dark like the README figures,
with the same themes (`figures.THEMES`): slot 1 is the market, slot 2 the model.
The drawing functions take their axes, so the one-pager and the tear sheet
embed the same charts as the reports.

Attribution (`report/attribution.py`, `report/tearsheet.py`): the book's net
equity by component group (`equity_by_component`), its net Sharpe against the
round trip assumed (`cost_curve`), the IC by sleeve and horizon (`ic_bars`)
and the net Sharpe by regime (`regime_bars`). Categorical colours follow the
entity in a fixed order (the caller's palette, `report.portfolio.CATEGORICAL`);
the IC's horizons are a magnitude, so they take the blue ramp, short to long
light to dark.
"""

from itertools import pairwise
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from policypath.report.figures import BLUE, THEMES, _style, _title


def _dates(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="both", length=0)


def paths_now(ax, today, effr, t, labels, lead_days=150, tail_days=40, gaps=(4, 8), legend=True, ends=None,
              size=9):
    """Market and model step paths as of one session, after the realized overnight rate.

    `gaps` are the meetings whose gap is labelled on the chart (none for the brief,
    which brackets them with `gap_marks`); `ends` overrides the two direct labels at
    the paths' ends, which are pushed apart where they would overlap.
    """
    w = labels
    day = today["session"]
    m = today["meetings"]
    realized = effr.set_index("date")["value"]
    realized = realized[(realized.index >= day - pd.Timedelta(days=lead_days)) & (realized.index < day)]
    daily = realized.reindex(pd.date_range(realized.index[0], day, freq="D")).ffill()
    ax.step(daily.index, daily.to_numpy(), where="post", color=t["ink"], lw=2)

    end = m["effective_date"].iloc[-1] + pd.Timedelta(days=tail_days)
    dates = [day, *m["effective_date"], end]
    market = [today["rate_now"], *m["market"], m["market"].iloc[-1]]
    model = [today["model_now"], *m["model"], m["model"].iloc[-1]]
    ax.step(dates, market, where="post", color=t["series"][0], lw=2)
    ax.step(dates, model, where="post", color=t["series"][1], lw=2)
    ax.axvline(day, color=t["axis"], lw=1)
    ends = ends or (f"Market ({w['market']})", w["rule"])
    _end_labels(ax, end, [market[-1], model[-1]], ends, t, size)
    for k in gaps:
        row = m[m["k"] == k].iloc[0]
        x = row["effective_date"] + (m["effective_date"].iloc[k] - row["effective_date"]) / 2 \
            if k < len(m) else row["effective_date"] + pd.Timedelta(days=tail_days / 2)
        # A wide gap is labelled between the two paths, a narrow one just under them.
        if abs(row["gap_bp"]) > 30:
            xy, offset, va = (x, (row["market"] + row["model"]) / 2), (0, 0), "center"
        else:
            xy, offset, va = (x, min(row["market"], row["model"])), (0, -10), "top"
        ax.annotate(f"{row['gap_bp']:+.0f}bp", xy=xy, xytext=offset, textcoords="offset points",
                    ha="center", va=va, fontsize=9, color=t["secondary"])
    ax.set_ylabel("Percent")
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.tick_params(axis="both", length=0)
    if legend:
        ax.legend(handles=[Line2D([], [], color=t["ink"], lw=2, label=f"Realized {w['rate']}"),
                           Line2D([], [], color=t["series"][0], lw=2, label=f"Market: {w['market']}-implied path"),
                           Line2D([], [], color=t["series"][1], lw=2,
                                  label="Model: balanced-approach rule, inertial, macro held flat")],
                  loc="upper left", fontsize=9, labelcolor=t["secondary"])


def _end_labels(ax, x, ys, labels, t, size):
    """Direct labels just right of `x` at each of `ys`, nudged apart in points where two would overlap."""
    ax.get_ylim()                                         # settle the autoscale before measuring
    px = ax.transData.transform([(0, y) for y in ys])[:, 1] * 72 / ax.figure.dpi      # points
    need = size * 1.25
    order = np.argsort(px)
    dy = np.zeros(len(ys))
    for lo, hi in pairwise(order):
        short = need - (px[hi] + dy[hi] - px[lo] - dy[lo])
        if short > 0:
            dy[lo] -= short / 2
            dy[hi] += short / 2
    for label, y, d in zip(labels, ys, dy):
        ax.annotate(label, xy=(x, y), xytext=(6, d), textcoords="offset points", va="center", fontsize=size,
                    color=t["secondary"])


def gap_marks(ax, today, t, ks=(4, 8), tail_days=40, size=7.5):
    """A bracket between the two paths at each of the `ks` meetings, labelled with the gap there.

    The label goes beside the bracket where the paths leave room for it across
    its width, else just under the lower path or over the upper one, whichever
    is nearer and clear of both. It never runs past the paths' end, where the
    direct labels are. Set the axes' limits first: the placement is measured.
    """
    m = today["meetings"].reset_index(drop=True)
    end = m["effective_date"].iloc[-1] + pd.Timedelta(days=tail_days)
    edges = mdates.date2num([today["session"], *m["effective_date"], end])   # step i holds on [edges[i], edges[i+1])
    steps = np.array([[today["rate_now"], *m["market"]], [today["model_now"], *m["model"]]])
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    to_px, to_data = ax.transData.transform, ax.transData.inverted().transform
    y0, y1 = ax.get_ylim()
    pad = 3 * fig.dpi / 72

    def span(xa, xb):
        """Each path's lowest and highest pixel in [xa, xb] (its steps and the risers between them), by path."""
        on = (edges[1:] > xa) & (edges[:-1] < xb)
        px = to_px([(0, v) for v in steps[:, on].ravel()])[:, 1].reshape(2, -1)
        return px.min(axis=1), px.max(axis=1)

    for k in ks:
        i = int(m.index[m["k"] == k][0]) + 1
        x = (edges[i] + edges[i + 1]) / 2
        a, b = steps[0, i], steps[1, i]
        ax.annotate("", xy=(x, a), xytext=(x, b), arrowprops={"arrowstyle": "|-|,widthA=0.25,widthB=0.25",
                    "color": t["secondary"], "lw": 0.8, "shrinkA": 0, "shrinkB": 0})
        label = ax.text(x, (a + b) / 2, f"{m.loc[i - 1, 'gap_bp']:+.0f}bp", fontsize=size, color=t["secondary"])
        box = label.get_window_extent(renderer)
        w, h = box.width, box.height
        xp = to_px([(x, 0)])[0, 0]
        top, bottom = to_px([(0, max(a, b))])[0, 1], to_px([(0, min(a, b))])[0, 1]
        upper = 0 if a >= b else 1
        right_end = to_px([(edges[-1], 0)])[0, 0] - pad
        spots = []
        for ha, xa in [("left", xp + pad), ("right", xp - pad - w)]:      # beside: the band between the paths
            if xa + w > right_end:
                continue
            lo, hi = span(*to_data([(xa, 0), (xa + w, 0)])[:, 0])
            floor, ceiling = hi[1 - upper], lo[upper]                    # the lower path's top, the upper's bottom
            if ceiling - floor >= h + pad:
                spots.append((0, ha, xa if ha == "left" else xa + w, (floor + ceiling) / 2, "center"))
        if not spots:                                                      # under or over, centred on the bracket
            xc = min(xp, right_end - w / 2)
            lo, hi = span(*to_data([(xc - w / 2, 0), (xc + w / 2, 0)])[:, 0])
            under, over = lo.min() - pad, hi.max() + pad
            spots += [(bottom - under, "center", xc, under, "top"), (over - top, "center", xc, over, "bottom")]
            fits = [s for s in spots if s[4] == "top" and s[3] - h >= to_px([(0, y0)])[0, 1]
                    or s[4] == "bottom" and s[3] + h <= to_px([(0, y1)])[0, 1]]
            spots = sorted(fits or spots, key=lambda s: s[0])
        _, ha, xa, ya, va = spots[0]
        label.set_position(to_data([(xa, ya)])[0])
        label.set_ha(ha)
        label.set_va(va)


def gap_history(ax, signal, k, t, moments=()):
    """The gap at horizon k in bp, with the trailing mean +- 1 sd its z is measured against.

    `moments` are (from, to, label): each is marked where the gap is widest inside it.
    """
    s = signal[signal["k"] == k].set_index("session").sort_index()
    s = s[s["gap_bp"].notna()]
    ax.axhline(0, color=t["axis"], lw=1)
    band = s["window_mean_bp"].notna()
    ax.fill_between(s.index[band], (s["window_mean_bp"] - s["window_sd_bp"])[band],
                    (s["window_mean_bp"] + s["window_sd_bp"])[band], color=t["grid"], lw=0,
                    label="Trailing two years: mean ± 1 sd (the z's yardstick)")
    ax.plot(s.index, s["gap_bp"], color=t["series"][0], lw=1.5, label=f"Gap at meeting {k} ahead")
    # Labels sit in two rows under the data, each tied to its point by a leader,
    # so none of them lands on the line.
    low = min(s["gap_bp"].min(), (s["window_mean_bp"] - s["window_sd_bp"]).min())
    rows = [low - 30, low - 55]   # label rows, below everything
    for i, (lo, hi, label) in enumerate(moments):
        w = s.loc[lo:hi, "gap_bp"]
        if w.empty:
            continue
        at = w.abs().idxmax()
        y = w[at]
        row = rows[i % 2]
        ax.plot([at, at], [y - 6, row + 11], color=t["muted"], lw=0.8)
        ax.plot([at], [y], "o", ms=5, color=t["series"][0], mec=t["surface"], mew=1.5, zorder=3)
        ax.text(at, row, label, ha="center", va="center", fontsize=8, color=t["secondary"])
    last = s.iloc[-1]
    ax.annotate(f"{last['gap_bp']:+.0f}bp\nz {last['z']:+.1f}", xy=(s.index[-1], last["gap_bp"]),
                xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=t["secondary"])
    ax.set_ylabel("bp, market − model")
    ax.set_ylim(rows[1] - 12, s["gap_bp"].max() + 55)   # headroom so the legend clears the data
    ax.legend(loc="upper left", fontsize=9, labelcolor=t["secondary"], ncols=2)
    _dates(ax)


def ship(today, effr, signal, k, labels, path=None, theme="light", axes=None, moments=()):
    """The one figure: paths as of today on top, the gap's history underneath."""
    t = _style(theme)
    if axes is None:
        fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 8.2),
                                          gridspec_kw={"height_ratios": [1.35, 1], "hspace": 0.42})
    else:
        fig, (top, bottom) = axes[0].figure, axes
    w = labels
    paths_now(top, today, effr, t, w)
    day = today["session"]
    _title(top, f"The {w['policy']} path: market against {w['rule'][0].lower() + w['rule'][1:]}, {day:%d %B %Y}",
           f"Next {len(today['meetings'])} {w['meetings']} meetings. The rule uses only data public on the day, "
           "and holds today's inflation, unemployment gap and r* flat.", t)
    gap_history(bottom, signal, k, t, moments)
    first = signal.loc[signal["gap_bp"].notna(), "session"].min()
    _title(bottom, f"The gap at the {k}th meeting ahead, {first:%Y}-{day:%Y}",
           "Negative: the market prices less tightening than the rule. The z is the gap against its band.", t)
    if path is not None:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def equity(result, summary, k, path, theme="light"):
    """The crude backtest's cumulative P&L, with the missing costs said on the chart."""
    t = _style(theme)
    fig, ax = plt.subplots(figsize=(11, 4.4))
    live = result[result["position"].notna()]
    eq = live["pnl"].cumsum()
    ax.axhline(0, color=t["axis"], lw=1)
    ax.plot(eq.index, eq.to_numpy(), color=t["series"][0], lw=2)
    ax.annotate(f"{eq.iloc[-1]:+.0f}bp", xy=(eq.index[-1], eq.iloc[-1]), xytext=(6, 0),
                textcoords="offset points", va="center", fontsize=9, color=t["secondary"])
    ax.text(0.01, 0.98, "No transaction costs: these overstate what was achievable.\nCosts are in reports/costs.md.",
            transform=ax.transAxes, ha="left", va="top", fontsize=9, color=t["ink"], fontweight="bold")
    ax.set_ylabel("bp x z, cumulative")
    _dates(ax)
    _title(ax, f"Crude backtest: trade the implied rate at the {k}th meeting ahead on the gap z",
           f"Receive when z > 0, pay when z < 0, size = z, DV01 fixed, one session's lag. "
           f"{summary['first']:%b %Y}-{summary['last']:%b %Y}: Sharpe {summary['sharpe']:.2f}, "
           f"max drawdown {summary['max_drawdown']:.0f}bp.", t)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_all(today, effr, signal, result, summary, k, out_dir, ccy, labels, moments=()):
    """The ship chart and the equity curve, light and dark. `labels` is the config's ``report.labels``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for theme in THEMES:
        p = out_dir / f"model_vs_market_{ccy}_{theme}.png"
        ship(today, effr, signal, k, labels, p, theme, moments=moments)
        q = out_dir / f"backtest_{ccy}_{theme}.png"
        equity(result, summary, k, q, theme)
        written += [p, q]
    return written


# ---- attribution -------------------------------------------------------------------

def _end_label(ax, x, text, t, dy=0):
    ax.annotate(text, xy=(x.index[-1], x.iloc[-1]), xytext=(6, dy), textcoords="offset points", va="center",
                fontsize=8, color=t["secondary"])


def equity_by_component(ax, curves, t, colors, windows=(), title=True, ncols=3):
    """Cumulative net P&L, % of capital: the book (bold, ink), its gross (dashed) and each component group.

    `curves` is {"book": series, "book gross": series, group: series, ...};
    `windows` are (first, last, label) shaded behind the lines.
    """
    for lo, hi, label in windows:
        ax.axvspan(lo, hi, color=t["grid"], lw=0, zorder=0)
        ax.text(lo + (hi - lo) / 2, 1.0, label, transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=7.5, color=t["secondary"])
    ax.axhline(0, color=t["axis"], lw=1)
    groups = [g for g in curves if not g.startswith("book")]
    handles = []
    for g, color in zip(groups, colors):
        x = curves[g]
        ax.plot(x.index, x.to_numpy(), color=color, lw=1.6)
        handles.append(Line2D([], [], color=color, lw=1.6, label=f"{g} {x.iloc[-1]:+.1f}%"))
    g = curves["book gross"]
    ax.plot(g.index, g.to_numpy(), color=t["ink"], lw=1.2, ls=(0, (4, 2)))
    b = curves["book"]
    ax.plot(b.index, b.to_numpy(), color=t["ink"], lw=2.4)
    handles = [Line2D([], [], color=t["ink"], lw=2.4, label=f"book, net {b.iloc[-1]:+.1f}%"),
               Line2D([], [], color=t["ink"], lw=1.2, ls=(0, (4, 2)), label=f"book, gross {g.iloc[-1]:+.1f}%"),
               *handles]
    lo = min(float(x.min()) for x in curves.values())
    hi = max(float(x.max()) for x in curves.values())
    ax.set_ylim(lo - 0.38 * (hi - lo), hi + 0.08 * (hi - lo))      # room under the data for the legend
    ax.legend(handles=handles, fontsize=7.5, labelcolor=t["secondary"], loc="lower left", ncols=ncols,
              handlelength=1.6, columnspacing=1.0)
    ax.set_ylabel("cumulative P&L, % of capital", fontsize=8)
    _dates(ax)
    ax.tick_params(labelsize=8)
    if title:
        ax.set_title("Net equity by component", loc="left", fontsize=10, fontweight="bold", color=t["ink"])


def cost_curve(ax, c, t, colors, title=True):
    """The book's net Sharpe against a uniform round trip on every leg (solid) and on the assumed legs only (dashed),
    the configured mix and the breakeven marked. `c` is `report.attribution.cost_curve`'s dict."""
    xs = np.asarray(c["round_trip_bp"], dtype=float)
    ax.axhline(0, color=t["axis"], lw=1)
    ax.plot(xs, c["net_sr"], color=colors[0], lw=2, label="every leg at the round trip")
    ax.plot(xs, c["assumed_net_sr"], color=colors[1], lw=1.6, ls=(0, (4, 2)),
            label="the assumed legs only (tick-costed legs fixed)")
    ax.plot([c["mix_bp"]], [c["net_sr_at_mix"]], "o", ms=7, color=colors[0], mec=t["surface"], mew=2, zorder=3)
    ax.annotate(f"configured mix {c['mix_bp']:.2f}bp: {c['net_sr_at_mix']:+.2f}", xy=(c["mix_bp"], c["net_sr_at_mix"]),
                xytext=(8, -4), textcoords="offset points", va="top", fontsize=8, color=t["secondary"])
    star = c["cstar_bp"]
    note = ("no breakeven: gross P&L is below zero" if star is None or not np.isfinite(star)
            else f"breakeven {star:.2f}bp")
    if star is not None and np.isfinite(star):
        ax.axvline(star, color=t["muted"], lw=1, ls=":")
    ax.text(0.98, 0.95, note, transform=ax.transAxes, ha="right", va="top", fontsize=8, color=t["secondary"])
    ax.set_xlabel("round trip, bp", fontsize=8)
    ax.set_ylabel("net Sharpe", fontsize=8)
    ax.legend(fontsize=7.5, labelcolor=t["secondary"], loc="lower left")
    ax.tick_params(labelsize=8, length=0)
    if title:
        ax.set_title("Net Sharpe against the cost assumed", loc="left", fontsize=10, fontweight="bold",
                     color=t["ink"])


def ic_bars(ax, ics, t, theme, title=True):
    """IC by sleeve and horizon (`metrics.ic_table` rows per sleeve), grouped bars, the blue ramp by horizon."""
    names = list(ics)
    hs = [r["h"] for r in ics[names[0]]]
    ramp = (BLUE[1:4] if theme == "dark" else BLUE[3:0:-1])[:len(hs)]    # on a dark surface, more is lighter
    width = 0.8 / len(hs)
    ax.axhline(0, color=t["axis"], lw=1)
    for j, (h, color) in enumerate(zip(hs, ramp)):
        v = [ics[n][j]["ic"] for n in names]
        ax.bar(np.arange(len(names)) + (j - (len(hs) - 1) / 2) * width, v, width * 0.9, color=color,
               label=f"{h} sessions", edgecolor=t["surface"], lw=1)
    ax.set_xticks(np.arange(len(names)), names, fontsize=7.5)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.2 * (hi - lo))                            # headroom for the legend
    ax.set_ylabel("rank IC", fontsize=8)
    ax.tick_params(labelsize=8, length=0)
    ax.legend(fontsize=7.5, labelcolor=t["secondary"], loc="upper left", ncols=len(hs))
    if title:
        ax.set_title("IC of z against the next sessions' rate change, outside the ELB", loc="left", fontsize=10,
                     fontweight="bold", color=t["ink"])


def regime_bars(ax, book, sleeves, t, marks, title=True, min_sessions=250):
    """The book's net Sharpe by collapsed regime with a one-SE whisker (bars), each sleeve's as a mark on it.

    `book` is {regime: row} in order (`report.attribution.regime_rows`'s rows:
    net_sr, net_se, sessions); `sleeves` {name: rows in the same order};
    `marks` {name: (colour, marker)}, so a sleeve keeps its component's colour.
    The scale is set by the regimes with `min_sessions` or more; a regime with
    fewer is drawn, and a mark past the scale is pinned to its edge and hollow,
    with the regime's label saying so.
    """
    order = list(book)
    x = np.arange(len(order))
    vals = np.array([book[g]["net_sr"] for g in order], dtype=float)
    ses = np.array([book[g]["net_se"] for g in order], dtype=float)
    dots = {n: np.array([r["net_sr"] for r in rs], dtype=float) for n, rs in sleeves.items()}
    big = np.array([book[g]["sessions"] >= min_sessions for g in order])
    seen = np.concatenate([vals[big] - ses[big], vals[big] + ses[big], *[v[big] for v in dots.values()]])
    lo, hi = np.nanmin(seen) - 0.3, np.nanmax(seen) + 0.3
    ax.axhline(0, color=t["axis"], lw=1)
    ax.bar(x, np.clip(vals, lo, hi), 0.6, color=t["muted"], edgecolor=t["surface"], lw=1, zorder=2)
    ax.errorbar(x, np.clip(vals, lo, hi), yerr=np.where(big, ses, np.nan), fmt="none", ecolor=t["secondary"],
                elinewidth=1, capsize=3, zorder=3)
    off = ~big | (vals < lo) | (vals > hi)
    for n, v in dots.items():
        color, marker = marks[n]
        out = (v < lo) | (v > hi)
        off |= out
        ax.plot(x[~out], v[~out], marker, ms=6, color=color, mec=t["surface"], mew=1.5, ls="none", zorder=4,
                label=n)
        ax.plot(x[out], np.clip(v[out], lo, hi), marker, ms=6, mfc="none", mec=color, mew=1.5, ls="none", zorder=4)
    ax.set_ylim(lo, hi)
    labels = [f"{g}{'*' if off[i] else ''}\n{book[g]['sessions']:,}" for i, g in enumerate(order)]
    ax.set_xticks(x, labels, fontsize=7.5)
    ax.set_ylabel("net Sharpe", fontsize=8)
    ax.tick_params(labelsize=8, length=0)
    handles, names = ax.get_legend_handles_labels()
    if off.any():
        handles.append(Line2D([], [], ls="none"))
        names.append(f"* under {min_sessions} sessions, or off the scale (hollow)")
    ax.legend(handles, names, fontsize=7, labelcolor=t["secondary"], loc="upper left", bbox_to_anchor=(0, -0.2),
              ncols=3, handletextpad=0.2, columnspacing=0.8, borderaxespad=0.0)
    if title:
        ax.set_title("Net Sharpe by regime: the book (bars, 1 SE) and each sleeve", loc="left", fontsize=10,
                     fontweight="bold", color=t["ink"])
