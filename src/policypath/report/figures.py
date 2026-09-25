"""Figures for the README, drawn from the panel. Generated, never hand-edited.

Each figure is written twice, for light and dark surfaces, so the README can
serve the one matching the reader's theme. Colours are the reference palette's
(categorical slots 1-2 and the blue ramp, unchanged); the red arm of the
diverging ramp is matched to the blue steps in OKLab lightness.
"""

from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

BLUE = ["#0d366b", "#1c5cab", "#3987e5", "#86b6ef", "#b7d3f6"]  # ramp steps 700..150
RED = ["#641819", "#a1302f", "#da5450", "#ec9992", "#f6c2bd"]   # matched to BLUE in L
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7", "series": ["#2a78d6", "#eb6834"],
              "diverging": [*BLUE, "#f0efec", *RED[::-1]]},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "secondary": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835", "series": ["#3987e5", "#d95926"],
             # on a dark surface more magnitude is more light, so the arms run outward to pale
             "diverging": [*BLUE[::-1], "#383835", *RED]},
}


def _style(theme):
    t = THEMES[theme]
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10,
        "figure.facecolor": t["surface"], "axes.facecolor": t["surface"],
        "savefig.facecolor": t["surface"], "text.color": t["ink"],
        "axes.edgecolor": t["axis"], "axes.labelcolor": t["secondary"],
        "xtick.color": t["muted"], "ytick.color": t["muted"],
        "xtick.labelcolor": t["secondary"], "ytick.labelcolor": t["secondary"],
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": t["grid"], "grid.linewidth": 0.8,
        "grid.linestyle": "-", "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "axes.axisbelow": True, "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round", "legend.frameon": False,
    })
    return t


def _title(ax, title, subtitle, t):
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=t["ink"], pad=22)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color=t["secondary"], va="bottom")


def monthly_paths(sessions, meetings, tail_days=45):
    """The step path on the first session of each month, as (session, dates, rates)."""
    ok = sessions[sessions["error"].isna()].set_index("session")
    firsts = ok.groupby(ok.index.to_period("M")).head(1).index
    by = dict(tuple(meetings.groupby("session")))
    out = []
    for day in firsts:
        m = by[day].sort_values("k")
        dates = [ok.loc[day, "first_unknown"], *m["effective_date"],
                 m["effective_date"].iloc[-1] + pd.Timedelta(days=tail_days)]
        rates = [ok.loc[day, "rate_now"], *m["rate"], m["rate"].iloc[-1]]
        out.append((day, dates, rates))
    return out


def implied_paths(sessions, meetings, effr, path, theme="light"):
    """Monthly implied paths over realized EFFR, and the implied move by meeting horizon."""
    t = _style(theme)
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 8.2), sharex=True,
                                      gridspec_kw={"height_ratios": [2.1, 1], "hspace": 0.32})
    paths = monthly_paths(sessions, meetings)
    for _, dates, rates in paths:
        top.step(dates, rates, where="post", color=t["series"][0], lw=1.1, alpha=0.6)
    realized = effr.set_index("date")["value"]
    # A fixing holds until the next one (weekends carry Friday's, as in ZQ's average),
    # so it is a step on calendar days; a line would slope a Friday print into Monday.
    daily = realized.reindex(pd.date_range(realized.index[0], realized.index[-1], freq="D")).ffill()
    top.step(daily.index, daily.to_numpy(), where="post", color=t["ink"], lw=2)
    top.set_ylabel("Percent")
    top.set_ylim(bottom=-0.1)
    top.legend(handles=[Line2D([], [], color=t["ink"], lw=2, label="Realized EFFR"),
                        Line2D([], [], color=t["series"][0], lw=1.5,
                               label="Implied path, first session of each month (next 8 meetings)")],
               loc="upper left", fontsize=9, labelcolor=t["secondary"])

    # The one annotation: what the market priced a year out, just before 2022.
    day, dates, rates = next(p for p in paths if p[0] >= pd.Timestamp("2021-12-01"))
    target = day + pd.DateOffset(years=1)
    priced = pd.Series(rates, index=pd.DatetimeIndex(dates)).asof(target)
    printed = realized.asof(target)
    top.annotate(f"On {day.day} {day:%b %Y} ZQ priced {priced:.2f}% for {target:%b %Y}.\n"
                 f"EFFR then printed {printed:.2f}%.",
                 xy=(target, priced), xytext=(12, -2), textcoords="offset points",
                 va="center", fontsize=9, color=t["secondary"])
    top.plot([target], [priced], "o", ms=6, color=t["series"][0], mec=t["surface"], mew=2)
    _title(top, f"The fed funds path implied by ZQ futures, {paths[0][0].year}-{paths[-1][0].year}",
           "Each blue step is one session's solved path; the heavy line is what the rate then did. "
           "Its 2015-17 notches are real: EFFR printed 5-12bp low on the last day of most months.", t)

    grid = meetings.pivot(index="k", columns="session", values="cum_bp")
    x = mdates.date2num(grid.columns.to_numpy())
    edges = np.append(x, x[-1] + 1)
    vmax = float(np.ceil(np.nanpercentile(np.abs(grid.to_numpy()), 99.5) / 50) * 50)
    cmap = LinearSegmentedColormap.from_list("diverging", t["diverging"])
    mesh = bottom.pcolormesh(edges, np.arange(0.5, grid.shape[0] + 1), grid.to_numpy(),
                             cmap=cmap, vmin=-vmax, vmax=vmax, shading="flat", rasterized=True)
    bottom.set_yticks(range(1, grid.shape[0] + 1))
    bottom.set_ylim(grid.shape[0] + 0.5, 0.5)
    bottom.set_ylabel("Meetings ahead")
    bottom.grid(False)
    # An inset, so the colorbar takes no width from the panel and both panels share x.
    bar = fig.colorbar(mesh, cax=bottom.inset_axes([1.012, 0.0, 0.014, 1.0]))
    bar.set_label("bp", color=t["secondary"])
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=t["muted"], labelcolor=t["secondary"])
    _title(bottom, "Implied move from today's rate, by meeting",
           "Red is hikes priced, blue is cuts; each column is one session.", t)
    bottom.xaxis.set_major_locator(mdates.YearLocator())
    bottom.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for ax in (top, bottom):
        ax.tick_params(axis="both", length=0)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def sofr_basis(nq, root, path, theme="light"):
    """The SOFR - EFFR basis implied by `root` against the ZQ path, and what then printed."""
    t = _style(theme)
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.axhline(0, color=t["axis"], lw=1)
    ax.plot(nq.index, nq["basis_bp"], color=t["series"][0], lw=1.5, label=f"Implied by {root} and ZQ")
    ax.plot(nq.index, nq["realized_bp"], color=t["series"][1], lw=2,
            label="Realized over the same days, ex post")
    ax.set_ylabel("bp")
    ax.legend(loc="upper left", fontsize=9, labelcolor=t["secondary"])
    last = nq["basis_bp"].dropna()
    ax.annotate(f"{last.iloc[-1]:+.1f}bp", xy=(last.index[-1], last.iloc[-1]), xytext=(6, 0),
                textcoords="offset points", va="center", fontsize=9, color=t["secondary"])
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="both", length=0)
    _title(ax, f"SOFR minus EFFR: what {root} implies against the ZQ path",
           f"First {root} contract wholly ahead of each session. A wrong ZQ path would show up as noise here.", t)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_all(sessions, meetings, effr, nq, root, out_dir, ccy):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for theme in THEMES:
        p = out_dir / f"implied_paths_{ccy}_{theme}.png"
        implied_paths(sessions, meetings, effr, p, theme)
        q = out_dir / f"sofr_basis_{ccy}_{theme}.png"
        sofr_basis(nq, root, q, theme)
        written += [p, q]
    return written
