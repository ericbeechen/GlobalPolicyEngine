"""The week 4 figures: the market path against the model path, the gap underneath, the crude equity curve.

Generated, never hand-edited. Written in light and dark like the README figures,
with the same themes (`figures.THEMES`): slot 1 is the market, slot 2 the model.
The drawing functions take their axes, so the one-pager embeds the same chart.
"""

from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from policypath.report.figures import THEMES, _style, _title

# The words a chart uses for a currency (config ``report.labels``); these are USD's.
LABELS = {"market": "ZQ", "rate": "EFFR", "rule": "Fed's own rule", "policy": "fed funds",
          "meetings": "FOMC", "bank": "the Fed"}



def _dates(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="both", length=0)


def paths_now(ax, today, effr, t, lead_days=150, tail_days=40, labels=None):
    """Market and model step paths as of one session, after the realized overnight rate."""
    w = {**LABELS, **(labels or {})}
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
    for label, series, color in [(f"Market ({w['market']})", market, t["series"][0]),
                                 (w["rule"], model, t["series"][1])]:
        ax.annotate(label, xy=(end, series[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=t["secondary"])
    for k in (4, 8):
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
    ax.legend(handles=[Line2D([], [], color=t["ink"], lw=2, label=f"Realized {w['rate']}"),
                       Line2D([], [], color=t["series"][0], lw=2, label=f"Market: {w['market']}-implied path"),
                       Line2D([], [], color=t["series"][1], lw=2,
                              label="Model: balanced-approach rule, inertial, macro held flat")],
              loc="upper left", fontsize=9, labelcolor=t["secondary"])


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


def ship(today, effr, signal, k, path=None, theme="light", axes=None, moments=(), labels=None):
    """The one figure: paths as of today on top, the gap's history underneath."""
    t = _style(theme)
    if axes is None:
        fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 8.2),
                                          gridspec_kw={"height_ratios": [1.35, 1], "hspace": 0.42})
    else:
        fig, (top, bottom) = axes[0].figure, axes
    w = {**LABELS, **(labels or {})}
    paths_now(top, today, effr, t, labels=w)
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
    ax.text(0.01, 0.98, "No transaction costs: these overstate what was achievable.\nCosts arrive in week 8.",
            transform=ax.transAxes, ha="left", va="top", fontsize=9, color=t["ink"], fontweight="bold")
    ax.set_ylabel("bp x z, cumulative")
    _dates(ax)
    _title(ax, f"Crude backtest: trade the implied rate at the {k}th meeting ahead on the gap z",
           f"Receive when z > 0, pay when z < 0, size = z, DV01 fixed, one session's lag. "
           f"{summary['first']:%b %Y}-{summary['last']:%b %Y}: Sharpe {summary['sharpe']:.2f}, "
           f"max drawdown {summary['max_drawdown']:.0f}bp.", t)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_all(today, effr, signal, result, summary, k, out_dir, ccy, moments=(), labels=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for theme in THEMES:
        p = out_dir / f"model_vs_market_{ccy}_{theme}.png"
        ship(today, effr, signal, k, p, theme, moments=moments, labels=labels)
        q = out_dir / f"backtest_{ccy}_{theme}.png"
        equity(result, summary, k, q, theme)
        written += [p, q]
    return written
