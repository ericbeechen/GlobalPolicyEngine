"""The one-page note: what the framework does, the chart, what it says now, what would make it wrong.

Generated, never hand-edited: the prose is templated here and every number in
it comes from the build. Edit the wording in this file and rerun
`scripts/build_model.py`. The page is one US-letter matplotlib figure saved as
PDF, so it needs nothing beyond matplotlib. Content that does not fit on one
page raises instead of spilling.
"""

from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.font_manager import FontProperties
from policypath.report import charts
from policypath.report.figures import THEMES, _style

PAGE = (8.5, 11.0)   # inches
MARGIN = 0.6
ORDINAL = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth", 7: "seventh", 8: "eighth"}


def _day(d):
    return f"{d.day} {d:%B %Y}"


def _meeting(row):
    """A meeting by the month it is announced in, as the market names it."""
    return f"{row['announcement_date']:%B %Y}"


def framework():
    return ("Every session since 2010, the fed funds path over the next eight FOMC meetings is solved from ZQ "
            "futures settles, using only the fixings published by that day. Beside it, the Fed's own "
            "balanced-approach rule from its Monetary Policy Report gives the path policy would take if today's "
            "macro picture held, fed by a real-time nowcast rebuilt from ALFRED vintages and by the committee's "
            "own longer-run dot for r*. The gap between the two, meeting by meeting and scored against its own "
            "trailing two years, is the signal.")


def now(today, k):
    """What the framework says on the latest session, in specifics."""
    m = today["meetings"].set_index("k")
    r = today["rule"]
    last = m.loc[len(m)]
    move = (last["market"] - today["rate_now"]) * 100
    ugap = r["u_gap"]
    lines = [
        f"On {_day(today['session'])} ZQ priced {abs(move):.0f}bp of {'tightening' if move > 0 else 'easing'} "
        f"by the {_meeting(last)} meeting, to {last['market']:.2f}%. The rule, fed core PCE at {r['inflation']:.1f}%, "
        f"unemployment {abs(ugap):.1f}pp {'below' if ugap < 0 else 'above'} CBO's natural rate and an r* of "
        f"{r['rstar']:.1f}% from the {_day(r['sep_date'])} SEP, asks for {r['notional']:.1f}%; at its inertial "
        f"pace it reaches {last['model']:.2f}% by then."
    ]
    close = m["gap_bp"].abs() <= 5
    through = (close.cumprod() == 1)
    if through.iloc[0] and not through.all():
        n = int(through.sum())
        lines.append(f"The two paths agree to within 5bp through the {_meeting(m.loc[n])} meeting; by "
                     f"{_meeting(last)} the market is {abs(last['gap_bp']):.0f}bp "
                     f"{'below' if last['gap_bp'] < 0 else 'above'} the rule.")
    else:
        lines.append(f"The market is {m.loc[k, 'gap_bp']:+.0f}bp from the rule at the {ORDINAL[k]} meeting "
                     f"and {last['gap_bp']:+.0f}bp at the {ORDINAL[len(m)]}.")
    row = m.loc[k]
    usual = row["window_mean_bp"]
    trade = "receive" if row["z"] > 0 else "pay"
    lines.append(
        f"Against its own history that is unusual: over the past two years the market sat on average "
        f"{abs(usual):.0f}bp {'below' if usual < 0 else 'above'} the rule at the {ORDINAL[k]} meeting, so today's "
        f"{row['gap_bp']:+.0f}bp is a z of {row['z']:+.1f}, and the signal is to {trade} the {_meeting(row)} meeting.")
    return " ".join(lines)


def wrong(today, headline, signal, result, k):
    g = signal[(signal["k"] == k) & signal["gap_bp"].notna()].set_index("session")["gap_bp"]
    sample_mean = g.groupby(g.index.year).mean().mean()
    pnl = result["pnl"]
    lost = pnl[(pnl.index.year >= 2019) & (pnl.index.year <= 2020)].sum()
    return (f"The rule has sat above the market for most of the sample: at the {ORDINAL[k]} meeting the market "
            f"has averaged {abs(sample_mean):.0f}bp below it, year by year since {headline['first_model']:%Y}. The z "
            "trades departures from that discount, not the discount itself. If the discount is a term premium, or the "
            "market's reasonable view that the committee does not follow its own benchmark, then a closing gap "
            f"is the market catching up with {today['rule']['inflation']:.1f}% core inflation, not overreaching, "
            "and the trade is wrong. Hold-flat is the other exposure: the rule path assumes today's inflation and "
            "slack persist, so a market that rightly expects disinflation looks too dovish, and one that expects "
            "re-acceleration too hawkish. And the rule cannot see shocks: the crude trade lost "
            f"{abs(lost):.0f}bp in 2019-20, when the market priced cuts the rule never asked for, and was right.")


def limitations(headline, k):
    b = headline["backtest"]
    return ("Macro conditioning is hold-flat: inflation, the unemployment gap and r* stay at today's values over "
            "the horizon. The rule is imposed from the Monetary Policy Report, not estimated, and r* is the SEP "
            "median as FRED rounds it (to within 5bp). The backtest is crude: one rate (the "
            f"{ORDINAL[k]} meeting ahead), DV01 fixed, one session's lag, and no transaction costs yet, so its "
            f"Sharpe of {b['sharpe']:.2f} over {b['first']:%Y}-{b['last']:%Y} overstates what was achievable. US only. "
            "Scheduled FOMC dates are treated as known on every date.")


class _Column:
    """Places blocks top to bottom on a figure, measuring each, and refuses to run off the page."""

    def __init__(self, fig, t):
        self.fig, self.t = fig, t
        self.renderer = fig.canvas.get_renderer()
        self.left, self.width = MARGIN / PAGE[0], 1 - 2 * MARGIN / PAGE[0]
        self.y = 1 - MARGIN / PAGE[1]
        self.bottom = MARGIN / PAGE[1]

    def _wrap(self, text, size, weight):
        font = FontProperties(family=plt.rcParams["font.family"], size=size, weight=weight)
        limit = self.width * PAGE[0] * self.fig.dpi
        lines, line = [], ""
        for word in text.split():
            trial = f"{line} {word}".strip()
            w, _, _ = self.renderer.get_text_width_height_descent(trial, font, ismath=False)
            if w > limit and line:
                lines.append(line)
                line = word
            else:
                line = trial
        return "\n".join([*lines, line])

    def text(self, text, size=9, weight="normal", color=None, gap=0.08):
        t = self.fig.text(self.left, self.y, self._wrap(text, size, weight), fontsize=size, weight=weight,
                          color=color or self.t["ink"], va="top", ha="left", linespacing=1.35)
        h = t.get_window_extent(self.renderer).height / (self.fig.dpi * PAGE[1])
        self.y -= h + gap / PAGE[1]
        self._check()

    def axes(self, height, n=1, hspace=0.9):
        """`n` stacked axes filling `height` inches, with `hspace` inches between them for titles."""
        each = (height - hspace * (n - 1)) / n
        out = []
        for i in range(n):
            top = self.y - (0.42 if i == 0 else 0) / PAGE[1]     # room for the chart's own title
            out.append(self.fig.add_axes([self.left + 0.07, top - each / PAGE[1], self.width - 0.16,
                                          each / PAGE[1]]))
            self.y = top - (each + hspace) / PAGE[1]
        self.y += (hspace - 0.35) / PAGE[1]
        self._check()
        return out

    def _check(self):
        if self.y < self.bottom:
            raise ValueError("the one-pager runs past one page; cut words, not the margin")


def write(today, effr, signal, result, headline, out_dir, ccy, author=None, generated=None, preview=None,
          moments=(), labels=None):
    """reports/onepager_<ccy>_<session>.pdf, dated by the session it describes. Returns the path.

    `preview`, if given, is a PNG path the same page is also rendered to, for checking the layout.
    """
    t = _style("light")
    k = headline["k"]
    generated = pd.Timestamp(generated) if generated is not None else pd.Timestamp.today().normalize()
    fig = plt.figure(figsize=PAGE)
    col = _Column(fig, t)
    col.text("Is the market pricing the Fed's own rule?", size=16, weight="bold", gap=0.04)
    by = f"{author}  ·  " if author else ""
    col.text(f"US policy path as of {_day(today['session'])}  ·  {by}generated {generated:%Y-%m-%d}",
             size=9, color=t["secondary"], gap=0.16)
    col.text(framework(), size=9, gap=0.12)
    top, bottom = col.axes(4.25, n=2)
    charts.ship(today, effr, signal, k, theme="light", axes=(top, bottom),
                moments=[tuple(m) for m in moments], labels=labels)
    for ax in (top, bottom):
        ax.title.set_fontsize(10.5)
        for text in ax.texts:                        # the subtitle `_title` draws, and the direct labels
            text.set_fontsize(8 if text.get_position() == (0, 1.02) else 7.5)
        ax.tick_params(labelsize=8)
        ax.yaxis.label.set_fontsize(8)
        if ax.get_legend():
            for text in ax.get_legend().get_texts():
                text.set_fontsize(7.5)
    for head, body in [("What it says now", now(today, k)),
                       ("What would make this wrong", wrong(today, headline, signal, result, k)),
                       ("Limitations", limitations(headline, k))]:
        col.text(head, size=10, weight="bold", gap=0.03)
        col.text(body, size=8.5, gap=0.1)
    out_dir = Path(out_dir)
    path = out_dir / f"onepager_{ccy}_{today['session']:%Y-%m-%d}.pdf"
    fig.savefig(path)
    if preview is not None:
        fig.savefig(preview, dpi=110)
    plt.close(fig)
    return path
