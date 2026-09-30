"""The robustness grid: the headline book with one choice at a time moved off the chosen specification. Week 9.

The plan's question is not whether every cell works but whether the result
hinges on one. So each row (``config/strategy.yml: robustness.choices``) moves
one choice (r*, the coefficients, the z window, the curve fit, the
conditioning, the slope, the carry filter) and leaves every other where the
book has it: one at a time, so interactions are not run. `report/robustness.py`
runs the rows and reports them; this module is what a row is.

**A row changes the signal and nothing else.** Its override reaches only
``rule``, ``signal`` and ``market.curve.method`` of a currency block, and
``signal`` and ``positions.carry_filter`` of the book (the config refuses
anything wider, `config.ROBUSTNESS_CURRENCY_PATHS`). The legs, and so the
marks, the instruments selected, carry, roll and costs, are the baseline's:
a row's sleeves are built through `View`, which hands back the legs the
baseline built and refuses to build another. So a row reuses the baseline's
unit P&L, sigma and covariance exactly, and the only thing that moves between
two cells is what the signal decided. Under NSS the GBP signal reads a
re-fitted market path while its P&L stays marked on the Bank's own curve.

**Rebuilding a row's signal** (`variant`, `rebuild`) goes through the chain's
own entry points from the baseline's ``data/panel/`` sessions, meetings and
macro, rerunning only the stages the override reaches: the market path
(`panel.build`, from the cache, on the baseline's sessions) under ``market``;
the model path (`model.path.build`) under ``market`` or ``rule``; the gap and
its z (`signal.gap.build`) always. A stage not rerun is the baseline's own
panel. `regress.merge` merges the override, so an r* block replaces
``rule.rstar`` whole, and the merged block must pass `config.validate`.
`rebuilt_equals` is the check that the same path, with no override,
reproduces ``data/panel/``'s model and signal bit for bit: what a row moves is
then its override's doing. A rebuilt row is kept in `Store` under a
fingerprint of everything it read (the merged block, the input panels, the
cache files of its currency, the chain's code), so the build reruns a row
only when one of them changed. Rebuilt cold, the 17 row-currency signals take
about 95s (an r* or conditioning row 5s a currency, an estimated one 10s,
the NSS row 16s with its market path, a z window under 0.1s), and the check 5s
a currency; from the store the whole grid runs in 9s.

**The common sample** (`eligible`, `common`). A session has an eligible
signal where the ELB treatment counts it (`positions.samples`) and a z stands
behind the position held into it or the one put on at its close. The ELB
state is the model's, so it moves with r*, the coefficients and the
conditioning: a row can count sessions the baseline does not, and the other
way round. The primary cell is over the sessions every row counts, so every
cell is over the same days and differs only by what the signal did on them.
Each row on its own sample is reported beside it. Under converge-to-target the
book counts 3,498 sessions of its own against the chosen specification's
2,970 (its dollar ELB state is 397 sessions, not 977); the common sample is
2,866.

**The rows are one at a time.** A choice that is one currency's by nature (r*,
the curve fit) moves that currency alone; a method (the coefficients, the z
window, the conditioning) moves every currency the book trades at once
(``every:``), as the chosen specification applies it to both.
"""

from dataclasses import dataclass
import hashlib
import inspect
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from policypath import config, panel, regress
from policypath.backtest import metrics
from policypath.model import path
from policypath.signal import components, gap
from policypath.sources import cache
from policypath.strategy import positions

STORE_DIR = panel.PANEL_DIR / "robustness"
INPUTS = ("sessions", "meetings", "macro", "model", "paths", "signal")   # the baseline panels a row reads and checks
REPLACED = ("model", "signal")                                   # the panels a row replaces
PACKAGE = Path(__file__).resolve().parents[1]
NOT_CHAIN = ("strategy", "report", "backtest")                   # code a row's signal does not run through
PAIRED_LAGS = 21                                                 # sessions, a month: the Newey-West paired SE's lags


# ---- the rows ----------------------------------------------------------------------

@dataclass(frozen=True)
class Row:
    """One row of the grid: its key, the choice it moves and the value it takes there; its overrides, {ccy: {...}}
    of the currency blocks and {...} of the book. The baseline row overrides nothing."""
    key: str
    choice: str
    label: str
    currencies: dict
    book: dict

    def words(self):
        return f"{self.choice}: {self.label}" if self.choice else self.label


BASELINE = Row("baseline", "", "the chosen specification", {}, {})


def book_currencies(book):
    """The currencies the book's sleeves trade, in sleeve order."""
    return list(dict.fromkeys(c for s in book["sleeves"] for c in config._sleeve_ccys(s)))


def rows(book):
    """Every row of ``robustness.choices``, the baseline first: `Row`s."""
    ccys = book_currencies(book)
    return [BASELINE] + [Row(r["key"], c["name"], r["label"], config.row_overrides(r, ccys), r.get("book", {}))
                         for c in book["robustness"]["choices"] for r in c["rows"]]


def block(ccy, cfg, over):
    """Currency block `cfg` with a row's override `over`, merged as `regress.merge` does and validated."""
    merged = regress.merge(cfg, over)
    problems = config.validate(ccy, merged)
    if problems:
        raise config.ConfigError(f"{ccy} with {over} fails its schema:\n  " + "\n  ".join(problems))
    return merged


def book_of(row, book):
    """The book config under `row`: its book override merged in."""
    return regress.merge(book, row.book)


# ---- a row's signal ----------------------------------------------------------------

def rebuild(ccy, cfg, panels, root=cache.CACHE_DIR, market=False, model=True):
    """(model, signal) under currency block `cfg`, from `panels` (the baseline's `INPUTS`, by name).

    With `market` the market path is re-solved from the cache under `cfg` on
    the baseline's sessions; with `market` or `model` the model path is rerun
    on it; the gap and its z always. Otherwise a stage is the baseline's.
    """
    if not (market or model):
        return panels["model"], gap.build(panels["paths"], cfg["signal"])
    sessions, meetings = panels["sessions"], panels["meetings"]
    if market:
        days = pd.DatetimeIndex(sessions["session"])
        sessions, meetings = panel.build(ccy, start=days.min(), end=days.max(), root=root, cfg=cfg)
    i = path.inputs(ccy, root, rule=cfg["rule"])
    summaries, paths = path.build(sessions, meetings, panels["macro"], i["sep"], i["target"], i["fixings"],
                                  cfg["rule"])
    return summaries, gap.build(paths, cfg["signal"])


def variant(ccy, cfg, over, panels, root=cache.CACHE_DIR):
    """A row's (model, signal) in one currency: baseline block `cfg` with its override `over` (`block`, `rebuild`)."""
    return rebuild(ccy, block(ccy, cfg, over), panels, root, market="market" in over,
                   model="market" in over or "rule" in over)


def rebuilt_equals(ccy, cfg, panels, root=cache.CACHE_DIR):
    """None if `rebuild` with no override reproduces ``panels``' model and signal bit for bit, else what differs."""
    model, signal = rebuild(ccy, cfg, panels, root)
    diffs = {name: regress._describe(panels[name], regress._roundtrip(got))
             for name, got in (("model", model), ("signal", signal))}
    return {k: v for k, v in diffs.items() if v} or None


def fingerprint(ccy, cfg, panel_root=panel.PANEL_DIR, root=cache.CACHE_DIR):
    """A key for everything a row's rebuild in `ccy` under block `cfg` reads: the block, the input panels, the
    cache files of the currency (by size and time), and the code of the chain it runs."""
    h = hashlib.sha256(json.dumps([ccy, cfg], sort_keys=True, default=str).encode())
    for name in INPUTS:
        h.update((Path(panel_root) / f"{ccy}_{name}.parquet").read_bytes())
    for p in sorted(Path(root).glob(f"*/{ccy}/*.parquet")):
        s = p.stat()
        h.update(f"{p.relative_to(root)} {s.st_size} {s.st_mtime_ns}".encode())
    for p in sorted(PACKAGE.rglob("*.py")):
        if p.relative_to(PACKAGE).parts[0] not in NOT_CHAIN:
            h.update(p.read_bytes())
    h.update(inspect.getsource(rebuild).encode())
    return h.hexdigest()[:16]


def signals(world, grid, store=None, keys=None, log=None):
    """{row key: {ccy: (model, signal)}} for the rows of `grid` that change a currency block (`variant`).

    `world` gives the baseline blocks and panels. With a `store`, a row is
    read from it where its `fingerprint` is there and stored where it is not;
    before the first model rebuild in a currency, `rebuilt_equals` is checked
    once per fingerprint (raising if the rebuild does not reproduce the
    baseline). `keys` limits the rows built to those keys (the rest are
    left out of the result); `log(message)` is told what was built and how long it took.
    """
    out, checked = {}, {}
    for row in grid:
        if keys is not None and row.key not in keys:
            continue
        out[row.key] = {}
        for ccy, over in row.currencies.items():
            base, merged = world.cfg(ccy), block(ccy, world.cfg(ccy), over)
            panels = {name: world.panel(ccy, name) for name in INPUTS}
            if {"rule", "market"} & set(over) and ccy not in checked:
                check = fingerprint(ccy, base, world.panel_root, world.root) if store else None
                checked[ccy] = bool(store and store.marked(ccy, check)), check
            key = fingerprint(ccy, merged, world.panel_root, world.root) if store else None
            got = store.get(ccy, key) if store else None
            if got is None:
                if ccy in checked and not checked[ccy][0]:
                    t = time.perf_counter()
                    diffs = rebuilt_equals(ccy, base, panels, world.root)
                    if diffs:
                        raise RuntimeError(f"{ccy}: the chain does not rebuild data/panel's panels: {diffs}")
                    if store:
                        store.mark(ccy, checked[ccy][1])
                    checked[ccy] = True, checked[ccy][1]
                    if log:
                        log(f"{ccy}: the rebuild reproduces the baseline's model and signal "
                            f"({time.perf_counter() - t:.1f}s)")
                t = time.perf_counter()
                got = variant(ccy, base, over, panels, world.root)
                got = store.put(ccy, key, got) if store else got
                if log:
                    log(f"{row.key} {ccy}: rebuilt ({time.perf_counter() - t:.1f}s)")
            out[row.key][ccy] = got
    return out


class Store:
    """Rebuilt rows under `root`, one model and one signal parquet per currency and `fingerprint`; `used` is every
    (currency, key) asked for since it was opened, what `prune` keeps."""

    def __init__(self, root=STORE_DIR):
        self.root, self.used = Path(root), set()

    def _paths(self, ccy, key):
        self.used.add((ccy, key))
        return [self.root / f"{ccy}_{key}_{name}.parquet" for name in REPLACED]

    def get(self, ccy, key):
        """The stored (model, signal), or None."""
        paths = self._paths(ccy, key)
        return tuple(pd.read_parquet(p) for p in paths) if all(p.exists() for p in paths) else None

    def put(self, ccy, key, frames):
        """Store (model, signal) and return them as read back, so a stored row and a fresh one are the same bits."""
        self.root.mkdir(parents=True, exist_ok=True)
        for p, frame in zip(self._paths(ccy, key), frames):
            frame.to_parquet(p, index=False)
        return self.get(ccy, key)

    def mark(self, ccy, key):
        """Record that a check under `key` passed."""
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / f"{ccy}_{key}.ok").write_text("", encoding="utf-8")

    def marked(self, ccy, key):
        self.used.add((ccy, key))
        return (self.root / f"{ccy}_{key}.ok").exists()

    def prune(self):
        """Delete every stored file this store has not been asked for (`used`): rows and checks gone stale. Returns
        how many."""
        gone = 0
        for p in self.root.glob("*_*"):
            if tuple(p.stem.split("_")[:2]) not in self.used:
                p.unlink()
                gone += 1
        return gone


# ---- a row's sleeves ---------------------------------------------------------------

class View:
    """What `strategy.expression.sleeve` reads, for one row: its book, currency blocks, and model and signal panels
    ({ccy: {name: frame}}) over `world`, the baseline's; everything else is `world`'s.

    Legs come from `legs` ({(ccy, expression, options): [(sessions, Leg),
    ...]}: one leg can serve two sleeves on different sessions, as a 2y
    gilt does the GBP 2s10s and the cross), shared by every row's view. With
    `build` (the baseline's view) a leg is built on first use and kept;
    without it a leg the baseline did not build on those sessions raises: a
    row cannot change an instrument, a mark, carry, roll or a cost.
    """

    def __init__(self, world, legs, book=None, cfgs=None, panels=None, build=False):
        self.world, self.legs, self.build = world, legs, build
        self.book = world.book if book is None else book
        self._cfgs, self._panels = dict(cfgs or {}), {c: dict(p) for c, p in (panels or {}).items()}

    def cfg(self, ccy):
        return self._cfgs[ccy] if ccy in self._cfgs else self.world.cfg(ccy)

    def panel(self, ccy, name):
        own = self._panels.get(ccy, {})
        return own[name] if name in own else self.world.panel(ccy, name)

    def leg(self, ccy, expression, days, **kw):
        key, days = (ccy, expression, tuple(sorted(kw.items()))), pd.DatetimeIndex(days)
        built = self.legs.setdefault(key, [])
        for on, leg in built:
            if on.equals(days):
                return leg
        if not self.build:
            raise RuntimeError(f"a robustness row asked for a leg the baseline has not built on its sessions: {key}")
        built.append((days, self.world.leg(ccy, expression, days, **kw)))
        return built[-1][1]


def onto(signal, base):
    """A row's signal panel on the baseline's (session, k) rows: a row neither adds a session nor drops one (a
    session the row lacks has no z there). Returns (frame, sessions added, sessions dropped)."""
    keys = ["session", "k"]
    s, b = signal.set_index(keys), base.set_index(keys)
    days, have = set(b.index.get_level_values(0)), set(s.index.get_level_values(0))
    return s.reindex(b.index).reset_index(), len(have - days), len(days - have)


def touches(row, sleeve):
    """True if `row` can move `sleeve`'s signal: it changes one of its currencies, or the book's signal block."""
    return bool(set(row.currencies) & set(sleeve.ccys)) or "signal" in row.book


# ---- the samples and the cells -----------------------------------------------------

def eligible(z, state, treatment, lag):
    """A sleeve's sessions with an eligible signal: counted by the ELB `treatment` (`positions.samples`), with a z
    behind the position held into the session (decided lag + 1 sessions before) or the one put on at its close."""
    kept = positions.samples(state, treatment, lag)[0]
    known = z.notna()
    return kept & (known.shift(lag + 1, fill_value=False) | known.shift(lag, fill_value=False))


def common(masks):
    """The sessions every mask in `masks` (bool series on one index) has: their intersection."""
    masks = list(masks)
    out = masks[0].copy()
    for m in masks[1:]:
        if not m.index.equals(out.index):
            raise ValueError("the samples are on different sessions")
        out &= m
    return out


def cell(pnl, mask, periods):
    """(Sharpe, its SE, sessions) of daily P&L `pnl` over the sessions `mask` has (`backtest.metrics`)."""
    x = pnl[mask.reindex(pnl.index, fill_value=False)]
    sr = metrics.sharpe(x, periods)
    return sr, metrics.sharpe_se(sr, len(x) / periods) if pd.notna(sr) else np.nan, len(x)


def paired_se(x, x0, mask, periods, lags=None):
    """The SE of a cell's difference from the baseline's over one sample, to first order: sd(x - x0) / sd(x0) /
    sqrt(years), `x` and `x0` the two daily P&Ls on one index; with `lags`, the Newey-West version, the long-run sd
    of (x - x0) / sd(x0) with Bartlett weights to `lags` (`metrics.long_run_var`) in place of the sd.

    Two cells over the same sessions share most of their positions, so their
    difference is much less noisy than either Sharpe (SE ~ 1 / sqrt(years)).
    Without `lags` it is iid, as `metrics.sharpe_se`. For a Sharpe's own SE
    iid is the least the uncertainty can be; for this difference it is no bound
    either way: on the week 9 grid the Newey-West version (`PAIRED_LAGS`) is
    below the iid one on 10 of the 13 rows (the carry filter's 0.043 against
    0.053). 0 for the baseline against itself.
    """
    m = mask.reindex(x.index, fill_value=False)
    d, b = (x - x0)[m], x0[m]
    sd = b.std(ddof=1)
    if not (len(b) > 1 and sd > 0):
        return np.nan
    spread = d.std(ddof=1) if lags is None else np.sqrt(metrics.long_run_var(d.to_numpy(), lags))
    return float(spread / sd * np.sqrt(periods / len(b)))


def ic(z, x, h, lag, mask):
    """IC(h) of `z` against the next `h` sessions of `x` after the execution `lag` (`metrics.forward`, `metrics.ic`),
    over the sessions `mask` has: (IC, t, sessions)."""
    fwd = metrics.forward(x, h, lag)
    return metrics.ic(z.where(mask.reindex(z.index, fill_value=False)), fwd, h)


def ic_offsets(z, x, h, lag, mask):
    """`ic`'s non-overlapping cross-check (`metrics.ic_offsets`): (mean, lowest, highest, fewest sessions) over the
    h start offsets."""
    return metrics.ic_offsets(z.where(mask.reindex(z.index, fill_value=False)), metrics.forward(x, h, lag), h)


def ic_persistence(z, x, h, lag, mask):
    """The autocorrelation `h` sessions apart of the rank products `ic` averages (`metrics.rank_products`): what its
    Newey-West t, whose Bartlett weights stop at lag h, leaves out."""
    p = metrics.rank_products(z.where(mask.reindex(z.index, fill_value=False)), metrics.forward(x, h, lag))
    return float(p.autocorr(h)) if len(p) > h + 2 else np.nan


# ---- what a row does to the signal -------------------------------------------------

def moves(base, row, k, base_model, row_model, on=None, window=None):
    """How far a row moves one currency's level signal (the gap at horizon `k`) from the baseline's.

    From the signal panels `base` and `row` and their model panels: the mean
    and sd of the change in gap_bp (row minus baseline), the latest session's
    gap before and after, corr(z, baseline z) and the share of sessions whose
    z changes sign (over the sessions both have a z), r*'s offset (mean and
    sd, pp), the bp the gap moves per pp of r* where the rule's goal is on the
    floor in neither (the median; NaN where r* does not move), and the share
    of sessions with the goal floored in each and in exactly one.

    Over every session those shares count years in which both models sit in
    the ELB state and no side can differ. With `on` (bool by session: the
    sessions every row counts for the currency's outright) the z's sign
    changes and the floored shares are also over those (``*_traded``), with
    ``floor_free``: the sessions there on which neither model's goal is
    floored on the session or anywhere in the trailing z `window`, where a
    constant r* offset is a constant shift of every gap the z reads, and
    ``floor_free_max_dz``, the largest change in z on them.
    """
    b, v = components.level(base, k), components.level(row, k)
    both = b.index.intersection(v.index)
    b, v = b.loc[both], v.loc[both]
    d = v["value_bp"] - b["value_bp"]
    z = b["z"].notna() & v["z"].notna()
    m0 = base_model.set_index("session").reindex(both)
    m1 = row_model.set_index("session").reindex(both)
    dr = (m1["rstar"] - m0["rstar"]).astype(float)
    f0, f1 = m0["at_elb"].eq(True), m1["at_elb"].eq(True)
    free = ~f0 & ~f1 & (dr.abs() > 1e-9)
    last = both[-1]
    out = {"sessions": len(both), "z_sessions": int(z.sum()), "mean_change_bp": float(d.mean()),
           "sd_change_bp": float(d.std()), "latest": last, "latest_base_bp": float(b.loc[last, "value_bp"]),
           "latest_row_bp": float(v.loc[last, "value_bp"]), "latest_change_bp": float(d.loc[last]),
           "corr_z": float(b.loc[z, "z"].corr(v.loc[z, "z"])),
           "sign_changes": float((np.sign(b.loc[z, "z"]) != np.sign(v.loc[z, "z"])).mean()),
           "rstar_offset_mean": float(dr.mean()), "rstar_offset_sd": float(dr.std()),
           "bp_per_pp": float((d[free] / dr[free]).median()) if free.any() else np.nan,
           "floored_base": float(f0.mean()), "floored_row": float(f1.mean()), "floored_differ": float((f0 != f1).mean())}
    if on is None:
        return out
    t = on.reindex(both, fill_value=False).astype(bool) & z
    clear = (f0 | f1).astype(float).rolling(window, closed="both").max().eq(0.0) & t      # the z's window and t
    dz = (v.loc[clear, "z"] - b.loc[clear, "z"]).abs()
    return out | {"traded_sessions": int(t.sum()),
                  "sign_changes_traded": float((np.sign(b.loc[t, "z"]) != np.sign(v.loc[t, "z"])).mean()),
                  "floored_base_traded": float(f0[t].mean()), "floored_row_traded": float(f1[t].mean()),
                  "floor_free": int(clear.sum()), "floor_free_window": window,
                  "floor_free_max_dz": float(dz.max()) if len(dz) else np.nan}


def first_estimate(model):
    """The first session whose estimated coefficients rest on at least one quarter (`model/estimate.py`), or None."""
    if "coef_quarters" not in model:
        return None
    days = model.loc[model["coef_quarters"] >= 1, "session"]
    return pd.Timestamp(days.min()) if len(days) else None
