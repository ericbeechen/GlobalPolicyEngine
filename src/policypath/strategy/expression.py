"""Sleeves: a signal component, the side it says, and the legs that trade it, with P&L by component and leg.

A sleeve (``config/strategy.yml: sleeves``) is one of three expressions:

- **outright**: the level component (the gap at the fourth meeting) traded in
  one leg, the currency's ``expression.outright`` instrument at that meeting
  (a ZQ month, a meeting-dated OIS forward). ``q = s``.
- **curve**: the slope component (`signal.components.slope`) traded as a
  DV01-neutral 2s10s: receive ``legs[0]``, pay ``legs[1]``, ``|q| = s`` each.
- **cross**: the differential (first currency's gap minus the second's) traded
  as receive the first currency's ``tenor``-year par leg, pay the second's,
  ``|q| = s`` in book DV01 each (matched at spot), on the sessions both have.
  Government 2y against 2y keeps it apart from the outright instruments: a
  cross built from the outright legs would make the sleeves' covariance singular.

Positions are book-currency DV01 per bp, positive = receive the sleeve's first
leg. For every sleeve z > 0 means the market prices more tightening than its
reference (the rule, or the other currency), so the priced rate is too high:
receive. The rule here is `linear`, s = z (week 6's sizing); hysteresis, the
ELB state and costs come from week 8's modules, which hand `run` any series of
decided positions and read its per-leg frame (executed DV01, the instrument
it is in, its native units).

Timing (`config/currencies.yml: backtest.execution_lag`): a position decided at
session t's close executes ``lag`` sessions later, at that session's close, in
the instrument selected there. The P&L credited to t is earned by the
position held from the previous session p, in the instrument selected at p:
decided ``lag + 1`` sessions before t, as `backtest.engine.run` has it. A cross
sleeve executes at the slower of its two currencies' lags.

For the curve and cross sleeves the component's bp is a proxy for the
instruments' move (the slope gap for a 2s10s, the fourth-meeting differential
for a 2y spread): the edge of those sleeves is that proxy. An outright's edge
is its instrument's own: for a ZQ month that straddles the next meeting (44% of
the 3,921 USD sessions), the day-weighted de-meaned gap of the regimes it
averages.
"""

from dataclasses import dataclass, replace
import numpy as np
import pandas as pd
from policypath import config, panel
from policypath.signal import components
from policypath.sources import cache
from policypath.strategy import carry, instruments


@dataclass
class Sleeve:
    """One sleeve, built: its component by session, its legs as (label, sign, `instruments.Leg`), its lag.

    `dev_bp` is the de-meaned gap its instrument expects to close, by session
    (the component's own, or an outright's regime-weighted one).
    """
    name: str
    kind: str
    ccys: tuple
    component: pd.DataFrame
    legs: list
    lag: int
    dev_bp: pd.Series

    @property
    def sessions(self):
        return self.component.index


class World:
    """What the sleeves read, loaded once: the book, each currency's config, panels, marks and meeting calendar.

    `panels` ({ccy: {name: frame}}) and `calendars` ({ccy: frame}) replace what
    would be read from ``data/panel/`` and ``config/meetings/`` (the tests' synthetic ones).
    """

    def __init__(self, book=None, root=cache.CACHE_DIR, panel_root=panel.PANEL_DIR, panels=None, calendars=None):
        self.book = config.strategy() if book is None else book
        self.root, self.panel_root = root, panel_root
        self._panels = {c: dict(p) for c, p in (panels or {}).items()}
        self._calendars = dict(calendars or {})
        self._marks = {}

    def cfg(self, ccy):
        return config.currency(ccy)

    def panel(self, ccy, name):
        tables = self._panels.setdefault(ccy, {})
        if name not in tables:
            tables[name] = panel.load(ccy, name, self.panel_root)
        return tables[name]

    def marks(self, ccy):
        if ccy not in self._marks:
            self._marks[ccy] = instruments.Marks(ccy, self.cfg(ccy), self.root)
        return self._marks[ccy]

    def calendar(self, ccy):
        if ccy not in self._calendars:
            self._calendars[ccy] = config.meetings(ccy)
        return self._calendars[ccy]

    def leg(self, ccy, expression, days, **kw):
        """`instruments.build` for one of `ccy`'s expressions on sleeve sessions `days`."""
        return instruments.build(self.cfg(ccy), expression, self.marks(ccy), days,
                                 self.book["carry"]["horizon_days"], self.panel(ccy, "sessions"),
                                 self.panel(ccy, "meetings"), self.calendar(ccy), **kw)


def dev_by_k(signal):
    """Session x k de-meaned gaps, gap_bp - window_mean_bp, from the signal panel."""
    wide = signal.assign(dev=signal["gap_bp"] - signal["window_mean_bp"])
    return wide.pivot(index="session", columns="k", values="dev").sort_index()


def _outright(spec, world):
    ccy = spec["ccy"]
    k = world.cfg(ccy)["backtest"]["horizon"]
    signal = world.panel(ccy, "signal")
    component = components.level(signal, k)
    leg = world.leg(ccy, config.SLEEVE_EXPRESSION["outright"], component.index, k=k)
    w, dev = leg.regimes, dev_by_k(signal).reindex(component.index)
    weighted = (w * dev.where(w > 0, 0.0)).sum(axis=1, skipna=False, min_count=1)
    return component, [(f"{ccy} {spec['kind']}", 1, leg)], weighted


def _curve(spec, world):
    ccy, book = spec["ccy"], world.book
    cfg = world.cfg(ccy)
    component = components.slope(world.panel(ccy, "signal"), book["signal"]["slope"], cfg["backtest"]["horizon"],
                                 cfg["signal"], book["signal"]["slope_orthogonal"])
    expression = config.SLEEVE_EXPRESSION["curve"]
    legs = [(f"{ccy} {tenor:g}y", sign, world.leg(ccy, expression, component.index, tenor=tenor))
            for tenor, sign in zip(cfg["expression"][expression]["legs"], (1, -1))]
    return component, legs, component["dev_bp"]


def _cross(spec, world):
    first, second = spec["pair"]
    a, b = world.cfg(first), world.cfg(second)
    component = components.differential(world.panel(first, "signal"), world.panel(second, "signal"),
                                        a["backtest"]["horizon"], b["backtest"]["horizon"], a["signal"])
    expression = config.SLEEVE_EXPRESSION["cross"]
    legs = [(f"{ccy} {spec['tenor']:g}y", sign, world.leg(ccy, expression, component.index, tenor=spec["tenor"]))
            for ccy, sign in ((first, 1), (second, -1))]
    return component, legs, component["dev_bp"]


KINDS = {"outright": _outright, "curve": _curve, "cross": _cross}


def sleeve(spec, world):
    """One ``sleeves`` entry of the book, built from `world`."""
    ccys = tuple(spec["pair"]) if spec["kind"] == "cross" else (spec["ccy"],)
    component, legs, dev = KINDS[spec["kind"]](spec, world)
    lag = max(world.cfg(c)["backtest"]["execution_lag"] for c in ccys)
    return Sleeve(spec["name"], spec["kind"], ccys, component, legs, lag, dev.rename("dev_bp"))


def build(world=None):
    """Every sleeve of the book (``config/strategy.yml``), in order: {name: `Sleeve`}."""
    world = World() if world is None else world
    return {s["name"]: sleeve(s, world) for s in world.book["sleeves"]}


def linear(sleeve):
    """The linear rule: decide s = z at each session's close (week 6's sizing, one unit of DV01 per unit of z)."""
    return sleeve.component["z"].rename("decided")


def unit(sleeve):
    """Per session and leg, the P&L of a unit receive of the sleeve (+1 book DV01 on its first leg), bp.

    carry, roll, rate and total (`carry.attribution`, signed for the leg), and
    the checks: ``identity_bp`` (check 1), ``full_bp`` and ``bound_bp`` (check 2).
    Long frame: one row per session and leg.
    """
    frames = []
    for label, sign, leg in sleeve.legs:
        parts = carry.attribution(leg)
        full, bound = carry.revaluation(leg)
        f = leg.frame
        frames.append((parts * sign).assign(
            leg=label, sign=sign, held_id=f["held_id"], delta=f["delta"], fx=f["fx"], stale=f["stale"],
            fx_stale=f["fx_stale"], identity_bp=carry.identity(leg, parts) * sign, full_bp=full * sign,
            bound_bp=bound).rename_axis("session").reset_index())
    return pd.concat(frames, ignore_index=True)


def run(sleeve, decided, u=None):
    """P&L of the positions `decided` at each session's close, by session and leg. Returns (legs, sessions).

    `decided` is s per session (book DV01 on the first leg: `linear`, or week
    8's rules). ``legs`` has, per session and leg, the position ``held`` into
    the session (in ``held_id``) and its P&L by component (``carry``,
    ``roll``, ``rate``, ``pnl``), the unit P&L (``unit_*``), and what a trade at
    the session's close leaves on: ``q`` (book DV01), the instrument
    ``selected``, and ``native`` (contracts, or notional in the leg's
    currency). ``sessions`` sums the legs. `u` is `unit(sleeve)`, where the
    caller runs many rules on one sleeve.
    """
    days = sleeve.sessions
    s = decided.reindex(days)
    executed = s.shift(sleeve.lag)
    held = executed.shift(1)
    u = unit(sleeve) if u is None else u
    rows = []
    for label, sign, leg in sleeve.legs:
        one = u[u["leg"] == label].set_index("session")
        q = executed * sign
        out = pd.DataFrame({"leg": label, "held": held * sign, "q": q, "selected": leg.frame["id"],
                            "native": instruments.native(leg, q)}, index=days)
        for part in [*carry.PARTS, "total"]:
            out[f"unit_{part}"] = one[part]
            out["pnl" if part == "total" else part] = (held * one[part]).fillna(0.0)
        rows.append(out.assign(held_id=one["held_id"], stale=one["stale"]).rename_axis("session").reset_index())
    legs = pd.concat(rows, ignore_index=True)
    sums = legs.groupby("session")[[*carry.PARTS, "pnl"]].sum()
    sessions = pd.DataFrame({"decided": s, "position": held}, index=days).join(sums)
    return legs, sessions.assign(equity=sessions["pnl"].cumsum()).rename_axis("session")


def checks(sleeve):
    """Checks 1 and 2 on every session, both directions: one row per leg and direction.

    sessions checked, max |identity residual|, max |full - total|, the bound
    there and the largest residual/bound ratio, sessions outside the bound
    (+1e-9bp), and the stale marks. A pay (-1) is the receive negated, so both
    rows have the same residual sizes: they are there because the checks are
    asked of both.
    """
    u = unit(sleeve)
    out = []
    for direction in (1, -1):
        for label, g in u.groupby("leg", sort=False):
            live = g.dropna(subset=["total"])
            res = (live["full_bp"] - live["total"]) * direction
            ratio = res.abs() / (live["bound_bp"] + carry.TOL_BP)
            out.append({"sleeve": sleeve.name, "leg": label, "direction": direction, "sessions": len(live),
                        "identity_max_bp": (live["identity_bp"] * direction).abs().max(),
                        "full_residual_max_bp": res.abs().max(),
                        "bound_at_max_bp": live["bound_bp"].loc[res.abs().idxmax()] if len(live) else np.nan,
                        "ratio_max": ratio.max(), "outside_bound": int((ratio > 1).sum()),
                        "stale": int(g["stale"].sum()), "fx_stale": int(g["fx_stale"].sum())})
    return pd.DataFrame(out)


def ahead(sleeve, h, min_pairs, side=None):
    """Per session, what a trade at its close expects over `h` days (`strategy/carry.py`, the breakeven).

    z, side (sign of z unless given), the raw and de-meaned gap, edge, carry
    and roll over h of the instruments selected (unit receive, then signed for
    the side), CR_h, phi_h, phi*, E_h, pays, bleeding, and the instruments.
    """
    comp = sleeve.component
    side = np.sign(comp["z"]) if side is None else side.reindex(comp.index)
    carry_h = pd.Series(0.0, index=comp.index)
    roll_h = pd.Series(0.0, index=comp.index)
    for _, sign, leg in sleeve.legs:
        c, r = carry.carry_roll_ahead(leg, h)
        carry_h, roll_h = carry_h + sign * c, roll_h + sign * r
    cr = side * (carry_h + roll_h)
    e = carry.edge(side, sleeve.dev_bp)
    phi = carry.closure(comp["value_bp"], comp["dev_bp"], h, min_pairs)
    ids = pd.concat([leg.frame["id"] for _, _, leg in sleeve.legs], axis=1)
    selected = ids.apply(lambda row: " vs ".join(row) if row.notna().all() else None, axis=1)
    out = pd.DataFrame({"z": comp["z"], "side": side, "value_bp": comp["value_bp"], "dev_bp": sleeve.dev_bp,
                        "edge_bp": e, "carry_h_bp": side * carry_h, "roll_h_bp": side * roll_h, "cr_h_bp": cr,
                        "phi_h": phi, "selected": selected}, index=comp.index)
    return out.join(carry.bleed(e, cr, phi)).rename_axis("session")


def backtest_residual(sleeve, backtest):
    """The end-to-end check: this outright sleeve, linear rule, fx forced to 1, against a week 6 backtest panel.

    `backtest` is ``data/panel/<ccy>_backtest.parquet`` (session, pnl). Returns
    (sessions compared, max |difference| in bp), on every session from the
    sleeve's first z.
    """
    ones = {"fx": 1.0, "spot": 1.0, "spot_p": 1.0}
    local = replace(sleeve, legs=[(label, sign, replace(leg, frame=leg.frame.assign(**ones)))
                                  for label, sign, leg in sleeve.legs])
    _, sessions = run(local, linear(local))
    first = sleeve.component["z"].first_valid_index()
    want = backtest.set_index("session")["pnl"]
    got = sessions.loc[sessions.index >= first, "pnl"]
    return len(got), float((got - want.reindex(got.index)).abs().max(skipna=False))
