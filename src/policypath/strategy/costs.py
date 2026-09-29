"""What trading a sleeve costs: a one-way cost per unit of DV01 on each leg, from config, and what every close trades.

**A leg's cost** (`one_way`), in bp of yield per unit of |change in DV01|
(book currency per bp traded), from the currency's ``costs.<expression>``:

- a listed future, round trip given in ticks: ``ticks_round_trip x tick.other
  / 0.01 / 2 + fee_per_side / DV01 a contract``, every number from
  ``contracts``, none typed. ZQ: a round trip of two back-month ticks (each
  0.005 of price, 0.5bp) is 0.5bp one way, and $1 a side over $41.67 a
  contract is 0.024bp: 0.524bp one way, 1.048bp a round trip. The back-month tick because the
  fourth meeting's month is never the front one. The tick is the exchange's
  minimum increment, not a measured bid-offer: the archive holds settles, not
  quotes, so a quote-based estimate is a data purchase, and the leg is
  ``observed`` in the sense that its price is traded, not its spread.
- anything else, round trip given in bp: ``round_trip_bp / 2``. The GBP legs
  are valued off the Bank's fitted curves, so their cost is assumed, not
  observed (``observed: false``): the asymmetry the report states.

**What a close trades** (`traded`). At session t's close each leg goes from
q_p, held into t in the instrument selected at p, to q_t in the instrument
selected at t (book DV01, `expression.run`'s per-leg frame):

- the same instrument: ``|q_t - q_p|``;
- an instrument change (a ZQ month, a forward's window: the fourth meeting
  moved on): ``|q_p| + |q_t|``, close one and open the other;
- a par leg re-struck: the P&L marks a constant-maturity par bond re-struck at
  every close, which is not a thing a desk holds. A desk holds a bond and rolls
  it, and that roll is what is charged: at the first session on or after each
  ``roll_every_months`` (3) anniversary of the leg's entry while it is held,
  ``|q_p| + |q_t|`` (2|q| at an unchanged size). Entry is where the leg's side
  goes from 0 or crosses it; nothing is charged inside the first interval.

Each close's trade is split by cause: ``signal`` (the rule's s changed),
``rescale`` (s unchanged, the vol multiplier re-set), and on instrument
changes and re-strikes the part beyond what a same-instrument trade would
have been, 2 min(|q_p|, |q_t|) on one side, as ``roll`` and ``restrike``
(maintenance). A roll that is also an entry is an entry. A roll is charged as
two outright one-ways, not as a calendar spread, which trades tighter: the
conservative end.

Not charged: the daily re-sizing that keeps a leg's book DV01 constant as its
duration drifts and, for a GBP leg, as sterling moves. A desk lets DV01 drift
inside a band rather than trade it.

**Rates**: `configured`, each leg its own. The sensitivity's two modes,
``uniform`` (every leg c/2 one way: c is the whole round trip, the ZQ fee
included in it, not added) and ``assumed`` (the ``observed: false`` legs at
c/2, the observed ones at their own), are `report/costs.py`'s `curve`: costs
are linear in c, so it needs only the DV01 traded, never a rate per leg.

**The breakeven** (`breakeven`): costs are linear in c, so the net mean, and so
the net Sharpe, is zero exactly at ``c* = 2 x gross P&L / DV01 traded`` (both
a year, or both over one sample: the years cancel), maintenance included.
None where the gross P&L is 0 or less: no cost makes it work.

Every assumption here is a row in notes/DECISIONS.md, K1 to K4.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from policypath import config
from policypath.strategy import instruments

BP_PRICE = 0.01          # one bp of rate in the price of a 100 - rate future
CAUSES = ("signal", "rescale", "roll", "restrike")
MAINTENANCE = ("roll", "restrike")


@dataclass(frozen=True)
class LegCost:
    """One leg's cost: its label in the sleeve, one-way bp, whether observed, and its re-strike interval (par legs)."""
    leg: str
    ccy: str
    expression: str
    one_way_bp: float
    observed: bool
    restrike_months: int | None

    @property
    def round_trip_bp(self):
        return 2.0 * self.one_way_bp


def one_way(cfg, expression):
    """One-way cost of `expression`'s legs in currency block `cfg`, bp per unit of DV01 traded (module docstring)."""
    c = cfg["costs"][expression]
    if "ticks_round_trip" in c:
        contract = cfg["contracts"][cfg["expression"][expression]["contract"]]
        spread = c["ticks_round_trip"] * contract["tick"]["other"] / BP_PRICE / 2.0
        return spread + contract["fee_per_side"] / instruments.contract_dv01(contract)
    return c["round_trip_bp"] / 2.0


def leg_costs(sleeve):
    """Every leg's `LegCost`, in the sleeve's leg order, from each leg's currency block."""
    expression = config.SLEEVE_EXPRESSION[sleeve.kind]
    out = []
    for label, _, leg in sleeve.legs:
        cfg = config.currency(leg.ccy)
        c = cfg["costs"][expression]
        out.append(LegCost(label, leg.ccy, expression, one_way(cfg, expression), c["observed"],
                           c["roll_every_months"] if leg.funded else None))
    return out


def round_trip(spec):
    """A round trip of every leg of a book sleeve (``config/strategy.yml: sleeves`` entry), bp per unit, from config.

    An outright has one leg in its currency, a curve sleeve one per tenor of
    its ``expression.curve.legs``, a cross one in each currency of its pair (as
    `strategy/expression.py` builds them). What entering and leaving the whole
    sleeve costs, comparable with its edge.
    """
    e = config.SLEEVE_EXPRESSION[spec["kind"]]
    if spec["kind"] == "cross":
        ccys = list(spec["pair"])
    elif spec["kind"] == "curve":
        ccys = [spec["ccy"]] * len(config.currency(spec["ccy"])["expression"][e]["legs"])
    else:
        ccys = [spec["ccy"]]
    return sum(2.0 * one_way(config.currency(c), e) for c in ccys)


def configured(costs):
    """{leg: one-way bp}: every leg at its own cost."""
    return {c.leg: c.one_way_bp for c in costs}


def restrikes(days, q, months):
    """True on each session a par leg is re-struck: the first on or after each `months` anniversary of its entry.

    `q` is the leg's position put on at each close. A run is a stretch of one
    non-zero side; its entry is its first session, and its anniversaries fall
    inside it (the session that ends it is a new entry or flat, not a re-strike).
    """
    side = np.sign(np.nan_to_num(np.asarray(q, dtype=float)))
    out = np.zeros(len(side), dtype=bool)
    starts = np.flatnonzero(side != np.r_[0.0, side[:-1]])
    ends = np.r_[starts[1:], len(side)]
    for a, b in zip(starts, ends):
        if side[a] == 0:
            continue
        k = 1
        while (when := days[a] + pd.DateOffset(months=k * months)) <= days[b - 1]:
            out[days.searchsorted(when)] = True
            k += 1
    return out


def traded(sleeve, legs, s, costs=None):
    """Book DV01 traded at each close by cause and leg: session x (cause, leg), `CAUSES` (module docstring).

    `legs` is `expression.run`'s per-leg frame for the decided positions, `s`
    the rule's s by session (before sizing: its change is what makes a trade a
    ``signal`` one), `costs` the legs' `LegCost` (their re-strike intervals;
    `leg_costs` if None).
    """
    days = sleeve.sessions
    costs = leg_costs(sleeve) if costs is None else costs
    executed = s.reindex(days).shift(sleeve.lag).fillna(0.0).to_numpy()
    moved_s = executed != np.r_[0.0, executed[:-1]]
    out = {}
    for c in costs:
        one = legs[legs["leg"] == c.leg].set_index("session").reindex(days)
        q = one["q"].fillna(0.0).to_numpy()
        qp = np.r_[0.0, q[:-1]]
        base = np.abs(q - qp)
        moved = (one["selected"].notna() & one["held_id"].notna() & (one["selected"] != one["held_id"])).to_numpy()
        struck = restrikes(days, q, c.restrike_months) if c.restrike_months else np.zeros(len(days), dtype=bool)
        extra = np.abs(qp) + np.abs(q) - base
        out["signal", c.leg] = np.where(moved_s, base, 0.0)
        out["rescale", c.leg] = np.where(moved_s, 0.0, base)
        out["roll", c.leg] = np.where(moved, extra, 0.0)
        out["restrike", c.leg] = np.where(struck & ~moved, extra, 0.0)
    frame = pd.DataFrame(out, index=days)
    frame.columns = pd.MultiIndex.from_tuples(frame.columns, names=["cause", "leg"])
    return frame


def by_leg(trades, causes=CAUSES):
    """DV01 traded by leg (session x leg), summed over `causes`."""
    return trades.loc[:, list(causes)].T.groupby(level="leg", sort=False).sum().T


def charge(trades, rates):
    """Cost at each close, book currency (bp x DV01): each leg's DV01 traded x its one-way `rates` ({leg: bp})."""
    legs = by_leg(trades)
    return (legs * pd.Series(rates).reindex(legs.columns)).sum(axis=1).rename("cost")


def breakeven(gross, dv01_traded):
    """c*, bp: the round trip at which the net P&L is 0, 2 x gross / DV01 traded (one sample); None if gross <= 0."""
    if not gross > 0:
        return None
    return float(2.0 * gross / dv01_traded) if dv01_traded > 0 else float("inf")
