"""Read `config/`: the per-currency blocks, the book, and the meeting calendar.

Every enabled block in ``currencies.yml`` is validated against `SCHEMA` when the
file is first read, and a block that is missing a field, or has one of the
wrong shape, fails there with every problem listed by its dotted path. That is
what makes a new currency an hour of config rather than an afternoon of
KeyErrors: the first run says everything the block still needs. Every ``tags``
block, wherever it sits, is checked too: the brief and the limitations are
built from them.

``strategy.yml`` is the book: its currency, the sleeves, and what every sleeve
shares. `strategy` checks it against `STRATEGY_SCHEMA` and then against the
currency blocks (each sleeve's currencies are enabled and have the expression
the sleeve trades; a cross pair has one horizon; a leg outside the book
currency has an FX rate). That check lives in `strategy`, not in the currency
load, so the currency blocks still load without a book. Its ``robustness``
rows (week 9's grid) are checked the same way: each row's override, merged
into the block it changes, must pass that block's own schema, and may reach
only the signal (`ROBUSTNESS_CURRENCY_PATHS`, `ROBUSTNESS_BOOK_PATHS`).
"""

from functools import cache
from numbers import Number
from pathlib import Path
import re
import pandas as pd
import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
ROOT = CONFIG_DIR.parent


class ConfigError(ValueError):
    """A config file that does not meet its schema. The message lists every problem."""


# ---- the schema --------------------------------------------------------------------
# Dotted path -> what the value must be. A check is a type, a tuple of types, or a
# function returning an error message (or None). Paths under a key that is itself
# conditional (the extractor's, r*'s) are in the tables below it.

def _date(v):
    try:
        pd.Timestamp(v)
    except (TypeError, ValueError):
        return f"{v!r} is not a date"
    return None


def _schedule(kind):
    """A value, or a dated schedule of them: [{from: date, value: x}, ...], in date order."""
    def check(v):
        if isinstance(v, kind) and not isinstance(v, bool):
            return None
        if not isinstance(v, list) or not v:
            return f"{v!r} is neither a {kind.__name__} nor a dated schedule"
        for s in v:
            if not isinstance(s, dict) or set(s) != {"from", "value"}:
                return f"schedule entry {s!r} is not {{from: date, value: x}}"
            if _date(s["from"]) or not isinstance(s["value"], kind):
                return f"schedule entry {s!r} has a bad date or value"
        dates = [pd.Timestamp(s["from"]) for s in v]
        return None if dates == sorted(dates) else "schedule dates are not in order"
    return check


def _list_of(kind):
    name = kind.__name__ if isinstance(kind, type) else "number"
    def check(v):
        ok = isinstance(v, list) and all(isinstance(x, kind) and not isinstance(x, bool) for x in v)
        return None if ok else f"{v!r} is not a list of {name}"
    return check


def _one_of(*names):
    def check(v):
        return None if v in names else f"{v!r} is not one of {list(names)}"
    return check


def _number(v):
    return isinstance(v, NUMBER) and not isinstance(v, bool)


def _optional_number(v):
    return None if v is None or _number(v) else f"{v!r} is neither a number nor null"


def _series_ref(v):
    """{source, series}, where series is one name or a list of names."""
    if not isinstance(v, dict) or not {"source", "series"} <= set(v):
        return f"{v!r} is not {{source, series}}"
    s = v["series"]
    ok = isinstance(s, str) or (isinstance(s, list) and s and all(isinstance(x, str) for x in s))
    return None if ok else f"series {s!r} is not a name or a list of names"


def _policy_rate(v):
    """One {source, series} (a list of two series is a range, read at its midpoint), or a dated schedule of them."""
    if isinstance(v, list):
        for s in v:
            if not isinstance(s, dict) or "from" not in s or _date(s["from"]):
                return f"schedule entry {s!r} has no valid from date"
            err = _series_ref({k: x for k, x in s.items() if k != "from"})
            if err:
                return err
        return None
    return _series_ref(v)


def _natural_rate(v):
    ok = isinstance(v, str) or (isinstance(v, dict) and isinstance(v.get("constant"), Number))
    return None if ok else f"{v!r} is neither a series name nor {{constant: x}}"


def _lags(v):
    """{source: {series: whole days to publication}}: every series cached with a lag says when it is public.

    Business days for a daily series (``sources.daily``); calendar days after the
    end of the vintage's quarter for an estimate (``sources.estimates``).
    """
    if not isinstance(v, dict):
        return f"{v!r} is not {{source: {{series: lag}}}}"
    for source, series in v.items():
        if not isinstance(series, dict) or not series:
            return f"{source}: {series!r} is not {{series: lag}}"
        bad = [s for s, lag in series.items() if not isinstance(lag, int) or isinstance(lag, bool) or lag < 0]
        if bad:
            return f"{source}: no whole-day publication lag for {bad}"
    return None


def _named_lists(v):
    ok = isinstance(v, dict) and all(isinstance(x, list) and all(isinstance(s, str) for s in x) for x in v.values())
    return None if ok else f"{v!r} is not {{source: [series, ...]}}"


def _moments(width):
    def check(v):
        ok = isinstance(v, list) and all(isinstance(m, list) and len(m) == width and not any(_date(d) for d in m[:-1])
                                         for m in v)
        return None if ok else f"not a list of [{', '.join(['date'] * (width - 1))}, label]"
    return check


def _calendar(v):
    from policypath.calendars import BDAYS
    return None if v in BDAYS else f"{v!r} is not a calendar in calendars.BDAYS {sorted(BDAYS)}"


def _meetings_file(v):
    return None if (CONFIG_DIR / "meetings" / str(v)).exists() else f"config/meetings/{v} does not exist"


def _day_count(v):
    """The overnight rate's money-market basis, days a year: Act/360 or Act/365."""
    return None if v in (360, 365) and not isinstance(v, bool) else f"{v!r} is not a day count (360 or 365)"


def _tenor_map(v):
    """{tenor in years: series}: one cached series per maturity, shortest first."""
    if not isinstance(v, dict) or not v:
        return f"{v!r} is not {{tenor: series}}"
    bad = [t for t, s in v.items() if not _number(t) or t <= 0 or not isinstance(s, str)]
    if bad:
        return f"tenors {bad} are not positive years, each with a series name"
    return None if list(v) == sorted(v) else "tenors are not shortest first"


def _legs(v):
    """[short, long]: two tenors in years; the first is received, the second paid."""
    ok = isinstance(v, list) and len(v) == 2 and all(_number(t) and t > 0 for t in v) and v[0] < v[1]
    return None if ok else f"{v!r} is not [shorter tenor, longer tenor] in years"


def _bounds(v):
    """[low, high]: two positive numbers, low first."""
    ok = isinstance(v, list) and len(v) == 2 and all(_number(x) and x > 0 for x in v) and v[0] < v[1]
    return None if ok else f"{v!r} is not [low, high], two positive numbers"


def _one_series(v):
    """{source, series}: one series name."""
    ok = isinstance(v, dict) and set(v) == {"source", "series"} and all(isinstance(x, str) for x in v.values())
    return None if ok else f"{v!r} is not {{source, series}} with one series"


# A credit spread is the sum of its legs, each signed by its role.
SPREAD_SIGNS = {"long": 1, "plus": 1, "short": -1, "minus": -1}


def _spread(v):
    """{role: {source, series}}: long and plus are added, short and minus taken off; long is required."""
    if not isinstance(v, dict) or "long" not in v:
        return f"{v!r} has no long leg"
    bad = [r for r in v if r not in SPREAD_SIGNS]
    if bad:
        return f"legs {bad} are not {list(SPREAD_SIGNS)}"
    errors = [f"{r}: {_one_series(ref)}" for r, ref in v.items() if _one_series(ref)]
    return errors[0] if errors else None


def _windows(v):
    """{name: [first day, last day]}: date windows, each in order."""
    ok = isinstance(v, dict) and all(isinstance(w, list) and len(w) == 2 and not any(_date(d) for d in w)
                                     and pd.Timestamp(w[0]) <= pd.Timestamp(w[1]) for w in v.values())
    return None if ok else f"{v!r} is not {{name: [first day, last day]}}"


def _named(check):
    """{name: x}, every x passing `check`."""
    def named(v):
        if not isinstance(v, dict) or not v:
            return f"{v!r} is not {{name: ...}}"
        errors = [f"{name}: {check(x)}" for name, x in v.items() if check(x)]
        return errors[0] if errors else None
    return named


TAG_KINDS = ("lag", "quality", "modelling")


def _tags(v):
    """{name: {kind: text}}, kind in `TAG_KINDS`: one approximation, said once, for the brief and the limitations."""
    if not isinstance(v, dict):
        return f"{v!r} is not {{name: {{kind: text}}}}"
    for name, t in v.items():
        if not isinstance(t, dict) or not t:
            return f"{name}: {t!r} is not {{kind: text}}"
        bad = [k for k, text in t.items() if k not in TAG_KINDS or not isinstance(text, str) or not text.strip()]
        if bad:
            return f"{name}: {bad} are not {' | '.join(TAG_KINDS)} with a text"
    return None


def _tag_problems(node, path=""):
    """`_tags` on every ``tags`` block anywhere under `node`, named by its dotted path."""
    if not isinstance(node, dict):
        return []
    problems = []
    for key, value in node.items():
        at = f"{path}.{key}" if path else str(key)
        if key == "tags":
            err = _tags(value)
            problems += [f"{at}: {err}"] if err else []
        else:
            problems += _tag_problems(value, at)
    return problems


NUMBER = (int, float)

SCHEMA = {
    "enabled": bool,
    "meetings": _meetings_file,
    "history_start": _date,
    "calendar": _calendar,
    "overnight": _series_ref,
    "overnight.day_count": _day_count,
    "market.extractor": str,
    "market.label": str,
    "market.overnight_label": str,
    "market.final_after_bdays": int,
    "sources.daily": _lags,
    "path.start": _date,
    "path.n_meetings": int,
    "path.min_regime_days": int,
    "path.tail_days": int,
    "macro.source": str,
    "macro.history_start": _date,
    "macro.series": _list_of(str),
    "macro.validation": _list_of(str),
    "macro.projections": _list_of(str),
    "macro.start": _date,
    "macro.inflation": dict,
    "macro.gap.unemployment": str,
    "macro.gap.natural_rate": _natural_rate,
    "rule.policy_rate": _policy_rate,
    "rule.inflation": str,
    "rule.gap": str,
    "rule.inflation_target": _schedule(NUMBER),
    "rule.coefficients.inflation_gap": NUMBER,
    "rule.coefficients.unemployment_gap": NUMBER,
    "rule.inertia": NUMBER,
    "rule.meetings_per_quarter": _schedule(NUMBER),
    "rule.elb": _schedule(NUMBER),
    "rule.spread_fixings": int,
    "rule.spread_breaks": _list_of(str),
    "rule.rstar": dict,
    "signal.window": str,
    "signal.min_periods": int,
    "signal.sd_floor_bp": NUMBER,
    "backtest.horizon": int,
    "backtest.execution_lag": int,
    "backtest.dv01": NUMBER,
    "report.labels.market": str,
    "report.labels.rate": str,
    "report.labels.rule": str,
    "report.labels.policy": str,
    "report.labels.meetings": str,
    "report.labels.bank": str,
    "report.labels.inflation": str,
    "report.labels.policy_rate": str,
    "report.labels.nowcast_start": str,
    "report.labels.calendar": str,
    "report.labels.unemployment_period": str,
    "report.onepager": bool,
    "report.annotate_from": _date,
    "report.moments": _moments(2),
    "report.chart_moments": _moments(3),
}

# By ``market.extractor``: what each market mechanic needs besides the above.
EXTRACTOR_SCHEMA = {
    "futures_strip": {
        "market.futures": _series_ref,
        "market.exchange_calendar": str,
        "market.exchange": str,
        "sources.futures": _named_lists,
        "path.min_forward_days": int,
    },
    "forward_curve": {
        "market.curve": _series_ref,
        "market.curve.year_days": NUMBER,
        "market.pin_first_regime": bool,
        "sources.curves": _named_lists,
    },
}

# By the shape of ``macro.inflation``: a 12-month rate the statistics office publishes
# (``rate``), or an index bridged from an earlier-printing proxy (``target``).
INFLATION_SCHEMA = {
    "rate": {"macro.inflation.rate": str},
    "target": {"macro.inflation.target": str, "macro.inflation.bridge": str,
               "macro.inflation.bridge_window": int, "macro.inflation.wages": str},
}

# By the shape of ``rule.rstar``: a constant, or a published longer-run rate.
RSTAR_SCHEMA = {
    "constant": {"rule.rstar.constant": NUMBER},
    "published": {"rule.rstar.source": str, "rule.rstar.series": str, "rule.rstar.label": str,
                  "rule.rstar.before_first": NUMBER, "rule.rstar.before_first_label": str},
}

# Checked where a block has them.
OPTIONAL_SCHEMA = {"sources.estimates": _lags}

# Week 9's model-side robustness variants: optional keys a variant's override adds
# (a block without them is the baseline). Each is checked where a block has it, and
# a key under `VARIANT_SCHEMA` then needs the paths under it.

def _non_negative(v):
    return None if _number(v) and v >= 0 else f"{v!r} is not a number, 0 or more"


def _positive(v):
    return None if _number(v) and v > 0 else f"{v!r} is not a positive number"


CURVE_METHODS = ("log_linear", "nss")
VARIANT_OPTIONS = {"rule.rstar.real": bool, "market.curve.method": _one_of(*CURVE_METHODS)}
VARIANT_SCHEMA = {
    "rule.estimate": {"rule.estimate.prior_quarters": _non_negative,               # model/estimate.py
                      "rule.estimate.drop_cuts_to_floor": bool},
    "rule.conditioning": {"rule.conditioning.converge.half_life_quarters": _positive},   # model.path.converge_goals
}

# The expression layer's blocks, also checked only where a block has them. Under
# ``expression``, every key but `NOT_EXPRESSIONS` is an expression (what a sleeve
# trades in this currency), and its ``instrument`` (strategy/instruments.py) says
# what else it needs; ``{e}`` is the expression's key. Every expression has a
# ``label``: what the reports call its instrument (report/expression.py).
NOT_EXPRESSIONS = ("fx", "tags")
INSTRUMENT_SCHEMA = {
    "futures_month": {"expression.{e}.label": str, "expression.{e}.contract": str, "market.futures": _series_ref},
    "curve_forward": {"expression.{e}.label": str, "market.curve": _series_ref},
    "par_yield": {"expression.{e}.label": str, "expression.{e}.source": str, "expression.{e}.series": _tenor_map,
                  "expression.{e}.legs": _legs, "expression.{e}.coupons_per_year": int},
    "par_from_spot": {"expression.{e}.label": str, "expression.{e}.source": str, "expression.{e}.series": str,
                      "expression.{e}.legs": _legs, "expression.{e}.coupons_per_year": int},
}
# The instruments with a par leg at any tenor on their curve: what curve and cross sleeves trade.
PAR_INSTRUMENTS = ("par_yield", "par_from_spot")
# ``expression.fx``: the spot rate a leg outside the book currency converts at (`strategy` requires it there).
# ``plausible`` bounds the spot in book currency per unit: a quote outside it is read upside down.
FX_SCHEMA = {"expression.fx.source": str, "expression.fx.series": str, "expression.fx.book_per_unit": bool,
             "expression.fx.plausible": _bounds}
# ``contracts.{c}``: a futures contract an expression names. Its DV01 and tick values are derived from these;
# ``quoted`` is what its exchange publishes, the check they are compared with.
CONTRACT_SCHEMA = {"contracts.{c}.notional": NUMBER, "contracts.{c}.accrual.days": int,
                   "contracts.{c}.accrual.year_days": int, "contracts.{c}.tick.front": NUMBER,
                   "contracts.{c}.tick.other": NUMBER, "contracts.{c}.fee_per_side": NUMBER,
                   "contracts.{c}.exchange": str, "contracts.{c}.quoted.dv01": NUMBER,
                   "contracts.{c}.quoted.tick.front": NUMBER, "contracts.{c}.quoted.tick.other": NUMBER}
# ``costs.{e}``, one per expression, by how its round trip is given: in its contract's ticks, or in bp of yield.
COST_SCHEMA = {
    "ticks": {"costs.{e}.ticks_round_trip": NUMBER, "costs.{e}.observed": bool},
    "bp": {"costs.{e}.round_trip_bp": NUMBER, "costs.{e}.observed": bool},
}


def _months(v):
    """A par leg's re-strike interval (``costs.{e}.roll_every_months``): required for every par expression, since
    its P&L is a constant-maturity bond re-struck at each close and the re-strike a desk would do is what is charged."""
    ok = isinstance(v, int) and not isinstance(v, bool) and v >= 1
    return None if ok else f"{v!r} is not a whole number of months, 1 or more"
# ``credit``: spreads built from cached series (`SPREAD_SIGNS`), the one the bridge leads
# with, a maturity control, a short cross-check, the yield a stale quote would trail, the
# day weeks end on, and the windows the predictive test is also run without (credit.py).
WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
CREDIT_SCHEMA = {"credit.primary": str, "credit.spreads": _named(_spread), "credit.control": _spread,
                 "credit.crosscheck": _named(_one_series), "credit.staleness": _one_series,
                 "credit.week_ends": _one_of(*WEEKDAYS), "credit.exclude": _windows,
                 "credit.labels": _named(lambda v: None if isinstance(v, str) and v.strip() else f"{v!r} is not a label")}
# Besides every spread and cross-check, these are labelled.
CREDIT_LABELLED = ("control", "staleness")
# The bridge's own columns (credit.Inputs.weeks and daily_frame, and differential*): no spread
# or cross-check may take one, and no exclude window may be called elb (the ex_elb sample).
CREDIT_COLUMNS = ("market", "model", "slope", "level", "control", "yield", "yield_lag", "start", "state", "elb",
                  "gap_bp", "z", "sd_bp")


def _get(block, path):
    node = block
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            raise KeyError(path)
        node = node[key]
    return node


def _has(block, path):
    try:
        _get(block, path)
    except KeyError:
        return False
    return True


def _check(block, schema):
    problems = []
    for path, want in schema.items():
        try:
            value = _get(block, path)
        except KeyError:
            problems.append(f"missing {path}")
            continue
        if isinstance(want, (type, tuple)):
            wrong = not isinstance(value, want) or (isinstance(value, bool) and want is not bool)
            err = f"{path} is {value!r}, not {want.__name__ if isinstance(want, type) else want}" if wrong else None
        else:
            err = want(value)
            err = f"{path}: {err}" if err else None
        if err:
            problems.append(err)
    return problems


def _contract(x):
    """The futures contract expression `x` names, if its instrument takes one."""
    takes = "expression.{e}.contract" in INSTRUMENT_SCHEMA.get(x.get("instrument"), {})
    return x.get("contract") if takes and isinstance(x.get("contract"), str) else None


def _expressions(block):
    """The keys under ``expression`` that are expressions, in file order."""
    expression = block.get("expression")
    return [e for e in expression if e not in NOT_EXPRESSIONS] if isinstance(expression, dict) else []


def _cached(block):
    """Every (source, series) the block's sources cache."""
    out = set()
    for kind in ("daily", "curves", "futures", "estimates"):
        for source, series in block.get("sources", {}).get(kind, {}).items():
            out |= {(source, s) for s in series}
    return out


def _refs(block):
    """(what, source, series) for every series the chain reads from the cache."""
    out = [("overnight", block["overnight"]["source"], block["overnight"]["series"])]
    market = block["market"]
    for key in ("futures", "curve", "policy_rate"):
        if key in market:
            out.append((f"market.{key}", market[key]["source"], market[key]["series"]))
    rate = block["rule"]["policy_rate"]
    for r in rate if isinstance(rate, list) else [rate]:
        names = [r["series"]] if isinstance(r["series"], str) else r["series"]
        out += [("rule.policy_rate", r["source"], s) for s in names]
    rstar = block["rule"]["rstar"]
    if "constant" not in rstar:
        out.append(("rule.rstar", rstar["source"], rstar["series"]))
    if "sofr" in block:
        sofr = block["sofr"]
        out += [("sofr", sofr["source"], sofr["series"]),
                *(("sofr", block["market"]["futures"]["source"], sofr[k]) for k in ("futures", "crosscheck"))]
    expression = block.get("expression", {})
    for e in _expressions(block):
        x = expression[e]
        if _contract(x):                # a futures month settles off the market's own futures source
            out.append((f"expression.{e}.contract", market["futures"]["source"], _contract(x)))
        if x["instrument"] in PAR_INSTRUMENTS:
            names = x["series"].values() if isinstance(x["series"], dict) else [x["series"]]
            out += [(f"expression.{e}.series", x["source"], s) for s in names]
    if "fx" in expression:
        out.append(("expression.fx", expression["fx"]["source"], expression["fx"]["series"]))
    if "credit" in block:
        credit = block["credit"]
        legs = [(f"credit.spreads.{n}.{r}", ref) for n, spread in credit["spreads"].items()
                for r, ref in spread.items()]
        legs += [(f"credit.control.{r}", ref) for r, ref in credit["control"].items()]
        legs += [(f"credit.crosscheck.{n}", ref) for n, ref in credit["crosscheck"].items()]
        legs.append(("credit.staleness", credit["staleness"]))
        out += [(what, ref["source"], ref["series"]) for what, ref in legs]
    return out


def _expression_schema(block):
    """The paths the ``expression``, ``contracts`` and ``costs`` blocks need, given each expression's instrument.

    Returns (schema, problems): what cannot even be looked up (an expression with
    no known instrument, a cost with no round trip) is a problem here.
    """
    schema, problems = {}, []
    if "expression" in block and not isinstance(block["expression"], dict):
        return schema, [f"expression is {block['expression']!r}, not a mapping"]
    expression, costs = block.get("expression", {}), block.get("costs")
    for e in _expressions(block):
        instrument = expression[e].get("instrument") if isinstance(expression[e], dict) else None
        if instrument not in INSTRUMENT_SCHEMA:
            problems.append(f"missing expression.{e}.instrument" if instrument is None else
                            f"expression.{e}.instrument {instrument!r} is not one of {sorted(INSTRUMENT_SCHEMA)}")
            continue
        schema |= {p.format(e=e): want for p, want in INSTRUMENT_SCHEMA[instrument].items()}
        if _contract(expression[e]):
            schema |= {p.format(c=_contract(expression[e])): want for p, want in CONTRACT_SCHEMA.items()}
        if costs is None:
            continue
        entry = costs.get(e) if isinstance(costs, dict) else None
        if not isinstance(entry, dict) or not {"ticks_round_trip", "round_trip_bp"} & set(entry):
            problems.append(f"missing costs.{e}.ticks_round_trip or costs.{e}.round_trip_bp")
            continue
        shape = COST_SCHEMA["ticks" if "ticks_round_trip" in entry else "bp"]
        schema |= {p.format(e=e): want for p, want in shape.items()}
        if "roll_every_months" in entry or instrument in PAR_INSTRUMENTS:
            schema[f"costs.{e}.roll_every_months"] = _months
    if "fx" in expression:
        schema |= FX_SCHEMA
    return schema, problems


def _off_curve(x, tenor):
    """Why a par leg of `tenor` years cannot be read off expression `x`'s curve, or None."""
    if x["instrument"] == "par_yield" and not min(x["series"]) <= tenor <= max(x["series"]):
        return f"{tenor}y is outside its {min(x['series'])}-{max(x['series'])}y tenors"
    if (tenor * x["coupons_per_year"]) % 1:
        return f"{tenor}y is not a whole number of coupon periods"
    return None


def _expression_cross(block):
    """Cross-field checks on the ``expression`` and ``costs`` blocks, once every field is there."""
    expression, costs = block.get("expression", {}), block.get("costs", {})
    problems = [f"expression.{e}.legs: {_off_curve(expression[e], t)}" for e in _expressions(block)
                if expression[e]["instrument"] in PAR_INSTRUMENTS
                for t in expression[e]["legs"] if _off_curve(expression[e], t)]
    problems += [f"costs.{e}.ticks_round_trip: expression.{e} names no contract to take a tick from"
                 for e in _expressions(block) if "ticks_round_trip" in costs.get(e, {}) and not _contract(expression[e])]
    problems += [f"costs.{k} names no expression" for k in costs if k != "tags" and k not in _expressions(block)]
    return problems


def validate(ccy, block):
    """Every problem with one currency block, as a list of strings (empty if it is sound)."""
    problems = _check(block, SCHEMA)
    extractor = block.get("market", {}).get("extractor")
    if extractor is not None and extractor not in EXTRACTOR_SCHEMA:
        problems.append(f"market.extractor {extractor!r} is not one of {sorted(EXTRACTOR_SCHEMA)}")
    elif extractor is not None:
        problems += _check(block, EXTRACTOR_SCHEMA[extractor])
    inflation = block.get("macro", {}).get("inflation")
    if isinstance(inflation, dict):
        problems += _check(block, INFLATION_SCHEMA["rate" if "rate" in inflation else "target"])
    rstar = block.get("rule", {}).get("rstar")
    if isinstance(rstar, dict):
        problems += _check(block, RSTAR_SCHEMA["constant" if "constant" in rstar else "published"])
    problems += _check(block, {path: want for path, want in OPTIONAL_SCHEMA.items() if _has(block, path)})
    problems += _check(block, {path: want for path, want in VARIANT_OPTIONS.items() if _has(block, path)})
    problems += _check(block, {path: want for key, table in VARIANT_SCHEMA.items() if _has(block, key)
                               for path, want in table.items()})
    schema, unknown = _expression_schema(block)
    problems += unknown + _check(block, schema)
    if "credit" in block:
        problems += _check(block, CREDIT_SCHEMA)
    problems += _tag_problems(block)
    if problems:
        return problems
    # Cross-field checks, once every field is there.
    cached = _cached(block)
    problems += [f"{what} reads {source}/{series}, which no entry under sources caches"
                 for what, source, series in _refs(block) if (source, series) not in cached]
    if not 1 <= block["backtest"]["horizon"] <= block["path"]["n_meetings"]:
        problems.append(f"backtest.horizon {block['backtest']['horizon']} is outside 1..path.n_meetings")
    problems += _expression_cross(block)
    if "credit" in block and block["credit"]["primary"] not in block["credit"]["spreads"]:
        problems.append(f"credit.primary {block['credit']['primary']!r} is not one of credit.spreads")
    if "credit" in block:
        credit = block["credit"]
        unlabelled = [n for n in [*credit["spreads"], *credit["crosscheck"], *CREDIT_LABELLED] if n not in credit["labels"]]
        problems += [f"credit.labels has no label for {unlabelled}"] if unlabelled else []
        both = sorted(set(credit["spreads"]) & set(credit["crosscheck"]))
        problems += [f"credit.crosscheck {both} are also credit.spreads"] if both else []
        for part in ("spreads", "crosscheck"):
            taken = [n for n in credit[part] if n in CREDIT_COLUMNS or n.startswith("differential")]
            problems += [f"credit.{part} {taken} are the bridge's own column names"] if taken else []
        if not credit["exclude"]:
            problems.append("credit.exclude is empty: the headline and chart name its first window")
        if "elb" in credit["exclude"]:
            problems.append("credit.exclude has a window called elb, the name of the sample without ELB sessions")
    return problems


# ---- the book: config/strategy.yml --------------------------------------------------

# Sleeve kind -> the expression it trades in each of its currencies. A cross sleeve
# receives the first currency's par leg at its tenor and pays the second's.
SLEEVE_EXPRESSION = {"outright": "outright", "curve": "curve", "cross": "curve"}
POSITION_RULES = ("linear", "hysteresis")
CONSTRUCTIONS = ("erc", "mean_variance", "inverse_vol")
ELB_TREATMENTS = ("flat", "exclude", "hold")


def _sleeve_ccys(sleeve):
    return sleeve["pair"] if sleeve["kind"] == "cross" else [sleeve["ccy"]]


def _sleeves(v):
    """[{name, kind, ccy}, ...]; a cross sleeve has ``pair: [first, second]`` and ``tenor`` (years) instead of ccy."""
    if not isinstance(v, list) or not v:
        return f"{v!r} is not a list of sleeves"
    for s in v:
        if not isinstance(s, dict) or not isinstance(s.get("name"), str):
            return f"sleeve {s!r} has no name"
        kind = s.get("kind")
        if kind not in SLEEVE_EXPRESSION:
            return f"{s['name']}: kind {kind!r} is not one of {list(SLEEVE_EXPRESSION)}"
        if kind == "cross":
            pair = s.get("pair")
            if not (isinstance(pair, list) and len(pair) == 2 and all(isinstance(c, str) for c in pair)
                    and pair[0] != pair[1]):
                return f"{s['name']}: pair {pair!r} is not two different currencies"
            if not (_number(s.get("tenor")) and s["tenor"] > 0):
                return f"{s['name']}: tenor {s.get('tenor')!r} is not a positive number of years"
        elif not isinstance(s.get("ccy"), str):
            return f"{s['name']}: no ccy"
    names = [s["name"] for s in v]
    dupes = sorted({n for n in names if names.count(n) > 1})
    return f"sleeve names {dupes} are not unique" if dupes else None


# A robustness row moves the signal and nothing else: in a currency block it may override these paths (and what is
# under them), in the book these. The marks, instruments, carry, roll and costs stay the baseline's.
ROBUSTNESS_CURRENCY_PATHS = ("rule", "signal", "market.curve.method")
ROBUSTNESS_BOOK_PATHS = ("signal", "positions.carry_filter")
ROW_KEY = re.compile(r"[a-z][a-z0-9_]*")


def _overrides(v):
    """{...}: a non-empty mapping of config keys."""
    return None if isinstance(v, dict) and v else f"{v!r} is not a non-empty mapping"


def _row(r):
    """{key, label} and either ``every: {...}`` (each currency the book trades) or ``currencies: {ccy: {...}}``,
    and/or ``book: {...}``."""
    if not isinstance(r, dict) or not isinstance(r.get("key"), str) or not isinstance(r.get("label"), str):
        return f"row {r!r} has no key or label"
    if not ROW_KEY.fullmatch(r["key"]) or r["key"] == "baseline":
        return f"row key {r['key']!r} is not lower_case_with_underscores, or is 'baseline'"
    extra = sorted(set(r) - {"key", "label", "every", "currencies", "book"})
    if extra:
        return f"row {r['key']}: unknown keys {extra}"
    if "every" in r and "currencies" in r:
        return f"row {r['key']}: both every and currencies (give one)"
    if not {"every", "currencies", "book"} & set(r):
        return f"row {r['key']}: overrides nothing"
    for part in ("every", "book"):
        if part in r and _overrides(r[part]):
            return f"row {r['key']}: {part}: {_overrides(r[part])}"
    if "currencies" in r:
        c = r["currencies"]
        if not isinstance(c, dict) or not c or any(not isinstance(k, str) or _overrides(x) for k, x in c.items()):
            return f"row {r['key']}: currencies {c!r} is not {{ccy: {{...}}}}"
    return None


def _choices(v):
    """[{name, chosen, rows: [row, ...]}, ...] (`_row`): row keys unique over the grid, labels within a choice."""
    if not isinstance(v, list) or not v:
        return f"{v!r} is not a list of choices"
    keys = []
    for c in v:
        if not isinstance(c, dict) or not isinstance(c.get("name"), str) or not isinstance(c.get("chosen"), str):
            return f"choice {c!r} has no name or chosen"
        rows = c.get("rows")
        if not isinstance(rows, list) or not rows:
            return f"{c['name']}: rows {rows!r} is not a list"
        errors = [e for e in map(_row, rows) if e]
        if errors:
            return f"{c['name']}: {errors[0]}"
        labels = [r["label"] for r in rows]
        if len(set(labels)) < len(labels) or c["chosen"] in labels:
            return f"{c['name']}: row labels {labels} repeat, or repeat chosen"
        keys += [r["key"] for r in rows]
    dupes = sorted({k for k in keys if keys.count(k) > 1})
    return f"row keys {dupes} are not unique" if dupes else None


def row_overrides(row, ccys):
    """{ccy: override} of a robustness row: ``every`` on each of `ccys` (the book's currencies), else ``currencies``."""
    return {c: row["every"] for c in ccys} if "every" in row else dict(row.get("currencies", {}))


def _leaves(d, at=()):
    for k, v in d.items():
        if isinstance(v, dict) and v:
            yield from _leaves(v, (*at, k))
        else:
            yield ".".join((*at, k))


def _under(path, allowed):
    return any(path == a or path.startswith(a + ".") for a in allowed)


def _robustness_problems(book, blocks):
    """The robustness rows against the currency blocks and the rest of the book: every override reaches only the
    signal and, merged as `regress.merge` does, leaves a block that passes its own schema; a z window keeps
    ``min_periods`` at the baseline's share of it."""
    from policypath.regress import merge          # regress imports this module: here, not at the top
    r, problems = book["robustness"], []
    if r["ic_horizon"] not in book["evaluation"]["ic_horizons"]:
        problems.append(f"robustness.ic_horizon {r['ic_horizon']} is not one of evaluation.ic_horizons")
    traded = list(dict.fromkeys(c for s in book["sleeves"] for c in _sleeve_ccys(s)))
    schema = {p: w for p, w in STRATEGY_SCHEMA.items() if not p.startswith("robustness")}
    for row in (row for c in r["choices"] for row in c["rows"]):
        where = f"robustness row {row['key']}"
        for ccy, over in row_overrides(row, traded).items():
            wide = [p for p in _leaves(over) if not _under(p, ROBUSTNESS_CURRENCY_PATHS)]
            if ccy not in traded or ccy not in blocks:
                problems.append(f"{where}: {ccy} is not a currency the book trades")
            elif wide:
                problems.append(f"{where}: {ccy} {wide} reach past the signal ({list(ROBUSTNESS_CURRENCY_PATHS)})")
            else:
                merged = merge(blocks[ccy], over)
                problems += [f"{where}: {ccy} {p}" for p in validate(ccy, merged)]
                base, got = blocks[ccy]["signal"], merged["signal"]
                want = round(base["min_periods"] * pd.Timedelta(got["window"]) / pd.Timedelta(base["window"]))
                if "window" in over.get("signal", {}) and got["min_periods"] != want:
                    problems.append(f"{where}: {ccy} signal.min_periods {got['min_periods']} is not the baseline's "
                                    f"share of the window, {want}")
        if "book" in row:
            wide = [p for p in _leaves(row["book"]) if not _under(p, ROBUSTNESS_BOOK_PATHS)]
            if wide:
                problems.append(f"{where}: book {wide} reach past the signal ({list(ROBUSTNESS_BOOK_PATHS)})")
            else:
                problems += [f"{where}: book {p}" for p in _check(merge(book, row["book"]), schema)]
    return problems


def _meeting_pair(v):
    """[first, second]: two meeting horizons, 1-based, the first nearer."""
    ok = isinstance(v, list) and len(v) == 2 and all(isinstance(k, int) and not isinstance(k, bool) for k in v)
    return None if ok and 1 <= v[0] < v[1] else f"{v!r} is not [nearer meeting, further meeting], from 1"


def _some_of(names):
    def check(v):
        ok = isinstance(v, list) and v and all(x in names for x in v)
        return None if ok else f"{v!r} is not a list drawn from {list(names)}"
    return check


STRATEGY_SCHEMA = {
    "book.currency": str,
    "book.capital": NUMBER,
    "book.vol_target": NUMBER,
    "sleeves": _sleeves,
    "signal.slope": _meeting_pair,
    "signal.slope_orthogonal": bool,
    "carry.horizon_days": int,
    "carry.closure_min_pairs": int,
    "positions.rule": _one_of(*POSITION_RULES),
    "positions.enter": NUMBER,
    "positions.exit": NUMBER,
    "positions.grid.enter": _list_of(NUMBER),
    "positions.grid.exit": _list_of(NUMBER),
    "positions.carry_filter": bool,
    "costs.sensitivity_bp": _list_of(NUMBER),
    "risk.ewma_lambda": NUMBER,
    "risk.sigma_floor": NUMBER,
    "risk.nonsynchronous_lag": int,
    "risk.no_trade_band": NUMBER,
    "risk.max_gross_dv01_per_capital": _optional_number,
    "portfolio.constructions": _some_of(CONSTRUCTIONS),
    "portfolio.headline": _one_of(*CONSTRUCTIONS),
    "portfolio.z_cap": NUMBER,
    "portfolio.min_history": int,
    "portfolio.drawdown.trigger": NUMBER,
    "portfolio.drawdown.release": NUMBER,
    "portfolio.drawdown.scale": NUMBER,
    "evaluation.elb.chosen": _one_of(*ELB_TREATMENTS),
    "evaluation.ic_horizons": _list_of(int),
    "evaluation.exclude": _windows,
    "robustness.ic_horizon": int,
    "robustness.choices": _choices,
}


def validate_strategy(book, blocks):
    """Every problem with the book config, given the enabled currency blocks ({ccy: block})."""
    if not isinstance(book, dict):
        return [f"the book config is {book!r}, not a mapping"]
    problems = _check(book, STRATEGY_SCHEMA) + _tag_problems(book)
    if problems:
        return problems
    # Against the currency blocks, once every field is there.
    home, sleeves = book["book"]["currency"], book["sleeves"]
    if home not in blocks:
        problems.append(f"book.currency {home} is not an enabled currency")
    for s in sleeves:
        missing = [c for c in _sleeve_ccys(s) if c not in blocks]
        if missing:
            problems.append(f"sleeve {s['name']}: {missing} not enabled")
            continue
        want = SLEEVE_EXPRESSION[s["kind"]]
        for c in _sleeve_ccys(s):
            x = blocks[c].get("expression", {}).get(want)
            if x is None:
                problems.append(f"sleeve {s['name']}: {c} has no expression.{want}")
            elif s["kind"] == "cross" and x["instrument"] not in PAR_INSTRUMENTS:
                problems.append(f"sleeve {s['name']}: {c} expression.{want} is {x['instrument']}, not a par leg")
            elif s["kind"] == "cross" and _off_curve(x, s["tenor"]):
                problems.append(f"sleeve {s['name']}: {c} expression.{want}: {_off_curve(x, s['tenor'])}")
        if s["kind"] == "cross":
            horizons = {c: blocks[c]["backtest"]["horizon"] for c in s["pair"]}
            if len(set(horizons.values())) > 1:
                problems.append(f"sleeve {s['name']}: backtest.horizon differs across the pair, {horizons}")
        if s["kind"] == "curve" and book["signal"]["slope"][1] > blocks[s["ccy"]]["path"]["n_meetings"]:
            problems.append(f"signal.slope {book['signal']['slope']} reaches past {s['ccy']}'s path.n_meetings")
    used = {c for s in sleeves for c in _sleeve_ccys(s) if c in blocks}
    problems += [f"{c} trades in a book in {home} and has no expression.fx"
                 for c in sorted(used) if c != home and "fx" not in blocks[c].get("expression", {})]
    if home in blocks and "fx" in blocks[home].get("expression", {}):
        problems.append(f"{home} is the book currency: its expression.fx converts nothing")
    positions, portfolio = book["positions"], book["portfolio"]
    if not positions["exit"] < positions["enter"]:
        problems.append("positions.exit is not below positions.enter")
    if positions["enter"] not in positions["grid"]["enter"] or positions["exit"] not in positions["grid"]["exit"]:
        problems.append("positions.enter and exit are not a cell of positions.grid")
    if portfolio["headline"] not in portfolio["constructions"]:
        problems.append(f"portfolio.headline {portfolio['headline']} is not one of portfolio.constructions")
    if not 0 < portfolio["drawdown"]["release"] < portfolio["drawdown"]["trigger"]:
        problems.append("portfolio.drawdown.release is not between 0 and the trigger")
    return problems + _robustness_problems(book, blocks)


@cache
def _currencies():
    with open(CONFIG_DIR / "currencies.yml") as f:
        blocks = yaml.safe_load(f)
    problems = [f"{ccy}: {p}" for ccy, block in blocks.items() if block.get("enabled", False)
                for p in validate(ccy, block)]
    if problems:
        raise ConfigError(f"{CONFIG_DIR / 'currencies.yml'} fails its schema:\n  " + "\n  ".join(problems))
    return blocks


def currency(ccy):
    """The config block for one currency. Raises on an unknown or disabled currency."""
    blocks = _currencies()
    if ccy not in blocks:
        raise KeyError(f"{ccy} has no block in {CONFIG_DIR / 'currencies.yml'}")
    if not blocks[ccy].get("enabled", False):
        raise ValueError(f"{ccy} is not enabled in currencies.yml")
    return blocks[ccy]


def enabled():
    """Every currency whose block says ``enabled: true``, in file order."""
    return [ccy for ccy, block in _currencies().items() if block.get("enabled", False)]


@cache
def strategy():
    """The book config, ``strategy.yml``, checked against `STRATEGY_SCHEMA` and the enabled currency blocks."""
    path = CONFIG_DIR / "strategy.yml"
    with open(path) as f:
        book = yaml.safe_load(f)
    problems = validate_strategy(book, {ccy: currency(ccy) for ccy in enabled()})
    if problems:
        raise ConfigError(f"{path} fails its schema:\n  " + "\n  ".join(problems))
    return book


def meetings(ccy):
    """The meeting calendar for `ccy`: announcement_date, effective_date, scheduled, cancelled.

    Every meeting ever on it, including ones called off; `calendars.known_meetings`
    gives the calendar as it stood on a date.
    """
    path = CONFIG_DIR / "meetings" / currency(ccy)["meetings"]
    return pd.read_csv(path, parse_dates=["announcement_date", "effective_date", "cancelled"])
