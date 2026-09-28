"""Read `config/`: the per-currency block and the meeting calendar.

Every enabled block in ``currencies.yml`` is validated against `SCHEMA` when the
file is first read, and a block that is missing a field, or has one of the
wrong shape, fails there with every problem listed by its dotted path. That is
what makes a new currency an hour of config rather than an afternoon of
KeyErrors: the first run says everything the block still needs.
"""

from functools import cache
from numbers import Number
from pathlib import Path
import pandas as pd
import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
ROOT = CONFIG_DIR.parent


class ConfigError(ValueError):
    """A currency block that does not meet the schema. The message lists every problem."""


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
    def check(v):
        ok = isinstance(v, list) and all(isinstance(x, kind) for x in v)
        return None if ok else f"{v!r} is not a list of {kind.__name__}"
    return check


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
    """{source: {series: business days to publication}}: every series cached daily says when it is public."""
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


NUMBER = (int, float)

SCHEMA = {
    "enabled": bool,
    "meetings": _meetings_file,
    "history_start": _date,
    "calendar": _calendar,
    "overnight": _series_ref,
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


def _get(block, path):
    node = block
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            raise KeyError(path)
        node = node[key]
    return node


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


def _cached(block):
    """Every (source, series) the block's sources cache."""
    out = set()
    for kind in ("daily", "curves", "futures"):
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
    return out


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
    if problems:
        return problems
    # Cross-field checks, once every field is there.
    cached = _cached(block)
    problems += [f"{what} reads {source}/{series}, which no entry under sources caches"
                 for what, source, series in _refs(block) if (source, series) not in cached]
    if not 1 <= block["backtest"]["horizon"] <= block["path"]["n_meetings"]:
        problems.append(f"backtest.horizon {block['backtest']['horizon']} is outside 1..path.n_meetings")
    return problems


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


def meetings(ccy):
    """The meeting calendar for `ccy`: announcement_date, effective_date, scheduled, cancelled.

    Every meeting ever on it, including ones called off; `calendars.known_meetings`
    gives the calendar as it stood on a date.
    """
    path = CONFIG_DIR / "meetings" / currency(ccy)["meetings"]
    return pd.read_csv(path, parse_dates=["announcement_date", "effective_date", "cancelled"])
