"""Which `Source` a config's ``sources`` block means, by name.

A currency's config lists what to cache under ``sources`` (daily series with
their publication lags, fitted curves, futures roots, quarterly estimates with
theirs) and ``macro.source``; the names there are the keys here. Adding a data
provider is a module in ``sources/`` and a line in one of these tables;
``scripts/update_data.py`` is not edited. Imports are deferred so that reading
the registry does not pull in every provider's client library.
"""


def _fred(lag):
    from policypath.sources import fred
    return fred.Fred(lag_bdays=lag)


def _boe(lag):
    from policypath.sources import boe
    return boe.BoeSeries(lag_bdays=lag)


def _boe_curve():
    from policypath.sources import boe
    return boe.BoeCurve()


def _nyfed(lag_days):
    from policypath.sources import nyfed
    return nyfed.Hlw(lag_days=lag_days)


def _databento(live):
    from policypath.sources import rates
    return rates.Settlements(client=rates.live_client() if live else None)


def _alfred(series, start):
    from policypath.sources import fred
    meta = {**fred.series_info(series), "first_vintage": f"{fred.vintage_dates(series).iloc[0]:%Y-%m-%d}"}
    return fred.vintages(series, start), meta


_release_days = {}


def _ons(series, start):
    from policypath.sources import ons
    kind = ons.SERIES[series]["release"]
    if kind not in _release_days:
        _release_days[kind] = ons.release_days(kind)
    v = ons.vintages(series, start, _release_days[kind])
    meta = {"first_vintage": f"{v['realtime_start'].min():%Y-%m-%d}",
            "release_days_by_rule_before": f"{ons.lag_approximated_before(_release_days[kind]):%Y-%m-%d}"}
    return v, meta


# ``sources.daily``: name -> f(lag in business days) -> a Source of one daily series.
DAILY = {"fred": _fred, "boe": _boe}
# ``sources.curves``: name -> f() -> a Source of one fitted curve, keyed by (date, tenor).
CURVES = {"boe": _boe_curve}
# ``sources.futures``: name -> f(live) -> a Source of one futures root's settles, keyed by (date, contract).
# Its `ranges(series, end)` says what to update, and `spent` what its requests cost; with `live`
# it may reach past its archive over the provider's API, if the provider's key is set.
FUTURES = {"databento": _databento}
# ``sources.estimates``: name -> f(calendar days from the end of a vintage's quarter to its release)
# -> a Source of one quarterly estimate, one row per vintage.
ESTIMATES = {"nyfed": _nyfed}
# ``macro.source``: name -> f(series, start) -> (every vintage, manifest metadata).
MACRO = {"alfred": _alfred, "ons": _ons}


def lookup(table, name):
    try:
        return table[name]
    except KeyError:
        raise KeyError(f"no source {name!r} in sources/registry.py; known: {sorted(table)}") from None
