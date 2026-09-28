"""A currency's macro picture as it could have been read at the end of a date.

Inflation in percent, the unemployment gap in percentage points, and (where the
config asks for it) activity as a robust z. Reads a VintagePanel only, so it
runs from the cache with no network.
"""

import pandas as pd
from policypath import config
from policypath.calendars import BDAYS
from policypath.macro.activity import activity
from policypath.macro.gap import unemployment_gap
from policypath.macro.inflation import inflation
from policypath.macro.vintage import VintagePanel


def inputs(spec):
    """Every series `nowcast` reads under `spec`: what its answer on a date can depend on."""
    names = list(spec["series"])
    for block in ("inflation", "gap"):
        names += [v for k, v in spec[block].items() if isinstance(v, str) and k in
                  ("target", "bridge", "wages", "rate", "unemployment", "natural_rate")]
    if "activity" in spec:
        names += [*spec["activity"]["series"], *spec["activity"].get("deflate", {}).values()]
    return sorted(set(names))


def nowcast(as_of, ccy=None, panel=None, spec=None):
    """One date. `panel` defaults to `ccy`'s cache; `spec` to its config's macro block.

    ``published`` is the latest vintage among the inputs: the newest thing the
    answer could depend on. `ccy` is needed only for what is not handed in.
    """
    if ccy is None and (panel is None or spec is None):
        raise ValueError("nowcast needs a currency, or both a panel and a macro spec")
    spec = spec or config.currency(ccy)["macro"]
    panel = panel if panel is not None else VintagePanel.from_cache(ccy)
    out = ({"as_of": pd.Timestamp(as_of).normalize(), "published": panel.published(as_of, spec["series"])}
           | inflation(panel, as_of, spec["inflation"])
           | unemployment_gap(panel, as_of, spec["gap"]))
    # The activity composite is a diagnostic, not a rule input: only where the config asks for it.
    return out | activity(panel, as_of, spec["activity"]) if "activity" in spec else out


def build(ccy, start, end, panel=None, spec=None):
    """Every business day in [start, end] (the currency's calendar), one row each.

    No try/except: every date from the config's ``macro.start`` (the first day
    every input has a vintage) should succeed, so a failure is a bug, not a data gap.

    A day on which none of the inputs changed (`VintagePanel.version`) gets the
    day before's row with its own ``as_of``: the nowcast is a function of the
    inputs as they stood and of nothing else about the date, so recomputing it
    gives the same numbers. Most days nothing is released.
    """
    spec = spec or config.currency(ccy)["macro"]
    panel = panel if panel is not None else VintagePanel.from_cache(ccy)
    days = pd.date_range(start, end, freq=BDAYS[config.currency(ccy)["calendar"]])
    names = inputs(spec)
    rows, last_key, last = [], None, None
    for day in days:
        key = panel.version(day, names)
        if key == last_key:
            row = {**last, "as_of": pd.Timestamp(day).normalize()}
        else:
            row = last = nowcast(day, ccy, panel, spec)
            last_key = key
        rows.append(row)
    return pd.DataFrame(rows)
