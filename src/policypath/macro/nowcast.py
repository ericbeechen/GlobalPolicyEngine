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


def nowcast(as_of, ccy="USD", panel=None, spec=None):
    """One date. `panel` defaults to the cache; `spec` to the config's macro block.

    ``published`` is the latest vintage among the inputs: the newest thing the
    answer could depend on.
    """
    spec = spec or config.currency(ccy)["macro"]
    panel = panel if panel is not None else VintagePanel.from_cache(ccy)
    used = panel.as_of(as_of, spec["series"])
    out = ({"as_of": pd.Timestamp(as_of).normalize(), "published": used["published"].max()}
           | inflation(panel, as_of, spec["inflation"])
           | unemployment_gap(panel, as_of, spec["gap"]))
    # The activity composite is a diagnostic, not a rule input: only where the config asks for it.
    return out | activity(panel, as_of, spec["activity"]) if "activity" in spec else out


def build(ccy, start, end, panel=None, spec=None):
    """Every business day in [start, end] (the currency's calendar), one row each.

    No try/except: every date from the config's ``macro.start`` (the first day
    every input has a vintage) should succeed, so a failure is a bug, not a data gap.
    """
    panel = panel if panel is not None else VintagePanel.from_cache(ccy)
    days = pd.date_range(start, end, freq=BDAYS[config.currency(ccy)["calendar"]])
    return pd.DataFrame([nowcast(d, ccy, panel, spec) for d in days])
