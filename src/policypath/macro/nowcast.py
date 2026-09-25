"""The US macro picture as it could have been read at the end of a date.

Inflation in percent, the unemployment gap in percentage points, activity as a
robust z. Reads a VintagePanel only, so it runs from the cache with no network.
"""

import pandas as pd
from policypath import config
from policypath.calendars import US_BDAY
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
    return ({"as_of": pd.Timestamp(as_of).normalize(), "published": used["published"].max()}
            | inflation(panel, as_of, spec["inflation"])
            | unemployment_gap(panel, as_of, spec["gap"])
            | activity(panel, as_of, spec["activity"]))


def build(ccy, start, end, panel=None, spec=None):
    """Every Fed business day in [start, end], one row each.

    No try/except: every date since 2021 should succeed, so a failure is a bug,
    not a data gap.
    """
    panel = panel if panel is not None else VintagePanel.from_cache(ccy)
    days = pd.date_range(start, end, freq=US_BDAY)
    return pd.DataFrame([nowcast(d, ccy, panel, spec) for d in days])
