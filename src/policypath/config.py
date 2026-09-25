"""Read `config/`: the per-currency block and the meeting calendar."""

from functools import cache
from pathlib import Path
import pandas as pd
import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


@cache
def _currencies():
    with open(CONFIG_DIR / "currencies.yml") as f:
        return yaml.safe_load(f)


def currency(ccy):
    """The config block for one currency. Raises on an unknown or disabled currency."""
    blocks = _currencies()
    if ccy not in blocks:
        raise KeyError(f"{ccy} has no block in {CONFIG_DIR / 'currencies.yml'}")
    if not blocks[ccy].get("enabled", False):
        raise ValueError(f"{ccy} is not enabled in currencies.yml")
    return blocks[ccy]


def meetings(ccy):
    """The meeting calendar for `ccy`: announcement_date, effective_date, scheduled, cancelled.

    Every meeting ever on it, including ones called off; `calendars.known_meetings`
    gives the calendar as it stood on a date.
    """
    path = CONFIG_DIR / "meetings" / currency(ccy)["meetings"]
    return pd.read_csv(path, parse_dates=["announcement_date", "effective_date", "cancelled"])
