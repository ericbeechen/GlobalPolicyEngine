"""Fixtures shared across the tests.

Every committed ALFRED vintage, as a panel, for the macro tests; and the rule's
market-side inputs (fixings, the target range, the SEP medians) as the logs the
model reads, each with the day it became public.
"""

from pathlib import Path
import pandas as pd
import pytest
from policypath import config
from policypath.macro.vintage import VintagePanel

DATA = Path(__file__).parent / "data"
ALFRED = DATA / "alfred"


@pytest.fixture(scope="session")
def raw():
    frames = [pd.read_csv(p, parse_dates=["date", "realtime_start", "realtime_end"]).assign(series=p.stem)
              for p in sorted(ALFRED.glob("*.csv"))]
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="session")
def panel(raw):
    return VintagePanel(raw, config.currency("USD")["macro"]["projections"])


def _effr():
    return pd.read_csv(DATA / "effr.csv", parse_dates=["date", "published"])


@pytest.fixture(scope="session")
def fixings():
    """EFFR fixings: date, value, published (the next business day)."""
    return _effr().rename(columns={"effr": "value"})[["date", "value", "published"]]


@pytest.fixture(scope="session")
def target():
    """The target-range midpoint in force on each fixing date, public on that date."""
    e = _effr().dropna(subset=["target_low", "target_high"])
    return pd.DataFrame({"date": e["date"], "value": (e["target_low"] + e["target_high"]) / 2,
                         "published": e["date"]})


@pytest.fixture(scope="session")
def sep():
    """FOMC SEP median longer-run funds rate, one row per SEP, public on its release day."""
    return pd.read_csv(DATA / "sep.csv", parse_dates=["date", "published"])
