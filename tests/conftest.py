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


# ---- GBP: the Bank of England's series and curve, the ONS vintages ------------------

BOE, ONS = DATA / "boe", DATA / "ons"


@pytest.fixture(scope="session")
def sonia():
    """SONIA fixings: date, value, published (the next London business day)."""
    return pd.read_csv(BOE / "sonia.csv", parse_dates=["date", "published"])


@pytest.fixture(scope="session")
def bank_rate():
    """Bank Rate in force each day, public that day (a change is announced at noon and applies from then)."""
    return pd.read_csv(BOE / "bank_rate.csv", parse_dates=["date", "published"])


@pytest.fixture(scope="session")
def curve():
    """The Bank's OIS spot curve, 1-24 months, on the committed days: date, tenor, value, published."""
    return pd.read_csv(BOE / "curve.csv", parse_dates=["date", "published"])


@pytest.fixture(scope="session")
def ons_raw():
    frames = [pd.read_csv(p, parse_dates=["date", "realtime_start", "realtime_end"]).assign(series=p.stem)
              for p in sorted(ONS.glob("*.csv"))]
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="session")
def ons_panel(ons_raw):
    return VintagePanel(ons_raw, config.currency("GBP")["macro"]["projections"])
