"""Fixtures shared by the macro tests: every committed ALFRED vintage, as a panel."""

from pathlib import Path
import pandas as pd
import pytest
from policypath import config
from policypath.macro.vintage import VintagePanel

ALFRED = Path(__file__).parent / "data" / "alfred"


@pytest.fixture(scope="session")
def raw():
    frames = [pd.read_csv(p, parse_dates=["date", "realtime_start", "realtime_end"]).assign(series=p.stem)
              for p in sorted(ALFRED.glob("*.csv"))]
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="session")
def panel(raw):
    return VintagePanel(raw, config.currency("USD")["macro"]["projections"])
