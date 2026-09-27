"""The market-implied policy path on every session, solved from the cache.

Backend-agnostic: the config's ``market.extractor`` says which market
mechanic a currency's path comes from (`market.py`), and this module only
loops over its sessions. See `market.py` for what each session reads and when
it was knowable.

The ZQ helpers (`settles_on`, `monthly_strip`, `solve_session`) are re-exported
from `market.py` for the scripts and tests written against them.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from policypath import config
from policypath.market import (PolicyPath, Session, ShortStrip, extractor, monthly_strip,  # noqa: F401
                               settles_on, solve_curve, solve_session)
from policypath.sources import cache

PANEL_DIR = Path(__file__).resolve().parents[2] / "data" / "panel"
COLUMNS = ["session", "k", "announcement_date", "effective_date", "scheduled", "rate", "step_bp", "cum_bp"]


def build(ccy, start=None, end=None, root=cache.CACHE_DIR, **overrides):
    """Solve every session in the cache. Returns (sessions, meetings), both long frames.

    `sessions` has one row per trade date, solved or not; a failed solve keeps
    its row with the reason in ``error``. `meetings` has one row per session
    and upcoming meeting. `overrides` replace keys of the config's ``path`` block.
    Only `ValueError` (a short strip or curve, an underdetermined solve) and
    `LinAlgError` are recorded as failures; anything else is a bug and raises.
    """
    ext = extractor(ccy, root)
    ext.spec = {**ext.spec, **overrides}
    dates = ext.sessions()
    if start is not None:
        dates = dates[dates >= pd.Timestamp(start)]
    if end is not None:
        dates = dates[dates <= pd.Timestamp(end)]

    summaries, frames = [], []
    for day in dates:
        row = {"session": day, "published": ext.published(day), "error": None}
        try:
            result = ext.session(day)
        except (ValueError, np.linalg.LinAlgError) as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        else:
            row.update(result.summary)
            frames.append(result.meetings.assign(session=day))
        summaries.append(row)

    sessions = pd.DataFrame(summaries)
    meetings = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return sessions, meetings[COLUMNS] if len(meetings) else meetings


def load(ccy, name, root=PANEL_DIR):
    """One table `scripts/build_*.py` wrote to ``data/panel/<ccy>_<name>.parquet`` (sessions, meetings, macro, ...)."""
    path = Path(root) / f"{ccy}_{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run the build script that writes it")
    return pd.read_parquet(path)
