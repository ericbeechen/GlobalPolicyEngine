"""The committed fixtures under ``tests/data``, laid out as a cache, so the whole chain runs on them.

``tests/data/fixtures.yml`` says, per currency, which file holds which cached
series and which sessions the fixtures cover. `write` puts them in the layout
`sources/cache.py` reads, so the production entry points (`market.extractor`,
`model.path.inputs`, `VintagePanel.from_cache`) run on them unchanged. That is
what the regression harness freezes in its fixtures tier, and what the
per-currency look-ahead test truncates and poisons.
"""

from functools import cache as memo
from pathlib import Path
import pandas as pd
import yaml
from policypath.sources import cache

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "data"
RETRIEVED = pd.Timestamp("2026-09-27")   # every fixture is one pull
STRIP_RECEIVED = pd.Timedelta(hours=16)  # a strip's settles, as received on its session


@memo
def _specs():
    with open(FIXTURE_DIR / "fixtures.yml") as f:
        return yaml.safe_load(f)


def spec(ccy):
    """The currency's block in fixtures.yml. Raises on a currency with no fixtures."""
    specs = _specs()
    if ccy not in specs:
        raise KeyError(f"{ccy} has no block in {FIXTURE_DIR / 'fixtures.yml'}")
    return specs[ccy]


def currencies():
    return list(_specs())


def sessions(ccy):
    return pd.DatetimeIndex(spec(ccy)["sessions"])


def _strips(s, days):
    frames = []
    for day in days:
        f = pd.read_csv(FIXTURE_DIR / s["file"].format(session=f"{day:%Y-%m-%d}"))
        month = pd.PeriodIndex(f["month"], freq="M")
        frames.append(pd.DataFrame({"date": day, "contract": f["symbol"],
                                    "expiration": month.end_time.normalize(), "value": f["price"],
                                    "published": day + STRIP_RECEIVED}))
    return pd.concat(frames, ignore_index=True)


def _log(s):
    f = pd.read_csv(FIXTURE_DIR / s["file"])
    value, published = s.get("value", "value"), s.get("published", "published")
    keys = [k for k in s.get("keys", ["date"]) if k != "date"]
    out = pd.DataFrame({"date": pd.to_datetime(f["date"]), **{k: f[k] for k in keys},
                        "value": f[value], "published": pd.to_datetime(f[published])})
    return out.dropna(subset=["value"]).reset_index(drop=True)


def logs(ccy):
    """Every daily log the fixtures hold for `ccy`: {(source, series): frame}, as `cache.log` returns them."""
    s = spec(ccy)
    out = {}
    if "strips" in s:
        out[(s["strips"]["source"], s["strips"]["series"])] = _strips(s["strips"], sessions(ccy))
    for entry in s.get("logs", []):
        out[(entry["source"], entry["series"])] = _log(entry)
    return {key: f.assign(retrieved=RETRIEVED) for key, f in out.items()}


def vintages(ccy):
    """Every real-time vintage file: {(source, series): frame of date, value, realtime_start, realtime_end}."""
    v = spec(ccy)["vintages"]
    dates = ["date", "realtime_start", "realtime_end"]
    return {(v["source"], p.stem): pd.read_csv(p, parse_dates=dates)
            for p in sorted((FIXTURE_DIR / v["dir"]).glob("*.csv"))}


def write(ccy, root, logs_=None, vintages_=None):
    """Lay the fixtures (or the frames given in their place) out as a cache under `root`. Returns `root`."""
    root = Path(root)
    for (source, series), frame in {**(logs(ccy) if logs_ is None else logs_),
                                    **(vintages(ccy) if vintages_ is None else vintages_)}.items():
        path = cache._path(source, series, ccy, root)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
    return root
