"""Snapshot-then-refactor: a currency's whole chain, frozen once, then compared bit for bit.

`outputs` runs every stage the build scripts run, through the same entry
points: the market path on every session (`panel.build`), the nowcast, the
model path, the gap and its z, and the crude backtest at every horizon. It
returns them as named frames. `freeze` writes them to a reference directory;
`compare` reruns and diffs exactly, with no tolerance, so a change that moves
one number by one ulp shows up. Both sides go through parquet, so a dtype the
format cannot keep is not reported as a change.

Exactly holds on one machine, not across two. The Mac (ARM, clang, Accelerate)
and Windows (x64, MSVC, OpenBLAS) round the same code differently in the last
bits: clang fuses a multiply and an add that MSVC rounds twice, even inside
numpy's own loops (`np.interp`), and the two LAPACKs solve `lstsq` in different
orders. On the fixtures that is at most 1.2e-13 of a column's largest value
(week 5's `residual_bp`, itself rounding noise), and 1.2e-14 elsewhere. A
reference committed on one and checked on the other is therefore compared to
`PLATFORM_ULPS` ulps of each float column's largest magnitude (at most 9e-13 of
it): 4 times the noise, and a fifth of the smallest move a 1e-12 change in the
rule's inertia makes (2.4e-12 of the column). One reference serves both machines.

Two tiers, one function:

- fixtures: the committed ``tests/data``, laid out as a cache
  (`policypath.fixtures`). The reference is committed under
  ``tests/data/reference/`` and checked by ``tests/test_regression.py``, so it
  runs anywhere.
- full: the real cache under ``data/``, every session. The reference is under
  ``data/reference/`` (gitignored), with the last session it covers, and a check
  stops there, so a cache updated since then compares like for like.

``scripts/regress.py`` runs either.
"""

import io
import json
from pathlib import Path
import subprocess
import tempfile
import time
import numpy as np
import pandas as pd
from policypath import config, fixtures, market
from policypath import panel as market_panel
from policypath.backtest import policy
from policypath.macro import nowcast
from policypath.macro.vintage import VintagePanel
from policypath.model import path as model_path
from policypath.signal import gap
from policypath.sources import cache

ROOT = Path(__file__).resolve().parents[2]
FULL_DIR = ROOT / "data" / "reference"
FIXTURE_REF = fixtures.FIXTURE_DIR / "reference"
# The tolerance a reference committed on one platform is checked to on another (module docstring).
PLATFORM_ULPS = 4096
STAGES = ["sessions", "meetings", "macro", "model", "paths", "signal", "backtest"]
# Blocks an override replaces whole instead of merging into. An r* block is one of two
# shapes: HLW merged into GBP's ``{constant: -1.6}`` would leave the constant in force.
REPLACE = {("rule", "rstar")}


def merge(base, over, at=()):
    """`base` with `over`'s keys replaced, recursively into nested dicts except the `REPLACE` blocks."""
    out = dict(base)
    for key, value in (over or {}).items():
        nested = isinstance(value, dict) and isinstance(base.get(key), dict) and (*at, key) not in REPLACE
        out[key] = merge(base[key], value, (*at, key)) if nested else value
    return out


def outputs(ccy, root=cache.CACHE_DIR, end=None, macro_days=None, overrides=None, calendar=None):
    """Every stage for `ccy` from the cache under `root`, through session `end` (default: the last).

    `macro_days` are the dates to nowcast; by default every business day from
    the config's ``macro.start`` to `end`, as ``scripts/build_nowcast.py`` does.
    `overrides` replace config keys, nested (``{"rule": {"spread_fixings": 1}}``), and
    the merged block is what every stage reads: a robustness variant is an override.
    The merged block must pass `config.validate`, so a mistyped key raises instead of
    leaving the baseline in place. The crude backtest marks on the published curve
    whatever ``market.curve.method`` the signal reads (`market.marked`).
    `calendar` replaces the config's meeting calendar.
    """
    cfg, root = merge(config.currency(ccy), overrides), Path(root)
    problems = config.validate(ccy, cfg) if overrides else []
    if problems:
        raise config.ConfigError(f"{ccy} with {overrides} fails its schema:\n  " + "\n  ".join(problems))
    sessions, meetings = market_panel.build(ccy, start=cfg["path"]["start"], end=end, root=root, calendar=calendar,
                                            cfg=cfg)
    marked = market.marked(cfg)
    marks = (sessions, meetings) if marked is cfg else market_panel.build(
        ccy, start=cfg["path"]["start"], end=end, root=root, calendar=calendar, cfg=marked)
    # The last session built, not `end` as given: date_range takes its resolution from its end point,
    # so a date parsed from meta.json would give the nowcast a different dtype than a freeze did.
    end = sessions["session"].max()
    panel = VintagePanel.from_cache(ccy, root)
    if macro_days is None:
        macro = nowcast.build(ccy, cfg["macro"]["start"], end, panel, cfg["macro"])
    else:
        macro = pd.DataFrame([nowcast.nowcast(d, ccy, panel, cfg["macro"]) for d in macro_days])
    inputs = model_path.inputs(ccy, root, rule=cfg["rule"])
    summaries, paths = model_path.build(sessions, meetings, macro, inputs["sep"], inputs["target"],
                                        inputs["fixings"], cfg["rule"])
    signal = gap.build(paths, cfg["signal"])
    backtest = pd.concat({k: policy.run(signal, *marks, k, cfg["backtest"])
                          for k in sorted(signal["k"].unique())}, names=["k", "session"]).reset_index()
    return {"sessions": sessions, "meetings": meetings, "macro": macro, "model": summaries,
            "paths": paths, "signal": signal, "backtest": backtest}


def fixture_outputs(ccy, logs=None, vintages=None, macro_days=None, calendar=None, overrides=None):
    """`outputs` on the committed fixtures (or frames given in their place), in a scratch cache.

    The nowcast is read on the fixture sessions (or `macro_days`), not every business day.
    `overrides` go on top of the fixtures' own (a variant, on the fixtures).
    """
    spec = fixtures.spec(ccy)
    days = fixtures.sessions(ccy) if macro_days is None else macro_days
    with tempfile.TemporaryDirectory() as tmp:
        fixtures.write(ccy, tmp, logs, vintages)
        return outputs(ccy, tmp, macro_days=days, overrides=merge(spec.get("overrides") or {}, overrides),
                       calendar=calendar)


def _roundtrip(frame):
    buf = io.BytesIO()
    frame.to_parquet(buf, index=False)
    buf.seek(0)
    return pd.read_parquet(buf)


def _commit():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except OSError:
        return None


def freeze(frames, out_dir, **meta):
    """Write each stage to ``<out_dir>/<stage>.parquet`` and what was frozen to ``meta.json``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_parquet(out_dir / f"{name}.parquet", index=False)
    last = frames["sessions"]["session"].max()
    info = {"end": f"{last:%Y-%m-%d}", "commit": _commit(),
            "frozen": pd.Timestamp.now().isoformat(timespec="seconds"),
            "rows": {name: len(f) for name, f in frames.items()}, **meta}
    (out_dir / "meta.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return info


def reference(out_dir):
    """The frozen stages and their ``meta.json``."""
    out_dir = Path(out_dir)
    if not (out_dir / "meta.json").exists():
        raise FileNotFoundError(f"no reference under {out_dir}; freeze one first (scripts/regress.py freeze)")
    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    return {name: pd.read_parquet(out_dir / f"{name}.parquet") for name in meta["rows"]}, meta


def _close(a, b, ulps):
    """Float columns within `ulps` ulps of the column's largest magnitude, NaN where the other is NaN."""
    x, y = a.to_numpy(float), b.to_numpy(float)
    if not np.array_equal(np.isnan(x), np.isnan(y)):
        return False
    scale = np.nanmax(np.abs(x), initial=0.0)
    return bool((np.abs(x - y)[~np.isnan(x)] <= ulps * np.spacing(scale)).all())


def _describe(want, got, ulps=0):
    """What differs between two frames, in a line; None if they are identical, bit for bit.

    With `ulps`, a float column within that many ulps of its largest magnitude is the same (`PLATFORM_ULPS`).
    """
    if list(want.columns) != list(got.columns):
        lost, new = sorted(set(want.columns) - set(got.columns)), sorted(set(got.columns) - set(want.columns))
        return f"columns differ: lost {lost}, new {new}" if lost or new else "columns reordered"
    if len(want) != len(got):
        return f"{len(want)} rows -> {len(got)}"
    for col in want.columns:
        a, b = want[col], got[col]
        if a.dtype != b.dtype:
            return f"{col}: dtype {a.dtype} -> {b.dtype}"
        if a.equals(b):
            continue
        if ulps and pd.api.types.is_float_dtype(a) and _close(a, b, ulps):
            continue
        same = (a == b) | (a.isna() & b.isna())
        bad = np.flatnonzero(~same.to_numpy())
        detail = ""
        if pd.api.types.is_numeric_dtype(a) and not pd.api.types.is_bool_dtype(a):
            detail = f", max |diff| {np.nanmax(np.abs(a.to_numpy(float) - b.to_numpy(float))):.3g}"
        first = bad[0]
        return (f"{col}: {len(bad)} of {len(a)} rows differ{detail}; "
                f"first at row {first}: {a.iloc[first]!r} -> {b.iloc[first]!r}")
    return None


def compare(frames, want, ulps=0):
    """{stage: what differs} for every stage that is not bit-identical to the reference `want`.

    `ulps`: the float tolerance, in ulps of each column's largest magnitude (`PLATFORM_ULPS` for
    a reference that may have been frozen on the other machine); 0 is bit for bit.
    """
    out = {}
    for name in want:
        if name not in frames:
            out[name] = "stage missing"
            continue
        diff = _describe(want[name], _roundtrip(frames[name]), ulps)
        if diff is not None:
            out[name] = diff
    return out


def truncate(frames, end):
    """Every stage through session (or as-of date) `end`: what a check against a shorter reference compares."""
    end = pd.Timestamp(end)
    date_col = {"macro": "as_of"}
    return {name: f[f[date_col.get(name, "session")] <= end].reset_index(drop=True) for name, f in frames.items()}


def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - t0
