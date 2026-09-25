"""Core PCE inflation as it could have been read on a date, with CPI standing in until PCE prints.

The rule's input is the 12-month rate: a bridged month's error enters it once,
where it enters a 3-month annualized rate about four times. See notes/macro.md.
"""

import numpy as np
import pandas as pd

MONTH = pd.DateOffset(months=1)


def monthly(s):
    """A monthly series on a gapless month-start index: holes are NaN, never skipped."""
    return s.asfreq("MS")


def log_change(index, months=1):
    return np.log(index / index.shift(months)) * 100


def bridge(target, proxy, window):
    """Extend `target` past its last month using the months `proxy` already has.

    Both are monthly price indexes from the same vintage. Target m/m is fitted
    on proxy m/m over the `window` months up to the target's last month; each
    month the proxy has beyond that is filled with the fitted m/m. A hole
    inside the proxy (October 2025 CPI) is log-interpolated, so the change
    across it is split evenly. Returns the extended index, the months filled
    and the fit.
    """
    target, proxy = monthly(target), monthly(proxy)
    last = target.last_valid_index()
    x = log_change(np.exp(np.log(proxy).interpolate(limit_area="inside")))
    y = log_change(target)
    fit = pd.concat({"x": x, "y": y}, axis=1).loc[:last].dropna().tail(window)
    slope, intercept = np.polyfit(fit["x"], fit["y"], 1)
    ahead = x.loc[x.index > last].dropna()
    out = target.loc[:last].copy()
    for month, dx in ahead.items():
        out[month] = out.iloc[-1] * np.exp((intercept + slope * dx) / 100)
    return out, list(ahead.index), (intercept, slope)


def rates(index, month):
    """12-month and annualized 3- and 6-month inflation to `month`, percent."""
    p = monthly(index)
    return {"12m": (p[month] / p[month - 12 * MONTH] - 1) * 100,
            "6m_ann": ((p[month] / p[month - 6 * MONTH]) ** 2 - 1) * 100,
            "3m_ann": ((p[month] / p[month - 3 * MONTH]) ** 4 - 1) * 100}


def inflation(panel, as_of, spec):
    """Core PCE to the latest month core CPI has, and wages as a diagnostic."""
    target = panel.series(spec["target"], as_of).dropna()
    if target.empty:
        raise ValueError(f"no {spec['target']} published by {pd.Timestamp(as_of).date()}")
    proxy = panel.series(spec["bridge"], as_of).dropna()
    bridged, filled, (a, b) = bridge(target, proxy, spec["bridge_window"])
    month = bridged.index[-1]
    r = rates(bridged, month)
    wages = monthly(panel.series(spec["wages"], as_of))
    w_month = wages.last_valid_index()
    w = rates(wages, w_month)
    return {"infl_month": month, "infl_12m": r["12m"], "infl_6m_ann": r["6m_ann"], "infl_3m_ann": r["3m_ann"],
            "target_month": target.index[-1], "n_bridged": len(filled),
            "bridge_intercept": a, "bridge_slope": b,
            "wages_month": w_month, "wages_12m": w["12m"], "wages_3m_ann": w["3m_ann"]}


def bridge_backtest(panel, as_of, spec, start="2021-01-01", window=None):
    """The bridge's real-time, out-of-sample error, one row per month since `start`.

    Each month the target printed by `as_of` is bridged from the vintage of the
    day before its release, and compared with its first print. Months CPI had
    not reached by then are skipped: there was nothing to bridge. Two naive
    alternatives ride along: proxy m/m taken 1:1, and the last target m/m
    carried forward. `across_hole` marks months whose proxy change was
    interpolated over a missing month.
    """
    window = window or spec["bridge_window"]
    released = panel.release_dates(as_of, spec["target"]).set_index("date")["published"]
    rows = []
    for month, day in released[released.index >= start].items():
        before = day - pd.Timedelta(days=1)
        target = panel.series(spec["target"], before).dropna()
        raw_proxy = monthly(panel.series(spec["bridge"], before))
        bridged, filled, (a, b) = bridge(target, raw_proxy.dropna(), window)
        if month not in filled:
            continue
        first = monthly(panel.series(spec["target"], day))
        proxy = np.exp(np.log(raw_proxy).interpolate(limit_area="inside"))
        rows.append({"month": month, "released": day,
                     "first_print": log_change(first)[month],
                     "bridge": log_change(monthly(bridged))[month],
                     "proxy_1to1": log_change(proxy)[month],
                     "last_carried": log_change(monthly(target)).dropna().iloc[-1],
                     "slope": b, "intercept": a,
                     "across_hole": bool(raw_proxy.loc[:month].iloc[-2:].isna().any())})
    return pd.DataFrame(rows)
