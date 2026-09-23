import numpy as np
import pandas as pd


def implied_path(implied_avg: pd.Series, effective_dates: pd.Series) -> pd.Series:
    if implied_avg.empty:
        raise ValueError("implied_avg is empty: no contracts to bootstrap from")
    implied_avg = implied_avg.sort_index()
    months = implied_avg.index
    start, end = months[0].start_time, months[-1].end_time.normalize()
    eff = pd.DatetimeIndex(sorted(effective_dates))
    eff = eff[(eff > start) & (eff <= end)]

    # Regime k runs from eff[k-1] up to the day before eff[k]; regime 0 from `start`.
    days = pd.date_range(start, end, freq="D")
    regime = eff.searchsorted(days, side="right")
    weights = pd.crosstab(days.to_period("M"), regime, normalize="index")
    weights = weights.reindex(index=months, columns=range(len(eff) + 1), fill_value=0.0)

    W = weights.to_numpy()
    if np.linalg.matrix_rank(W) < W.shape[1]:
        raise ValueError(
            f"{len(months)} contracts cannot identify {W.shape[1]} rates "
            f"(effective dates {[d.date().isoformat() for d in eff]})"
        )
    rates, *_ = np.linalg.lstsq(W, implied_avg.to_numpy(), rcond=None)
    return pd.Series(rates, index=pd.DatetimeIndex([start, *eff]), name="rate")
