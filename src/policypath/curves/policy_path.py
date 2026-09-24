import numpy as np
import pandas as pd


def implied_path(implied_avg: pd.Series, effective_dates: pd.Series,
                 realized: pd.Series | None = None, min_forward_days: int = 0,
                 min_regime_days: int = 0) -> pd.Series:
    """Piecewise-constant overnight rate between meeting effective dates.

    `implied_avg` is the implied average rate per contract month (100 - price),
    indexed by monthly Period. Each month is one linear equation: its calendar-day
    average of the overnight rate. All months are solved jointly by least squares.

    `realized` is the rate already known for each calendar day, through the last
    known day (`calendars.known_daily`). Days of the strip it covers enter each
    month's average as known numbers instead of unknowns, and the path starts on
    the first day after them. Without it every day of the strip is solved for,
    which is right only for a strip starting at a month start.

    Two cutoffs keep a handful of days from carrying a whole rate. A contract
    whose unknown days are few says little about them, and what it says is price
    noise times days-in-month / days-left.

    - `min_forward_days`: drop the front contract when fewer than this many of
      its days are still unknown.
    - `min_regime_days`: pin the first regime -- today until the next meeting --
      to the last known fixing when it spans fewer than this many days, or when
      dropping the front contract leaves it in no contract at all.

    Returns the rate in force from each pillar -- the first unknown day, then
    each effective date after it. ``attrs`` carries ``front_dropped``,
    ``pinned`` and ``residual_bp``, the worst repricing error over the contracts used.
    """
    if implied_avg.empty:
        raise ValueError("implied_avg is empty: no contracts to bootstrap from")
    implied_avg = implied_avg.sort_index()
    months = implied_avg.index
    first, end = months[0].start_time, months[-1].end_time.normalize()
    days = pd.date_range(first, end, freq="D")

    known = _known_days(realized, first, end)
    start = first + pd.Timedelta(days=len(known))
    if start > end:
        raise ValueError(f"every day of the strip {months[0]}..{months[-1]} is already realized")
    eff = pd.DatetimeIndex(sorted(effective_dates))
    eff = eff[(eff > start) & (eff <= end)]

    in_month = months.days_in_month.to_numpy()
    unknown = days[days >= start]
    regime = eff.searchsorted(unknown, side="right")
    counts = (pd.crosstab(unknown.to_period("M"), regime)
              .reindex(index=months, columns=range(len(eff) + 1), fill_value=0))
    W = counts.to_numpy() / in_month[:, None]
    known_avg = known.groupby(known.index.to_period("M")).sum().reindex(months, fill_value=0.0)
    y = implied_avg.to_numpy() - known_avg.to_numpy() / in_month

    left = counts.sum(axis=1).to_numpy()
    keep = left > 0
    front_dropped = bool(min_forward_days and 0 < left[0] < min_forward_days)
    keep[0] &= not front_dropped
    if not keep.any():
        raise ValueError(f"no contract in {months[0]}..{months[-1]} has unknown days left to solve")
    W, y = W[keep], y[keep]

    last_fixing = None if realized is None or realized.empty else float(realized.sort_index().iloc[-1])
    orphaned = W[:, 0].sum() == 0
    pinned = last_fixing is not None and (orphaned or (regime == 0).sum() < min_regime_days)
    solve = np.arange(1 if pinned else 0, W.shape[1])
    if pinned:
        y = y - W[:, 0] * last_fixing
    if (W[:, solve].sum(axis=0) == 0).any() or np.linalg.matrix_rank(W[:, solve]) < len(solve):
        raise ValueError(
            f"{int(keep.sum())} contracts cannot identify {len(solve)} rates "
            f"(effective dates {[d.date().isoformat() for d in eff]})"
        )
    rates = np.empty(W.shape[1])
    rates[solve], *_ = np.linalg.lstsq(W[:, solve], y, rcond=None)
    if pinned:
        rates[0] = last_fixing

    path = pd.Series(rates, index=pd.DatetimeIndex([start, *eff]), name="rate")
    path.attrs = {
        "front_dropped": front_dropped,
        "pinned": bool(pinned),
        "residual_bp": float(np.abs(W[:, solve] @ rates[solve] - y).max() * 100.0),
    }
    return path


def _known_days(realized, first, end):
    """The part of `realized` inside [first, end], checked to run unbroken from `first`."""
    if realized is None:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    known = realized.sort_index().loc[first:end]
    if len(known) and not known.index.equals(pd.date_range(first, periods=len(known), freq="D")):
        raise ValueError(f"realized rates must cover every calendar day from {first.date()} "
                         f"with no gaps; got {known.index[0].date()}..{known.index[-1].date()}")
    if known.isna().any():
        raise ValueError("realized rates contain NaN")
    return known
