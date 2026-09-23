import pandas as pd

def zq_prices_from_path(start_rate, changes, months):
    """changes: {effective_date: new_rate}. Returns {Period('YYYY-MM'): price}."""
    days = pd.date_range(months[0].start_time, months[-1].end_time, freq="D")
    rate = pd.Series(start_rate, index=days)
    for eff, r in sorted(changes.items()):
        rate[rate.index >= eff] = r
    avg = rate.groupby(rate.index.to_period("M")).mean()
    return 100.0 - avg
