"""Print the ZQ and SR3 contracts and their final settlements for one trade date.

    uv run python scripts/show_databento.py 2024-03-15
"""

import sys

import pandas as pd

from policypath.sources import rates

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 200)

day = pd.Timestamp(sys.argv[1] if len(sys.argv) > 1 else "2024-03-15")
# Final settles for a session arrive on a later day, so read a few days past it.
settles = rates.settlements(day, day + pd.Timedelta(days=4))
settles = settles[settles["trade_date"].dt.date == day.date()]

for asset, curve in settles.groupby("asset"):
    print(f"\n{asset} final settlements for {day.date()} ({len(curve)} contracts)")
    print(curve[["symbol", "expiration", "price", "implied_rate", "published"]].to_string(index=False))
