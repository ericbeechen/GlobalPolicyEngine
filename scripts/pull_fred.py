import sys
from policypath.sources import cache, fred

START = "2016-01-01"

df = fred.effr(START)
added = cache.append(df, "fred", "EFFR", "USD")
print(f"EFFR {df['date'].min().date()} .. {df['date'].max().date()}: "
      f"{len(df)} fixings, {added} new rows in the cache")

if "--check" in sys.argv:
    bad = fred.check_publication_lag("EFFR", START)
    print(f"{len(bad)} dates where next-business-day != ALFRED realtime_start")
    if len(bad):
        lag = (bad["realtime_start"] - bad["published"]).dt.days
        print(lag.value_counts().sort_index().rename("days late vs rule").to_string())
        print(bad.tail(10).to_string(index=False))
