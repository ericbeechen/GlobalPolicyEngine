"""Check the EFFR publication-lag rule against ALFRED's first-seen dates.

Caching lives in scripts/update_data.py; this only audits the lag rule that
gives every FRED observation its ``published`` date.

    uv run --env-file .env python scripts/pull_fred.py
"""

from policypath.sources import fred

START = "2016-01-01"

bad = fred.check_publication_lag("EFFR", START)
print(f"{len(bad)} dates where next-business-day != ALFRED realtime_start")
if len(bad):
    lag = (bad["realtime_start"] - bad["published"]).dt.days
    print(lag.value_counts().sort_index().rename("days late vs rule").to_string())
    print(bad.tail(10).to_string(index=False))
