"""The macro nowcast as it could have been read at the end of one date.

Reads the cache only, so it needs no .env (run scripts/update_data.py
--macro-only first).

    uv run python scripts/run_nowcast.py 2024-01-20
"""

import sys
import pandas as pd
from policypath import config
from policypath.macro.nowcast import nowcast
from policypath.macro.vintage import VintagePanel

CCY = "USD"
spec = config.currency(CCY)["macro"]
day = pd.Timestamp(sys.argv[1] if len(sys.argv) > 1 else "2024-01-20")
panel = VintagePanel.from_cache(CCY)
n = nowcast(day, CCY, panel)

months = {s: f"{panel.series(s, day).dropna().index[-1]:%Y-%m}" for s in spec["series"] if s not in spec["projections"]}
activity = spec["activity"]["series"]
used = ", ".join(f"{s} {n[f'{s}_months']}" for s in activity)
print(f"as_of {n['as_of']:%Y-%m-%d}   published {n['published']:%Y-%m-%d}")
print(f"inflation   core PCE 12m {n['infl_12m']:.2f}%  (to {n['infl_month']:%Y-%m}, "
      f"{n['n_bridged']} month{'s' * (n['n_bridged'] != 1)} bridged from CPI; "
      f"PCE printed to {n['target_month']:%Y-%m})")
print(f"            3m ann {n['infl_3m_ann']:.2f}%  6m ann {n['infl_6m_ann']:.2f}%   "
      f"bridge {n['bridge_intercept']:.2f} + {n['bridge_slope']:.2f} x CPI   "
      f"wages 12m {n['wages_12m']:.2f}% (to {n['wages_month']:%Y-%m})")
print(f"gap         u {n['u']:.1f} ({n['gap_month']:%Y-%m})  u* {n['u_star']:.2f}  u - u* = {n['u_gap']:+.2f}pp")
print(f"activity    {n['activity_quarter'].to_period('Q')}: z {n['activity_z']:+.2f} from {n['activity_n']} series  "
      f"(months in quarter: {used})")
print("\nlast month  " + "  ".join(f"{s} {m}" for s, m in months.items()))
for s in activity:
    print(f"            {s:<9} growth {n[f'{s}_growth']:+6.2f}% ann   z {n[f'{s}_z']:+.2f}")
