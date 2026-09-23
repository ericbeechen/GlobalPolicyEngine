"""Regenerate the committed test fixtures.

Not run by the test suite. It needs the Databento archive (paid, gitignored) and
the network, which is exactly what the fixtures exist to keep out of `pytest`.
Everything it writes is small enough to read in a diff.

    uv run python tests/data/build_fixtures.py

Outputs, all in this directory:
  effr.csv              NY Fed EFFR and prevailing target range, one row per day.
  zq_expiry_settles.csv One row per expired ZQ contract: its settle on its own
                        expiry session, which must equal 100 - the realized
                        average EFFR for the contract month.
  zq_strip_<date>.csv   The whole ZQ strip on a few chosen sessions.
"""

import pandas as pd
import requests

from policypath.sources import rates

HERE = __import__("pathlib").Path(__file__).resolve().parent
START, END = "2020-12-01", "2026-09-21"
# Sessions worth keeping a full strip for. Each is a day the near meeting was live.
STRIP_DATES = ["2022-06-01", "2022-06-13", "2023-06-13", "2024-09-17", "2026-09-21"]

NY_FED = "https://markets.newyorkfed.org/api/rates/unsecured/effr/search.json"


def build_effr():
    j = requests.get(NY_FED, params={"startDate": START, "endDate": END}, timeout=60).json()
    e = pd.DataFrame(j["refRates"])
    e = e.rename(columns={"effectiveDate": "date", "percentRate": "effr",
                          "targetRateFrom": "target_low", "targetRateTo": "target_high"})
    e = e[["date", "effr", "target_low", "target_high"]].sort_values("date")
    e.to_csv(HERE / "effr.csv", index=False)
    return e


def build_expiry_settles():
    s = rates.settlements(START, END)
    zq = s[s["asset"] == "ZQ"].copy()
    zq["month"] = zq["expiration"].dt.tz_localize(None).dt.to_period("M")
    on_expiry = zq[zq["trade_date"].dt.normalize() == zq["expiration"].dt.normalize()]
    out = on_expiry.sort_values("published").groupby("month").tail(1)
    out = out.assign(
        expiry=out["expiration"].dt.tz_localize(None).dt.date,
        month=out["month"].astype(str),
    )[["month", "expiry", "symbol", "price", "is_final"]].sort_values("month")
    out.to_csv(HERE / "zq_expiry_settles.csv", index=False)
    return out


def build_strips():
    for day in STRIP_DATES:
        d = pd.Timestamp(day)
        s = rates.settlements(d, d + pd.Timedelta(days=4))
        zq = s[(s["asset"] == "ZQ") & (s["trade_date"].dt.date == d.date())].copy()
        if zq.empty:
            raise SystemExit(f"no ZQ settlements for session {day}")
        zq = zq.assign(
            month=zq["expiration"].dt.tz_localize(None).dt.to_period("M").astype(str),
        )[["month", "symbol", "price", "implied_rate"]].sort_values("month")
        zq.to_csv(HERE / f"zq_strip_{day}.csv", index=False)


if __name__ == "__main__":
    print(build_effr().tail(3).to_string(index=False))
    print(build_expiry_settles().tail(3).to_string(index=False))
    build_strips()
    print("wrote fixtures to", HERE)
