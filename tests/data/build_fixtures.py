"""Regenerate the committed test fixtures.

Not run by the test suite. It needs the Databento archive (paid, gitignored) and
the network, which is exactly what the fixtures exist to keep out of `pytest`.
Everything it writes is small enough to read in a diff.

    uv run --env-file .env python tests/data/build_fixtures.py
    uv run --env-file .env python tests/data/build_fixtures.py --market-only   # leave alfred/ as it is

Outputs, all in this directory:
  effr.csv              EFFR and prevailing target range from FRED, one row per
                        fixing, with the day the fixing was published. The range
                        is the one in force on `date`, public that same day.
  zq_expiry_settles.csv One row per expired ZQ contract: its settle on its own
                        expiry session, which must equal 100 - the realized
                        average EFFR for the contract month.
  sr1_expiry_settles.csv The same for SR1, against the average SOFR.
  zq_strip_<date>.csv   The whole ZQ strip on a few chosen sessions.
  sr1_strip_<date>.csv  The SR1 strip on the same sessions, where SR1 had listed (2018-05).
  sr3_strip_<date>.csv  The SR3 strip on the same sessions, for the curve tests, where
                        the archive has SR3 (2020-12-31 on).
  sofr.csv              SOFR from FRED, one row per fixing, with its publication day.
  sep.csv               The FOMC SEP median longer-run funds rate (FEDTARMDLR), one row
                        per SEP, published on its release day.

GBP, under boe/ and ons/ (--gbp-only rebuilds just these; needs the network, not the archive):
  boe/sonia.csv, boe/bank_rate.csv   IADB IUDSOIA and IUDBEDR from 2009, with publication days.
  boe/curve.csv         The Bank's OIS spot curve, 1-24 months, on the GBP_SESSIONS and on
                        every day of the three MPR windows the conditioning-path test reads.
  ons/<series>.csv      Every ONS vintage of the GBP macro series, reference dates from 2018.
  boe/conditioning_paths.csv is ground truth, built once by hand from the Bank's databank; not here.
"""

import pandas as pd
from policypath import config
from policypath.calendars import UK_BDAY
from policypath.sources import boe, fred, ons, rates

HERE = __import__("pathlib").Path(__file__).resolve().parent
START, END = "2010-06-01", "2026-09-21"
# Sessions worth keeping a full strip for. Each is a day the near meeting was live:
# the eve of liftoff in 2015, of the first cut in 2019, and the 2022-26 cycle.
STRIP_DATES = ["2015-12-15", "2019-07-30", "2022-06-01", "2022-06-13", "2023-06-13",
               "2024-09-17", "2026-09-21"]



ALFRED_START = "2018-01-01"  # reference dates kept; every vintage of each is kept


def build_alfred():
    """Every ALFRED vintage of the macro series, reference dates from 2018."""
    out = HERE / "alfred"
    out.mkdir(exist_ok=True)
    spec = config.currency("USD")["macro"]
    for series in [*spec["series"], *spec["validation"]]:
        fred.vintages(series, ALFRED_START).to_csv(out / f"{series}.csv", index=False, date_format="%Y-%m-%d")


def build_effr():
    e = fred.effr(START, END).rename(columns={"value": "effr"})
    for col, series in [("target_low", "DFEDTARL"), ("target_high", "DFEDTARU")]:
        t = fred.observations(series, START, END)[["date", "value"]]
        e = e.merge(t.rename(columns={"value": col}), on="date", how="left")
    e = e.assign(date=e["date"].dt.date, published=e["published"].dt.date)
    e = e[["date", "effr", "target_low", "target_high", "published"]].sort_values("date")
    e.to_csv(HERE / "effr.csv", index=False)
    return e


def settles_by_year(series, start, end):
    """`rates.settlements` for one root, a year of the archive at a time.

    Each year reads four days past its end, so the final of its last session,
    which lands in the next day's file, is still found.
    """
    frames = []
    for year in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
        lo = max(pd.Timestamp(start), pd.Timestamp(year, 1, 1))
        hi = min(pd.Timestamp(end), pd.Timestamp(year, 12, 31))
        try:
            s = rates.settlements(lo, hi + pd.Timedelta(days=4), series=series)
        except FileNotFoundError:  # the root had not listed yet
            continue
        day = s["trade_date"].dt.tz_localize(None).dt.normalize()
        frames.append(s[(day >= lo) & (day <= hi)])
    return pd.concat(frames, ignore_index=True)


def build_expiry_settles(series):
    s = settles_by_year(series, START, END)
    s["month"] = s["expiration"].dt.tz_localize(None).dt.to_period("M")
    on_expiry = s[s["trade_date"].dt.normalize() == s["expiration"].dt.normalize()]
    out = on_expiry.sort_values("published").groupby("month").tail(1)
    out = out.assign(
        expiry=out["expiration"].dt.tz_localize(None).dt.date,
        month=out["month"].astype(str),
    )[["month", "expiry", "symbol", "price", "is_final"]].sort_values("month")
    out.to_csv(HERE / f"{series.lower()}_expiry_settles.csv", index=False)
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

        for root in ["SR1", "SR3"]:
            sofr = s[(s["asset"] == root) & (s["trade_date"].dt.date == d.date())]
            if sofr.empty:  # not listed yet, or not in the archive
                continue
            pd.DataFrame({
                "contract": sofr["symbol"],
                "expiration": sofr["expiration"].dt.tz_convert(rates.LOCAL_TZ).dt.date,
                "price": sofr["price"],
            }).sort_values("expiration").to_csv(HERE / f"{root.lower()}_strip_{day}.csv", index=False)


def build_sofr():
    s = fred.observations("SOFR", START, END)
    s = s.assign(date=s["date"].dt.date, published=s["published"].dt.date)
    s[["date", "value", "published"]].to_csv(HERE / "sofr.csv", index=False)
    return s


def build_sep():
    rule = config.currency("USD")["rule"]
    lag = config.currency("USD")["fred_lags"][rule["rstar"]["series"]]
    s = fred.observations(rule["rstar"]["series"], START, END, lag_bdays=lag)
    s = s.assign(date=s["date"].dt.date, published=s["published"].dt.date)
    s[["date", "value", "published"]].to_csv(HERE / "sep.csv", index=False)
    return s


# GBP sessions with a whole curve kept: the day after the November 2021 hold, liftoff's eve,
# the mini-budget, the June 2023 50bp, the first cut, the latest.
GBP_SESSIONS = ["2021-11-05", "2022-03-01", "2022-09-27", "2023-06-23", "2024-08-02", "2025-06-02",
                "2026-09-24"]
# The MPR reports whose 15-day windows the conditioning-path test recomputes (windows read in each report).
MPR_WINDOWS = {"August 2019": "2019-07-24", "February 2023": "2023-01-24", "August 2024": "2024-07-22"}
CURVE_TENORS = 24
GBP_END = "2026-09-25"   # the Bank publishes to the day before; the archive-bound USD fixtures stop at END


def build_gbp():
    out = HERE / "boe"
    out.mkdir(exist_ok=True)
    cfg = config.currency("GBP")
    for series, name in [("IUDSOIA", "sonia"), ("IUDBEDR", "bank_rate")]:
        s = boe.BoeSeries(cfg["boe_lags"][series]).fetch(series, "2009-01-01", GBP_END)
        s.assign(date=s["date"].dt.date, published=s["published"].dt.date).to_csv(out / f"{name}.csv", index=False)
    days = set(pd.to_datetime(GBP_SESSIONS))
    for end in MPR_WINDOWS.values():
        days |= set(pd.date_range(end=end, periods=15, freq=UK_BDAY))
    curve = boe.BoeCurve().fetch("OIS_SPOT", "2009-01-01", "2026-12-31")
    curve = curve[curve["date"].isin(days) & (curve["tenor"] <= CURVE_TENORS)]
    curve.assign(date=curve["date"].dt.date, published=curve["published"].dt.date).to_csv(
        out / "curve.csv", index=False)
    macro = cfg["macro"]
    (HERE / "ons").mkdir(exist_ok=True)
    for series in [*macro["series"], *macro["validation"]]:
        ons.vintages(series, ALFRED_START).to_csv(HERE / "ons" / f"{series}.csv", index=False, date_format="%Y-%m-%d")
    return curve


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Regenerate the committed test fixtures.")
    parser.add_argument("--market-only", action="store_true",
                        help="skip ALFRED: rebuild the rates fixtures, leave tests/data/alfred/ as it is")
    parser.add_argument("--gbp-only", action="store_true", help="rebuild only the GBP fixtures (no archive needed)")
    args = parser.parse_args()
    if args.gbp_only:
        print(build_gbp().groupby("date").size().to_string())
        raise SystemExit
    print(build_effr().tail(3).to_string(index=False))
    print(build_sofr().tail(3).to_string(index=False))
    print(build_sep().tail(3).to_string(index=False))
    for series in ["ZQ", "SR1"]:
        print(build_expiry_settles(series).tail(3).to_string(index=False))
    if not args.market_only:
        build_alfred()
    build_strips()
    build_gbp()
    print("wrote fixtures to", HERE)

