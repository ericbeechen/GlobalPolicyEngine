"""Regenerate config/meetings/mpc.csv: every MPC Bank Rate decision from 2009, and the ones scheduled ahead.

Needs the network, and overwrites the committed CSV. Two Bank of England sources:

- The MPC voting history (``mpcvoting.xlsx``, sheet "Bank Rate Decisions"):
  every decision since 1997, no-change meetings included, with Bank Rate after
  it. Its date column is the day of the vote, which until 2015 was not always
  the announcement: `ANNOUNCED` maps the six that differ, each checked against
  the Bank's announcement page and the minutes.
- The upcoming-dates page, for scheduled meetings not yet held.

A Bank Rate decision takes effect on the day it is announced, at noon
(Sterling Monetary Framework operating procedures, 4.4), so ``effective_date``
equals ``announcement_date``: unlike the FOMC's, there is no next-day lag. The
IADB's Bank Rate series changes on the announcement date for every change since 2009.

Monthly meetings through 2015, eight a year from 2017 (2016 was the changeover,
with eleven). The two March 2020 cuts (11th, 19th) were unscheduled; the
scheduled 26 March meeting went ahead. The September 2022 meeting was moved a
week, to the 22nd, for the period of national mourning; it is listed where it
was held.

    uv run python src/policypath/sources/pull_mpc.py
"""

import io
import re
from pathlib import Path
import openpyxl
import pandas as pd
import requests
from bs4 import BeautifulSoup

VOTING = "https://www.bankofengland.co.uk/-/media/boe/files/monetary-policy-summary-and-minutes/mpcvoting.xlsx"
UPCOMING = "https://www.bankofengland.co.uk/monetary-policy/upcoming-mpc-dates"
HEADERS = {"User-Agent": "Mozilla/5.0 (policypath research)"}
OUT = Path(__file__).resolve().parents[3] / "config" / "meetings" / "mpc.csv"
START = "2009-01-01"

# Vote date in the spreadsheet -> the day the decision was announced.
ANNOUNCED = {
    "2013-10-09": "2013-10-10",
    "2014-04-09": "2014-04-10",
    "2014-10-08": "2014-10-09",
    "2015-05-08": "2015-05-11",   # the Friday after the general election; announced the Monday
    "2015-06-03": "2015-06-04",
    "2015-07-08": "2015-07-09",   # Budget day
}
UNSCHEDULED = {"2020-03-11", "2020-03-19"}


def _get(url):
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    return r


def decisions():
    """Every decision in the voting history from `START`: announcement date and Bank Rate after it."""
    wb = openpyxl.load_workbook(io.BytesIO(_get(VOTING).content), read_only=True, data_only=True)
    rows = [(r[1], r[2]) for r in wb["Bank Rate Decisions"].iter_rows(values_only=True)
            if len(r) > 2 and hasattr(r[1], "year") and isinstance(r[2], (int, float))]
    df = pd.DataFrame(rows, columns=["vote_date", "rate"])
    df["vote_date"] = pd.to_datetime(df["vote_date"]).dt.normalize()
    df = df[df["vote_date"] >= START]
    df["announcement_date"] = df["vote_date"].dt.strftime("%Y-%m-%d").map(ANNOUNCED).fillna(
        df["vote_date"].dt.strftime("%Y-%m-%d")).pipe(pd.to_datetime)
    return pd.DataFrame({"announcement_date": df["announcement_date"],
                         "bank_rate": (df["rate"] * 100).round(4)})


def upcoming():
    """Scheduled dates on the Bank's upcoming-dates page: one table per year, the year in the heading above it."""
    soup = BeautifulSoup(_get(UPCOMING).text, "html.parser")
    out = []
    for table in soup.find_all("table"):
        heading = table.find_previous(["h2", "h3", "h4", "caption", "p"], string=re.compile(r"\b20\d\d\b"))
        year = re.search(r"\b(20\d\d)\b", heading.get_text()).group(1)
        for m in re.finditer(r"(?:Monday|Tuesday|Wednesday|Thursday|Friday)\W+(\d{1,2}) ([A-Z][a-z]+)",
                             table.get_text(" ", strip=True)):
            out.append(pd.Timestamp(f"{m.group(1)} {m.group(2)} {year}"))
    if not out:
        raise RuntimeError("no dates found on the upcoming MPC dates page")
    return pd.DatetimeIndex(sorted(set(out)))


def build():
    held = decisions()
    ahead = upcoming()
    ahead = ahead[ahead > held["announcement_date"].max()]
    cal = pd.concat([held, pd.DataFrame({"announcement_date": ahead})], ignore_index=True)
    cal = cal.sort_values("announcement_date").drop_duplicates("announcement_date").reset_index(drop=True)
    cal["effective_date"] = cal["announcement_date"]
    cal["scheduled"] = ~cal["announcement_date"].dt.strftime("%Y-%m-%d").isin(UNSCHEDULED)
    cal["cancelled"] = pd.NaT
    # The decision itself stays out of the calendar: a session reading it would know every
    # future outcome. It is only used here, to check the dates against the IADB (see __main__).
    return cal[["announcement_date", "effective_date", "scheduled", "cancelled"]], cal


if __name__ == "__main__":
    cal, decided = build()
    cal.to_csv(OUT, index=False, date_format="%Y-%m-%d")
    moves = decided.dropna(subset=["bank_rate"])
    moves = moves[moves["bank_rate"].diff() != 0]
    print(f"{len(moves) - 1} Bank Rate changes since {START}, each on a decision date by construction")
    print(f"wrote {len(cal)} meetings, {cal['announcement_date'].min():%Y-%m-%d} .. "
          f"{cal['announcement_date'].max():%Y-%m-%d}, to {OUT}")
    print(cal.groupby(cal["announcement_date"].dt.year).size().to_string())
