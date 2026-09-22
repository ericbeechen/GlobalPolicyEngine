import re
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

# URL targeting the 2020s decade directory
url = "https://fraser.stlouisfed.org/title/federal-open-market-committee-meeting-minutes-transcripts-documents-677?browse=2020s"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}
out_path = Path(__file__).with_name("fomc.csv")

MONTHS = {
    m: i
    for i, m in enumerate(
        ["january", "february", "march", "april", "may", "june", "july",
         "august", "september", "october", "november", "december"],
        start=1,
    )
}
# FRASER dates some meetings by when they were held, not when the decision was
# announced. Map the FRASER date to the actual announcement date.
ANNOUNCEMENT_OVERRIDES = {
    # Conference call held Mar 2; 50bp cut announced Mar 3.
    pd.Timestamp("2020-03-02"): pd.Timestamp("2020-03-03"),
}
# Policy changes take effect the US business day after the announcement.
us_bday = CustomBusinessDay(calendar=USFederalHolidayCalendar())


def parse_slug(slug):
    """Return (announcement_date, tags) from a FRASER slug body.

    e.g. "january-31-february-1-2023" -> (2023-02-01, [])
         "march-15-2020-unscheduled"  -> (2020-03-15, ["unscheduled"])
    The announcement is the last day of the meeting.
    """
    parts = slug.lower().split("-")
    year_idx = next(i for i, p in enumerate(parts) if re.fullmatch(r"\d{4}", p))
    year = int(parts[year_idx])
    tags = parts[year_idx + 1:]

    month = day = None
    for p in parts[:year_idx]:
        if p in MONTHS:
            month = MONTHS[p]
        elif p.isdigit():
            day = int(p)
    return pd.Timestamp(year, month, day), tags


try:
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    meetings_data = []

    for item in soup.find_all("a", href=True):
        match = re.search(
            r"/(?:meeting|conference|telephone-conference)-(.*?)-\d+$",
            item["href"],
            re.IGNORECASE,
        )
        if not match:
            continue

        announcement_date, tags = parse_slug(match.group(1))
        announcement_date = ANNOUNCEMENT_OVERRIDES.get(
            announcement_date, announcement_date
        )

        # Cancelled meetings and notation votes carry no rate decision.
        if "cancelled" in tags or "notation" in tags:
            continue

        meetings_data.append(
            {
                "announcement_date": announcement_date,
                "effective_date": announcement_date + us_bday,
                "scheduled": "unscheduled" not in tags,
            }
        )

    if meetings_data:
        df = (
            pd.DataFrame(meetings_data)
            .drop_duplicates(subset=["announcement_date"])
            .sort_values("announcement_date")
        )
        df.to_csv(out_path, index=False, date_format="%Y-%m-%d")
        print(df.to_string(index=False))
    else:
        print("No meetings found. The page layout may have changed entirely.")

except requests.exceptions.RequestException as e:
    print(f"Network error pulling data from FRASER: {e}")
