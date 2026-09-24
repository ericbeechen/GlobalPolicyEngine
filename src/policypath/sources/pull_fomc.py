import re
from pathlib import Path
import pandas as pd
import requests
from bs4 import BeautifulSoup
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

url = "https://fraser.stlouisfed.org/title/federal-open-market-committee-meeting-minutes-transcripts-documents-677?browse=2020s"
FED_CALENDAR = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}
out_path = Path(__file__).resolve().parents[3] / "config" / "meetings" / "fomc.csv"

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


def scheduled_from_fed_calendar():
    """Scheduled meetings from the Fed's own calendar page, including future ones.

    FRASER indexes minutes and transcripts, so it only ever knows about meetings
    that have already happened. Every meeting the market is currently pricing is
    therefore missing from it, and without those pillars the path solver has
    nothing to solve for. The Fed publishes the calendar about two years ahead.

    Dates render as ('January', '28-29'), ('Jan/Feb', '31-1') for a meeting that
    straddles month end, and ('March', '18-19*') where the asterisk marks a
    Summary of Economic Projections. The announcement is the last day.
    """
    r = requests.get(FED_CALENDAR, headers=headers)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    for panel in soup.select("div.panel.panel-default"):
        heading = panel.select_one(".panel-heading")
        year_match = re.search(r"(\d{4})", heading.get_text() if heading else "")
        if not year_match:
            continue
        year = int(year_match.group(1))

        for block in panel.select("div.fomc-meeting"):
            month_el = block.select_one(".fomc-meeting__month")
            date_el = block.select_one(".fomc-meeting__date")
            if not (month_el and date_el):
                continue
            raw_date = date_el.get_text(strip=True)
            # "22 (notation vote)" carries no rate decision, same as in FRASER.
            if "notation" in raw_date.lower() or "cancel" in raw_date.lower():
                continue

            months = [m.strip() for m in month_el.get_text(strip=True).split("/")]
            days = re.findall(r"\d+", raw_date)
            if not days:
                continue
            # last day of the meeting, in the later month where it straddles one
            month_name = months[-1].lower()
            month = next(n for m, n in MONTHS.items() if m.startswith(month_name[:3]))
            day = int(days[-1])
            # "31-1" rolls into January of the next year
            cal_year = year + 1 if len(months) > 1 and month == 1 else year

            announcement = pd.Timestamp(cal_year, month, day)
            yield {
                "announcement_date": announcement,
                "effective_date": announcement + us_bday,
                "scheduled": True,
            }


try:
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    meetings_data = list(scheduled_from_fed_calendar())

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
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False, date_format="%Y-%m-%d")
        print(df.to_string(index=False))
    else:
        print("No meetings found. The page layout may have changed entirely.")

except requests.exceptions.RequestException as e:
    print(f"Network error pulling data from FRASER: {e}")
