import pandas as pd
from dateutil.relativedelta import MO
from pandas.tseries.holiday import (AbstractHolidayCalendar, EasterMonday, GoodFriday, Holiday,
                                    USFederalHolidayCalendar, nearest_workday, next_monday,
                                    next_monday_or_tuesday, sunday_to_monday)
from pandas.tseries.offsets import CustomBusinessDay, DateOffset


class FedHolidayCalendar(AbstractHolidayCalendar):
    """US federal holidays as the Federal Reserve observes them.

    Same days as pandas' federal calendar, except a holiday falling on a
    Saturday is not moved to the Friday: the Fed stays open, and the NY Fed
    publishes rates, on e.g. 2021-06-18 and 2023-11-10.
    """
    rules = [
        Holiday(h.name, year=h.year, month=h.month, day=h.day, offset=h.offset,
                start_date=h.start_date, end_date=h.end_date,
                observance=sunday_to_monday if h.observance is nearest_workday else h.observance)
        for h in USFederalHolidayCalendar.rules
    ]


# Fed business days: the calendar FOMC effective dates and NY Fed fixings follow.
US_BDAY = CustomBusinessDay(calendar=FedHolidayCalendar())


def _one_off(name, day):
    d = pd.Timestamp(day)
    return Holiday(name, year=d.year, month=d.month, day=d.day)


class UKHolidayCalendar(AbstractHolidayCalendar):
    """England and Wales bank holidays: the days SONIA is not fixed and the Bank does not publish.

    A New Year's Day, Christmas or Boxing Day on a weekend moves to the next
    weekday. The early May holiday moved to Friday 8 May in 2020 (VE Day), the
    spring holiday to 4 June 2012 and 2 June 2022 (jubilees), and five days were
    one-offs. Checked against the SONIA fixing dates (`tests/test_gbp.py`).
    """
    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=next_monday),
        GoodFriday,
        EasterMonday,
        Holiday("Early May", month=5, day=1, offset=DateOffset(weekday=MO(1)), end_date="2019-12-31"),
        Holiday("Early May", month=5, day=1, offset=DateOffset(weekday=MO(1)), start_date="2021-01-01"),
        _one_off("Early May (VE Day)", "2020-05-08"),
        Holiday("Spring", month=5, day=31, offset=DateOffset(weekday=MO(-1)), end_date="2011-12-31"),
        Holiday("Spring", month=5, day=31, offset=DateOffset(weekday=MO(-1)),
                start_date="2013-01-01", end_date="2021-12-31"),
        Holiday("Spring", month=5, day=31, offset=DateOffset(weekday=MO(-1)), start_date="2023-01-01"),
        _one_off("Spring (Diamond Jubilee)", "2012-06-04"),
        _one_off("Spring (Platinum Jubilee)", "2022-06-02"),
        Holiday("Summer", month=8, day=31, offset=DateOffset(weekday=MO(-1))),
        Holiday("Christmas Day", month=12, day=25, observance=next_monday),
        Holiday("Boxing Day", month=12, day=26, observance=next_monday_or_tuesday),
        _one_off("Royal Wedding", "2011-04-29"),
        _one_off("Diamond Jubilee", "2012-06-05"),
        _one_off("Platinum Jubilee", "2022-06-03"),
        _one_off("State Funeral", "2022-09-19"),
        _one_off("Coronation", "2023-05-08"),
    ]


# London business days: SONIA fixings, the Bank of England's curves, MPC dates.
UK_BDAY = CustomBusinessDay(calendar=UKHolidayCalendar())

# By the name a currency's config gives in ``calendar``.
BDAYS = {"fed": US_BDAY, "uk": UK_BDAY}


class CMEHolidayCalendar(AbstractHolidayCalendar):
    """Days CME's interest-rate futures do not settle, among those the Fed may keep open.

    Good Friday, and every federal holiday with pandas' observance, which moves a
    Saturday holiday to the Friday as CME does (the Fed does not: `FedHolidayCalendar`).
    Not every day CME is shut: only the ones a coverage report must explain.
    """
    rules = [GoodFriday, *USFederalHolidayCalendar.rules]


# By the name a currency's config gives in ``market.exchange_calendar``.
EXCHANGES = {"cme": CMEHolidayCalendar}


def known_daily(fixings, as_of, bday):
    """The overnight rate for every calendar day already known at the end of `as_of`.

    `fixings` has ``date``, ``value`` and ``published``; only rows published by
    `as_of` count. A fixing for business day b is the rate for every calendar
    day until the next business day (weekends and holidays carry the last
    fixing, as in the ZQ and SR1 settlement rules), so once the latest fixing b
    is published, the days through next_bday(b) - 1 are known and nothing after
    is. On an ordinary session t that is t - 1: the NY Fed publishes t - 1's
    fixing on the morning of t, and t's own fixing only on the next business day.
    """
    f = fixings[fixings["published"] < pd.Timestamp(as_of).normalize() + pd.Timedelta(days=1)]
    if f.empty:
        raise ValueError(f"no fixings published by {pd.Timestamp(as_of).date()}")
    f = f.sort_values("date").drop_duplicates("date", keep="last").set_index("date")["value"]
    through = f.index[-1] + bday - pd.Timedelta(days=1)
    return f.reindex(pd.date_range(f.index[0], through, freq="D")).ffill()


def known_meetings(meetings, as_of):
    """The meeting calendar as it stood at the end of `as_of`.

    An unscheduled meeting counts from its announcement: before then nobody
    could price it, and handing it to the solver as a pillar is look-ahead. A
    scheduled meeting that was called off counts until it was, since the market
    priced a decision there. Scheduled meetings count on every date: the Fed
    publishes them about two years ahead, and the calendar does not record when.
    """
    as_of = pd.Timestamp(as_of)
    announced = meetings["scheduled"].astype(bool) | (meetings["announcement_date"] <= as_of)
    standing = meetings["cancelled"].isna() | (meetings["cancelled"] > as_of)
    return meetings[announced & standing]


def label_path(path, meetings):
    """Join a solved policy path to the meeting calendar.
    """
    announced = (meetings.set_index("effective_date")["announcement_date"]
                 .reindex(path.index))
    out = pd.DataFrame({
        "announced": announced.to_numpy(),
        "effective": path.index,
        "rate": path.to_numpy(),
        "move_bp": path.diff().to_numpy() * 100.0,
    })
    out.loc[out.index[0], "announced"] = pd.NaT
    return out.reset_index(drop=True)
