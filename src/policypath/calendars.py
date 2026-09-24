import pandas as pd
from pandas.tseries.holiday import (AbstractHolidayCalendar, Holiday,
                                    USFederalHolidayCalendar, nearest_workday,
                                    sunday_to_monday)
from pandas.tseries.offsets import CustomBusinessDay


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


def next_meetings(as_of, meetings, n):
    """The next `n` meetings strictly after `as_of`, by announcement date."""
    upcoming = meetings[meetings["announcement_date"] > pd.Timestamp(as_of)]
    return upcoming.sort_values("announcement_date").head(n).reset_index(drop=True)
