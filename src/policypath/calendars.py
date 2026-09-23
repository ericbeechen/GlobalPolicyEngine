import pandas as pd


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
