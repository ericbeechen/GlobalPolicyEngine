

def require_published(df):
    """No series leaves `sources/` indexed by reference date alone.

    `df` needs a ``date`` column and a ``published`` column with no gaps, and
    nothing can be published before the day it describes.
    """
    if "published" not in df.columns or df["published"].isna().any():
        raise ValueError("observations must carry a publication date for every row")
    if (df["published"] < df["date"]).any():
        raise ValueError("an observation cannot be published before its reference date")
    return df
