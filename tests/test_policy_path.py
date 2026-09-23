"""Canary for the policy-path solver. Invariant 4: this passes on every commit.

Three layers, cheapest first.

1. **Algebra.** Build ZQ prices from a known path, solve, recover the path. Catches sign errors, off-by-one-month errors and bad day weighting in seconds.
2. **Identity.** An expired ZQ contract settles to 100 minus the realized average EFFR over its contract month. That is the contract's definition, not a modelling choice, so it is external ground truth in the strict sense -- and it exercises the whole chain (`sources/rates.py`, the contract-to-month mapping, the calendar-day
   weighting) against numbers nobody in this repo picked.
3. **Published pricing.** The solved path on a real session, checked against the realized EFFR it must start from, and against captured market pricing.

Every layer reads committed fixtures only, so the suite runs for anyone who clones the repo without the (paid, gitignored) Databento archive. Regenerate the fixtures with `uv run python tests/data/build_fixtures.py`.
"""

from pathlib import Path
import pandas as pd
import pytest
from policypath.curves.policy_path import implied_path

DATA = Path(__file__).parent / "data"
MEETINGS = Path(__file__).parents[1] / "config" / "meetings" / "fomc.csv"
BP = 0.01 
HORIZON_MONTHS = 12


def zq_prices_from_path(start_rate, changes, months):
    """changes: {effective_date: new_rate}. Returns {Period('YYYY-MM'): price}."""
    days = pd.date_range(months[0].start_time, months[-1].end_time, freq="D")
    rate = pd.Series(start_rate, index=days)
    for eff, r in sorted(changes.items()):
        rate[rate.index >= eff] = r
    avg = rate.groupby(rate.index.to_period("M")).mean()
    return 100.0 - avg


def path_at(start_rate, changes, dates):
    """The step function implied by `changes`, sampled at `dates`."""
    out = []
    for d in dates:
        r = start_rate
        for eff, new in sorted(changes.items()):
            if d >= eff:
                r = new
        out.append(r)
    return pd.Series(out, index=pd.DatetimeIndex(dates))


def months_between(first, last):
    return pd.period_range(first, last, freq="M")


@pytest.fixture(scope="module")
def effr():
    """Realized EFFR and prevailing target range, indexed by date."""
    e = pd.read_csv(DATA / "effr.csv", parse_dates=["date"]).set_index("date").sort_index()
    return e


@pytest.fixture(scope="module")
def effr_monthly_avg(effr):
    """Calendar-day average EFFR per month -- the ZQ settlement convention.

    Non-business days carry the previous business day's rate forward.
    """
    daily = effr["effr"].reindex(
        pd.date_range(effr.index[0], effr.index[-1], freq="D")
    ).ffill()
    full = daily.groupby(daily.index.to_period("M")).count()
    avg = daily.groupby(daily.index.to_period("M")).mean()
    # drop partial months at either end of the fixture
    return avg[full == full.index.days_in_month]


@pytest.fixture(scope="module")
def meetings():
    m = pd.read_csv(MEETINGS, parse_dates=["announcement_date", "effective_date"])
    return m


# --------------------------------------------------------------------------
# 1. algebra -- synthetic path in, same path out
# --------------------------------------------------------------------------

def test_recovers_a_single_hike():
    months = months_between("2022-06", "2022-12")
    start, changes = 0.83, {pd.Timestamp("2022-06-16"): 1.58}
    prices = zq_prices_from_path(start, changes, months)

    got = implied_path(100.0 - prices, pd.Series(list(changes)))

    expected = path_at(start, changes, got.index)
    pd.testing.assert_series_equal(got, expected, check_names=False, atol=1e-9)


def test_recovers_a_path_with_idle_meetings():
    """Pillars where nothing happened must come back unchanged, not smeared."""
    months = months_between("2024-08", "2025-06")
    start = 5.33
    changes = {pd.Timestamp("2024-09-19"): 4.83, pd.Timestamp("2024-12-19"): 4.58}
    pillars = [pd.Timestamp(d) for d in
               ["2024-09-19", "2024-11-08", "2024-12-19", "2025-01-30", "2025-03-20"]]
    prices = zq_prices_from_path(start, changes, months)

    got = implied_path(100.0 - prices, pd.Series(pillars))

    expected = path_at(start, changes, got.index)
    pd.testing.assert_series_equal(got, expected, check_names=False, atol=1e-9)


def test_recovers_two_moves_in_one_month():
    """March 2020 shape: an inter-meeting cut and a scheduled one, days apart."""
    months = months_between("2020-02", "2020-09")
    start = 1.58
    changes = {pd.Timestamp("2020-03-04"): 1.10, pd.Timestamp("2020-03-16"): 0.05}
    prices = zq_prices_from_path(start, changes, months)

    got = implied_path(100.0 - prices, pd.Series(list(changes)))

    expected = path_at(start, changes, got.index)
    pd.testing.assert_series_equal(got, expected, check_names=False, atol=1e-9)


def test_flat_path_when_no_meetings_fall_in_the_strip():
    months = months_between("2019-01", "2019-04")
    prices = zq_prices_from_path(2.40, {}, months)

    got = implied_path(100.0 - prices, pd.Series([], dtype="datetime64[ns]"))

    assert len(got) == 1
    assert got.iloc[0] == pytest.approx(2.40, abs=1e-9)


def test_raises_rather_than_guessing_when_underdetermined():
    """Two meetings inside the only contract month cannot be told apart."""
    months = months_between("2020-03", "2020-03")
    prices = zq_prices_from_path(1.58, {pd.Timestamp("2020-03-16"): 0.05}, months)
    pillars = pd.Series([pd.Timestamp("2020-03-04"), pd.Timestamp("2020-03-16")])

    with pytest.raises(ValueError, match="cannot identify"):
        implied_path(100.0 - prices, pillars)


def test_survives_tick_rounding():
    """ZQ trades in quarter-basis-point ticks, so the inputs are never exact."""
    months = months_between("2024-08", "2025-09")
    start = 5.33
    changes = {pd.Timestamp("2024-09-19"): 4.83, pd.Timestamp("2024-11-08"): 4.58,
               pd.Timestamp("2024-12-19"): 4.33, pd.Timestamp("2025-03-20"): 4.08}
    prices = (zq_prices_from_path(start, changes, months) / 0.0025).round() * 0.0025

    got = implied_path(100.0 - prices, pd.Series(list(changes)))

    expected = path_at(start, changes, got.index)
    assert (got - expected).abs().max() < 1 * BP


def test_expired_contracts_settle_to_realized_effr(effr_monthly_avg):
    """100 - settle == average realized EFFR over the contract month.

    This is the ZQ contract definition. A break here means the settlement feed,
    the contract-to-month mapping or the averaging convention is wrong, and every
    implied path in the repo is wrong with it.
    """
    settles = pd.read_csv(DATA / "zq_expiry_settles.csv")
    settles["month"] = settles["month"].apply(pd.Period)
    settles = settles[settles["month"].isin(effr_monthly_avg.index)]
    assert len(settles) > 24, "fixture too thin to be a meaningful check"

    implied = 100.0 - settles["price"].to_numpy()
    realized = effr_monthly_avg.loc[settles["month"]].to_numpy()
    err = pd.Series(implied - realized, index=settles["month"].to_numpy())

    # One tick is 0.25bp, and the expiry-session settle can predate the final
    # fixing of the month's last day or two. Anything beyond that is a real break.
    worst = err.abs().idxmax()
    assert err.abs().max() < 0.3 * BP, (
        f"worst month {worst}: implied {implied[list(err.index).index(worst)]:.4f} "
        f"vs realized {realized[list(err.index).index(worst)]:.4f}"
    )

def strip_sessions():
    return sorted(p.name.removeprefix("zq_strip_").removesuffix(".csv")
                  for p in DATA.glob("zq_strip_*.csv"))


def solve_session(day, meetings):
    """The implied path from the committed strip fixture for one session."""
    strip = pd.read_csv(DATA / f"zq_strip_{day}.csv")
    implied_avg = pd.Series(
        strip["implied_rate"].to_numpy(),
        index=pd.PeriodIndex(strip["month"], freq="M"),
    ).sort_index()
    horizon = pd.Timestamp(day).to_period("M") + HORIZON_MONTHS
    return implied_path(implied_avg[implied_avg.index <= horizon], meetings["effective_date"])


@pytest.mark.parametrize("day", strip_sessions())
def test_path_starts_at_the_prevailing_effr(day, meetings, effr):
    """The front regime is the rate in force today, which is published.

    The solver is told nothing about the current policy rate -- it backs the whole
    path out of futures prices -- so this catches a level error of any kind.
    """
    path = solve_session(day, meetings)
    prevailing = effr["effr"].asof(pd.Timestamp(day))

    assert path.iloc[0] == pytest.approx(prevailing, abs=3 * BP), (
        f"{day}: solved front rate {path.iloc[0]:.4f} vs realized EFFR {prevailing:.4f}"
    )


@pytest.mark.parametrize("day", strip_sessions())
def test_path_is_monotone_in_time_order_and_finite(day, meetings):
    path = solve_session(day, meetings)
    assert path.index.is_monotonic_increasing
    assert path.notna().all()
    assert (path.abs() < 25).all(), "a policy rate this size means the solve blew up"


def test_implied_meeting_moves_match_published_pricing(meetings):
    """Implied move at a meeting vs. market pricing captured on the day.
    `published_pricing.csv` is hand-captured: one row per (session, meeting) with the expected move in basis points implied by a published source (CME FedWatch or a contemporaneous desk note), plus where it came from. Populate it as you go -- a snapshot is hard to recover after the fact.
    """
    table = pd.read_csv(DATA / "published_pricing.csv", comment="#",
                        parse_dates=["session", "meeting"])
    if table.empty:
        pytest.skip("published_pricing.csv has no rows yet")

    errors = []
    for session, rows in table.groupby("session"):
        day = session.date().isoformat()
        path = solve_session(day, meetings)
        steps = path.diff().dropna() * 100.0
        for _, row in rows.iterrows():
            eff = meetings.loc[meetings["announcement_date"] == row["meeting"], "effective_date"]
            assert len(eff) == 1, f"{row['meeting'].date()} is not in fomc.csv"
            got = steps.get(eff.iloc[0])
            assert got is not None, f"no pillar at {eff.iloc[0].date()}"
            if abs(got - row["expected_move_bp"]) > row["tolerance_bp"]:
                errors.append(
                    f"{day} / {row['meeting'].date()}: implied {got:+.1f}bp vs "
                    f"published {row['expected_move_bp']:+.1f}bp ({row['source']})"
                )
    assert not errors, "\n".join(errors)
