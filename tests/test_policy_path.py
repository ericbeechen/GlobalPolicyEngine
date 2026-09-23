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

FEDWATCH = DATA / "fedwatch"


def strip_sessions():
    """Every session with a committed strip, from either source."""
    databento = (p.name.removeprefix("zq_strip_").removesuffix(".csv")
                 for p in DATA.glob("zq_strip_*.csv"))
    return sorted({*databento, *fedwatch_sessions()})


def fedwatch_sessions():
    return sorted(p.name.removesuffix("_futures.csv")
                  for p in FEDWATCH.glob("*_futures.csv"))


def load_strip(day):
    """Implied average rate per contract month, from whichever fixture exists.

    A FedWatch capture wins over the Databento settle for the same session: it is
    paired with the probabilities measured against it, so using it removes the
    timing drift between the two.
    """
    fw = FEDWATCH / f"{day}_futures.csv"
    if fw.exists():
        strip = pd.read_csv(fw, comment="#")
        rates = 100.0 - strip["price"].to_numpy()
    else:
        strip = pd.read_csv(DATA / f"zq_strip_{day}.csv")
        rates = strip["implied_rate"].to_numpy()
    return pd.Series(rates, index=pd.PeriodIndex(strip["month"], freq="M")).sort_index()


def solve_session(day, meetings):
    """The implied path from the committed strip fixture for one session."""
    implied_avg = load_strip(day)
    horizon = pd.Timestamp(day).to_period("M") + HORIZON_MONTHS
    return implied_path(implied_avg[implied_avg.index <= horizon], meetings["effective_date"])


@pytest.mark.parametrize("day", strip_sessions())
def test_path_matches_the_prevailing_effr(day, meetings, effr):
    """The regime covering today holds the rate in force today, which is published.

    The solver is told nothing about the current policy rate -- it backs the whole
    path out of futures prices -- so this catches a level error of any kind. Note
    it is the regime containing `day`, not the first one: on 2026-09-22 the front
    contract month already contains a meeting, so the first regime is September's
    *pre*-meeting rate and only the second one is current.
    """
    path = solve_session(day, meetings)
    prevailing = effr["effr"].asof(pd.Timestamp(day))

    assert path.asof(pd.Timestamp(day)) == pytest.approx(prevailing, abs=3 * BP), (
        f"{day}: solved {path.asof(pd.Timestamp(day)):.4f} vs realized EFFR {prevailing:.4f}"
    )


@pytest.mark.parametrize("day", strip_sessions())
def test_regimes_already_in_the_past_match_realized_effr(day, meetings, effr):
    """A regime that has entirely happened is no longer a forecast.

    Its rate is published, so the solver must reproduce it from futures alone.
    Tolerance is 2bp rather than a fraction of one because elapsed days of the
    front contract are still solved as unknowns -- see the decisions log. Tighten
    this once they are pinned to realized fixings.
    """
    path = solve_session(day, meetings)
    as_of = pd.Timestamp(day)
    daily = effr["effr"].reindex(
        pd.date_range(effr.index[0], effr.index[-1], freq="D")
    ).ffill()

    spans = list(zip(path.index, [*path.index[1:], None]))
    checked = 0
    for start, end in spans:
        if end is None or end > as_of:
            continue  # still partly in the future
        realized = daily.loc[start:end - pd.Timedelta(days=1)].mean()
        assert path[start] == pytest.approx(realized, abs=2 * BP), (
            f"{day}: regime {start.date()}..{(end - pd.Timedelta(days=1)).date()} "
            f"solved {path[start]:.4f} vs realized {realized:.4f}"
        )
        checked += 1
    if not checked:
        pytest.skip("no regime has fully elapsed in this strip")


@pytest.mark.parametrize("day", strip_sessions())
def test_path_is_monotone_in_time_order_and_finite(day, meetings):
    path = solve_session(day, meetings)
    assert path.index.is_monotonic_increasing
    assert path.notna().all()
    assert (path.abs() < 25).all(), "a policy rate this size means the solve blew up"


@pytest.mark.parametrize("day", strip_sessions())
def test_path_reprices_the_strip_it_was_solved_from(day, meetings):
    """Average the solved step function back over each month and recover the input.

    Deliberately loose. The residual is not solver error -- the system is
    overdetermined (13 contracts, 10 pillars) and the shortfall is the
    piecewise-constant-between-meetings assumption itself: expected average EFFR
    is an expectation over many paths, not one step function, so priced
    inter-meeting risk shows up here. It peaks near 2.3bp on 2022-06-13, days
    before the CPI print that forced a 75bp move, which is the assumption
    straining exactly where you would expect it to.

    So this is a regression guard, not a precision check. A single mistyped price
    moves the fit by less than the residual and will not be caught here.
    """
    avg = load_strip(day)
    avg = avg[avg.index <= pd.Timestamp(day).to_period("M") + HORIZON_MONTHS]
    path = solve_session(day, meetings)

    days = pd.date_range(avg.index[0].start_time, avg.index[-1].end_time.normalize(), freq="D")
    step = pd.Series(float("nan"), index=days)
    for pillar in path.index:
        step[step.index >= pillar] = path[pillar]
    assert step.notna().all(), "the solved path does not cover the whole strip"

    resid = (step.groupby(step.index.to_period("M")).mean() - avg).abs() * 100
    assert resid.max() < 3.0, f"{day}: worst month {resid.idxmax()} off {resid.max():.2f}bp"


def fedwatch_expected_midpoint(day):
    """Probability-weighted target midpoint at each meeting, from a capture."""
    pr = pd.read_csv(FEDWATCH / f"{day}_probabilities.csv", comment="#",
                     parse_dates=["meeting"])
    buckets = [c for c in pr.columns if c != "meeting"]
    mids = pd.Series({c: sum(int(x) for x in c.split("-")) / 200.0 for c in buckets})
    weights = pr[buckets].fillna(0.0) / 100.0

    # A mistyped cell almost always breaks the row sum, so check before using it.
    assert weights.sum(axis=1).sub(1.0).abs().max() < 0.002, (
        f"{day}: a probability row does not sum to 100%, check the transcription"
    )
    return pd.Series((weights * mids).sum(axis=1).to_numpy(), index=pr["meeting"])


@pytest.mark.parametrize("day", fedwatch_sessions())
def test_implied_meeting_moves_match_fedwatch(day, meetings, effr):
    """Per-meeting implied move against CME FedWatch, on FedWatch's own inputs.

    Compared as *moves* rather than levels, which is what makes this well posed:
    the solver produces EFFR and FedWatch produces target-range midpoints, and the
    two differ by an operating spread that is a nuisance in levels but cancels in
    differences.

    FedWatch reports probabilities to 0.1% over 25bp buckets, so its own expected
    value carries a few tenths of a basis point of quantization noise. 1.5bp is
    therefore about as tight as this can be; the fixture is a frozen snapshot, so
    there is no flakiness to absorb, only genuine methodology difference.
    """
    path = solve_session(day, meetings)
    steps = path.diff().dropna() * 100.0
    eff = meetings.set_index("announcement_date")["effective_date"]

    prevailing_mid = round(effr["effr"].asof(pd.Timestamp(day)) * 8) / 8  # nearest 12.5bp
    expected = fedwatch_expected_midpoint(day)
    fw_moves = expected.diff()
    fw_moves.iloc[0] = expected.iloc[0] - prevailing_mid
    fw_moves *= 100.0

    errors, compared = [], []
    for meeting, want in fw_moves.items():
        assert meeting in eff.index, f"{meeting.date()} is missing from fomc.csv"
        got = steps.get(eff[meeting])
        if got is None:
            # Beyond the last contract, so there is no pillar to solve for. The
            # solver is right to drop it; the strip simply does not reach.
            assert eff[meeting] > path.index[-1]
            continue
        compared.append(abs(got - want))
        if abs(got - want) > 1.5:
            errors.append(f"  {meeting.date()}: solver {got:+.2f}bp vs "
                          f"FedWatch {want:+.2f}bp ({got - want:+.2f})")

    assert not errors, f"{day} disagrees with FedWatch:\n" + "\n".join(errors)
    assert len(compared) >= 8, f"{day}: only {len(compared)} meetings compared"
    # A systematic bias shows up in the mean long before it breaks any single meeting.
    assert sum(compared) / len(compared) < 0.7, (
        f"{day}: mean absolute error {sum(compared) / len(compared):.2f}bp"
    )
