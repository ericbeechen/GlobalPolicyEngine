from pathlib import Path
import pandas as pd
import pytest
from policypath import config
from policypath.calendars import US_BDAY, known_daily, known_meetings
from policypath.curves.policy_path import implied_path

DATA = Path(__file__).parent / "data"
BP = 0.01 
HORIZON_MONTHS = 12


def daily_path(start_rate, changes, months):
    """The overnight rate on every calendar day of `months`, from {effective_date: new_rate}."""
    days = pd.date_range(months[0].start_time, months[-1].end_time.normalize(), freq="D")
    rate = pd.Series(start_rate, index=days)
    for eff, r in sorted(changes.items()):
        rate[rate.index >= eff] = r
    return rate


def zq_prices_from_path(start_rate, changes, months, daily=None):
    """changes: {effective_date: new_rate}. Returns {Period('YYYY-MM'): price}.

    `daily` overrides the step function on the days it covers, for building
    prices off a realized history that is not a clean step.
    """
    rate = daily_path(start_rate, changes, months)
    if daily is not None:
        rate.loc[daily.index] = daily
    avg = rate.groupby(rate.index.to_period("M")).mean()
    return 100.0 - avg


def realized_before(start_rate, changes, months, as_of):
    """The step path on every day before `as_of`: what is already known on `as_of`."""
    rate = daily_path(start_rate, changes, months)
    return rate[rate.index < pd.Timestamp(as_of)]


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
    """Realized EFFR, prevailing target range and publication date, indexed by date."""
    e = pd.read_csv(DATA / "effr.csv", parse_dates=["date", "published"]).set_index("date").sort_index()
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
    return config.meetings("USD")


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


# --------------------------------------------------------------------------
# 1b. algebra, mid-month -- elapsed days are known, not solved for
# --------------------------------------------------------------------------

JUL23_PILLARS = pd.Series(pd.to_datetime(
    ["2023-07-27", "2023-09-21", "2023-11-02", "2023-12-14", "2024-02-01"]))


def test_mid_month_path_starts_on_the_first_unknown_day():
    months = months_between("2023-07", "2024-03")
    start, changes = 5.08, {pd.Timestamp("2023-07-27"): 5.33}
    as_of = pd.Timestamp("2023-07-12")
    prices = zq_prices_from_path(start, changes, months)

    got = implied_path(100.0 - prices, JUL23_PILLARS,
                       realized=realized_before(start, changes, months, as_of))

    assert got.index[0] == as_of
    expected = path_at(start, changes, got.index)
    pd.testing.assert_series_equal(got, expected, check_names=False, atol=1e-9)


def test_evaluation_date_after_a_meeting_in_the_same_month():
    """On 28 July the hike of the 27th is history: July's forward part is all new rate."""
    months = months_between("2023-07", "2024-03")
    start, changes = 5.08, {pd.Timestamp("2023-07-27"): 5.33}
    as_of = pd.Timestamp("2023-07-28")
    prices = zq_prices_from_path(start, changes, months)

    got = implied_path(100.0 - prices, JUL23_PILLARS,
                       realized=realized_before(start, changes, months, as_of))

    assert got.index[0] == as_of
    assert pd.Timestamp("2023-07-27") not in got.index
    assert got.iloc[0] == pytest.approx(5.33, abs=1e-9)
    expected = path_at(start, changes, got.index)
    pd.testing.assert_series_equal(got, expected, check_names=False, atol=1e-9)


def test_realized_days_absorb_what_a_step_function_cannot():
    """Realized fixings wander inside the range; the forward path does not see that.

    Build July's price off a first half that printed a basis point soft. Solved
    from futures alone, that leaks into the current regime. With the realized
    days substituted, the forward path comes back exact.
    """
    months = months_between("2023-07", "2024-03")
    start, changes = 5.08, {pd.Timestamp("2023-07-27"): 5.33}
    as_of = pd.Timestamp("2023-07-15")
    known = realized_before(start, changes, months, as_of) - 0.01
    prices = zq_prices_from_path(start, changes, months, daily=known)

    blind = implied_path(100.0 - prices, JUL23_PILLARS)
    got = implied_path(100.0 - prices, JUL23_PILLARS, realized=known)

    assert abs(blind.iloc[0] - start) > 0.1 * BP, "the noise should have leaked in"
    expected = path_at(start, changes, got.index)
    pd.testing.assert_series_equal(got, expected, check_names=False, atol=1e-9)


def test_front_contract_dropped_near_expiry_anchors_on_the_next():
    months = months_between("2023-07", "2024-03")
    start, changes = 5.08, {pd.Timestamp("2023-07-27"): 5.33}
    as_of = pd.Timestamp("2023-07-29")  # 29, 30, 31 left
    prices = zq_prices_from_path(start, changes, months)
    # Garbage in the front contract must not reach the path once it is dropped.
    prices[pd.Period("2023-07")] += 0.05

    got = implied_path(100.0 - prices, JUL23_PILLARS, min_forward_days=5,
                       realized=realized_before(start, changes, months, as_of))

    assert got.attrs["front_dropped"] and not got.attrs["pinned"]
    expected = path_at(start, changes, got.index)
    pd.testing.assert_series_equal(got, expected, check_names=False, atol=1e-9)


def test_first_regime_pinned_when_its_only_contract_is_dropped():
    """July 2025: decision announced the 30th, effective the 31st.

    On the 29th only three July days are unknown, so July is dropped -- and with
    it the only contract that sees the 29th and 30th. That regime is pinned to
    the last known fixing; the post-meeting rate still comes from August.
    """
    months = months_between("2025-07", "2026-02")
    start, changes = 4.33, {pd.Timestamp("2025-07-31"): 4.08}
    pillars = pd.Series(pd.to_datetime(["2025-07-31", "2025-09-18", "2025-10-30", "2025-12-11"]))
    as_of = pd.Timestamp("2025-07-29")
    known = realized_before(start, changes, months, as_of)
    known.iloc[-1] = 4.31  # a stray last fixing, so pinning is visible
    prices = zq_prices_from_path(start, changes, months, daily=known)

    got = implied_path(100.0 - prices, pillars, realized=known, min_forward_days=5)

    assert got.attrs["pinned"]
    assert got.iloc[0] == 4.31
    assert got[pd.Timestamp("2025-07-31")] == pytest.approx(4.08, abs=1e-9)


def test_short_first_regime_pinned_rather_than_amplified():
    """17 September 2024: the cut is effective on the 19th, two days away.

    Only September sees those two days, at 2/30 weight, so half a basis point of
    price noise in September becomes 7.5bp in the current rate and in the size
    of the cut. Pinned to the last fixing, the cut comes back from October.
    """
    months = months_between("2024-09", "2025-06")
    start, changes = 5.33, {pd.Timestamp("2024-09-19"): 4.83}
    pillars = pd.Series(pd.to_datetime(["2024-09-19", "2024-11-08", "2024-12-19",
                                        "2025-01-30", "2025-03-20", "2025-05-08"]))
    as_of = pd.Timestamp("2024-09-17")
    known = realized_before(start, changes, months, as_of)
    prices = zq_prices_from_path(start, changes, months)
    prices[pd.Period("2024-09")] += 0.005  # half a basis point

    free = implied_path(100.0 - prices, pillars, realized=known)
    got = implied_path(100.0 - prices, pillars, realized=known, min_regime_days=5)

    assert abs(free.iloc[0] - start) > 5 * BP, "the noise should have been amplified"
    assert got.attrs["pinned"] and got.iloc[0] == start
    assert got[pd.Timestamp("2024-09-19")] == pytest.approx(4.83, abs=0.5 * BP)


def test_meeting_announced_at_month_end_steps_in_the_next_contract():
    """Announced 31 January 2024, effective 1 February: January never sees the new rate."""
    months = months_between("2024-01", "2024-08")
    start, changes = 5.33, {pd.Timestamp("2024-02-01"): 5.08}
    pillars = pd.Series(pd.to_datetime(["2024-02-01", "2024-03-21", "2024-05-02", "2024-06-13"]))
    prices = zq_prices_from_path(start, changes, months)
    assert 100.0 - prices[pd.Period("2024-01")] == pytest.approx(start, abs=1e-12)

    got = implied_path(100.0 - prices, pillars,
                       realized=realized_before(start, changes, months, "2024-01-17"))

    expected = path_at(start, changes, got.index)
    pd.testing.assert_series_equal(got, expected, check_names=False, atol=1e-9)


def test_realized_with_a_gap_is_refused():
    months = months_between("2023-07", "2024-03")
    known = realized_before(5.08, {}, months, "2023-07-12").drop(pd.Timestamp("2023-07-04"))
    prices = zq_prices_from_path(5.08, {}, months)
    with pytest.raises(ValueError, match="no gaps"):
        implied_path(100.0 - prices, JUL23_PILLARS, realized=known)


# --------------------------------------------------------------------------
# 1c. the publication boundary -- on date t, fixings are known through t - 1
# --------------------------------------------------------------------------

def fixings_for(first, last):
    """One fixing per Fed business day, valued by date so each is recognisable,
    published the next business day as the NY Fed does."""
    dates = pd.date_range(first, last, freq=US_BDAY)
    return pd.DataFrame({"date": dates, "value": dates.day / 100.0,
                         "published": [d + US_BDAY for d in dates]})


@pytest.mark.parametrize("as_of, known_through, carried", [
    ("2023-10-04", "2023-10-03", "2023-10-03"),  # ordinary Wednesday: Tuesday's fixing
    ("2023-10-02", "2023-10-01", "2023-09-29"),  # Monday: Friday's fixing covers the weekend
    ("2023-10-09", "2023-10-05", "2023-10-05"),  # Columbus Day: Friday is published Tuesday
    ("2023-10-10", "2023-10-09", "2023-10-06"),  # and then covers Friday through the holiday
])
def test_fixings_known_through_the_day_before(as_of, known_through, carried):
    fixings = fixings_for("2023-09-01", "2023-10-31")
    daily = known_daily(fixings, as_of)
    assert daily.index[-1] == pd.Timestamp(known_through)
    assert daily.iloc[-1] == pd.Timestamp(carried).day / 100.0


def unseen_move(daily, months, expiries):
    """Per contract, how far the fixings its expiry session could not see moved, in the month average.

    The session settles before its own day's fixing is published (the next
    morning), so the fixings dated from the expiry day to the month end are
    unseen. Each counts |fixing - last published fixing| / days in month.
    """
    out = []
    for month, expiry in zip(months, pd.to_datetime(expiries)):
        last_seen = daily[:expiry - pd.Timedelta(days=1)].iloc[-1]
        unseen = daily[expiry:month.end_time.normalize()]
        out.append((unseen - last_seen).abs().sum() / month.days_in_month)
    return pd.Series(out, index=months)


def test_expired_contracts_settle_to_realized_effr(effr, effr_monthly_avg):
    """100 - settle == average realized EFFR over the contract month.

    This is the ZQ contract definition. A break here means the settlement feed,
    the contract-to-month mapping or the averaging convention is wrong, and every
    implied path in the repo is wrong with it.

    The archive holds the expiry session's settle, not CME's final settlement,
    and that session has not seen the fixings of its own day onward. EFFR moves
    at month and quarter ends, by several basis points before 2016, so those
    days are allowed for: one tick (0.25bp) plus how far they moved from the
    last fixing the session saw. That still fails 128 of 193 months with the
    contract mapped one month off, and 20 with business-day instead of
    calendar-day averaging. Tightest case: ZQH8, a quarter end on Good Friday,
    0.363bp against 0.365bp.
    """
    settles = pd.read_csv(DATA / "zq_expiry_settles.csv")
    settles["month"] = settles["month"].apply(pd.Period)
    settles = settles[settles["month"].isin(effr_monthly_avg.index)]
    assert len(settles) > 150, "fixture too thin to be a meaningful check"

    implied = 100.0 - settles["price"].to_numpy()
    realized = effr_monthly_avg.loc[settles["month"]].to_numpy()
    err = pd.Series(implied - realized, index=settles["month"].to_numpy())
    daily = effr["effr"].reindex(pd.date_range(effr.index[0], effr.index[-1], freq="D")).ffill()
    allowed = 0.3 * BP + unseen_move(daily, err.index, settles["expiry"])

    excess = err.abs() - allowed
    worst = excess.idxmax()
    assert excess.max() < 0, (
        f"worst month {worst}: implied {implied[list(err.index).index(worst)]:.4f} "
        f"vs realized {realized[list(err.index).index(worst)]:.4f}, "
        f"allowed {allowed[worst] / BP:.2f}bp"
    )
    # Where nothing unseen moved, the settle is the average to a tick.
    calm = allowed <= 0.3 * BP + 1e-12
    assert calm.sum() > 60 and err[calm].abs().max() < 0.3 * BP  # 84 of 194 months

FEDWATCH = DATA / "fedwatch"


def read_fedwatch(path, **kwargs):
    """Read a FedWatch capture.

    Every capture opens with a one-line provenance note in plain prose, not a
    commented block, so it is skipped by position: the header is always line two.
    A capture saved without that note will fail loudly on a missing column here
    rather than silently mis-parse.
    """
    return pd.read_csv(path, skiprows=1, **kwargs)


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
        strip = read_fedwatch(fw)
        rates = 100.0 - strip["price"].to_numpy()
    else:
        strip = pd.read_csv(DATA / f"zq_strip_{day}.csv")
        rates = strip["implied_rate"].to_numpy()
    return pd.Series(rates, index=pd.PeriodIndex(strip["month"], freq="M")).sort_index()


def fixings_known_on(effr, day):
    """The EFFR fixture as it stood on `day`: every calendar day known by then."""
    fixings = effr.reset_index().rename(columns={"effr": "value"})
    return known_daily(fixings, day)


def solve_session(day, meetings, effr=None):
    """The implied path from the committed strip fixture for one session.

    With `effr`, days already fixed by `day` are substituted as known and the
    configured cutoffs apply, as in the panel. Without it the whole strip is
    solved from futures alone.
    """
    implied_avg = load_strip(day)
    horizon = pd.Timestamp(day).to_period("M") + HORIZON_MONTHS
    implied_avg = implied_avg[implied_avg.index <= horizon]
    pillars = known_meetings(meetings, day)["effective_date"]
    if effr is None:
        return implied_path(implied_avg, pillars)
    spec = config.currency("USD")["path"]
    return implied_path(implied_avg, pillars, fixings_known_on(effr, day),
                        spec["min_forward_days"], spec["min_regime_days"])


@pytest.mark.parametrize("day", strip_sessions())
def test_path_matches_the_prevailing_effr(day, meetings, effr):
    """The rate from today to the next meeting is the rate in force today.

    The solver is told nothing about the current policy rate beyond the fixings
    already published -- which enter only as the elapsed part of the front
    month -- so the level of the forward path still comes from futures, and
    this catches a level error of any kind.
    """
    path = solve_session(day, meetings, effr)
    prevailing = effr["effr"].asof(pd.Timestamp(day))

    assert path.index[0] == pd.Timestamp(day)
    assert path.iloc[0] == pytest.approx(prevailing, abs=1 * BP), (
        f"{day}: solved {path.iloc[0]:.4f} vs realized EFFR {prevailing:.4f}"
    )


@pytest.mark.parametrize("day", strip_sessions())
def test_regimes_already_in_the_past_match_realized_effr(day, meetings, effr):
    """A regime that has entirely happened is no longer a forecast.

    Its rate is published, so the solver must reproduce it from futures alone.
    This is the futures-only solve, deliberately: with realized days substituted
    no past regime is solved at all, so this is the check that the extraction
    itself gets history right. Tolerance is 2bp because a futures-only solve
    lets front-contract pricing noise into the elapsed days.
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
def test_path_is_monotone_in_time_order_and_finite(day, meetings, effr):
    path = solve_session(day, meetings, effr)
    assert path.index.is_monotonic_increasing
    assert path.notna().all()
    assert (path.abs() < 25).all(), "a policy rate this size means the solve blew up"


@pytest.mark.parametrize("day", strip_sessions())
def test_path_reprices_the_strip_it_was_solved_from(day, meetings, effr):
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
    path = solve_session(day, meetings, effr)

    days = pd.date_range(avg.index[0].start_time, avg.index[-1].end_time.normalize(), freq="D")
    step = fixings_known_on(effr, day).reindex(days)
    for pillar in path.index:
        step[step.index >= pillar] = path[pillar]
    assert step.notna().all(), "the solved path does not cover the whole strip"

    resid = (step.groupby(step.index.to_period("M")).mean() - avg).abs() * 100
    assert resid.max() < 3.0, f"{day}: worst month {resid.idxmax()} off {resid.max():.2f}bp"


def fedwatch_expected_midpoint(day):
    """Probability-weighted target midpoint at each meeting, from a capture."""
    pr = read_fedwatch(FEDWATCH / f"{day}_probabilities.csv",
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
    path = solve_session(day, meetings, effr)
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
