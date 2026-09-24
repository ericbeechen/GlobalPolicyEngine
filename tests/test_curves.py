"""The SOFR curve reprices its own inputs, by QuantLib and by the contract's definition.

QuantLib repricing its own helpers is nearly circular: a bootstrap solves for
exactly that. The second check is the one that catches day-count and
convention errors. It recomputes each SR3 settlement rate from scratch --
reference quarter from `curves/futures.py`, realized SOFR compounded Actual/360
over the elapsed days, the curve's discount factors over the rest -- and asks
that this, too, gives back the settle.
"""

from pathlib import Path
import pandas as pd
import pytest
import QuantLib as ql
from policypath import config
from policypath.curves.build import build_curve, sofr_curve
from policypath.curves.futures import reference_period, settlement_rate, third_wednesday
from policypath.curves.helpers import SOFR_CALENDAR, ql_date, sofr_business_days, sofr_index

DATA = Path(__file__).parent / "data"
SESSIONS = sorted(p.name.removeprefix("sr3_strip_").removesuffix(".csv")
                  for p in DATA.glob("sr3_strip_*.csv"))
BP = 0.01  # in percent


@pytest.fixture(scope="module")
def sofr():
    return pd.read_csv(DATA / "sofr.csv", parse_dates=["date", "published"])


def strip(day):
    s = pd.read_csv(DATA / f"sr3_strip_{day}.csv", parse_dates=["expiration"])
    return s.rename(columns={"price": "value"})


@pytest.fixture(scope="module", params=SESSIONS)
def built(request, sofr):
    day = pd.Timestamp(request.param)
    curve, helpers, contracts = sofr_curve(day, strip(request.param), sofr,
                                           config.currency("USD")["sofr"])
    return day, curve, helpers, contracts


def test_there_is_something_to_test():
    assert len(SESSIONS) >= 4


def test_curve_reprices_every_helper(built):
    day, curve, helpers, contracts = built
    assert len(helpers) >= 8, f"{day.date()}: only {len(helpers)} SR3 contracts on the curve"
    for h, c in zip(helpers, contracts.itertuples()):
        assert h.impliedQuote() == pytest.approx(c.price, abs=1e-8), c.contract


def test_reference_quarters_match_quantlib(built):
    _, _, helpers, contracts = built
    for h, c in zip(helpers, contracts.itertuples()):
        assert ql_date(c.start) == h.earliestDate(), c.contract
        assert ql_date(c.end) == h.maturityDate(), c.contract


def test_curve_reprices_every_contract_by_its_definition(built, sofr):
    """Compound SOFR over the IMM quarter, independently of QuantLib's futures code.

    One known exception. A quarter opening on a SOFR holiday (SR3M4, Juneteenth
    2024) takes the previous business day's fixing for that day under CME's
    rule; QuantLib compounds its curve from the IMM date itself. The forecast of
    that earlier fixing straddles a curve node, so the two differ by the kink
    there -- 0.04 to 0.22bp on these sessions. Asserted small, not zero.
    """
    day, curve, _, contracts = built
    known = sofr[sofr["published"] < day + pd.Timedelta(days=1)].set_index("date")["value"]
    for c in contracts.itertuples():
        past = known.loc[c.start - pd.Timedelta(days=7):day - pd.Timedelta(days=1)]
        future = sofr_business_days(max(c.start - pd.Timedelta(days=7), day), c.end)
        dates = past.index.append(future)
        # Each future fixing is the curve's simple Actual/360 forward to the next one.
        bounds = future.append(pd.DatetimeIndex([c.end]))
        fwd = [(curve.discount(ql_date(a)) / curve.discount(ql_date(b)) - 1) * 360 / (b - a).days * 100
               for a, b in zip(bounds[:-1], bounds[1:])]
        rates = pd.concat([past, pd.Series(fwd, index=future)])
        daily = rates.reindex(pd.date_range(dates[0], c.end - pd.Timedelta(days=1))).ffill()

        got = settlement_rate("imm_quarter", c.start, c.end, daily, dates)
        holiday_open = not SOFR_CALENDAR.isBusinessDay(ql_date(c.start))
        tolerance = 0.5 * BP if holiday_open else 0.01 * BP
        assert got == pytest.approx(100.0 - c.price, abs=tolerance), (
            f"{day.date()} {c.contract}: {got:.5f} vs {100 - c.price:.5f}")


def test_sr3_reference_quarter_is_imm_to_imm():
    # SR3Z3 expires Tuesday 19 March 2024: the quarter is 20 Dec 2023 -> 20 Mar 2024.
    assert reference_period("imm_quarter", "2024-03-19") == (pd.Timestamp("2023-12-20"),
                                                             pd.Timestamp("2024-03-20"))
    assert third_wednesday(2024, 6) == pd.Timestamp("2024-06-19")  # Juneteenth: a holiday


def test_a_quarter_opening_on_a_holiday_takes_the_previous_fixing():
    """SR3M4 opens on Juneteenth 2024; Tuesday's SOFR covers the Wednesday."""
    start, end = pd.Timestamp("2024-06-19"), pd.Timestamp("2024-09-18")
    dates = sofr_business_days(start - pd.Timedelta(days=3), end)
    assert pd.Timestamp("2024-06-19") not in dates
    days = pd.date_range(dates[0], end - pd.Timedelta(days=1))
    flat = settlement_rate("imm_quarter", start, end, pd.Series(5.0, index=days), dates)
    # 91 days compounded at 5% act/360: a touch above 5, never a missing day.
    growth = 1.0
    for a, b in zip(dates[dates >= "2024-06-18"], [*dates[dates >= "2024-06-18"][1:], end]):
        growth *= 1 + 0.05 * (b - max(a, start)).days / 360
    assert flat == pytest.approx((growth - 1) * 360 / 91 * 100, abs=1e-12)


def test_sofr_index_holds_only_fixings_published_by_as_of(sofr):
    day = pd.Timestamp("2024-09-17")
    index = sofr_index(sofr, day)
    assert index.hasHistoricalFixing(ql_date("2024-09-16"))      # published the 17th
    assert not index.hasHistoricalFixing(ql_date("2024-09-17"))  # published the 18th


def test_nss_fits_bonds_priced_off_a_known_curve():
    """The Svensson branch, on the bonds it is built for."""
    today = ql.Date(16, 9, 2024)
    ql.Settings.instance().evaluationDate = today
    truth = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.04, ql.Actual365Fixed()))
    engine = ql.DiscountingBondEngine(truth)
    helpers, bonds = [], []
    for years in [1, 2, 3, 5, 7, 10, 15, 20, 30]:
        schedule = ql.Schedule(today, today + ql.Period(years, ql.Years), ql.Period(ql.Semiannual),
                               ql.NullCalendar(), ql.Unadjusted, ql.Unadjusted,
                               ql.DateGeneration.Backward, False)
        bond = ql.FixedRateBond(0, 100.0, schedule, [0.04], ql.Actual365Fixed())
        bond.setPricingEngine(engine)
        helpers.append(ql.FixedRateBondHelper(ql.QuoteHandle(ql.SimpleQuote(bond.cleanPrice())),
                                              0, 100.0, schedule, [0.04], ql.Actual365Fixed()))
        bonds.append(bond)
    curve = build_curve(today, helpers, ql.Actual365Fixed(), "nss")
    fitted = ql.DiscountingBondEngine(ql.YieldTermStructureHandle(curve))
    for bond in bonds:
        want = bond.cleanPrice()
        bond.setPricingEngine(fitted)
        assert bond.cleanPrice() == pytest.approx(want, abs=0.01)  # a cent per 100


def test_nss_refuses_futures_helpers(built):
    day, _, helpers, _ = built
    with pytest.raises(TypeError, match="bond helpers"):
        build_curve(ql_date(day), helpers, ql.Actual360(), "nss")
