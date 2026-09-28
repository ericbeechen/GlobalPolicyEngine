"""`market.path(date, ccy)`: one interface, two backends, from the cache, with no currency branch.

A temporary cache is built from the committed fixtures (ZQ strips and EFFR for
USD, the Bank's curve, SONIA and Bank Rate for GBP), and the one call is made
for both. Each must match its backend solved directly, and the two results must
have the same shape, so nothing downstream can tell them apart.
"""

from pathlib import Path
import pandas as pd
import pytest
from policypath import config, market
from policypath.calendars import UK_BDAY, US_BDAY
from policypath.sources import cache

DATA = Path(__file__).parent / "data"
T = pd.Timestamp


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise RuntimeError("market.path touched the network")
    monkeypatch.setattr("socket.socket.connect", refuse)


@pytest.fixture(scope="module")
def root(tmp_path_factory, fixings, curve, sonia, bank_rate):
    """A cache holding exactly the committed fixtures, under the names the config gives them."""
    root = tmp_path_factory.mktemp("cache")
    usd, gbp = config.currency("USD"), config.currency("GBP")
    for day in ["2024-09-17", "2026-09-21"]:
        s = pd.read_csv(DATA / f"zq_strip_{day}.csv")
        rows = pd.DataFrame({
            "date": T(day), "contract": s["symbol"], "value": s["price"],
            "published": T(day) + pd.Timedelta(hours=19),
            "expiration": pd.PeriodIndex(s["month"], freq="M").end_time.normalize(),
        })
        futures = usd["market"]["futures"]
        cache.append(rows, futures["source"], futures["series"], "USD", keys=("date", "contract"), root=root,
                     dated=True)
    cache.append(fixings, usd["overnight"]["source"], usd["overnight"]["series"], "USD", root=root)
    curve_spec = gbp["market"]["curve"]
    cache.append(curve, curve_spec["source"], curve_spec["series"], "GBP", keys=("date", "tenor"), root=root)
    cache.append(sonia, gbp["overnight"]["source"], gbp["overnight"]["series"], "GBP", root=root)
    policy = gbp["market"]["policy_rate"]
    cache.append(bank_rate, policy["source"], policy["series"], "GBP", root=root)
    return root


def test_one_call_gives_both_currencies_paths_in_the_same_shape(root):
    usd = market.path("2024-09-17", "USD", root)
    gbp = market.path("2024-08-02", "GBP", root)
    for p in (usd, gbp):
        assert isinstance(p, market.PolicyPath)
        assert p.meetings["k"].tolist() == list(range(1, 9))
        assert {"session", "published", "rate_now", "last_fixing", "first_unknown", "pinned",
                "residual_bp"} <= set(p.summary)
    assert list(usd.meetings.columns) == list(gbp.meetings.columns)
    assert usd.summary["session"] == T("2024-09-17") and gbp.summary["session"] == T("2024-08-02")


def test_each_matches_its_backend_solved_directly(root, fixings, curve, sonia, bank_rate):
    usd_cfg, gbp_cfg = config.currency("USD"), config.currency("GBP")
    s = pd.read_csv(DATA / "zq_strip_2024-09-17.csv")
    strip = pd.Series(s["implied_rate"].to_numpy(), index=pd.PeriodIndex(s["month"], freq="M"))
    spec = usd_cfg["path"]
    direct = market.solve_session("2024-09-17", strip, config.meetings("USD"), fixings, spec["n_meetings"],
                                  US_BDAY, spec["min_forward_days"], spec["min_regime_days"])
    got = market.path("2024-09-17", "USD", root)
    assert got.meetings["rate"].to_numpy() == pytest.approx(direct.meetings["rate"].to_numpy(), abs=1e-9)

    spot = curve[curve["date"] == T("2024-08-02")].set_index("tenor")["value"].sort_index()
    g = gbp_cfg["path"]
    direct = market.solve_curve("2024-08-02", spot, config.meetings("GBP"), sonia, g["n_meetings"], UK_BDAY,
                                gbp_cfg["market"]["curve"]["year_days"], g["tail_days"], g["min_regime_days"],
                                policy=bank_rate, pin_always=gbp_cfg["market"]["pin_first_regime"])
    got = market.path("2024-08-02", "GBP", root)
    pd.testing.assert_frame_equal(got.meetings, direct.meetings)


def test_a_date_between_sessions_reads_the_last_one_before_it(root):
    assert market.path("2024-09-22", "USD", root).summary["session"] == T("2024-09-17")
    with pytest.raises(ValueError, match="no GBP session"):
        market.path("2019-01-01", "GBP", root)


def test_the_backend_is_chosen_by_config_alone():
    assert config.currency("USD")["market"]["extractor"] == "futures_strip"
    assert config.currency("GBP")["market"]["extractor"] == "forward_curve"
    assert set(market.EXTRACTORS) == {"futures_strip", "forward_curve"}
