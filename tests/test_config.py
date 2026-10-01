"""The config schema: every enabled currency meets it, and a block that does not fails loudly, naming the field.

The point of validating on load is that a third currency's block says
everything it is missing on the first run. So the tests take each real block,
break it one field at a time, and check the error names that field. The book
(``strategy.yml``) is broken the same way, and then against the currency blocks.
"""

import copy
import pytest
import yaml
from policypath import config, regress

with open(config.CONFIG_DIR / "currencies.yml") as f:
    BLOCKS = yaml.safe_load(f)
ENABLED = [c for c, b in BLOCKS.items() if b.get("enabled")]
with open(config.CONFIG_DIR / "strategy.yml") as f:
    BOOK = yaml.safe_load(f)


# Removing one of these switches the block to the other branch of the schema, whose fields it then lacks.
DISCRIMINATORS = {"macro.inflation.rate", "macro.inflation.target", "rule.rstar.constant"}


def required(block):
    """Every dotted path the schema requires of `block`, given its extractor, inflation measure and r*,
    and, where it has them, its expressions (by instrument), their contracts and costs, its FX rate and credit."""
    paths = list(config.SCHEMA)
    paths += list(config.EXTRACTOR_SCHEMA[block["market"]["extractor"]])
    paths += list(config.INFLATION_SCHEMA["rate" if "rate" in block["macro"]["inflation"] else "target"])
    paths += list(config.RSTAR_SCHEMA["constant" if "constant" in block["rule"]["rstar"] else "published"])
    for e in config._expressions(block):
        x = block["expression"][e]
        paths += [f"expression.{e}.instrument", *(p.format(e=e) for p in config.INSTRUMENT_SCHEMA[x["instrument"]])]
        if config._contract(x):
            paths += [p.format(c=x["contract"]) for p in config.CONTRACT_SCHEMA]
        if "costs" in block:
            shape = "ticks" if "ticks_round_trip" in block["costs"][e] else "bp"
            paths += [p.format(e=e) for p in config.COST_SCHEMA[shape]]
            if x["instrument"] in config.PAR_INSTRUMENTS:
                paths.append(f"costs.{e}.roll_every_months")
    if "fx" in block.get("expression", {}):
        paths += list(config.FX_SCHEMA)
    if "credit" in block:
        paths += list(config.CREDIT_SCHEMA)
    paths += [p for key, table in config.VARIANT_SCHEMA.items() if config._has(block, key) for p in table]
    return [p for p in dict.fromkeys(paths) if p not in DISCRIMINATORS]


def without(block, path):
    b = copy.deepcopy(block)
    node = b
    *parents, last = path.split(".")
    for key in parents:
        node = node[key]
    del node[last]
    return b


@pytest.mark.parametrize("ccy", ENABLED)
def test_every_enabled_block_meets_the_schema(ccy):
    assert config.validate(ccy, BLOCKS[ccy]) == []


def test_loading_validates_and_lists_the_enabled_currencies():
    assert config.enabled() == ENABLED
    for ccy in ENABLED:
        assert config.currency(ccy) is BLOCKS[ccy] or config.currency(ccy) == BLOCKS[ccy]


@pytest.mark.parametrize("ccy, path", [(c, p) for c in ENABLED for p in required(BLOCKS[c])])
def test_a_missing_field_is_named(ccy, path):
    problems = config.validate(ccy, without(BLOCKS[ccy], path))
    assert any(path in p for p in problems), f"removing {path} gave {problems}"


@pytest.mark.parametrize("ccy", ENABLED)
def test_losing_the_key_that_picks_a_branch_asks_for_the_other_branch(ccy):
    block = BLOCKS[ccy]
    for path in DISCRIMINATORS:
        try:
            broken = without(block, path)
        except KeyError:
            continue
        assert config.validate(ccy, broken), f"removing {path} went unnoticed"


@pytest.mark.parametrize("ccy", ENABLED)
def test_a_wrong_type_is_named(ccy):
    b = copy.deepcopy(BLOCKS[ccy])
    b["path"]["n_meetings"] = "eight"
    b["rule"]["elb"] = [{"from": "2016-01-01", "value": 0.1}, {"from": "2009-01-01", "value": 0.5}]
    problems = config.validate(ccy, b)
    assert any("path.n_meetings" in p for p in problems)
    assert any("rule.elb" in p and "order" in p for p in problems)


@pytest.mark.parametrize("ccy", ENABLED)
def test_a_series_nothing_caches_is_refused(ccy):
    """The chain reads the overnight rate from the cache; a block whose sources do not pull it would fail later."""
    b = copy.deepcopy(BLOCKS[ccy])
    o = b["overnight"]
    del b["sources"]["daily"][o["source"]][o["series"]]
    problems = config.validate(ccy, b)
    assert any(f"{o['source']}/{o['series']}" in p for p in problems)


@pytest.mark.parametrize("ccy", ENABLED)
def test_a_daily_series_without_a_publication_lag_is_refused(ccy):
    b = copy.deepcopy(BLOCKS[ccy])
    source = next(iter(b["sources"]["daily"]))
    series = next(iter(b["sources"]["daily"][source]))
    b["sources"]["daily"][source][series] = None
    assert any("publication lag" in p for p in config.validate(ccy, b))


def uncached(block, source, series):
    """`block` with `series` taken out of whatever caches it (and its source dropped if that empties it)."""
    b = copy.deepcopy(block)
    for kind in ("daily", "curves", "futures", "estimates"):
        table = b["sources"].get(kind, {})
        if source in table and series in table[source]:
            group = table[source]
            group.pop(series) if isinstance(group, dict) else group.remove(series)
            if not group:
                del table[source]
    return b


def series_refs(node, path):
    """(path, source, series) for every {source, series} under `node`: series one name, a list, or {tenor: name}."""
    if not isinstance(node, dict):
        return []
    if {"source", "series"} <= set(node):
        s = node["series"]
        names = s.values() if isinstance(s, dict) else [s] if isinstance(s, str) else s
        return [(path, node["source"], n) for n in names]
    return [r for k, v in node.items() if k != "tags" for r in series_refs(v, f"{path}.{k}")]


def leg_refs(block):
    """Every series the expression and credit blocks read, found by walking them, not from `config._refs`.

    A contract an expression names settles off the market's futures source.
    """
    contracts = [(f"expression.{e}.contract", block["market"]["futures"]["source"], x["contract"])
                 for e, x in block.get("expression", {}).items() if isinstance(x, dict) and "contract" in x]
    return (contracts + series_refs(block.get("expression", {}), "expression")
            + series_refs(block.get("credit", {}), "credit"))


@pytest.mark.parametrize("ccy, what, source, series", [(c, *r) for c in ENABLED for r in leg_refs(BLOCKS[c])])
def test_every_series_an_expression_or_credit_leg_reads_must_be_cached(ccy, what, source, series):
    """A par curve, an FX rate or a credit leg nothing caches would fail only when weeks 7 or 12 first read it.

    The error must name the leg itself: a series another block also reads (DGS10 in the curve and the
    credit control, ZQ in the market) would otherwise be caught by that block's reference alone.
    """
    problems = config.validate(ccy, uncached(BLOCKS[ccy], source, series))
    assert any(p.startswith(what) and f" reads {source}/{series}," in p for p in problems), problems


def blocks_with(test):
    """(currency, expression key) for every expression passing `test`, so each check runs where it applies."""
    return [(c, e) for c in ENABLED for e in config._expressions(BLOCKS[c]) if test(BLOCKS[c]["expression"][e])]


@pytest.mark.parametrize("ccy, e", blocks_with(lambda x: x["instrument"] == "par_yield"))
@pytest.mark.parametrize("bad", [{}, ["DGS2"], {2: 5}, {-1: "X"}, {True: "X"}, {10: "B", 2: "A"}])
def test_a_tenor_map_is_positive_tenors_to_series_names_shortest_first(ccy, e, bad):
    b = copy.deepcopy(BLOCKS[ccy])
    b["expression"][e]["series"] = bad
    assert any(f"expression.{e}.series" in p for p in config.validate(ccy, b))


@pytest.mark.parametrize("ccy, e", blocks_with(lambda x: x["instrument"] in config.PAR_INSTRUMENTS))
def test_a_par_leg_is_on_its_curve_and_a_whole_number_of_coupons(ccy, e):
    b = copy.deepcopy(BLOCKS[ccy])
    x = b["expression"][e]
    x["legs"] = [x["legs"][0] + 0.25, x["legs"][1]]
    assert any("whole number of coupon periods" in p for p in config.validate(ccy, b))
    if x["instrument"] == "par_yield":
        x["legs"] = [x["legs"][0], max(x["series"]) + 1]
        assert any(f"expression.{e}.legs" in p and "outside" in p for p in config.validate(ccy, b))
    x["legs"] = x["legs"][::-1]
    assert any(f"expression.{e}.legs" in p for p in config.validate(ccy, b))


@pytest.mark.parametrize("ccy", ENABLED)
def test_an_expression_needs_a_known_instrument_and_a_cost_needs_an_expression(ccy):
    b = copy.deepcopy(BLOCKS[ccy])
    e = config._expressions(b)[0]
    b["expression"][e]["instrument"] = "swaption"
    b["costs"]["butterfly"] = {"round_trip_bp": 1.0, "observed": False}
    problems = config.validate(ccy, b)
    assert any("swaption" in p for p in problems)
    b["expression"][e] = copy.deepcopy(BLOCKS[ccy]["expression"][e])
    assert any("costs.butterfly names no expression" in p for p in config.validate(ccy, b))


@pytest.mark.parametrize("ccy, e", blocks_with(lambda x: not config._contract(x)))
def test_a_cost_in_ticks_needs_a_contract_to_take_the_tick_from(ccy, e):
    b = copy.deepcopy(BLOCKS[ccy])
    kept = {k: v for k, v in b["costs"][e].items() if k == "roll_every_months"}
    b["costs"][e] = {"ticks_round_trip": 2, "observed": True, **kept}
    assert any(f"costs.{e}.ticks_round_trip" in p and "no contract" in p for p in config.validate(ccy, b))


@pytest.mark.parametrize("ccy, e", blocks_with(lambda x: x["instrument"] in config.PAR_INSTRUMENTS))
def test_a_par_leg_is_re_struck_every_whole_number_of_months(ccy, e):
    for bad in (0, 1.5, True):
        b = copy.deepcopy(BLOCKS[ccy])
        b["costs"][e]["roll_every_months"] = bad
        assert any(f"costs.{e}.roll_every_months" in p for p in config.validate(ccy, b)), bad


@pytest.mark.parametrize("ccy", [c for c in ENABLED if "credit" in BLOCKS[c]])
def test_a_credit_spread_has_a_long_leg_signed_roles_and_the_primary_is_one_of_them(ccy):
    b = copy.deepcopy(BLOCKS[ccy])
    spreads = b["credit"]["spreads"]
    name, legs = next(iter(spreads.items()))
    spreads[name] = {"over": legs["long"]}
    assert any("credit.spreads" in p and "long" in p for p in config.validate(ccy, b))
    spreads[name] = {**legs, "times": legs["long"]}
    assert any("times" in p for p in config.validate(ccy, b))
    b = copy.deepcopy(BLOCKS[ccy])
    b["credit"]["primary"] = "not_a_spread"
    assert any("credit.primary" in p for p in config.validate(ccy, b))


@pytest.mark.parametrize("ccy", [c for c in ENABLED if "credit" in BLOCKS[c]])
@pytest.mark.parametrize("path, bad", [("week_ends", "Caturday"), ("exclude", {"w": ["2020-03-01", "2020-02-01"]}),
                                       ("staleness", {"source": "fred"}), ("labels", {"quality": ""})])
def test_the_credit_bridge_settings_are_checked(ccy, path, bad):
    b = copy.deepcopy(BLOCKS[ccy])
    b["credit"][path] = bad
    assert any(f"credit.{path}" in p for p in config.validate(ccy, b))


@pytest.mark.parametrize("ccy", [c for c in ENABLED if "credit" in BLOCKS[c]])
@pytest.mark.parametrize("path, change", [
    ("exclude", lambda c: c.update(exclude={})),
    ("exclude", lambda c: c.update(exclude={"elb": ["2020-02-15", "2020-04-30"]})),
    ("crosscheck", lambda c: c["crosscheck"].update({c["primary"]: next(iter(c["crosscheck"].values()))})),
    ("spreads", lambda c: (c["spreads"].update(level=c["spreads"][c["primary"]]), c["labels"].update(level="L"))),
    ("crosscheck", lambda c: (c["crosscheck"].update(differential_2=next(iter(c["crosscheck"].values()))),
                              c["labels"].update(differential_2="D"))),
])
def test_credit_names_the_bridge_would_overwrite_are_refused(ccy, path, change):
    """An empty exclude, a window called elb, a cross-check named like a spread or a spread named like a column."""
    b = copy.deepcopy(BLOCKS[ccy])
    change(b["credit"])
    assert any(f"credit.{path}" in p for p in config.validate(ccy, b))


@pytest.mark.parametrize("ccy", [c for c in ENABLED if "credit" in BLOCKS[c]])
def test_every_credit_spread_has_a_label(ccy):
    b = copy.deepcopy(BLOCKS[ccy])
    del b["credit"]["labels"][b["credit"]["primary"]]
    assert any("credit.labels" in p and b["credit"]["primary"] in p for p in config.validate(ccy, b))


@pytest.mark.parametrize("ccy", ENABLED)
@pytest.mark.parametrize("bad", ["a bare string", {"caveat": "a text"}, {"lag": ""}, {}])
def test_every_tags_block_anywhere_is_names_to_kinds_to_text(ccy, bad):
    """The brief and the limitations read tags as {name: {lag | quality | modelling: text}}."""
    b = copy.deepcopy(BLOCKS[ccy])
    b["backtest"]["tags"]["probe"] = bad
    b["expression"]["tags"]["probe"] = bad
    problems = config.validate(ccy, b)
    assert any(p.startswith("backtest.tags:") for p in problems)
    assert any(p.startswith("expression.tags:") for p in problems)


def test_all_three_tag_kinds_are_accepted():
    assert config._tags({"x": {k: "a text" for k in config.TAG_KINDS}}) is None


@pytest.mark.parametrize("ccy", [c for c in ENABLED if "estimates" in BLOCKS[c]["sources"]])
def test_an_estimate_is_cached_like_a_daily_series_and_needs_a_lag(ccy):
    b = copy.deepcopy(BLOCKS[ccy])
    source, lags = next(iter(b["sources"]["estimates"].items()))
    series = next(iter(lags))
    b["rule"]["rstar"] = {"source": source, "series": series, "label": "x", "before_first": 2.0,
                          "before_first_label": "x"}
    assert config.validate(ccy, b) == []
    b["sources"]["estimates"][source][series] = -1
    assert any("sources.estimates" in p and "publication lag" in p for p in config.validate(ccy, b))


# Week 9's model-side variants, as the overrides the robustness grid applies (notes/DECISIONS.md, V1-V7).
VARIANTS = {
    "hlw": {"rule": {"rstar": {"source": "nyfed", "series": "HLW_RSTAR", "label": "HLW", "real": True,
                               "before_first": 2.0, "before_first_label": "Taylor's 2%"}}},
    "estimated": {"rule": {"estimate": {"prior_quarters": 8, "drop_cuts_to_floor": True}}},
    "converge": {"rule": {"conditioning": {"converge": {"half_life_quarters": 4}}}},
    "nss": {"market": {"curve": {"method": "nss"}}},
    "window_365": {"signal": {"window": "365D", "min_periods": 125}},
    "projected": {"rule": {"projected": {"source": "alfred", "series": "FEDTARMD", "label": "SEP median"}}},
}


def with_variant(block, over):
    """`block` with the override merged in as `regress.outputs` merges it (an r* block is replaced whole)."""
    return regress.merge(block, over)


def variant_blocks():
    """(currency, variant, block with the variant) for every variant that applies to the currency's block."""
    applies = {"hlw": lambda b: ("nyfed", "HLW_RSTAR") in config._cached(b),
               "nss": lambda b: b["market"]["extractor"] == "forward_curve",
               "projected": lambda b: ("alfred", "FEDTARMD") in config._cached(b)}
    return [(c, name, with_variant(BLOCKS[c], over)) for c in ENABLED for name, over in VARIANTS.items()
            if applies.get(name, lambda b: True)(BLOCKS[c])]


@pytest.mark.parametrize("ccy, name, block", variant_blocks())
def test_every_variant_override_meets_the_schema(ccy, name, block):
    assert config.validate(ccy, block) == []


@pytest.mark.parametrize("ccy, name, block, path", [(c, n, b, p) for c, n, b in variant_blocks()
                                                    for p in required(b) if p not in required(BLOCKS[c])])
def test_a_variant_key_needs_the_paths_under_it(ccy, name, block, path):
    problems = config.validate(ccy, without(block, path))
    assert any(path in p for p in problems), f"removing {path} gave {problems}"


@pytest.mark.parametrize("name, path, bad", [("hlw", "rule.rstar.real", "yes"),
                                             ("estimated", "rule.estimate.prior_quarters", -1),
                                             ("estimated", "rule.estimate.drop_cuts_to_floor", "yes"),
                                             ("converge", "rule.conditioning.converge.half_life_quarters", 0),
                                             ("nss", "market.curve.method", "cubic_spline"),
                                             ("projected", "rule.projected.series", 7)])
def test_a_bad_variant_value_is_named(name, path, bad):
    ccy, _, block = copy.deepcopy(next(v for v in variant_blocks() if v[1] == name))
    *parents, last = path.split(".")
    config._get(block, ".".join(parents))[last] = bad
    assert any(path in p for p in config.validate(ccy, block))


def test_a_currency_that_caches_no_hlw_cannot_take_the_hlw_override():
    """GBP's real-time HLW vintages end in 2020Q2, and nothing caches them: the override is refused, by name."""
    for ccy in [c for c in ENABLED if ("nyfed", "HLW_RSTAR") not in config._cached(BLOCKS[c])]:
        problems = config.validate(ccy, with_variant(BLOCKS[ccy], VARIANTS["hlw"]))
        assert any("rule.rstar reads nyfed/HLW_RSTAR" in p for p in problems)


def test_a_projected_path_nothing_caches_is_refused():
    """The projected-path override reads a vintage series ``sources.vintages`` must cache; without it, refused by name."""
    for ccy in [c for c in ENABLED if ("alfred", "FEDTARMD") not in config._cached(BLOCKS[c])]:
        problems = config.validate(ccy, with_variant(BLOCKS[ccy], VARIANTS["projected"]))
        assert any("rule.projected reads alfred/FEDTARMD" in p for p in problems)
    ccy = next(c for c in ENABLED if ("alfred", "FEDTARMD") in config._cached(BLOCKS[c]))
    block = copy.deepcopy(with_variant(BLOCKS[ccy], VARIANTS["projected"]))
    del block["sources"]["vintages"]
    assert any("rule.projected reads alfred/FEDTARMD" in p for p in config.validate(ccy, block))


def test_an_unknown_calendar_or_extractor_is_refused():
    b = copy.deepcopy(BLOCKS[ENABLED[0]])
    b["calendar"], b["market"]["extractor"] = "target2", "swaption_cube"
    problems = config.validate("XXX", b)
    assert any("calendars.BDAYS" in p for p in problems)
    assert any("swaption_cube" in p for p in problems)


def test_a_disabled_block_is_a_draft_and_is_not_checked(monkeypatch, tmp_path):
    (tmp_path / "meetings").mkdir()
    for name in {b["meetings"] for b in BLOCKS.values() if "meetings" in b}:
        (tmp_path / "meetings" / name).write_text((config.CONFIG_DIR / "meetings" / name).read_text())
    draft = {**BLOCKS, "XXX": {"enabled": False, "meetings": "xxx.csv"}}
    (tmp_path / "currencies.yml").write_text(yaml.safe_dump(draft, sort_keys=False))
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    config._currencies.cache_clear()
    try:
        assert config.enabled() == ENABLED
        with pytest.raises(ValueError, match="not enabled"):
            config.currency("XXX")
        (tmp_path / "currencies.yml").write_text(yaml.safe_dump({**draft, "XXX": {"enabled": True}}, sort_keys=False))
        config._currencies.cache_clear()
        with pytest.raises(config.ConfigError, match="XXX: missing meetings"):
            config.currency(ENABLED[0])
    finally:
        config._currencies.cache_clear()


# ---- the book ------------------------------------------------------------------------

ENABLED_BLOCKS = {c: BLOCKS[c] for c in ENABLED}


def sleeve(book, kind):
    return next(s for s in book["sleeves"] if s["kind"] == kind)


def test_the_book_meets_its_schema_and_the_currency_blocks():
    assert config.validate_strategy(BOOK, ENABLED_BLOCKS) == []
    assert config.strategy() == BOOK


@pytest.mark.parametrize("path", list(config.STRATEGY_SCHEMA))
def test_a_missing_book_field_is_named(path):
    problems = config.validate_strategy(without(BOOK, path), ENABLED_BLOCKS)
    assert any(path in p for p in problems), f"removing {path} gave {problems}"


@pytest.mark.parametrize("change, message", [
    (lambda s: s.append({"name": "XXX outright", "kind": "outright", "ccy": "XXX"}), "not enabled"),
    (lambda s: s.append({"name": "fly", "kind": "butterfly", "ccy": "XXX"}), "butterfly"),
    (lambda s: s.append(dict(s[0])), "not unique"),
    (lambda s: s.append({"name": "x", "kind": "cross", "pair": ["A", "A"], "tenor": 2}), "two different"),
])
def test_a_bad_sleeve_is_named(change, message):
    book = copy.deepcopy(BOOK)
    change(book["sleeves"])
    assert any(message in p for p in config.validate_strategy(book, ENABLED_BLOCKS))


@pytest.mark.parametrize("kind", ["outright", "curve"])
def test_a_sleeve_needs_the_expression_it_trades(kind):
    blocks = copy.deepcopy(ENABLED_BLOCKS)
    ccy = sleeve(BOOK, kind)["ccy"]
    del blocks[ccy]["expression"][config.SLEEVE_EXPRESSION[kind]]
    assert any(f"{ccy} has no expression.{kind}" in p for p in config.validate_strategy(BOOK, blocks))


def test_a_cross_pair_shares_one_horizon_and_its_tenor_is_a_par_leg_on_both_curves():
    first, second = sleeve(BOOK, "cross")["pair"]
    blocks = copy.deepcopy(ENABLED_BLOCKS)
    blocks[second]["backtest"]["horizon"] += 1
    assert any("horizon differs" in p for p in config.validate_strategy(BOOK, blocks))
    book = copy.deepcopy(BOOK)
    sleeve(book, "cross")["tenor"] = 2.1
    assert any("coupon periods" in p for p in config.validate_strategy(book, ENABLED_BLOCKS))
    blocks = copy.deepcopy(ENABLED_BLOCKS)
    blocks[first]["expression"]["curve"] = blocks[first]["expression"]["outright"]
    assert any("not a par leg" in p for p in config.validate_strategy(BOOK, blocks))


@pytest.mark.parametrize("bad", [[1.0], [2.5, 1.0], [0, 2.5], "1-2.5", [1.0, True]])
def test_a_plausible_fx_range_is_low_then_high_and_positive(bad):
    ccy = next(c for c in ENABLED if "fx" in BLOCKS[c].get("expression", {}))
    b = copy.deepcopy(BLOCKS[ccy])
    b["expression"]["fx"]["plausible"] = bad
    assert any("expression.fx.plausible" in p for p in config.validate(ccy, b))


def test_a_leg_outside_the_book_currency_needs_an_fx_rate_and_the_book_currency_has_none():
    home = BOOK["book"]["currency"]
    other = next(c for s in BOOK["sleeves"] for c in config._sleeve_ccys(s) if c != home)
    blocks = copy.deepcopy(ENABLED_BLOCKS)
    blocks[home]["expression"]["fx"] = blocks[other]["expression"].pop("fx")
    problems = config.validate_strategy(BOOK, blocks)
    assert any(f"{other} trades in a book in {home} and has no expression.fx" in p for p in problems)
    assert any(f"{home} is the book currency" in p for p in problems)


@pytest.mark.parametrize("change, message", [
    (lambda b: b["positions"].update(exit=1.0), "positions.exit is not below"),
    (lambda b: b["positions"].update(enter=1.1), "not a cell of positions.grid"),
    (lambda b: b["portfolio"].update(constructions=["erc"]), "portfolio.headline"),
    (lambda b: b["portfolio"]["drawdown"].update(release=0.2), "portfolio.drawdown.release"),
    (lambda b: b["signal"].update(slope=[1, 99]), "path.n_meetings"),
    (lambda b: b["evaluation"]["exclude"].update(backwards=["2023-01-01", "2022-01-01"]), "evaluation.exclude"),
    (lambda b: b.update(tags={"probe": "a bare string"}), "tags"),
])
def test_the_book_settings_are_consistent(change, message):
    book = copy.deepcopy(BOOK)
    change(book)
    assert any(message in p for p in config.validate_strategy(book, ENABLED_BLOCKS))


def first_row(book):
    return book["robustness"]["choices"][0]["rows"][0]


@pytest.mark.parametrize("change, message", [
    (lambda b: first_row(b).update(currencies={"USD": {"costs": {"outright": {"ticks_round_trip": 9}}}}),
     "reach past the signal"),
    (lambda b: first_row(b).update(currencies={"XXX": {"signal": {"window": "365D"}}}),
     "not a currency the book trades"),
    (lambda b: first_row(b).update(currencies={"GBP": {"market": {"curve": {"method": "NSS"}}}}),
     "market.curve.method"),
    (lambda b: first_row(b).update(currencies={"GBP": {"market": {"curve": {"series": "X"}}}}),
     "reach past the signal"),
    (lambda b: first_row(b).update(currencies={"USD": {"signal": {"window": "365D", "min_periods": 100}}}),
     "share of the window, 125"),
    (lambda b: first_row(b).update(currencies={"USD": {"rule": {"estimate": {"prior_quarters": 8}}}}),
     "rule.estimate.drop_cuts_to_floor"),
    (lambda b: first_row(b).update(book={"portfolio": {"z_cap": 2.0}}), "reach past the signal"),
    (lambda b: first_row(b).update(book={"positions": {"carry_filter": "yes"}}), "positions.carry_filter"),
    (lambda b: first_row(b).update(every={"signal": {"window": "365D", "min_periods": 125}}),
     "both every and currencies"),
    (lambda b: first_row(b).update(key="Bad Key"), "lower_case_with_underscores"),
    (lambda b: b["robustness"]["choices"][1]["rows"][0].update(key=first_row(b)["key"]), "not unique"),
    (lambda b: b["robustness"]["choices"][0]["rows"].append({"key": "empty", "label": "nothing"}),
     "overrides nothing"),
    (lambda b: b["robustness"].update(ic_horizon=10), "robustness.ic_horizon"),
])
def test_a_robustness_row_that_reaches_past_the_signal_or_breaks_a_block_is_named(change, message):
    """A row moves the signal and nothing else, and the block it leaves must meet its own schema."""
    book = copy.deepcopy(BOOK)
    assert "currencies" in first_row(book)
    change(book)
    problems = config.validate_strategy(book, ENABLED_BLOCKS)
    assert any(message in p for p in problems), problems


def test_the_book_rows_override_only_what_the_config_allows():
    rows = [r for c in BOOK["robustness"]["choices"] for r in c["rows"]]
    assert rows and len({r["key"] for r in rows}) == len(rows)
    traded = {c for s in BOOK["sleeves"] for c in config._sleeve_ccys(s)}
    for r in rows:
        overs = config.row_overrides(r, sorted(traded))
        assert set(overs) <= traded
        assert all(config._under(p, config.ROBUSTNESS_CURRENCY_PATHS) for o in overs.values()
                   for p in config._leaves(o))
        assert all(config._under(p, config.ROBUSTNESS_BOOK_PATHS) for p in config._leaves(r.get("book", {})))


def test_a_bad_book_tag_is_named_once():
    problems = config.validate_strategy({**BOOK, "tags": {"probe": "a bare string"}}, ENABLED_BLOCKS)
    assert problems == ["tags: probe: 'a bare string' is not {kind: text}"]


def test_a_broken_book_fails_on_load_naming_the_file(monkeypatch, tmp_path):
    """Checked in `strategy()`, not with the currency blocks: they still load without a book."""
    (tmp_path / "meetings").mkdir()
    for name in {b["meetings"] for b in BLOCKS.values() if "meetings" in b}:
        (tmp_path / "meetings" / name).write_text((config.CONFIG_DIR / "meetings" / name).read_text())
    (tmp_path / "currencies.yml").write_text((config.CONFIG_DIR / "currencies.yml").read_text())
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    config._currencies.cache_clear()
    config.strategy.cache_clear()
    try:
        assert config.enabled() == ENABLED
        (tmp_path / "strategy.yml").write_text(yaml.safe_dump({**BOOK, "book": {"currency": "XXX"}}))
        with pytest.raises(config.ConfigError, match="strategy.yml fails its schema"):
            config.strategy()
    finally:
        config._currencies.cache_clear()
        config.strategy.cache_clear()
