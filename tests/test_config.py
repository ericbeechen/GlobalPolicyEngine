"""The config schema: every enabled currency meets it, and a block that does not fails loudly, naming the field.

The point of validating on load is that a third currency's block says
everything it is missing on the first run. So the tests take each real block,
break it one field at a time, and check the error names that field.
"""

import copy
import pytest
import yaml
from policypath import config

with open(config.CONFIG_DIR / "currencies.yml") as f:
    BLOCKS = yaml.safe_load(f)
ENABLED = [c for c, b in BLOCKS.items() if b.get("enabled")]


# Removing one of these switches the block to the other branch of the schema, whose fields it then lacks.
DISCRIMINATORS = {"macro.inflation.rate", "macro.inflation.target", "rule.rstar.constant"}


def required(block):
    """Every dotted path the schema requires of `block`, given its extractor, inflation measure and r*."""
    paths = list(config.SCHEMA)
    paths += list(config.EXTRACTOR_SCHEMA[block["market"]["extractor"]])
    paths += list(config.INFLATION_SCHEMA["rate" if "rate" in block["macro"]["inflation"] else "target"])
    paths += list(config.RSTAR_SCHEMA["constant" if "constant" in block["rule"]["rstar"] else "published"])
    return [p for p in paths if p not in DISCRIMINATORS]


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
