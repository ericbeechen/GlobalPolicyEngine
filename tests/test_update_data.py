"""update_data.py ``--only``: it pulls the series it names and nothing else, and refuses what the block does not list.

Every source in the registry is replaced by one that records what it is asked
to update and downloads nothing, and a macro pull fails the test, so this runs
offline. The script still reads the cache's manifest to print what is covered.
"""

import runpy
import sys
from pathlib import Path
import pytest
from policypath import config
from policypath.sources import registry

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "update_data.py"


@pytest.fixture
def pulled(monkeypatch):
    """The series the sources were asked to update, in order."""
    calls = []

    class Recorder:
        def __init__(self, *lag):
            pass

        def update(self, series, *args, **kwargs):
            calls.append(series)
            return 0

    for table in ("DAILY", "ESTIMATES", "CURVES", "FUTURES"):
        monkeypatch.setattr(registry, table, dict.fromkeys(getattr(registry, table), Recorder))
    monkeypatch.setattr(registry, "MACRO", dict.fromkeys(registry.MACRO, lambda *a: pytest.fail("a macro pull")))
    return calls


def run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", [SCRIPT.name, *argv])
    runpy.run_path(str(SCRIPT), run_name="__main__")


def one_of_each(ccy):
    """The last series of every source the block caches daily, as estimates or as curves."""
    sources = config.currency(ccy)["sources"]
    return [list(group)[-1] for kind in ("daily", "estimates", "curves") for group in sources.get(kind, {}).values()]


@pytest.mark.parametrize("ccy", config.enabled())
def test_only_pulls_just_the_series_it_names_and_no_macro_vintages(ccy, monkeypatch, pulled):
    wanted = one_of_each(ccy)
    run(monkeypatch, "--ccy", ccy, "--only", *wanted)
    assert sorted(pulled) == sorted(wanted)


def test_only_refuses_a_series_the_block_does_not_list_and_goes_without_macro_only(monkeypatch, pulled, capsys):
    ccy = config.enabled()[0]
    with pytest.raises(SystemExit):
        run(monkeypatch, "--ccy", ccy, "--only", "NOT_A_SERIES", *one_of_each(ccy))
    assert f"--only ['NOT_A_SERIES']: not under sources in the {ccy} block" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        run(monkeypatch, "--ccy", ccy, "--only", *one_of_each(ccy), "--macro-only")
    assert "not allowed with argument --only" in capsys.readouterr().err
    assert pulled == []
