"""No currency lives in shared code: the acceptance grep, as a test.

Currency-specific behaviour belongs in ``config/currencies.yml``. A module that
names a currency, its overnight rate, its futures, its committee or its
calendar in *code* (not in a docstring or a comment, which may explain with
examples) has a branch or a default the config should be giving it.

Backends are exempt: a module the config names, which exists to speak to one
provider or to make one currency's own check. Each is listed with the reason, so
adding one is a decision someone can see.
"""

import ast
import io
from pathlib import Path
import re
import tokenize
import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "policypath"
SCRIPTS = SRC.parents[1] / "scripts"

# Currencies, their overnight and policy rates, futures roots, committees and calendars;
# the series behind the curve, cross and credit legs, the FX rate, and the curves and estimates.
FORBIDDEN = re.compile(
    r"\b(USD|GBP|EUR|AUD|CAD|JPY|CHF|EFFR|SOFR|SONIA|ESTR|AONIA|CORRA|ZQ|SR1|SR3|FOMC|MPC|"
    r"US_BDAY|UK_BDAY|IUDSOIA|IUDBEDR|DFEDTAR[LU]|FEDTARMDLR|PCEPILFE|UNRATE|NROU|MGSX|D7G7|"
    r"DGS\d+|DEXUSUK|BAA10Y|AAA10Y|BAML\w+|GLC_SPOT|OIS_SPOT|HLW_RSTAR)\b")

BACKENDS = {
    "calendars.py": "the holiday calendars `calendar:` and `market.exchange_calendar` name",
    "sources/fred.py": "the FRED/ALFRED client: FRED publishes on Fed business days",
    "sources/boe.py": "the Bank of England client: it publishes on London business days",
    "sources/nyfed.py": "the New York Fed client: HLW's real-time r* workbook, by its series name",
    "sources/ons.py": "the ONS client",
    "sources/rates.py": "the Databento archive reader for CME futures",
    "sources/pull_fomc.py": "regenerates config/meetings/fomc.csv",
    "sources/pull_mpc.py": "regenerates config/meetings/mpc.csv",
    "curves/basis.py": "the SOFR cross-check, run where the config has a `sofr` block",
    "curves/helpers.py": "QuantLib's SOFR index and calendar, for the SOFR discount curve",
    "curves/build.py": "the SOFR discount curve",
    "report/crosscheck.py": "the SOFR cross-check's report and figure",
    "report/nowcast.py": "the US nowcast report (`macro.report: nowcast`)",
    "report/labour.py": "the UK labour-market report (`macro.report: labour`)",
    "report/conditioning.py": "the check against the Bank's MPR paths (`validation.conditioning`)",
    "report/onepager.py": "the week 4 one-pager, USD prose (`report.onepager: true`)",
    "fixtures.py": None,   # not exempt: listed so a typo in this table is caught
}
SCRIPT_BACKENDS = {
    "pull_fred.py": "audits the EFFR publication-lag rule against ALFRED",
    "show_databento.py": "prints the CME archive's contracts",
    "file_databento.py": "files Databento downloads",
}


def code_hits(path):
    """(line, token) for every forbidden name in `path`'s code: its docstrings and comments are skipped."""
    text = path.read_text()
    docstrings = set()
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            value = first.value if isinstance(first, ast.Expr) else None
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                docstrings.add((first.lineno, first.col_offset))
    hits = []
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type == tokenize.COMMENT or (tok.type == tokenize.STRING and tok.start in docstrings):
            continue
        if tok.type in (tokenize.NAME, tokenize.STRING, tokenize.FSTRING_MIDDLE):
            hits += [(tok.start[0], m) for m in FORBIDDEN.findall(tok.string)]
    return hits


def shared(root, exempt):
    return sorted(p for p in root.rglob("*.py") if exempt.get(p.relative_to(root).as_posix()) is None)


@pytest.mark.parametrize("path", shared(SRC, BACKENDS), ids=lambda p: str(p.relative_to(SRC)))
def test_shared_modules_name_no_currency(path):
    hits = code_hits(path)
    assert not hits, (f"{path.relative_to(SRC)} names {hits}: move it to config/currencies.yml, "
                      "or list the module as a backend")


@pytest.mark.parametrize("path", shared(SCRIPTS, SCRIPT_BACKENDS), ids=lambda p: p.name)
def test_scripts_name_no_currency(path):
    hits = code_hits(path)
    assert not hits, f"scripts/{path.name} names {hits}: take it from the config"


def test_every_listed_backend_exists():
    missing = [name for name in BACKENDS if not (SRC / name).exists()]
    missing += [name for name in SCRIPT_BACKENDS if not (SCRIPTS / name).exists()]
    assert not missing


def test_the_check_sees_code_and_not_prose():
    probe = SRC.parents[1] / "tests" / "data" / "fixtures.yml"   # any file: only its path is borrowed
    sample = '"""The ZQ path."""\n# EFFR in a comment\nx = "USD"\ny = f"{z} SONIA"\n'
    tmp = probe.with_name("_probe.py")
    tmp.write_text(sample)
    try:
        assert [t for _, t in code_hits(tmp)] == ["USD", "SONIA"]
    finally:
        tmp.unlink()
