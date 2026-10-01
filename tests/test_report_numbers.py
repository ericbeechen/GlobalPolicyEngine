"""The numbers sheet: current with the committed reports, and stopped by a number its report does not print.

Reads the committed reports/ only (no data, no strategy chain), so it is fast.
Checks: reports/numbers.md is what the build makes from the reports beside it,
so a report regenerated without the sheet fails here; a JSON value its report
does not print stops the build; a line lookup finds only lines inside its
section.
"""

import json
from pathlib import Path
import shutil
import pytest
from policypath.report import numbers

ROOT = Path(__file__).resolve().parents[1]


def test_the_committed_sheet_is_current():
    committed = (ROOT / "reports" / "numbers.md").read_text(encoding="utf-8")
    assert numbers.markdown(ROOT) == committed, "reports/numbers.md is stale: run scripts/build_numbers.py"


@pytest.fixture
def copy(tmp_path):
    shutil.copytree(ROOT / "reports", tmp_path / "reports", ignore=shutil.ignore_patterns("*.png", "*.pdf"))
    for rel in ("README.md", "notes/DECISIONS.md"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, tmp_path / rel)
    return tmp_path


def test_a_json_value_the_report_does_not_print_stops_the_build(copy):
    p = copy / "reports" / "results" / "metrics.json"
    m = json.loads(p.read_text(encoding="utf-8"))
    m["performance"][0]["net_sr"] += 0.05
    p.write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(ValueError, match="metrics.md under 'Performance': no line prints"):
        numbers.markdown(copy)


def test_a_static_row_whose_record_is_gone_stops_the_build(copy):
    p = copy / "notes" / "DECISIONS.md"
    p.write_text(p.read_text(encoding="utf-8").replace("1,276,426", "1,276,000"), encoding="utf-8")
    with pytest.raises(ValueError, match="DECISIONS.md: no line prints '1,276,426'"):
        numbers.markdown(copy)


def test_a_line_is_found_only_inside_its_section(tmp_path):
    (tmp_path / "r.md").write_text("# R\n\n## A\n\nx 1\n\n## B\n\nx 2\n### B1\n\nx 3\n\n## C\n", encoding="utf-8")
    r = numbers.Reports(tmp_path)
    assert r.line("r.md", "x") == 5
    assert r.line("r.md", "x", section="B") == 9
    assert r.line("r.md", "x 3", section="B") == 12
    with pytest.raises(ValueError):
        r.line("r.md", "x 1", section="B")
    with pytest.raises(ValueError):
        r.line("r.md", "x", section="C")
