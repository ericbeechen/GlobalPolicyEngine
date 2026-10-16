"""The explorer (`report/explorer.py`): its map names only what exists, every row of the decision log is read,
the time machine's columns line up, and the page is one self-contained file.

The map is checked against the committed reports, numbers sheet and log; the time
machine runs on the flat world of `test_closure.py`.
"""

import copy
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
import pytest
import yaml
from policypath import closure, config
from policypath.report import closure as closure_report
from policypath.report import explorer
from test_closure import H, K, world

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "notes" / "DECISIONS.md"


@pytest.fixture(scope="module")
def spec():
    return yaml.safe_load(explorer.MAP.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def sheet():
    return explorer.sheet((ROOT / "reports" / "numbers.md").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def log():
    return explorer.decisions(LOG.read_text(encoding="utf-8"))


def built(spec, sheet, log, stages=None, sections=None):
    items, tables = sheet
    secs, rows = log
    return explorer.stages(stages or spec["stages"], ROOT, items, tables, sections or secs, rows, config.enabled())


def test_the_map_names_only_what_exists(spec, sheet, log):
    stages = built(spec, sheet, log)
    assert [s["key"] for s in stages] == [s["key"] for s in spec["stages"]]
    for s, raw in zip(stages, spec["stages"]):
        assert len(s["numbers"]) >= len(raw.get("numbers") or [])
        assert all(n["value"] for n in s["numbers"])
        assert {d["section"] for d in s["decisions"]} == set(raw["decisions"])


def test_every_section_of_the_log_is_on_one_stage(spec, sheet, log):
    with pytest.raises(ValueError, match="on no stage"):
        built(spec, sheet, log, sections=log[0] + ["A section added later"])
    twice = copy.deepcopy(spec["stages"])
    twice[1]["decisions"] = twice[1]["decisions"] + twice[0]["decisions"][:1]
    with pytest.raises(ValueError, match="already on stage"):
        built(spec, sheet, log, stages=twice)


def test_a_name_that_does_not_exist_stops_the_build(spec, sheet, log):
    for key, value, words in [("numbers", [{"item": "No such item"}], "has no item"),
                              ("numbers", [{"column": "No such column"}], "has no column"),
                              ("code", ["src/policypath/no_such_module.py"], "do not exist"),
                              ("writes", ["no_such_report_{ccy}.md"], "do not exist"),
                              ("decisions", ["No such section"], "has no section"),
                              ("feeds", ["no_such_stage"], "not stages")]:
        bad = copy.deepcopy(spec["stages"])
        bad[0][key] = value
        with pytest.raises(ValueError, match=words):
            built(spec, sheet, log, stages=bad)


def test_every_row_of_the_log_is_read(log):
    text = LOG.read_text(encoding="utf-8")
    ids = (re.findall(r"^- \*\*([A-Z]{1,2}\d+)\*\*", text, re.M) + re.findall(r"^\| (L\d+) ", text, re.M)
           + re.findall(r"^- ([A-Z]\d+)\. ", text, re.M))
    sections, rows = log
    assert sorted(r["id"] for r in rows) == sorted(ids) and len(set(ids)) == len(ids)
    assert all(r["section"] in sections for r in rows) and all(r["date"] for r in rows if not r["id"].startswith("L"))


def test_each_row_has_its_kind_and_keeps_its_symbols(log):
    rows = {r["id"]: r for r in log[1]}
    assert rows["M1"]["kind"] == "choice" and rows["A16"]["kind"] == "headline"
    assert rows["D1"]["kind"] == "registered" and rows["H1"]["kind"] == "registered"
    assert rows["D17"]["kind"] == "after" and rows["W1"]["kind"] == "proposed"
    assert "<code>curves/policy_path.py</code>" in rows["M1"]["html"] and "curves/policy_path.py" in rows["M1"]["refs"]
    assert "u* the CBO NROU" in rows["X2"]["text"] and "<" not in rows["X2"]["text"]
    assert rows["R8"]["date"] == "2026-09-27" and rows["L7"]["section"] == "Publication lags and approximations"


def test_the_machine_lines_up():
    paths, signal, model = world(n=400)
    cfg = config.currency(config.enabled()[0])
    days = pd.DatetimeIndex(model["session"])
    sides = pd.DataFrame({"one": 1.0, "two": -1.0}, index=days[::2])
    first = days[100]
    m = explorer.machine(paths, signal, model, cfg, sides, first)
    n = len(model)
    for key in ("d", "mk", "md", "ah", "r0", "g", "z", "lo", "hi", "pi", "ug", "rs", "nt", "elb", "sd"):
        assert len(m[key]) == n, key
    assert m["d"][0] == 0 and all(b > a for a, b in zip(m["d"], m["d"][1:]))
    assert all(len(row) == m["meetings"] for row in m["mk"])
    assert all(x >= 0 for row in m["ah"] for x in row if x is not None)
    assert m["mk"][0][0] == round(paths["market"].iloc[0] * 100) and m["k"] == cfg["backtest"]["horizon"]
    assert m["sd"][99] == [0, 0] and m["sd"][100] == [1, -1] and m["sd"][101] == [1, -1]
    assert all(0 <= i < n for i, _ in m["moments"])


def test_the_page_is_one_self_contained_file(tmp_path):
    paths, signal, model = world(n=400)
    cfg = config.currency(config.enabled()[0])
    machines = {"AAA": explorer.machine(paths, signal, model, cfg)}
    labels = {"market": "the futures", "rule": "The rule", "policy_rate": "Policy rate"}
    ccfg = {"backtest": {"horizon": K}, "report": {"labels": labels}}
    panels = {"paths": paths, "signal": signal, "model": model}
    fitted = {"AAA": closure.run("AAA", load=lambda ccy, name: panels[name], cfg=ccfg, horizons=[H])}
    data = explorer.payload(ROOT, machines, closure_report.results_json(fitted, {"AAA": ccfg}), ["one", "two"])
    out = tmp_path / "reports"
    out.mkdir()
    page = explorer.write(tmp_path, data).read_text(encoding="utf-8")
    assert "/*__DATA__*/" not in page and "/*__THEME__*/" not in page
    assert not re.search(r"<script[^>]+src|<link\b|<iframe|<img|@import|url\(\s*['\"]?(?!#)|fetch\(|XMLHttpRequest|"
                         r"import\(", page)
    template = explorer.TEMPLATE.read_text(encoding="utf-8")
    assert set(re.findall(r"https?://[^\s\"'<>]+", template)) == {"http://www.w3.org/2000/svg"}
    assert page.count("</script>") == 1
    blob = re.search(r"const DATA=(.*?);\nconst NS=", page, re.S).group(1)
    assert json.loads(blob) == json.loads(json.dumps(data))
    assert not np.isnan(json.loads(blob)["machine"]["AAA"]["r0"][0])
