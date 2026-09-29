"""The credit report on the synthetic world of `test_credit.py`: the headline's own counts and words, the report, the chart, the note's keys.

The world runs the currency's real credit block on made-up series with no
lead, and a policy rate through two hiking cycles, so every key the note
quotes exists.
"""

import copy
import json
from pathlib import Path
import re
import pytest
from policypath import config, credit
from policypath.report import credit as report
from test_credit import world

NOTE = Path(__file__).resolve().parents[1] / "notes" / "credit_section.md"
CCYS = [c for c in config.enabled() if "credit" in config.currency(c)]
# Off the floor in April 2010, then two cycles of four quarterly hikes (a year early, a year late) with a cut between.
MOVES = [("2010-04-01", 0.25), ("2010-10-01", 0.25), ("2011-04-01", 0.25), ("2011-10-03", 0.25), ("2012-06-01", -0.25),
         ("2013-01-02", 0.25), ("2013-07-01", 0.25), ("2014-01-02", 0.25), ("2014-07-01", 0.25)]


def two_cycles(sessions):
    return 0.1 + sum(step * (sessions >= day) for day, step in MOVES)


@pytest.fixture(scope="module", params=CCYS)
def built(request):
    inputs, cfg = world(lead=0.0, block=copy.deepcopy(config.currency(request.param)["credit"]), r0=two_cycles)
    results, frames = credit.run(inputs)
    return request.param, results, frames, cfg


def test_the_headline_counts_and_words_follow_the_results(built):
    _, r, _, cfg = built
    h = report.headline(r, cfg)
    years = r["headline_without_each_year"]
    assert h["t2_verdict"] == ("passes" if r["supported"] else "does not pass")
    assert h["t2_years_failing"] == sum(not c["supported"] for c in years.values())
    assert h["t2_years_negative"] == sum(c["slope"] < 0 for c in years.values())
    assert h["t2_without_year1"] == max(years, key=lambda y: years[y]["t"])
    no = r["predictive"][r["primary"]][max(credit.HORIZONS)]["z"]["all"]["nonoverlap"]
    assert h["t2_nonoverlap_clearing"] == round(no["share_clearing"] * no["offsets"])
    checks = r["headline_checks"]
    fails = any(checks[c]["t"] > -credit.Z95 for c in ("uniform", "bartlett_2h"))
    assert h["t2_kernel_sentence"].startswith("The pass depends") == (r["supported"] and fails)
    assert h["t3_cycles"] == 2 and h["t3_cycle2_late_label"].startswith("Jan 2014")
    placebo = r["headline_placebo"]
    assert h["t2_placebo_p"] == f"{placebo['p']:.2f}" and h["t2_placebo_shifts"] == placebo["shifts"] > 0
    assert h["t2_years_passing"] == h["t2_years"] - h["t2_years_failing"]
    assert h["t2_t"] == report._t(r["predictive"][r["primary"]][max(credit.HORIZONS)]["z"]["all"]["t"])
    assert h["t3_weeks"] <= r["sample"]["weeks"] - r["sample"]["elb_weeks"]
    json.dumps(report.jsonable({"headline": h, **r}), allow_nan=False)


def test_every_placeholder_in_the_note_is_a_key_the_headline_writes(built):
    ccy, r, _, cfg = built
    keys = re.findall(r"\{credit\.(\w+)\.headline\.(\w+)\}", NOTE.read_text())
    assert keys and {c for c, _ in keys} == {ccy}
    assert sorted({k for _, k in keys} - set(report.headline(r, cfg))) == []


def test_the_report_states_the_headline_and_an_identity_only_where_it_holds(built):
    ccy, r, _, cfg = built
    md = report.markdown(ccy, r, cfg)
    h = report.headline(r, cfg)
    assert f"slope {h['t2_slope']}bp per unit of z" in md and h["t2_kernel_sentence"] in md
    assert "its one-sided placebo p" in md and f"{h['t2_placebo_reject_pct']}% of the shifts" in md
    said = report.identities(r, cfg)
    for sentence in said:
        assert sentence in md
    # The block's Treasury-benchmarked pair: one is the other plus the control, and the made-up series agree.
    assert len(said) == 1
    broken = copy.deepcopy(r)
    a = next(n for n in cfg["credit"]["spreads"] if said[0].startswith(cfg["credit"]["labels"][n] + " is"))
    broken["contemporaneous"][a]["components_control"]["coef"]["control"] += 0.5
    assert report.identities(broken, cfg) == []


@pytest.mark.parametrize("ccy", CCYS)
def test_no_identity_where_the_legs_or_the_fits_do_not_give_one(ccy):
    inputs, cfg = world()                                   # no spread is another plus the control
    results, _ = credit.run(inputs)
    assert report.identities(results, cfg) == []
    assert "differs by exactly one" not in report.markdown("AAA", results, cfg)
    # With the lead, the spread that is Baa alone gets the staleness control and its pair does not.
    inputs, cfg = world(block=copy.deepcopy(config.currency(ccy)["credit"]), r0=two_cycles)
    results, _ = credit.run(inputs)
    assert len({results["staleness"][n]["shows"] for n in cfg["credit"]["spreads"] if n != cfg["credit"]["primary"]}) == 2
    assert report.identities(results, cfg) == []


def test_numbers_are_printed_as_the_report_prints_them_and_a_t_is_never_rounded_onto_the_line():
    assert report._fmt(-2.2049, 2, sign=True) == "-2.20" and report._fmt(3.628, 2, sign=True) == "+3.63"
    assert report._fmt(0.2985, 1) == "0.3" and report._fmt(float("nan")) is None
    assert report._t(-2.0927) == "-2.09" and report._t(1.957997) == "+1.958" and report._t(-1.96) == "-1.960"
    assert report._t(1.9649) == "+1.965" and report._t(1.9651) == "+1.97" and report._t(1.9549) == "+1.95"


def test_the_sign_pattern_of_a_phase_in_words():
    assert report._signs([-0.1, -0.2]) == "negative in both cycles"
    assert report._signs([-0.1, 0.2]) == "negative in the first cycle and positive in the second"
    assert report._signs([0.1, -0.2, 0.3]) == "positive in the first cycle, negative in the second and positive in the third"
    assert report._signs([0.1]) == "positive in the one cycle" and report._signs([]) == "not estimated"


def test_the_chart_is_drawn_in_both_themes(built, tmp_path):
    ccy, r, frames, cfg = built
    written = report.write_figures(frames, r, cfg, tmp_path, ccy)
    assert sorted(p.name for p in written) == sorted(f"credit_{ccy}_{t}.png" for t in ("light", "dark"))
    assert all(p.stat().st_size > 0 for p in written)
