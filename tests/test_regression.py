"""The whole chain, on the committed fixtures, bit for bit against the reference frozen before the refactor.

`tests/data/reference/<ccy>/` was written by ``scripts/regress.py freeze
--fixtures`` at the week 5 code, on the Mac. It is checked on the Mac and on
Windows, whose floating point differs in the last bits, so floats compare to
`regress.PLATFORM_ULPS` ulps of their column's largest magnitude (at most 9e-13
of it, `policypath.regress`) and everything else exactly. Refreezing is a
decision, not a fix, and goes in notes/DECISIONS.md with the reason the numbers moved.

The harness is only worth having if it catches a change, so the last two
tests make one on purpose and check it is caught.
"""

import pytest
from policypath import config, fixtures, regress


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise RuntimeError("the regression harness touched the network")
    monkeypatch.setattr("socket.socket.connect", refuse)


@pytest.fixture(scope="module", params=fixtures.currencies())
def ccy(request):
    return request.param


@pytest.fixture(scope="module")
def frozen(ccy):
    return regress.reference(regress.FIXTURE_REF / ccy)[0]


def test_every_stage_is_bit_identical_to_the_frozen_reference(ccy, frozen):
    diffs = regress.compare(regress.fixture_outputs(ccy), frozen, regress.PLATFORM_ULPS)
    assert diffs == {}, "\n".join(f"{ccy} {stage}: {what}" for stage, what in diffs.items())


def test_the_reference_covers_every_stage(frozen):
    assert list(frozen) == regress.STAGES
    assert all(len(f) for f in frozen.values())


def test_a_coefficient_moved_in_its_twelfth_digit_is_caught(ccy, frozen, monkeypatch):
    """The rule's inertia times 1 + 1e-12: the path, the gap and the P&L move, and nothing upstream.

    Not one ulp: 0.85 ** 0.5 rounds a one-ulp change in 0.85 away, so the path
    really is bit-identical then, and the harness is right to say so. The
    smallest column it moves moves by 2.4e-12 of its largest value, above the
    platform tolerance (at most 9e-13).
    """
    spec = fixtures.spec(ccy)
    inertia = regress.merge({}, spec.get("overrides", {})).get("rule", {}).get("inertia", 0.85)
    bumped = {**spec, "overrides": regress.merge(spec.get("overrides", {}),
                                                 {"rule": {"inertia": inertia * (1 + 1e-12)}})}
    monkeypatch.setattr(fixtures, "spec", lambda c: bumped)
    diffs = regress.compare(regress.fixture_outputs(ccy), frozen, regress.PLATFORM_ULPS)
    assert set(diffs) == {"paths", "signal", "backtest"}, "inertia moves the path, not the rule's goal or its inputs"


def test_a_nudged_fixing_is_caught(ccy, frozen):
    """The last fixing the last session reads, moved by 1e-9pp, well inside any print's rounding."""
    logs = fixtures.logs(ccy)
    overnight = config.currency(ccy)["overnight"]
    key = (overnight["source"], overnight["series"])
    f = logs[key].copy()
    last = fixtures.sessions(ccy)[-1]
    i = f.index[f["published"] <= last][-1]
    f.loc[i, "value"] += 1e-9
    diffs = regress.compare(regress.fixture_outputs(ccy, logs={**logs, key: f}), frozen, regress.PLATFORM_ULPS)
    assert diffs, "a moved input left every stage identical"
