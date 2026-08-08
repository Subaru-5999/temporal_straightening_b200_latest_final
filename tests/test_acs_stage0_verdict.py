"""Task 4.4 - unit tests for the Stage-0 pre-registered verdict rules.

The design's properties P1-P19 cover the *model* layer. This module covers the layer
that can kill the feature: the decision rule read off the Stage-0 statistics
(`rule_a_verdict`, `rule_b_verdict`, `combine_rule_verdicts`, `evaluate_stage0` in
`probe_ccr_curvature.py`, plus `parse_table1_gains` and `ordering_vs_table1_gains`).

Those functions are pure over plain statistic dicts - no dataset, no torch, no hydra,
no `DATASET_DIR` - so every test here hands them a literal and runs in milliseconds on
CPU.

What is pinned, and why:

1. **Totality.** Over an exhaustive grid of four `frac(cos<0)` values (9^4 = 6561
   assignments, including 0.0, exact ties and 1.0) and of PushT `R` values (including
   `None`), each rule returns exactly one of `GO` / `MIDDLE` / `STOP`: it never raises
   on valid input, never returns something outside the enum, and never lands in a
   region the prose leaves unnamed. The verdict is cross-checked against
   `_reference_rule_a_verdict`, written from the prose table of `PROGRESS_ACS.md`
   section 4.1 rather than from the implementation, so a gap or an overlap in the
   shipped branch order shows up as a disagreement. The clause code is checked to
   determine the verdict uniquely, which is what makes "no overlap" mechanical:
   clause 2.2 is the only GO, 2.3/2.4 the only MIDDLEs, 2.6/2.7 the only STOPs.

2. **Exact boundaries.** `1.5x` and `1.1x` on rule A and `0.15` and `0.08` on rule B
   are each tested *at*, *just below* and *just above* the value, so an inclusive /
   exclusive slip is caught rather than sitting in the one function nobody re-reads.
   The rule-A boundary fixtures scale by a **power of two** on purpose, as
   `rule_a_verdict`'s docstring instructs: in float64 `0.30 / 0.20` is
   `1.4999999999999998` and would land on the MIDDLE side of `1.5x`, whereas
   `0.375 / 0.25` is exactly `1.5` and `(1.1 * 0.5) / 0.5` is exactly `1.1`.
   "Just below" and "just above" are one ULP away via `math.nextafter`, which is the
   strongest form of the boundary test: nothing can sit between the fixture and the
   threshold.

3. **Combination.** All nine `(rule A, rule B)` verdict pairs: either `STOP` gives a
   combined `STOP` with `stage1_permitted` false (Requirements 2.12, 2.13), both at
   least `MIDDLE` permits Stage 1 (Requirement 2.14), and `GO` requires both.

4. **The thresholds themselves.** `RULE_A_CLEAR_MARGIN`, `RULE_A_INDISTINGUISHABLE`,
   `RULE_B_GO` and `RULE_B_MIDDLE` are asserted equal to the values pre-registered in
   `PROGRESS_ACS.md` section 4 on 2026-08-08 (Requirements 2.1, 2.17). Refitting a
   threshold to the measured data - the documented CCR failure mode - then fails a
   test instead of passing silently.

5. **Requirement 3.6's downgrade is a cap, never an upgrade.** When `raw` shows
   reversal structure and `sum` does not, the headline verdict stays `STOP`: the
   pre-declared downgrade can only lower a verdict, so it can never convert the
   headline STOP into permission to build.

Validates: Requirements 2.2, 2.3, 2.4, 2.6, 2.7, 2.8, 2.9, 2.11, 2.14
"""

from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from probe_ccr_curvature import (  # noqa: E402
    ACS_ENV_RULE_KEYS,
    ACS_HEADLINE_REDUCTION,
    ACS_RULE_ENV_KEYS,
    ACS_TABLE1_GAINS_DEFAULT,
    ACS_WITHIN_STEP_REDUCTION,
    RULE_A_CLEAR_MARGIN,
    RULE_A_INDISTINGUISHABLE,
    RULE_B_GO,
    RULE_B_MIDDLE,
    VERDICT_GO,
    VERDICT_MIDDLE,
    VERDICT_STOP,
    VERDICTS,
    combine_rule_verdicts,
    evaluate_stage0,
    ordering_vs_table1_gains,
    parse_table1_gains,
    rule_a_verdict,
    rule_b_verdict,
)

# Clause -> verdict, straight out of Requirement 2. The map is what makes "exactly one
# verdict" mechanical: a clause that could produce two verdicts, or a verdict reached
# through a clause that does not name it, fails the totality test.
CLAUSE_VERDICT_A = {"2.2": VERDICT_GO,
                    "2.3": VERDICT_MIDDLE,
                    "2.4": VERDICT_MIDDLE,
                    "2.6": VERDICT_STOP,
                    "2.7": VERDICT_STOP}
CLAUSE_VERDICT_B = {"2.8": VERDICT_GO,
                    "2.9": VERDICT_MIDDLE,
                    "2.11": VERDICT_STOP}

# The exhaustive rule-A grid. Chosen to hit every structural case rather than to be
# large: 0.0 (a zero denominator, so the ratio is unbounded rather than a number),
# exact power-of-two neighbours (0.125 / 0.25 / 0.375 / 0.5 sit exactly on 1.5x and
# 3x of one another), 0.55 (exactly 1.1x of 0.5), and 1.0 (the top of the domain).
# Every pair of grid points is also a tie candidate, which is where the strict
# "highest" and "lowest" comparisons live.
FRACTION_GRID = (0.0, 0.05, 0.1, 0.125, 0.25, 0.375, 0.5, 0.55, 1.0)

# The rule-B grid: both thresholds exactly, one ULP either side of each, the ends of
# the domain, and `None` (the probe's encoding of `mean(w) = 0`).
R_GRID = (None,
          0.0,
          math.nextafter(RULE_B_MIDDLE, 0.0),
          RULE_B_MIDDLE,
          math.nextafter(RULE_B_MIDDLE, 1.0),
          0.1,
          math.nextafter(RULE_B_GO, 0.0),
          RULE_B_GO,
          math.nextafter(RULE_B_GO, 1.0),
          0.5,
          0.999)


def _fracs(pusht, wall, umaze, medium):
    """A rule-A input in the rule's own env keys (`point_maze` is read as `umaze`)."""
    return {"pusht": pusht, "wall": wall, "umaze": umaze, "medium": medium}


def _ratio_or_inf(numerator, denominator):
    """The probe's `_ratio` semantics, restated: `inf` for x/0, `None` for 0/0."""
    if denominator > 0:
        return numerator / denominator
    return math.inf if numerator > 0 else None


def _reference_rule_a_verdict(values):
    """
    Rule A transcribed from the prose table of `PROGRESS_ACS.md` section 4.1, in the
    order the prose states the outcomes, and deliberately *not* from the shipped
    branch order. Returns `(verdict, clause)`.

    | PushT highest AND >= 1.5x each of the other three AND UMaze lowest   | GO     |
    | PushT highest BUT the ordering inverts, or the margin is [1.1x, 1.5x) | MIDDLE |
    | PushT not the highest, OR within 1.1x of the smoothest                | STOP   |
    """
    pusht = values["pusht"]
    others = [value for key, value in values.items() if key != "pusht"]
    largest_other = max(others)
    smoothest = min(values.values())
    umaze_is_lowest = values["umaze"] < min(
        value for key, value in values.items() if key != "umaze")

    if not pusht > largest_other:
        return VERDICT_STOP, "2.6"
    ratio_to_smoothest = _ratio_or_inf(pusht, smoothest)
    if ratio_to_smoothest is not None and ratio_to_smoothest < RULE_A_INDISTINGUISHABLE:
        return VERDICT_STOP, "2.7"
    margin = _ratio_or_inf(pusht, largest_other)
    if margin is not None and margin >= RULE_A_CLEAR_MARGIN and umaze_is_lowest:
        return VERDICT_GO, "2.2"
    return VERDICT_MIDDLE, ("2.4" if umaze_is_lowest else "2.3")


def _stats(fracs, pusht_R, reduction=ACS_HEADLINE_REDUCTION):
    """`evaluate_stage0`'s input: exactly the fields `summarize_cos_and_gate` writes."""
    return {reduction: {key: {"frac_cos_lt_0": fracs[key],
                              "reallocation_R": pusht_R if key == "pusht" else 0.2}
                        for key in ACS_RULE_ENV_KEYS}}


# --------------------------------------------------------------------------
# 4. The thresholds are the pre-registered ones (Requirements 2.1, 2.17)
# --------------------------------------------------------------------------
def test_thresholds_are_the_preregistered_values():
    """`PROGRESS_ACS.md` section 4, written 2026-08-08, before the data."""
    assert RULE_A_CLEAR_MARGIN == 1.5
    assert RULE_A_INDISTINGUISHABLE == 1.1
    assert RULE_B_GO == 0.15
    assert RULE_B_MIDDLE == 0.08


def test_rule_env_keys_map_point_maze_onto_umaze():
    assert ACS_RULE_ENV_KEYS == ("pusht", "wall", "umaze", "medium")
    assert ACS_ENV_RULE_KEYS == {"pusht": "pusht", "wall": "wall",
                                 "point_maze": "umaze",
                                 "point_maze_medium": "medium"}


# --------------------------------------------------------------------------
# 1. Totality
# --------------------------------------------------------------------------
def test_rule_a_is_total_over_the_fraction_grid():
    """
    Every assignment of four fractions lands on exactly one verdict, reached through
    exactly one clause, and agrees with the prose. 9^4 = 6561 assignments.
    """
    for pusht, wall, umaze, medium in itertools.product(FRACTION_GRID, repeat=4):
        values = _fracs(pusht, wall, umaze, medium)
        block = rule_a_verdict(values)
        expected_verdict, expected_clause = _reference_rule_a_verdict(values)

        assert block["verdict"] in VERDICTS, values
        assert block["clause"] in CLAUSE_VERDICT_A, values
        # No overlap: the clause determines the verdict, and only one clause fires.
        assert CLAUSE_VERDICT_A[block["clause"]] == block["verdict"], values
        # No gap, no misrouting: the shipped branch order reproduces the prose.
        assert (block["verdict"], block["clause"]) == (expected_verdict,
                                                       expected_clause), values
        # `reversal_structure` is exactly "not one of the two STOP conditions", so it
        # introduces no threshold of its own (this is what Requirement 3.6 reads).
        assert block["reversal_structure"] == (block["verdict"] != VERDICT_STOP), values
        assert block["reason"], values
        assert block["thresholds"] == {"clear_margin": RULE_A_CLEAR_MARGIN,
                                       "indistinguishable": RULE_A_INDISTINGUISHABLE}
        assert block["caps_applied"] == []


def test_rule_a_reported_numbers_agree_with_the_verdict():
    """
    A printed ratio that disagrees with the verdict would be worse than no ratio, so
    the reported margin / smoothest-ratio are checked against the verdict on the same
    grid, including the unbounded (`x / 0`) cases, which JSON carries as `null` plus a
    flag rather than as `Infinity`.
    """
    for pusht, wall, umaze, medium in itertools.product(FRACTION_GRID, repeat=4):
        values = _fracs(pusht, wall, umaze, medium)
        block = rule_a_verdict(values)
        margin = _ratio_or_inf(pusht, max(wall, umaze, medium))
        ratio_to_smoothest = _ratio_or_inf(pusht, min(values.values()))

        assert block["margin_over_largest_other_unbounded"] == (margin == math.inf)
        assert block["ratio_to_smoothest_unbounded"] == (ratio_to_smoothest == math.inf)
        if margin == math.inf:
            assert block["margin_over_largest_other"] is None
        assert block["pusht_is_highest"] == (pusht > max(wall, umaze, medium))
        if block["verdict"] == VERDICT_GO:
            assert margin >= RULE_A_CLEAR_MARGIN and block["umaze_is_lowest"]
        if block["clause"] == "2.7":
            assert ratio_to_smoothest < RULE_A_INDISTINGUISHABLE


def test_rule_b_is_total_over_the_R_grid():
    """Every `R` in the grid, `None` included, lands on exactly one verdict."""
    for value in R_GRID:
        block = rule_b_verdict(value)
        assert block["verdict"] in VERDICTS, value
        assert CLAUSE_VERDICT_B[block["clause"]] == block["verdict"], value
        assert block["reason"], value
        assert block["thresholds"] == {"go": RULE_B_GO, "middle": RULE_B_MIDDLE}
        assert block["caps_applied"] == []


def test_rule_b_is_monotone_in_R():
    """More reallocation is never a more severe verdict."""
    severity = {VERDICT_STOP: 0, VERDICT_MIDDLE: 1, VERDICT_GO: 2}
    numeric = sorted(value for value in R_GRID if value is not None)
    verdicts = [severity[rule_b_verdict(value)["verdict"]] for value in numeric]
    assert verdicts == sorted(verdicts), list(zip(numeric, verdicts))


def test_evaluate_stage0_is_total_over_fractions_and_R():
    """
    End to end on synthetic statistic dicts: every combination of four fractions and a
    PushT `R` produces a combined verdict in the enum, consistent with the two rule
    verdicts and with `stage1_permitted`.
    """
    grid = (0.0, 0.1, 0.25, 0.55, 1.0)
    for pusht, wall, umaze, medium in itertools.product(grid, repeat=4):
        for value in (None, 0.0, RULE_B_MIDDLE, 0.1, RULE_B_GO, 0.5):
            fracs = _fracs(pusht, wall, umaze, medium)
            evaluation = evaluate_stage0(_stats(fracs, value))
            rule_a = evaluation["rule_a"]["verdict"]
            rule_b = evaluation["rule_b"]["verdict"]
            combined = evaluation["combined"]
            assert combined["verdict"] in VERDICTS
            assert combined == combine_rule_verdicts(rule_a, rule_b)
            assert combined["stage1_permitted"] == (rule_a != VERDICT_STOP
                                                    and rule_b != VERDICT_STOP)
            assert evaluation["headline_reduction"] == ACS_HEADLINE_REDUCTION


# --------------------------------------------------------------------------
# 2. Exact boundaries - rule A, the 1.5x clear margin (Requirements 2.2, 2.4)
# --------------------------------------------------------------------------
# largest_other = 0.25, umaze strictly lowest, and `0.375 / 0.25` is exactly 1.5 in
# float64 because both are powers of two times a small integer. `0.30 / 0.20` is
# 1.4999999999999998 and would silently test the wrong side.
_MARGIN_LARGEST_OTHER = 0.25
_MARGIN_AT = 0.375                       # exactly 1.5 * 0.25


def _margin_case(pusht):
    return _fracs(pusht, _MARGIN_LARGEST_OTHER, 0.125, 0.1875)


@pytest.mark.parametrize("pusht, expected_verdict, expected_clause", [
    (math.nextafter(_MARGIN_AT, 0.0), VERDICT_MIDDLE, "2.4"),   # one ULP below 1.5x
    (0.37, VERDICT_MIDDLE, "2.4"),                              # 1.48x
    (_MARGIN_AT, VERDICT_GO, "2.2"),                            # exactly 1.5x
    (math.nextafter(_MARGIN_AT, 1.0), VERDICT_GO, "2.2"),       # one ULP above
    (0.5, VERDICT_GO, "2.2"),                                   # 2.0x
])
def test_rule_a_clear_margin_boundary(pusht, expected_verdict, expected_clause):
    """`>= 1.5x` is inclusive: exactly 1.5x is a GO, one ULP below is a MIDDLE."""
    block = rule_a_verdict(_margin_case(pusht))
    assert (block["verdict"], block["clause"]) == (expected_verdict, expected_clause)


def test_rule_a_margin_fixture_is_exactly_on_the_boundary():
    """Guards the fixture itself: the 'at' case really is 1.5x, not 1.4999...."""
    assert _MARGIN_AT / _MARGIN_LARGEST_OTHER == RULE_A_CLEAR_MARGIN
    assert math.nextafter(_MARGIN_AT, 0.0) / _MARGIN_LARGEST_OTHER < RULE_A_CLEAR_MARGIN
    assert math.nextafter(_MARGIN_AT, 1.0) / _MARGIN_LARGEST_OTHER > RULE_A_CLEAR_MARGIN


# --------------------------------------------------------------------------
# 2. Exact boundaries - rule A, the 1.1x indistinguishable STOP (Requirement 2.7)
# --------------------------------------------------------------------------
# All three others at 0.5, so the smoothest is 0.5 and `(1.1 * 0.5) / 0.5` is exactly
# 1.1: multiplying and dividing by 0.5 are both exact.
_SMOOTHEST = 0.5
_INDIST_AT = RULE_A_INDISTINGUISHABLE * _SMOOTHEST      # 0.55, exactly 1.1 * 0.5


def _indistinguishable_case(pusht):
    return _fracs(pusht, _SMOOTHEST, _SMOOTHEST, _SMOOTHEST)


@pytest.mark.parametrize("pusht, expected_verdict, expected_clause", [
    (math.nextafter(_INDIST_AT, 0.0), VERDICT_STOP, "2.7"),      # one ULP below 1.1x
    (0.52, VERDICT_STOP, "2.7"),                                 # 1.04x
    (_INDIST_AT, VERDICT_MIDDLE, "2.3"),                         # exactly 1.1x
    (math.nextafter(_INDIST_AT, 1.0), VERDICT_MIDDLE, "2.3"),    # one ULP above
    (0.6, VERDICT_MIDDLE, "2.3"),                                # 1.2x
])
def test_rule_a_indistinguishable_boundary(pusht, expected_verdict, expected_clause):
    """
    `within 1.1x` is strict: exactly 1.1x is *not* within, so it clears the STOP and
    becomes a MIDDLE (clause 2.3 here, because with the other three tied at 0.5 UMaze
    is not strictly the lowest).
    """
    block = rule_a_verdict(_indistinguishable_case(pusht))
    assert (block["verdict"], block["clause"]) == (expected_verdict, expected_clause)


def test_rule_a_indistinguishable_fixture_is_exactly_on_the_boundary():
    assert _INDIST_AT / _SMOOTHEST == RULE_A_INDISTINGUISHABLE
    assert math.nextafter(_INDIST_AT, 0.0) / _SMOOTHEST < RULE_A_INDISTINGUISHABLE
    assert math.nextafter(_INDIST_AT, 1.0) / _SMOOTHEST > RULE_A_INDISTINGUISHABLE


# --------------------------------------------------------------------------
# 2. Exact boundaries - the strict "highest" / "lowest" comparisons
# --------------------------------------------------------------------------
def test_rule_a_tie_at_the_top_is_stop():
    """"Highest" is strict: PushT equal to the largest other is not the highest."""
    block = rule_a_verdict(_fracs(0.3, 0.3, 0.1, 0.2))
    assert (block["verdict"], block["clause"]) == (VERDICT_STOP, "2.6")
    assert block["pusht_is_highest"] is False


def test_rule_a_all_four_equal_is_stop():
    block = rule_a_verdict(_fracs(0.2, 0.2, 0.2, 0.2))
    assert (block["verdict"], block["clause"]) == (VERDICT_STOP, "2.6")


def test_rule_a_umaze_tie_for_lowest_is_middle_not_go():
    """
    "Lowest" is strict too. The margin here is 3x, comfortably past 1.5x, but UMaze
    ties Medium at the bottom, so the ordering behind PushT is not the predicted one
    and the mechanism claim is downgraded (Requirement 2.3).
    """
    block = rule_a_verdict(_fracs(0.75, 0.25, 0.125, 0.125))
    assert (block["verdict"], block["clause"]) == (VERDICT_MIDDLE, "2.3")
    assert block["umaze_is_lowest"] is False


def test_rule_a_go_requires_umaze_lowest():
    """Same margin, UMaze strictly lowest: GO (Requirement 2.2)."""
    block = rule_a_verdict(_fracs(0.75, 0.25, 0.1, 0.125))
    assert (block["verdict"], block["clause"]) == (VERDICT_GO, "2.2")
    assert block["umaze_is_lowest"] is True


def test_rule_a_wall_above_medium_still_go_when_umaze_is_lowest():
    """
    Requirement 2.3's "e.g. Wall > Medium" is only a MIDDLE where it makes UMaze not
    the lowest. The pre-registered GO condition names PushT highest, the 1.5x margin
    and UMaze lowest, and this case satisfies all three.
    """
    block = rule_a_verdict(_fracs(0.75, 0.5, 0.125, 0.25))
    assert (block["verdict"], block["clause"]) == (VERDICT_GO, "2.2")


# --------------------------------------------------------------------------
# 2. Exact boundaries - rule B, 0.15 and 0.08 (Requirements 2.8, 2.9, 2.11)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("value, expected_verdict, expected_clause", [
    (0.0, VERDICT_STOP, "2.11"),
    (math.nextafter(RULE_B_MIDDLE, 0.0), VERDICT_STOP, "2.11"),     # one ULP below
    (0.0799, VERDICT_STOP, "2.11"),
    (RULE_B_MIDDLE, VERDICT_MIDDLE, "2.9"),                         # exactly 0.08
    (math.nextafter(RULE_B_MIDDLE, 1.0), VERDICT_MIDDLE, "2.9"),    # one ULP above
    (0.1, VERDICT_MIDDLE, "2.9"),
    (math.nextafter(RULE_B_GO, 0.0), VERDICT_MIDDLE, "2.9"),        # one ULP below
    (0.1499, VERDICT_MIDDLE, "2.9"),
    (RULE_B_GO, VERDICT_GO, "2.8"),                                 # exactly 0.15
    (math.nextafter(RULE_B_GO, 1.0), VERDICT_GO, "2.8"),            # one ULP above
    (0.4, VERDICT_GO, "2.8"),
])
def test_rule_b_boundaries(value, expected_verdict, expected_clause):
    """Both rule-B thresholds are inclusive at the bottom of their band."""
    block = rule_b_verdict(value)
    assert (block["verdict"], block["clause"]) == (expected_verdict, expected_clause)
    assert block["R"] == pytest.approx(value, abs=1e-9)


def test_rule_b_none_is_stop():
    """`R = None` is the probe's `mean(w) = 0`: the gate zeroes every triple."""
    block = rule_b_verdict(None)
    assert (block["verdict"], block["clause"]) == (VERDICT_STOP, "2.11")
    assert block["R"] is None


# --------------------------------------------------------------------------
# 3. Combination (Requirements 2.12, 2.13, 2.14)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("rule_a", VERDICTS)
@pytest.mark.parametrize("rule_b", VERDICTS)
def test_combine_covers_all_nine_pairs(rule_a, rule_b):
    combined = combine_rule_verdicts(rule_a, rule_b)
    either_stop = VERDICT_STOP in (rule_a, rule_b)
    both_go = rule_a == VERDICT_GO and rule_b == VERDICT_GO

    if either_stop:
        assert combined["verdict"] == VERDICT_STOP
        assert combined["stage1_permitted"] is False
    elif both_go:
        assert combined["verdict"] == VERDICT_GO
        assert combined["stage1_permitted"] is True
    else:
        # Both at least MIDDLE: Stage 1 permitted with the downgraded claim.
        assert combined["verdict"] == VERDICT_MIDDLE
        assert combined["stage1_permitted"] is True
    assert combined["rule_a"] == rule_a and combined["rule_b"] == rule_b
    assert combined["reason"]


def test_combine_rejects_a_verdict_outside_the_enum():
    with pytest.raises(ValueError, match="rule A verdict"):
        combine_rule_verdicts("go", VERDICT_GO)
    with pytest.raises(ValueError, match="rule B verdict"):
        combine_rule_verdicts(VERDICT_GO, None)


# --------------------------------------------------------------------------
# 5. Requirement 3.6's downgrade is a cap, never an upgrade
# --------------------------------------------------------------------------
def test_requirement_3_6_caps_raw_go_and_never_lifts_the_headline_stop():
    """
    `sum` has no reversal structure (PushT not the highest) while `raw` does and would
    be a GO on its own. The downgrade fires, `raw`'s verdict is capped at MIDDLE, and
    the headline verdict stays STOP: the cap can only lower a verdict.
    """
    stats = {ACS_HEADLINE_REDUCTION: {
        "pusht": {"frac_cos_lt_0": 0.1, "reallocation_R": 0.3},
        "wall": {"frac_cos_lt_0": 0.4, "reallocation_R": 0.3},
        "umaze": {"frac_cos_lt_0": 0.2, "reallocation_R": 0.3},
        "medium": {"frac_cos_lt_0": 0.3, "reallocation_R": 0.3}},
        ACS_WITHIN_STEP_REDUCTION: {
        "pusht": {"frac_cos_lt_0": 0.75, "reallocation_R": 0.3},
        "wall": {"frac_cos_lt_0": 0.25, "reallocation_R": 0.3},
        "umaze": {"frac_cos_lt_0": 0.1, "reallocation_R": 0.3},
        "medium": {"frac_cos_lt_0": 0.125, "reallocation_R": 0.3}}}

    evaluation = evaluate_stage0(stats)
    detail = evaluation["requirement_3_6"]
    assert detail["measured"] is True
    assert detail["applies"] is True
    assert detail["sum_has_reversal_structure"] is False
    assert detail["raw_has_reversal_structure"] is True
    assert detail["raw_rule_a_verdict"] == VERDICT_GO
    assert detail["raw_rule_a_verdict_capped"] == VERDICT_MIDDLE
    assert evaluation["rule_a"]["verdict"] == VERDICT_STOP
    assert evaluation["combined"]["verdict"] == VERDICT_STOP
    assert evaluation["combined"]["stage1_permitted"] is False


def test_requirement_3_6_does_not_fire_when_sum_has_structure():
    stats = _stats(_fracs(0.75, 0.25, 0.1, 0.125), 0.3)
    stats[ACS_WITHIN_STEP_REDUCTION] = dict(stats[ACS_HEADLINE_REDUCTION])
    evaluation = evaluate_stage0(stats)
    assert evaluation["requirement_3_6"]["applies"] is False
    assert evaluation["rule_a"]["verdict"] == VERDICT_GO
    assert evaluation["rule_a"]["caps_applied"] == []
    assert evaluation["combined"]["verdict"] == VERDICT_GO


def test_evaluate_stage0_requires_the_headline_reduction():
    stats = _stats(_fracs(0.75, 0.25, 0.1, 0.125), 0.3,
                   reduction=ACS_WITHIN_STEP_REDUCTION)
    with pytest.raises(ValueError, match=ACS_HEADLINE_REDUCTION):
        evaluate_stage0(stats)


def test_evaluate_stage0_skips_an_incomplete_non_headline_reduction():
    stats = _stats(_fracs(0.75, 0.25, 0.1, 0.125), 0.3)
    stats["first"] = {"pusht": {"frac_cos_lt_0": 0.4, "reallocation_R": 0.3}}
    evaluation = evaluate_stage0(stats)
    assert "skipped" in evaluation["per_reduction"]["first"]
    assert evaluation["rule_a"]["verdict"] == VERDICT_GO


# --------------------------------------------------------------------------
# Input validation on the rule inputs
# --------------------------------------------------------------------------
def test_rule_a_rejects_a_missing_environment():
    with pytest.raises(ValueError, match="missing medium"):
        rule_a_verdict({"pusht": 0.4, "wall": 0.2, "umaze": 0.1})


def test_rule_a_rejects_an_unknown_environment():
    values = _fracs(0.4, 0.2, 0.1, 0.15)
    values["rope"] = 0.3
    with pytest.raises(ValueError, match="rope"):
        rule_a_verdict(values)


@pytest.mark.parametrize("bad", [-0.01, 1.01, float("nan"), float("inf")])
def test_rule_a_rejects_values_outside_the_unit_interval(bad):
    with pytest.raises(ValueError, match="fraction in .0, 1."):
        rule_a_verdict(_fracs(bad, 0.2, 0.1, 0.15))


def test_rule_a_rejects_a_non_dict_and_a_non_number():
    with pytest.raises(ValueError, match="dict of frac"):
        rule_a_verdict([0.4, 0.2, 0.1, 0.15])
    with pytest.raises(ValueError, match="not a number"):
        rule_a_verdict(_fracs("high", 0.2, 0.1, 0.15))


@pytest.mark.parametrize("bad", [-0.01, float("nan"), float("inf")])
def test_rule_b_rejects_non_finite_and_negative_R(bad):
    with pytest.raises(ValueError, match="non-negative finite"):
        rule_b_verdict(bad)


@pytest.mark.parametrize("bad", ["lots", [0.2], {"R": 0.2}])
def test_rule_b_rejects_a_non_number(bad):
    with pytest.raises(ValueError, match="not a number"):
        rule_b_verdict(bad)


def test_rule_b_accepts_a_numeric_string_from_json():
    """
    Both rules coerce with `float()`, so a JSON report that stored `R` as a string
    still evaluates rather than crashing halfway through a summarize run. Pinned as
    intended behaviour, not discovered later from a traceback.
    """
    assert rule_b_verdict("0.2")["verdict"] == VERDICT_GO
    assert rule_a_verdict(_fracs("0.75", 0.25, 0.1, 0.125))["verdict"] == VERDICT_GO


# --------------------------------------------------------------------------
# `--table1-gains` parsing and the reported (non-gating) ordering count
# --------------------------------------------------------------------------
def test_parse_table1_gains_default():
    assert parse_table1_gains(ACS_TABLE1_GAINS_DEFAULT) == {
        "umaze": 50.0, "medium": 10.67, "wall": 10.67, "pusht": 7.33}


@pytest.mark.parametrize("raw, match", [
    ("umaze=50.00,medium=10.67,wall=10.67", "missing pusht"),
    ("umaze=50.00,medium=10.67,wall=10.67,pusht=7.33,pusht=1", "twice"),
    ("umaze=50.00,medium=10.67,wall=10.67,rope=1", "not one of"),
    ("umaze=50.00,medium=10.67,wall=10.67,pusht=lots", "not a number"),
    ("umaze", "not KEY=VALUE"),
])
def test_parse_table1_gains_rejects_bad_input(raw, match):
    with pytest.raises(ValueError, match=match):
        parse_table1_gains(raw)


def test_ordering_vs_table1_gains_is_concordant_on_the_predicted_ordering():
    """
    The predicted inverse ordering (UMaze lowest, PushT highest) is fully concordant,
    and the Wall / Medium pair - equal gains at +10.67 - carries no prediction and is
    skipped rather than counted.
    """
    gains = parse_table1_gains(ACS_TABLE1_GAINS_DEFAULT)
    ordering = ordering_vs_table1_gains(_fracs(0.75, 0.25, 0.1, 0.2), gains)
    assert ordering["tied_gain_pairs"] == [["wall", "medium"]]
    assert ordering["pairs_compared"] == 5
    assert ordering["pairs_discordant"] == 0
    assert ordering["matches_inverse_gains"] is True
    assert ordering["observed_order_desc_frac"] == ["pusht", "wall", "medium", "umaze"]


def test_ordering_vs_table1_gains_counts_a_discordant_pair():
    gains = parse_table1_gains(ACS_TABLE1_GAINS_DEFAULT)
    ordering = ordering_vs_table1_gains(_fracs(0.75, 0.1, 0.25, 0.2), gains)
    assert ordering["matches_inverse_gains"] is False
    assert ordering["pairs_discordant"] > 0


def test_evaluate_stage0_reports_the_ordering_only_when_gains_are_given():
    stats = _stats(_fracs(0.75, 0.25, 0.1, 0.2), 0.3)
    assert "table1_ordering" not in evaluate_stage0(stats)
    with_gains = evaluate_stage0(stats, parse_table1_gains(ACS_TABLE1_GAINS_DEFAULT))
    assert with_gains["table1_ordering"]["matches_inverse_gains"] is True
    # Reported, never gating: the verdict is read off rule A.
    assert with_gains["combined"]["verdict"] == with_gains["rule_a"]["verdict"]
