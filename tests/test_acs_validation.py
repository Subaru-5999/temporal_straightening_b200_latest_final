"""Task 2.4 - eager validation of the ACS enums and of the `straighten` mode string.

Feature: action-conditioned-straightening, Property 13: Enum and mode-string validation is
eager. ∀ invalid `acs_action_reduce` / `acs_gate`: `__init__` raises **even when
`straighten="aggcos1e-1"`**, i.e. when ACS is not selected. ∀ non-empty `straighten` string
matching no known prefix, and ∀ `acsaggcos<suffix>` with a non-numeric suffix: `__init__`
raises. Closes F4's silent-disable hole.

This module is a **gate**, deliberately not optional. F4 recorded the live landmine it closes:
before the parser's ``else: raise``, a typo such as ``acsagcos1e-1`` left ``curvature_mode =
None``, set ``straighten = False``, logged ``"Straightening disabled"`` in a wall of startup
lines and then trained a full 12-hour run **with no curvature term at all**. The difference
between that and a ``ValueError`` at second zero is the whole point of the test, so every
assertion here is about ``__init__`` and none of them needs a forward pass.

Two things the tests are careful about:

1. **"While ACS is not selected."** The enum properties build the model with
   ``straighten="aggcos1e-1"`` (and, parametrized, ``False`` and ``"cos1e-1"``), so the raise
   cannot be an artifact of the ACS path being live. ``test_valid_enums_build_with_acs_not_selected``
   is the matching negative control: it asserts the same configuration builds *and* that
   ``curvature_mode`` really is ``"aggcos"`` rather than ``"acsaggcos"``.
2. **No flaky "invalid" strings.** ``_parser_accepts`` mirrors the shipped parser exactly -
   prefix order ``acsaggcos`` → ``aggcos`` → ``cos``, empty suffix means scale 1.0, otherwise
   ``float(suffix)`` and then ``scale <= 0`` - and every generated string is filtered through
   it. So ``"cos  1e-1"`` (``float`` tolerates surrounding whitespace) and ``"aggcos"`` (bare
   prefix, scale 1.0) are never asserted to raise.

3. **The non-finite residue.** ``float()`` accepts ``"nan"`` and ``"inf"``, so ``"cosnan"``
   used to parse to ``scale = nan``, slip past ``scale <= 0`` and then evaluate ``nan > 0`` as
   ``False`` - F4's exact failure mode reached through the *accepted* branch rather than the
   fall-through. ``test_non_finite_scale_raises`` and
   ``test_recorded_non_finite_strings_never_disable_silently`` pin the finiteness check that
   closes it, and ``_parser_accepts`` mirrors it.

``straighten`` values ``False``, ``None`` and ``""`` all mean "off" and must **not** raise:
Requirements 6.2 / 9.2 scope the raise to non-empty strings, which is why the invalid-string
strategy generates only non-empty ones and ``test_off_values_do_not_raise`` pins the other side.

Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 9.1, 9.2, 9.4
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.visual_world_model import (  # noqa: E402
    ACS_ACTION_REDUCTIONS,
    ACS_GATES,
    STRAIGHTEN_FORMS,
)
from tests.conftest import build_stub_world_model  # noqa: E402

# The parser's prefixes in the order it tests them. Specificity order, so `acsaggcos1e-1`
# cannot be swallowed by the `aggcos` branch.
KNOWN_PREFIXES = ("acsaggcos", "aggcos", "cos")

# `straighten` values that mean "off". None of these may raise (Requirements 6.2, 9.2 scope
# the raise to non-empty strings).
OFF_VALUES = (False, None, "")

# The motivating typo from design F4: one dropped `g`, no known prefix, and before the
# `else: raise` it silently trained a null run.
MOTIVATING_TYPO = "acsagcos1e-1"

# Hand-picked unrecognized strings, kept alongside the generated ones so the shrunk
# counterexample of a regression is readable rather than a random blob.
UNKNOWN_STRAIGHTEN_EXAMPLES = (
    MOTIVATING_TYPO,
    "acsaggos1e-1",
    "acsagg1e-1",
    "acscos1e-1",
    "aggcs1e-1",
    "agcos1e-1",
    "co1e-1",
    "ACSAGGCOS1e-1",
    "AggCos1e-1",
    "Cos1e-1",
    " cos1e-1",
    "_cos1e-1",
    "curvature1e-1",
    "true",
    "True",
    "on",
    "1e-1",
    "0.1",
    "-",
    " ",
)

# Suffixes `float()` *accepts* and that are not finite numbers. This is the residual half of
# F4's silent-disable hole, found while writing this module: `float("nan")` succeeds, so
# `straighten="cosnan"` used to parse to `scale = nan`, slip past `scale <= 0` (which is
# `False` for `nan`), and then set `self.straighten = curvature_mode is not None and
# straighten_scale > 0` to `False` because `nan > 0` is `False`. The result is exactly the
# failure the `else: raise` was added to close: `curvature_mode == "cos"`, a startup log
# reading "Straightening disabled", and a 12-hour run with no curvature term. `"cosinf"` is
# the other direction - it enables the term and makes the loss infinite on step one.
NON_FINITE_SUFFIX_EXAMPLES = (
    "nan",
    "NaN",
    "NAN",
    "-nan",
    "+nan",
    "inf",
    "INF",
    "Inf",
    "+inf",
    "-inf",
    "infinity",
    "Infinity",
    "-Infinity",
    " nan ",
    " inf",
)

# Suffixes `float()` refuses. Appended to each known prefix, these select a mode and then
# fail to parse a scale for it (Requirement 6.3).
NON_NUMERIC_SUFFIX_EXAMPLES = (
    "abc",
    "1e",
    "1e-",
    "e-1",
    "1,0",
    "1.2.3",
    "--1",
    "0x10",
    "1e-1x",
    "1 e-1",
    "1/10",
    "one",
    "?",
    "1e-1;",
)


def _is_float(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def _parser_accepts(value) -> bool:
    """Mirror of the shipped `straighten` parser: True iff `__init__` does **not** raise.

    Kept deliberately literal rather than clever, because its whole job is to be a faithful
    restatement of ``VWorldModel.__init__``'s branch structure so that the strategies below
    cannot generate a legitimately-parsing string and call it invalid.
    """
    if not isinstance(value, str) or value == "":
        return True  # 'off', and off never raises
    for prefix in KNOWN_PREFIXES:
        if value.startswith(prefix):
            suffix = value.replace(prefix, "", 1)
            if suffix == "":
                return True  # bare prefix -> scale 1.0
            if not _is_float(suffix):
                return False  # Requirement 6.3
            scale = float(suffix)
            if not math.isfinite(scale):
                return False  # non-finite scale: 'cosnan' / 'cosinf'
            return not (scale <= 0)  # Requirement 6.4
    return False  # Requirement 6.2 / 9.2: no known prefix


def _invalid_enum_strategy(valid: tuple[str, ...]):
    """Non-members of a closed enum: free text plus the near-misses a human actually types."""
    near_misses = tuple(v.upper() for v in valid)
    near_misses += tuple(v.capitalize() for v in valid)
    near_misses += tuple(v + " " for v in valid)
    near_misses += tuple(" " + v for v in valid)
    near_misses += tuple(v.replace("_", "-") for v in valid)
    return st.one_of(
        st.sampled_from(near_misses),
        st.sampled_from(("", "mean", "avg", "none", "None", "relu", "cos", "0", "1", "true")),
        st.text(min_size=0, max_size=12),
    ).filter(lambda s: s not in valid)


invalid_action_reduce_strategy = _invalid_enum_strategy(ACS_ACTION_REDUCTIONS)
invalid_gate_strategy = _invalid_enum_strategy(ACS_GATES)

def _prefix_typos() -> tuple[str, ...]:
    """Every single-character deletion of every known prefix, with a valid scale appended.

    `acsagcos1e-1` is one of these; generating the family rather than the one recorded typo is
    what makes the property a statement about the parser instead of about a single string.
    """
    out = []
    for prefix in KNOWN_PREFIXES:
        for i in range(len(prefix)):
            out.append(prefix[:i] + prefix[i + 1 :] + "1e-1")
    return tuple(out)


# Non-empty strings matching **no** known prefix. This is the Requirement 6.2 / 9.2 class, the
# one F4 recorded: before the `else: raise` these fell through to `curvature_mode = None`.
unknown_prefix_strategy = st.one_of(
    st.sampled_from(UNKNOWN_STRAIGHTEN_EXAMPLES),
    st.sampled_from(_prefix_typos()),
    st.text(min_size=1, max_size=14),
).filter(lambda s: s != "" and not any(s.startswith(p) for p in KNOWN_PREFIXES))

# Every invalid non-empty string, all three classes at once: unknown prefix, non-numeric
# suffix, non-positive scale. Everything the parser would legitimately accept is filtered out
# through `_parser_accepts`, so a failure here is always a real defect and never a generator
# artifact.
invalid_straighten_strategy = st.one_of(
    unknown_prefix_strategy,
    st.builds(
        lambda prefix, suffix: prefix + suffix,
        st.sampled_from(KNOWN_PREFIXES),
        st.one_of(
            st.sampled_from(NON_NUMERIC_SUFFIX_EXAMPLES),
            st.sampled_from(NON_FINITE_SUFFIX_EXAMPLES),
            st.text(min_size=1, max_size=8),
            st.floats(min_value=-1e3, max_value=0.0, allow_nan=False, allow_infinity=False).map(
                repr
            ),
        ),
    ),
).filter(lambda s: s != "" and not _parser_accepts(s))

# `<known prefix><non-finite suffix>`, the residual silent-disable case on its own.
non_finite_scale_strategy = st.builds(
    lambda prefix, suffix: prefix + suffix,
    st.sampled_from(KNOWN_PREFIXES),
    st.sampled_from(NON_FINITE_SUFFIX_EXAMPLES),
)

# `<known prefix><non-numeric suffix>`, the Requirement 6.3 case on its own.
non_numeric_suffix_strategy = st.builds(
    lambda prefix, suffix: prefix + suffix,
    st.sampled_from(KNOWN_PREFIXES),
    st.one_of(
        st.sampled_from(NON_NUMERIC_SUFFIX_EXAMPLES),
        st.text(min_size=1, max_size=8).filter(lambda s: not _is_float(s)),
    ),
)

# `scale <= 0`, formatted through `repr` so `float()` always parses it back (Requirement 6.4).
non_positive_scale_strategy = st.builds(
    lambda prefix, scale: prefix + repr(scale),
    st.sampled_from(KNOWN_PREFIXES),
    st.one_of(
        st.just(0.0),
        st.just(-0.0),
        st.floats(
            min_value=-1e3, max_value=0.0, allow_nan=False, allow_infinity=False
        ),
    ),
)

# The three straighten settings under which the enum validation must still fire: a baseline
# aggcos run, a patch-space run, and straightening off entirely. None of them selects ACS.
ACS_NOT_SELECTED = ("aggcos1e-1", "cos1e-1", False)


# ---------------------------------------------------------------------------
# Property 13 - the enums (Requirements 6.5, 6.6, 9.1)
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(bad=invalid_action_reduce_strategy, straighten=st.sampled_from(ACS_NOT_SELECTED))
def test_invalid_action_reduce_raises_while_acs_not_selected(bad, straighten):
    """An out-of-enum `acs_action_reduce` raises in `__init__` even with ACS unselected."""
    with pytest.raises(ValueError) as excinfo:
        build_stub_world_model(straighten=straighten, acs_action_reduce=bad)

    message = str(excinfo.value)
    assert "acs_action_reduce" in message, message
    for accepted in ACS_ACTION_REDUCTIONS:
        assert accepted in message, message
    assert repr(bad) in message, message


@settings(max_examples=100, deadline=None)
@given(bad=invalid_gate_strategy, straighten=st.sampled_from(ACS_NOT_SELECTED))
def test_invalid_gate_raises_while_acs_not_selected(bad, straighten):
    """An out-of-enum `acs_gate` raises in `__init__` even with ACS unselected."""
    with pytest.raises(ValueError) as excinfo:
        build_stub_world_model(straighten=straighten, acs_gate=bad)

    message = str(excinfo.value)
    assert "acs_gate" in message, message
    for accepted in ACS_GATES:
        assert accepted in message, message
    assert repr(bad) in message, message


@settings(max_examples=100, deadline=None)
@given(
    action_reduce=st.sampled_from(ACS_ACTION_REDUCTIONS + (None,)),
    gate=st.sampled_from(ACS_GATES + (None,)),
)
def test_valid_enums_build_with_acs_not_selected(action_reduce, gate):
    """The negative control: in-enum values (and `None`, meaning "default") do not raise.

    Also pins that the raise in the two properties above is genuinely *eager* rather than a
    side effect of ACS being live: ``curvature_mode`` here is ``"aggcos"``, so no ACS code
    path is reachable, and the model still carries the two validated knobs.
    """
    model = build_stub_world_model(
        straighten="aggcos1e-1", acs_action_reduce=action_reduce, acs_gate=gate
    )

    assert model.curvature_mode == "aggcos"
    assert model.acs_action_reduce == ("sum" if action_reduce is None else action_reduce)
    assert model.acs_gate == ("relu_cos" if gate is None else gate)


# ---------------------------------------------------------------------------
# Property 13 - the mode string (Requirements 6.1, 6.2, 6.3, 6.4, 9.2)
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(bad=invalid_straighten_strategy)
def test_every_invalid_straighten_string_raises(bad):
    """No non-empty `straighten` string that fails to parse may reach the training loop.

    All three failure classes at once - unknown prefix, non-numeric suffix, non-positive scale
    - because the guarantee F4 needs is about the *absence of a fall-through*, not about which
    of the three messages fires.
    """
    with pytest.raises(ValueError) as excinfo:
        build_stub_world_model(straighten=bad)

    assert "straighten" in str(excinfo.value), str(excinfo.value)


@settings(max_examples=100, deadline=None)
@given(bad=unknown_prefix_strategy)
def test_unrecognized_straighten_string_raises(bad):
    """A non-empty `straighten` string matching no known prefix raises, never disables.

    This is F4's hole. The message must name the accepted forms, so a reader of the traceback
    does not have to open the parser to find out what to type instead.
    """
    with pytest.raises(ValueError) as excinfo:
        build_stub_world_model(straighten=bad)

    message = str(excinfo.value)
    assert "straighten" in message, message
    assert "matches no known curvature mode" in message, message
    for form in STRAIGHTEN_FORMS:
        assert form in message, message


def test_motivating_typo_raises_rather_than_training_a_null_run():
    """`acsagcos1e-1` - one dropped `g` - is the recorded 12-hour null run. It must raise."""
    with pytest.raises(ValueError, match="matches no known curvature mode"):
        build_stub_world_model(straighten=MOTIVATING_TYPO)


@settings(max_examples=100, deadline=None)
@given(bad=non_numeric_suffix_strategy)
def test_non_numeric_scale_suffix_raises(bad):
    """A recognized prefix with a suffix `float()` refuses raises during `__init__`."""
    with pytest.raises(ValueError) as excinfo:
        build_stub_world_model(straighten=bad)

    message = str(excinfo.value)
    assert "straighten" in message, message
    assert "non-numeric scale suffix" in message, message


@settings(max_examples=100, deadline=None)
@given(bad=non_positive_scale_strategy)
def test_non_positive_scale_raises(bad):
    """`scale <= 0` names the term while disabling it, so it raises instead of proceeding."""
    with pytest.raises(ValueError) as excinfo:
        build_stub_world_model(straighten=bad)

    message = str(excinfo.value)
    assert "straighten" in message, message
    assert "curvature scale" in message, message


@settings(max_examples=100, deadline=None)
@given(bad=non_finite_scale_strategy)
def test_non_finite_scale_raises(bad):
    """`float()` accepts 'nan' and 'inf', so the parser must reject them explicitly.

    Without this check `"cosnan"` selects `curvature_mode = "cos"`, sets
    `straighten_scale = nan`, passes `scale <= 0` (False for `nan`), and then fails
    `nan > 0` - so the run logs "Straightening disabled" while naming a curvature mode and
    trains with no curvature term. That is F4's failure mode reached through the *accepted*
    branch instead of the fall-through. `"cosinf"` is the mirror image: it enables the term
    and the loss is infinite from the first step.
    """
    with pytest.raises(ValueError) as excinfo:
        build_stub_world_model(straighten=bad)

    message = str(excinfo.value)
    assert "straighten" in message, message
    assert "non-finite curvature scale" in message, message
    for form in STRAIGHTEN_FORMS:
        assert form in message, message


@pytest.mark.parametrize("bad", ("cosnan", "cosinf", "aggcos-inf", "acsaggcosnan"))
def test_recorded_non_finite_strings_never_disable_silently(bad):
    """The four recorded strings raise instead of building a model with a non-finite scale."""
    with pytest.raises(ValueError, match="non-finite curvature scale"):
        build_stub_world_model(straighten=bad)


@settings(max_examples=100, deadline=None)
@given(off=st.sampled_from(OFF_VALUES))
def test_off_values_do_not_raise(off):
    """`False`, `None` and `""` all mean "off" and must build, with straightening disabled."""
    model = build_stub_world_model(straighten=off)

    assert model.straighten is False
    assert model.curvature_mode is None
    assert model.straighten_scale == 0.0


@settings(max_examples=100, deadline=None)
@given(
    scale_text=st.sampled_from(("1e-1", "0.1", "1", "1e-2", "5e-1", "10")),
    action_reduce=st.sampled_from(ACS_ACTION_REDUCTIONS),
    gate=st.sampled_from(ACS_GATES),
)
def test_acsaggcos_prefix_selects_acs_and_parses_its_scale(scale_text, action_reduce, gate):
    """Requirement 6.1: the new prefix sets `curvature_mode` and reads the scale suffix.

    Included here rather than in a separate module because it is the positive half of the same
    parser branch the four raising properties above cover: the prefix must be *recognized*, not
    merely not-silently-ignored.
    """
    model = build_stub_world_model(
        straighten=f"acsaggcos{scale_text}",
        acs_action_reduce=action_reduce,
        acs_gate=gate,
    )

    assert model.curvature_mode == "acsaggcos"
    assert model.straighten_scale == float(scale_text)
    assert model.straighten is True
    assert model.acs_action_reduce == action_reduce
    assert model.acs_gate == gate
