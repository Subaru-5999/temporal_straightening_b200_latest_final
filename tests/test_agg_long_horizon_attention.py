"""Task 11.3b: the long-horizon attention deviation is gated on the horizon regime.

Feature: ``aggregated-space-planning-cost``.

**Why this file exists.** Task 11.4's reading (a) (``goal_H 50``) OOM'd the ``1g.45gb`` MIG slice on
the materialised attention path, so the long-horizon Positive_Control runs through
``F.scaled_dot_product_attention``. That is a deviation from the implementation every recorded number
on this platform was measured on: the Platform_Baseline (75.33 open-loop / 82.00 MPC) and task 11.1's
paired zero-weight check, which found this wrapper at ``agg_weight=0`` *numerically identical* to
frozen ``plan.py``.

The whole safety of the deviation is that it is **keyed on the horizon regime** and therefore cannot
reach the reported short-horizon result (Requirement 7.2). Nothing else pins that. A single
character -- ``"sdpa"`` written into the short row, or an ``attention_scope`` that enables SDPA
unconditionally -- would move the reported result onto an unmeasured path with no error and no
visible symptom, because SDPA computes the same function and the run would simply produce slightly
different numbers.

Deliberately runnable with no CUDA and no hydra: ``models.vit``'s module scope imports ``torch`` but
touches no device (only ``Attention.__init__`` does, via ``.to('cuda')``), and the class-level switch
plus its context manager are plain Python.
"""

from __future__ import annotations

import contextlib

import pytest

import plan_agg


SHORT = plan_agg.HORIZON_REGIMES[plan_agg.SHORT_GOAL_H]
LONG = plan_agg.HORIZON_REGIMES[plan_agg.LONG_GOAL_H]


# ---------------------------------------------------------------------------
# The table itself
# ---------------------------------------------------------------------------


def test_short_horizon_keeps_the_materialised_path():
    """The reported result stays on the path it was measured on. This is the point of the file."""
    assert plan_agg.ATTENTION_IMPL_BY_REGIME[SHORT] == "materialized"
    assert plan_agg.attention_impl_for_regime(SHORT) == "materialized"


def test_long_horizon_uses_sdpa():
    assert plan_agg.ATTENTION_IMPL_BY_REGIME[LONG] == "sdpa"
    assert plan_agg.attention_impl_for_regime(LONG) == "sdpa"


def test_every_horizon_regime_has_an_assignment_and_a_recorded_reason():
    """No regime may fall through to a default, and every choice must carry its justification."""
    for regime in plan_agg.HORIZON_REGIMES.values():
        impl = plan_agg.attention_impl_for_regime(regime)
        reason = plan_agg.ATTENTION_IMPL_REASON[impl]
        assert reason.strip(), f"regime {regime!r} -> {impl!r} has an empty reason"

    assert set(plan_agg.ATTENTION_IMPL_REASON) == set(
        plan_agg.ATTENTION_IMPL_BY_REGIME.values()
    )


def test_unknown_regime_aborts_naming_the_regimes_that_exist():
    with pytest.raises(plan_agg.ProtocolError) as excinfo:
        plan_agg.attention_impl_for_regime("medium")
    message = str(excinfo.value)
    assert "medium" in message
    for regime in plan_agg.ATTENTION_IMPL_BY_REGIME:
        assert regime in message


# ---------------------------------------------------------------------------
# The scope, against the real `models.vit` switch
# ---------------------------------------------------------------------------


@pytest.fixture()
def vit_attention():
    """The real ``models.vit.Attention`` class, with ``use_sdpa`` restored afterwards."""
    vit = pytest.importorskip("models.vit")
    previous = vit.Attention.use_sdpa
    try:
        yield vit.Attention
    finally:
        vit.Attention.use_sdpa = previous


def test_default_switch_is_off(vit_attention):
    """The premise of the short-horizon row: the shipped default is the materialised path."""
    assert vit_attention.use_sdpa is False


def test_short_horizon_scope_does_not_touch_the_switch(vit_attention):
    """Not merely restored afterwards -- never modified, so nothing can observe a flip."""
    vit_attention.use_sdpa = False
    with plan_agg.attention_scope(SHORT) as impl:
        assert impl == "materialized"
        assert vit_attention.use_sdpa is False
    assert vit_attention.use_sdpa is False


def test_long_horizon_scope_enables_sdpa_and_restores_it(vit_attention):
    vit_attention.use_sdpa = False
    with plan_agg.attention_scope(LONG) as impl:
        assert impl == "sdpa"
        assert vit_attention.use_sdpa is True
    assert vit_attention.use_sdpa is False


def test_long_horizon_scope_restores_on_exception(vit_attention):
    """A failed evaluation must not leave the fast path enabled for anything after it."""
    vit_attention.use_sdpa = False
    with contextlib.suppress(RuntimeError):
        with plan_agg.attention_scope(LONG):
            assert vit_attention.use_sdpa is True
            raise RuntimeError("evaluation blew up")
    assert vit_attention.use_sdpa is False


def test_unknown_regime_scope_aborts_before_touching_the_switch(vit_attention):
    vit_attention.use_sdpa = False
    with pytest.raises(plan_agg.ProtocolError):
        with plan_agg.attention_scope("medium"):
            pytest.fail("the scope must abort before its body runs")
    assert vit_attention.use_sdpa is False


# ---------------------------------------------------------------------------
# The manifest record (Requirement 8.6): the deviation travels with the numbers
# ---------------------------------------------------------------------------


def _record(config_name: str, goal_h: int) -> plan_agg.ProtocolRecord:
    expected = plan_agg.PROTOCOL_EXPECTED[(config_name, goal_h)]
    return plan_agg.ProtocolRecord(
        config_name=config_name,
        setting=plan_agg.SETTING_NAMES[config_name],
        resolved=dict(expected),
        expected=dict(expected),
        deviations=(),
        expected_source=plan_agg.PROTOCOL_EXPECTED_SOURCE[(config_name, goal_h)],
        horizon_regime=plan_agg.HORIZON_REGIMES[goal_h],
        goal_H=goal_h,
    )


@pytest.mark.parametrize("config_name", sorted(plan_agg.SETTING_NAMES))
@pytest.mark.parametrize(
    "goal_h", sorted(plan_agg.HORIZON_REGIMES), ids=["short", "long"]
)
def test_manifest_records_the_implementation_and_its_reason(config_name, goal_h):
    manifest = plan_agg.build_manifest(
        protocol=_record(config_name, goal_h),
        agg_weight=0.1,
        cfg={"model_name": "m", "ckpt_base_path": "./checkpoints", "seed": 100},
    )
    regime = plan_agg.HORIZON_REGIMES[goal_h]
    expected_impl = plan_agg.ATTENTION_IMPL_BY_REGIME[regime]

    assert manifest["attention_impl"] == expected_impl
    assert manifest["horizon_regime"] == regime
    assert manifest["attention_impl_reason"] == plan_agg.ATTENTION_IMPL_REASON[expected_impl]


def test_the_sdpa_reason_records_what_is_not_verified():
    """The backend-selection risk is the live failure mode; it must be in the manifest text.

    ``scaled_dot_product_attention`` is called with an explicit ``attn_mask``, which rules out the
    flash backend, and a fallback to the ``math`` backend materialises the same score matrix and
    OOMs again. That is the diagnosis a reader of a failed run needs, so it is recorded rather than
    left in a conversation.
    """
    reason = plan_agg.ATTENTION_IMPL_REASON["sdpa"]
    assert "NOT verified" in reason
    assert "math" in reason
    assert "attn_mask" in reason
    # And the fallback, so the next decision is already written down.
    assert "reading (b)" in reason


# ---------------------------------------------------------------------------
# The other half of the mechanism: the rollout must INHERIT the outer scope
# ---------------------------------------------------------------------------
#
# The gating above is necessary and was not sufficient. `plan_agg.attention_scope` opened the
# outer scope, printed `sdpa`, and the long-horizon run still OOM'd in the *materialised*
# branch at `models/vit.py:100`. Cause: `_predict_maybe_checkpointed` wraps every `predict`
# in `with sdpa_attention(fast_attention)`, and `_rollout_latents` defaulted
# `fast_attention=False`, so an **inner** scope forced the switch off on every predictor call
# and silently overrode the outer one. A context manager that sets an absolute value cannot be
# nested by a caller that only wants to opt in, which is the general shape of the bug.
#
# Fixed by `VWorldModel.resolve_fast_attention`: `None` means inherit. These tests pin the
# composition, because nothing else does and the symptom was a 13-second OOM whose traceback
# said nothing about the switch.


def test_resolve_fast_attention_inherits_the_ambient_switch(vit_attention):
    """`None` means inherit. This is the single line whose absence caused the OOM."""
    from models.visual_world_model import VWorldModel

    vit_attention.use_sdpa = False
    assert VWorldModel.resolve_fast_attention(None) is False
    vit_attention.use_sdpa = True
    assert VWorldModel.resolve_fast_attention(None) is True


def test_resolve_fast_attention_honours_an_explicit_bool(vit_attention):
    """Explicit values still force their branch, so `compute_ccr`'s call site is unchanged."""
    from models.visual_world_model import VWorldModel

    for ambient in (False, True):
        vit_attention.use_sdpa = ambient
        assert VWorldModel.resolve_fast_attention(True) is True
        assert VWorldModel.resolve_fast_attention(False) is False


class _PredictSpy:
    """Minimal stand-in for ``VWorldModel``: records the switch each ``predict`` observes.

    Only the two members ``_predict_maybe_checkpointed`` touches. Deliberately not a real
    ``VWorldModel``: ``Attention.__init__`` calls ``.to('cuda')``, so constructing the real
    predictor needs a GPU and this check needs none.
    """

    resolve_fast_attention = staticmethod(
        __import__("models.visual_world_model", fromlist=["VWorldModel"])
        .VWorldModel.resolve_fast_attention
    )

    def __init__(self):
        self.observed = []

    def predict(self, z):
        import models.vit as vit

        self.observed.append(vit.Attention.use_sdpa)
        # An *intermediate* whose backward needs it, so a non-reentrant checkpoint actually
        # recomputes this callable. `z * 1` would not: multiplying by a constant saves
        # nothing from inside the segment, nothing is unpacked, and no recomputation runs --
        # which would make the recomputation test vacuous rather than failing.
        h = z.exp()
        return h * h


def _predict_once(spy, *, checkpoint, fast_attention, requires_grad=False):
    import torch

    from models.visual_world_model import VWorldModel

    z = torch.zeros(2, 2, requires_grad=requires_grad)
    out = VWorldModel._predict_maybe_checkpointed(
        spy, z, checkpoint, fast_attention
    )
    return out


@pytest.mark.parametrize("ambient", [False, True])
def test_predict_observes_the_ambient_switch_when_not_asked(vit_attention, ambient):
    """The regression test proper: an outer scope now reaches the predictor call."""
    pytest.importorskip("torch")
    spy = _PredictSpy()

    vit_attention.use_sdpa = ambient
    _predict_once(spy, checkpoint=False, fast_attention=None)

    assert spy.observed == [ambient]


def test_explicit_false_still_forces_the_materialised_branch(vit_attention):
    """CCR's `fast_attention=False` configuration keeps forcing off, ambient notwithstanding."""
    pytest.importorskip("torch")
    spy = _PredictSpy()

    vit_attention.use_sdpa = True
    _predict_once(spy, checkpoint=False, fast_attention=False)

    assert spy.observed == [False]


def test_checkpointed_recomputation_is_pinned_to_the_forward_branch(vit_attention):
    """The ambient value is resolved at forward time, not looked up again in backward.

    A checkpointed segment is recomputed during backward, long after the enclosing `with` has
    exited. If the branch were resolved lazily, the recomputation would run on the *other*
    implementation and produce activations that do not match the ones the forward recorded.
    """
    torch = pytest.importorskip("torch")
    spy = _PredictSpy()

    vit_attention.use_sdpa = False
    with plan_agg.attention_scope(LONG):
        out = _predict_once(spy, checkpoint=True, fast_attention=None, requires_grad=True)

    # Scope has exited; the ambient switch is back to the shipped default.
    assert vit_attention.use_sdpa is False
    out.sum().backward()

    # Two observations: the forward and the backward recomputation. Both on the branch the
    # forward took, not the one that happens to be ambient at backward time.
    assert len(spy.observed) == 2, spy.observed
    assert spy.observed == [True, True]
    assert not isinstance(out, tuple)
    assert isinstance(out, torch.Tensor)
