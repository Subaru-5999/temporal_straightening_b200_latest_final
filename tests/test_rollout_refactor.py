"""Property 7: the ``_rollout_latents`` extraction preserved ``rollout``.

Feature: counterfactual-curvature-regularization, Property 7: The rollout refactor preserves
rollout

**Validates: Requirements 1.2, 1.7, 5.2**

Task 2.1 moved the predictor loop out of ``VWorldModel.rollout`` into
``VWorldModel._rollout_latents`` so that ``compute_ccr`` can roll latents forward without a
second encoder pass (Requirements 1.2, 1.7). That refactor is only safe if it is *nothing but*
a refactor: ``plan.py``, ``planning/gd.py``, ``planning/cem.py``, ``planning/evaluator.py`` and
``Trainer.openloop_rollout`` all call ``rollout`` and must be numerically unaffected
(Requirement 5.2).

The reference is :func:`tests.reference_impl.reference_rollout`, a frozen verbatim copy of the
pre-feature loop. Equality is checked with ``torch.equal`` (bitwise), not
``assert_close``: the refactor performs the identical tensor operations in the identical order
on identically-seeded weights, so anything other than bit-for-bit equality means the op
sequence changed and is a real finding rather than a tolerance question.

This is the gate for section 2 of the plan: no CCR code may be built on top of
``_rollout_latents`` until this passes.
"""

from __future__ import annotations

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from tests.conftest import (
    MAX_NUM_FRAMES,
    agg_type_strategy,
    batch_size_strategy,
    build_stub_world_model,
    concat_dim_strategy,
    make_stub_batch,
    num_hist_strategy,
)
from tests.reference_impl import reference_rollout

# ``rollout`` is called with ``obs_0`` holding n context frames and ``act`` holding t + n
# action frames (see ``Trainer.openloop_rollout``, which calls it with both n = num_hist and
# n = 1, and ``planning/gd.py`` / ``planning/cem.py``, which call it with n = num_hist). n is
# therefore *not* tied to num_hist and is generated independently down to 1.
MIN_CONTEXT_FRAMES = 1
MAX_CONTEXT_FRAMES = MAX_NUM_FRAMES - 2  # leaves room for at least one rollout step
MIN_ROLLOUT_STEPS = 1
MAX_ROLLOUT_STEPS = 4


@st.composite
def rollout_cases(draw):
    """Joint strategy over the shapes and model variants ``rollout`` is called with.

    Draws batch size, context length n, rollout length t, ``num_hist``, ``concat_dim``,
    ``agg_type`` and a weight seed together, so the ``concat_dim`` branches of ``encode`` /
    ``replace_actions_from_z`` / ``separate_emb`` are exercised against every context length.
    """
    return {
        "batch_size": draw(batch_size_strategy),
        "num_context_frames": draw(
            st.integers(min_value=MIN_CONTEXT_FRAMES, max_value=MAX_CONTEXT_FRAMES)
        ),
        "num_rollout_steps": draw(
            st.integers(min_value=MIN_ROLLOUT_STEPS, max_value=MAX_ROLLOUT_STEPS)
        ),
        "num_hist": draw(num_hist_strategy),
        "concat_dim": draw(concat_dim_strategy),
        "agg_type": draw(agg_type_strategy),
        "seed": draw(st.integers(min_value=0, max_value=3)),
    }


def _build_pair(case):
    """Two identically-seeded models: the reference loop and the refactored path share weights."""
    kwargs = dict(
        num_hist=case["num_hist"],
        num_pred=1,
        concat_dim=case["concat_dim"],
        agg_type=case["agg_type"],
        seed=case["seed"],
        image_size=8,
    )
    reference_model = build_stub_world_model(**kwargs)
    refactored_model = build_stub_world_model(**kwargs)
    # Deterministic modules only (Linear / Conv1d / LayerNorm), but eval() removes any doubt
    # that a train-mode-only op could differ between the two calls.
    reference_model.eval()
    refactored_model.eval()

    ref_state = reference_model.state_dict()
    new_state = refactored_model.state_dict()
    assert ref_state.keys() == new_state.keys()
    for key in ref_state:
        assert torch.equal(ref_state[key], new_state[key]), (
            f"the two stub models are not identically seeded: weight '{key}' differs, so a "
            "downstream mismatch would say nothing about the refactor"
        )
    return reference_model, refactored_model


def _make_rollout_inputs(model, case):
    """``obs_0`` with n context frames and ``act`` with t + n frames, as callers pass them."""
    n = case["num_context_frames"]
    obs_0, act = make_stub_batch(
        model,
        batch_size=case["batch_size"],
        num_frames=n,
        num_action_frames=n + case["num_rollout_steps"],
        seed=case["seed"],
    )
    assert obs_0["visual"].shape[1] == n
    assert act.shape[1] == n + case["num_rollout_steps"]
    return obs_0, act


def _assert_rollouts_identical(reference, refactored, case):
    """Element-for-element equality of both returned values."""
    ref_z_obses, ref_z = reference
    new_z_obses, new_z = refactored

    assert new_z.shape == ref_z.shape, (
        f"rollout returned {tuple(new_z.shape)}, reference returned {tuple(ref_z.shape)}"
    )
    # t + n + 1 frames, per the rollout docstring.
    expected_frames = case["num_context_frames"] + case["num_rollout_steps"] + 1
    assert ref_z.shape[1] == expected_frames
    assert torch.equal(new_z, ref_z), (
        "refactored rollout latents differ from the frozen reference loop; max abs diff "
        f"{(new_z - ref_z).abs().max().item()!r}"
    )

    assert new_z_obses.keys() == ref_z_obses.keys()
    for key in ref_z_obses:
        assert new_z_obses[key].shape == ref_z_obses[key].shape, (
            f"z_obses['{key}']: {tuple(new_z_obses[key].shape)} vs "
            f"{tuple(ref_z_obses[key].shape)}"
        )
        assert torch.equal(new_z_obses[key], ref_z_obses[key]), (
            f"refactored rollout z_obses['{key}'] differs from the frozen reference loop; "
            f"max abs diff {(new_z_obses[key] - ref_z_obses[key]).abs().max().item()!r}"
        )


@given(case=rollout_cases())
def test_property_7_rollout_refactor_preserves_rollout(case):
    """Feature: counterfactual-curvature-regularization, Property 7: The rollout refactor
    preserves rollout.

    For any initial observation window and action sequence, ``rollout(obs_0, act)`` returns
    latents equal to those produced by a frozen reference copy of the original rollout loop,
    element for element.

    **Validates: Requirements 1.2, 1.7, 5.2**
    """
    reference_model, refactored_model = _build_pair(case)
    obs_0, act = _make_rollout_inputs(refactored_model, case)

    # Independent copies of the inputs, so an accidental in-place write on one path cannot
    # feed the other path a mutated tensor and hide itself.
    ref_obs_0 = {k: v.clone() for k, v in obs_0.items()}
    new_obs_0 = {k: v.clone() for k, v in obs_0.items()}

    with torch.no_grad():
        reference = reference_rollout(reference_model, ref_obs_0, act.clone())
        refactored = refactored_model.rollout(new_obs_0, act.clone())

    _assert_rollouts_identical(reference, refactored, case)


@pytest.mark.parametrize("num_context_frames", [1, 3])
def test_rollout_refactor_matches_reference_on_target_cell(num_context_frames):
    """PushT target-cell shapes, both context lengths ``Trainer.openloop_rollout`` uses.

    A worked example alongside the property: ``num_hist=3``, ``concat_dim=1``,
    ``agg_type=mlp``, n in {num_hist, 1} and a 5-step rollout (Planner_Horizon).
    """
    case = {
        "batch_size": 2,
        "num_context_frames": num_context_frames,
        "num_rollout_steps": 5,
        "num_hist": 3,
        "concat_dim": 1,
        "agg_type": "mlp",
        "seed": 0,
    }
    reference_model, refactored_model = _build_pair(case)
    obs_0, act = _make_rollout_inputs(refactored_model, case)

    with torch.no_grad():
        reference = reference_rollout(
            reference_model, {k: v.clone() for k, v in obs_0.items()}, act.clone()
        )
        refactored = refactored_model.rollout(
            {k: v.clone() for k, v in obs_0.items()}, act.clone()
        )

    _assert_rollouts_identical(reference, refactored, case)
