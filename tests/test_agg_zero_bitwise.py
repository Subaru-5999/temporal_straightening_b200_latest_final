"""Task 4.2 - the bitwise-zero guarantee for the aggregated-space planning cost.

This module is a **gate**, deliberately not optional. It is the reason the Baseline_Arm is a
valid control rather than an approximation: every downstream comparison in this feature - the
paired zero-weight end-to-end check, the sweep's same-seed reference point, the
Paired_Comparison in the Negative_Result_Record - rests on ``Agg_Weight = 0`` reproducing the
unmodified planning objective *exactly*, not merely closely.

Three deliberate choices, each of which the property would be weaker without:

1. **Raw bytes, not ``torch.equal``.** ``torch.equal`` reports ``nan != nan``, so a path that
   turned a ``nan`` loss into a *different* ``nan`` would look "unequal" either way and a path
   that quietly replaced one ``nan`` with another would never be caught. The comparison here is
   ``t.detach().cpu().numpy().tobytes()``, which is exact for every float32 bit pattern
   including the non-finite ones.
2. **Non-finite and denormal inputs.** ``L_spatial + 0 * L_agg`` is *not* bitwise safe:
   ``0 * inf`` and ``0 * nan`` are both ``nan``, so the arithmetic form of the zero weight would
   poison a perfectly good loss. ``agg_objectives.create_agg_objective_fn`` therefore resolves
   ``enabled = float(agg_weight) > 0.0`` once at factory time and, when disabled, performs no
   tensor operation on the spatial loss at all. That is a gate, and only non-finite inputs
   distinguish a gate from arithmetic. ``test_why_bytes_and_not_arithmetic`` pins the arithmetic
   hazard itself so the point of the exercise cannot be lost.
3. **Identity as well as bytes.** Because the disabled path returns the frozen delegate's *own*
   tensor object, equality is by identity. Bytes alone would pass for a recomputation; identity
   pins the mechanism the design argues from.

Requirement 5.5 is checked alongside: at ``Agg_Weight = 0`` a raw L_agg magnitude must still
reach the Instrumentation_Record at the recorded steps, while L_plan stays equal to L_spatial.
Task 5.1 owns ``AggInstrumentation``; until it lands, :class:`StubInstrumentation` below stands in
with the same three-method contract the objective calls (``should_record`` / ``log`` / ``advance``).

Validates: Requirements 1.2, 3.1, 3.2, 5.5
"""

from __future__ import annotations

import contextlib
import functools
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import agg_objectives  # noqa: E402
from agg_objectives import AGG_CONTEXT, create_agg_objective_fn  # noqa: E402
from planning.objectives import create_objective_fn  # noqa: E402
from tests.conftest import (  # noqa: E402
    AGG_TARGET_CELL_CHANNELS,
    AGG_TARGET_CELL_HIDDEN_DIM,
    AGG_TARGET_CELL_IN_DIM,
    AGG_TARGET_CELL_OUT_DIM,
    AGG_TARGET_CELL_PATCHES,
    AGG_SPECIAL_VALUES,
    agg_alpha_strategy,
    agg_base_strategy,
    agg_hidden_dim_strategy,
    agg_latent_dicts,
    agg_mode_strategy,
    agg_out_dim_strategy,
    agg_step_strategy,
    make_agg_head,
)

# Minimum 100 examples per the feature's testing convention, without capping the
# ``ccr-thorough`` profile back down to 100.
MIN_EXAMPLES = max(100, settings.default.max_examples or 100)
agg_settings = settings(max_examples=MIN_EXAMPLES)

#: Every spelling of "zero" a config or a CLI override can produce. ``-0.0`` is included on
#: purpose: it is the one float for which ``x + 0.0`` is *not* bit-exact, so a gate that compared
#: ``weight == 0.0`` loosely and then fell through to arithmetic would show up here.
ZERO_WEIGHTS = (0, 0.0, -0.0)
zero_weight_strategy = st.sampled_from(ZERO_WEIGHTS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def raw_bytes(tensor: torch.Tensor) -> bytes:
    """The tensor's exact float32 bit pattern, ``nan`` payloads included."""
    return tensor.detach().cpu().numpy().tobytes()


def clone_latents(latents: dict) -> dict:
    """An independent copy, so an in-place write on one path cannot feed the other."""
    return {key: value.clone() for key, value in latents.items()}


@dataclass
class DelegateCall:
    """One callable built from the frozen factory, plus the tensors it actually returned."""

    alpha: Any
    base: Any
    mode: str
    outputs: List[torch.Tensor] = field(default_factory=list)


@contextlib.contextmanager
def recorded_frozen_delegates():
    """Capture the tensor objects the frozen delegates return, for the identity assertion.

    Patches ``agg_objectives._create_frozen_objective_fn`` - a private name in the **new**
    module - and restores it afterwards. Nothing inside ``planning.objectives`` is rebound, so
    Property 8's frozen-module immutability is untouched and the wrapped callables are the frozen
    ones: the wrapper only appends each returned tensor to a list before handing it back.
    """
    original = agg_objectives._create_frozen_objective_fn
    created: List[DelegateCall] = []

    def factory(alpha, base, mode="last"):
        delegate = original(alpha=alpha, base=base, mode=mode)
        record = DelegateCall(alpha=alpha, base=base, mode=mode)
        created.append(record)

        @functools.wraps(delegate)
        def wrapped(z_obs_pred, z_obs_tgt, step=None):
            out = delegate(z_obs_pred, z_obs_tgt, step=step)
            record.outputs.append(out)
            return out

        return wrapped

    agg_objectives._create_frozen_objective_fn = factory
    try:
        yield created
    finally:
        agg_objectives._create_frozen_objective_fn = original


class StubInstrumentation:
    """Stand-in for task 5.1's ``AggInstrumentation``, with the same three-method contract.

    ``planning/gd.py`` calls the objective exactly once per inner iteration, so the call index is
    the optimizer step index; the real recorder fires at ``step_index == 0`` and
    ``step_index == opt_steps - 1``, and this double mirrors that rather than recording every call.
    """

    def __init__(self, opt_steps: int):
        self.opt_steps = int(opt_steps)
        self.step_index = 0
        self.advances = 0
        self.records: List[dict] = []

    def should_record(self) -> bool:
        return self.step_index in (0, self.opt_steps - 1)

    def log(self, step, l_spatial, l_agg) -> None:
        self.records.append(
            {
                "step": step,
                "step_index": self.step_index,
                "l_spatial": l_spatial,
                "l_agg": l_agg,
                "grad_enabled": torch.is_grad_enabled(),
            }
        )

    def advance(self) -> None:
        self.advances += 1
        self.step_index += 1
        if self.step_index >= self.opt_steps:
            self.step_index = 0


@contextlib.contextmanager
def published_instrumentation(recorder: Optional[StubInstrumentation]):
    """Attach a recorder to :data:`AGG_CONTEXT` for the duration, then restore the previous one."""
    previous = AGG_CONTEXT.instrumentation
    AGG_CONTEXT.instrumentation = recorder
    try:
        yield recorder
    finally:
        AGG_CONTEXT.instrumentation = previous


@dataclass
class ZeroWeightCase:
    """One generated configuration of the objective at a zero Agg_Weight."""

    latents: Any
    mode: str
    alpha: float
    base: Any
    step: Optional[int]
    agg_weight: Any
    hidden_dim: int
    out_dim: int
    head_seed: int
    nonfinite: bool

    def make_head(self):
        return make_agg_head(
            in_dim=self.latents.in_dim,
            hidden_dim=self.hidden_dim,
            out_dim=self.out_dim,
            seed=self.head_seed,
        )


@st.composite
def zero_weight_cases(draw, *, nonfinite: Optional[bool] = None) -> ZeroWeightCase:
    """Latents, head widths and every frozen-factory argument, drawn jointly.

    ``nonfinite=None`` draws the flag, so ordinary and special-valued latents are both exercised;
    the head's ``in_dim`` is pinned to ``patches * channels`` by construction, since a shape
    mismatch is Property 6's subject and would only mask this one.
    """
    special = draw(st.booleans()) if nonfinite is None else bool(nonfinite)
    latents = draw(agg_latent_dicts(nonfinite=special))
    return ZeroWeightCase(
        latents=latents,
        mode=draw(agg_mode_strategy),
        alpha=draw(agg_alpha_strategy),
        base=draw(agg_base_strategy),
        step=draw(agg_step_strategy(latents.num_frames)),
        agg_weight=draw(zero_weight_strategy),
        hidden_dim=draw(agg_hidden_dim_strategy),
        out_dim=draw(agg_out_dim_strategy),
        head_seed=draw(st.integers(min_value=0, max_value=2**31 - 1)),
        nonfinite=special,
    )


# ---------------------------------------------------------------------------
# Property 1
# ---------------------------------------------------------------------------


@given(case=zero_weight_cases())
@agg_settings
def test_property_1_zero_weight_is_bitwise_identity(case):
    """Feature: aggregated-space-planning-cost, Property 1: Zero weight is bitwise identity.

    For any pair of predicted and goal latent dictionaries (including entries containing ``inf``,
    ``-inf``, ``nan`` and denormals), any ``alpha``, any ``base``, any mode in {``last``, ``all``,
    ``staged``} and any ``step``, the tensor returned by the Agg_Objective_Module at Agg_Weight
    ``0`` has the same raw byte representation as the tensor returned by the unmodified
    ``planning.objectives.create_objective_fn`` callable for the same inputs - and is in fact the
    delegate's own tensor object, since the disabled path performs no tensor operation on it.

    **Validates: Requirements 1.2, 3.1, 3.2**
    """
    head = case.make_head()

    # Independent input copies per path: an accidental in-place write on one must not be able to
    # feed the other the mutated tensor and hide itself.
    agg_pred = clone_latents(case.latents.z_pred)
    agg_tgt = clone_latents(case.latents.z_tgt)
    frozen_pred = clone_latents(case.latents.z_pred)
    frozen_tgt = clone_latents(case.latents.z_tgt)

    with published_instrumentation(None), recorded_frozen_delegates() as delegates:
        objective = create_agg_objective_fn(
            alpha=case.alpha,
            base=case.base,
            mode=case.mode,
            agg_weight=case.agg_weight,
            agg_head=head,
        )
        result = objective(agg_pred, agg_tgt, step=case.step)

    expected = create_objective_fn(alpha=case.alpha, base=case.base, mode=case.mode)(
        frozen_pred, frozen_tgt, step=case.step
    )

    assert objective.enabled is False, (
        f"agg_weight={case.agg_weight!r} must resolve to the disabled path; "
        f"objective.enabled={objective.enabled!r}"
    )

    # Bytes, not torch.equal: nan != nan under torch.equal, which is exactly the failure this
    # property exists to catch.
    assert raw_bytes(result) == raw_bytes(expected), (
        f"L_plan at agg_weight={case.agg_weight!r} is not BITWISE equal to the unmodified "
        f"objective: mode={case.mode}, alpha={case.alpha!r}, base={case.base!r}, "
        f"step={case.step!r}, nonfinite={case.nonfinite}, "
        f"shape={tuple(result.shape)}; got {result.tolist()!r} vs {expected.tolist()!r}"
    )
    assert result.shape == expected.shape == (case.latents.batch_size,)
    assert result.dtype == expected.dtype == case.latents.z_pred["visual"].dtype
    assert result.device == case.latents.z_pred["visual"].device

    # Two delegates, built from the one frozen factory: L_spatial with the configured alpha,
    # L_agg with alpha pinned to 0.
    assert len(delegates) == 2, f"expected two frozen delegates, saw {len(delegates)}"
    spatial_delegate, agg_delegate = delegates
    assert spatial_delegate.alpha == case.alpha
    assert agg_delegate.alpha == 0
    assert spatial_delegate.base == agg_delegate.base == case.base
    assert spatial_delegate.mode == agg_delegate.mode == case.mode

    # Identity, not just bytes: the disabled path returns the object the frozen callable returned.
    assert len(spatial_delegate.outputs) == 1
    assert result is spatial_delegate.outputs[0], (
        "at agg_weight 0 the objective must return the frozen delegate's own tensor object; "
        "a bitwise-equal recomputation is not the guarantee the design argues from"
    )

    # With no recorder attached, the aggregated-space term is never even evaluated, so a
    # non-finite L_agg cannot reach the result by any route.
    assert agg_delegate.outputs == []


@given(case=zero_weight_cases(), opt_steps=st.integers(min_value=2, max_value=4))
@agg_settings
def test_property_1_zero_weight_still_records_a_raw_l_agg(case, opt_steps):
    """Feature: aggregated-space-planning-cost, Property 1: Zero weight is bitwise identity.

    The other half of the property: while L_plan stays bitwise equal to L_spatial at Agg_Weight
    ``0``, the Instrumentation_Record for the recorded steps still carries a raw L_agg magnitude,
    computed off the autograd graph.

    **Validates: Requirements 1.2, 3.2, 5.5**
    """
    head = case.make_head()
    recorder = StubInstrumentation(opt_steps=opt_steps)

    agg_pred = clone_latents(case.latents.z_pred)
    agg_tgt = clone_latents(case.latents.z_tgt)
    frozen_pred = clone_latents(case.latents.z_pred)
    frozen_tgt = clone_latents(case.latents.z_tgt)

    with published_instrumentation(recorder), recorded_frozen_delegates() as delegates:
        objective = create_agg_objective_fn(
            alpha=case.alpha,
            base=case.base,
            mode=case.mode,
            agg_weight=case.agg_weight,
            agg_head=head,
        )
        results = [objective(agg_pred, agg_tgt, step=case.step) for _ in range(opt_steps)]

    frozen_fn = create_objective_fn(alpha=case.alpha, base=case.base, mode=case.mode)
    expected = frozen_fn(frozen_pred, frozen_tgt, step=case.step)
    expected_bytes = raw_bytes(expected)

    # Recording must not perturb the returned loss on any call.
    for index, result in enumerate(results):
        assert raw_bytes(result) == expected_bytes, (
            f"call {index} of {opt_steps}: recording changed L_plan at "
            f"agg_weight={case.agg_weight!r}, mode={case.mode}, step={case.step!r}"
        )

    spatial_delegate, _agg_delegate = delegates
    assert len(spatial_delegate.outputs) == opt_steps
    assert all(
        result is delegate_output
        for result, delegate_output in zip(results, spatial_delegate.outputs)
    ), "every returned tensor must be the frozen delegate's own object, recorder attached or not"

    # The recorder fires at step_index 0 and opt_steps - 1, and advance() runs once per call.
    assert recorder.advances == opt_steps
    assert [record["step_index"] for record in recorder.records] == [0, opt_steps - 1]

    with torch.no_grad():
        reference_agg = objective.agg_loss_fn(
            clone_latents(case.latents.z_pred), clone_latents(case.latents.z_tgt), step=case.step
        )

    for record in recorder.records:
        assert record["step"] == case.step
        # Requirement 5.5: the raw L_agg magnitude is recorded even though L_plan == L_spatial.
        l_agg = record["l_agg"]
        assert isinstance(l_agg, torch.Tensor)
        assert l_agg.shape == (case.latents.batch_size,)
        assert l_agg.dtype == case.latents.z_pred["visual"].dtype
        assert raw_bytes(l_agg) == raw_bytes(reference_agg), (
            "the recorded L_agg must be the raw mean-squared aggregated-space distance"
        )
        # Off the autograd graph, so 2 records per 100 optimizer steps cost nothing and cannot
        # leak into the planner's gradient.
        assert record["grad_enabled"] is False
        assert l_agg.grad_fn is None and l_agg.requires_grad is False
        # The recorded L_spatial is the tensor that was returned, not a copy of it.
        assert record["l_spatial"] is results[record["step_index"]]

    if not case.nonfinite:
        magnitude = float(reference_agg.mean())
        assert magnitude == magnitude and magnitude >= 0.0, (
            f"finite latents must give a finite non-negative L_agg magnitude, got {magnitude!r}"
        )


# ---------------------------------------------------------------------------
# Worked examples alongside the property
# ---------------------------------------------------------------------------


def test_why_bytes_and_not_arithmetic():
    """Pin the two facts the property's method rests on, so its point cannot be lost.

    ``L_spatial + 0 * L_agg`` is not bitwise safe, and ``torch.equal`` cannot see the difference.
    """
    spatial = torch.tensor([1.0, 2.0, 3.0, 0.0], dtype=torch.float32)
    agg = torch.tensor(
        [float("inf"), float("nan"), float("-inf"), 1.0], dtype=torch.float32
    )

    naive = spatial + 0.0 * agg
    assert raw_bytes(naive) != raw_bytes(spatial), (
        "0 * inf and 0 * nan are nan, so the arithmetic form of a zero weight poisons the loss; "
        "if this ever stops holding, the gate in create_agg_objective_fn is no longer load-bearing"
    )

    # And this is why the comparison is bytes rather than torch.equal.
    nan = torch.tensor([float("nan")], dtype=torch.float32)
    assert not torch.equal(nan, nan)
    assert raw_bytes(nan) == raw_bytes(nan.clone())


@pytest.mark.parametrize("mode", ("last", "all", "staged"))
def test_zero_weight_is_bitwise_identity_at_target_cell_shapes(mode):
    """The Target_Cell worked example: 196 patches x 8 channels -> 1568 -> 512 -> 512 -> 128.

    Explicit ``inf``, ``-inf``, ``nan`` and denormal values in both the predicted and the goal
    visual features, at the shapes ``wm.rollout`` and ``encode_obs`` actually produce for PushT
    (``T = 6``, ``B = 2``), in the two modes the Evaluation_Protocol uses plus ``all``.
    """
    batch, frames = 2, 6
    generator = torch.Generator(device="cpu").manual_seed(0)

    def visual(t_frames: int) -> torch.Tensor:
        tensor = torch.randn(
            batch,
            t_frames,
            AGG_TARGET_CELL_PATCHES,
            AGG_TARGET_CELL_CHANNELS,
            generator=generator,
            dtype=torch.float32,
        )
        flat = tensor.view(-1)
        for offset, value in enumerate(AGG_SPECIAL_VALUES):
            flat[offset * 7] = value
        return tensor

    z_pred = {
        "visual": visual(frames),
        "proprio": torch.randn(batch, frames, 3, generator=generator, dtype=torch.float32),
    }
    z_tgt = {
        "visual": visual(1),
        "proprio": torch.randn(batch, 1, 3, generator=generator, dtype=torch.float32),
    }

    head = make_agg_head(
        in_dim=AGG_TARGET_CELL_IN_DIM,
        hidden_dim=AGG_TARGET_CELL_HIDDEN_DIM,
        out_dim=AGG_TARGET_CELL_OUT_DIM,
    )
    assert head.in_dim == AGG_TARGET_CELL_PATCHES * AGG_TARGET_CELL_CHANNELS == 1568

    # step 4 takes the terminal-only staged branch (4 < T - 1 == 5); step 5 takes the weighted one.
    for step in (None, 0, 4, 5):
        with published_instrumentation(None):
            objective = create_agg_objective_fn(
                alpha=1, base=2, mode=mode, agg_weight=0, agg_head=head
            )
            result = objective(clone_latents(z_pred), clone_latents(z_tgt), step=step)

        expected = create_objective_fn(alpha=1, base=2, mode=mode)(
            clone_latents(z_pred), clone_latents(z_tgt), step=step
        )

        assert result.shape == (batch,)
        assert raw_bytes(result) == raw_bytes(expected), (
            f"Target_Cell shapes, mode={mode}, step={step!r}: L_plan at agg_weight 0 is not "
            f"bitwise equal to the unmodified objective"
        )
