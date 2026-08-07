"""Task 4.3 - Property 3: the staged dispatch and the per-frame coefficients are reused exactly.

``agg_objectives.create_agg_objective_fn`` does not reimplement the frozen reduction. It builds a
*second* callable from ``planning.objectives.create_objective_fn`` with ``alpha`` pinned to ``0``
and feeds it aggregated-space feature dicts, so the stage predicate (``step < T - 1``), the
coefficient vector (``[base ** i for i in range(T)]``, normalized) and the reduction order are
literally ``planning/objectives.py``'s rather than a copy of it.

This module is the only automated check that the reuse is **exact** rather than approximate, and it
gates the MPC confirmation run: that run is measured in ``staged`` mode (Requirement 8.3), where a
coefficient or stage discrepancy would be invisible in the success rate.

The head used here is the identity on flattened patch features
(``tests/conftest.py::IdentityAggHead``), i.e. exactly the ``x.contiguous().view(x.shape[0], -1)``
the real ``DinoV2Encoder.agg`` performs with the MLP and the LayerNorm removed. A mean over
``p * d`` flattened features is the same reduction as a mean over the ``(p, d)`` axes, so with this
head L_agg must reproduce the unmodified objective's own value for the same mode, ``base`` and
``step`` - which is what makes the equality checkable at all.

Tolerance: the two reductions are numerically the same sum in the same memory order, but they are
not the *same call* (``mean`` over one flattened axis versus over two axes), so equality is asserted
to ``rtol=1e-6, atol=1e-7`` rather than bitwise. The tolerance is stated in every assertion message.
Bitwise identity is the subject of Property 1, on a different code path (the ``w == 0`` early
return), and is not what this property is about.

Validates: Requirements 1.2, 1.6
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from hypothesis import event, given

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agg_objectives import create_agg_objective_fn  # noqa: E402
from planning.objectives import create_objective_fn  # noqa: E402

from .conftest import (  # noqa: E402
    agg_alpha_strategy,
    agg_base_strategy,
    agg_latent_dicts,
    agg_mode_strategy,
    agg_step_strategy,
    make_identity_agg_head,
    positive_agg_weight_strategy,
)

#: The tightest tolerance the two reduction spellings hold to across the generated shapes.
RTOL = 1e-6
ATOL = 1e-7

#: float32 machine epsilon, used only for the cancellation bound in the decomposition check below.
EPS32 = float(torch.finfo(torch.float32).eps)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _agg_loss(latents, *, alpha, base, mode, agg_weight=0.0):
    """The raw L_agg the module itself computes, read off the factory's own seam.

    ``create_agg_objective_fn`` exposes ``agg_loss_fn``: the aggregated-space term before the
    weight is applied. Reading it is exact, so no ``L_plan - L_spatial`` subtraction (and no float
    ordering argument) is needed here. ``test_l_plan_minus_l_spatial_recovers_the_weighted_agg_term``
    below pins that seam to the value the objective actually adds, so this read is not vacuous.
    """
    head = make_identity_agg_head(in_dim=latents.in_dim)
    objective = create_agg_objective_fn(
        alpha=alpha, base=base, mode=mode, agg_weight=agg_weight, agg_head=head
    )
    return objective, objective.agg_loss_fn


def _frozen(mode, *, base, alpha=0):
    """The unmodified frozen callable, built exactly as ``planning/objectives.py`` builds it."""
    return create_objective_fn(alpha=alpha, base=base, mode=mode)


def _assert_matches(actual, expected, *, what):
    assert actual.shape == expected.shape, (
        f"{what}: L_agg has shape {tuple(actual.shape)} but the frozen objective returned "
        f"{tuple(expected.shape)}"
    )
    if torch.allclose(actual, expected, rtol=RTOL, atol=ATOL):
        return
    diff = (actual - expected).abs()
    raise AssertionError(
        f"{what}: L_agg does not match the frozen planning.objectives value to "
        f"rtol={RTOL:g}, atol={ATOL:g} (max abs diff {float(diff.max()):.3e}).\n"
        f"  L_agg  = {actual.tolist()}\n"
        f"  frozen = {expected.tolist()}\n"
        "The stage selection or the per-frame coefficients have drifted from "
        "planning/objectives.py."
    )


def _expected_staged_stage(step, num_frames):
    """Which frozen mode ``objective_fn_staged`` dispatches to, read off the frozen source.

    ``step is None`` -> the full-horizon weighted objective; ``step < T - 1`` -> terminal only;
    otherwise -> the full-horizon weighted objective.
    """
    if step is None:
        return "all"
    return "last" if step < num_frames - 1 else "all"


# ---------------------------------------------------------------------------
# Property 3
# ---------------------------------------------------------------------------


@given(
    latents=agg_latent_dicts(),
    mode=agg_mode_strategy,
    base=agg_base_strategy,
    step=agg_step_strategy(),
    alpha=agg_alpha_strategy,
)
def test_agg_loss_equals_the_frozen_objective_under_an_identity_head(
    latents, mode, base, step, alpha
):
    """Feature: aggregated-space-planning-cost, Property 3: Stage selection and coefficients are the frozen module's

    For any frame count ``T``, any ``base``, any ``step`` and any latent dictionaries, with the
    identity head on flattened patch features L_agg equals the value the unmodified
    ``planning.objectives.create_objective_fn`` callable returns for the same mode, ``base`` and
    ``step``.

    The reference is built with ``alpha = 0``, which is the alpha the module pins the aggregated
    delegate to. The configured ``alpha`` is drawn freely and must not reach L_agg at all: it
    weights the proprio channel of L_spatial only, and ``_agg_dicts`` passes zero-valued proprio
    tensors through an ``alpha = 0`` delegate.

    Validates: Requirements 1.2, 1.6
    """
    _, agg_loss_fn = _agg_loss(latents, alpha=alpha, base=base, mode=mode)

    actual = agg_loss_fn(latents.z_pred, latents.z_tgt, step=step)
    expected = _frozen(mode, base=base)(latents.z_pred, latents.z_tgt, step=step)

    event(f"mode={mode}, step_is_none={step is None}")
    _assert_matches(actual, expected, what=f"mode={mode}, base={base}, step={step}")
    assert actual.shape == (latents.batch_size,)


@given(
    latents=agg_latent_dicts(),
    base=agg_base_strategy,
    step=agg_step_strategy(),
    alpha=agg_alpha_strategy,
)
def test_staged_agg_loss_takes_the_frozen_stage_for_the_step(latents, base, step, alpha):
    """Feature: aggregated-space-planning-cost, Property 3: Stage selection and coefficients are the frozen module's

    In ``staged`` mode L_agg equals the frozen ``last``-mode value when ``step < T - 1`` and the
    frozen ``all``-mode value otherwise (``step is None`` takes the ``all`` branch, which is what
    ``objective_fn_staged`` does with no stage information).

    The other stage's value is asserted to be *different* whenever the two frozen stages disagree,
    so a mode that silently ignored ``step`` could not pass by accident.

    Validates: Requirements 1.2, 1.6
    """
    _, agg_loss_fn = _agg_loss(latents, alpha=alpha, base=base, mode="staged")

    actual = agg_loss_fn(latents.z_pred, latents.z_tgt, step=step)

    stage = _expected_staged_stage(step, latents.num_frames)
    event(f"staged stage={stage}, T={latents.num_frames}, step={step}")

    expected = _frozen(stage, base=base)(latents.z_pred, latents.z_tgt, step=step)
    _assert_matches(
        actual,
        expected,
        what=f"staged, T={latents.num_frames}, base={base}, step={step} -> frozen {stage!r}",
    )

    other = "all" if stage == "last" else "last"
    other_value = _frozen(other, base=base)(latents.z_pred, latents.z_tgt, step=step)
    if not torch.allclose(expected, other_value, rtol=RTOL, atol=ATOL):
        assert not torch.allclose(actual, other_value, rtol=RTOL, atol=ATOL), (
            f"staged L_agg at T={latents.num_frames}, step={step} matched the {other!r} stage "
            f"rather than the {stage!r} stage the frozen dispatch selects"
        )


@given(
    latents=agg_latent_dicts(),
    mode=agg_mode_strategy,
    base=agg_base_strategy,
    step=agg_step_strategy(),
    alpha=agg_alpha_strategy,
    agg_weight=positive_agg_weight_strategy,
)
def test_l_plan_minus_l_spatial_recovers_the_weighted_agg_term(
    latents, mode, base, step, alpha, agg_weight
):
    """Feature: aggregated-space-planning-cost, Property 3: Stage selection and coefficients are the frozen module's

    The raw ``agg_loss_fn`` read the two tests above rest on is the same value the objective adds:
    at a known ``agg_weight``, ``L_plan - L_spatial`` recovers ``agg_weight * L_agg``, where
    L_spatial is the unmodified frozen callable's value for the configured ``alpha``, ``base`` and
    mode. Without this, reading ``agg_loss_fn`` would prove the coefficients of a term nothing uses.

    Tolerance, and why it is not a flat ``rtol``: the *subtraction* is the test's own, and when
    ``w * L_agg`` is orders of magnitude below L_spatial the difference is formed by cancelling two
    nearly equal float32 numbers, which leaves an absolute error of a few units in the last place of
    L_spatial regardless of how small the true difference is. A flat relative tolerance on a tiny
    expected value is therefore a statement about float32 cancellation, not about the objective.
    So the bound is ``rtol * |w * L_agg| + atol + 16 * eps32 * max(|L_plan|, |L_spatial|)``, which
    is tight (about 2e-6 relative to L_spatial) and is the error the subtraction can actually carry.

    Validates: Requirements 1.2, 1.6
    """
    objective, agg_loss_fn = _agg_loss(
        latents, alpha=alpha, base=base, mode=mode, agg_weight=agg_weight
    )

    l_plan = objective(latents.z_pred, latents.z_tgt, step=step)
    l_spatial = _frozen(mode, base=base, alpha=alpha)(latents.z_pred, latents.z_tgt, step=step)
    l_agg = agg_loss_fn(latents.z_pred, latents.z_tgt, step=step)

    recovered = l_plan - l_spatial
    expected = agg_weight * l_agg

    cancellation = 16.0 * EPS32 * torch.maximum(l_plan.abs(), l_spatial.abs())
    tolerance = RTOL * expected.abs() + ATOL + cancellation
    diff = (recovered - expected).abs()
    assert bool((diff <= tolerance).all()), (
        f"mode={mode}, step={step}, w={agg_weight:g}: L_plan - L_spatial = {recovered.tolist()} "
        f"but w * L_agg = {expected.tolist()}; the difference {diff.tolist()} exceeds the "
        f"cancellation-aware bound {tolerance.tolist()} "
        f"(rtol={RTOL:g}, atol={ATOL:g}, plus 16 * eps32 * max(|L_plan|, |L_spatial|))"
    )


# ---------------------------------------------------------------------------
# Worked examples: the Target_Cell stage boundary and the coefficient vector
# ---------------------------------------------------------------------------


def _fixed_latents(batch=2, num_frames=6, patches=4, channels=3, proprio_dim=2, seed=0):
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=gen, dtype=torch.float32)

    z_pred = {
        "visual": randn(batch, num_frames, patches, channels),
        "proprio": randn(batch, num_frames, proprio_dim),
    }
    z_tgt = {
        "visual": randn(batch, 1, patches, channels),
        "proprio": randn(batch, 1, proprio_dim),
    }
    return z_pred, z_tgt, patches * channels


def test_target_cell_stage_boundary_is_at_step_five():
    """Both staged branches at the Target_Cell shapes: ``T = 6``, ``base = 2``, steps 0 through 6.

    ``planning/objectives.py`` dispatches on ``step < T - 1``, so with the six frames
    ``wm.rollout`` returns for the PushT cell, MPC iterations 0-4 take the terminal-only branch and
    5 onwards take the weighted branch. Both are exercised in a single confirmation run, so both
    are pinned here.
    """
    z_pred, z_tgt, in_dim = _fixed_latents(num_frames=6)
    head = make_identity_agg_head(in_dim=in_dim)
    objective = create_agg_objective_fn(
        alpha=1, base=2, mode="staged", agg_weight=0.1, agg_head=head
    )

    frozen_last = create_objective_fn(alpha=0, base=2, mode="last")
    frozen_all = create_objective_fn(alpha=0, base=2, mode="all")

    for step in range(7):
        actual = objective.agg_loss_fn(z_pred, z_tgt, step=step)
        expected_fn = frozen_last if step < 5 else frozen_all
        expected = expected_fn(z_pred, z_tgt, step=step)
        _assert_matches(actual, expected, what=f"T=6, base=2, step={step}")

    # `step=None` is what the open-loop setting passes; the frozen staged branch treats it as "no
    # stage information" and falls through to the weighted objective.
    _assert_matches(
        objective.agg_loss_fn(z_pred, z_tgt, step=None),
        frozen_all(z_pred, z_tgt, step=None),
        what="T=6, base=2, step=None",
    )


def test_all_mode_uses_the_geometric_coefficient_vector():
    """The coefficients really are ``base ** i`` normalized, and not a uniform average.

    Independent replication rather than a second call into the frozen factory: if the reuse ever
    degraded into a plain mean over frames, the equality tests above would still pass at ``base = 1``
    and this one would not.
    """
    z_pred, z_tgt, in_dim = _fixed_latents(num_frames=6)
    head = make_identity_agg_head(in_dim=in_dim)
    objective = create_agg_objective_fn(
        alpha=1, base=2, mode="all", agg_weight=0.5, agg_head=head
    )
    actual = objective.agg_loss_fn(z_pred, z_tgt, step=None)

    per_frame = (z_pred["visual"] - z_tgt["visual"]).pow(2).mean(dim=(2, 3))  # (B, T)
    geometric = torch.tensor([2.0**i for i in range(6)], dtype=torch.float32)
    geometric = geometric / geometric.sum()
    expected = (per_frame * geometric).mean(dim=1)
    _assert_matches(actual, expected, what="all mode, base=2, geometric coefficients")

    uniform = torch.full((6,), 1.0 / 6.0, dtype=torch.float32)
    uniform_value = (per_frame * uniform).mean(dim=1)
    assert not torch.allclose(actual, uniform_value, rtol=RTOL, atol=ATOL), (
        "all-mode L_agg matched a uniform frame average, so this test cannot tell the geometric "
        "coefficient vector from a plain mean; the fixture has degenerated"
    )
