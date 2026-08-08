"""Task 3.4 - Property 4: the gate carries no gradient.

This module is a **gate**, deliberately not optional. An attached gate re-introduces the
lambda-reduction confound the whole ACS design is built to eliminate - invisibly and adaptively.
The argument, from design section 4.2: if `w` were differentiable, the trained
``action_encoder`` (``action_encoder_lr: 5e-4``) could lower ``L_acs`` by driving `w -> 0` on the
hard triples. That lowers total straightening pressure without improving one bit of geometry, and
it is indistinguishable at the telemetry level from a smaller lambda - which is precisely the
objection section 4.1 removed by construction. So the shipped gate reads the **raw** ``act``
tensor and calls ``.detach()`` anyway, and this module is the executable form of that contract.

What is asserted, and why it is in two halves
---------------------------------------------

1. **The cheap half.** ``w.requires_grad is False`` and ``w.grad_fn is None`` - a direct read of
   Requirement 5.3. It is necessary and it is not sufficient: a flag says nothing about what the
   surrounding expression does with the tensor.
2. **The half that carries the property.** The gradient of the ACS expression is computed twice -
   once with the gate exactly as ``action_gate`` returns it, and once with ``w`` replaced by a
   **freshly constructed leaf tensor** holding the same numeric values - and the two gradients must
   agree bitwise. That is the operational meaning of "the gate is a constant": the descent
   direction is the one the model would see if the weights had been typed in as literals.

   Taken over ``z``, over ``act`` **and** over the encoder / ``action_encoder`` /
   ``proprio_encoder`` parameters, not over ``z`` alone. On ``z`` alone the comparison is nearly
   vacuous - `w` is a function of `act`, so deleting the ``.detach()`` leaves ``d(L_acs)/dz``
   bitwise unchanged and the assertion would pass on precisely the attached gate Requirement 5.3
   forbids. ``d(L_acs)/d(act)`` is where an attached gate is visible, and it is checked here and
   again, from the other side, in the ``act.grad is None`` test below.

Alongside those: no gradient reaches ``act`` even when ``act`` requires one (the ``raw``
reduction makes ``cos_a`` genuinely differentiable in ``act`` before the ``detach``, so this is a
live path, not a vacuous one); ``action_encoder`` is never called by the gate; and the gate-derived
telemetry scalars are detached, which is the part of Requirement 8.18 the shipped code can carry.

How the ACS expression is built here
------------------------------------

``compute_acs`` does not exist yet - it lands in task 6.1 - so the term is assembled in this
module from the shipped pieces, exactly as design section 4.1 specifies it::

    c, mask = _cos_curvature_terms(*_agg_velocities(z))
    w       = action_gate(act, mask=mask, env_action_dim=d)
    num     = (w[mask] * c[mask]).sum()
    den     = w[mask].sum().clamp_min(WEIGHT_SUM_FLOOR)
    loss    = num / den

Nothing is implemented in the model here. When task 6.1 lands, this module keeps testing the same
term; if ``compute_acs`` ever disagrees with this expression, that is a defect in ``compute_acs``.

Two shipped-code notes this module respects (``PROGRESS_ACS.md`` section 13):

- ``env_action_dim`` is **threaded** into ``reduce_action`` / ``action_gate``. The design's
  one-argument signature is not implementable: ``act.shape[-1] = 10`` at the target cell cannot
  distinguish 5 substeps x 2 dims from 2 x 5 (section 13.1).
- ``action_gate`` takes the static ``mask``, because the ``permuted`` null control shuffles across
  exactly the unmasked triples (section 13.2). ``permuted`` also calls ``torch.randperm``, so every
  ``action_gate`` call in this module is preceded by a seed - the two runs of the substitution test
  must compare the *same* permutation, or they would be comparing two different gates.

**Validates: Requirements 5.2, 5.3, 8.18, 13.4**
"""

from __future__ import annotations

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.conftest import (
    ACS_GATES,
    ACS_STEP_THRESH,
    ACS_TARGET_CELL_BATCH,
    ACS_TARGET_CELL_CHANNELS,
    ACS_TARGET_CELL_ENV_ACTION_DIM,
    ACS_TARGET_CELL_FRAMES,
    ACS_TARGET_CELL_PATCHES,
    ACS_TARGET_CELL_SUBSTEPS,
    ACS_WEIGHT_SUM_FLOOR,
    acs_action_reduce_strategy,
    acs_cases,
    acs_gate_strategy,
    agg_type_strategy,
    build_stub_world_model,
    make_acs_case,
)

# Minimum 100 examples per the feature's testing convention, without capping a thorough profile
# back down to 100. ``deadline=None``: the expression is small but a CPU-only box under load is
# not a wall clock.
MIN_EXAMPLES = max(100, settings.default.max_examples or 100)
acs_settings = settings(max_examples=MIN_EXAMPLES, deadline=None)

# The stub encoder's aggregated-space grid. ``agg_type="mlp"`` builds its first Linear at
# ``num_patches * emb_dim``, so the feature tensor's patch and channel axes are pinned to the
# stub's defaults (4 x 4) and the whole ``AGG_TYPES`` enum stays reachable.
STUB_PATCHES = 4
STUB_CHANNELS = 4

# Both mode strings the parser accepts for the aggregated space. ``acsaggcos1e-1`` is the arm's
# own setting; ``aggcos1e-1`` is the baseline's. The gate is a method, reachable under either, and
# must not acquire a gradient because of how the model was configured.
STRAIGHTEN_VALUES = ("aggcos1e-1", "acsaggcos1e-1")


# ---------------------------------------------------------------------------
# The ACS expression, assembled from the shipped pieces
# ---------------------------------------------------------------------------


def raw_bytes(tensor: torch.Tensor) -> bytes:
    """The tensor's exact bit pattern, ``nan`` payloads included."""
    return tensor.detach().cpu().contiguous().numpy().tobytes()


def assert_bitwise(actual: torch.Tensor, expected: torch.Tensor, what: str) -> None:
    """Assert two tensors agree in dtype, shape and every bit.

    ``torch.equal`` first; where it says no, the raw bytes decide, because ``nan != nan`` and a
    degenerate batch can legitimately produce one.
    """
    assert actual.dtype == expected.dtype, (
        f"{what}: dtype moved, {actual.dtype} vs {expected.dtype}"
    )
    assert actual.shape == expected.shape, (
        f"{what}: shape moved, {tuple(actual.shape)} vs {tuple(expected.shape)}"
    )
    if torch.equal(actual, expected):
        return
    assert raw_bytes(actual) == raw_bytes(expected), (
        f"{what}: not bitwise equal.\n"
        f"  got      {actual.detach().cpu().reshape(-1)[:8].tolist()!r} ...\n"
        f"  expected {expected.detach().cpu().reshape(-1)[:8].tolist()!r} ..."
    )


def gate_for(model, act, mask, env_action_dim, *, seed: int) -> torch.Tensor:
    """``action_gate`` under a fixed RNG state.

    Only ``permuted`` reads the RNG (``_permute_gate`` calls ``torch.randperm``), but seeding
    unconditionally keeps every gate mode on one code path here and makes the substitution test's
    two runs comparable for all four.
    """
    torch.manual_seed(int(seed))
    return model.action_gate(act, mask=mask, env_action_dim=env_action_dim)


def acs_expression(model, z, act, *, env_action_dim, gate_seed=0, w_override=None):
    """The ACS term as design section 4.1 specifies it, built from the shipped pieces.

    Returns ``(loss, w, c, mask)``. ``w_override`` substitutes a caller-supplied weight tensor for
    the gate's own output, which is how the substitution half of Property 4 is expressed: the two
    calls differ in nothing but the provenance of ``w``.
    """
    v1, v2 = model._agg_velocities(z)
    c, mask = model._cos_curvature_terms(v1, v2)
    if w_override is None:
        w = gate_for(model, act, mask, env_action_dim, seed=gate_seed)
    else:
        w = w_override
    w_masked = w[mask]
    num = (w_masked * c[mask]).sum()
    den = w_masked.sum().clamp_min(ACS_WEIGHT_SUM_FLOOR)
    return num / den, w, c, mask


def fresh_leaf_like(w: torch.Tensor) -> torch.Tensor:
    """A brand-new leaf tensor holding ``w``'s numeric values and sharing no storage with it.

    Built through ``tolist``, so it cannot inherit a view, a graph or a storage from ``w`` by
    accident - it is the tensor a reader would get by typing the weights in as literals, which is
    exactly what "the gate is a constant" means.
    """
    values = torch.tensor(w.detach().cpu().tolist(), dtype=w.dtype)
    assert values.shape == w.shape, "the reconstructed leaf changed shape"
    assert values.requires_grad is False and values.grad_fn is None
    assert values.is_leaf
    assert values.data_ptr() != w.data_ptr(), "the substituted gate shares storage with w"
    assert torch.equal(values, w.detach()), "the substituted gate is not numerically identical"
    return values


def grads_wrt(loss, inputs):
    """``d(loss)/d(input)`` for every input, with an unused input reported as an explicit zero.

    ``allow_unused=True`` plus materialization by hand rather than
    ``materialize_grads=True``: a ``None`` and a zero tensor are the *same* statement about the
    descent direction, and collapsing them here is what lets the substitution test compare the
    two expressions elementwise over inputs the gate expression may or may not reach.
    """
    grads = torch.autograd.grad(loss, list(inputs), allow_unused=True, retain_graph=True)
    return [
        torch.zeros_like(inp) if grad is None else grad
        for inp, grad in zip(inputs, grads)
    ]


def build_model(case):
    """A CPU float32 stub model configured with this case's ACS knobs and aggregation head."""
    return build_stub_world_model(
        agg_type=case["agg_type"],
        straighten=case["straighten"],
        acs_action_reduce=case["action_reduce"],
        acs_gate=case["gate"],
        image_size=8,
        seed=case["model_seed"],
    )


@st.composite
def gate_cases(draw):
    """A jointly generated ``(z, act)`` pair plus the model variant and both ACS knobs.

    ``kind`` is left free, so the draws cover the ordinary case and every degenerate one:
    all-static (every triple masked), one-moving (partially masked), all-parallel (``w = 1``),
    all-antiparallel (``w = 0`` under ``relu_cos`` / ``hard``, so ``WEIGHT_SUM_FLOOR`` binds) and
    zero-norm actions. Detachment is a structural property and must hold on all of them.
    """
    case = draw(acs_cases(patches=STUB_PATCHES, channels=STUB_CHANNELS))
    return {
        "acs": case,
        "gate": draw(acs_gate_strategy),
        "action_reduce": draw(acs_action_reduce_strategy),
        "agg_type": draw(agg_type_strategy),
        "straighten": draw(st.sampled_from(STRAIGHTEN_VALUES)),
        "model_seed": draw(st.integers(min_value=0, max_value=3)),
        "gate_seed": draw(st.integers(min_value=0, max_value=2**31 - 1)),
    }


# ---------------------------------------------------------------------------
# Property 4
# ---------------------------------------------------------------------------


@given(case=gate_cases())
@acs_settings
def test_property_4_gate_tensor_carries_no_grad_fn(case):
    """Feature: action-conditioned-straightening, Property 4: The gate carries no gradient.

    The flag half: for any ``z, act``, ``w.requires_grad is False`` and ``w.grad_fn is None`` -
    holding for all four ``acs_gate`` members including ``permuted``, all three reductions, and
    even when both ``z`` and ``act`` are themselves differentiable leaves. ``w`` is additionally a
    leaf of shape ``(b, t-2)``, matching ``c`` and the mask elementwise.

    **Validates: Requirements 5.3, 13.4**
    """
    acs = case["acs"]
    model = build_model(case)
    z = acs.z.clone().requires_grad_(True)
    act = acs.act.clone().requires_grad_(True)

    _, w, c, mask = acs_expression(
        model, z, act, env_action_dim=acs.env_action_dim, gate_seed=case["gate_seed"]
    )

    assert w.requires_grad is False, (
        f"gate={case['gate']!r} returned a tensor with requires_grad=True; the encoder and the "
        "trained action_encoder could then lower the loss by driving w -> 0"
    )
    assert w.grad_fn is None, (
        f"gate={case['gate']!r} returned a tensor with grad_fn={w.grad_fn!r}, so it is still "
        "attached to the graph that produced it"
    )
    assert w.is_leaf, f"gate={case['gate']!r} returned a non-leaf tensor"
    assert w.shape == c.shape == mask.shape, (
        f"gate shape {tuple(w.shape)} must match c {tuple(c.shape)} and mask "
        f"{tuple(mask.shape)} elementwise"
    )
    assert tuple(w.shape) == (acs.batch_size, acs.num_frames - 2)


@given(case=gate_cases())
@acs_settings
def test_property_4_gradient_equals_the_substituted_constant_gate(case):
    """Feature: action-conditioned-straightening, Property 4: The gate carries no gradient.

    The half that carries the property: the gradient of the ACS expression is **bitwise** equal to
    the gradient of the same expression with ``w`` replaced by its numeric values in a freshly
    constructed leaf tensor. So the descent direction is the one the model would see if the gate
    had been typed in as literals - there is no path through the gate for either the encoder or
    the trained ``action_encoder`` to lower the loss by moving ``w``.

    **Taken over every differentiable input, not just ``z``.** The design states the property for
    ``z``, and on ``z`` alone it is nearly vacuous: `w` is a function of `act`, so removing the
    ``.detach()`` leaves ``d(L_acs)/dz`` bitwise unchanged and the assertion would pass on exactly
    the implementation Requirement 5.3 forbids. The gradients w.r.t. ``act`` and w.r.t. the
    encoder / ``action_encoder`` / ``proprio_encoder`` parameters are therefore compared too --
    that is where an attached gate shows up, as a nonzero ``d(L_acs)/d(act)`` against the
    substituted expression's zero.

    Two independent leaf sets with equal values, so a mismatch cannot be an in-place write on a
    shared tensor. Bitwise rather than ``allclose``: the two expressions perform the identical ops
    on identical values, so anything short of bit equality means the gate changed the graph. An
    input the expression never reaches is compared as an explicit zero (:func:`grads_wrt`), since
    "no gradient" and "zero gradient" are the same statement about the descent direction.

    **Validates: Requirements 5.2, 5.3, 13.4**
    """
    acs = case["acs"]
    model = build_model(case)
    env_action_dim = acs.env_action_dim
    params = [
        p
        for module in (model.encoder, model.action_encoder, model.proprio_encoder)
        for p in module.parameters()
    ]

    z_gate = acs.z.clone().requires_grad_(True)
    act_gate = acs.act.clone().requires_grad_(True)
    loss_gate, w, _, _ = acs_expression(
        model, z_gate, act_gate, env_action_dim=env_action_dim, gate_seed=case["gate_seed"],
    )
    grads_gate = grads_wrt(loss_gate, [z_gate, act_gate, *params])

    w_const = fresh_leaf_like(w)
    z_const = acs.z.clone().requires_grad_(True)
    act_const = acs.act.clone().requires_grad_(True)
    loss_const, w_used, _, _ = acs_expression(
        model, z_const, act_const, env_action_dim=env_action_dim, w_override=w_const,
    )
    assert w_used is w_const, "the override did not reach the expression"
    grads_const = grads_wrt(loss_const, [z_const, act_const, *params])

    where = (
        f"gate={case['gate']!r}, reduce={case['action_reduce']!r}, kind={acs.kind!r}"
    )
    assert_bitwise(loss_const, loss_gate, f"loss ({where})")
    names = ["d(L_acs)/dz", "d(L_acs)/d(act)"] + [
        f"d(L_acs)/d(param {i})" for i in range(len(params))
    ]
    for name, got, expected in zip(names, grads_gate, grads_const):
        assert_bitwise(got, expected, f"{name} ({where})")


@given(case=gate_cases())
@acs_settings
def test_property_4_no_gradient_reaches_the_action_tensor(case):
    """Feature: action-conditioned-straightening, Property 4: The gate carries no gradient.

    Backward through the ACS expression with ``act`` a differentiable leaf leaves ``act.grad``
    ``None``, and ``autograd.grad(loss, act, allow_unused=True)`` returns ``None``. This is a live
    path rather than a vacuous one: under ``acs_action_reduce="raw"`` the reduction is the identity
    on ``act``, so ``cos_a`` really is a differentiable function of ``act`` right up to the
    ``.detach()`` - the flag test above would pass even if that detach were the only thing standing
    between the gate and the graph, and this test is what says the graph is actually cut.

    ``z`` still receives a gradient (except where the batch is fully masked and the geometry term
    is a constant), so the assertion is "no gradient *through the gate*", not "no gradient at all".

    **Validates: Requirements 5.2, 5.3**
    """
    acs = case["acs"]
    model = build_model(case)
    z = acs.z.clone().requires_grad_(True)
    act = acs.act.clone().requires_grad_(True)

    loss, w, _, mask = acs_expression(
        model, z, act, env_action_dim=acs.env_action_dim, gate_seed=case["gate_seed"]
    )
    assert loss.requires_grad, "the ACS expression lost its grad path to z entirely"

    (act_grad,) = torch.autograd.grad(loss, act, allow_unused=True, retain_graph=True)
    assert act_grad is None, (
        f"gate={case['gate']!r}, reduce={case['action_reduce']!r}: the ACS expression has a "
        f"gradient path into act (grad norm {None if act_grad is None else act_grad.norm()}), so "
        "the trained action_encoder's upstream data could move the gate"
    )

    loss.backward()
    assert act.grad is None, (
        f"gate={case['gate']!r}: act.grad is populated after backward through the ACS expression"
    )
    assert w.grad is None, "the gate accumulated a gradient of its own"
    if bool(mask.any()):
        assert z.grad is not None, "z received no gradient at all, so the comparison is vacuous"

    # The detach is load-bearing, not decorative: the quantity the gate is a function of - the
    # cosine of the *shipped* reduction's consecutive vectors - is itself differentiable in `act`
    # and does hand back a gradient. So `act.grad is None` above is the detach's doing, and
    # deleting that one line would open the very path this property forbids. Built from
    # `reduce_action` and `cosine_similarity` only; the gate itself is never re-implemented here.
    a = model.reduce_action(act, env_action_dim=acs.env_action_dim)
    cos_a = torch.nn.functional.cosine_similarity(a[:, :-2], a[:, 1:-1], dim=-1)
    assert cos_a.requires_grad and cos_a.grad_fn is not None, (
        f"reduce={case['action_reduce']!r}: the pre-gate cosine is not differentiable in act, so "
        "this test cannot distinguish a detached gate from an unreachable one"
    )
    (pre_gate_grad,) = torch.autograd.grad(cos_a.sum(), act, allow_unused=True)
    assert pre_gate_grad is not None, (
        "the pre-gate cosine has no gradient path into act at all"
    )


@given(case=gate_cases())
@acs_settings
def test_property_4_gate_ignores_the_encoder_and_the_action_encoder(case):
    """Feature: action-conditioned-straightening, Property 4: The gate carries no gradient.

    The mechanism behind the detachment, asserted rather than assumed: ``w`` is computed from the
    raw ``act`` tensor of the batch, so perturbing every parameter of the encoder **and** of the
    trained ``action_encoder`` leaves ``w`` bitwise unchanged, and calling the gate never touches
    ``action_encoder`` at all (it is replaced here by a callable that raises). An encoded-action
    gate would let ``action_encoder``, which trains at ``5e-4``, drive ``w -> 0`` on the hard
    triples and lower total straightening pressure without improving any geometry.

    **Validates: Requirements 5.2**
    """
    acs = case["acs"]
    model = build_model(case)
    env_action_dim = acs.env_action_dim

    _, _, _, mask = acs_expression(
        model, acs.z.clone(), acs.act.clone(), env_action_dim=env_action_dim,
        gate_seed=case["gate_seed"],
    )

    def _boom(*_args, **_kwargs):  # pragma: no cover - reaching it is the failure
        raise AssertionError(
            "action_gate called action_encoder; the gate must read the raw act tensor "
            "(Requirement 5.2)"
        )

    model.action_encoder.forward = _boom
    before = gate_for(model, acs.act.clone(), mask, env_action_dim, seed=case["gate_seed"])

    with torch.no_grad():
        for module in (model.encoder, model.action_encoder, model.proprio_encoder):
            for param in module.parameters():
                param.add_(torch.full_like(param, 0.25))

    after = gate_for(model, acs.act.clone(), mask, env_action_dim, seed=case["gate_seed"])
    assert_bitwise(after, before, f"w after perturbing the encoders (gate={case['gate']!r})")


@given(case=gate_cases())
@acs_settings
def test_property_4_gate_telemetry_scalars_are_detached(case):
    """Feature: action-conditioned-straightening, Property 4: The gate carries no gradient.

    The gate-derived telemetry - ``acs_gate_mean``, ``acs_gate_tv``, ``acs_gate_zero_frac`` and the
    quantiles - is computed from ``w``, so a detached gate makes every one of them a detached
    scalar for free. That is the part of Requirement 8.18 the shipped code can carry; the full
    check over ``compute_acs``'s own return values belongs to task 6.1.

    **Validates: Requirements 8.18**
    """
    acs = case["acs"]
    model = build_model(case)
    _, w, _, mask = acs_expression(
        model, acs.z.clone().requires_grad_(True), acs.act.clone(),
        env_action_dim=acs.env_action_dim, gate_seed=case["gate_seed"],
    )

    w_masked = w[mask]
    if w_masked.numel() == 0:
        # A fully static batch masks out every triple; there is no gate population to summarise,
        # which is the E9 case and not this property's business.
        return

    total = w_masked.sum()
    n = w_masked.numel()
    scalars = {
        "acs_gate_mean": w_masked.mean(),
        "acs_gate_tv": 0.5 * (w_masked / total.clamp_min(ACS_WEIGHT_SUM_FLOOR) - 1.0 / n)
        .abs()
        .sum(),
        "acs_gate_zero_frac": (w_masked == 0).to(w.dtype).mean(),
        "acs_gate_p50": w_masked.median(),
    }
    for name, value in scalars.items():
        assert value.requires_grad is False, f"{name} carries requires_grad=True"
        assert value.grad_fn is None, f"{name} carries grad_fn={value.grad_fn!r}"
        assert value.dim() == 0, f"{name} is not a scalar, shape {tuple(value.shape)}"


# ---------------------------------------------------------------------------
# Worked examples at the PushT target cell
# ---------------------------------------------------------------------------


def target_cell_case(gate: str, *, action_reduce: str = "sum", kind: str = "generic"):
    """The Target_Cell's shapes: ``b=32``, ``t=4``, ``196x8`` features, ``act`` ``(32, 4, 10)``.

    ``f = 5`` (frameskip) and ``d = 2`` (PushT's env action dim), so ``act.shape[-1] = 10`` and
    every latent step's block splits into 5 two-dimensional substeps.
    """
    acs = make_acs_case(
        kind=kind,
        batch_size=ACS_TARGET_CELL_BATCH,
        num_frames=ACS_TARGET_CELL_FRAMES,
        patches=ACS_TARGET_CELL_PATCHES,
        channels=ACS_TARGET_CELL_CHANNELS,
        substeps=ACS_TARGET_CELL_SUBSTEPS,
        env_action_dim=ACS_TARGET_CELL_ENV_ACTION_DIM,
        seed=0,
    )
    assert acs.act.shape[-1] == 10, "the target cell packs 5 substeps x 2 dims into 10 channels"
    model = build_stub_world_model(
        agg_type="mlp",
        straighten="acsaggcos1e-1",
        acs_action_reduce=action_reduce,
        acs_gate=gate,
        emb_dim=ACS_TARGET_CELL_CHANNELS,
        num_patches=ACS_TARGET_CELL_PATCHES,
        agg_out_dim=32,
        image_size=8,
        seed=0,
    )
    return model, acs


@pytest.mark.parametrize("gate", ACS_GATES)
def test_target_cell_gradient_is_unchanged_by_substituting_gate_values(gate):
    """Property 4 at the Target_Cell's shapes, for each of the four gate modes.

    A worked example beside the property: ``(32, 4, 196, 8)`` features into the 32-d aggregated
    space, 2 curvature triples per sample and 64 weights per batch, at the configuration the
    recorded 75.33 OL / 82.00 MPC baseline was trained in.
    """
    model, acs = target_cell_case(gate)
    d = acs.env_action_dim

    z_gate = acs.z.clone().requires_grad_(True)
    act_gate = acs.act.clone().requires_grad_(True)
    loss_gate, w, c, mask = acs_expression(
        model, z_gate, act_gate, env_action_dim=d, gate_seed=7
    )
    grad_gate, act_grad_gate = grads_wrt(loss_gate, [z_gate, act_gate])

    z_const = acs.z.clone().requires_grad_(True)
    act_const = acs.act.clone().requires_grad_(True)
    loss_const, _, _, _ = acs_expression(
        model, z_const, act_const, env_action_dim=d, w_override=fresh_leaf_like(w)
    )
    grad_const, act_grad_const = grads_wrt(loss_const, [z_const, act_const])

    assert tuple(w.shape) == (32, 2) and w.numel() == 64
    assert w.requires_grad is False and w.grad_fn is None
    assert_bitwise(loss_const, loss_gate, f"target-cell loss (gate={gate!r})")
    assert_bitwise(grad_const, grad_gate, f"target-cell d(L_acs)/dz (gate={gate!r})")
    assert_bitwise(
        act_grad_const, act_grad_gate, f"target-cell d(L_acs)/d(act) (gate={gate!r})"
    )
    assert float(act_grad_gate.abs().sum()) == 0.0, (
        "the target-cell ACS term has a gradient path into act"
    )
    assert grad_gate.abs().sum() > 0 or not bool(mask.any()), (
        "the target-cell gradient is identically zero, so the comparison says nothing"
    )
    assert torch.isfinite(c[mask]).all()


def test_permuted_gate_needs_the_seed_for_a_bitwise_comparison():
    """Pin the fact :func:`gate_for` exists for, so its point cannot be lost.

    ``_permute_gate`` calls ``torch.randperm``, so two unseeded ``permuted`` calls return
    different gates - the substitution test would then be comparing two different expressions and
    a failure would say nothing about detachment. Under a fixed seed the call is reproducible, and
    the permutation-invariant quantities (the sorted weight multiset, hence ``mean(w)``) agree with
    ``relu_cos`` either way, which is what makes the arm a null control (Requirement 13.4).
    """
    model, acs = target_cell_case("permuted")
    d = acs.env_action_dim
    _, _, _, mask = acs_expression(model, acs.z.clone(), acs.act.clone(), env_action_dim=d)

    torch.manual_seed(0)
    first = model.action_gate(acs.act.clone(), mask=mask, env_action_dim=d)
    torch.manual_seed(1)
    second = model.action_gate(acs.act.clone(), mask=mask, env_action_dim=d)
    assert not torch.equal(first, second), (
        "two differently seeded permuted gates came out identical; if the permutation ever "
        "stops depending on the RNG, the seeding in gate_for should be re-derived, not deleted"
    )
    assert_bitwise(
        gate_for(model, acs.act.clone(), mask, d, seed=0),
        first,
        "permuted gate under a repeated seed",
    )

    # Both are detached, and the permutation moves no weight in or out of the population.
    for w in (first, second):
        assert w.requires_grad is False and w.grad_fn is None

    relu_model, _ = target_cell_case("relu_cos")
    plain = relu_model.action_gate(acs.act.clone(), mask=mask, env_action_dim=d)
    assert_bitwise(
        torch.sort(first[mask]).values,
        torch.sort(plain[mask]).values,
        "the permuted gate's unmasked weight multiset",
    )


def test_static_batch_masks_every_triple_and_the_floor_binds():
    """The two degenerate reductions the expression above has to survive, made explicit.

    A fully static batch masks out every triple (``step_thresh`` is ``1e-6``), so the weighted mean
    sums an empty selection and ``WEIGHT_SUM_FLOOR`` carries the denominator; an all-antiparallel
    batch under ``relu_cos`` has every weight exactly 0 and the floor binds again. Both must be
    finite and both must still compare bitwise between the gate and its substituted values - which
    is why the property draws from every degenerate kind rather than from ``generic`` alone.
    """
    assert ACS_STEP_THRESH == 1e-6 and ACS_WEIGHT_SUM_FLOOR == 1e-3

    for kind in ("static", "antiparallel"):
        model, acs = target_cell_case("relu_cos", kind=kind)
        d = acs.env_action_dim
        z = acs.z.clone().requires_grad_(True)
        loss, w, _, mask = acs_expression(model, z, acs.act.clone(), env_action_dim=d)

        if kind == "static":
            assert not bool(mask.any()), "a fully static batch should mask out every triple"
        else:
            assert bool(mask.any())
            assert torch.equal(w[mask], torch.zeros_like(w[mask])), (
                "all-antiparallel actions must give w = 0 under relu_cos"
            )

        value = float(loss.detach())
        assert torch.isfinite(loss.detach()), f"kind={kind!r} produced a non-finite ACS term"
        assert value == 0.0, (
            f"kind={kind!r}: with no surviving weight the term must be exactly 0, got {value}"
        )

        (grad_gate,) = torch.autograd.grad(loss, z)
        z_const = acs.z.clone().requires_grad_(True)
        loss_const, _, _, _ = acs_expression(
            model, z_const, acs.act.clone(), env_action_dim=d, w_override=fresh_leaf_like(w)
        )
        (grad_const,) = torch.autograd.grad(loss_const, z_const)
        assert_bitwise(grad_const, grad_gate, f"d(L_acs)/dz (kind={kind!r})")
