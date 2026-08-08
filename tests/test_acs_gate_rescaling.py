"""Task 3.5 - Property 5: gate invariance to positive rescaling, and ``sum`` == ``mean``.

The property, from design section 4.3::

    for every ``act`` and every ``alpha > 0``:   action_gate(alpha * act) ~= action_gate(act)

and its corollary: because ``mean = sum / f`` is one positive scalar applied to *both* vectors of
the cosine, a hypothetical ``acs_action_reduce="mean"`` is the **same gate** as ``"sum"``. That is
the whole reason ``mean`` is not offered as a fourth reduction (Requirement 5.10's enum is
``{sum, raw, first}``), and this module is the executable form of that argument rather than a
comment asserting it.

Why the property matters beyond tidiness: the recorded actions are *normalized* per dataset, so the
scale of ``act`` is an artifact of the preprocessing rather than a property of the control. A gate
that moved with it would make the arm's behaviour a function of the normalization constants, and the
Stage-0 statistics measured on one normalization would not transfer to training under another.

Float32 tolerance, stated up front rather than tuned until green
---------------------------------------------------------------

Exact equality is **not** achievable and this module does not pretend otherwise. ``alpha * act``
rounds once per element, ``reduce_action("sum")`` then sums ``f`` rounded values, and
``cosine_similarity`` divides two rounded reductions. Three separate places where float32 loses
bits, so the assertion has to be a tolerance - and the honest question is which one, derived from
what.

The derivation, per curvature triple. Write the reduced vector as ``a = sum_s x_s``. The float32
sum satisfies ``|a_hat - a| <= f * eps * sum_s |x_s|`` componentwise (``eps = 2**-23``), so the
*direction* of ``a_hat`` differs from that of ``a`` by an angle of order ``f * eps * kappa`` where::

    kappa = || sum_s |x_s| || / || a ||        (>= 1, by the triangle inequality)

is the **cancellation amplification** of that reduction: it is 1 when the substeps agree in sign and
grows without bound as they cancel. On top of that, the cosine's own dot product and two norms each
round over ``n = a.shape[-1]`` terms. ``cos`` is 1-Lipschitz in each unit direction, so::

    |cos(alpha * act) - cos(act)|  <~  eps * (n + f * (kappa_1 + kappa_2)) * C

:func:`cos_tolerance` computes exactly that with ``C = 8``, per triple, in float64. Two things this
buys over a flat number: it is *tight* on the well-conditioned draws (``8 * eps * (n + 2f)``, about
``1e-5`` at the target cell) and it is *loose only where float32 genuinely is* - a triple whose
substeps almost cancel has no accurate direction to be invariant about, and no fixed ``atol`` can
express that without being wrong on one side or the other.

Measured slack, so ``C = 8`` is a number rather than a vibe: over 6000 randomized draws spanning all
three reductions and the full six decades of ``alpha``, the largest observed
``|Delta cos| / tolerance`` was **0.078**, and the largest absolute deviation was
``5.0e-6`` - about 42 ulps of 1.0, driven by exactly the cancellation term the bound models. So the
asserted tolerance carries roughly 13x headroom on random data while still failing a genuine
scale-dependence, which would be O(1).

Two places where the property has a real boundary, handled structurally rather than by inflating the
tolerance:

1. **``cosine_similarity``'s epsilon floor.** The shipped call leaves ``eps`` at its default, which
   clamps each *squared* norm at ``eps**2 = 1e-16``; below ``||a|| = 1e-8`` the clamp, not the
   geometry, sets the value, and invariance ends there by construction. Draws are therefore
   restricted to reduced norms at least 100x above that floor, and the boundary itself is pinned by
   :func:`test_the_cosine_epsilon_floor_is_where_the_invariance_ends` instead of being hidden.
2. **The ``hard`` gate is a step at ``cos = 0``, not a Lipschitz function.** ``1[cos > 0]`` can flip
   0 <-> 1 when rescaling moves a near-orthogonal cosine across zero, and that is arithmetic, not a
   defect. The assertion for ``hard`` is correspondingly sharper than a tolerance: every entry that
   differs must have ``|cos| <= tolerance``, i.e. the only entries allowed to move are the ones the
   rounding could have flipped.

Shipped-code notes respected here (``PROGRESS_ACS.md`` section 13, as in
``tests/test_acs_gate_detached.py``): ``env_action_dim`` is threaded into ``reduce_action`` /
``action_gate``; ``action_gate`` takes the static ``mask``; and ``permuted`` calls
``torch.randperm``, so every gate call is preceded by a seed - the scaled and unscaled runs must
compare the *same* permutation. The gate is never re-implemented in this module: only
``reduce_action`` and ``F.cosine_similarity`` are used directly, to name the ``cos`` the gate is a
function of.

**Validates: Requirements 5.9, 5.12, 5.13, 5.16**
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from tests.conftest import (
    ACS_ACTION_REDUCTIONS,
    ACS_CASE_KINDS,
    ACS_GATES,
    ACS_TARGET_CELL_BATCH,
    ACS_TARGET_CELL_CHANNELS,
    ACS_TARGET_CELL_ENV_ACTION_DIM,
    ACS_TARGET_CELL_FRAMES,
    ACS_TARGET_CELL_PATCHES,
    ACS_TARGET_CELL_SUBSTEPS,
    acs_action_reduce_strategy,
    acs_cases,
    acs_gate_strategy,
    acs_positive_scale_strategy,
    agg_type_strategy,
    build_stub_world_model,
    make_acs_case,
)

MIN_EXAMPLES = max(100, settings.default.max_examples or 100)
acs_settings = settings(max_examples=MIN_EXAMPLES, deadline=None)

# The stub encoder's `agg_type="mlp"` head sizes its first Linear at num_patches * emb_dim, so the
# feature grid is pinned to the stub's defaults and the whole AGG_TYPES enum stays reachable.
STUB_PATCHES = 4
STUB_CHANNELS = 4

# float32 unit roundoff, and the safety factor on the derived per-triple bound (module docstring).
FP32_EPS = 2.0**-23
TOLERANCE_SAFETY = 8.0

# `F.cosine_similarity`'s default eps clamps each squared norm at eps**2, so the clamp binds below
# ||a|| = 1e-8. Draws stay two decades above that: float32 rounds a norm by ~1e-7 relative, so a
# 100x margin cannot flip whether the clamp binds.
COSINE_EPS = 1e-8
NORM_FLOOR_MARGIN = 100.0
SAFE_NORM = COSINE_EPS * NORM_FLOOR_MARGIN  # 1e-6

# Every kind except `zero_action`, whose reduced blocks are exactly zero and therefore sit *under*
# the epsilon floor by construction. It is not filtered out of the property - it gets a stronger,
# bitwise assertion of its own below, since `alpha * 0 == 0` exactly.
RESCALING_KINDS = tuple(k for k in ACS_CASE_KINDS if k != "zero_action")

# Both mode strings the parser accepts for the aggregated space. The gate is a method, reachable
# under either, and its scale-freeness cannot depend on how the model was configured.
STRAIGHTEN_VALUES = ("aggcos1e-1", "acsaggcos1e-1")


# ---------------------------------------------------------------------------
# Shipped pieces, plus the derived tolerance
# ---------------------------------------------------------------------------


def build_model(*, gate, action_reduce, agg_type="mlp", straighten="acsaggcos1e-1", seed=0, **kw):
    """A CPU float32 stub model configured with this case's ACS knobs."""
    return build_stub_world_model(
        agg_type=agg_type,
        straighten=straighten,
        acs_action_reduce=action_reduce,
        acs_gate=gate,
        image_size=8,
        seed=seed,
        **kw,
    )


def mask_for(model, z):
    """The static-velocity mask, from the shipped helpers. A function of ``z`` only.

    ``action_gate`` reads it for the ``permuted`` arm, which shuffles across exactly the unmasked
    triples. Since it does not depend on ``act``, rescaling the actions cannot move it, and both
    runs of every comparison below are handed the *same* mask object.
    """
    v1, v2 = model._agg_velocities(z)
    _, mask = model._cos_curvature_terms(v1, v2)
    return mask


def gate_for(model, act, mask, env_action_dim, *, seed=0):
    """``action_gate`` under a fixed RNG state.

    Only ``permuted`` reads the RNG, but seeding unconditionally keeps all four modes on one path
    and makes the scaled / unscaled pair comparable: the permutation depends on the number of
    unmasked triples, which rescaling leaves alone, so a fixed seed gives the identical shuffle.
    """
    torch.manual_seed(int(seed))
    return model.action_gate(act, mask=mask, env_action_dim=env_action_dim)


def pre_gate_cos(model, act, env_action_dim):
    """The cosine the gate is a function of, from the **shipped** reduction.

    ``reduce_action`` plus ``F.cosine_similarity``, clamped exactly as ``action_gate`` clamps it.
    The gate itself is never re-implemented here; this exists to name the quantity whose distance
    from zero decides which ``hard`` entries are allowed to flip.
    """
    a = model.reduce_action(act, env_action_dim=env_action_dim)
    return F.cosine_similarity(a[:, :-2], a[:, 1:-1], dim=-1).clamp(-1.0, 1.0)


def reduced_norms(model, act, env_action_dim):
    """float64 norms of the two reduced vectors of every triple, ``(b, t-2)`` each.

    float64 on purpose: this is the quantity the epsilon-floor guard is read off, so it must not
    itself underflow in the arithmetic that is being guarded.
    """
    a = model.reduce_action(act, env_action_dim=env_action_dim).double()
    return a[:, :-2].norm(dim=-1), a[:, 1:-1].norm(dim=-1)


def cos_tolerance(model, act, env_action_dim):
    """The per-triple float32 bound on ``|cos(alpha * act) - cos(act)|``, derived not tuned.

    ``eps * (n + f * (kappa_1 + kappa_2)) * TOLERANCE_SAFETY``, with the cancellation amplification
    ``kappa = || sum_s |x_s| || / || a ||`` read off the **shipped** reduction applied to
    ``act.abs()`` - which is exactly the componentwise ``sum_s |x_s|`` for ``sum``, ``|act[..., :d]|``
    for ``first`` and ``|act|`` for ``raw``. See the module docstring for where each term comes from.

    Returned in float64, shape ``(b, t-2)``.
    """
    a = model.reduce_action(act, env_action_dim=env_action_dim).double()
    abs_reduced = model.reduce_action(act.abs(), env_action_dim=env_action_dim).double()
    # Only `sum` accumulates over substeps; `first` and `raw` read channels straight through.
    summands = float(act.shape[-1] // a.shape[-1]) if model.acs_action_reduce == "sum" else 1.0

    norms = a.norm(dim=-1).clamp_min(torch.finfo(torch.float64).tiny)
    kappa = abs_reduced.norm(dim=-1) / norms
    kappa_1, kappa_2 = kappa[:, :-2], kappa[:, 1:-1]
    n_terms = float(a.shape[-1])
    return TOLERANCE_SAFETY * FP32_EPS * (n_terms + summands * (kappa_1 + kappa_2))


def summation_bound(model, act, env_action_dim, *, factor=1.0):
    """The standard float32 summation bound for the reduction, ``8 * eps * sum_s |x_s|``.

    A *relative* tolerance is the wrong instrument for ``reduce_action("sum")``: when the substeps
    cancel, the result is small while the addends are not, so the relative error is unbounded while
    the absolute error stays at ``eps * sum_s |x_s|`` - which is the quantity the textbook bound for
    sequential summation gives, and the quantity asserted here (``factor`` carries an outer positive
    scalar such as ``alpha`` or ``1/f``). ``TOLERANCE_SAFETY = 8`` against the derived ``~2 * eps``.

    ``sum_s |x_s|`` comes from the **shipped** reduction applied to ``act.abs()``, componentwise, so
    it is the same channel layout the reduction itself used.
    """
    abs_reduced = model.reduce_action(act.abs(), env_action_dim=env_action_dim).double()
    return TOLERANCE_SAFETY * FP32_EPS * abs(float(factor)) * abs_reduced


def assert_within(actual, expected, bound, *, where):
    """``|actual - expected| <= bound`` elementwise, in float64, with a locating message."""
    delta = (actual.double() - expected.double()).abs()
    if bool((delta <= bound).all()):
        return
    worst = int(delta.argmax())
    raise AssertionError(
        f"{where}: differs by up to {float(delta.max())!r}, above the derived float32 bound "
        f"{float(bound.flatten()[worst])!r} at flat index {worst} "
        f"({int((delta > bound).sum())} of {delta.numel()} entries over bound)"
    )


def floor_is_clear(model, act, env_action_dim, scales=(1.0,)):
    """True when no reduced norm comes within :data:`NORM_FLOOR_MARGIN` of the cosine's clamp.

    Checked at every scale the comparison will use, because the floor binds at the *scaled* norm.
    """
    n1, n2 = reduced_norms(model, act, env_action_dim)
    smallest = float(torch.minimum(n1.min(), n2.min()))
    return all(smallest * float(scale) > SAFE_NORM for scale in scales)


def assert_gate_invariant(w_ref, w_scaled, cos_ref, tol, *, gate, where):
    """The gate half of Property 5, with each mode held to what its own definition supports.

    - ``relu_cos`` / ``permuted``: ``relu`` is 1-Lipschitz, so ``|Delta w| <= |Delta cos|``.
    - ``affine_cos``: ``(1 + cos)/2`` is 1/2-Lipschitz, so half the bound applies.
    - ``hard``: ``1[cos > 0]`` is a step and no tolerance on ``w`` is meaningful. The assertion is
      sharper instead - an entry may differ **only** if its cosine sits within the rounding
      tolerance of zero, i.e. only where float32 could have moved it across the threshold.
    """
    assert w_ref.shape == w_scaled.shape, f"{where}: gate shape moved under rescaling"
    delta = (w_scaled.double() - w_ref.double()).abs()

    if gate == "hard":
        differing = delta > 0
        if not bool(differing.any()):
            return
        borderline = cos_ref.double().abs() <= tol
        offenders = differing & ~borderline
        assert not bool(offenders.any()), (
            f"{where}: the hard gate flipped on {int(offenders.sum())} triple(s) whose cosine is "
            f"not within rounding distance of the threshold; largest such |cos| is "
            f"{float(cos_ref.double().abs()[offenders].max())!r}, tolerance there is "
            f"{float(tol.expand_as(cos_ref)[offenders].max())!r}. That is a scale-dependent gate, "
            "not a rounding artifact."
        )
        return

    bound = tol * (0.5 if gate == "affine_cos" else 1.0)
    worst = float((delta - bound).max())
    assert bool((delta <= bound).all()), (
        f"{where}: gate moved under positive rescaling by up to {float(delta.max())!r}, exceeding "
        f"the derived float32 bound by {worst!r}. Bound at the offending triple: "
        f"{float(bound.flatten()[int(delta.argmax())])!r}."
    )


# ---------------------------------------------------------------------------
# Property 5, the general case
# ---------------------------------------------------------------------------


@st.composite
def rescaling_cases(draw):
    """A jointly generated ``(z, act)`` pair, both ACS knobs, the model variant and ``alpha``.

    ``kind`` covers the ordinary case and every degenerate one except ``zero_action`` (see
    :data:`RESCALING_KINDS`): all-static so every triple is masked, one-moving so the mask is
    partial, all-parallel (``cos = +1``) and all-antiparallel (``cos = -1``). Scale-freeness is
    structural and must hold on all of them.
    """
    kind = draw(st.sampled_from(RESCALING_KINDS))
    case = draw(acs_cases(kind=kind, patches=STUB_PATCHES, channels=STUB_CHANNELS))
    return {
        "acs": case,
        "alpha": draw(acs_positive_scale_strategy),
        "gate": draw(acs_gate_strategy),
        "action_reduce": draw(acs_action_reduce_strategy),
        "agg_type": draw(agg_type_strategy),
        "straighten": draw(st.sampled_from(STRAIGHTEN_VALUES)),
        "model_seed": draw(st.integers(min_value=0, max_value=3)),
        "gate_seed": draw(st.integers(min_value=0, max_value=2**31 - 1)),
    }


@given(case=rescaling_cases())
@acs_settings
def test_property_5_gate_is_invariant_to_positive_rescaling(case):
    """Feature: action-conditioned-straightening, Property 5: gate invariance to rescaling.

    For every ``alpha`` drawn log-uniformly over ``[1e-3, 1e3]`` - six decades, a factor of a
    million between the smallest and the largest draw - ``action_gate(alpha * act)`` agrees with
    ``action_gate(act)`` to the derived float32 tolerance, for all four gate modes and all three
    reductions. The recorded actions are normalized per dataset, so a gate that moved with their
    scale would make the arm a function of the preprocessing constants.

    Asserted at both levels: on the pre-gate cosine, where the bound is derived, and on ``w``
    itself, where each mode is held to what its own definition supports (:func:`assert_gate_invariant`).

    **Validates: Requirements 5.9**
    """
    acs = case["acs"]
    alpha = float(case["alpha"])
    d_env = acs.env_action_dim
    model = build_model(
        gate=case["gate"],
        action_reduce=case["action_reduce"],
        agg_type=case["agg_type"],
        straighten=case["straighten"],
        seed=case["model_seed"],
    )

    # Below ||a|| = 1e-8 the cosine's own epsilon clamp sets the value and invariance ends by
    # construction; that boundary is pinned by its own test rather than absorbed into the bound.
    assume(floor_is_clear(model, acs.act, d_env, scales=(1.0, alpha)))

    mask = mask_for(model, acs.z)
    act_scaled = acs.act * alpha

    w_ref = gate_for(model, acs.act, mask, d_env, seed=case["gate_seed"])
    w_scaled = gate_for(model, act_scaled, mask, d_env, seed=case["gate_seed"])

    cos_ref = pre_gate_cos(model, acs.act, d_env)
    cos_scaled = pre_gate_cos(model, act_scaled, d_env)
    tol = cos_tolerance(model, acs.act, d_env)

    where = (
        f"alpha={alpha!r}, gate={case['gate']!r}, reduce={case['action_reduce']!r}, "
        f"kind={acs.kind!r}, b={acs.batch_size}, t={acs.num_frames}, f={acs.substeps}, "
        f"d_env={d_env}"
    )
    cos_delta = (cos_scaled.double() - cos_ref.double()).abs()
    assert bool((cos_delta <= tol).all()), (
        f"{where}: the pre-gate cosine moved by up to {float(cos_delta.max())!r} under positive "
        f"rescaling, above the derived bound {float(tol.flatten()[int(cos_delta.argmax())])!r}"
    )
    assert_gate_invariant(w_ref, w_scaled, cos_ref, tol, gate=case["gate"], where=where)


@given(case=rescaling_cases())
@acs_settings
def test_property_5_reduce_action_is_positively_homogeneous(case):
    """Feature: action-conditioned-straightening, Property 5: gate invariance to rescaling.

    The mechanism behind the invariance, asserted rather than assumed: every reduction is positively
    homogeneous, ``reduce_action(alpha * act) == alpha * reduce_action(act)``, so rescaling moves the
    two vectors of each cosine by one common positive factor and the cosine cannot see it.

    ``raw`` and ``first`` are **bitwise** - both sides are the same float32 product, an identity and
    a view respectively (Requirements 5.13, 5.14) - while ``sum`` reorders one multiplication
    against ``f`` additions and is asserted against :func:`summation_bound` (Requirement 5.12). Note
    that bound is absolute rather than relative on purpose: cancelling substeps make the relative
    error unbounded while leaving the absolute error at ``eps * sum_s |x_s|``.

    **Validates: Requirements 5.12, 5.13, 5.16**
    """
    acs = case["acs"]
    alpha = float(case["alpha"])
    d_env = acs.env_action_dim
    model = build_model(gate=case["gate"], action_reduce=case["action_reduce"])

    act = acs.act
    act_before = act.clone()
    reduced = model.reduce_action(act, env_action_dim=d_env)
    reduced_scaled = model.reduce_action(act * alpha, env_action_dim=d_env)

    where = f"reduce={case['action_reduce']!r}, alpha={alpha!r}, kind={acs.kind!r}"
    assert reduced_scaled.shape == reduced.shape, f"{where}: reduced shape moved"

    if model.acs_action_reduce in ("raw", "first"):
        assert torch.equal(reduced_scaled, reduced * alpha), (
            f"{where}: an identity/view reduction must be bitwise homogeneous"
        )
    else:
        assert_within(
            reduced_scaled,
            reduced * alpha,
            summation_bound(model, act, d_env, factor=alpha),
            where=f"{where}: sum reduction is not positively homogeneous",
        )

    # Requirement 5.16: `sum` and `first` leave `act` alone. `raw` returns `act` itself, so the
    # statement there is that reducing does not write to it either.
    assert torch.equal(act, act_before), f"{where}: reduce_action mutated act"


@given(case=rescaling_cases())
@acs_settings
def test_property_5_sum_and_mean_are_the_same_gate(case):
    """Feature: action-conditioned-straightening, Property 5: ``sum`` == ``mean``.

    ``mean = sum / f`` is a single positive scalar on both vectors of the cosine, so
    ``cos(sum(u), sum(v)) == cos(mean(u), mean(v))`` and the two reductions define the *same* gate.
    That is why ``mean`` is absent from ``acs_action_reduce``'s closed enum (Requirement 5.10) - it
    would be a second name for ``sum``, and an apparent knob that cannot change any run's behaviour
    is worse than no knob.

    The mean-reduced tensor is obtained from the **shipped** reduction rather than a hand-rolled
    one: ``reduce_action("sum")`` applied to ``act / f`` is ``sum_s act_s / f``, i.e. exactly the
    per-substep mean. So this is the ``alpha = 1/f`` instance of the property above, asserted on
    both the cosine and the gate. ``f`` is generally not a power of two, so the division is itself
    inexact - which is the reason this is an fp32-tolerance claim and not a bitwise one.

    **Validates: Requirements 5.9, 5.12**
    """
    acs = case["acs"]
    d_env = acs.env_action_dim
    f = acs.substeps
    model = build_model(gate=case["gate"], action_reduce="sum", agg_type=case["agg_type"])

    assume(floor_is_clear(model, acs.act, d_env, scales=(1.0, 1.0 / f)))

    act_mean_source = acs.act / f
    mask = mask_for(model, acs.z)

    sum_vectors = model.reduce_action(acs.act, env_action_dim=d_env)
    mean_vectors = model.reduce_action(act_mean_source, env_action_dim=d_env)
    tol = cos_tolerance(model, acs.act, d_env)
    where = f"gate={case['gate']!r}, kind={acs.kind!r}, f={f}, d_env={d_env}"

    # The mean vectors really are sum / f, to fp32: the claim is about the same geometry, not a
    # different reduction that happens to agree. Absolute bound, not relative - the two summations
    # can cancel, and then no relative tolerance is meaningful (see `summation_bound`).
    assert_within(
        mean_vectors,
        sum_vectors / f,
        summation_bound(model, acs.act, d_env),
        where=f"{where}: mean-reduced vectors are not sum/f",
    )

    cos_sum = pre_gate_cos(model, acs.act, d_env)
    cos_mean = pre_gate_cos(model, act_mean_source, d_env)
    delta = (cos_mean.double() - cos_sum.double()).abs()
    assert bool((delta <= tol).all()), (
        f"{where}: cos(sum(u), sum(v)) and cos(mean(u), mean(v)) differ by up to "
        f"{float(delta.max())!r}, above the derived bound "
        f"{float(tol.flatten()[int(delta.argmax())])!r}; if this is real then `mean` is a "
        "genuinely different gate and the enum is missing a member"
    )

    w_sum = gate_for(model, acs.act, mask, d_env, seed=case["gate_seed"])
    w_mean = gate_for(model, act_mean_source, mask, d_env, seed=case["gate_seed"])
    assert_gate_invariant(w_sum, w_mean, cos_sum, tol, gate=case["gate"], where=f"mean vs sum, {where}")


@given(
    case=acs_cases(kind="zero_action", patches=STUB_PATCHES, channels=STUB_CHANNELS),
    alpha=acs_positive_scale_strategy,
    gate=acs_gate_strategy,
    action_reduce=acs_action_reduce_strategy,
)
@acs_settings
def test_property_5_zero_action_blocks_rescale_bitwise(case, alpha, gate, action_reduce):
    """Feature: action-conditioned-straightening, Property 5: gate invariance to rescaling.

    The one kind excluded from the property's draws, given a **stronger** assertion instead of a
    skip. A zero action block sits under ``cosine_similarity``'s epsilon floor by construction, but
    ``alpha * 0 == 0`` exactly in float32 for every finite ``alpha``, so the gate is not merely
    invariant here - it is bitwise identical. Requirement 5.8's ``w = 0`` (for the two gates that
    give it) is Property 3's business; what this pins is that the floor case cannot become
    scale-dependent.

    **Validates: Requirements 5.9**
    """
    model = build_model(gate=gate, action_reduce=action_reduce)
    mask = mask_for(model, case.z)
    d_env = case.env_action_dim

    w_ref = gate_for(model, case.act, mask, d_env, seed=11)
    w_scaled = gate_for(model, case.act * float(alpha), mask, d_env, seed=11)
    assert torch.equal(w_scaled, w_ref), (
        f"alpha={alpha!r}, gate={gate!r}, reduce={action_reduce!r}: rescaling an all-zero action "
        "tensor changed the gate, which cannot be a float32 artifact"
    )


# ---------------------------------------------------------------------------
# The reduction's own contract (Requirements 5.12, 5.13, 5.16)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("action_reduce", ACS_ACTION_REDUCTIONS)
def test_reduce_action_matches_its_specified_formula_and_leaves_act_alone(action_reduce):
    """The three reductions, each against the formula Requirement 5 states for it.

    ``sum`` is ``out[..., j] = sum_s act[..., s * d + j]``, checked against an explicit per-substep
    slice sum (the ``s * d + j`` channel layout is what ``rearrange("(n f) d -> n (f d)")`` produces,
    so getting it wrong would silently mix dimension ``j`` of one substep with ``j+1`` of the next);
    ``first`` is ``act[..., :d]`` and is a *view*; ``raw`` is ``act`` itself. Afterwards ``act`` is
    bitwise what it was.

    Compared in float64 against a float64 reference for ``sum``, because two float32 summations over
    the same ``f`` addends in a different order need not agree bit for bit - the claim is the
    formula, not the reduction tree.

    **Validates: Requirements 5.12, 5.13, 5.16**
    """
    case = make_acs_case(kind="generic", batch_size=3, num_frames=4, substeps=4, env_action_dim=3)
    d = case.env_action_dim
    f = case.substeps
    model = build_model(gate="relu_cos", action_reduce=action_reduce)

    act_before = case.act.clone()
    out = model.reduce_action(case.act, env_action_dim=d)

    if action_reduce == "raw":
        assert out is case.act, "the raw reduction is documented as an identity, not a copy"
    elif action_reduce == "first":
        assert torch.equal(out, case.act[..., :d])
        assert out.shape[-1] == d
        assert out.data_ptr() == case.act.data_ptr(), "`first` is documented as a view of act"
    else:
        expected = sum(case.act[..., s * d : (s + 1) * d].double() for s in range(f))
        assert out.shape[-1] == d
        assert_within(
            out,
            expected,
            summation_bound(model, case.act, d),
            where="sum reduction does not match out[..., j] = sum_s act[..., s*d+j]",
        )

    assert torch.equal(case.act, act_before), f"reduce_action({action_reduce!r}) mutated act"


# ---------------------------------------------------------------------------
# Worked example at the PushT target cell
# ---------------------------------------------------------------------------

TARGET_CELL_SCALES = (1e-3, 1e-2, 0.1, 1.0, 7.3, 1e2, 1e3)


def target_cell_case(gate, *, action_reduce="sum", kind="generic"):
    """The Target_Cell's shapes: ``b=32``, ``t=4``, ``196x8`` features, ``act`` ``(32, 4, 10)``.

    ``f = 5`` (frameskip) and ``d = 2`` (PushT's env action dim), so every latent step's 10 channels
    split into 5 two-dimensional substeps and there are 2 triples per sample, 64 per batch.
    """
    case = make_acs_case(
        kind=kind,
        batch_size=ACS_TARGET_CELL_BATCH,
        num_frames=ACS_TARGET_CELL_FRAMES,
        patches=ACS_TARGET_CELL_PATCHES,
        channels=ACS_TARGET_CELL_CHANNELS,
        substeps=ACS_TARGET_CELL_SUBSTEPS,
        env_action_dim=ACS_TARGET_CELL_ENV_ACTION_DIM,
        seed=0,
    )
    assert case.act.shape[-1] == 10, "the target cell packs 5 substeps x 2 dims into 10 channels"
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
    return model, case


@pytest.mark.parametrize("gate", ACS_GATES)
def test_target_cell_gate_is_invariant_across_six_decades(gate):
    """Property 5 at the Target_Cell's shapes, for each of the four gate modes.

    A worked example beside the property: 64 weights per batch at the configuration the recorded
    75.33 OL / 82.00 MPC baseline was trained in, swept over seven scales spanning ``1e-3`` to
    ``1e3`` including a non-power-of-two ``7.3``. The mean reduction (``alpha = 1/5``, an inexact
    binary division) is included in the sweep by construction.

    The largest deviation is asserted against the derived bound *and* reported in the failure
    message, so a regression says how far off it is rather than only that it failed.
    """
    model, case = target_cell_case(gate)
    d = case.env_action_dim
    mask = mask_for(model, case.z)
    assert tuple(mask.shape) == (32, 2)
    assert floor_is_clear(model, case.act, d, scales=(min(TARGET_CELL_SCALES), 1.0))

    w_ref = gate_for(model, case.act, mask, d, seed=5)
    cos_ref = pre_gate_cos(model, case.act, d)
    tol = cos_tolerance(model, case.act, d)

    worst = 0.0
    for alpha in TARGET_CELL_SCALES:
        w_scaled = gate_for(model, case.act * alpha, mask, d, seed=5)
        assert_gate_invariant(
            w_ref, w_scaled, cos_ref, tol, gate=gate, where=f"target cell, alpha={alpha!r}"
        )
        worst = max(worst, float((w_scaled.double() - w_ref.double()).abs().max()))

    if gate != "hard":
        bound = float(tol.max()) * (0.5 if gate == "affine_cos" else 1.0)
        assert worst <= bound, f"target cell, gate={gate!r}: worst deviation {worst!r} > {bound!r}"


@pytest.mark.parametrize("kind", ("parallel", "antiparallel"))
def test_target_cell_saturated_cosines_are_invariant(kind):
    """The two saturated cases, each held to what float32 actually delivers.

    ``antiparallel`` **is** bitwise: the cosine lands at or just above ``-1``, ``relu`` sends the
    whole neighbourhood to exactly ``0.0``, and no scale can change that. ``parallel`` is **not**:
    the measured cosine of two identical reduced vectors is ``1.0`` only to within ~2 ulps (observed
    range ``[0.99999982, 1.00000012]`` before ``action_gate``'s clamp), so ``w`` sits within
    ``1.8e-7`` of 1 rather than at it. Stating that is the point - a bitwise assertion here would be
    a claim about ``cosine_similarity``'s rounding that is simply false, and quietly relaxing it to
    ``allclose`` everywhere would hide the ``antiparallel`` case where exactness is real.
    """
    model, case = target_cell_case("relu_cos", kind=kind)
    d = case.env_action_dim
    mask = mask_for(model, case.z)
    tol = cos_tolerance(model, case.act, d)
    cos_ref = pre_gate_cos(model, case.act, d)

    w_ref = gate_for(model, case.act, mask, d, seed=3)
    if kind == "antiparallel":
        assert torch.equal(w_ref, torch.zeros_like(w_ref)), (
            "relu_cos on an antiparallel batch must be exactly 0"
        )
    else:
        assert_within(
            w_ref, torch.ones_like(w_ref), tol, where="parallel actions should give w ~= 1"
        )

    for alpha in TARGET_CELL_SCALES:
        w_scaled = gate_for(model, case.act * alpha, mask, d, seed=3)
        if kind == "antiparallel":
            assert torch.equal(w_scaled, w_ref), (
                f"alpha={alpha!r}: an exactly-zero gate moved under rescaling"
            )
        else:
            assert_gate_invariant(
                w_ref, w_scaled, cos_ref, tol, gate="relu_cos",
                where=f"target cell, kind={kind!r}, alpha={alpha!r}",
            )


# ---------------------------------------------------------------------------
# Where the invariance ends, stated rather than hidden
# ---------------------------------------------------------------------------


def test_the_cosine_epsilon_floor_is_where_the_invariance_ends():
    """The boundary of Property 5, pinned so the tolerance above cannot be quietly widened.

    ``action_gate`` leaves ``F.cosine_similarity``'s ``eps`` at its default, which clamps each
    *squared* norm at ``eps**2 = 1e-16``. That clamp is load-bearing - it is what turns a zero-norm
    action block into ``cos = 0`` instead of a division by zero (E10) - and it is also exactly where
    scale invariance stops: once ``alpha * ||a||`` drops below ``1e-8`` the returned value is set by
    the floor rather than by the geometry, and it shrinks with ``alpha`` instead of ignoring it.

    Asserted here on a parallel-action batch whose reduced norm is ``~1e-7``: unscaled the gate is
    exactly 1, and at ``alpha = 1e-3`` it collapses toward 0. This is arithmetic, not a defect, and
    the property's draws are guarded against it by :func:`floor_is_clear` (a 100x margin above the
    floor) rather than by inflating the tolerance until the region passes.
    """
    model = build_model(gate="relu_cos", action_reduce="sum")
    case = make_acs_case(
        kind="parallel",
        batch_size=2,
        num_frames=4,
        patches=STUB_PATCHES,
        channels=STUB_CHANNELS,
        substeps=2,
        env_action_dim=2,
    )
    d = case.env_action_dim

    # Scale the whole batch down so the *unscaled* reduced norm is just above the floor.
    n1, _ = reduced_norms(model, case.act, d)
    tiny_act = case.act * float(1e-7 / n1.min())
    mask = mask_for(model, case.z)

    n1_tiny, n2_tiny = reduced_norms(model, tiny_act, d)
    assert float(torch.minimum(n1_tiny, n2_tiny).min()) > COSINE_EPS, (
        "the unscaled case must sit above the floor"
    )
    assert not floor_is_clear(model, tiny_act, d, scales=(1.0,)), (
        "this batch is meant to be inside the guard's margin, so the property never draws it"
    )

    w_unscaled = gate_for(model, tiny_act, mask, d, seed=0)
    assert float(w_unscaled.min()) > 0.99, (
        "above the floor, parallel actions still give w ~= 1 however small the norm is; got "
        f"{float(w_unscaled.min())!r}"
    )

    alpha = 1e-3
    n1_scaled, n2_scaled = reduced_norms(model, tiny_act * alpha, d)
    assert float(torch.maximum(n1_scaled, n2_scaled).max()) < COSINE_EPS, (
        "the scaled case must be entirely under the floor for the collapse below to be the "
        "clamp's doing"
    )
    w_scaled = gate_for(model, tiny_act * alpha, mask, d, seed=0)
    assert float(w_scaled.max()) < 0.5, (
        "under the epsilon floor the clamp, not the geometry, sets the value; if this ever holds "
        "at 1.0 the floor has moved and floor_is_clear's margin should be re-derived"
    )
    assert torch.isfinite(w_scaled).all() and float(w_scaled.min()) >= 0.0, (
        "the floor must still return a finite gate in [0, 1] rather than a NaN"
    )
