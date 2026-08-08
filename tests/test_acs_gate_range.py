"""Task 3.3 - Property 3: gate range and the parallel-action identity.

Design section 4.2 states the gate's contract as three facts, and this module is their executable
form:

1. **Range.** `0 <= w <= 1` elementwise, for every member of the `acs_gate` enum
   (Requirement 5.4). The gate is a *weight*: a value above 1 would let one triple carry more
   straightening pressure than the baseline's uniform mean ever gives it, and a negative value
   would push a triple's curvature *up*. Neither is a gate, and the weighted mean's convex-
   combination reading - "ACS can only reallocate pressure, never reduce it in aggregate"
   (section 4.1) - depends on the range holding, not on it holding approximately.
2. **The parallel identity.** Positively parallel reduced actions give `w = 1`
   (Requirement 5.5): a latent step whose control did not change direction gets the *full*
   baseline pressure. This is the upper anchor of the reallocation story, and it is what makes
   "flat gate at 1 reduces to `L_curv`" (Property 2) reachable on real data rather than only in
   the limit.
3. **The zero set, for `relu_cos` exactly.** `w = 0` on the whole `cos <= 0` half-space
   (Requirement 5.6): every action-reversing triple gets exactly zero pressure, which is the ACS
   hypothesis stated sharply rather than softly. `affine_cos` deliberately does *not* have this
   property - it returns `0.5` at orthogonality and reaches 0 only at exact antiparallelism - and
   the contrast is asserted here so the enum members cannot be confused for each other.

Shape is checked alongside them (Requirement 5.7): `w` is `(b, t - 2)` and matches the per-triple
curvature `c` and the static mask from ``_cos_curvature_terms`` elementwise, because the weighted
mean indexes all three with the same mask.

Where the `permuted` member sits in each clause
-----------------------------------------------

`permuted` (Requirement 13.4) is `relu_cos` shuffled across the batch's unmasked triples, so:

- **Range**: it holds, and trivially - a permutation moves values, it does not create them. Checked
  for all four members together.
- **Parallel identity**: it holds on an all-parallel batch, and *only* because every weight in that
  population is already 1, so the permutation is the identity on the multiset. It is asserted here
  for all four members with that reasoning made explicit, not smuggled.
- **Zero set**: it does **not** hold, by design. A permuted gate can land a 0 on a positive-cos
  triple and a positive weight on a reversing one - that is the whole point of the null control,
  which preserves the weight *distribution* and destroys the correspondence. There is a test below
  that demonstrates exactly this, so the restriction of clause 3 to `relu_cos` reads as a design
  fact rather than as an untested omission.

Why the parallel identity is asserted to a tolerance and not bitwise
--------------------------------------------------------------------

``F.cosine_similarity(a, a)`` is not bitwise 1 in float32: the dot product is `sum(a_i^2)` while
the denominator is `sqrt(sum(a_i^2))` squared, and `sqrt(s)**2 != s` in general. Measured on the
shipped path across all three reductions, the deviation is at most 3 ulps
(`3.58e-07`); the gate's `clamp(-1, 1)` removes the *upper* side of that error, so `w <= 1` is
exact while `w >= 1` is only exact to a few ulps. The tolerance below is stated in ulps rather than
as a round decimal so it cannot drift into hiding a real defect, and `w <= 1` is asserted exactly.

Two shipped-code notes this module respects (``PROGRESS_ACS.md`` section 13, same as
``tests/test_acs_gate_detached.py``):

- ``env_action_dim`` is **threaded** into ``reduce_action`` / ``action_gate``: `act.shape[-1] = 10`
  at the target cell cannot distinguish 5 substeps x 2 dims from 2 x 5, so the batch's protocol
  value is passed rather than guessed.
- ``action_gate`` takes the static ``mask``, because ``permuted`` shuffles across exactly the
  unmasked triples via ``torch.randperm``. Every call in this module is therefore preceded by a
  seed, so a `permuted` example is reproducible and a shrunk counterexample is re-runnable.

**Validates: Requirements 5.1, 5.4, 5.5, 5.6, 5.7, 13.4**
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.conftest import (
    ACS_ACTION_REDUCTIONS,
    ACS_GATES,
    ACS_TARGET_CELL_BATCH,
    ACS_TARGET_CELL_CHANNELS,
    ACS_TARGET_CELL_ENV_ACTION_DIM,
    ACS_TARGET_CELL_FRAMES,
    ACS_TARGET_CELL_PATCHES,
    ACS_TARGET_CELL_SUBSTEPS,
    ACS_TARGET_CELL_TRIPLES,
    ACS_TARGET_CELL_TRIPLES_PER_SAMPLE,
    acs_action_reduce_strategy,
    acs_cases,
    acs_gate_strategy,
    acs_parallel_action_cases,
    agg_type_strategy,
    build_stub_world_model,
    make_acs_case,
)

# Minimum 100 examples per the feature's testing convention, without capping a richer profile back
# down to 100. ``deadline=None``: the gate is a handful of elementwise ops, but a CPU-only box
# under load is not a wall clock.
MIN_EXAMPLES = max(100, settings.default.max_examples or 100)
acs_settings = settings(max_examples=MIN_EXAMPLES, deadline=None)

# The stub encoder's aggregated-space grid. ``agg_type="mlp"`` builds its first Linear at
# ``num_patches * emb_dim``, so the feature tensor's patch and channel axes are pinned to the
# stub's defaults and the whole ``AGG_TYPES`` enum stays reachable.
STUB_PATCHES = 4
STUB_CHANNELS = 4

# Both mode strings the parser accepts for the aggregated space: the arm's own setting and the
# baseline's. ``action_gate`` is a method, reachable under either, and its range must not depend on
# how the model was configured.
STRAIGHTEN_VALUES = ("aggcos1e-1", "acsaggcos1e-1")

# ``cos(a, a)`` is 1 only to a few float32 ulps (see the module docstring); measured worst case on
# the shipped path is 3 ulps, and 32 leaves headroom for a wider reduction without leaving the
# regime where "1" is the only plausible intent.
EPS32 = float(torch.finfo(torch.float32).eps)
PARALLEL_ULPS = 32
PARALLEL_ATOL = PARALLEL_ULPS * EPS32

# Gates whose zero set is exactly the ``cos <= 0`` half-space. ``affine_cos`` is excluded because
# it maps orthogonality to 0.5 by definition, and ``permuted`` because it deliberately breaks the
# weight-to-triple correspondence.
HALF_SPACE_ZERO_GATES = ("relu_cos", "hard")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def gate_for(model, act, mask, env_action_dim, *, seed: int = 0) -> torch.Tensor:
    """``action_gate`` under a fixed RNG state.

    Only ``permuted`` reads the RNG (``_permute_gate`` calls ``torch.randperm``), but seeding
    unconditionally keeps all four members on one code path here and makes every example
    reproducible, including a shrunk one.
    """
    torch.manual_seed(int(seed))
    return model.action_gate(act, mask=mask, env_action_dim=env_action_dim)


def build_model(*, gate: str, action_reduce: str, agg_type: str = "mlp", straighten="acsaggcos1e-1", seed: int = 0):
    """A CPU float32 stub model carrying this case's two ACS knobs."""
    return build_stub_world_model(
        agg_type=agg_type,
        straighten=straighten,
        acs_action_reduce=action_reduce,
        acs_gate=gate,
        image_size=8,
        seed=seed,
    )


def curvature_terms(model, z):
    """``(c, mask)`` from the shipped shared helpers, i.e. what the gate has to match in shape."""
    v1, v2 = model._agg_velocities(z)
    return model._cos_curvature_terms(v1, v2)


def reduced_cosine(model, act, env_action_dim) -> torch.Tensor:
    """`cos(a_k, a_{k+1})` for every triple, from the **shipped** reduction.

    This is not a second implementation of the gate: it is how the `cos <= 0` half-space is
    *identified* so the gate's behaviour on it can be asserted. The reduction itself is
    ``model.reduce_action``, so there is still exactly one definition of `a_k` in the repository
    (Property 19), and the gate function - the `relu`, the affine map, the indicator, the
    permutation - is never reproduced here.
    """
    a = model.reduce_action(act, env_action_dim=env_action_dim)
    return F.cosine_similarity(a[:, :-2], a[:, 1:-1], dim=-1).clamp(-1.0, 1.0)


def angled_actions(angles_deg, *, substeps: int = 3, magnitude: float = 2.0) -> torch.Tensor:
    """`act` of shape ``(1, len(angles_deg) + 2, substeps * 2)`` realising a prescribed angle sweep.

    ``angles_deg[k]`` is the turn between frame `k`'s action and frame `k + 1`'s: headings
    accumulate, starting at 0, and one extra frame is appended so that the `k`-th *triple* -
    the pair ``(a_k, a_{k+1})`` the gate compares - subtends exactly ``angles_deg[k]``. So
    ``len(angles_deg)`` angles give ``len(angles_deg)`` triples and `cos` is known in closed form
    rather than measured.

    Tiling the same 2-d vector over the substeps keeps the three reductions in agreement: `sum`
    scales it by ``substeps``, `first` takes it as is, `raw` sees a repeated profile, and a positive
    scalar cannot change a cosine (P5).
    """
    headings = [0.0]
    for angle in angles_deg:
        headings.append(headings[-1] + float(angle))
    # ``action_gate`` compares ``a[:, :-2]`` against ``a[:, 1:-1]``, i.e. the pairs ``(k, k+1)`` for
    # ``k <= t - 3``. One trailing frame is appended (turn 0, never compared) so that every
    # requested angle becomes a triple: ``t = len(angles_deg) + 2`` gives ``t - 2`` pairs subtending
    # ``angles_deg`` in order. ``act[:, t-1]`` drives the transition out of the window and is
    # correctly ignored by the gate (F1).
    headings.append(headings[-1])
    blocks = []
    for heading in headings:
        rad = math.radians(heading)
        step = torch.tensor([math.cos(rad), math.sin(rad)], dtype=torch.float32) * magnitude
        blocks.append(step.repeat(substeps))
    act = torch.stack(blocks).unsqueeze(0).contiguous()
    assert act.shape[1] == len(angles_deg) + 2
    return act


def direction_actions(directions, *, substeps: int = 3, magnitude: float = 2.0) -> torch.Tensor:
    """`act` built from explicit 2-d directions, one per frame, tiled over the substeps.

    Used where an angle in degrees cannot be represented exactly: ``math.cos(radians(90))`` is
    ``6.1e-17``, not 0, so the orthogonal boundary has to be built from axis-aligned vectors whose
    dot product is exactly zero. One trailing frame is appended for the same reason as in
    :func:`angled_actions`.
    """
    dirs = [torch.tensor(d, dtype=torch.float32) for d in directions]
    dirs.append(dirs[-1])
    blocks = [(d * magnitude).repeat(substeps) for d in dirs]
    return torch.stack(blocks).unsqueeze(0).contiguous()


@st.composite
def gate_cases(draw):
    """A jointly generated ``(z, act)`` pair plus the model variant and both ACS knobs.

    ``kind`` is left free, so the draws cover the ordinary case and every degenerate one:
    all-static (every triple masked), one-moving (partially masked), `b = 1`, all-parallel
    (`w = 1`), all-antiparallel (`w = 0` under `relu_cos` / `hard`) and zero-norm action blocks.
    The range is a structural property and must hold on all of them.
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
# Property 3
# ---------------------------------------------------------------------------


@given(case=gate_cases())
@acs_settings
def test_property_3_gate_range_is_the_unit_interval(case):
    """Feature: action-conditioned-straightening, Property 3: Gate range and parallel identity.

    For every member of the ``acs_gate`` enum, every reduction and every case kind:
    ``0 <= w <= 1`` elementwise, every entry finite, and the shape is ``(b, t - 2)`` matching the
    per-triple curvature ``c`` and the static mask elementwise - the three tensors the weighted
    mean indexes with one mask.

    Asserted exactly, not to a tolerance: the gate clamps ``cos`` to ``[-1, 1]`` before applying
    ``relu`` / the affine map / the indicator precisely so the range is a hard fact rather than
    ``cosine_similarity``'s float32 slop (Requirement 5.4).

    **Validates: Requirements 5.1, 5.4, 5.7, 13.4**
    """
    acs = case["acs"]
    model = build_model(
        gate=case["gate"],
        action_reduce=case["action_reduce"],
        agg_type=case["agg_type"],
        straighten=case["straighten"],
        seed=case["model_seed"],
    )
    c, mask = curvature_terms(model, acs.z.clone())
    w = gate_for(model, acs.act.clone(), mask, acs.env_action_dim, seed=case["gate_seed"])

    where = (
        f"gate={case['gate']!r}, reduce={case['action_reduce']!r}, kind={acs.kind!r}, "
        f"b={acs.batch_size}, t={acs.num_frames}, f={acs.substeps}, d_env={acs.env_action_dim}"
    )
    assert torch.isfinite(w).all(), f"{where}: the gate returned a non-finite weight"
    assert bool((w >= 0.0).all()), (
        f"{where}: min(w)={float(w.min())!r} < 0, so that triple's curvature would be pushed up"
    )
    assert bool((w <= 1.0).all()), (
        f"{where}: max(w)={float(w.max())!r} > 1, so that triple would carry more pressure than "
        "the baseline's uniform mean gives it"
    )
    assert w.dtype == torch.float32, f"{where}: gate dtype moved to {w.dtype}"
    assert tuple(w.shape) == (acs.batch_size, acs.num_frames - 2), (
        f"{where}: gate shape {tuple(w.shape)} is not (b, t-2)"
    )
    assert w.shape == c.shape == mask.shape, (
        f"{where}: gate {tuple(w.shape)}, c {tuple(c.shape)} and mask {tuple(mask.shape)} must "
        "match elementwise"
    )


@given(
    case=acs_parallel_action_cases(patches=STUB_PATCHES, channels=STUB_CHANNELS),
    gate=acs_gate_strategy,
    action_reduce=acs_action_reduce_strategy,
    gate_seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@acs_settings
def test_property_3_positively_parallel_actions_give_unit_weight(case, gate, action_reduce, gate_seed):
    """Feature: action-conditioned-straightening, Property 3: Gate range and parallel identity.

    Every consecutive reduced action pair positively parallel (the ``parallel`` case tiles one
    nonzero 2-d... `d_env`-d block over the substeps and repeats it across the frames, so
    ``cos = +1`` under all three reductions) implies ``w = 1`` for every triple, for all four gate
    members: ``relu(1) = 1``, ``(1 + 1)/2 = 1``, ``1[1 > 0] = 1``, and ``permuted`` shuffles a
    population that is already all ones, so it is the identity on that multiset.

    ``w <= 1`` is asserted **exactly** - that side is the clamp's doing - while ``w >= 1`` is
    asserted to 32 float32 ulps, because ``cos(a, a)`` computes ``sum(a_i^2)`` over
    ``sqrt(sum(a_i^2))**2`` and those differ by a couple of ulps. The cosine itself is checked to
    the same tolerance first, so a failure here separates "the case stopped being parallel" from
    "the gate stopped returning 1".

    **Validates: Requirements 5.1, 5.4, 5.5, 13.4**
    """
    model = build_model(gate=gate, action_reduce=action_reduce)
    _, mask = curvature_terms(model, case.z.clone())

    cos_a = reduced_cosine(model, case.act.clone(), case.env_action_dim)
    assert torch.allclose(cos_a, torch.ones_like(cos_a), atol=PARALLEL_ATOL, rtol=0.0), (
        f"reduce={action_reduce!r}, kind={case.kind!r}: the parallel case did not produce "
        f"cos = +1 (min {float(cos_a.min())!r}), so this example says nothing about the gate"
    )

    w = gate_for(model, case.act.clone(), mask, case.env_action_dim, seed=gate_seed)
    where = f"gate={gate!r}, reduce={action_reduce!r}, b={case.batch_size}, t={case.num_frames}"
    assert bool((w <= 1.0).all()), f"{where}: max(w)={float(w.max())!r} exceeded 1"
    assert torch.allclose(w, torch.ones_like(w), atol=PARALLEL_ATOL, rtol=0.0), (
        f"{where}: positively parallel actions must give w = 1, got min {float(w.min())!r} "
        f"(worst deviation {float((w - 1.0).abs().max())!r}, tolerance {PARALLEL_ATOL!r})"
    )
    if gate == "hard":
        # The indicator has no float error to forgive, so it is pinned exactly.
        assert torch.equal(w, torch.ones_like(w)), f"{where}: hard gate is not exactly 1"


@given(
    case=acs_cases(patches=STUB_PATCHES, channels=STUB_CHANNELS),
    action_reduce=acs_action_reduce_strategy,
    gate=st.sampled_from(HALF_SPACE_ZERO_GATES),
)
@acs_settings
def test_property_3_relu_cos_zeroes_exactly_the_reversing_half_space(case, action_reduce, gate):
    """Feature: action-conditioned-straightening, Property 3: Gate range and parallel identity.

    For ``relu_cos`` - the pre-registered default - the zero set is **exactly** the ``cos <= 0``
    half-space: every triple whose reduced action pair reversed direction (or is orthogonal, or has
    a zero-norm block, which ``cosine_similarity``'s ``eps`` sends to ``cos = 0``) gets ``w = 0``
    exactly, and every triple with ``cos > 0`` gets ``w > 0``. That is the ACS hypothesis stated
    sharply: no straightening pressure at all on the reversing half-space, grading preserved on the
    rest.

    ``hard`` is swept alongside it because it has the same *support* by construction (design
    section 4.2) and differs only in how the surviving mass is graded - so a change that broke the
    support of one and not the other would be a real divergence.

    Exact, not to a tolerance: ``relu`` of a non-positive number is ``0.0``, and ``1[cos > 0]`` of
    one is ``0.0``. Drawn over every case kind, so the reversing set is sometimes empty, sometimes
    the whole batch (``antiparallel``) and sometimes all-zero-norm (``zero_action``).

    **Validates: Requirements 5.1, 5.6, 5.8**
    """
    model = build_model(gate=gate, action_reduce=action_reduce)
    _, mask = curvature_terms(model, case.z.clone())
    cos_a = reduced_cosine(model, case.act.clone(), case.env_action_dim)
    w = gate_for(model, case.act.clone(), mask, case.env_action_dim)

    reversing = cos_a <= 0.0
    where = f"gate={gate!r}, reduce={action_reduce!r}, kind={case.kind!r}"
    if bool(reversing.any()):
        assert torch.equal(w[reversing], torch.zeros_like(w[reversing])), (
            f"{where}: {int(reversing.sum())} triple(s) with cos <= 0 kept a nonzero weight "
            f"(max {float(w[reversing].max())!r}); the reversing half-space must get exactly zero "
            "pressure"
        )
    if bool((~reversing).any()):
        assert bool((w[~reversing] > 0.0).all()), (
            f"{where}: a triple with cos > 0 was zeroed (min {float(w[~reversing].min())!r}), so "
            "the zero set is larger than the reversing half-space"
        )
    if case.kind == "antiparallel":
        assert bool(reversing.all()), "the antiparallel case must reverse on every triple"
        assert torch.equal(w, torch.zeros_like(w)), (
            f"{where}: an all-antiparallel batch must give w = 0 everywhere"
        )
    if case.kind == "zero_action":
        assert torch.equal(cos_a, torch.zeros_like(cos_a)), (
            "a zero-norm action block must fall out as cos = 0 through cosine_similarity's eps"
        )
        assert torch.equal(w, torch.zeros_like(w)), f"{where}: zero-norm blocks must give w = 0"


# ---------------------------------------------------------------------------
# The half-space, swept deterministically over angles
# ---------------------------------------------------------------------------


# 0 deg is parallel, 180 deg antiparallel, and 89.9 / 90.1 straddle the half-space boundary - which
# is where an off-by-a-sign in the gate would hide. Exactly 90 deg is deliberately absent: it is not
# representable (``cos(radians(90)) == 6.1e-17``), so the boundary itself is pinned by
# :func:`test_gate_at_exact_orthogonality` using axis-aligned vectors instead.
SWEEP_ANGLES = (0.0, 10.0, 45.0, 80.0, 89.9, 90.1, 100.0, 135.0, 179.0, 180.0)


@pytest.mark.parametrize("gate", ACS_GATES)
@pytest.mark.parametrize("action_reduce", ACS_ACTION_REDUCTIONS)
def test_gate_over_an_explicit_angle_sweep(gate, action_reduce):
    """The three clauses read off one deterministic sweep from parallel to antiparallel.

    Each consecutive action pair subtends a prescribed angle, so ``cos`` is known in closed form
    rather than measured. Asserted per gate member:

    - all four: ``0 <= w <= 1``;
    - ``relu_cos``: ``w = cos`` on the ``cos > 0`` side and exactly 0 from 90 deg onwards, plus
      ``w = 1`` at 0 deg;
    - ``hard``: ``w`` is the indicator of the positive half-space, so exactly 0 or 1 with the same
      boundary, and 1 at 0 deg;
    - ``affine_cos``: ``(1 + cos)/2``, hence 1 at 0 deg and ``0.5`` at 90 deg - the documented
      reason it is *not* the default, since the orthogonal case still carries half the baseline
      pressure;
    - ``permuted``: the weight multiset matches ``relu_cos``'s and nothing more is claimed, because
      the correspondence between a weight and its triple is exactly what it destroys.

    The angles are shared across the three reductions: the action block is one direction tiled over
    the substeps, and ``cos`` is invariant to the positive scalar that ``sum`` applies (P5).
    """
    act = angled_actions(SWEEP_ANGLES, substeps=3)
    model = build_model(gate=gate, action_reduce=action_reduce)
    relu_model = build_model(gate="relu_cos", action_reduce=action_reduce)

    expected_cos = torch.tensor(
        [math.cos(math.radians(a)) for a in SWEEP_ANGLES], dtype=torch.float32
    ).unsqueeze(0)
    cos_a = reduced_cosine(model, act, 2)
    assert cos_a.shape == expected_cos.shape == (1, len(SWEEP_ANGLES))
    assert torch.allclose(cos_a, expected_cos, atol=1e-6, rtol=0.0), (
        f"reduce={action_reduce!r}: the sweep's cosines are not the prescribed angles"
    )

    w = gate_for(model, act, None, 2, seed=0)
    assert bool((w >= 0.0).all()) and bool((w <= 1.0).all()), (
        f"gate={gate!r}: the sweep left the unit interval, w={w.tolist()!r}"
    )

    reversing = expected_cos <= 0.0
    if gate == "relu_cos":
        assert torch.equal(w[reversing], torch.zeros_like(w[reversing]))
        assert torch.allclose(w[~reversing], expected_cos[~reversing], atol=1e-6, rtol=0.0), (
            "relu_cos must pass the positive cosine through ungraded-by-anything-else"
        )
    elif gate == "hard":
        assert torch.equal(w, (expected_cos > 0.0).to(w.dtype)), (
            "the hard gate must be the indicator of the positive half-space"
        )
    elif gate == "affine_cos":
        assert torch.allclose(w, (1.0 + expected_cos) * 0.5, atol=1e-6, rtol=0.0)
    if gate == "permuted":
        plain = gate_for(relu_model, act, None, 2, seed=0)
        assert torch.equal(torch.sort(w.reshape(-1)).values, torch.sort(plain.reshape(-1)).values), (
            "the permuted gate must preserve relu_cos's weight multiset exactly"
        )
    else:
        # The parallel anchor: the first pair subtends 0 deg. Excluded for ``permuted``, whose
        # weights no longer belong to their own triple.
        assert abs(float(w[0, 0]) - 1.0) <= PARALLEL_ATOL, (
            f"gate={gate!r}: the parallel pair did not get w = 1, got {float(w[0, 0])!r}"
        )


@pytest.mark.parametrize("action_reduce", ACS_ACTION_REDUCTIONS)
def test_gate_at_exact_orthogonality(action_reduce):
    """The half-space boundary itself, built from axis-aligned vectors so ``cos`` is exactly 0.

    ``cos <= 0`` includes the boundary (Requirement 5.6), so orthogonal controls get **zero**
    pressure under ``relu_cos`` - the gate's ``clamp`` then ``relu`` sends `0.0` to `0.0`, not to a
    denormal. ``hard`` agrees (``1[0 > 0] == 0``); ``affine_cos`` returns exactly ``0.5``, which is
    the concrete form of the objection design section 4.2 raises against it: the softer gate leaves
    half the baseline pressure on a transition whose control turned by a right angle.

    This is the same value a zero-norm action block produces through ``cosine_similarity``'s ``eps``
    (E10), so the two degenerate inputs land on one assertion.

    **Validates: Requirements 5.4, 5.6**
    """
    act = direction_actions([(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)], substeps=4)

    cos_a = reduced_cosine(build_model(gate="relu_cos", action_reduce=action_reduce), act, 2)
    assert torch.equal(cos_a, torch.zeros_like(cos_a)), (
        f"reduce={action_reduce!r}: perpendicular controls did not give cos = 0 exactly, "
        f"got {cos_a.tolist()!r}"
    )

    expected = {"relu_cos": 0.0, "hard": 0.0, "affine_cos": 0.5}
    for gate, value in expected.items():
        model = build_model(gate=gate, action_reduce=action_reduce)
        w = gate_for(model, act, None, 2)
        assert torch.equal(w, torch.full_like(w, value)), (
            f"gate={gate!r}, reduce={action_reduce!r}: orthogonal controls gave {w.tolist()!r}, "
            f"expected {value}"
        )


def test_permuted_gate_can_move_weight_off_its_own_triple():
    """Why clause 3 is restricted to ``relu_cos``, demonstrated rather than asserted by omission.

    The ``permuted`` member is the attribution null control (Requirement 13.4): it keeps the weight
    population - hence ``mean(w)``, the quantiles and ``gate_tv`` - and destroys only which triple
    each weight lands on. So a reversing triple *can* receive positive weight and a parallel one
    *can* receive 0, and the ``cos <= 0`` zero-set clause must not be asserted for it. If that ever
    stopped happening, the arm would have quietly become a second copy of ACS and would answer
    nothing.

    The permutation is over the whole tensor here (``mask=None``); the masked variant is exercised
    by the property tests above, where a partially static batch supplies a real mask.
    """
    act = angled_actions((0.0, 180.0, 0.0, 180.0, 0.0, 180.0, 0.0, 180.0), substeps=2)
    model = build_model(gate="permuted", action_reduce="sum")
    relu_model = build_model(gate="relu_cos", action_reduce="sum")

    plain = relu_model.action_gate(act, mask=None, env_action_dim=2).reshape(-1)
    assert bool((plain == 0).any()) and bool((plain > 0).any()), (
        "the sweep must contain both reversing and non-reversing triples for this to say anything"
    )

    moved = False
    for seed in range(64):
        w = gate_for(model, act, None, 2, seed=seed).reshape(-1)
        assert torch.equal(torch.sort(w).values, torch.sort(plain).values), (
            f"seed={seed}: the permuted gate changed the weight multiset"
        )
        assert bool((w >= 0).all()) and bool((w <= 1).all())
        if bool(((plain == 0) & (w > 0)).any()):
            moved = True
            break
    assert moved, (
        "no seed moved a positive weight onto a reversing triple; the permuted arm would then be "
        "indistinguishable from relu_cos and could not attribute anything"
    )


def test_permuted_gate_leaves_masked_entries_alone():
    """The permutation is over the **unmasked** triples, which is what makes the arm comparable.

    ``_permute_gate`` gathers the unmasked entries, shuffles them among themselves and scatters
    them back; masked entries keep their own values, because the weighted mean drops them by the
    same mask and their positions must not absorb weight belonging to a live triple.

    **Validates: Requirements 13.4**
    """
    act = angled_actions((0.0, 180.0, 45.0, 135.0, 0.0, 90.0), substeps=2)
    model = build_model(gate="permuted", action_reduce="sum")
    relu_model = build_model(gate="relu_cos", action_reduce="sum")
    plain = relu_model.action_gate(act, mask=None, env_action_dim=2)

    mask = torch.ones_like(plain, dtype=torch.bool)
    mask[0, 1] = False
    mask[0, -1] = False

    for seed in range(8):
        w = gate_for(model, act, mask, 2, seed=seed)
        assert torch.equal(w[~mask], plain[~mask]), (
            f"seed={seed}: a masked triple's weight was overwritten by the permutation"
        )
        assert torch.equal(torch.sort(w[mask]).values, torch.sort(plain[mask]).values), (
            f"seed={seed}: the unmasked weight multiset was not preserved"
        )
        assert bool((w >= 0).all()) and bool((w <= 1).all())


# ---------------------------------------------------------------------------
# Worked examples at the PushT target cell
# ---------------------------------------------------------------------------


def target_cell_case(gate: str, *, action_reduce: str = "sum", kind: str = "generic"):
    """The Target_Cell's shapes: ``b=32``, ``t=4``, ``196x8`` features, ``act`` ``(32, 4, 10)``.

    ``f = 5`` (frameskip) and ``d = 2`` (PushT's env action dim), so every latent step's 10
    channels split into 5 two-dimensional substeps and the batch carries 64 weights.
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
def test_target_cell_gate_range_shape_and_parallel_identity(gate):
    """Property 3 at the Target_Cell's shapes, for each of the four gate members.

    ``(32, 4, 196, 8)`` features into the 32-d aggregated space give 2 curvature triples per sample
    and 64 weights per batch, at the configuration the recorded 75.33 OL / 82.00 MPC baseline was
    trained in. Both the ordinary batch (range and shape) and the all-parallel batch (``w = 1``)
    are read at those shapes, so the property is anchored to the cell rather than only to the small
    generated ones.
    """
    model, acs = target_cell_case(gate)
    _, mask = curvature_terms(model, acs.z)
    w = gate_for(model, acs.act, mask, acs.env_action_dim, seed=7)

    assert tuple(w.shape) == (ACS_TARGET_CELL_BATCH, ACS_TARGET_CELL_TRIPLES_PER_SAMPLE)
    assert w.numel() == ACS_TARGET_CELL_TRIPLES == 64
    assert w.shape == mask.shape
    assert torch.isfinite(w).all()
    assert bool((w >= 0.0).all()) and bool((w <= 1.0).all()), (
        f"gate={gate!r}: target-cell weights left the unit interval, "
        f"[{float(w.min())!r}, {float(w.max())!r}]"
    )

    par_model, par = target_cell_case(gate, kind="parallel")
    _, par_mask = curvature_terms(par_model, par.z)
    par_w = gate_for(par_model, par.act, par_mask, par.env_action_dim, seed=7)
    assert bool((par_w <= 1.0).all())
    assert torch.allclose(par_w, torch.ones_like(par_w), atol=PARALLEL_ATOL, rtol=0.0), (
        f"gate={gate!r}: target-cell parallel actions gave min(w)={float(par_w.min())!r}"
    )


def test_target_cell_relu_cos_zero_set_matches_the_half_space():
    """The default gate's zero set at the target cell, counted rather than sampled.

    64 weights, and the set ``{w = 0}`` is exactly ``{cos <= 0}`` - the fraction is what the Stage-0
    probe reports as ``frac(w = 0)`` and what the ``acs_gate_zero_frac`` telemetry tracks during
    training, so it is worth reading once at the shipped shapes.
    """
    model, acs = target_cell_case("relu_cos")
    cos_a = reduced_cosine(model, acs.act, acs.env_action_dim)
    w = gate_for(model, acs.act, None, acs.env_action_dim)

    zero = w == 0.0
    reversing = cos_a <= 0.0
    assert torch.equal(zero, reversing), (
        "the relu_cos zero set is not the reversing half-space: "
        f"{int(zero.sum())} zeros against {int(reversing.sum())} reversing triples"
    )
    assert 0 < int(reversing.sum()) < w.numel(), (
        "this fixture is only informative when the batch contains both kinds of triple"
    )
