"""Task 2.3 - Property 1: the ACS-disabled path is bitwise the baseline.

This module is a **gate**, deliberately not optional. It is the reason the measured PushT
baseline - 75.33 +/- 6.11 OL, 82.00 +/- 2.00 MPC at
``pusht_aggmlpcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05`` - stands as the control for
the ACS arm without a 12 h retrain. If any byte of the ``straighten="aggcos1e-1"`` objective
moved when sections 1 and 2 landed, that number describes a model the repository can no longer
build, and every downstream comparison in this feature is against a control that does not exist.

What is compared, and against what
----------------------------------

The reference is ``tests/reference_impl.py``, not a re-derivation:

- :func:`tests.reference_impl.reference_forward_loss` - the frozen ``VWorldModel.forward`` at
  ``d73b9c6``, run here with its straightening tail suppressed (see :class:`_PreFeatureModel`).
- :func:`tests.reference_impl.reference_curvature_forward_tail` - the frozen straightening tail
  at ``d3c3ce5``, which is the tail that writes **both** ``curvature_loss_used_for_training``
  and ``curvature_loss_scaled``. The ``d73b9c6`` copy predates the scaled key, so composing the
  two snapshots this way is what the second snapshot's own header instructs.
- :func:`tests.reference_impl.reference_total_curvature` and
  :func:`tests.reference_impl.reference_cos_curvature` - the frozen curvature path, taken
  *before* task 2.1 split it into ``_agg_velocities`` / ``_cos_curvature_terms``. That split is
  the only reason this property is more than a tautology: the frozen copies call neither helper.

Three deliberate choices
------------------------

1. **Bitwise, not ``allclose``.** The refactor performs the identical operations in the
   identical order on identically seeded weights, so anything short of bit-for-bit equality
   means the op sequence changed. ``allclose`` would pass a reordered reduction, and a
   reordered reduction is exactly what a 12 h control cannot absorb.
2. **A bit-pattern fallback next to ``torch.equal``.** ``torch.equal`` reports ``nan != nan``.
   A fully static batch masks out every triple, so ``loss[mask].mean()`` is a legitimate
   ``nan`` on both paths; comparing those with ``torch.equal`` alone would fail on agreement
   and, worse, would silently accept one ``nan`` in place of a different one. Where
   ``torch.equal`` says no, the raw float32 bytes are compared instead (see
   :func:`assert_bitwise`).
3. **Two identically seeded models, not one model called twice.** The reference path and the
   current path each get their own ``VWorldModel``, weight-checked for equality first, so a
   mismatch cannot be blamed on shared mutable state.

Deliberately out of scope: nothing here calls ``forward`` with
``straighten="acsaggcos1e-1"``. The ACS loss term does not exist until task 6.1, and until it
does that mode string raises from ``total_curvature`` - which is the correct behaviour for a
half-built feature, not a defect this module should paper over.

**Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5, 7.7, 7.9**
"""

from __future__ import annotations

import itertools

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.conftest import (
    ACS_ACTION_REDUCTIONS,
    ACS_GATES,
    agg_type_strategy,
    batch_size_strategy,
    build_stub_world_model,
    concat_dim_strategy,
    make_acs_case,
    make_stub_batch,
    num_hist_strategy,
)
from tests.reference_impl import (
    reference_cos_curvature,
    reference_curvature_forward_tail,
    reference_forward_loss,
    reference_total_curvature,
)

# Minimum 100 examples per the feature's testing convention, without capping the
# ``ccr-thorough`` profile back down to 100. ``deadline=None``: a stub forward is fast but a
# CPU-only box under load is not a wall clock.
MIN_EXAMPLES = max(100, settings.default.max_examples or 100)
acs_settings = settings(max_examples=MIN_EXAMPLES, deadline=None)

#: The two ``straighten`` values Property 1 covers. ``"aggcos1e-1"`` is the Baseline_Arm's own
#: setting (Requirements 7.1, 7.2); ``False`` is the shipped default (Requirement 7.9).
DISABLED_STRAIGHTEN_VALUES = ("aggcos1e-1", False)

#: Every ACS knob combination that must leave the disabled path alone: not passed at all,
#: passed as ``None`` (what ``self.cfg.training.get(key)`` yields for an absent yaml key), and
#: every member of the two closed enums - including the non-defaults, since a non-default knob
#: on a non-ACS mode is exactly the "typo in an unused knob" case the parser now validates
#: eagerly and must still not perturb the loss.
ACS_KWARG_CHOICES = (
    None,
    (None, None),
) + tuple(itertools.product(ACS_ACTION_REDUCTIONS, ACS_GATES))

#: Keys no non-ACS forward may emit (Requirements 7.3, 7.4).
FORBIDDEN_KEY_PREFIX = "acs_"
FORBIDDEN_KEY = "curvature_loss_unweighted"

# The stub encoder's grid: ``visual_only(z)`` is ``(b, t, num_patches, emb_dim)`` under both
# ``concat_dim`` branches, so aggregated-space features are 4 patches x 4 channels and the
# ``agg_type="mlp"`` head's input width (num_patches * emb_dim) matches by construction.
STUB_PATCHES = 4
STUB_CHANNELS = 4

# Which degenerate constructions the curvature-refactor properties draw from. ``static`` makes
# every aggregated velocity exactly zero, so the whole batch is masked and both paths return
# ``nan``; ``one_moving`` leaves a single unmasked sample, the partially-masked case.
CURVATURE_CASE_KINDS = ("generic", "static", "one_moving")


# ---------------------------------------------------------------------------
# Bitwise comparison
# ---------------------------------------------------------------------------


def raw_bytes(tensor: torch.Tensor) -> bytes:
    """The tensor's exact bit pattern, ``nan`` payloads included."""
    return tensor.detach().cpu().contiguous().numpy().tobytes()


def assert_bitwise(actual: torch.Tensor, expected: torch.Tensor, what: str) -> None:
    """Assert ``actual`` and ``expected`` agree in dtype, shape and every bit.

    ``torch.equal`` first, because that is the comparison this feature's tests are specified
    in. Where it says no, the raw bytes decide: a fully masked curvature batch is ``nan`` on
    both paths, and ``nan != nan`` under ``torch.equal``.
    """
    assert isinstance(actual, torch.Tensor), f"{what}: expected a tensor, got {type(actual)!r}"
    assert actual.dtype == expected.dtype, (
        f"{what}: dtype moved, {actual.dtype} vs {expected.dtype}"
    )
    assert actual.shape == expected.shape, (
        f"{what}: shape moved, {tuple(actual.shape)} vs {tuple(expected.shape)}"
    )
    if torch.equal(actual, expected):
        return
    assert raw_bytes(actual) == raw_bytes(expected), (
        f"{what}: not BITWISE equal to the frozen pre-feature reference. "
        f"got {actual.detach().cpu().tolist()!r}, "
        f"reference {expected.detach().cpu().tolist()!r}"
    )


# ---------------------------------------------------------------------------
# The pre-feature forward
# ---------------------------------------------------------------------------


class _PreFeatureModel:
    """Proxy that runs :func:`reference_forward_loss` with its straightening tail suppressed.

    Two reasons this exists rather than a straight call:

    - The ``d73b9c6`` snapshot's tail writes only ``curvature_loss_used_for_training`` and
      routes the curvature through ``model.total_curvature`` - i.e. through the *current*,
      refactored method, which would make the comparison circular. Suppressing the tail here
      and applying :func:`reference_curvature_forward_tail` afterwards routes the curvature
      through the frozen ``d3c3ce5`` copies and produces both keys.
    - The tail needs ``z``, which ``reference_forward_loss`` computes internally and does not
      return. The proxy records it instead of the test recomputing ``encode``.

    Every other attribute and method resolves to the wrapped model, so the frozen forward is
    reading the real thing.
    """

    def __init__(self, model):
        self._model = model
        self.straighten = False  # suppress the d73b9c6 tail; the d3c3ce5 tail is applied below
        self.z = None

    def __getattr__(self, name):
        return getattr(self._model, name)

    def encode(self, obs, act):
        self.z = self._model.encode(obs, act)
        return self.z


def pre_feature_forward(model, obs, act):
    """``forward`` as it behaved before this feature, assembled from the frozen snapshots.

    Returns the same 5-tuple as ``VWorldModel.forward``. The arithmetic order matches the
    current implementation's - ``loss = 0``, ``+ z_loss``, then ``+ curvature * scale`` - and
    the keys are inserted in the same order, so ``loss_components`` can be compared as an
    ordered mapping and not just as a set.
    """
    proxy = _PreFeatureModel(model)
    z_pred, visual_pred, visual_reconstructed, loss, loss_components = reference_forward_loss(
        proxy, obs, act
    )
    assert "curvature_loss_used_for_training" not in loss_components, (
        "the proxy failed to suppress the d73b9c6 straightening tail, so the reference would "
        "have counted the curvature term twice"
    )
    del loss_components["loss"]  # re-inserted last, exactly where forward writes it
    loss = reference_curvature_forward_tail(model, proxy.z, loss, loss_components)
    loss_components["loss"] = loss
    return z_pred, visual_pred, visual_reconstructed, loss, loss_components


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------


def _model_kwargs(case, *, with_acs_kwargs: bool = True) -> dict:
    kwargs = dict(
        num_hist=case["num_hist"],
        num_pred=1,
        concat_dim=case["concat_dim"],
        agg_type=case["agg_type"],
        straighten=case["straighten"],
        seed=case["seed"],
        image_size=8,
    )
    if with_acs_kwargs and case["acs_kwargs"] is not None:
        action_reduce, gate = case["acs_kwargs"]
        kwargs["acs_action_reduce"] = action_reduce
        kwargs["acs_gate"] = gate
    return kwargs


def build_pair(case):
    """Two identically seeded models: one for the frozen reference, one for the current path.

    The reference model is built **without** the ACS ctor kwargs and the current one **with**
    them, so the pair also pins that the knobs change nothing about the weights they share.
    """
    reference_model = build_stub_world_model(**_model_kwargs(case, with_acs_kwargs=False))
    current_model = build_stub_world_model(**_model_kwargs(case, with_acs_kwargs=True))
    reference_model.eval()
    current_model.eval()

    ref_state = reference_model.state_dict()
    cur_state = current_model.state_dict()
    assert ref_state.keys() == cur_state.keys(), (
        "the ACS ctor kwargs changed the state dict: "
        f"{sorted(set(cur_state) ^ set(ref_state))}"
    )
    for key in ref_state:
        assert torch.equal(ref_state[key], cur_state[key]), (
            f"the two stub models are not identically seeded: weight '{key}' differs, so a "
            "downstream mismatch would say nothing about the refactor"
        )
    return reference_model, current_model


@st.composite
def forward_cases(draw):
    """Shapes, model variants, ``straighten`` value and ACS knobs, drawn jointly.

    ``num_pred`` is pinned to 1 and ``num_frames`` to ``num_hist + 1``, which is what
    ``forward`` requires for its source and target windows to line up, and which puts
    ``num_frames`` at 3 or 4 - at or above the three frames a curvature triple needs.
    """
    return {
        "batch_size": draw(batch_size_strategy),
        "num_hist": draw(num_hist_strategy),
        "concat_dim": draw(concat_dim_strategy),
        "agg_type": draw(agg_type_strategy),
        "straighten": draw(st.sampled_from(DISABLED_STRAIGHTEN_VALUES)),
        "acs_kwargs": draw(st.sampled_from(ACS_KWARG_CHOICES)),
        "seed": draw(st.integers(min_value=0, max_value=3)),
    }


@st.composite
def curvature_cases(draw):
    """A model variant plus an aggregated-space feature tensor of the shape it consumes."""
    case = draw(forward_cases())
    case["kind"] = draw(st.sampled_from(CURVATURE_CASE_KINDS))
    case["num_frames"] = draw(st.integers(min_value=3, max_value=6))
    case["feature_seed"] = draw(st.integers(min_value=0, max_value=2**31 - 1))
    return case


def features_for(case) -> torch.Tensor:
    """``(b, t, 4, 4)`` visual-only features, ordinary or degenerate per ``case["kind"]``."""
    return make_acs_case(
        kind=case["kind"],
        batch_size=case["batch_size"],
        num_frames=case["num_frames"],
        patches=STUB_PATCHES,
        channels=STUB_CHANNELS,
        moving_index=0,
        seed=case["feature_seed"],
    ).z


# ---------------------------------------------------------------------------
# Property 1
# ---------------------------------------------------------------------------


@given(case=forward_cases())
@acs_settings
def test_property_1_disabled_forward_is_bitwise_the_baseline(case):
    """Feature: action-conditioned-straightening, Property 1: The disabled path is bitwise the baseline.

    For any observation window and action sequence, with ``straighten="aggcos1e-1"`` and with
    the default ``straighten=False``, ``loss`` and every ``loss_components`` value are bitwise
    equal to a reference built from the pre-feature code path
    (``tests/reference_impl.py``), and no key with the prefix ``acs_`` and no
    ``curvature_loss_unweighted`` key exists. Holds for every value of the two ACS ctor knobs,
    including the non-defaults and the ``None`` an absent yaml key produces.

    **Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.9**
    """
    reference_model, current_model = build_pair(case)
    obs, act = make_stub_batch(
        current_model,
        batch_size=case["batch_size"],
        num_frames=case["num_hist"] + 1,
        image_size=8,
        seed=case["seed"],
    )

    # Independent input copies per path: an accidental in-place write on one must not be able
    # to feed the other the mutated tensor and hide itself.
    ref_obs = {key: value.clone() for key, value in obs.items()}
    cur_obs = {key: value.clone() for key, value in obs.items()}

    with torch.no_grad():
        _, _, _, ref_loss, ref_components = pre_feature_forward(
            reference_model, ref_obs, act.clone()
        )
        _, _, _, cur_loss, cur_components = current_model(cur_obs, act.clone())

    assert_bitwise(cur_loss, ref_loss, f"loss (straighten={case['straighten']!r})")

    # Same keys, in the same order: ACS must not have inserted, dropped or moved one.
    assert list(cur_components) == list(ref_components), (
        "loss_components keys moved on the disabled path: "
        f"got {list(cur_components)}, reference {list(ref_components)}"
    )
    for key, expected in ref_components.items():
        assert_bitwise(cur_components[key], expected, f"loss_components['{key}']")

    # Requirements 7.3 / 7.4: not one ACS diagnostic on a non-ACS mode.
    offending = [key for key in cur_components if key.startswith(FORBIDDEN_KEY_PREFIX)]
    assert offending == [], (
        f"curvature_mode={current_model.curvature_mode!r} emitted ACS keys {offending}"
    )
    assert FORBIDDEN_KEY not in cur_components, (
        f"curvature_mode={current_model.curvature_mode!r} emitted '{FORBIDDEN_KEY}', which is "
        "an ACS-only diagnostic and would break the term-share sum for the control arm"
    )

    # The straightening keys are present exactly when the baseline arm's mode is selected.
    expects_curvature = case["straighten"] is not False
    assert ("curvature_loss_used_for_training" in cur_components) is expects_curvature
    assert ("curvature_loss_scaled" in cur_components) is expects_curvature
    assert current_model.curvature_mode == ("aggcos" if expects_curvature else None)


@given(case=curvature_cases())
@acs_settings
def test_property_1_cos_curvature_refactor_is_bitwise(case):
    """Feature: action-conditioned-straightening, Property 1: The disabled path is bitwise the baseline.

    The ``_cos_curvature_terms`` half of the section-2.1 refactor: for any pair of velocity
    fields, ``VWorldModel._cos_curvature`` returns a value bitwise equal to the frozen
    pre-refactor ``_cos_curvature``. Same ops, same order, same dtypes - including the fully
    static case, where every triple is masked out and both paths return ``nan``.

    **Validates: Requirements 7.5**
    """
    reference_model, current_model = build_pair(case)

    flat = features_for(case).reshape(case["batch_size"], case["num_frames"], -1)
    v1 = flat[:, 1:-1] - flat[:, :-2]
    v2 = flat[:, 2:] - flat[:, 1:-1]

    with torch.no_grad():
        expected = reference_cos_curvature(reference_model, v1.clone(), v2.clone())
        actual = current_model._cos_curvature(v1.clone(), v2.clone())

    assert_bitwise(actual, expected, f"_cos_curvature (kind={case['kind']})")


@given(case=curvature_cases(), mode=st.sampled_from(("cos", "aggcos")))
@acs_settings
def test_property_1_total_curvature_refactor_is_bitwise(case, mode):
    """Feature: action-conditioned-straightening, Property 1: The disabled path is bitwise the baseline.

    The ``_agg_velocities`` half of the section-2.1 refactor: for any visual-only feature
    tensor and both curvature modes, ``VWorldModel.total_curvature`` returns a value bitwise
    equal to the frozen pre-refactor ``total_curvature``. ``aggcos`` is the mode the
    Baseline_Arm trains under and the mode the refactor actually touched; ``cos`` is carried
    along because the parser still accepts it.

    **Validates: Requirements 7.5**
    """
    reference_model, current_model = build_pair(case)
    features = features_for(case)

    with torch.no_grad():
        expected = reference_total_curvature(reference_model, features.clone(), mode=mode)
        actual = current_model.total_curvature(features.clone(), mode=mode)

    assert_bitwise(actual, expected, f"total_curvature(mode={mode!r}, kind={case['kind']})")


@given(case=forward_cases())
@acs_settings
def test_property_1_acs_kwargs_add_no_module_parameter_or_buffer(case):
    """Feature: action-conditioned-straightening, Property 1: The disabled path is bitwise the baseline.

    The structural half of the property: a model built **with** the ACS ctor kwargs has the
    same ``named_modules``, ``named_parameters``, ``named_buffers`` and ``state_dict`` as one
    built **without** them at all. ``VWorldModel`` is constructed after
    ``accelerator.prepare()`` and is never itself prepared, so a parameter or buffer created
    here would never be moved, wrapped or synchronised - which is why the knobs are plain
    Python strings and why that is asserted rather than assumed.

    **Validates: Requirements 7.7**
    """
    plain = build_stub_world_model(**_model_kwargs(case, with_acs_kwargs=False))
    configured = build_stub_world_model(**_model_kwargs(case, with_acs_kwargs=True))

    def modules(model):
        return [(name, type(module).__name__) for name, module in model.named_modules()]

    def tensors(named):
        return [(name, tuple(t.shape), str(t.dtype)) for name, t in named]

    assert modules(configured) == modules(plain), (
        "the ACS ctor kwargs created a module: "
        f"{sorted(set(modules(configured)) ^ set(modules(plain)))}"
    )
    assert tensors(configured.named_parameters()) == tensors(plain.named_parameters()), (
        "the ACS ctor kwargs created a parameter"
    )
    assert tensors(configured.named_buffers()) == tensors(plain.named_buffers()), (
        "the ACS ctor kwargs created a buffer"
    )
    assert list(configured.state_dict()) == list(plain.state_dict())

    # The mechanism behind the counts above: plain strings, not tensors or modules.
    for attribute in ("acs_action_reduce", "acs_gate"):
        value = getattr(configured, attribute)
        assert isinstance(value, str), f"{attribute} is {type(value).__name__}, expected str"
        assert not isinstance(value, (torch.Tensor, torch.nn.Module))

    # Defaults resolve to themselves, and an absent yaml key (None) resolves to the default.
    if case["acs_kwargs"] in (None, (None, None)):
        assert configured.acs_action_reduce == "sum"
        assert configured.acs_gate == "relu_cos"
    else:
        action_reduce, gate = case["acs_kwargs"]
        assert configured.acs_action_reduce == action_reduce
        assert configured.acs_gate == gate


# ---------------------------------------------------------------------------
# Worked examples alongside the property
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("straighten", DISABLED_STRAIGHTEN_VALUES)
def test_target_cell_shaped_forward_is_bitwise_the_baseline(straighten):
    """The Target_Cell's model variant: ``num_hist=3``, ``concat_dim=1``, ``agg_type=mlp``.

    A worked example beside the property, at the configuration the recorded 75.33 / 82.00 run
    was trained in, with both ACS knobs pushed off their defaults.
    """
    case = {
        "batch_size": 2,
        "num_hist": 3,
        "concat_dim": 1,
        "agg_type": "mlp",
        "straighten": straighten,
        "acs_kwargs": ("raw", "hard"),
        "seed": 0,
    }
    reference_model, current_model = build_pair(case)
    obs, act = make_stub_batch(current_model, batch_size=2, num_frames=4, image_size=8, seed=0)

    with torch.no_grad():
        _, _, _, ref_loss, ref_components = pre_feature_forward(
            reference_model, {k: v.clone() for k, v in obs.items()}, act.clone()
        )
        _, _, _, cur_loss, cur_components = current_model(
            {k: v.clone() for k, v in obs.items()}, act.clone()
        )

    assert_bitwise(cur_loss, ref_loss, f"loss (straighten={straighten!r})")
    assert list(cur_components) == list(ref_components)
    for key, expected in ref_components.items():
        assert_bitwise(cur_components[key], expected, f"loss_components['{key}']")
    assert not any(key.startswith(FORBIDDEN_KEY_PREFIX) for key in cur_components)
    assert FORBIDDEN_KEY not in cur_components


def test_why_a_bit_pattern_fallback_and_not_torch_equal_alone():
    """Pin the fact :func:`assert_bitwise` rests on, so its point cannot be lost.

    A fully static batch masks out every curvature triple, so the baseline path returns
    ``nan`` - and ``torch.equal(nan, nan)`` is ``False``. Without the byte fallback the
    static case would fail on agreement; without the comparison being bitwise at all, one
    ``nan`` could be swapped for another unnoticed.
    """
    nan = torch.tensor(float("nan"), dtype=torch.float32)
    assert not torch.equal(nan, nan)
    assert raw_bytes(nan) == raw_bytes(nan.clone())

    model = build_stub_world_model(straighten="aggcos1e-1", image_size=8, seed=0)
    static = make_acs_case(
        kind="static", batch_size=2, num_frames=4, patches=STUB_PATCHES, channels=STUB_CHANNELS
    ).z
    with torch.no_grad():
        value = model.total_curvature(static, mode="aggcos")
    assert torch.isnan(value), (
        "a fully static batch is expected to mask out every triple and average an empty "
        "selection; if that ever changes, the fallback in assert_bitwise is no longer needed "
        "for this case and the reason for it should be re-derived rather than deleted"
    )
