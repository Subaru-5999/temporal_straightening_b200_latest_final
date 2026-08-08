"""MCA rung 1 - the `_mca_terms` extraction, the probe's statistics and the gate.

`PROGRESS_MCA.md` section 4 is a **pre-registration**: section 4.3's thresholds were
written on 2026-08-08, before the offline probe was run. This module is what makes that
pre-registration mean something operationally. It pins four things:

1. **The extraction is bitwise neutral.** `VWorldModel.compute_mca` after the
   `_mca_terms` split is compared against
   :func:`tests.reference_impl.reference_compute_mca`, the verbatim frozen copy of the
   pre-refactor body at commit ``6a5741c``, **bitwise** rather than with `allclose`, over
   the shared stub fixtures and every ``agg_type`` in
   :data:`tests.conftest.AGG_TYPES`. Section 4.1 requires the split to be neutral; the
   `mca_weight` knob has ridden along at 0 in every CCR pilot, so a change here would
   silently redefine a term that is about to be switched on.

2. **The probe and the training penalty are the same number.**
   `probe_ccr_curvature.mca_reduction(r)` on the `r` that `_mca_terms` returns is
   **bitwise** equal to `compute_mca(z)` on the same tensor. This is the structural fix
   for the CCR calibration error, where the probe's statistic and the trained penalty
   were two implementations of "the same" quantity and were not the same number
   (`PROGRESS_CCR.md` sections 5a, 6a; `PROGRESS_MCA.md` section 4.1).

3. **The gate is total and its boundaries are exact.** `rung1_verdict` is a pure
   function of one float, so both thresholds are tested *at*, *just below* and *just
   above* the value via `math.nextafter` - one ULP away, so nothing can sit between the
   fixture and the threshold - plus the `rho > 0` branch, which section 4.3 requires to
   be a STOP with a **distinct** reason because it contradicts the LayerNorm argument of
   section 4.2. Mirrors `tests/test_acs_stage0_verdict.py`'s structure, including the
   clause-to-verdict map that makes "exactly one verdict, no overlap" mechanical.

4. **Average-rank tie handling.** The Spearman implementation is checked against
   hand-computed values on small tied examples, and against the no-tie shortcut
   `1 - 6*sum(d^2)/(n^3 - n)` which it must *disagree* with on tied data. Tie handling
   is the easy thing to get wrong in a hand-rolled rank correlation, and getting it
   wrong biases rho toward zero exactly where the data is discrete - i.e. it would move
   the gate.

Everything here is CPU float32 (or float64 numpy) against the tiny stub encoder: no
DINOv2 download, no GPU, no dataset, no `DATASET_DIR`. The verdict tests touch neither
torch nor numpy.

**Validates: PROGRESS_MCA.md sections 4.1, 4.2, 4.3, 4.4, 4.5**
"""

from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.conftest import (
    AGG_TYPES,
    agg_type_strategy,
    batch_size_strategy,
    build_stub_world_model,
    concat_dim_strategy,
    make_stub_batch,
    num_frames_strategy,
    num_hist_strategy,
)
from tests.reference_impl import reference_compute_mca

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from probe_ccr_curvature import (  # noqa: E402
    MCA_ADVISORY_MIN_PAIRS,
    MCA_CHECK_A_NOT_GATING_REASON,
    MCA_CV_RELATIVE_TOLERANCE,
    MCA_DECILES,
    MCA_EPS,
    MCA_HIST_BINS,
    MCA_HIST_RANGE,
    READOUTS,
    STATE_DIM_NAMES,
    Budget,
    RUNG1_RHO_GO,
    RUNG1_RHO_MIDDLE,
    VERDICT_GO,
    VERDICT_MIDDLE,
    VERDICT_STOP,
    VERDICTS,
    _average_ranks,
    build_aggmetric_report,
    check_a_block,
    cv_of_r,
    mca_decile_table,
    mca_measure,
    mca_ratio_histogram,
    mca_reduction,
    mca_statistics,
    print_aggmetric_report,
    rank_disagreement_rate,
    rho_standard_error,
    rung1_verdict,
    spearman_rho,
)

# Minimum 100 examples per the feature's testing convention, without capping the
# `ccr-thorough` profile back down to 100.
MIN_EXAMPLES = max(100, settings.default.max_examples or 100)
mca_settings = settings(max_examples=MIN_EXAMPLES, deadline=None)

#: Clause -> verdict for `rung1_verdict`, straight out of `PROGRESS_MCA.md` section 4.3.
#: The map is what makes "exactly one verdict" mechanical: a clause that could produce
#: two verdicts, or a verdict reached through a clause that does not name it, fails the
#: totality test. `4.3-stop-positive` is a separate clause from `4.3-stop` because
#: section 4.3 requires the positive-rho STOP to carry a DISTINCT recorded reason.
CLAUSE_VERDICT_RUNG1 = {"4.3-go": VERDICT_GO,
                        "4.3-middle": VERDICT_MIDDLE,
                        "4.3-stop": VERDICT_STOP,
                        "4.3-stop-positive": VERDICT_STOP}


# ---------------------------------------------------------------------------
# Bitwise comparison (same convention as tests/test_acs_off_bitwise.py)
# ---------------------------------------------------------------------------
def raw_bytes(tensor: torch.Tensor) -> bytes:
    """The tensor's exact bit pattern, `nan` payloads included."""
    return tensor.detach().cpu().contiguous().numpy().tobytes()


def assert_bitwise(actual: torch.Tensor, expected: torch.Tensor, what: str) -> None:
    """Assert `actual` and `expected` agree in dtype, shape and every bit.

    `torch.equal` first, because that is the comparison this feature's tests are
    specified in. Where it says no, the raw bytes decide: `nan != nan` under
    `torch.equal`, and a degenerate batch can legitimately produce `nan` on both paths.
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
        f"{what}: not BITWISE equal. got {actual.detach().cpu().tolist()!r}, "
        f"expected {expected.detach().cpu().tolist()!r}"
    )


def _encode(model, batch_size=2, num_frames=4, seed=0):
    """`z` for a freshly built stub batch, under `no_grad` (the probe's regime)."""
    obs, act = make_stub_batch(model, batch_size=batch_size, num_frames=num_frames,
                               seed=seed)
    with torch.no_grad():
        return model.encode(obs, act)


# ===========================================================================
# 1. `compute_mca` is BITWISE the frozen pre-refactor copy (section 4.1)
# ===========================================================================
@pytest.mark.parametrize("agg_type", AGG_TYPES)
@pytest.mark.parametrize("concat_dim", (0, 1))
def test_compute_mca_is_bitwise_the_frozen_reference(agg_type, concat_dim):
    """Every `agg_type` in the enum, both `concat_dim` branches.

    Two identically seeded models rather than one model called twice, so a mismatch
    cannot be blamed on shared mutable state.
    """
    kwargs = dict(agg_type=agg_type, concat_dim=concat_dim, seed=0)
    current = build_stub_world_model(**kwargs)
    frozen = build_stub_world_model(**kwargs)
    for (name, a), (_name, b) in zip(current.state_dict().items(),
                                     frozen.state_dict().items()):
        assert torch.equal(a, b), f"the two stub models disagree on {name}"

    z = _encode(current)
    with torch.no_grad():
        assert_bitwise(current.compute_mca(z), reference_compute_mca(frozen, z),
                       f"compute_mca (agg_type={agg_type}, concat_dim={concat_dim})")


def test_compute_mca_is_bitwise_the_frozen_reference_at_the_target_cell(
        target_cell_model, target_cell_batch):
    """The PushT target cell's shapes: num_hist=3, num_pred=1, concat_dim=1, agg mlp."""
    obs, act = target_cell_batch
    frozen = build_stub_world_model(num_hist=3, num_pred=1, concat_dim=1,
                                    agg_type="mlp", straighten="aggcos1e-1",
                                    stop_grad=True)
    with torch.no_grad():
        z = target_cell_model.encode(obs, act)
        assert_bitwise(target_cell_model.compute_mca(z),
                       reference_compute_mca(frozen, z),
                       "compute_mca at the target cell")


@given(agg_type=agg_type_strategy, concat_dim=concat_dim_strategy,
       batch_size=batch_size_strategy, num_frames=num_frames_strategy,
       num_hist=num_hist_strategy,
       eps=st.sampled_from((MCA_EPS, 1e-8, 1e-3)))
@mca_settings
def test_compute_mca_is_bitwise_the_frozen_reference_over_shapes(
        agg_type, concat_dim, batch_size, num_frames, num_hist, eps):
    """The same equality over generated shapes and `eps` values.

    `eps` is varied because it appears **twice** in the shipped term -- in
    `v_patch + eps` and in `r_bar.clamp_min(eps)` -- and `_mca_terms(z, eps=eps)` has to
    carry it through to the first while `compute_mca` keeps it for the second. A split
    that forgot to forward it would still pass at the default.
    """
    kwargs = dict(agg_type=agg_type, concat_dim=concat_dim, num_hist=num_hist, seed=0)
    current = build_stub_world_model(**kwargs)
    frozen = build_stub_world_model(**kwargs)
    z = _encode(current, batch_size=batch_size, num_frames=num_frames)
    with torch.no_grad():
        assert_bitwise(current.compute_mca(z, eps=eps),
                       reference_compute_mca(frozen, z, eps=eps),
                       f"compute_mca (agg_type={agg_type}, eps={eps})")


# ===========================================================================
# 2. The probe's CV(r)^2 is BITWISE compute_mca's return (section 4.1)
# ===========================================================================
@pytest.mark.parametrize("agg_type", AGG_TYPES)
def test_probe_reduction_is_bitwise_compute_mca(agg_type):
    """`mca_reduction(r)` == `compute_mca(z)`, on the same tensor, bitwise.

    This is the assertion the probe itself makes on every window. If it fails, the
    rung-1 headline statistic and the training penalty have stopped being the same
    number, which is the whole thing section 4.1 exists to prevent.
    """
    model = build_stub_world_model(agg_type=agg_type)
    z = _encode(model)
    with torch.no_grad():
        r, _v_patch, _v_agg = model._mca_terms(z)
        assert_bitwise(mca_reduction(r), model.compute_mca(z),
                       f"probe CV(r)^2 vs compute_mca (agg_type={agg_type})")


@given(agg_type=agg_type_strategy, concat_dim=concat_dim_strategy,
       batch_size=batch_size_strategy, num_frames=num_frames_strategy,
       eps=st.sampled_from((MCA_EPS, 1e-8, 1e-3)))
@mca_settings
def test_probe_reduction_is_bitwise_compute_mca_over_shapes(
        agg_type, concat_dim, batch_size, num_frames, eps):
    model = build_stub_world_model(agg_type=agg_type, concat_dim=concat_dim)
    z = _encode(model, batch_size=batch_size, num_frames=num_frames)
    with torch.no_grad():
        r, _v_patch, _v_agg = model._mca_terms(z, eps=eps)
        assert_bitwise(mca_reduction(r, eps=eps), model.compute_mca(z, eps=eps),
                       f"probe CV(r)^2 vs compute_mca (eps={eps})")


def test_cv_of_r_squared_matches_compute_mca_to_tolerance():
    """The independent float64 route: `CV(r)^2 == compute_mca(z)`.

    `cv_of_r` uses the **population** variance, which is what makes the identity exact
    rather than approximate; the sample variance would be off by `n/(n-1)`. A bitwise
    check against the same op sequence cannot see a wrong definition, and this can.
    """
    model = build_stub_world_model(agg_type="mlp")
    z = _encode(model, batch_size=3, num_frames=5)
    with torch.no_grad():
        r, _v_patch, _v_agg = model._mca_terms(z)
        mca = float(model.compute_mca(z))
    cv, mean, variance = cv_of_r(r.reshape(-1).double().numpy())
    assert cv is not None and mean > 0
    assert cv * cv == pytest.approx(mca, rel=1e-6, abs=1e-12)
    assert variance / (mean * mean) == pytest.approx(mca, rel=1e-6, abs=1e-12)


# ===========================================================================
# 3. `_mca_terms` shape / domain contract
# ===========================================================================
@given(agg_type=agg_type_strategy, concat_dim=concat_dim_strategy,
       batch_size=batch_size_strategy, num_frames=num_frames_strategy)
@mca_settings
def test_mca_terms_shapes_are_b_by_t_minus_one_and_finite(
        agg_type, concat_dim, batch_size, num_frames):
    """All three terms are `(b, t - 1)`, all finite, and `r >= 0`.

    `r` is a ratio of two norms plus a positive `eps` in the denominator, so it cannot
    be negative and cannot divide by zero. Asserting it here means the probe's decile
    table, histogram and rank statistics never have to defend against a negative ratio.
    """
    model = build_stub_world_model(agg_type=agg_type, concat_dim=concat_dim)
    z = _encode(model, batch_size=batch_size, num_frames=num_frames)
    with torch.no_grad():
        r, v_patch, v_agg = model._mca_terms(z)

    expected = (batch_size, num_frames - 1)
    for name, term in (("r", r), ("v_patch", v_patch), ("v_agg", v_agg)):
        assert tuple(term.shape) == expected, f"{name}: {tuple(term.shape)} != {expected}"
        assert torch.isfinite(term).all(), f"{name} is not finite"
    assert (r >= 0).all(), "r must be non-negative"
    assert (v_patch >= 0).all() and (v_agg >= 0).all(), "norms must be non-negative"


class _NoAggEncoder(torch.nn.Module):
    """An encoder with no `agg`, which is the checkpoint MCA cannot be measured on.

    `models/dino.py` builds `agg_mlp` / `agg_post_norm` only for `agg_type == "mlp"`,
    and other encoders (`resnet`, `r3m`, `dummy`) have no `agg` at all, so this is a real
    configuration rather than a synthetic one.
    """


def test_mca_terms_rejects_an_encoder_without_agg():
    """The runtime guard, shaped like `total_curvature`'s `aggcos` check.

    It fires *before* `visual_only`, so the failure costs nothing and names the reason.
    """
    model = build_stub_world_model(agg_type="mlp")
    z = _encode(model)
    model.encoder = _NoAggEncoder()
    assert not hasattr(model.encoder, "agg")
    with pytest.raises(ValueError, match=r"requires encoder\.agg\(\)"):
        model._mca_terms(z)
    with pytest.raises(ValueError, match=r"requires encoder\.agg\(\)"):
        model.compute_mca(z)


# ===========================================================================
# 4. `rung1_verdict` - the pre-registered gate (section 4.3)
# ===========================================================================
def test_thresholds_are_the_preregistered_values():
    """`PROGRESS_MCA.md` section 4.3, written 2026-08-08, before the data.

    Refitting a threshold to the measured rho - the documented CCR failure mode - then
    fails a test instead of passing silently.
    """
    assert RUNG1_RHO_GO == -0.30
    assert RUNG1_RHO_MIDDLE == -0.10
    assert "aggmetric" in READOUTS


#: The rho grid: both thresholds exactly, one ULP either side of each, zero from both
#: sides (including `-0.0`), the ends of the domain, and points in every band.
RHO_GRID = (-1.0,
            -0.9,
            math.nextafter(RUNG1_RHO_GO, -1.0),
            RUNG1_RHO_GO,
            math.nextafter(RUNG1_RHO_GO, 0.0),
            -0.2,
            math.nextafter(RUNG1_RHO_MIDDLE, -1.0),
            RUNG1_RHO_MIDDLE,
            math.nextafter(RUNG1_RHO_MIDDLE, 0.0),
            -0.05,
            -5e-324,
            -0.0,
            0.0,
            5e-324,
            0.01,
            0.5,
            1.0)


def _reference_rung1_verdict(rho):
    """The rule transcribed from the prose of section 4.3, not from the implementation.

    | `rho <= -0.30`         | GO     |
    | `-0.30 < rho <= -0.10` | MIDDLE |
    | `-0.10 < rho <= 0`     | STOP   |
    | `rho > 0`              | STOP, distinct reason (contradicts section 4.2) |
    """
    if rho <= -0.30:
        return VERDICT_GO, "4.3-go"
    if rho <= -0.10:
        return VERDICT_MIDDLE, "4.3-middle"
    if rho <= 0.0:
        return VERDICT_STOP, "4.3-stop"
    return VERDICT_STOP, "4.3-stop-positive"


def test_rung1_verdict_is_total_over_the_rho_grid():
    """Every rho lands on exactly one verdict, through exactly one clause."""
    for rho in RHO_GRID:
        block = rung1_verdict(rho)
        expected_verdict, expected_clause = _reference_rung1_verdict(rho)
        assert block["verdict"] in VERDICTS, rho
        assert block["clause"] in CLAUSE_VERDICT_RUNG1, rho
        # No overlap: the clause determines the verdict, and only one clause fires.
        assert CLAUSE_VERDICT_RUNG1[block["clause"]] == block["verdict"], rho
        # No gap, no misrouting: the shipped branch order reproduces the prose.
        assert (block["verdict"], block["clause"]) == (expected_verdict,
                                                       expected_clause), rho
        assert block["reason"], rho
        assert block["gating"] is True, rho
        assert block["thresholds"] == {"go": RUNG1_RHO_GO, "middle": RUNG1_RHO_MIDDLE}
        assert block["rung2_permitted"] == (block["verdict"] != VERDICT_STOP), rho
        assert block["contradicts_layernorm_argument"] == (rho > 0.0), rho
        assert block["caps_applied"] == []


def test_rung1_verdict_is_total_over_a_dense_grid():
    """A dense sweep of the whole domain, so no unnamed region survives."""
    steps = 401
    for index in range(steps):
        rho = -1.0 + 2.0 * index / (steps - 1)
        block = rung1_verdict(rho)
        assert (block["verdict"], block["clause"]) == _reference_rung1_verdict(rho), rho


def test_rung1_verdict_is_monotone_in_rho():
    """A more negative rho is never a more severe verdict."""
    severity = {VERDICT_STOP: 0, VERDICT_MIDDLE: 1, VERDICT_GO: 2}
    ordered = sorted(RHO_GRID, reverse=True)          # least to most negative
    verdicts = [severity[rung1_verdict(rho)["verdict"]] for rho in ordered]
    assert verdicts == sorted(verdicts), list(zip(ordered, verdicts))


# --- the -0.30 GO boundary --------------------------------------------------
@pytest.mark.parametrize("rho, expected_verdict, expected_clause", [
    (math.nextafter(RUNG1_RHO_GO, -1.0), VERDICT_GO, "4.3-go"),      # one ULP below
    (-0.31, VERDICT_GO, "4.3-go"),
    (RUNG1_RHO_GO, VERDICT_GO, "4.3-go"),                            # exactly -0.30
    (math.nextafter(RUNG1_RHO_GO, 0.0), VERDICT_MIDDLE, "4.3-middle"),  # one ULP above
    (-0.29, VERDICT_MIDDLE, "4.3-middle"),
])
def test_rung1_go_boundary(rho, expected_verdict, expected_clause):
    """`rho <= -0.30` is inclusive: exactly -0.30 is a GO, one ULP above is a MIDDLE."""
    block = rung1_verdict(rho)
    assert (block["verdict"], block["clause"]) == (expected_verdict, expected_clause)


# --- the -0.10 MIDDLE boundary ----------------------------------------------
@pytest.mark.parametrize("rho, expected_verdict, expected_clause", [
    (math.nextafter(RUNG1_RHO_MIDDLE, -1.0), VERDICT_MIDDLE, "4.3-middle"),
    (-0.11, VERDICT_MIDDLE, "4.3-middle"),
    (RUNG1_RHO_MIDDLE, VERDICT_MIDDLE, "4.3-middle"),                # exactly -0.10
    (math.nextafter(RUNG1_RHO_MIDDLE, 0.0), VERDICT_STOP, "4.3-stop"),
    (-0.09, VERDICT_STOP, "4.3-stop"),
])
def test_rung1_middle_boundary(rho, expected_verdict, expected_clause):
    """`rho <= -0.10` is inclusive: exactly -0.10 is a MIDDLE, one ULP above a STOP."""
    block = rung1_verdict(rho)
    assert (block["verdict"], block["clause"]) == (expected_verdict, expected_clause)


def test_boundary_fixtures_really_straddle_the_thresholds():
    """Guards the fixtures themselves: the ULP neighbours are on the sides claimed."""
    assert math.nextafter(RUNG1_RHO_GO, -1.0) < RUNG1_RHO_GO
    assert math.nextafter(RUNG1_RHO_GO, 0.0) > RUNG1_RHO_GO
    assert math.nextafter(RUNG1_RHO_MIDDLE, -1.0) < RUNG1_RHO_MIDDLE
    assert math.nextafter(RUNG1_RHO_MIDDLE, 0.0) > RUNG1_RHO_MIDDLE
    # Nothing sits between the ULP neighbour and the threshold.
    assert math.nextafter(math.nextafter(RUNG1_RHO_GO, 0.0), -1.0) == RUNG1_RHO_GO
    assert math.nextafter(math.nextafter(RUNG1_RHO_MIDDLE, 0.0), -1.0) == RUNG1_RHO_MIDDLE


# --- the zero boundary and the positive branch --------------------------------
@pytest.mark.parametrize("rho, expected_clause", [
    (-5e-324, "4.3-stop"),          # the smallest negative float64
    (-0.0, "4.3-stop"),             # negative zero is still `<= 0`
    (0.0, "4.3-stop"),
    (5e-324, "4.3-stop-positive"),  # the smallest positive float64
    (1e-12, "4.3-stop-positive"),
    (0.5, "4.3-stop-positive"),
    (1.0, "4.3-stop-positive"),
])
def test_rung1_zero_boundary_and_positive_branch(rho, expected_clause):
    """`rho > 0` is a STOP with a DISTINCT clause; the sign of zero changes nothing."""
    block = rung1_verdict(rho)
    assert block["verdict"] == VERDICT_STOP
    assert block["clause"] == expected_clause


def test_positive_rho_stop_records_a_distinct_reason():
    """Section 4.3: a positive rho is "a more interesting STOP" and must say so.

    It contradicts the LayerNorm saturation argument of section 4.2, means the
    architecture is not understood, and is to be recorded as a finding rather than
    retried with another statistic. The text has to carry all three, or an operator
    reading only the verdict word would treat it as the ordinary STOP.
    """
    positive = rung1_verdict(0.2)
    ordinary = rung1_verdict(-0.05)
    assert positive["verdict"] == ordinary["verdict"] == VERDICT_STOP
    assert positive["clause"] != ordinary["clause"]
    assert positive["reason"] != ordinary["reason"]
    assert positive["contradicts_layernorm_argument"] is True
    assert ordinary["contradicts_layernorm_argument"] is False
    lowered = positive["reason"].lower()
    assert "4.2" in positive["reason"]
    assert "expand" in lowered
    assert "finding" in lowered
    assert "not retry" in lowered or "do not retry" in lowered


@pytest.mark.parametrize("bad", [None, "not a number", object(), float("nan"),
                                 float("inf"), float("-inf"), -1.5, 1.5,
                                 -1.0000000000000002, 1.0000000000000002])
def test_rung1_verdict_rejects_a_non_correlation(bad):
    """An undefined or out-of-domain rho is refused, not silently given a verdict.

    `None` is the probe's encoding of "Spearman is undefined here" -- fewer than 3
    velocity pairs, or a constant column -- and the gate must not manufacture a verdict
    from it. `nan` and values outside `[-1, 1]` are not rank correlations at all.
    """
    with pytest.raises(ValueError):
        rung1_verdict(bad)


def test_rung1_verdict_coerces_a_numeric_string_like_rule_b_does():
    """Deliberate leniency, matching `rule_b_verdict`: a JSON round-trip stays readable.

    What is *not* lenient is the domain check -- a string that parses to something
    outside `[-1, 1]` still raises -- so this cannot become a back door.
    """
    assert rung1_verdict("-0.4")["verdict"] == VERDICT_GO
    assert rung1_verdict("-0.4")["rho"] == pytest.approx(-0.4)
    with pytest.raises(ValueError):
        rung1_verdict("-2.0")


def test_rho_standard_error_is_one_over_sqrt_n_minus_one():
    """Section 4.3 asks for `1/sqrt(n-1)` beside rho, and for it to be ~0.01 at n~1e4."""
    assert rho_standard_error(2) == pytest.approx(1.0)
    assert rho_standard_error(101) == pytest.approx(0.1)
    assert rho_standard_error(10001) == pytest.approx(0.01)
    assert rho_standard_error(1) is None
    assert rho_standard_error(None) is None


# ===========================================================================
# 5. Check A is flagged as never gating (section 4.2)
# ===========================================================================
def test_check_a_is_flagged_not_gating_and_carries_the_layernorm_reason():
    """So nobody downstream can read `CV(r)` as a gate."""
    trained = {"cv_r": 0.4, "mca": 0.16, "n_pairs": 300, "n_windows": 100}
    pristine = {"cv_r": 0.5, "mca": 0.25, "n_pairs": 300, "n_windows": 100}
    block = check_a_block(trained, pristine)
    assert block["gating"] is False
    assert block["reason"] == MCA_CHECK_A_NOT_GATING_REASON
    assert "NEVER GATING" in block["reason"]
    assert "LayerNorm" in block["reason"] or "nn.LayerNorm" in block["reason"]
    assert block["cv_r_delta_trained_minus_pristine"] == pytest.approx(-0.1)
    assert "TOWARD similarity" in block["direction"]
    # Every statistic carries its denominators (section 4.4).
    for key in ("n_pairs_trained", "n_pairs_pristine", "n_windows_trained",
                "n_windows_pristine"):
        assert block[key] == 300 or block[key] == 100


def test_check_a_survives_a_missing_pristine_reference():
    block = check_a_block({"cv_r": 0.4, "mca": 0.16, "n_pairs": 3, "n_windows": 1}, None)
    assert block["gating"] is False
    assert block["cv_r_pristine"] is None
    assert block["direction"] is None


# ===========================================================================
# 6. Spearman: average-rank tie handling, against hand-computed values
# ===========================================================================
@pytest.mark.parametrize("values, expected", [
    ([10.0, 20.0, 30.0], [1.0, 2.0, 3.0]),
    ([30.0, 10.0, 20.0], [3.0, 1.0, 2.0]),
    # One tie block of three in the middle: 1-based ranks 2, 3, 4 -> mean 3.0.
    ([10.0, 20.0, 20.0, 20.0, 30.0], [1.0, 3.0, 3.0, 3.0, 5.0]),
    # A leading and a trailing tie block: (1+2)/2 = 1.5 and (3+4)/2 = 3.5.
    ([1.0, 1.0, 2.0, 2.0], [1.5, 1.5, 3.5, 3.5]),
    # Everything tied: (1+2+3+4)/4 = 2.5.
    ([7.0, 7.0, 7.0, 7.0], [2.5, 2.5, 2.5, 2.5]),
])
def test_average_ranks_on_hand_computed_examples(values, expected):
    """Average ranks inside every tie block, computed by hand from the definition."""
    assert list(_average_ranks(values)) == pytest.approx(expected)


def test_average_ranks_sum_is_the_triangular_number():
    """A necessary condition: average ranks preserve the total `n(n+1)/2`."""
    for values in ([1.0, 1.0, 2.0], [5.0] * 7, [3.0, 1.0, 2.0, 2.0, 9.0, 9.0]):
        n = len(values)
        assert float(sum(_average_ranks(values))) == pytest.approx(n * (n + 1) / 2)


def test_spearman_matches_a_hand_computed_value_with_ties_in_x():
    """`x = [1, 2, 2, 3]`, `y = [10, 20, 30, 40]`.

    Worked by hand from the definition:
        ranks(x) = [1, 2.5, 2.5, 4]   (the tie block at positions 2-3 averages to 2.5)
        ranks(y) = [1, 2, 3, 4]
        centred:  dx = [-1.5, 0, 0, 1.5]   dy = [-1.5, -0.5, 0.5, 1.5]
        sum dx*dy = 2.25 + 0 + 0 + 2.25 = 4.5
        sum dx^2  = 4.5      sum dy^2 = 5.0
        rho = 4.5 / sqrt(4.5 * 5.0) = 4.5 / sqrt(22.5) = 3 / sqrt(10)
    """
    rho, n = spearman_rho([1.0, 2.0, 2.0, 3.0], [10.0, 20.0, 30.0, 40.0])
    assert n == 4
    assert rho == pytest.approx(3.0 / math.sqrt(10.0))
    assert rho == pytest.approx(0.9486832980505138)


def test_spearman_matches_a_hand_computed_value_with_ties_in_both():
    """`x = [1, 1, 2, 2]`, `y = [1, 2, 2, 3]`.

        ranks(x) = [1.5, 1.5, 3.5, 3.5]   ranks(y) = [1, 2.5, 2.5, 4]
        centred:  dx = [-1, -1, 1, 1]     dy = [-1.5, 0, 0, 1.5]
        sum dx*dy = 1.5 + 0 + 0 + 1.5 = 3.0
        sum dx^2  = 4.0      sum dy^2 = 4.5
        rho = 3.0 / sqrt(18.0) = 1 / sqrt(2)
    """
    rho, n = spearman_rho([1.0, 1.0, 2.0, 2.0], [1.0, 2.0, 2.0, 3.0])
    assert n == 4
    assert rho == pytest.approx(1.0 / math.sqrt(2.0))


def test_spearman_disagrees_with_the_no_tie_shortcut_on_tied_data():
    """The proof that tie handling is actually implemented.

    `1 - 6*sum(d^2)/(n^3 - n)` is only correct without ties. On
    `x = [1, 2, 2, 3]`, `y = [10, 20, 30, 40]` it gives `1 - 6*0.5/60 = 0.95`, while the
    correct average-rank Pearson value is `3/sqrt(10) = 0.9486832...`. A hand-rolled
    Spearman that took the shortcut would land on 0.95 and would bias rho toward zero
    wherever the data is discrete -- i.e. it would move the gate.
    """
    x = [1.0, 2.0, 2.0, 3.0]
    y = [10.0, 20.0, 30.0, 40.0]
    rho, n = spearman_rho(x, y)
    d = [a - b for a, b in zip(_average_ranks(x), _average_ranks(y))]
    shortcut = 1.0 - 6.0 * sum(value * value for value in d) / (n ** 3 - n)
    assert shortcut == pytest.approx(0.95)
    assert rho != pytest.approx(shortcut, abs=1e-6)
    assert rho == pytest.approx(3.0 / math.sqrt(10.0))


@pytest.mark.parametrize("x, y, expected", [
    ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], 1.0),
    ([1.0, 2.0, 3.0], [3.0, 2.0, 1.0], -1.0),
    # Monotone but wildly nonlinear: rank correlation is still exactly 1, which is why
    # section 4.3 fixed on Spearman rather than Pearson.
    ([1.0, 2.0, 3.0, 4.0], [1.0, 10.0, 1e3, 1e9], 1.0),
    ([1.0, 2.0, 3.0, 4.0], [1e9, 1e3, 10.0, 1.0], -1.0),
])
def test_spearman_endpoints_and_monotone_nonlinearity(x, y, expected):
    rho, _n = spearman_rho(x, y)
    assert rho == pytest.approx(expected)
    assert -1.0 <= rho <= 1.0


def test_spearman_is_undefined_rather_than_zero_on_a_constant_column():
    """A constant column has no ranks to correlate; `None` says so, `0.0` would lie."""
    assert spearman_rho([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])[0] is None
    assert spearman_rho([1.0, 2.0, 3.0], [5.0, 5.0, 5.0])[0] is None
    assert spearman_rho([1.0, 2.0], [1.0, 2.0]) == (None, 2)


def test_spearman_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        spearman_rho([1.0, 2.0, 3.0], [1.0, 2.0])


def test_spearman_is_invariant_to_a_monotone_reparameterisation():
    """Rank correlation only sees the order, which is the property section 4.3 wants."""
    x = [0.5, 3.0, 1.25, 9.0, 2.0, 2.0]
    y = [4.0, 1.0, 3.0, 0.25, 2.0, 7.0]
    base, _n = spearman_rho(x, y)
    warped, _n = spearman_rho([value ** 3 for value in x],
                              [math.log1p(value) for value in y])
    assert warped == pytest.approx(base)


# ===========================================================================
# 7. The mandatory disaggregation (section 4.4) and the rank-disagreement rate (4.5)
# ===========================================================================
def test_decile_table_partitions_every_pair_by_v_patch_rank():
    """Equal-count groups, ascending in `v_patch`, with every pair accounted for."""
    n = 95
    v_patch = [float(n - index) for index in range(n)]     # descending on purpose
    r = [float(index) for index in range(n)]
    table = mca_decile_table(r, v_patch)
    assert len(table) == MCA_DECILES
    assert sum(row["n_pairs"] for row in table) == n
    assert [row["decile"] for row in table] == list(range(1, MCA_DECILES + 1))
    # Group boundaries ascend in v_patch, so decile 1 is the smallest motion.
    maxima = [row["v_patch_max"] for row in table]
    assert maxima == sorted(maxima)
    assert table[0]["v_patch_min"] == pytest.approx(1.0)
    assert table[-1]["v_patch_max"] == pytest.approx(float(n))


def test_decile_table_reports_mean_and_median_r_per_decile():
    """`r = v_patch` makes both statistics hand-checkable: decile 1 is 1..10."""
    v_patch = [float(index) for index in range(1, 101)]
    table = mca_decile_table(v_patch, v_patch)
    assert table[0]["n_pairs"] == 10
    assert table[0]["r_mean"] == pytest.approx(5.5)
    assert table[0]["r_median"] == pytest.approx(5.5)
    assert table[-1]["r_mean"] == pytest.approx(95.5)


def test_histogram_is_lossless_over_the_fixed_range():
    """`counts + below_range + above_range == n_pairs`, so no mass is silently dropped.

    The range is FIXED, not data-derived, so the trained and pristine histograms share
    an x-axis and can be overlaid. Mass beyond it is counted, not discarded.
    """
    r = [0.0, 0.5, 1.0, 1.0, 1.5, 2.0, 40.0]
    histogram = mca_ratio_histogram(r)
    assert histogram["bins"] == MCA_HIST_BINS
    assert histogram["range"] == [float(MCA_HIST_RANGE[0]), float(MCA_HIST_RANGE[1])]
    assert histogram["range_is_fixed"] is True
    assert len(histogram["edges"]) == MCA_HIST_BINS + 1
    assert len(histogram["counts"]) == MCA_HIST_BINS
    assert (sum(histogram["counts"]) + histogram["below_range"]
            + histogram["above_range"]) == histogram["n_pairs"] == len(r)
    assert histogram["above_range"] >= 1        # the 40.0 outlier is counted, not lost


def test_histogram_x_axis_is_r_over_r_bar():
    """A constant `r` puts every pair in the bin containing 1.0, whatever the scale."""
    for scale in (1e-3, 1.0, 1e3):
        histogram = mca_ratio_histogram([scale] * 50)
        assert histogram["r_bar"] == pytest.approx(max(scale, MCA_EPS))
        occupied = [index for index, count in enumerate(histogram["counts"]) if count]
        assert len(occupied) == 1
        low = histogram["edges"][occupied[0]]
        high = histogram["edges"][occupied[0] + 1]
        assert low <= 1.0 <= high


@pytest.mark.parametrize("v_patch, v_agg, expected_rate", [
    # Perfectly reversed: every unordered pair disagrees.
    ([1.0, 2.0, 3.0], [3.0, 2.0, 1.0], 1.0),
    # Perfectly concordant: none does.
    ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], 0.0),
    # One reversal out of three pairs: (0,1) disagrees, (0,2) and (1,2) agree.
    ([1.0, 2.0, 3.0], [2.0, 1.0, 3.0], 1.0 / 3.0),
])
def test_rank_disagreement_rate_on_hand_computed_examples(v_patch, v_agg, expected_rate):
    result = rank_disagreement_rate(v_patch, v_agg)
    assert result["exhaustive"] is True
    assert result["pairs_compared"] == result["total_pairs"] == 3
    assert result["rate"] == pytest.approx(expected_rate)


def test_rank_disagreement_ties_are_in_the_denominator_never_disagreements():
    """No ordering is not a disagreement, and the tie count is reported."""
    result = rank_disagreement_rate([1.0, 1.0, 2.0], [5.0, 3.0, 4.0])
    assert result["tied_pairs"] == 1                    # the (0, 1) v_patch tie
    assert result["pairs_compared"] == 3
    assert result["disagreements"] == 1                 # (1, 2): v_agg down, v_patch up
    assert result["rate"] == pytest.approx(1.0 / 3.0)


def test_rank_disagreement_samples_deterministically_above_the_cap():
    """Above the exhaustive cap the sample is seeded, so two probes compare the same
    pairs, and the report says it was sampled rather than exhaustive."""
    v_patch = [float(index) for index in range(200)]
    v_agg = [float(200 - index) for index in range(200)]
    first = rank_disagreement_rate(v_patch, v_agg, max_exhaustive=500)
    second = rank_disagreement_rate(v_patch, v_agg, max_exhaustive=500)
    assert first["exhaustive"] is False
    assert first["pairs_compared"] == 500
    assert first["total_pairs"] == 200 * 199 // 2
    assert first["rate"] == second["rate"] == pytest.approx(1.0)
    assert first["seed"] == second["seed"]


def test_rank_disagreement_is_empty_rather_than_raising_on_one_value():
    result = rank_disagreement_rate([1.0], [1.0])
    assert result["rate"] is None
    assert result["total_pairs"] == 0


# ===========================================================================
# 8. The advisory on sample size is an advisory, not a gate
# ===========================================================================
def test_low_power_advisory_does_not_change_the_verdict():
    """Section 4.3's thresholds are effect sizes, not significance thresholds.

    So a small `n` warns and is recorded, and the verdict is still read off rho alone.
    """
    assert MCA_ADVISORY_MIN_PAIRS > 0
    block = rung1_verdict(-0.35)
    assert block["verdict"] == VERDICT_GO
    assert "n_pairs" not in block
    assert set(block["thresholds"]) == {"go", "middle"}


def test_the_verdict_is_a_pure_function_of_one_float():
    """No file, no tensor, no dataset: the same rho gives the same block, always."""
    first = rung1_verdict(-0.2)
    second = rung1_verdict(-0.2)
    assert first == second
    assert isinstance(first["rho"], float)
    # And it accepts a plain int / numpy-free float without complaint.
    assert rung1_verdict(0)["clause"] == "4.3-stop"
    assert rung1_verdict(-1)["verdict"] == VERDICT_GO


# ===========================================================================
# 9. The probe's measurement path, end to end on the stub model
#
# The probe itself cannot be run here: it needs DATASET_DIR and the PushT checkpoint,
# neither of which is on a development box. So `mca_measure` and `mca_statistics` are
# exercised against synthetic windows in exactly the shape `load_windows` returns them,
# which is what keeps the wiring -- the per-pair-to-window mapping the terciles index
# through, the finiteness filter, the pooled reduction, the identity check -- from being
# discovered broken on the pod.
# ===========================================================================
def _stub_windows(model, count=12, num_frames=4, seed=0):
    """Windows in `load_windows`' shape: batch-of-1 obs/act plus a `(t, 5)` state."""
    windows = []
    for index in range(count):
        obs, act = make_stub_batch(model, batch_size=1, num_frames=num_frames,
                                   seed=seed + index)
        generator = torch.Generator(device="cpu").manual_seed(1000 + seed + index)
        state = torch.rand(num_frames, len(STATE_DIM_NAMES), generator=generator,
                           dtype=torch.float32).numpy().astype("float64")
        windows.append({"index": index, "obs": obs, "act": act, "state": state})
    return windows


def test_mca_measure_pools_the_shipped_terms_and_agrees_with_compute_mca():
    """Every window's reduction is bitwise `compute_mca`, and the pool is consistent."""
    model = build_stub_world_model(agg_type="mlp")
    windows = _stub_windows(model, count=12, num_frames=4)
    budget = Budget(0.0)
    measurement = mca_measure(model, windows, budget, budget.deadline(1.0), "test")

    assert measurement["n_windows"] == 12
    assert measurement["pairs_per_window"] == 3
    assert measurement["n_pairs"] == 36
    assert measurement["nonfinite_windows"] == 0
    assert measurement["partial"] is False
    # Section 4.1: the probe's reduction and the shipped term are the same number.
    assert measurement["reduction_bitwise_mismatches"] == []
    assert len(measurement["mca_per_window"]) == 12
    assert measurement["r"].shape == measurement["v_patch"].shape == (36,)
    assert measurement["window_of_pair"].tolist() == [
        index for index in range(12) for _ in range(3)]
    assert measurement["state_motion"].shape == (12, len(STATE_DIM_NAMES))
    # The pooled value is what `compute_mca` returns with every window in one batch.
    assert tuple(measurement["r_tensor"].shape) == (12, 3)
    assert measurement["mca_pooled"] == pytest.approx(
        float(mca_reduction(measurement["r_tensor"])), rel=0, abs=0)


def test_mca_statistics_reports_every_mandatory_readout_with_its_denominators():
    """Section 4.4: nothing is reported without `n_pairs` and `n_windows` beside it."""
    model = build_stub_world_model(agg_type="mlp")
    windows = _stub_windows(model, count=30, num_frames=4)
    budget = Budget(0.0)
    stats = mca_statistics(mca_measure(model, windows, budget, budget.deadline(1.0),
                                       "test"))

    assert stats["n_windows"] == 30 and stats["n_pairs"] == 90
    assert stats["cv_r"] is not None and stats["cv_r"] >= 0.0
    # The identity, checked the independent float64 way against the shipped reduction.
    assert stats["identity_holds"] is True
    assert abs(stats["cv_squared_relative_residual"]) <= MCA_CV_RELATIVE_TOLERANCE
    assert stats["cv_r_squared"] == stats["mca"]
    # Check B's statistic and its standard error.
    assert stats["rho"] is None or -1.0 <= stats["rho"] <= 1.0
    assert stats["rho_n_pairs"] == 90
    assert stats["rho_standard_error"] == pytest.approx(1.0 / math.sqrt(89.0))
    # The three mandatory disaggregations.
    assert len(stats["deciles"]) == MCA_DECILES
    assert sum(row["n_pairs"] for row in stats["deciles"]) == 90
    assert len(stats["histogram"]["counts"]) == MCA_HIST_BINS
    assert set(stats["per_dim_tercile"]) == set(STATE_DIM_NAMES)
    for name, entry in stats["per_dim_tercile"].items():
        # `_top_tercile_mask` takes ceil(n/3) windows, and every pair of a selected
        # window comes with it.
        assert entry["n_windows"] == math.ceil(30 / 3), name
        assert entry["n_pairs"] == entry["n_windows"] * 3, name
        assert entry["rho"] is None or -1.0 <= entry["rho"] <= 1.0, name
    assert stats["rank_disagreement"]["exhaustive"] is True
    assert stats["rank_disagreement"]["pairs_compared"] == 90 * 89 // 2
    # 90 pairs is far below the advisory floor, so the flag fires and says so.
    assert stats["low_power"] is True


def test_mca_statistics_survives_an_empty_measurement():
    """No finite window is an error the caller reports, not a crash in the statistics."""
    model = build_stub_world_model(agg_type="mlp")
    budget = Budget(0.0)
    measurement = mca_measure(model, [], budget, budget.deadline(1.0), "empty")
    assert measurement["n_pairs"] == 0
    stats = mca_statistics(measurement)
    assert stats["n_pairs"] == 0
    assert stats["cv_r"] is None and stats["rho"] is None
    assert stats["deciles"] == []
    assert set(stats["per_dim_tercile"]) == set(STATE_DIM_NAMES)


def test_mca_measure_excludes_a_non_finite_window_and_counts_it():
    """A non-finite latent carries no information; it is counted, reported, excluded."""
    model = build_stub_world_model(agg_type="mlp")
    windows = _stub_windows(model, count=4, num_frames=4)
    windows[1]["obs"] = {"visual": windows[1]["obs"]["visual"] * float("nan"),
                         "proprio": windows[1]["obs"]["proprio"]}
    budget = Budget(0.0)
    measurement = mca_measure(model, windows, budget, budget.deadline(1.0), "test")
    assert measurement["nonfinite_windows"] == 1
    assert measurement["n_windows"] == 3
    assert measurement["n_pairs"] == 9
    # The per-pair owner map still indexes the KEPT windows contiguously, which is what
    # the tercile masks are indexed through.
    assert measurement["window_of_pair"].tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert measurement["state_motion"].shape[0] == 3


class _StubCfgNode(dict):
    """The two attribute-plus-`get` operations `build_aggmetric_report` performs on the
    resolved `hydra.yaml`. omegaconf is not installed in the test environment, and the
    report builder only ever reads `env.name`, `num_hist`, `num_pred` and
    `encoder.agg_out_dim`, so a dict with attribute access is a faithful stand-in."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:                       # pragma: no cover
            raise AttributeError(name) from exc


class _StubArgs:
    def __init__(self, num_windows, max_minutes=0.0):
        self.num_windows = num_windows
        self.max_minutes = max_minutes


def test_the_report_serialises_to_json_and_prints_without_crashing(tmp_path, capsys):
    """The report is machine-readable first and printed second, and both must work.

    The probe cannot be executed here (no `DATASET_DIR`, no checkpoint), so a broken
    format string or a non-serialisable value in the report would otherwise be found on
    the pod, after the measurement. `json.dumps` with `allow_nan=False` is the strict
    form: a bare `Infinity` or `NaN` is not valid JSON and no strict parser will read it
    back.
    """
    import json

    model = build_stub_world_model(agg_type="mlp")
    windows = _stub_windows(model, count=12, num_frames=4)
    budget = Budget(0.0)
    measurement = mca_measure(model, windows, budget, budget.deadline(1.0), "checkpoint")
    trained = mca_statistics(measurement)

    pristine_measurement = mca_measure(build_stub_world_model(agg_type="mlp", seed=7),
                                       windows, budget, budget.deadline(1.0),
                                       "pristine")
    pristine = mca_statistics(pristine_measurement)

    train_cfg = _StubCfgNode(
        env=_StubCfgNode(name="pusht"), num_hist=3, num_pred=1,
        encoder=_StubCfgNode(agg_out_dim=128))
    fingerprint = {"sha256": "0" * 64, "size_bytes": 1, "mtime": "1970-01-01T00:00:00Z"}
    verdict = rung1_verdict(trained["rho"]) if trained["rho"] is not None else None
    report = build_aggmetric_report(
        _StubArgs(num_windows=12), tmp_path / "model_2.pth", tmp_path / "hydra.yaml",
        fingerprint, fingerprint, train_cfg, measurement, trained, pristine,
        "pristine", check_a_block(trained, pristine), verdict, budget, 4, 2,
        {"state_dim": 5})

    encoded = json.dumps(report, allow_nan=False)
    assert json.loads(encoded)["schema"] == report["schema"]
    assert report["check_a"]["gating"] is False
    assert report["agg_out_dim"] == 128
    assert report["checkpoint_modified"] is False
    assert report["histogram_range_fixed"] == list(MCA_HIST_RANGE)
    assert report["notes"], "section 4.6's caveats travel with the report"

    print_aggmetric_report(report)
    printed = capsys.readouterr().out
    # The operator must not have to remember the rule, so the whole ladder is printed.
    assert "CHECK A" in printed and "NEVER GATING" in printed
    assert "CHECK B -- THE GATE" in printed
    assert "-> GO" in printed and "-> MIDDLE" in printed and "-> STOP" in printed
    assert "RUNG-1 VERDICT" in printed
    assert "RANK-DISAGREEMENT RATE" in printed
    assert "r BY DECILE" in printed
    assert "HISTOGRAM OF r/r_bar" in printed
    assert "PER STATE-DIMENSION TERCILE" in printed
    for dim in STATE_DIM_NAMES:
        assert dim in printed


def test_the_report_prints_when_the_pristine_reference_is_missing(tmp_path, capsys):
    """A reference the hub cache cannot supply degrades check A, it does not break it."""
    import json

    model = build_stub_world_model(agg_type="mlp")
    windows = _stub_windows(model, count=6, num_frames=4)
    budget = Budget(0.0)
    measurement = mca_measure(model, windows, budget, budget.deadline(1.0), "checkpoint")
    trained = mca_statistics(measurement)
    train_cfg = _StubCfgNode(env=_StubCfgNode(name="pusht"), num_hist=3, num_pred=1,
                             encoder=_StubCfgNode(agg_out_dim=128))
    fingerprint = {"sha256": "a" * 64, "size_bytes": 1, "mtime": "1970-01-01T00:00:00Z"}
    report = build_aggmetric_report(
        _StubArgs(num_windows=6), tmp_path / "model_2.pth", tmp_path / "hydra.yaml",
        fingerprint, fingerprint, train_cfg, measurement, trained, None, None,
        check_a_block(trained, None), rung1_verdict(-0.4), budget, 4, 2,
        {"state_dim": 5})
    json.dumps(report, allow_nan=False)
    print_aggmetric_report(report)
    printed = capsys.readouterr().out
    assert "RUNG-1 VERDICT        : GO" in printed
    assert report["statistics"]["pristine"] is None


def test_every_agg_type_in_the_enum_is_parametrized():
    """A new `agg_type` must not silently escape the bitwise comparison above.

    `test_compute_mca_is_bitwise_the_frozen_reference` is parametrized over
    :data:`tests.conftest.AGG_TYPES` directly, so this only has to pin that the enum is
    still the three-branch contract of `models/dino.py::DinoV2Encoder.agg`. If a fourth
    branch is added there and mirrored into the stub, this fails and the new branch gets
    a bitwise comparison rather than being assumed neutral.
    """
    assert set(AGG_TYPES) == {"mean", "flatten", "mlp"}
    covered = {case for case, _concat in itertools.product(AGG_TYPES, (0, 1))}
    assert covered == set(AGG_TYPES)
