"""Can the probe tell "orientation is absent" from "the probe cannot read orientation"?

`PROGRESS_CCR.md` section 6f rests a mechanism claim on `state_readout_r2["block_angle"]` being
0.183 against 0.50-0.80 for the four positional dimensions. `state_readout` is a **linear** ridge
onto the raw state value, and `block_angle` is periodic, so that number is ambiguous between:

  (a) the representation discards orientation, or
  (b) the representation encodes orientation perfectly as `(cos t, sin t)` and a linear map onto
      `t` cannot read it, because `t` jumps a full period where the representation is continuous.

If (b), the entire rotation-straightening direction is void and no GPU should be spent on it. These
tests construct both worlds synthetically, with the ground truth known, and assert the pair of
readouts separates them. That is the discriminating check, and it has to pass before the probe's
output on a real checkpoint means anything.

CPU only, numpy only, no checkpoint, no dataset, no GPU.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import probe_ccr_curvature as probe  # noqa: E402


ANGLE = "block_angle"


def _measurement(latent, state):
    """The fields the readouts read, plus the ones `curvature_readout` needs.

    `readouts_from_measurement` also calls `curvature_readout`, so the stub carries its
    `unperturbed` / `perturbed` fields too. They are not exercised here -- this module is about
    the state readouts -- but omitting them would make the report-level test fail for an
    unrelated reason and hide whichever assertion it was written for.
    """
    n = latent.shape[0]
    return {
        "agg": latent,
        "state": state,
        "windows": n,
        "unperturbed": np.zeros((n, 4)),
        "perturbed": np.zeros((n, 4)),
        "motion": np.zeros((n, len(probe.STATE_DIM_NAMES))),
    }


def _readable(m):
    """The `orientation_readable` block, without routing through `curvature_readout`.

    Keeps this module's failures attributable to the state readouts it is about.
    """
    return probe._orientation_readable(
        probe.state_readout(m), probe.circular_state_readout(m)
    )


def _angles(n_windows, t, rng, period=2 * np.pi):
    """Angles sweeping the full circle, so the wrap point is inside the data."""
    theta = rng.uniform(-period / 2, period / 2, size=(n_windows, t))
    return theta


def _positional(n_windows, t, rng):
    return rng.uniform(-1.0, 1.0, size=(n_windows, t))


def _build(n_windows=64, t=8, seed=0, encode="cos_sin", period=2 * np.pi):
    """A latent that encodes orientation either as (cos, sin) or as the raw angle."""
    rng = np.random.default_rng(seed)
    theta = _angles(n_windows, t, rng, period)
    px, py, bx, by = (_positional(n_windows, t, rng) for _ in range(4))

    k = 2 * np.pi / period
    if encode == "cos_sin":
        angle_feats = [np.cos(k * theta), np.sin(k * theta)]
    elif encode == "raw":
        angle_feats = [k * theta, np.zeros_like(theta)]
    elif encode == "absent":
        angle_feats = [rng.normal(size=theta.shape), rng.normal(size=theta.shape)]
    else:  # pragma: no cover
        raise AssertionError(encode)

    latent = np.stack([px, py, bx, by, *angle_feats], axis=-1)
    # A little padding so the ridge sees more features than targets, as it does in practice.
    latent = np.concatenate([latent, rng.normal(scale=1e-3, size=latent.shape)], axis=-1)

    order = {n: i for i, n in enumerate(probe.STATE_DIM_NAMES)}
    state = np.zeros((n_windows, t, len(probe.STATE_DIM_NAMES)))
    for name, col in (("agent_x", px), ("agent_y", py), ("block_x", bx), ("block_y", by)):
        state[..., order[name]] = col
    state[..., order[ANGLE]] = theta
    return _measurement(latent, state)


def _linear_r2(m):
    return probe.state_readout(m)["per_dim"][ANGLE]


def _circular_r2(m):
    return probe.circular_state_readout(m)["per_dim"][ANGLE]["circular_r2"]


# ---------------------------------------------------------------------------
# The discriminating pair
# ---------------------------------------------------------------------------


def test_cos_sin_encoding_looks_bad_to_the_linear_readout_and_good_to_the_circular_one():
    """World (b): orientation IS encoded. The linear readout under-reports it.

    Note the *measured* size of the effect, which is the calibration the real result has to be
    read against: a linear readout of an exactly `(cos, sin)`-encoded angle scores around 0.60,
    i.e. **inside** the 0.50-0.80 band the positional dimensions occupy. It is under-reporting,
    not blindness. `PROGRESS_CCR.md` section 5c observed 0.183, far below 0.60, so the linear
    readout's mis-specification does **not** by itself explain that number.
    """
    m = _build(encode="cos_sin")
    lin, circ = _linear_r2(m), _circular_r2(m)

    assert circ > 0.95, f"circular readout should recover a (cos,sin) encoding, got {circ:.4f}"
    assert 0.4 < lin < 0.8, (
        f"calibration: a linear readout of a (cos,sin) encoding lands in the positional band, "
        f"got {lin:.4f}. If this drifts, the interpretation of the real 0.183 changes with it"
    )
    assert circ - lin > 0.3, (
        f"the two readouts must disagree in this world (linear {lin:.4f}, circular {circ:.4f}); "
        f"if they agreed the probe could not separate 'absent' from 'unreadable'"
    )


def test_absent_orientation_looks_bad_to_both_readouts():
    """World (a): orientation is genuinely absent. Both readouts must say so.

    Without this, a high circular R^2 could be an artifact of the richer target basis rather
    than evidence the representation holds orientation.
    """
    m = _build(encode="absent")
    lin, circ = _linear_r2(m), _circular_r2(m)

    assert circ < 0.3, f"circular readout must not manufacture signal, got {circ:.4f}"
    assert lin < 0.3, f"linear readout should also be near zero, got {lin:.4f}"


def test_raw_angle_encoding_reverses_which_readout_wins():
    """The converse artifact, which is why neither readout may be used alone.

    If the latent carries the angle *linearly*, the circular readout under-reports instead: a
    linear map cannot produce `cos t` from `t` any more than it can produce `t` from
    `(cos t, sin t)`. Measured at ~0.50 against the linear readout's ~1.0.

    This is the test that forces the decision rule to be `max(linear, circular)` rather than
    "use the circular one, it's better specified".
    """
    m = _build(encode="raw")
    lin, circ = _linear_r2(m), _circular_r2(m)

    assert lin > 0.9, f"a linearly-encoded angle must read linearly, got {lin:.4f}"
    assert circ < 0.8, (
        f"calibration of the converse artifact: circular readout under-reports a raw-angle "
        f"encoding, got {circ:.4f}"
    )
    assert lin > circ, "which readout wins must depend on the encoding, or neither is diagnostic"


def test_the_decision_rule_is_the_max_of_both():
    """`best_r2` must recover orientation under *either* encoding, and stay low when absent.

    This is the only quantity a claim about orientation loss may be built on.
    """
    for encode, expect_high in (("cos_sin", True), ("raw", True), ("absent", False)):
        best = _readable(_build(encode=encode, seed=11))[ANGLE]["best_r2"]
        if expect_high:
            assert best > 0.9, f"{encode}: orientation is present, best_r2 {best:.4f}"
        else:
            assert best < 0.3, f"{encode}: orientation is absent, best_r2 {best:.4f}"


# ---------------------------------------------------------------------------
# Properties of the circular score itself
# ---------------------------------------------------------------------------


def test_circular_r2_is_zero_for_a_constant_predictor():
    """The denominator is the circular mean, so no-signal must score ~0, not negative infinity."""
    m = _build(encode="absent", seed=3)
    circ = _circular_r2(m)
    assert -1.0 < circ < 0.3, f"expected a bounded near-zero score, got {circ:.4f}"


def test_units_are_inferred_and_recorded_not_assumed():
    """Degrees must not be silently read as radians -- that would make the score meaningless."""
    rad = probe.circular_state_readout(_build(encode="cos_sin", period=2 * np.pi))
    deg = probe.circular_state_readout(_build(encode="cos_sin", period=360.0))

    assert rad["per_dim"][ANGLE]["unit"] == "radians"
    assert deg["per_dim"][ANGLE]["unit"] == "degrees"
    assert deg["per_dim"][ANGLE]["period"] == 360.0
    # Same underlying world, so the score must not depend on the unit it was recorded in.
    assert abs(rad["per_dim"][ANGLE]["circular_r2"] - deg["per_dim"][ANGLE]["circular_r2"]) < 0.05


def test_only_angular_dimensions_are_measured():
    """Positional dimensions are not periodic and must be left to the linear readout."""
    out = probe.circular_state_readout(_build())
    assert out["angular_dims"] == [ANGLE]
    assert set(out["per_dim"]) == {ANGLE}


def test_linear_readout_is_untouched_by_the_addition():
    """`state_readout` must still return exactly what the progress logs cite.

    The additive contract: previously recorded `state_readout_r2` values stay reproducible, so
    the new reading sits beside the old rather than silently replacing a number under a claim.
    """
    m = _build(encode="cos_sin", seed=7)
    first = probe.state_readout(m)
    probe.circular_state_readout(m)
    second = probe.state_readout(m)
    assert first["per_dim"] == second["per_dim"]
    assert first["aggregate"] == second["aggregate"]


def test_report_carries_both_readings():
    """`readouts_from_measurement` must expose the circular block, or the pod run cannot use it."""
    r = probe.readouts_from_measurement(_build())
    assert "state_readout_r2" in r
    assert "state_readout_circular" in r
    assert ANGLE in r["state_readout_circular"]["per_dim"]


def test_too_few_windows_returns_none_rather_than_a_number():
    m = _build(n_windows=2)
    out = probe.circular_state_readout(m)
    assert out["per_dim"][ANGLE] is None


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_separation_is_not_a_seed_artifact(seed):
    """The whole decision rests on this gap, so it is checked across seeds."""
    m = _build(encode="cos_sin", seed=seed)
    assert _circular_r2(m) - _linear_r2(m) > 0.3
