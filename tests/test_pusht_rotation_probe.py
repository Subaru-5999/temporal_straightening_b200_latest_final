"""The rotation probe must not manufacture rotation, and must not miss it.

This probe can void the whole rotation direction (`PROGRESS_ROT.md` §5.2 explanation 3), so its
arithmetic is checked against synthetic trajectories whose true rotation is known. The failure that
matters most is the wrap artifact: a naive angle difference across the period boundary produces a
spurious full-period jump, which would inflate every statistic and manufacture exactly the answer
the direction wants.

CPU only, no dataset, no torch load, no GPU.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import probe_pusht_rotation as rot  # noqa: E402

TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------------------
# wrapped_delta -- the load-bearing primitive
# ---------------------------------------------------------------------------


def test_wrapped_delta_is_zero_across_a_full_wrap():
    """A jump of exactly one period is no rotation at all."""
    assert abs(rot.wrapped_delta(0.1, 0.1 + TWO_PI, TWO_PI)) < 1e-9
    assert abs(rot.wrapped_delta(10.0, 370.0, 360.0)) < 1e-9


def test_wrapped_delta_takes_the_short_way_round():
    """Crossing the boundary must read as a small step, not a near-full-period one."""
    d = rot.wrapped_delta(-math.pi + 0.05, math.pi - 0.05, TWO_PI)
    assert abs(abs(d) - 0.10) < 1e-6, f"expected ~0.10 rad the short way, got {d}"

    d_deg = rot.wrapped_delta(359.0, 1.0, 360.0)
    assert abs(abs(d_deg) - 2.0) < 1e-6, f"expected ~2 deg the short way, got {d_deg}"


def test_naive_difference_would_have_been_wrong():
    """Documents the artifact this primitive exists to prevent."""
    a, b = -math.pi + 0.05, math.pi - 0.05
    naive = b - a                                  # ~6.18 rad, a fictitious near-full rotation
    wrapped = rot.wrapped_delta(a, b, TWO_PI)      # ~0.10 rad, the truth
    assert naive > 6.0 and abs(wrapped) < 0.2


def test_wrapped_delta_is_antisymmetric_and_bounded():
    rng = np.random.default_rng(0)
    a = rng.uniform(-math.pi, math.pi, 500)
    b = rng.uniform(-math.pi, math.pi, 500)
    fwd = rot.wrapped_delta(a, b, TWO_PI)
    rev = rot.wrapped_delta(b, a, TWO_PI)
    assert np.allclose(fwd, -rev, atol=1e-9)
    assert np.all(np.abs(fwd) <= math.pi + 1e-9)


# ---------------------------------------------------------------------------
# period inference
# ---------------------------------------------------------------------------


def test_period_inferred_from_range_not_assumed():
    assert rot.infer_period(np.array([-3.0, 3.0])) == (TWO_PI, "radians")
    assert rot.infer_period(np.array([-170.0, 350.0])) == (360.0, "degrees")


# ---------------------------------------------------------------------------
# end-to-end on synthetic trajectories with known rotation
# ---------------------------------------------------------------------------


def _states(n_roll, T, per_step_rot_rad, translate=0.01, wrap=True, seed=0):
    """Rollouts rotating at a known constant rate, optionally crossing the wrap point."""
    rng = np.random.default_rng(seed)
    s = np.zeros((n_roll, T, 5), dtype=np.float64)
    for i in range(n_roll):
        start = rng.uniform(-math.pi, math.pi) if wrap else 0.0
        theta = start + per_step_rot_rad * np.arange(T)
        if wrap:
            theta = np.arctan2(np.sin(theta), np.cos(theta))   # keep inside (-pi, pi]
        s[i, :, 4] = theta
        s[i, :, 2] = translate * np.arange(T)                  # block_x
        s[i, :, 3] = 0.0
        s[i, :, 0] = rng.normal(size=T) * 0.01
        s[i, :, 1] = rng.normal(size=T) * 0.01
    return s


def test_known_constant_rotation_is_recovered_despite_wrapping():
    """1 degree per step over 25 steps must read as ~25 degrees, wrap or no wrap."""
    per_step = math.radians(1.0)
    m = rot.measure(_states(24, 80, per_step, wrap=True), None, TWO_PI)
    med = m["horizons"]["25"]["rotation_deg"]["median"]
    assert abs(med - 25.0) < 1.0, f"expected ~25 deg over 25 steps, got {med:.3f}"


def test_a_static_block_reads_as_no_rotation():
    m = rot.measure(_states(16, 60, 0.0, wrap=True), None, TWO_PI)
    med = m["horizons"]["25"]["rotation_deg"]["median"]
    assert med < 1e-6, f"a non-rotating block must read ~0, got {med}"
    assert m["horizons"]["25"]["fraction_exceeding_deg"]["15.0"] == 0.0


def test_rotation_grows_with_horizon():
    m = rot.measure(_states(16, 90, math.radians(0.5)), None, TWO_PI)
    meds = [m["horizons"][str(h)]["rotation_deg"]["median"] for h in (5, 25, 50)]
    assert meds[0] < meds[1] < meds[2], f"rotation should accumulate with horizon: {meds}"


def test_translation_is_reported_alongside_rotation():
    """Rotation has to be comparable against the translation the model does encode."""
    m = rot.measure(_states(8, 60, math.radians(1.0), translate=0.02), None, TWO_PI)
    t = m["horizons"]["25"]["block_translation"]["median"]
    assert abs(t - 0.02 * 25) < 1e-6, f"expected 0.5 of translation over 25 steps, got {t}"


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------


def _report(states):
    r = {"measurements": rot.measure(states, None, TWO_PI)}
    return rot.verdict(r)


def test_gate_says_substantial_for_real_rotation():
    g = _report(_states(24, 80, math.radians(1.0)))
    assert g["verdict"] == "SUBSTANTIAL", g


def test_gate_says_negligible_for_a_static_block():
    """This is the outcome that would void the direction, so it must be reachable."""
    g = _report(_states(24, 80, 0.0))
    assert g["verdict"] == "NEGLIGIBLE", g
    assert "VOID" in g["reason"]


def test_gate_has_a_marginal_band():
    g = _report(_states(24, 80, math.radians(0.2)))
    assert g["verdict"] in {"MARGINAL", "SUBSTANTIAL", "NEGLIGIBLE"}
    assert "criterion" in g


def test_gate_criterion_is_stated_in_the_output():
    """The threshold travels with the number, so a reader cannot mistake which was applied."""
    g = _report(_states(8, 60, math.radians(1.0)))
    assert "median >= 10" in g["criterion"] and "0.20" in g["criterion"]


# ---------------------------------------------------------------------------
# padding
# ---------------------------------------------------------------------------


def test_constant_tail_is_trimmed_when_seq_lengths_are_absent():
    """Padding that repeats the last frame must not be counted as 'no rotation'."""
    s = _states(4, 60, math.radians(1.0), wrap=False)
    s[:, 40:, :] = s[:, 39:40, :]                    # 20 frames of exact padding
    assert rot.valid_length(s, None, 0) == 40

    seq = np.full(4, 40, dtype=np.int64)
    assert rot.valid_length(s, seq, 0) == 40


def test_seq_lengths_take_precedence_over_inference():
    s = _states(4, 60, math.radians(1.0), wrap=False)
    seq = np.full(4, 30, dtype=np.int64)
    assert rot.valid_length(s, seq, 0) == 30


# ---------------------------------------------------------------------------
# Dataset layout -- the two assumptions that made the first pod run fail
# ---------------------------------------------------------------------------
#
# `datasets.pusht_dset.load_pusht_slice_train_val` appends "/train" and "/val" to the configured
# `data_path`, so `$DATASET_DIR/pusht_noise` holds no `states.pth` itself; and the sequence lengths
# are a **pickle** (`seq_lengths.pkl`), not a tensor. I guessed both wrong, so both are pinned here.


def _write_split(tmp_path, name, n_roll=4, T=40, with_pkl=True):
    import pickle

    import torch

    d = tmp_path / name if name else tmp_path
    d.mkdir(parents=True, exist_ok=True)
    states = torch.zeros(n_roll, T, 5)
    states[..., 4] = torch.linspace(0, 1, T)
    torch.save(states, d / "states.pth")
    if with_pkl:
        with open(d / "seq_lengths.pkl", "wb") as fh:
            pickle.dump([T] * n_roll, fh)
    return d


def test_dataset_root_resolves_to_both_splits(tmp_path):
    _write_split(tmp_path, "train")
    _write_split(tmp_path, "val")
    got = rot.resolve_splits(tmp_path)
    assert [n for n, _ in got] == ["train", "val"], got


def test_a_single_split_directory_is_accepted_directly(tmp_path):
    d = _write_split(tmp_path, "train")
    got = rot.resolve_splits(d)
    assert len(got) == 1 and got[0][1] == d


def test_missing_states_names_the_split_layout(tmp_path):
    """The error must say where it looked, since the first failure was a wrong assumption."""
    with pytest.raises(FileNotFoundError) as e:
        rot.resolve_splits(tmp_path)
    msg = str(e.value)
    assert "train" in msg and "val" in msg and "load_pusht_slice_train_val" in msg


def test_seq_lengths_are_read_from_the_pickle(tmp_path):
    d = _write_split(tmp_path, "train", n_roll=3, T=25)
    seq, source = rot.load_seq_lengths(d)
    assert source == "seq_lengths.pkl"
    assert list(seq) == [25, 25, 25]


def test_absent_seq_lengths_falls_back_and_says_so(tmp_path):
    d = _write_split(tmp_path, "train", with_pkl=False)
    seq, source = rot.load_seq_lengths(d)
    assert seq is None
    assert "inferred" in source


def test_load_states_returns_the_source_it_used(tmp_path):
    d = _write_split(tmp_path, "train", n_roll=2, T=30)
    states, seq, source = rot.load_states(d)
    assert states.shape == (2, 30, 5)
    assert source == "seq_lengths.pkl" and list(seq) == [30, 30]
