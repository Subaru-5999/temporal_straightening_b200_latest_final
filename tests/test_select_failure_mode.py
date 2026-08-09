"""Tests for select_failure_mode.py, the selection-experiment analysis (HANDOFF.md §10.2).

Built on synthetic fixtures with known ground truth, CPU-only, no hydra and no planning
stack: each fake run directory holds the two artifacts a real plan_agg.py evaluation
produces -- plan_targets.pkl and agg_episode_outcomes.jsonl.

The wrap-awareness tests follow the discipline of tests/test_pusht_rotation_probe.py: a
naive (unwrapped) angle difference would give the wrong covariate across the period
boundary, and the test records that.
"""

import json
import math
import os
import pickle

import numpy as np
import pytest
import torch

import select_failure_mode as sfm


# -------------------------------------------------------------------------------------------
# Fixture builders
# -------------------------------------------------------------------------------------------

def make_states(angles_0, angles_g, n=None, xy_scale=100.0, seed=0):
    """(state_0, state_g) arrays in PushT layout from angle specs and random positions."""
    rs = np.random.RandomState(seed)
    if n is None:
        n = len(angles_0)
    state_0 = rs.rand(n, 5) * xy_scale
    state_g = rs.rand(n, 5) * xy_scale
    state_0[:, sfm.BLOCK_ANGLE] = np.asarray(angles_0, dtype=float)
    state_g[:, sfm.BLOCK_ANGLE] = np.asarray(angles_g, dtype=float)
    return state_0, state_g


def write_run_dir(tmp_path, name, state_0, state_g, successes, extra_rows=()):
    """Write one fake eval run directory with both artifacts. Returns its path."""
    run_dir = tmp_path / name
    run_dir.mkdir(parents=True)
    targets = {
        "obs_0": np.zeros((state_0.shape[0], 1)),
        "obs_g": np.zeros((state_g.shape[0], 1)),
        "state_0": state_0,
        "state_g": state_g,
        "gt_actions": torch.zeros(state_0.shape[0], 5, 10),  # real pickles store tensors
        "goal_H": 25,
    }
    with open(run_dir / sfm.PLAN_TARGETS_FILENAME, "wb") as handle:
        pickle.dump(targets, handle)
    rows = list(extra_rows) + [
        {
            "filename": sfm.REPORTED_OUTCOME_FILENAME,
            "plan_call": 0,
            "n_evals": len(successes),
            "successes": [bool(s) for s in successes],
        }
    ]
    with open(run_dir / sfm.EPISODE_OUTCOMES_FILENAME, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return str(run_dir)


# -------------------------------------------------------------------------------------------
# Statistics
# -------------------------------------------------------------------------------------------

def test_wrap_angle_recovers_small_rotation_across_the_period_boundary():
    # 3.05 rad -> -3.05 rad crosses pi; the true change is 0.18... rad, not ~6.1.
    delta = np.array([-3.05 - 3.05])
    wrapped = sfm.wrap_angle(delta)
    assert wrapped[0] == pytest.approx(2 * math.pi - 6.10, abs=1e-9)
    naive = abs(-3.05 - 3.05)
    assert naive > 6.0 and wrapped[0] < 0.2  # the naive form is the recorded hazard


def test_wrap_angle_zero_and_pi():
    assert sfm.wrap_angle(np.array([0.0]))[0] == pytest.approx(0.0)
    assert abs(sfm.wrap_angle(np.array([math.pi]))[0]) == pytest.approx(math.pi)


def test_paired_stats_known_table():
    # 10 episodes: ON wins 3 flips, loses 1, 6 agree.
    on = [True, True, True, True, False, False, False, True, True, False]
    off = [False, False, False, True, False, True, False, True, True, False]
    stats = sfm.paired_stats(on, off)
    assert stats["on_only"] == 3
    assert stats["off_only"] == 1
    assert stats["matching"] == 6
    assert stats["delta_pts"] == pytest.approx(20.0)  # 100 * (3-1)/10
    assert stats["on_rate"] == pytest.approx(60.0)
    assert stats["off_rate"] == pytest.approx(40.0)
    # SE: p01=0.3, p10=0.1 -> sqrt((0.4 - 0.04)/10) * 100
    assert stats["se_pts"] == pytest.approx(100 * math.sqrt(0.36 / 10))
    # Exact McNemar on 3 vs 1 out of 4 discordant: 2 * (C(4,0)+C(4,1)) / 16 = 0.625
    assert stats["mcnemar_exact_p"] == pytest.approx(0.625)


def test_paired_stats_no_discordant_pairs():
    on = [True, False, True]
    stats = sfm.paired_stats(on, list(on))
    assert stats["delta_pts"] == 0.0
    assert stats["se_pts"] == 0.0
    assert stats["mcnemar_exact_p"] == 1.0


def test_paired_stats_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="equal length"):
        sfm.paired_stats([True, False], [True])


def test_binom_exact_known_values():
    assert sfm._binom_two_sided_exact(0, 0) == 1.0
    assert sfm._binom_two_sided_exact(5, 10) == pytest.approx(1.0)  # mode, everything counts
    # k=0, n=10: outcomes {0,10} are at most as probable as 0 -> 2/1024
    assert sfm._binom_two_sided_exact(0, 10) == pytest.approx(2 / 1024)


# -------------------------------------------------------------------------------------------
# Covariates
# -------------------------------------------------------------------------------------------

def test_covariates_hand_computed():
    state_0 = np.array([[10.0, 20.0, 30.0, 40.0, 0.0]])
    state_g = np.array([[13.0, 20.0, 34.0, 40.0, 0.5]])
    cov = sfm.covariates_from_states(state_0, state_g)
    assert cov["req_rotation_deg"][0] == pytest.approx(math.degrees(0.5))
    assert cov["req_block_trans"][0] == pytest.approx(4.0)
    # positional dims: agent moved 3 in x, block 4 in x -> dist 5
    assert cov["state_goal_dist"][0] == pytest.approx(5.0)
    assert cov["init_agent_block_dist"][0] == pytest.approx(math.hypot(20.0, 20.0))


def test_covariates_rotation_is_wrap_aware():
    state_0, state_g = make_states([3.05], [-3.05])
    cov = sfm.covariates_from_states(state_0, state_g)
    expected = math.degrees(2 * math.pi - 6.10)
    assert cov["req_rotation_deg"][0] == pytest.approx(expected, abs=1e-6)
    # The naive reading would be ~350 degrees; the wrap-aware one is ~10.3.
    assert cov["req_rotation_deg"][0] < 15.0


def test_covariates_reject_short_state():
    with pytest.raises(ValueError, match="d >= 5"):
        sfm.covariates_from_states(np.zeros((3, 4)), np.zeros((3, 4)))


def test_covariates_reject_shape_mismatch():
    with pytest.raises(ValueError, match="share a shape"):
        sfm.covariates_from_states(np.zeros((3, 5)), np.zeros((4, 5)))


# -------------------------------------------------------------------------------------------
# Artifact loading
# -------------------------------------------------------------------------------------------

def test_load_reported_outcomes_uses_the_last_output_final_row(tmp_path):
    state_0, state_g = make_states([0.0], [0.1])
    run_dir = write_run_dir(
        tmp_path, "run_a", state_0, state_g, [True],
        extra_rows=(
            {"filename": "plan0", "plan_call": 0, "n_evals": 1, "successes": [False]},
            {"filename": "output_final", "plan_call": 0, "n_evals": 1,
             "successes": [False]},  # an earlier final row must be superseded
        ),
    )
    # The appended writer puts extra_rows first, so the REAL final row is last.
    assert sfm.load_reported_outcomes(run_dir) == [True]


def test_load_reported_outcomes_missing_file(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        sfm.load_reported_outcomes(str(tmp_path))


def test_load_reported_outcomes_no_final_row(tmp_path):
    run_dir = tmp_path / "run_b"
    run_dir.mkdir()
    with open(run_dir / sfm.EPISODE_OUTCOMES_FILENAME, "w", encoding="utf-8") as handle:
        handle.write(json.dumps({"filename": "plan0", "n_evals": 1,
                                 "successes": [True]}) + "\n")
    with pytest.raises(ValueError, match="no 'output_final' row"):
        sfm.load_reported_outcomes(str(run_dir))


def test_load_targets_roundtrip_with_torch_tensor(tmp_path):
    state_0, state_g = make_states([0.1, 0.2], [0.3, 0.4])
    run_dir = write_run_dir(tmp_path, "run_c", state_0, state_g, [True, False])
    s0, sg = sfm.load_targets(run_dir)
    np.testing.assert_allclose(s0, state_0)
    np.testing.assert_allclose(sg, state_g)


def test_glob_one_requires_exactly_one_match(tmp_path):
    (tmp_path / "a1").mkdir()
    (tmp_path / "a2").mkdir()
    with pytest.raises(ValueError, match="exactly one"):
        sfm._glob_one(str(tmp_path / "a*"), "test")
    with pytest.raises(ValueError, match="no directory matches"):
        sfm._glob_one(str(tmp_path / "zz*"), "test")


def test_seed_label_extraction():
    assert sfm._seed_label("plan_outputs/aggw0.0_seed200_gH25", "x") == "200"
    assert sfm._seed_label("no-seed-here", "fallback") == "fallback"


# -------------------------------------------------------------------------------------------
# End-to-end analysis
# -------------------------------------------------------------------------------------------

def _two_arm_fixture(tmp_path, on_success_fn, off_success_fn, rotations):
    """Two arms x two seeds x len(rotations) episodes. Returns (on_globs, off_globs)."""
    on_globs, off_globs = [], []
    for seed, tag in ((100, "s100"), (200, "s200")):
        angles_0 = [0.0] * len(rotations)
        angles_g = [math.radians(r) for r in rotations]
        state_0, state_g = make_states(angles_0, angles_g, seed=seed)
        on_success = [on_success_fn(i, seed) for i in range(len(rotations))]
        off_success = [off_success_fn(i, seed) for i in range(len(rotations))]
        on_dir = write_run_dir(
            tmp_path, f"on_seed{tag}", state_0, state_g, on_success
        )
        off_dir = write_run_dir(
            tmp_path, f"off_seed{tag}", state_0, state_g, off_success
        )
        on_globs.append(on_dir)
        off_globs.append(off_dir)
    return on_globs, off_globs


def test_analyse_detects_a_qualifying_rotation_gap(tmp_path):
    # 40 episodes per arm: rotations 1..40 deg. ON always succeeds; OFF succeeds only on
    # low-rotation episodes (i < 20, i.e. rotations <= 20). The ON-minus-OFF benefit is
    # concentrated on the HIGH side of the 15 deg split, so high-minus-low gap is strongly
    # positive and must qualify under the pre-registered gate.
    rotations = [float(r) for r in range(1, 41)]

    def on_fn(i, seed):
        return True

    def off_fn(i, seed):
        return i < 20

    on_globs, off_globs = _two_arm_fixture(tmp_path, on_fn, off_fn, rotations)
    report = sfm.analyse(sfm.load_arm(on_globs, "ON"), sfm.load_arm(off_globs, "OFF"))

    assert report["n_episodes"] == 80
    # ON succeeds on all 80 episodes; OFF succeeds on 20 of 40 per seed -> 40/80.
    assert report["overall"]["delta_pts"] == pytest.approx(100.0 * (80 - 40) / 80)

    rot15 = next(
        s for s in report["splits"]
        if s["covariate"] == "req_rotation_deg" and s["split_value"] == 15.0
    )
    # low side: rotations <= 15 -> episodes 1..15: ON 15/15, OFF 15/15 -> delta 0
    assert rot15["low"]["delta_pts"] == pytest.approx(0.0)
    # high side: rotations 16..40; OFF still succeeds on 16..20 -> delta (25-5)/25 = 80
    assert rot15["high"]["delta_pts"] == pytest.approx(80.0)
    assert rot15["gap_pts"] == pytest.approx(80.0)
    assert rot15["qualifies"] is True
    assert report["n_qualifying"] >= 1
    assert report["rotation_verdict"] == "CONFIRMED"
    assert "CONTINUE" in report["stopping_rule"]


def test_analyse_flags_reversed_rotation_ordering(tmp_path):
    # The mirror of the qualifying case: OFF fails only on LOW-rotation episodes, so the
    # ON-minus-OFF benefit is concentrated on the LOW side and every rotation gap is
    # strongly negative. Pre-registered reading: the ROT mechanism prediction is
    # CONTRADICTED and must be recorded -- not absorbed into a generic STOP.
    rotations = [float(r) for r in range(1, 41)]
    on_globs, off_globs = _two_arm_fixture(
        tmp_path, lambda i, s: True, lambda i, s: i >= 20, rotations
    )
    report = sfm.analyse(sfm.load_arm(on_globs, "ON"), sfm.load_arm(off_globs, "OFF"))

    rot15 = next(
        s for s in report["splits"]
        if s["covariate"] == "req_rotation_deg" and s["split_value"] == 15.0
    )
    # low side: rotations <= 15: ON 15/15, OFF 0/15 -> delta +100
    assert rot15["low"]["delta_pts"] == pytest.approx(100.0)
    # high side: rotations 16..40: OFF succeeds on 21..40 -> delta (25-5)/25 = +20
    assert rot15["high"]["delta_pts"] == pytest.approx(20.0)
    assert rot15["gap_pts"] == pytest.approx(-80.0)
    assert rot15["qualifies"] is False
    assert rot15["reversed"] is True
    assert report["n_qualifying"] == 0
    assert report["rotation_verdict"] == "CONTRADICTED"
    assert report["stopping_rule"].startswith("STOP")


def test_analyse_stops_when_no_gap_qualifies(tmp_path):
    # Uniform benefit: OFF fails everywhere, ON succeeds everywhere -> delta +100 on BOTH
    # sides of every split, gap 0 everywhere. The stopping rule must fire.
    rotations = [float(r) for r in range(1, 41)]
    on_globs, off_globs = _two_arm_fixture(
        tmp_path, lambda i, s: True, lambda i, s: False, rotations
    )
    report = sfm.analyse(sfm.load_arm(on_globs, "ON"), sfm.load_arm(off_globs, "OFF"))
    for split in report["splits"]:
        assert split["gap_pts"] == pytest.approx(0.0)
        assert split["qualifies"] is False
    assert report["n_qualifying"] == 0
    assert report["rotation_verdict"] == "NULL"
    assert report["stopping_rule"].startswith("STOP")


def test_analyse_rejects_unpaired_episodes(tmp_path):
    # Same seeds, but the OFF arm evaluated different initial states: the pairing premise
    # is false and the analysis must refuse rather than compute.
    rotations = [5.0, 10.0, 20.0, 30.0]
    on_globs, off_globs = _two_arm_fixture(
        tmp_path, lambda i, s: True, lambda i, s: False, rotations
    )
    on_seeds = sfm.load_arm(on_globs, "ON")
    off_seeds = sfm.load_arm(off_globs, "OFF")
    off_seeds[0].state_0 = off_seeds[0].state_0 + 1.0  # break episode identity
    with pytest.raises(ValueError, match="not identical"):
        sfm.analyse(on_seeds, off_seeds)


def test_analyse_rejects_mismatched_seed_sets(tmp_path):
    rotations = [5.0, 10.0, 20.0, 30.0]
    on_globs, off_globs = _two_arm_fixture(
        tmp_path, lambda i, s: True, lambda i, s: False, rotations
    )
    on_seeds = sfm.load_arm(on_globs, "ON")
    off_seeds = sfm.load_arm(off_globs[:1], "OFF")  # drop a seed
    with pytest.raises(ValueError, match="seed sets differ"):
        sfm.analyse(on_seeds, off_seeds)


def test_median_splits_are_outcome_blind_and_labelled(tmp_path):
    rotations = [float(r) for r in range(1, 41)]
    on_globs, off_globs = _two_arm_fixture(
        tmp_path, lambda i, s: i % 2 == 0, lambda i, s: i % 3 == 0, rotations
    )
    report = sfm.analyse(sfm.load_arm(on_globs, "ON"), sfm.load_arm(off_globs, "OFF"))
    median_splits = [s for s in report["splits"] if s["split_kind"] == "median"]
    assert len(median_splits) == 3  # every covariate except rotation gets one median split
    trans = next(s for s in median_splits if s["covariate"] == "req_block_trans")
    # The median is computed from covariates alone -- here from the pooled episode set.
    assert trans["low"]["n"] + trans["high"]["n"] == report["n_episodes"]


def test_cli_end_to_end(tmp_path, capsys):
    rotations = [float(r) for r in range(1, 41)]
    on_globs, off_globs = _two_arm_fixture(
        tmp_path, lambda i, s: True, lambda i, s: i < 20, rotations
    )
    out_path = str(tmp_path / "sel_outputs" / "report.json")
    rc = sfm.main(
        ["--on-globs", *on_globs, "--off-globs", *off_globs,
         "--setting", "ol", "--out", out_path]
    )
    assert rc == 0
    assert os.path.isfile(out_path)
    with open(out_path, encoding="utf-8") as handle:
        report = json.load(handle)
    assert report["setting"] == "ol"
    assert report["gap_gate"]["gap_min_pts"] == sfm.GAP_MIN_PTS
    captured = capsys.readouterr()
    assert "Selection experiment" in captured.out


def test_cli_rejects_unequal_glob_counts(capsys):
    rc = sfm.main(["--on-globs", "a", "b", "--off-globs", "a"])
    assert rc == 2
