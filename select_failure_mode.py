#!/usr/bin/env python3
"""select_failure_mode.py -- covariate-split analysis of the paired ON/OFF PushT evaluation.

The method-selection experiment (HANDOFF.md section 10.2): evaluate the straightening-ON
baseline and the straightening-OFF checkpoint on *identical* episodes, then split the
per-episode paired outcomes by covariates computable free from the eval's own artifacts.
The covariate with the largest ON-minus-OFF benefit gap names the failure mode -- measured
rather than hypothesised. This script performs the split and the paired statistics; the
gates that judge its output are pre-registered in PROGRESS_SEL.md and the thresholds are
command-line arguments with the pre-registered values as defaults, so the script cannot be
quietly re-tuned after the data is seen.

Inputs, per arm and per seed (produced by ``PLAN_ENTRY=plan_agg.py`` evaluations):

* ``agg_episode_outcomes.jsonl`` -- one JSON object per ``eval_actions`` call. The reported
  per-episode vector is the last row with ``filename == "output_final"`` (MPC's intermediate
  ``plan{N}`` rows are recorded too but are not the reported result).
* ``plan_targets.pkl`` -- the frozen ``plan.py`` dump holding ``state_0`` / ``state_g``
  (numpy, ``(n_evals, d)``) for exactly the episodes that were evaluated. Every covariate is
  computed from these two arrays and nothing else.

Covariates (the frozen list; PushT state layout: agent_x, agent_y, block_x, block_y,
block_angle -- index 4, radians, per HANDOFF.md section 7.3):

* ``req_rotation_deg``  wrap-aware |block_angle_g - block_angle_0| in degrees (PRIMARY axis)
* ``req_block_trans``   |delta(block_x, block_y)|
* ``state_goal_dist``   Euclidean distance over the four positional dims only
* ``init_agent_block_dist``  |agent_xy - block_xy| at the initial state

Splits: rotation at fixed absolute thresholds (pre-registered 15 deg primary, 30 deg
secondary, chosen from the data distribution before any outcome was seen -- PROGRESS_ROT.md
section 9.2); every other covariate at its median over the pooled episodes, computed from
covariates alone (both arms evaluate the same episodes, so one median serves both).

Paired statistics: training on this pod is bitwise deterministic and ``plan.py`` seeds
episodes from ``cfg.seed``, so the two arms at one seed drew the same episodes -- the
comparison is genuinely paired and is analysed as such (HANDOFF.md section 2.3). Success
rate differences are reported in percentage points with the paired SE
``100 * sqrt((p01 + p10) - (p01 - p10)^2) / sqrt(n)``, and discordant counts with the exact
two-sided binomial p-value (the mid-p correction is NOT used: the exact test is the
conservative one and n is small per split).

Standard library + numpy + torch only (torch solely to unpickle ``plan_targets.pkl``, which
stores the ground-truth action tensor); no hydra, no model import -- runs on the CPU dev box
and on the pod alike.

Usage::

    python select_failure_mode.py \\
        --on-globs  'plan_outputs_gd_aggw/aggw0.0_seed100_gH25' \\
                    'plan_outputs_gd_aggw/aggw0.0_seed200_gH25' \\
                    'plan_outputs_gd_aggw/aggw0.0_seed300_gH25' \\
        --off-globs 'plan_outputs_gd_offw/aggw0.0_seed100_gH25' \\
                    'plan_outputs_gd_offw/aggw0.0_seed200_gH25' \\
                    'plan_outputs_gd_offw/aggw0.0_seed300_gH25' \\
        --setting ol --out sel_outputs/selection_ol.json

One glob per seed, in matching order for both arms; the seed label is taken from the first
``seed<digits>`` match in the glob. ``--setting`` is a label recorded into the report only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: The outcome row the reported success rate comes from (agg_objectives.REPORTED_OUTCOME_FILENAME).
REPORTED_OUTCOME_FILENAME = "output_final"
EPISODE_OUTCOMES_FILENAME = "agg_episode_outcomes.jsonl"
PLAN_TARGETS_FILENAME = "plan_targets.pkl"

#: PushT state layout (HANDOFF.md section 7.3).
AGENT_X, AGENT_Y, BLOCK_X, BLOCK_Y, BLOCK_ANGLE = 0, 1, 2, 3, 4
POSITIONAL_DIMS = (AGENT_X, AGENT_Y, BLOCK_X, BLOCK_Y)

#: The frozen covariate list, in report order. req_rotation_deg is the PRIMARY axis with
#: fixed absolute splits; the rest split at their pooled median.
COVARIATE_NAMES = (
    "req_rotation_deg",
    "req_block_trans",
    "state_goal_dist",
    "init_agent_block_dist",
)
PRIMARY_COVARIATE = "req_rotation_deg"

#: Pre-registered rotation splits (PROGRESS_ROT.md section 9.2), degrees.
ROTATION_SPLITS_DEG = (15.0, 30.0)

#: Pre-registered gate defaults (PROGRESS_SEL.md). A covariate qualifies as the failure mode
#: when the benefit gap between its two sides is at least GAP_MIN_PTS points AND at least
#: GAP_MIN_SE times its own paired SE.
GAP_MIN_PTS = 8.0
GAP_MIN_SE = 2.0

_SEED_RE = re.compile(r"seed(\d+)")


# -------------------------------------------------------------------------------------------
# Pure statistics
# -------------------------------------------------------------------------------------------

def wrap_angle(delta: np.ndarray) -> np.ndarray:
    """Wrap angle differences to (-pi, pi]. Vectorised; the atan2 form is the wrap-aware one."""
    return np.arctan2(np.sin(delta), np.cos(delta))


def paired_stats(on: Sequence[bool], off: Sequence[bool]) -> Dict[str, Any]:
    """Paired comparison of two per-episode boolean outcome vectors.

    Returns the per-arm rates, the ON-minus-OFF difference in percentage points, its paired
    SE, the discordant counts and the exact two-sided binomial p-value on them.

    Raises:
        ValueError: if the vectors differ in length or are empty.
    """
    on_arr = np.asarray(on, dtype=bool)
    off_arr = np.asarray(off, dtype=bool)
    if on_arr.shape != off_arr.shape:
        raise ValueError(
            f"outcome vectors must have equal length, got {on_arr.size} vs {off_arr.size}"
        )
    n = int(on_arr.size)
    if n == 0:
        raise ValueError("outcome vectors must not be empty")

    on_only = int(np.count_nonzero(on_arr & ~off_arr))
    off_only = int(np.count_nonzero(~on_arr & off_arr))
    discordant = on_only + off_only

    delta_pts = 100.0 * (on_only - off_only) / n
    p01 = on_only / n
    p10 = off_only / n
    variance = (p01 + p10) - (p01 - p10) ** 2
    se_pts = 100.0 * math.sqrt(max(variance, 0.0) / n)

    return {
        "n": n,
        "on_rate": 100.0 * int(np.count_nonzero(on_arr)) / n,
        "off_rate": 100.0 * int(np.count_nonzero(off_arr)) / n,
        "delta_pts": delta_pts,
        "se_pts": se_pts,
        "on_only": on_only,
        "off_only": off_only,
        "matching": n - discordant,
        "mcnemar_exact_p": _binom_two_sided_exact(on_only, discordant),
    }


def _binom_two_sided_exact(k: int, n: int) -> float:
    """Exact two-sided binomial test of k successes out of n against p = 0.5.

    Sum of the probabilities of all outcomes at most as probable as the observed one. n = 0
    (no discordant episodes) returns 1.0: the data carry no evidence either way.
    """
    if n == 0:
        return 1.0
    observed = _binom_pmf(k, n)
    total = 0.0
    for i in range(n + 1):
        p_i = _binom_pmf(i, n)
        if p_i <= observed + 1e-15:
            total += p_i
    return min(total, 1.0)


def _binom_pmf(k: int, n: int) -> float:
    return math.comb(n, k) * 0.5 ** n


# -------------------------------------------------------------------------------------------
# Covariates
# -------------------------------------------------------------------------------------------

def covariates_from_states(state_0: np.ndarray, state_g: np.ndarray) -> Dict[str, np.ndarray]:
    """The frozen covariate dictionary, one array of length n_evals per covariate.

    Args:
        state_0: ``(n_evals, d)`` initial states, d >= 5, PushT layout.
        state_g: ``(n_evals, d)`` goal states, same layout.

    Raises:
        ValueError: on shape mismatch or fewer than 5 state dimensions.
    """
    state_0 = np.asarray(state_0, dtype=float)
    state_g = np.asarray(state_g, dtype=float)
    if state_0.shape != state_g.shape:
        raise ValueError(
            f"state_0 and state_g must share a shape, got {state_0.shape} vs {state_g.shape}"
        )
    if state_0.ndim != 2 or state_0.shape[1] < 5:
        raise ValueError(
            f"states must be (n_evals, d) with d >= 5 (PushT layout), got {state_0.shape}"
        )

    d_state = state_g - state_0
    req_rotation_deg = np.degrees(np.abs(wrap_angle(d_state[:, BLOCK_ANGLE])))
    req_block_trans = np.linalg.norm(d_state[:, [BLOCK_X, BLOCK_Y]], axis=1)
    state_goal_dist = np.linalg.norm(d_state[:, list(POSITIONAL_DIMS)], axis=1)
    init_agent_block_dist = np.linalg.norm(
        state_0[:, [AGENT_X, AGENT_Y]] - state_0[:, [BLOCK_X, BLOCK_Y]], axis=1
    )
    return {
        "req_rotation_deg": req_rotation_deg,
        "req_block_trans": req_block_trans,
        "state_goal_dist": state_goal_dist,
        "init_agent_block_dist": init_agent_block_dist,
    }


# -------------------------------------------------------------------------------------------
# Artifact loading
# -------------------------------------------------------------------------------------------

def _glob_one(pattern: str, what: str) -> str:
    """Resolve ``pattern`` to exactly one existing directory. Raises ValueError otherwise."""
    import glob as _glob

    matches = sorted(p for p in _glob.glob(pattern) if os.path.isdir(p))
    if len(matches) == 0:
        raise ValueError(f"{what}: no directory matches {pattern!r}")
    if len(matches) > 1:
        raise ValueError(
            f"{what}: {pattern!r} matches {len(matches)} directories; the pattern must "
            f"select exactly one (first three: {matches[:3]})"
        )
    return matches[0]


def load_reported_outcomes(run_dir: str) -> List[bool]:
    """The last ``output_final`` success vector in the run's ``agg_episode_outcomes.jsonl``.

    Raises:
        ValueError: if the file or a usable row is absent -- a missing vector must fail the
            analysis loudly rather than read as an empty success set.
    """
    path = os.path.join(run_dir, EPISODE_OUTCOMES_FILENAME)
    if not os.path.isfile(path):
        raise ValueError(
            f"{path} not found: this run was not evaluated through plan_agg.py, so no "
            f"per-episode vector exists for it."
        )
    reported: Optional[List[bool]] = None
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("filename") == REPORTED_OUTCOME_FILENAME:
                reported = [bool(v) for v in row["successes"]]
    if reported is None:
        raise ValueError(
            f"{path} holds no '{REPORTED_OUTCOME_FILENAME}' row; the reported per-episode "
            f"vector is missing."
        )
    return reported


def load_targets(run_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """``(state_0, state_g)`` numpy arrays from the run's ``plan_targets.pkl``.

    torch may be stored in the pickle (gt_actions); it is imported lazily here so importing
    this module for its statistics costs nothing.
    """
    path = os.path.join(run_dir, PLAN_TARGETS_FILENAME)
    if not os.path.isfile(path):
        raise ValueError(f"{path} not found: the evaluated episodes cannot be identified.")
    import torch  # noqa: F401  -- registers torch classes with pickle before the load.

    with open(path, "rb") as handle:
        data = pickle.load(handle)
    return np.asarray(data["state_0"], dtype=float), np.asarray(data["state_g"], dtype=float)


def _seed_label(glob_pattern: str, fallback: str) -> str:
    match = _SEED_RE.search(glob_pattern)
    return match.group(1) if match else fallback


@dataclass
class ArmSeed:
    """One arm's outcome vector + covariates for one seed, with its episode provenance."""

    seed: str
    run_dir: str
    successes: List[bool]
    covariates: Dict[str, np.ndarray]
    state_0: np.ndarray
    state_g: np.ndarray


def load_arm(globs: Sequence[str], arm_label: str) -> List[ArmSeed]:
    """Load one arm's seeds in glob order. Raises ValueError on any unusable directory."""
    seeds: List[ArmSeed] = []
    for i, pattern in enumerate(globs):
        run_dir = _glob_one(pattern, f"{arm_label} seed #{i}")
        successes = load_reported_outcomes(run_dir)
        state_0, state_g = load_targets(run_dir)
        cov = covariates_from_states(state_0, state_g)
        if len(successes) != state_0.shape[0]:
            raise ValueError(
                f"{run_dir}: {len(successes)} outcomes but {state_0.shape[0]} episodes in "
                f"plan_targets.pkl; the artifacts disagree about which episodes ran."
            )
        seeds.append(
            ArmSeed(
                seed=_seed_label(pattern, str(i)),
                run_dir=run_dir,
                successes=successes,
                covariates=cov,
                state_0=state_0,
                state_g=state_g,
            )
        )
    return seeds


def assert_episodes_identical(on_seeds: Sequence[ArmSeed], off_seeds: Sequence[ArmSeed]) -> None:
    """The pairing premise, checked rather than assumed.

    Both arms must have run the same episodes: equal seed sets, and bitwise-close initial and
    goal states per seed. The pod is bitwise deterministic, so anything beyond float-print
    tolerance means the comparison is not paired and must not be analysed as one.
    """
    on_by_seed = {s.seed: s for s in on_seeds}
    off_by_seed = {s.seed: s for s in off_seeds}
    if set(on_by_seed) != set(off_by_seed):
        raise ValueError(
            f"seed sets differ: ON {sorted(on_by_seed)} vs OFF {sorted(off_by_seed)}; "
            f"the arms are not paired."
        )
    for seed in sorted(on_by_seed):
        on_s, off_s = on_by_seed[seed], off_by_seed[seed]
        for name, a, b in (
            ("state_0", on_s.state_0, off_s.state_0),
            ("state_g", on_s.state_g, off_s.state_g),
        ):
            if a.shape != b.shape or not np.allclose(a, b, rtol=0.0, atol=1e-6):
                worst = float(np.max(np.abs(a - b))) if a.shape == b.shape else float("nan")
                raise ValueError(
                    f"seed {seed}: {name} differs between the arms (max |diff| {worst:.3g}); "
                    f"the episodes are not identical and the paired analysis is invalid."
                )


# -------------------------------------------------------------------------------------------
# Split analysis
# -------------------------------------------------------------------------------------------

@dataclass
class SplitResult:
    covariate: str
    split_value: float
    split_kind: str  # "fixed" (rotation) or "median"
    low: Dict[str, Any] = field(default_factory=dict)
    high: Dict[str, Any] = field(default_factory=dict)
    gap_pts: float = float("nan")
    gap_se_pts: float = float("nan")
    qualifies: bool = False
    reversed: bool = False  # primary covariate only: gap significantly NEGATIVE

    def as_dict(self) -> Dict[str, Any]:
        return {
            "covariate": self.covariate,
            "split_value": self.split_value,
            "split_kind": self.split_kind,
            "low": self.low,
            "high": self.high,
            "gap_pts": self.gap_pts,
            "gap_se_pts": self.gap_se_pts,
            "qualifies": self.qualifies,
            "reversed": self.reversed,
        }


def _pooled(on_seeds: Sequence[ArmSeed], off_seeds: Sequence[ArmSeed]):
    """Concatenate seeds into (on_vec, off_vec, cov_dict) in matching seed order."""
    on_vec: List[bool] = []
    off_vec: List[bool] = []
    covs: Dict[str, List[np.ndarray]] = {name: [] for name in COVARIATE_NAMES}
    off_by_seed = {s.seed: s for s in off_seeds}
    for on_s in on_seeds:
        off_s = off_by_seed[on_s.seed]
        on_vec.extend(on_s.successes)
        off_vec.extend(off_s.successes)
        for name in COVARIATE_NAMES:
            covs[name].append(on_s.covariates[name])
    return (
        on_vec,
        off_vec,
        {name: np.concatenate(arrs) for name, arrs in covs.items()},
    )


def analyse(
    on_seeds: Sequence[ArmSeed],
    off_seeds: Sequence[ArmSeed],
    rotation_splits: Sequence[float] = ROTATION_SPLITS_DEG,
    gap_min_pts: float = GAP_MIN_PTS,
    gap_min_se: float = GAP_MIN_SE,
) -> Dict[str, Any]:
    """The full selection-experiment report as a JSON-serializable dict.

    The gate: a split *qualifies* when ``gap_pts >= gap_min_pts`` AND
    ``gap_pts >= gap_min_se * gap_se_pts`` -- both clauses, exactly as pre-registered. The
    gap is high-side benefit minus low-side benefit (benefit = ON-minus-OFF paired delta),
    and the two sides are disjoint episode sets, so the gap SE is the quadrature sum of the
    side SEs.
    """
    assert_episodes_identical(on_seeds, off_seeds)
    on_vec, off_vec, covs = _pooled(on_seeds, off_seeds)

    report: Dict[str, Any] = {
        "n_seeds": len(on_seeds),
        "seeds": [s.seed for s in on_seeds],
        "n_episodes": len(on_vec),
        "overall": paired_stats(on_vec, off_vec),
        "gap_gate": {"gap_min_pts": gap_min_pts, "gap_min_se": gap_min_se},
        "splits": [],
    }

    on_arr = np.asarray(on_vec, dtype=bool)
    off_arr = np.asarray(off_vec, dtype=bool)

    def _side_stats(mask: np.ndarray) -> Dict[str, Any]:
        if not mask.any():
            raise ValueError("a split side is empty; the split is unusable")
        return paired_stats(on_arr[mask].tolist(), off_arr[mask].tolist())

    for name in COVARIATE_NAMES:
        values = covs[name]
        split_values: List[Tuple[float, str]]
        if name == PRIMARY_COVARIATE:
            split_values = [(float(v), "fixed") for v in rotation_splits]
        else:
            split_values = [(float(np.median(values)), "median")]

        for split_value, kind in split_values:
            low_mask = values <= split_value
            high_mask = ~low_mask
            result = SplitResult(
                covariate=name, split_value=split_value, split_kind=kind
            )
            result.low = _side_stats(low_mask)
            result.high = _side_stats(high_mask)
            result.gap_pts = result.high["delta_pts"] - result.low["delta_pts"]
            result.gap_se_pts = math.sqrt(
                result.high["se_pts"] ** 2 + result.low["se_pts"] ** 2
            )
            result.qualifies = (
                result.gap_pts >= gap_min_pts
                and result.gap_pts >= gap_min_se * result.gap_se_pts
            )
            if name == PRIMARY_COVARIATE:
                # Pre-registered reversal check: the ROT mechanism predicts benefit
                # GROWING with required rotation. A strongly negative gap (mirror of the
                # qualify gate) contradicts it and must be recorded, not hidden.
                result.reversed = (
                    result.gap_pts <= -gap_min_pts
                    and result.gap_pts <= -gap_min_se * result.gap_se_pts
                )
            report["splits"].append(result.as_dict())

    qualifying = [s for s in report["splits"] if s["qualifies"]]
    report["n_qualifying"] = len(qualifying)
    rot_splits = [s for s in report["splits"] if s["covariate"] == PRIMARY_COVARIATE]
    if any(s["qualifies"] for s in rot_splits):
        report["rotation_verdict"] = "CONFIRMED"
    elif any(s["reversed"] for s in rot_splits):
        report["rotation_verdict"] = "CONTRADICTED"
    else:
        report["rotation_verdict"] = "NULL"
    # The stopping rule (HANDOFF.md section 10.2), evaluated by the pre-registered gate:
    report["stopping_rule"] = (
        "STOP: no covariate split shows a qualifying benefit gap; write the analysis paper."
        if not qualifying
        else "CONTINUE: at least one covariate split shows a qualifying benefit gap."
    )
    return report


# -------------------------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Covariate-split analysis of the paired ON/OFF PushT evaluation "
        "(HANDOFF.md section 10.2). Gates pre-registered in PROGRESS_SEL.md.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--on-globs",
        nargs="+",
        required=True,
        metavar="GLOB",
        help="one directory glob per seed for the straightening-ON arm, in seed order",
    )
    p.add_argument(
        "--off-globs",
        nargs="+",
        required=True,
        metavar="GLOB",
        help="one directory glob per seed for the straightening-OFF arm, matching --on-globs",
    )
    p.add_argument(
        "--setting",
        default="ol",
        help="label recorded into the report (ol | mpc); does not change any computation",
    )
    p.add_argument(
        "--rotation-splits",
        nargs="+",
        type=float,
        default=list(ROTATION_SPLITS_DEG),
        help=f"fixed rotation split thresholds in degrees (default {list(ROTATION_SPLITS_DEG)}, "
        "pre-registered; changing them after seeing outcomes is the cardinal sin)",
    )
    p.add_argument("--gap-min-pts", type=float, default=GAP_MIN_PTS,
                   help=f"qualifying benefit gap in points (default {GAP_MIN_PTS}, pre-registered)")
    p.add_argument("--gap-min-se", type=float, default=GAP_MIN_SE,
                   help=f"gap must also be at least this many paired SEs (default {GAP_MIN_SE})")
    p.add_argument("--out", default=None, help="report JSON path (default: print only)")
    return p


def _fmt_stats(label: str, stats: Dict[str, Any]) -> str:
    return (
        f"{label:<6} n={stats['n']:>3}  ON {stats['on_rate']:5.1f}  OFF {stats['off_rate']:5.1f}  "
        f"delta {stats['delta_pts']:+6.2f} +/- {stats['se_pts']:.2f} pts  "
        f"(on_only {stats['on_only']}, off_only {stats['off_only']}, "
        f"McNemar p={stats['mcnemar_exact_p']:.3f})"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if len(args.on_globs) != len(args.off_globs):
        print(
            f"error: --on-globs has {len(args.on_globs)} entries but --off-globs has "
            f"{len(args.off_globs)}; one glob per seed, matching order",
            file=sys.stderr,
        )
        return 2

    on_seeds = load_arm(args.on_globs, "ON")
    off_seeds = load_arm(args.off_globs, "OFF")
    report = analyse(
        on_seeds,
        off_seeds,
        rotation_splits=args.rotation_splits,
        gap_min_pts=args.gap_min_pts,
        gap_min_se=args.gap_min_se,
    )
    report["setting"] = args.setting
    report["on_run_dirs"] = [s.run_dir for s in on_seeds]
    report["off_run_dirs"] = [s.run_dir for s in off_seeds]

    ov = report["overall"]
    print(f"Selection experiment -- setting={args.setting}, {report['n_seeds']} seeds, "
          f"{report['n_episodes']} paired episodes")
    print(f"  overall  ON {ov['on_rate']:.2f}  OFF {ov['off_rate']:.2f}  "
          f"paired delta {ov['delta_pts']:+.2f} +/- {ov['se_pts']:.2f} pts  "
          f"(on_only {ov['on_only']}, off_only {ov['off_only']}, "
          f"McNemar exact p={ov['mcnemar_exact_p']:.4f})")
    print()
    gate = report["gap_gate"]
    print(f"  split gate (pre-registered): gap >= {gate['gap_min_pts']:g} pts AND "
          f">= {gate['gap_min_se']:g} x paired SE")
    for split in report["splits"]:
        print(
            f"\n  {split['covariate']} {('<=' if split['split_kind'] == 'fixed' else '@ median')} "
            f"{split['split_value']:.3f}  [{split['split_kind']}]"
        )
        print("    " + _fmt_stats("low", split["low"]))
        print("    " + _fmt_stats("high", split["high"]))
        verdict = "QUALIFIES" if split["qualifies"] else "does not qualify"
        if split["reversed"]:
            verdict += " (REVERSED ordering)"
        print(
            f"    benefit gap (high - low) {split['gap_pts']:+.2f} +/- "
            f"{split['gap_se_pts']:.2f} pts -> {verdict}"
        )
    rot_verdict = report["rotation_verdict"]
    rot_text = {
        "CONFIRMED": "CONFIRMED: benefit grows with required rotation as predicted.",
        "CONTRADICTED": "CONTRADICTED: rotation ordering is reversed; the ROT mechanism "
        "prediction fails and must be reported as such.",
        "NULL": "NULL: rotation splits neither qualify nor reverse.",
    }[rot_verdict]
    print(f"\n  rotation verdict: {rot_text}")
    print(f"  {report['stopping_rule']}")

    if args.out:
        out_dir = os.path.dirname(os.path.abspath(args.out))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"\n  report written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
