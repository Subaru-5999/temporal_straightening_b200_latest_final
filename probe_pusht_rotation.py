#!/usr/bin/env python3
"""
probe_pusht_rotation.py -- does the PushT T-block actually rotate in the logged data?

`PROGRESS_ROT.md` §5 measured that the paper's trained model barely encodes `block_angle`
(best_r2 0.166 against 0.506-0.709 for the four positional dimensions) and §7 measured that
readability falls 58% over training (0.394 @8k -> 0.166 @124k). The whole "straightening trades
away rotation" direction rests on those numbers meaning something.

**They mean nothing if the block does not rotate.** §5.2 explanation (3): if the logged angular
motion is negligible, then "the representation discards orientation" is uninteresting -- there was
nothing to discard -- and the 58% fall is just an encoder correctly reallocating capacity away from a
dimension with no signal. That would make the direction void, and it is answerable from the dataset
alone.

So this script reads `states.pth` and reports, in the units the data is stored in:

  * how much the block rotates, per latent step (frameskip) and per planning horizon (25 / 50 env
    steps -- the paper's short and long goal distances);
  * the same displacement statistics for `block_x` / `block_y`, so rotation can be compared against
    the translation the model *does* encode;
  * the per-span rotation distribution, which is also the axis the within-PushT test of
    `RESEARCH_GOAL.md` §2.3 needs (per-episode required rotation versus per-episode benefit, n=150
    paired episodes instead of n=4 environments).

No checkpoint, no model, no GPU, no video decode, no hydra. Reads two tensors and does arithmetic.

**Angle differencing is wrap-aware throughout** (`atan2(sin d, cos d)`). A naive difference across a
wrap point produces a spurious full-period jump, which would inflate every rotation statistic here
and manufacture exactly the answer the direction wants. That is the same class of error as reading a
periodic variable with a linear ridge, which is what §5.1 had to rule out.

Usage
    DATASET_DIR=/workspace/arun/data python probe_pusht_rotation.py
    python probe_pusht_rotation.py --data-dir /workspace/arun/data/pusht_noise --out probe_outputs/rot_data_pusht.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

#: PushT state layout, from `datasets/pusht_dset.py`: the first two dims are proprio
#: (agent x/y), then the block pose, then optional velocities. `block_angle` is index 4.
STATE_NAMES = ("agent_x", "agent_y", "block_x", "block_y", "block_angle")
ANGLE_INDEX = 4

#: The paper's short and long goal distances, in env steps (`goal_H` in `conf/plan_gd*.yaml`).
HORIZONS = (5, 25, 50)

#: Thresholds for "did this span involve real rotation", in degrees. Reported as fractions so the
#: within-PushT axis can be split on whichever one has usable support on both sides.
DEGREE_THRESHOLDS = (5.0, 15.0, 30.0, 45.0, 90.0)


def wrapped_delta(a, b, period):
    """`b - a` mapped into (-period/2, period/2], elementwise. Wrap-aware by construction."""
    import numpy as np

    k = 2.0 * np.pi / period
    d = k * (np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64))
    return np.arctan2(np.sin(d), np.cos(d)) / k


def infer_period(theta):
    """`(period, unit)` inferred from the observed range, and recorded rather than assumed."""
    import numpy as np

    span = float(np.nanmax(np.abs(theta))) if theta.size else 0.0
    if span > 2.0 * np.pi + 1e-6:
        return 360.0, "degrees"
    return 2.0 * np.pi, "radians"


def load_states(data_dir):
    """`(states, seq_lengths_or_None)`. `states` is (n_rollouts, T, state_dim), unnormalized."""
    import torch

    path = Path(data_dir) / "states.pth"
    if not path.is_file():
        raise FileNotFoundError(
            f"no states.pth at {path}. Pass --data-dir pointing at the PushT dataset "
            f"directory (the one datasets/pusht_dset.py reads), e.g. "
            f"$DATASET_DIR/pusht_noise."
        )
    states = torch.load(path, map_location="cpu").float().numpy()

    seq = None
    for name in ("seq_lengths.pth", "seq_length.pth"):
        cand = Path(data_dir) / name
        if cand.is_file():
            seq = torch.load(cand, map_location="cpu").long().numpy()
            break
    return states, seq


def valid_length(states, seq, i):
    """Frames of rollout `i` that are real rather than padding.

    Prefers the dataset's own `seq_lengths`. Without it, trailing padding in these dumps repeats
    the last frame exactly, so the fallback trims an exactly-constant tail. Which path was taken
    is recorded in the report -- a wrong length would bias every per-step statistic.
    """
    import numpy as np

    T = states.shape[1]
    if seq is not None:
        return int(max(2, min(T, int(seq[i]))))
    row = states[i]
    last = row[-1]
    n = T
    while n > 2 and np.allclose(row[n - 2], last, atol=0.0, rtol=0.0):
        n -= 1
    return n


def summarize(values, name):
    import numpy as np

    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"name": name, "n": 0}
    return {
        "name": name,
        "n": int(v.size),
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "p90": float(np.percentile(v, 90)),
        "p99": float(np.percentile(v, 99)),
        "max": float(v.max()),
        "std": float(v.std()),
    }


def measure(states, seq, period):
    """Per-horizon rotation and translation displacement statistics."""
    import numpy as np

    n_roll = states.shape[0]
    lengths = [valid_length(states, seq, i) for i in range(n_roll)]

    theta_all, out = [], {"horizons": {}}
    for i in range(n_roll):
        theta_all.append(states[i, : lengths[i], ANGLE_INDEX])
    theta_cat = np.concatenate(theta_all) if theta_all else np.zeros(0)

    deg = 180.0 / math.pi * (2.0 * np.pi / period)   # value -> degrees

    out["angle_distribution"] = {
        **summarize(theta_cat, "block_angle"),
        "period": period,
        "circular_std_deg": float(
            math.degrees(
                math.sqrt(
                    max(0.0, -2.0 * math.log(
                        max(1e-12, abs(
                            np.mean(np.exp(1j * (2 * np.pi / period) * theta_cat))
                        ))
                    ))
                )
            )
        ) if theta_cat.size else None,
        "frames": int(theta_cat.size),
        "rollouts": int(n_roll),
    }

    for h in HORIZONS:
        rot, tx = [], []
        for i in range(n_roll):
            L = lengths[i]
            if L <= h:
                continue
            a = states[i, : L - h, ANGLE_INDEX]
            b = states[i, h:L, ANGLE_INDEX]
            rot.append(np.abs(wrapped_delta(a, b, period)) * deg)
            bx = states[i, h:L, 2] - states[i, : L - h, 2]
            by = states[i, h:L, 3] - states[i, : L - h, 3]
            tx.append(np.hypot(bx, by))
        rot = np.concatenate(rot) if rot else np.zeros(0)
        tx = np.concatenate(tx) if tx else np.zeros(0)

        entry = {
            "rotation_deg": summarize(rot, f"|d block_angle| over {h} env steps (deg)"),
            "block_translation": summarize(tx, f"|d block_xy| over {h} env steps"),
            "fraction_exceeding_deg": {
                str(t): (float((rot > t).mean()) if rot.size else None)
                for t in DEGREE_THRESHOLDS
            },
        }
        out["horizons"][str(h)] = entry
    return out


def verdict(report):
    """The pre-registered reading, applied to the numbers.

    Written as code so the threshold cannot drift between the log and the script:
    `PROGRESS_ROT.md` §8 fixes it at a median rotation over the 25-step planning horizon of
    **10 degrees**, with a **20%** floor on the fraction of spans exceeding 15 degrees.
    """
    h = report["measurements"]["horizons"].get("25")
    if not h:
        return {"verdict": "UNKNOWN", "reason": "no 25-step spans measured"}
    med = h["rotation_deg"].get("median")
    frac15 = h["fraction_exceeding_deg"].get("15.0")
    if med is None or frac15 is None:
        return {"verdict": "UNKNOWN", "reason": "insufficient data"}
    if med >= 10.0 and frac15 >= 0.20:
        v, why = "SUBSTANTIAL", "the block rotates materially over a planning horizon"
    elif med < 3.0 or frac15 < 0.05:
        v, why = "NEGLIGIBLE", ("the block barely rotates; 'the representation discards "
                                "orientation' is uninteresting and the direction is VOID")
    else:
        v, why = "MARGINAL", "rotation is present but modest; record and decide explicitly"
    return {"verdict": v, "reason": why, "median_rotation_deg_25": med,
            "fraction_over_15deg_25": frac15,
            "criterion": "SUBSTANTIAL iff median >= 10 deg AND fraction(>15 deg) >= 0.20; "
                         "NEGLIGIBLE iff median < 3 deg OR fraction(>15 deg) < 0.05"}


def build_parser():
    ap = argparse.ArgumentParser(
        description="Does the PushT T-block rotate in the logged data? Dataset-only, no GPU.")
    ap.add_argument("--data-dir", default=None,
                    help="PushT dataset directory containing states.pth "
                         "(default $DATASET_DIR/pusht_noise)")
    ap.add_argument("--out", default="probe_outputs/rot_data_pusht.json")
    return ap


def main(argv=None):
    import numpy as np  # noqa: F401  - imported for the error message if numpy is absent

    args = build_parser().parse_args(argv)
    data_dir = args.data_dir
    if data_dir is None:
        root = os.environ.get("DATASET_DIR")
        if not root:
            print("ERROR: set DATASET_DIR or pass --data-dir.", file=sys.stderr)
            return 2
        data_dir = str(Path(root) / "pusht_noise")

    states, seq = load_states(data_dir)
    if states.ndim != 3 or states.shape[-1] <= ANGLE_INDEX:
        print(f"ERROR: states.pth has shape {states.shape}; expected "
              f"(rollouts, frames, >={ANGLE_INDEX + 1}).", file=sys.stderr)
        return 2

    period, unit = infer_period(states[..., ANGLE_INDEX])
    report = {
        "probe": "probe_pusht_rotation.py",
        "data_dir": str(data_dir),
        "states_shape": list(states.shape),
        "seq_lengths_source": "seq_lengths.pth" if seq is not None
        else "inferred from constant tail",
        "angle_period": period,
        "angle_unit": unit,
        "state_names": list(STATE_NAMES),
        "measurements": measure(states, seq, period),
    }
    report["gate"] = verdict(report)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("=" * 78)
    print(f"PUSHT ROTATION IN THE LOGGED DATA   {data_dir}")
    print("=" * 78)
    print(f"  states               : {tuple(states.shape)}  "
          f"(angle at index {ANGLE_INDEX}, {unit}, period {period:g})")
    print(f"  seq_lengths          : {report['seq_lengths_source']}")
    d = report["measurements"]["angle_distribution"]
    print(f"  block_angle spread   : std {d.get('std'):.4f} {unit}, "
          f"circular std {d.get('circular_std_deg'):.2f} deg, "
          f"{d.get('frames')} frames over {d.get('rollouts')} rollouts")
    print()
    print(f"  {'horizon':>8}{'rot median':>12}{'rot p90':>10}{'rot max':>10}"
          f"{'>15deg':>9}{'>30deg':>9}{'block |dxy| med':>17}")
    for h in HORIZONS:
        e = report["measurements"]["horizons"][str(h)]
        r, t = e["rotation_deg"], e["block_translation"]
        f15 = e["fraction_exceeding_deg"].get("15.0")
        f30 = e["fraction_exceeding_deg"].get("30.0")
        print(f"  {h:>8}{r.get('median', float('nan')):>12.2f}"
              f"{r.get('p90', float('nan')):>10.2f}{r.get('max', float('nan')):>10.2f}"
              f"{(f15 if f15 is not None else float('nan')):>9.3f}"
              f"{(f30 if f30 is not None else float('nan')):>9.3f}"
              f"{t.get('median', float('nan')):>17.3f}")
    print()
    g = report["gate"]
    print(f"  criterion            : {g['criterion']}")
    print(f"  VERDICT              : {g['verdict']}  -- {g['reason']}")
    print()
    print(f"Report written to {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
