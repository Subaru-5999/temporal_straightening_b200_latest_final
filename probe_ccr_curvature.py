#!/usr/bin/env python3
"""
probe_ccr_curvature.py  --  rung 1 of the escalation ladder (Requirement 11.3).

It answers one question, before a single GPU hour is spent: does perturbing the
actions actually bend the imagined latent trajectory more than the recorded
actions do? If that gap is ~0 there is nothing for CCR to fix, and no Pilot_Run
should be launched.

Standalone, read-only and CPU-only:

  * no optimizer is constructed anywhere in this file, every tensor op runs under
    `torch.no_grad()` and the model is put in `eval()` (Requirement 7.1);
  * the checkpoint is hashed before and after the measurement and the run
    directory is never written to -- reports go to `probe_outputs/`
    (Requirement 7.4);
  * a missing `--ckpt` / `--train-cfg` exits 1 with the absolute path *before*
    any model is constructed or any weight is loaded (Requirement 7.5);
  * `--max-minutes` is a wall-clock guard: it stops sampling, marks the report
    `partial: true` and exits 0 (Requirement 7.6).

Usage (task 13.1):

    python probe_ccr_curvature.py \
      --ckpt   checkpoints/test/pusht_aggmlpcos1e-1_.../checkpoints/model_2.pth \
      --train-cfg checkpoints/test/pusht_aggmlpcos1e-1_.../hydra.yaml \
      --rho 0.05 --rollout-len 5 --action-source synthetic \
      --num-windows 64 --draws 4 --reference pristine --max-minutes 30 \
      --out probe_outputs/ccr_pusht.json

The probe measures **the arm that will be trained**: it calls the training
model's own `_ccr_actions`, `_sample_action_perturbation`, `_rollout_latents`,
`visual_only` and `compute_ccr`, so `--action-source logged` is rejected with the
exact message training gives when `num_hist + L - 1 > num_frames`. Nothing about
the curvature definition is reimplemented here.

Readouts are reported **per state dimension** (Requirement 7.2). Aggregating over
all windows is the documented mistake of `SHORT_BUDGET_PILOTS.md` section 4,
where aggregate `probe_r2` improved 0.244 -> 0.280 while `agent_x` collapsed
0.943 -> -0.011. Same data, opposite conclusions. The per-dimension breakdown is
the whole point of this file.

Probe gate, written down before running (Requirement 8.1): the mechanism is
present if the aggregate `curvature_gap` is positive AND, on at least 3 of the 5
disaggregated dimensions, the gap is positive and at least 20% of that
dimension's unperturbed curvature magnitude. The verdict is printed as an
explicit PASS/FAIL so the operator does not have to remember the rule.

Third-party imports (torch, numpy, hydra, omegaconf) are deliberately *lazy*:
path validation must be able to fail fast, and does so without importing an ML
stack. numpy only -- no sklearn; the ridge regression is a 6-line closed form.

`--readout` (ACS task 4.1) selects what is measured. Its default, `curvature`, is
everything described above -- `curvature_gap` and `state_readout_r2` against a
reference, gated by the written probe gate -- so an invocation that predates the
flag behaves exactly as it did.

`--readout actions` is a different, cheaper thing: the Stage-0 action-similarity
readout of the action-conditioned-straightening spec (ACS Requirement 1). It needs
no checkpoint, builds no model, allocates no GPU and decodes no video. It composes
each environment's config from `conf/train.yaml`, reads the dataset's action tensor
directly and reports the distribution of `cos(a_t, a_{t+1})` and of the gate weight
`w` per environment per action reduction:

    python probe_ccr_curvature.py --readout actions --env pusht \
      --acs-action-reduce all --split train \
      --out probe_outputs/acs_actions_pusht.json

Two things about that readout are load-bearing rather than incidental:

  * it does **not** go through `load_windows`, whose `state_dim` guard is correct
    for the readouts it protects and PushT-specific (Wall's `state_dim` is 4, so
    reusing it would raise before measuring anything -- ACS error case E14); and
  * it does **not** go through `dset[idx]`, which routes PushT through
    `PushTDataset.get_frames`, opens a `VideoReader` and decodes `num_frames *
    frameskip` frames per window. It reads the underlying dataset's action tensor
    plus `dset.slices`, `dset.frameskip` and `dset.num_frames` and applies the same
    `rearrange("(n f) d -> n (f d)")`, which is what keeps Stage 0 in minutes.

`a_t` and `w_t` come from the **shipped** `VWorldModel.reduce_action` and
`VWorldModel.action_gate`. There is deliberately no cosine of actions anywhere in
this file: Stage 0's prediction and training's measurement have to be the same
number, and the only way to guarantee that is for them to be the same code (ACS
Requirements 15.1, 15.2, Property 19).

`--readout actions --summarize <reports...>` (ACS task 4.3) reads those per-env
reports back and evaluates the **pre-registered** Stage-0 verdict rules A and B on
them, emitting one combined verdict JSON plus a printed verdict:

    python probe_ccr_curvature.py --readout actions \
      --summarize probe_outputs/acs_actions_*.json \
      --table1-gains "umaze=50.00,medium=10.67,wall=10.67,pusht=7.33" \
      --out probe_outputs/acs_stage0_verdict.json

Every threshold it applies -- `1.5x`, `1.1x`, `0.15`, `0.08` -- is reproduced from
`PROGRESS_ACS.md` section 4, which was written before the data was collected, and is
a judgment call rather than a derivation (ACS Requirement 2.17). The rules live in
`rule_a_verdict`, `rule_b_verdict`, `combine_rule_verdicts` and `evaluate_stage0`,
which are **pure functions over plain statistic dicts**: no file, no dataset, no
tensor. That is what lets the boundaries be unit-tested at, just below and just
above each threshold, and it is why the rule cannot be quietly refitted to the
numbers it is judging.
"""
import argparse
import glob as globlib
import hashlib
import json
import logging
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("probe_ccr")

# PushT `state` layout (datasets/pusht_dset.py): dims 0-4 are the pose, dims 5-6
# are the agent's velocities and are NOT reported (with_velocity appends them).
STATE_DIM_NAMES = ("agent_x", "agent_y", "block_x", "block_y", "block_angle")

# --- probe gate (design section 10 / Requirement 8.1) -------------------------
GATE_MIN_RATIO = 0.20   # gap must be >= 20% of the unperturbed curvature magnitude
GATE_MIN_DIMS = 3       # ... on at least 3 of the 5 dimensions

# Windows in the top tercile of |state_d[last] - state_d[first]| are the windows in
# which dimension d is the dominant motion. That is the subset on which a
# per-dimension curvature statistic means anything.
TERCILE = 3

# Fixed seed: the window sample must be identical run to run, or two probes are
# not comparable.
PROBE_SEED = 0

# Ridge readout. Fixed, documented constants rather than a CLI knob, so two
# probes never disagree because one of them tuned the regularizer.
RIDGE_ALPHA = 1.0
RIDGE_TRAIN_FRACTION = 0.7   # split by *window*, so frames of one window never straddle it

# Requirement 7.3: cheapest first. `pristine` is free (hub cache + fresh heads),
# `early_telemetry` is a free matched-budget control read off the reference run's
# own JSONL, `control_run` is the last resort.
REFERENCE_SOURCES = ("pristine", "early_telemetry", "control_run")
EARLY_TELEMETRY_MAX_ITER = 8000   # "the reference run's own first-8,000-step rows"
TELEMETRY_BASENAME = "training_log.jsonl"

# Wall-clock split of `--max-minutes`: the measurement of the checkpoint under
# test comes first and gets the larger share; the reference measurement gets what
# is left, minus a small tail for the ridge fit and the report write.
MAIN_BUDGET_FRACTION = 0.60
REFERENCE_BUDGET_FRACTION = 0.95

DEFAULT_OUT = "probe_outputs/ccr_probe.json"

RULE = "=" * 78
THIN = "-" * 78


# --- `--readout` (ACS task 4.1) -----------------------------------------------
# `curvature` is the pre-existing pair of readouts (`curvature_gap` +
# `state_readout_r2`) and is the default, so an invocation written before this flag
# existed runs exactly what it ran before. `actions` is the Stage-0
# action-similarity readout, which needs no checkpoint at all.
READOUTS = ("curvature", "actions")
DEFAULT_READOUT = "curvature"

# The four environments Stage 0 measures (ACS Requirement 1.1). `save_name` differs
# from `name` for the mazes (umaze / medium); these are the `env=` config-group
# names, which is what `--env` takes.
ACS_ENVS = ("pusht", "wall", "point_maze", "point_maze_medium")

# Protocol values, fixed rather than exposed: ACS Requirement 1.13 pins the Stage-0
# composition to `num_hist=3`, `num_pred=1`, `frameskip=5`, which is what training
# and every Table-1 cell use. A flag here would let two Stage-0 runs disagree.
ACS_NUM_HIST = 3
ACS_NUM_PRED = 1
ACS_FRAMESKIP = 5

# `VWorldModel.__init__` takes an `image_size`, and the (reduction, gate) instances this
# readout builds are dummy-module shells whose encoder is never called, so the value is
# inert. It is the repo's `img_size` anyway, so nothing here reads as a special case.
ACS_IMAGE_SIZE_STUB = 224

# Mirrors `models.visual_world_model.ACS_ACTION_REDUCTIONS` for argparse only, so
# `--help` renders without importing torch (the whole module keeps its ML imports
# lazy). The shipped `reduce_action` validates the value that actually executes, so
# a drift here surfaces as its ValueError rather than as a silent wrong reduction.
ACS_ACTION_REDUCTIONS = ("sum", "raw", "first")
# `permuted` is deliberately absent: it is the training-time null control, and
# shuffling `w` across triples measures nothing about a dataset (§11.3). Stage 0
# measures the real gate.
ACS_GATES = ("relu_cos", "affine_cos", "hard")
DEFAULT_ACS_GATE = "relu_cos"

# CLI split names and the dataset-dict keys they map to. `load_*_slice_train_val`
# returns `{"train": ..., "valid": ...}`; the requirements say "validation".
ACS_SPLITS = {"train": "train", "validation": "valid"}

# 20 bins over [-1, 1] (ACS Requirement 1.6), so the shape of the distribution is on
# the record and not just its first two moments.
COS_HIST_BINS = 20
COS_HIST_RANGE = (-1.0, 1.0)

# Windows per gather. Bounds peak memory at `chunk * num_frames * frameskip *
# action_dim` floats without changing any statistic: every readout below is
# accumulated over the full slice list.
ACS_CHUNK_WINDOWS = 65536

# `probe_outputs/`, one JSON per environment (ACS Requirement 1.18).
ACS_OUT_DIR = "probe_outputs"
ACS_OUT_PREFIX = "acs_actions"


# --- Stage-0 verdict rules (ACS task 4.3) -------------------------------------
# The four rule keys. They are the *rule's* names for the environments, which is
# what `--table1-gains` and PROGRESS_ACS.md section 4 use; `ACS_ENV_RULE_KEYS` maps
# the `env=` config-group names onto them, so `point_maze` is read as `umaze`.
ACS_RULE_ENV_KEYS = ("pusht", "wall", "umaze", "medium")
ACS_ENV_RULE_KEYS = {
    "pusht": "pusht",
    "wall": "wall",
    "point_maze": "umaze",
    "point_maze_medium": "medium",
}

VERDICT_GO = "GO"
VERDICT_MIDDLE = "MIDDLE"
VERDICT_STOP = "STOP"
VERDICTS = (VERDICT_GO, VERDICT_MIDDLE, VERDICT_STOP)
# Ordering only, so "cap this verdict at MIDDLE" is one `min` rather than a chain of
# ifs. Higher is more permissive.
VERDICT_SEVERITY = {VERDICT_STOP: 0, VERDICT_MIDDLE: 1, VERDICT_GO: 2}

# PRE-REGISTERED THRESHOLDS. Written 2026-08-08 in PROGRESS_ACS.md section 4,
# before the Stage-0 statistics were collected (ACS Requirements 2.1, 2.17), and
# reproduced here verbatim. They are judgment calls, not derivations: `1.5x` is
# "clearly separated rather than marginally separated", `1.1x` is
# "indistinguishable", `0.08` is the point below which the reallocated mass is too
# small to plausibly move a +4/+5 bar given that the whole first-order straightening
# effect was +7.33, and `0.15` is roughly twice that.
#
# DO NOT TUNE THESE AGAINST MEASURED DATA. An arbitrary threshold fixed in advance
# is a test; the same threshold chosen afterwards is a fit, and that is the
# documented CCR failure mode (PROGRESS_CCR.md sections 5a, 6a).
RULE_A_CLEAR_MARGIN = 1.5          # GO needs PushT >= 1.5x each of the other three
RULE_A_INDISTINGUISHABLE = 1.1     # within 1.1x of the smoothest is a STOP
RULE_B_GO = 0.15                   # R >= 0.15 is a GO on rule B
RULE_B_MIDDLE = 0.08               # 0.08 <= R < 0.15 is a MIDDLE; below is a STOP

# The verdict is read off the `train` split (that is what training sees) and off the
# `sum` reduction (the shipped default, `acs_action_reduce=sum`). `raw` is the
# cross-measurement Requirement 3.6 compares against: reversals that only exist
# under `raw` happen *inside* a latent step, where the latent velocity cannot see
# them either.
ACS_HEADLINE_REDUCTION = "sum"
ACS_WITHIN_STEP_REDUCTION = "raw"

# Table 1's straightening gains (`L_curv` off -> on, open-loop), the ordering rule A
# is set against. A default rather than a required flag: these are published numbers,
# not a knob, and `--table1-gains` exists so the recorded command self-describes.
ACS_TABLE1_GAINS_DEFAULT = "umaze=50.00,medium=10.67,wall=10.67,pusht=7.33"

ACS_VERDICT_SCHEMA = "acs_stage0_verdict/1"
ACS_VERDICT_BASENAME = "acs_stage0_verdict.json"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _non_negative_float(raw):
    value = float(raw)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {value}")
    return value


def _positive_int(raw):
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {value}")
    return value


def build_parser():
    ap = argparse.ArgumentParser(
        description="Read-only CPU probe: curvature of imagined rollouts under "
                    "unperturbed vs perturbed actions, disaggregated per state "
                    "dimension, with a reference value for every readout.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--readout", choices=READOUTS, default=DEFAULT_READOUT,
                    help=f"what to measure (default {DEFAULT_READOUT}): 'curvature' is "
                         f"curvature_gap + state_readout_r2 against a reference, i.e. "
                         f"this script's original behaviour and everything the flags "
                         f"below describe; 'actions' is the Stage-0 action-similarity "
                         f"readout, which takes no checkpoint and builds no model")
    # Not `required=True` any more, because `--readout actions` has no checkpoint to
    # take. `main` reinstates the requirement for every other readout through
    # `parser.error`, which prints the same message and exits 2 exactly as argparse
    # did; only the usage line's brackets change.
    ap.add_argument("--ckpt",
                    help="path to model_<epoch>.pth (read-only; never modified). "
                         "Required unless --readout actions")
    ap.add_argument("--train-cfg",
                    help="path to the run's resolved hydra.yaml. Required unless "
                         "--readout actions")
    ap.add_argument("--rho", type=_non_negative_float, default=0.05,
                    help="perturbation radius in normalized-action units (default 0.05)")
    ap.add_argument("--rollout-len", type=_positive_int, default=5,
                    help="imagined predictor steps L; default 5 == Planner_Horizon")
    ap.add_argument("--action-source", choices=("logged", "synthetic"),
                    default="synthetic",
                    help="which arm to measure (default synthetic, as in training)")
    ap.add_argument("--num-windows", type=_positive_int, default=64,
                    help="validation windows to sample (default 64)")
    ap.add_argument("--draws", type=_positive_int, default=4,
                    help="independent perturbations per window (default 4)")
    ap.add_argument("--reference", choices=REFERENCE_SOURCES, default="pristine",
                    help="preferred reference source (default pristine); the probe "
                         "falls back through the remaining sources and reports the "
                         "one it actually used")
    ap.add_argument("--max-minutes", type=_non_negative_float, default=30.0,
                    help="wall-clock guard; 0 disables it. On expiry the probe stops "
                         "sampling, marks the report partial and exits 0")
    ap.add_argument("--out", default=None,
                    help=f"report path or directory (default {DEFAULT_OUT}); must not "
                         f"be inside the checkpoint directory. With --readout actions "
                         f"the default is {ACS_OUT_DIR}/{ACS_OUT_PREFIX}_<env>.json and "
                         f"a directory or a multi-env run gets one file per env")

    acs = ap.add_argument_group(
        "--readout actions (Stage 0)",
        "Action-similarity readout: no checkpoint, no model, no GPU, no video decode. "
        "Ignored by every other readout.")
    acs.add_argument("--env", nargs="+", choices=ACS_ENVS, default=list(ACS_ENVS),
                     metavar="NAME",
                     help=f"environment config group(s) to measure, one JSON report "
                          f"each (default: all of {', '.join(ACS_ENVS)}). Composed from "
                          f"conf/train.yaml with env=<NAME>, num_hist={ACS_NUM_HIST}, "
                          f"num_pred={ACS_NUM_PRED}, frameskip={ACS_FRAMESKIP}")
    acs.add_argument("--split", choices=tuple(ACS_SPLITS), default="train",
                     help="which split is the headline (default train, because that is "
                          "what training sees). The other split is measured and "
                          "reported anyway, as the cross-check")
    acs.add_argument("--acs-action-reduce", choices=("all",) + ACS_ACTION_REDUCTIONS,
                     default="all",
                     help=f"action reduction(s) to measure (default all, which is "
                          f"{', '.join(ACS_ACTION_REDUCTIONS)} in one invocation)")
    acs.add_argument("--acs-gate", choices=ACS_GATES, default=DEFAULT_ACS_GATE,
                     help=f"gate whose weights `w` are reported (default "
                          f"{DEFAULT_ACS_GATE}, the pre-registered one). 'permuted' is "
                          f"not offered: it is the training-time null control and says "
                          f"nothing about a dataset")

    verdict = ap.add_argument_group(
        "--readout actions --summarize (Stage-0 verdict, task 4.3)",
        "Evaluates the PRE-REGISTERED rules A and B on reports already written by "
        "--readout actions. Reads JSON only: no dataset, no DATASET_DIR, no model.")
    verdict.add_argument("--summarize", nargs="+", default=None, metavar="REPORT",
                         help=f"{ACS_OUT_PREFIX}_<env>.json report(s) to evaluate, one "
                              f"per environment, all four of "
                              f"{', '.join(ACS_RULE_ENV_KEYS)} required. Shell globs "
                              f"are expanded here too, so the same command works where "
                              f"the shell does not expand them. Requires --readout "
                              f"actions; --env/--split/--acs-* are not read")
    verdict.add_argument("--table1-gains", default=ACS_TABLE1_GAINS_DEFAULT,
                         metavar="K=V,...",
                         help=f"Table 1's open-loop straightening gains that rule A's "
                              f"ordering is set against (default the published "
                              f"'{ACS_TABLE1_GAINS_DEFAULT}'). Reported, never gating: "
                              f"n = 4 with no replicates cannot establish an ordering")
    return ap


# --------------------------------------------------------------------------
# 1. path validation, before any model is constructed (Requirement 7.5)
# --------------------------------------------------------------------------
def validate_paths(ckpt, train_cfg):
    """
    Resolve `--ckpt` and `--train-cfg` and exit(1) naming the missing ABSOLUTE
    path. This runs before torch is even imported, so no model is constructed and
    no weight is loaded on the failure path.
    """
    missing = []
    resolved = {}
    for flag, raw in (("--ckpt", ckpt), ("--train-cfg", train_cfg)):
        path = Path(raw).expanduser()
        absolute = path if path.is_absolute() else Path(os.getcwd()) / path
        absolute = Path(os.path.normpath(str(absolute)))
        resolved[flag] = absolute
        if not path.is_file():
            missing.append((flag, absolute))
    if missing:
        for flag, absolute in missing:
            print(f"ERROR: {flag} does not exist: {absolute}", file=sys.stderr)
        print("       No model was constructed and no weight was loaded "
              "(Requirement 7.5).", file=sys.stderr)
        sys.exit(1)
    return resolved["--ckpt"].resolve(), resolved["--train-cfg"].resolve()


def resolve_out_path(raw_out, ckpt_path):
    """
    Report destination. Never inside the checkpoint directory (Requirement 7.4):
    the probe must not be able to write a byte anywhere near the artifact it is
    supposed to leave untouched.
    """
    out = Path(raw_out).expanduser()
    if not out.is_absolute():
        out = Path(os.getcwd()) / out
    out = Path(os.path.normpath(str(out)))
    if out.is_dir() or raw_out.endswith(("/", "\\")) or out.suffix.lower() != ".json":
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = out / f"ccr_probe_{ckpt_path.stem}_{stamp}.json"
    ckpt_dir = ckpt_path.parent.resolve()
    try:
        inside = out.resolve().is_relative_to(ckpt_dir)
    except AttributeError:                       # Python < 3.9
        inside = str(out.resolve()).startswith(str(ckpt_dir) + os.sep)
    except OSError:
        inside = False
    if inside:
        print(f"ERROR: --out {out} is inside the checkpoint directory {ckpt_dir}. "
              f"The probe never writes there (Requirement 7.4); use probe_outputs/.",
              file=sys.stderr)
        sys.exit(1)
    return out


# --------------------------------------------------------------------------
# 2. checkpoint fingerprint (Requirement 7.4)
# --------------------------------------------------------------------------
def file_fingerprint(path):
    """sha256 / size / mtime of a file, read in chunks (checkpoints are ~1 GB)."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = os.stat(path)
    return {
        "sha256": digest.hexdigest(),
        "size_bytes": int(stat.st_size),
        "mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }


# --------------------------------------------------------------------------
# wall-clock budget (Requirement 7.6)
# --------------------------------------------------------------------------
class Budget:
    """`--max-minutes` guard. `limit_s <= 0` means "no limit"."""

    def __init__(self, max_minutes):
        self.start = time.perf_counter()
        self.limit_s = float(max_minutes) * 60.0

    def elapsed(self):
        return time.perf_counter() - self.start

    def deadline(self, fraction=1.0):
        """Sub-deadline in elapsed seconds, or None when unbounded."""
        if self.limit_s <= 0:
            return None
        return self.limit_s * float(fraction)

    def expired(self, deadline):
        return deadline is not None and self.elapsed() >= deadline


# --------------------------------------------------------------------------
# 3. model loading -- same shape as plan.py's load_ckpt / load_model
# --------------------------------------------------------------------------
# Keys in a training checkpoint that hold whole nn.Module objects (train.py's
# `_keys_to_save`). Everything else in the payload (epoch, global_iter, the
# *_optimizer state dicts) is ignored here: the probe constructs no optimizer.
MODEL_KEYS = ("encoder", "predictor", "decoder", "proprio_encoder", "action_encoder")


def _warm_dino_hub(train_cfg):
    """
    Make the DINOv2 hub module importable before unpickling.

    `plan.load_ckpt` does exactly this: the checkpoint pickles whole `nn.Module`
    objects whose classes live in the `torch.hub` dinov2 cache, so that cache has
    to be on `sys.path` before `torch.load` walks the pickle. Failure is a warning
    rather than an error, because `torch.load` may still succeed if the module was
    already imported.
    """
    try:
        from models.dino import DinoV2Encoder
        name = "dinov2_vits14"
        try:
            cfg_name = train_cfg.encoder.get("name", None)
        except Exception:
            cfg_name = None
        _ = DinoV2Encoder(str(cfg_name or name), "x_norm_patchtokens")
    except Exception as exc:  # noqa: BLE001 - any hub/cache failure is non-fatal here
        log.warning("Could not pre-import the DINOv2 hub module (%s); torch.load may "
                    "still succeed if it is already importable.", exc)


def load_probe_model(ckpt_path, train_cfg):
    """
    Rebuild the trained `VWorldModel` on CPU, read-only.

    `weights_only=False` is required: these checkpoints store whole `nn.Module`
    objects, not state dicts, which is also why nothing is re-downloaded here.
    """
    import torch

    _warm_dino_hub(train_cfg)
    with open(ckpt_path, "rb") as fh:
        payload = torch.load(fh, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"{ckpt_path} does not hold a checkpoint dict "
                           f"(got {type(payload).__name__}).")

    parts = {k: payload[k] for k in MODEL_KEYS if payload.get(k) is not None}
    if "predictor" not in parts:
        raise RuntimeError(f"No predictor in {ckpt_path}; CCR is a predictor rollout, "
                           f"so there is nothing to probe.")
    for key in ("encoder", "proprio_encoder", "action_encoder"):
        if key not in parts:
            raise RuntimeError(f"No {key} in {ckpt_path}; cannot rebuild the model "
                               f"without re-training it, which this probe will not do.")
    for key, module in parts.items():
        parts[key] = module.to("cpu")
    # The decoder is never exercised: the probe encodes, rolls and measures
    # curvature, and `VWorldModel` treats `decoder=None` as "no decoder".
    parts.setdefault("decoder", None)

    model = _instantiate_world_model(train_cfg, parts)
    epoch = payload.get("epoch")
    log.info("Loaded checkpoint %s (epoch=%s) on CPU, read-only.", ckpt_path, epoch)
    return model, epoch


def _instantiate_world_model(train_cfg, parts):
    """Instantiate `train_cfg.model` from already-built submodules (plan.load_model)."""
    import hydra

    model = hydra.utils.instantiate(
        train_cfg.model,
        encoder=parts["encoder"],
        proprio_encoder=parts["proprio_encoder"],
        action_encoder=parts["action_encoder"],
        predictor=parts["predictor"],
        decoder=parts.get("decoder"),
        proprio_dim=train_cfg.proprio_emb_dim,
        action_dim=train_cfg.action_emb_dim,
        concat_dim=train_cfg.concat_dim,
        num_action_repeat=train_cfg.num_action_repeat,
        num_proprio_repeat=train_cfg.num_proprio_repeat,
    )
    return _freeze_for_probe(model)


def _plain_tensor_attrs_to_cpu(model):
    """
    Move plain (non-parameter, non-buffer) tensor attributes onto the CPU.

    `models/vit.py` builds its causal attention mask as
    `self.bias = generate_mask_matrix(...).to('cuda')` -- a bare attribute, not a
    registered buffer, so `nn.Module.to()` and `.cpu()` both leave it behind. A model
    rebuilt from a checkpoint never trips over this, because the predictor is pickled
    whole and `map_location="cpu"` rewrites the attribute on the way in. A *freshly
    instantiated* predictor (the `pristine` reference) does: its mask lands on cuda:0
    while every parameter is on the CPU, and the first `masked_fill` raises
    "expected self and mask to be on the same device".

    Fixed here rather than in `models/vit.py` because that file is outside the
    Requirement 5.6 changed-file allowlist and its `cuda` default is what every
    training and planning run relies on. Read-only with respect to numerics: only the
    device changes, and this probe is CPU-only by construction (Requirement 7.1).
    """
    import torch

    moved = []
    for name, module in model.named_modules():
        for attr, value in list(module.__dict__.items()):
            if isinstance(value, torch.Tensor) and value.device.type != "cpu":
                setattr(module, attr, value.to("cpu"))
                moved.append(f"{name or '<root>'}.{attr}")
    if moved:
        log.info("Moved %d non-buffer tensor attribute(s) to CPU (e.g. %s).",
                 len(moved), ", ".join(moved[:3]))
    return model


def _freeze_for_probe(model):
    """
    eval() + requires_grad(False). Belt and braces: every measurement already runs
    under `torch.no_grad()`, so this only makes "no populated gradient" true by
    construction rather than by discipline (Property 14).
    """
    _plain_tensor_attrs_to_cpu(model)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def build_pristine_model(train_cfg, dset_dims):
    """
    The `pristine` reference (Requirement 7.3): DINOv2 from the hub cache plus
    freshly initialised projector, predictor, action and proprio encoders. Free --
    no training, no baseline checkpoint. Mirrors `Trainer.init_models`.

    Returns None if it cannot be built (no hub cache, no network); the caller then
    falls through to the next reference source.
    """
    import hydra

    try:
        encoder_kwargs = {}
        projector_config = None
        try:
            projector_config = train_cfg.encoder.get("projector_config", None)
        except Exception:
            projector_config = None
        if projector_config is not None:
            encoder_kwargs["projector_config"] = hydra.utils.instantiate(projector_config)
        encoder = hydra.utils.instantiate(train_cfg.encoder, **encoder_kwargs)

        proprio_encoder = hydra.utils.instantiate(
            train_cfg.proprio_encoder,
            in_chans=dset_dims["proprio_dim"],
            emb_dim=train_cfg.proprio_emb_dim,
        )
        action_encoder = hydra.utils.instantiate(
            train_cfg.action_encoder,
            in_chans=dset_dims["action_dim"],
            emb_dim=train_cfg.action_emb_dim,
        )
        proprio_emb_dim = proprio_encoder.emb_dim
        action_emb_dim = action_encoder.emb_dim

        # Same patch arithmetic as Trainer.init_models.
        if encoder.latent_ndim == 1:
            num_patches = 1
        else:
            num_side_patches = int(train_cfg.img_size) // 16   # decoder_scale, from vqvae
            num_patches = num_side_patches ** 2
        if int(train_cfg.concat_dim) == 0:
            num_patches += 2
        predictor = hydra.utils.instantiate(
            train_cfg.predictor,
            num_patches=num_patches,
            num_frames=train_cfg.num_hist,
            dim=encoder.emb_dim
            + (
                proprio_emb_dim * train_cfg.num_proprio_repeat
                + action_emb_dim * train_cfg.num_action_repeat
            )
            * int(train_cfg.concat_dim),
        )
        parts = {
            "encoder": encoder,
            "predictor": predictor,
            "decoder": None,
            "proprio_encoder": proprio_encoder,
            "action_encoder": action_encoder,
        }
        model = _instantiate_world_model(train_cfg, parts)
    except Exception as exc:  # noqa: BLE001 - an unavailable reference is not fatal
        log.warning("Could not build the pristine reference model (%s); falling back "
                    "to the next reference source.", exc)
        return None
    log.info("Built the pristine reference model (untrained heads, DINOv2 from cache).")
    return model


def configure_ccr(model, rho, rollout_len, action_source):
    """
    Point the model's CCR knobs at the arm being probed.

    These are plain attributes on `VWorldModel` (no module, no buffer), and
    `compute_ccr` reads them every call, so setting them here makes the probe
    measure exactly the configuration a pilot would train.
    """
    model.ccr_rho = float(rho)
    model.ccr_rollout_len = int(rollout_len)
    model.ccr_action_source = str(action_source)
    return model


def reject_infeasible_action_source(model, num_frames):
    """
    `--action-source logged` with L past the window edge is rejected with the exact
    message training gives.

    The message is not duplicated here: `compute_ccr`'s guard raises before it
    touches `z`, so calling it with a dummy action tensor of the right length (and
    `z=None`) produces training's own message and nothing else runs.
    """
    import torch

    required = model.num_hist + model.ccr_rollout_len - 1
    if model.ccr_action_source != "logged" or required <= num_frames:
        return
    try:
        model.compute_ccr(None, torch.zeros(1, num_frames, 1))
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    raise RuntimeError(  # pragma: no cover - only reachable if the guard changed
        "compute_ccr accepted an infeasible ccr_action_source='logged' configuration; "
        "the probe and training no longer agree on the horizon guard."
    )


# --------------------------------------------------------------------------
# 4. windows from the unmodified validation loader, at a fixed seed
# --------------------------------------------------------------------------
def _seed_everything(value):
    random.seed(value)
    try:
        import numpy as np
        np.random.seed(value)
    except Exception:  # pragma: no cover - numpy is a hard dependency of the readouts
        pass
    try:
        import torch
        torch.manual_seed(value)
    except Exception:  # pragma: no cover
        pass


def load_windows(train_cfg, num_windows):
    """
    Sample `num_windows` windows from the validation split through the *unmodified*
    dataset loader (Requirement 5.4: the data path is not touched).

    Each window is one batch of size 1:
        obs["visual"]  (1, num_frames, 3, H, W)
        obs["proprio"] (1, num_frames, proprio_dim)
        act            (1, num_frames, action_dim * frameskip)
        state          (num_frames, 5)   raw PushT pose dims, velocities dropped

    Windows are held in memory so the reference model can be measured on exactly
    the same ones without decoding the videos twice.
    """
    import hydra
    import numpy as np
    import torch

    _seed_everything(PROBE_SEED)
    datasets, _traj = hydra.utils.call(
        train_cfg.env.dataset,
        num_hist=train_cfg.num_hist,
        num_pred=train_cfg.num_pred,
        frameskip=train_cfg.frameskip,
    )
    dset = datasets["valid"]
    dims = {"proprio_dim": dset.proprio_dim, "action_dim": dset.action_dim,
            "state_dim": dset.state_dim, "num_slices": len(dset)}
    if len(dset) == 0:
        raise RuntimeError("The validation split produced no windows; check "
                           f"env.dataset.data_path={train_cfg.env.dataset.data_path}")
    if dset.state_dim < len(STATE_DIM_NAMES):
        # The five reported dimension names are the PushT `state` layout, which is
        # the only cell in scope (Requirement 5.7). Fail loudly rather than report
        # five labels over four columns.
        raise RuntimeError(
            f"env={train_cfg.env.name} has state_dim={dset.state_dim}, but this probe "
            f"reports the PushT dimensions {STATE_DIM_NAMES} (dims 0-4 of `state`). "
            f"Extend STATE_DIM_NAMES before probing another environment."
        )

    take = min(num_windows, len(dset))
    if take < num_windows:
        log.warning("Requested %s windows but the validation split only has %s.",
                    num_windows, len(dset))
    rng = np.random.default_rng(PROBE_SEED)
    indices = rng.choice(len(dset), size=take, replace=False)

    windows = []
    for idx in indices:
        obs, act, state = dset[int(idx)]
        window = {
            "index": int(idx),
            "obs": {
                "visual": obs["visual"].unsqueeze(0).float(),
                "proprio": obs["proprio"].unsqueeze(0).float(),
            },
            "act": act.unsqueeze(0).float(),
            # Only the reported pose dims; 5-6 are velocities (design section 10).
            "state": torch.as_tensor(state).float()[:, : len(STATE_DIM_NAMES)].numpy(),
        }
        windows.append(window)
    log.info("Sampled %s validation window(s) at seed %s (proprio_dim=%s, "
             "action_dim=%s, state_dim=%s).", len(windows), PROBE_SEED,
             dims["proprio_dim"], dims["action_dim"], dims["state_dim"])
    return windows, dims


# --------------------------------------------------------------------------
# 5. per-window measurement
# --------------------------------------------------------------------------
def _aggregate_latent(model, z):
    """
    The aggregated latent the straightening penalty lives in: `encoder.agg` over
    the visual channels, i.e. exactly the feature `total_curvature(..., "aggcos")`
    differentiates. (b, t, agg_dim).
    """
    feats = model.visual_only(z)
    b, t, p, d = feats.shape
    return model.encoder.agg(feats.reshape(b * t, p, d)).reshape(b, t, -1)


def measure(model, windows, rho, draws, budget, deadline, label):
    """
    Encode each window ONCE, then evaluate
    `total_curvature(visual_only(z_imag[:, -(L+2):]), "aggcos")` -- via the model's
    own `compute_ccr` -- under (a) unperturbed actions and (b) `draws` independent
    perturbations at `rho`.

    Only the action channels change between draws (`compute_ccr` clones `z` and
    overwrites them), so the encoder runs `num_frames` times per window and never
    again. That is what keeps the probe inside its CPU budget.

    The unperturbed pass is the `rho = 0` control arm by construction: the same
    `_ccr_actions` construction with an all-zero perturbation.
    """
    import numpy as np
    import torch

    requested_rho = float(rho)
    unpert, pert, aggs, states, indices = [], [], [], [], []
    nonfinite = 0
    partial = False

    with torch.no_grad():
        for position, window in enumerate(windows):
            if budget.expired(deadline):
                partial = True
                log.warning("[%s] wall-clock guard hit after %s/%s window(s) "
                            "(%.1fs elapsed); stopping early and marking the report "
                            "partial.", label, position, len(windows), budget.elapsed())
                break
            obs, act = window["obs"], window["act"]
            z = model.encode(obs, act)

            model.ccr_rho = 0.0                      # unperturbed: exact zeros
            base = float(model.compute_ccr(z, act))
            model.ccr_rho = requested_rho
            draw_values = [float(model.compute_ccr(z, act)) for _ in range(draws)]

            values = [base] + draw_values
            if not all(math.isfinite(v) for v in values):
                # total_curvature masks velocity pairs below its step threshold; a
                # window where every pair is masked yields NaN and carries no
                # information. Counted, reported, excluded.
                nonfinite += 1
                continue

            unpert.append(base)
            pert.append(draw_values)
            aggs.append(_aggregate_latent(model, z)[0].cpu().numpy())
            states.append(window["state"])
            indices.append(window["index"])

    model.ccr_rho = requested_rho
    if nonfinite:
        log.warning("[%s] excluded %s window(s) whose curvature was not finite "
                    "(every velocity pair below total_curvature's step threshold).",
                    label, nonfinite)
    return {
        "label": label,
        "windows": len(unpert),
        "nonfinite": nonfinite,
        "partial": partial,
        "unperturbed": np.asarray(unpert, dtype=np.float64),
        "perturbed": np.asarray(pert, dtype=np.float64),      # (n, draws)
        "agg": np.asarray(aggs, dtype=np.float64),            # (n, t, agg_dim)
        "state": np.asarray(states, dtype=np.float64),        # (n, t, 5)
        "indices": indices,
    }


# --------------------------------------------------------------------------
# 6. readouts, disaggregated per state dimension (Requirement 7.2)
# --------------------------------------------------------------------------
def _top_tercile_mask(motion):
    """
    Windows in the top tercile of `motion` -- the windows in which this dimension
    is the dominant motion. Ties are broken by index, deterministically.
    """
    import numpy as np

    n = motion.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)
    k = max(1, int(math.ceil(n / TERCILE)))
    order = np.argsort(-motion, kind="stable")
    mask = np.zeros(n, dtype=bool)
    mask[order[:k]] = True
    return mask


def curvature_readout(m):
    """
    `curvature_gap` = mean(perturbed) - mean(unperturbed), aggregate and per
    dimension, plus the unperturbed curvature magnitude the gate compares against.

    Per dimension the same difference is restricted to the top-tercile motion
    subset for that dimension. Aggregating over all windows is the mistake
    documented in SHORT_BUDGET_PILOTS.md section 4.
    """
    import numpy as np

    if m["windows"] == 0:
        empty = {name: None for name in STATE_DIM_NAMES}
        return {"aggregate": None, "per_dim": dict(empty),
                "unperturbed_aggregate": None, "unperturbed_per_dim": dict(empty),
                "windows_per_dim": {name: 0 for name in STATE_DIM_NAMES}}

    unpert = m["unperturbed"]                     # (n,)
    pert = m["perturbed"].mean(axis=1)            # (n,) mean over draws
    state = m["state"]                            # (n, t, 5)
    motion = np.abs(state[:, -1, :] - state[:, 0, :])   # (n, 5)

    per_dim, unpert_per_dim, counts = {}, {}, {}
    for d, name in enumerate(STATE_DIM_NAMES):
        mask = _top_tercile_mask(motion[:, d])
        counts[name] = int(mask.sum())
        if not mask.any():
            per_dim[name] = None
            unpert_per_dim[name] = None
            continue
        per_dim[name] = float(pert[mask].mean() - unpert[mask].mean())
        unpert_per_dim[name] = float(np.abs(unpert[mask]).mean())
    return {
        "aggregate": float(pert.mean() - unpert.mean()),
        "per_dim": per_dim,
        "unperturbed_aggregate": float(np.abs(unpert).mean()),
        "unperturbed_per_dim": unpert_per_dim,
        "windows_per_dim": counts,
    }


def _ridge_predict(x_train, y_train, x_test, alpha=RIDGE_ALPHA):
    """
    Closed-form ridge, numpy only (no sklearn).

    Features are standardized with the *train* split's statistics and targets are
    centred, so the intercept is handled by the centring and the penalty only ever
    touches the slopes. Ridge is linear in `y`, so a per-dimension R^2 is invariant
    to how the state dimensions are scaled -- which is what makes the
    per-dimension numbers comparable across dimensions with wildly different units
    (pixels vs radians).
    """
    import numpy as np

    x_mean = x_train.mean(axis=0, keepdims=True)
    x_std = x_train.std(axis=0, keepdims=True)
    x_std = np.where(x_std < 1e-8, 1.0, x_std)
    xt = (x_train - x_mean) / x_std
    xs = (x_test - x_mean) / x_std
    y_mean = y_train.mean(axis=0, keepdims=True)
    yt = y_train - y_mean

    d = xt.shape[1]
    gram = xt.T @ xt + alpha * np.eye(d)
    rhs = xt.T @ yt
    try:
        weights = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:                  # pragma: no cover - alpha > 0 makes this rare
        weights = np.linalg.lstsq(gram, rhs, rcond=None)[0]
    return xs @ weights + y_mean


def state_readout(m):
    """
    `state_readout_r2`: ridge R^2 from the aggregated latent to each state
    dimension, on a held-out split.

    The split is by *window* (the first `RIDGE_TRAIN_FRACTION` of the sampled
    windows train, the rest test), never by frame, so frames of one window cannot
    appear on both sides.

    `aggregate` is the pooled multi-output R^2 in raw state units,
    `1 - sum_d SSE_d / sum_d SStot_d`. It exists only so the report shows what an
    aggregate would have said; the per-dimension entries are the readout that
    decides anything.
    """
    import numpy as np

    per_dim = {name: None for name in STATE_DIM_NAMES}
    if m["windows"] < 4:
        return {"aggregate": None, "per_dim": per_dim, "train_windows": 0,
                "test_windows": 0, "feature_dim": 0}

    agg, state = m["agg"], m["state"]              # (n, t, dim), (n, t, 5)
    n = agg.shape[0]
    n_train = max(1, min(n - 1, int(round(RIDGE_TRAIN_FRACTION * n))))
    x_train = agg[:n_train].reshape(-1, agg.shape[-1])
    x_test = agg[n_train:].reshape(-1, agg.shape[-1])
    y_train = state[:n_train].reshape(-1, state.shape[-1])
    y_test = state[n_train:].reshape(-1, state.shape[-1])

    pred = _ridge_predict(x_train, y_train, x_test)
    sse = ((y_test - pred) ** 2).sum(axis=0)
    sst = ((y_test - y_test.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
    for d, name in enumerate(STATE_DIM_NAMES):
        # A dimension that does not move on the held-out split has no variance to
        # explain; reporting 0.0 says "no signal" without inventing a number.
        per_dim[name] = 0.0 if sst[d] < 1e-12 else float(1.0 - sse[d] / sst[d])
    total_sst = float(sst.sum())
    aggregate = 0.0 if total_sst < 1e-12 else float(1.0 - sse.sum() / total_sst)
    return {"aggregate": aggregate, "per_dim": per_dim,
            "train_windows": n_train, "test_windows": n - n_train,
            "feature_dim": int(agg.shape[-1])}


def readouts_from_measurement(m):
    """Both readouts for one model, in the report's shape (minus the reference)."""
    curvature = curvature_readout(m)
    state = state_readout(m)
    return {
        "curvature_gap": {"aggregate": curvature["aggregate"],
                          "per_dim": curvature["per_dim"]},
        "state_readout_r2": {"aggregate": state["aggregate"],
                             "per_dim": state["per_dim"]},
        "_curvature_detail": curvature,
        "_state_detail": state,
    }


# --------------------------------------------------------------------------
# 7. reference values (Requirement 7.3)
# --------------------------------------------------------------------------
def _straighten_scale(train_cfg):
    """`aggcos1e-1` -> 0.1, mirroring VWorldModel.__init__'s parsing."""
    try:
        raw = train_cfg.training.get("straighten", False)
    except Exception:
        return None
    if not isinstance(raw, str):
        return None
    for prefix in ("aggcos", "cos"):
        if raw.startswith(prefix):
            suffix = raw[len(prefix):]
            try:
                return float(suffix) if suffix else 1.0
            except ValueError:
                return None
    return None


def _pristine_reference(ctx):
    """
    Recompute both readouts with an untrained model: DINOv2 from the hub cache plus
    freshly initialised projector / predictor / action / proprio encoders. Free, and
    it is what lets a number be called an *improvement* rather than merely
    "not yet degraded" (SHORT_BUDGET_PILOTS.md section 4).
    """
    model = build_pristine_model(ctx["train_cfg"], ctx["dims"])
    if model is None:
        return {}
    configure_ccr(model, ctx["rho"], ctx["rollout_len"], ctx["action_source"])
    _freeze_for_probe(model)
    m = measure(model, ctx["windows"], ctx["rho"], ctx["draws"], ctx["budget"],
                ctx["reference_deadline"], "pristine")
    if m["partial"]:
        log.warning("The pristine reference was cut short at %s of %s window(s), so it "
                    "is measured on a subset of the windows the checkpoint was measured "
                    "on; read it as indicative, not matched.",
                    m["windows"], len(ctx["windows"]))
    if m["windows"] == 0:
        log.warning("The pristine reference measured 0 usable window(s); it cannot "
                    "serve as a reference.")
        return {}
    ctx["reference_details"]["pristine"] = {
        "windows": m["windows"], "partial": m["partial"], "nonfinite": m["nonfinite"]}
    readouts = readouts_from_measurement(m)
    return {
        "curvature_gap": {"value": readouts["curvature_gap"]["aggregate"],
                          "per_dim": readouts["curvature_gap"]["per_dim"]},
        "state_readout_r2": {"value": readouts["state_readout_r2"]["aggregate"],
                             "per_dim": readouts["state_readout_r2"]["per_dim"]},
    }


def _early_telemetry_reference(ctx):
    """
    The reference run's own first-8,000-step rows, read out of its
    `training_log.jsonl`: a free matched-budget control.

    It can only speak for `curvature_gap`, and only for a run that logged a `ccr`
    block: the difference between the mean raw CCR curvature (imagined, perturbed)
    and the mean raw baseline curvature (on-log) over those rows is the telemetry
    analogue of the gap. There is no `state_readout_r2` in telemetry, so that
    readout falls through to the next source.
    """
    path = Path(ctx["run_dir"]) / TELEMETRY_BASENAME
    if not path.is_file():
        log.info("No %s in %s; the early-telemetry reference is unavailable.",
                 TELEMETRY_BASENAME, ctx["run_dir"])
        return {}
    scale = _straighten_scale(ctx["train_cfg"])
    ccr_raw, curvature_raw = [], []
    rows = 0
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue    # a truncated last line is the signature of a killed job
                if not isinstance(rec, dict):
                    continue
                it = rec.get("global_iter")
                if not isinstance(it, (int, float)) or it > EARLY_TELEMETRY_MAX_ITER:
                    continue
                rows += 1
                block = rec.get("ccr")
                if isinstance(block, dict) and isinstance(block.get("raw"), (int, float)):
                    ccr_raw.append(float(block["raw"]))
                terms = rec.get("terms")
                if isinstance(terms, dict) and scale:
                    entry = terms.get("curvature")
                    if isinstance(entry, dict) and isinstance(entry.get("scaled"), (int, float)):
                        curvature_raw.append(float(entry["scaled"]) / scale)
    except OSError as exc:
        log.warning("Could not read %s (%s).", path, exc)
        return {}
    if not ccr_raw or not curvature_raw:
        log.info("%s has %s row(s) at global_iter <= %s but no usable ccr/curvature "
                 "pair; the early-telemetry reference is unavailable for "
                 "curvature_gap.", path, rows, EARLY_TELEMETRY_MAX_ITER)
        return {}
    value = sum(ccr_raw) / len(ccr_raw) - sum(curvature_raw) / len(curvature_raw)
    ctx["reference_details"]["early_telemetry"] = {"rows": rows, "log": str(path)}
    # Telemetry is a scalar per iteration: there is nothing to disaggregate, and
    # saying so is better than fabricating five equal numbers.
    return {"curvature_gap": {"value": float(value), "per_dim": None}}


def _control_run_reference(ctx):
    """
    Last resort (Requirement 7.3): the readouts of a previous probe report for a
    *different* checkpoint sitting in the same output directory, i.e. a
    matched-budget control run that was already probed.
    """
    out_dir = Path(ctx["out_path"]).parent
    if not out_dir.is_dir():
        return {}
    candidates = []
    for path in sorted(out_dir.glob("*.json")):
        if path.resolve() == Path(ctx["out_path"]).resolve():
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                report = json.load(fh)
        except (OSError, ValueError):
            continue
        if not isinstance(report, dict) or "readouts" not in report:
            continue
        if report.get("ckpt_sha256") == ctx["ckpt_sha256"]:
            continue        # the same checkpoint is not a control
        candidates.append((path.stat().st_mtime, path, report))
    if not candidates:
        log.info("No other probe report in %s to use as a control run.", out_dir)
        return {}
    _mtime, path, report = max(candidates, key=lambda item: item[0])
    ctx["reference_details"]["control_run"] = {"report": str(path),
                                              "ckpt": report.get("ckpt")}
    out = {}
    for name in ("curvature_gap", "state_readout_r2"):
        entry = report["readouts"].get(name)
        if isinstance(entry, dict) and entry.get("aggregate") is not None:
            out[name] = {"value": entry.get("aggregate"),
                         "per_dim": entry.get("per_dim")}
    log.info("Using %s as the control-run reference.", path)
    return out


REFERENCE_PROVIDERS = {
    "pristine": _pristine_reference,
    "early_telemetry": _early_telemetry_reference,
    "control_run": _control_run_reference,
}


def resolve_references(requested, ctx, readout_names=("curvature_gap", "state_readout_r2")):
    """
    Resolve a reference value per readout, trying `requested` first and then the
    remaining sources in their canonical cheapest-first order. A source is only
    materialised while some readout is still unresolved, so the (expensive)
    pristine measurement is skipped entirely when it is not needed.
    """
    order = [requested] + [s for s in REFERENCE_SOURCES if s != requested]
    resolved = {name: {"value": None, "per_dim": None, "source": None}
                for name in readout_names}
    for source in order:
        if all(entry["value"] is not None for entry in resolved.values()):
            break
        if ctx["budget"].expired(ctx["reference_deadline"]) and source == "pristine":
            log.warning("Skipping the pristine reference: the wall-clock budget is "
                        "already spent.")
            continue
        try:
            provided = REFERENCE_PROVIDERS[source](ctx)
        except Exception as exc:  # noqa: BLE001 - a broken reference must not lose the run
            log.warning("Reference source %r failed (%s); trying the next one.",
                        source, exc)
            provided = {}
        for name in readout_names:
            if resolved[name]["value"] is not None:
                continue
            entry = provided.get(name)
            if isinstance(entry, dict) and entry.get("value") is not None:
                resolved[name] = {"value": float(entry["value"]),
                                  "per_dim": entry.get("per_dim"),
                                  "source": source}
    for name, entry in resolved.items():
        if entry["value"] is None:
            log.warning("No reference value could be obtained for %s from any of %s.",
                        name, ", ".join(REFERENCE_SOURCES))
    return resolved


# --------------------------------------------------------------------------
# probe gate (design section 10, Requirement 8.1)
# --------------------------------------------------------------------------
GATE_CRITERION = (
    "aggregate curvature_gap > 0 AND, on at least "
    f"{GATE_MIN_DIMS} of the {len(STATE_DIM_NAMES)} disaggregated dimensions, "
    f"gap > 0 and gap >= {GATE_MIN_RATIO:.0%} of that dimension's unperturbed "
    "curvature magnitude"
)


def evaluate_gate(detail):
    """The written gate, evaluated so the operator does not have to remember it."""
    aggregate = detail["aggregate"]
    aggregate_positive = aggregate is not None and aggregate > 0.0
    per_dim = {}
    passing = 0
    for name in STATE_DIM_NAMES:
        gap = detail["per_dim"].get(name)
        base = detail["unperturbed_per_dim"].get(name)
        ratio = None
        ok = False
        if gap is not None and base is not None:
            ratio = None if base <= 0 else gap / base
            ok = gap > 0.0 and ratio is not None and ratio >= GATE_MIN_RATIO
        per_dim[name] = {
            "gap": gap,
            "unperturbed": base,
            "ratio": ratio,
            "windows": detail["windows_per_dim"].get(name),
            "pass": bool(ok),
        }
        passing += int(ok)
    return {
        "criterion": GATE_CRITERION,
        "min_ratio": GATE_MIN_RATIO,
        "min_dims": GATE_MIN_DIMS,
        "aggregate_curvature_gap": aggregate,
        "aggregate_curvature_unperturbed": detail["unperturbed_aggregate"],
        "aggregate_positive": bool(aggregate_positive),
        "dims_passing": passing,
        "per_dim": per_dim,
        "passed": bool(aggregate_positive and passing >= GATE_MIN_DIMS),
    }


# --------------------------------------------------------------------------
# report (schema: design section 10, "Probe report")
# --------------------------------------------------------------------------
def _round(value, places=6):
    if value is None:
        return None
    return round(float(value), places)


def _round_map(mapping, places=6):
    if mapping is None:
        return None
    return {key: _round(value, places) for key, value in mapping.items()}


def build_report(args, ckpt_path, cfg_path, before, after, train_cfg, measurement,
                 readouts, references, gate, budget, synthesized, num_frames,
                 reference_details, epoch):
    curvature = readouts["curvature_gap"]
    state = readouts["state_readout_r2"]
    return {
        "probe": "probe_ccr_curvature.py",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "ckpt": str(ckpt_path),
        "ckpt_sha256": before["sha256"],
        "ckpt_size_bytes": before["size_bytes"],
        "ckpt_mtime": before["mtime"],
        "ckpt_sha256_final": after["sha256"],
        "checkpoint_modified": bool(after["sha256"] != before["sha256"]),
        "ckpt_epoch": epoch,
        "train_cfg": str(cfg_path),
        "env": str(train_cfg.env.name),
        "rho": float(args.rho),
        "rollout_len": int(args.rollout_len),
        "action_source": str(args.action_source),
        "synthesized_action_frames": int(synthesized),
        "num_windows": int(args.num_windows),
        "draws": int(args.draws),
        "num_hist": int(train_cfg.num_hist),
        "num_frames": int(num_frames),
        "windows_evaluated": int(measurement["windows"]),
        "windows_nonfinite": int(measurement["nonfinite"]),
        "seed": PROBE_SEED,
        "partial": bool(measurement["partial"]),
        "elapsed_s": _round(budget.elapsed(), 1),
        "max_minutes": float(args.max_minutes),
        "reference_requested": str(args.reference),
        "reference_details": reference_details,
        "readouts": {
            "curvature_gap": {
                "aggregate": _round(curvature["aggregate"]),
                "per_dim": _round_map(curvature["per_dim"]),
                "reference_value": _round(references["curvature_gap"]["value"]),
                "reference_source": references["curvature_gap"]["source"],
                "reference_per_dim": _round_map(references["curvature_gap"]["per_dim"]),
            },
            "state_readout_r2": {
                "aggregate": _round(state["aggregate"]),
                "per_dim": _round_map(state["per_dim"]),
                "reference_value": _round(references["state_readout_r2"]["value"]),
                "reference_source": references["state_readout_r2"]["source"],
                "reference_per_dim": _round_map(
                    references["state_readout_r2"]["per_dim"]),
            },
        },
        "curvature_unperturbed": {
            "aggregate": _round(readouts["_curvature_detail"]["unperturbed_aggregate"]),
            "per_dim": _round_map(readouts["_curvature_detail"]["unperturbed_per_dim"]),
        },
        "ridge_readout": {
            "alpha": RIDGE_ALPHA,
            "train_fraction": RIDGE_TRAIN_FRACTION,
            "train_windows": readouts["_state_detail"]["train_windows"],
            "test_windows": readouts["_state_detail"]["test_windows"],
            "feature_dim": readouts["_state_detail"]["feature_dim"],
            "aggregate_definition": "pooled multi-output R^2 in raw state units",
        },
        "gate": gate,
    }


def write_report(out_path, report):
    """Reports go to `probe_outputs/`, never into the checkpoint directory (7.4)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
        fh.write("\n")
    return out_path


# --------------------------------------------------------------------------
# printing
# --------------------------------------------------------------------------
def _fmt(value, width=12, places=6):
    if value is None:
        return f"{'n/a':>{width}}"
    return f"{value:>{width}.{places}f}"


def print_report(report):
    r = report["readouts"]
    print()
    print(RULE)
    print(f"CCR CURVATURE PROBE  {report['ckpt']}")
    print(RULE)
    print(f"  env / epoch           : {report['env']} / {report['ckpt_epoch']}")
    print(f"  arm                   : rho={report['rho']} L={report['rollout_len']} "
          f"action_source={report['action_source']} "
          f"synthesized_action_frames={report['synthesized_action_frames']}")
    print(f"  windows / draws       : {report['windows_evaluated']}"
          f"/{report['num_windows']} evaluated, {report['draws']} draw(s)"
          + (f", {report['windows_nonfinite']} non-finite excluded"
             if report["windows_nonfinite"] else ""))
    print(f"  elapsed               : {report['elapsed_s']}s "
          f"(guard {report['max_minutes']} min)"
          + ("  PARTIAL" if report["partial"] else ""))
    print(f"  checkpoint sha256     : {report['ckpt_sha256'][:16]}... "
          f"({'UNCHANGED' if not report['checkpoint_modified'] else 'CHANGED'})")

    for name in ("curvature_gap", "state_readout_r2"):
        entry = r[name]
        print()
        print(f"{name}   (reference: {entry['reference_source'] or 'none'})")
        print(THIN)
        print(f"  {'dimension':<14}{'value':>12}{'reference':>12}")
        print(f"  {'AGGREGATE':<14}{_fmt(entry['aggregate'])}"
              f"{_fmt(entry['reference_value'])}")
        ref_per_dim = entry.get("reference_per_dim") or {}
        for dim in STATE_DIM_NAMES:
            print(f"  {dim:<14}{_fmt(entry['per_dim'].get(dim))}"
                  f"{_fmt(ref_per_dim.get(dim))}")
        if name == "curvature_gap":
            print("  (per-dimension values are restricted to the top-tercile motion "
                  "subset for that dimension)")

    gate = report["gate"]
    print()
    print(RULE)
    print("PROBE GATE (Requirement 8.1, written before running)")
    print(RULE)
    print(f"  criterion             : {gate['criterion']}")
    print(f"  aggregate gap         : {_fmt(gate['aggregate_curvature_gap'], 10)}"
          f"   (unperturbed magnitude "
          f"{_fmt(gate['aggregate_curvature_unperturbed'], 10).strip()})"
          f"  -> {'positive' if gate['aggregate_positive'] else 'NOT positive'}")
    print()
    print(f"  {'dimension':<14}{'gap':>12}{'unperturbed':>14}{'ratio':>10}"
          f"{'windows':>9}  verdict")
    for dim in STATE_DIM_NAMES:
        item = gate["per_dim"][dim]
        ratio = item["ratio"]
        ratio_txt = "       n/a" if ratio is None else f"{ratio:>9.3f}"
        print(f"  {dim:<14}{_fmt(item['gap'])}{_fmt(item['unperturbed'], 14)}"
              f"{ratio_txt}{str(item['windows']):>9}  "
              f"{'pass' if item['pass'] else 'fail'}")
    print()
    print(f"  dimensions passing    : {gate['dims_passing']} of "
          f"{len(STATE_DIM_NAMES)} (need >= {gate['min_dims']})")
    print(f"  PROBE GATE            : {'PASS' if gate['passed'] else 'FAIL'}")
    if gate["passed"]:
        print("  The mechanism is present. Rung 2 (a Pilot_Run) may be launched "
              "(Requirement 11.3).")
    else:
        print("  The mechanism is NOT present at this rho. Do NOT launch a "
              "Pilot_Run (Requirement 11.3).")
    if report["partial"]:
        print("  NOTE: the report is PARTIAL (wall-clock guard). Judge the gate on a "
              "complete run before acting on it.")
    print(RULE)


# --------------------------------------------------------------------------
# 8. `--readout actions`: the Stage-0 action-similarity readout (ACS Req 1)
# --------------------------------------------------------------------------
# Nothing below constructs an encoder, a predictor or a decoder, reads a
# checkpoint, touches a GPU or decodes a video frame. What it needs from the
# repository is exactly two functions -- `VWorldModel.reduce_action` and
# `VWorldModel.action_gate` -- and what it needs from the dataset is exactly one
# tensor, `actions`.
def _acs_conf_dir():
    """`conf/`, beside this file. Stage 0 composes from `conf/train.yaml` (Req 1.13)."""
    conf_dir = Path(__file__).resolve().parent / "conf"
    if not conf_dir.is_dir():
        print(f"ERROR: no config directory at {conf_dir}; --readout actions composes "
              f"each environment from conf/train.yaml (ACS Requirement 1.13).",
              file=sys.stderr)
        sys.exit(1)
    return conf_dir


def require_dataset_dir():
    """
    All four Stage-0 env configs set `data_path: ${oc.env:DATASET_DIR}/...`, and
    OmegaConf resolves that lazily -- so an unset variable surfaces deep inside
    `hydra.utils.call` as an interpolation error rather than as the setup problem it
    is. Checked once, up front, before any config is composed.
    """
    if os.environ.get("DATASET_DIR"):
        return
    print("ERROR: DATASET_DIR is not set. Every Stage-0 environment config resolves "
          "env.dataset.data_path from it (e.g. ${oc.env:DATASET_DIR}/pusht_noise), so "
          "--readout actions has nothing to read.", file=sys.stderr)
    sys.exit(1)


def compose_env_cfg(env_name):
    """
    `conf/train.yaml` composed with `env=<env_name>` and the protocol values
    (ACS Requirement 1.13).

    Composed rather than read off a run's `hydra.yaml`, because Wall, UMaze and
    Medium have no trained run and Stage 0 needs none. `num_hist`, `num_pred` and
    `frameskip` are pinned to the constants above rather than exposed as flags, so
    two Stage-0 runs cannot disagree about the window they measured.
    """
    from hydra import compose, initialize_config_dir
    # Registers the OmegaConf resolvers `conf/train.yaml` interpolates.
    import custom_resolvers  # noqa: F401

    conf_dir = _acs_conf_dir()
    overrides = [
        f"env={env_name}",
        f"num_hist={ACS_NUM_HIST}",
        f"num_pred={ACS_NUM_PRED}",
        f"frameskip={ACS_FRAMESKIP}",
    ]
    with initialize_config_dir(config_dir=str(conf_dir), version_base=None):
        cfg = compose(config_name="train", overrides=overrides)
    log.info("Composed env=%s from conf/train.yaml with %s.", env_name,
             " ".join(overrides[1:]))
    return cfg


# One tiny `VWorldModel` per (reduction, gate) pair, cached. The modules are
# `models.dummy.DummyModel`s of width 1 and are never called: the only methods this
# readout invokes are `reduce_action` and `action_gate`, neither of which touches the
# encoder, the predictor or the action encoder. Going through the real constructor
# rather than a hand-rolled shim is the point -- `__init__` is what validates the two
# enums, so a bad `--acs-gate` raises here exactly as it would in training.
_ACS_GATE_MODELS = {}


def acs_gate_model(action_reduce, gate):
    """
    A `VWorldModel` configured for one (reduction, gate) pair, on the CPU, frozen.

    This is how ACS Requirements 15.1 and 15.2 are met: `a_t` and `w_t` come from
    the shipped `VWorldModel.reduce_action` / `VWorldModel.action_gate` bound to a
    real instance. There is deliberately no cosine of actions in this file (P19) --
    Stage 0's prediction and training's `acs_gate_tv` have to be the same number,
    and the only way to guarantee that is for them to be the same code.
    """
    key = (str(action_reduce), str(gate))
    cached = _ACS_GATE_MODELS.get(key)
    if cached is not None:
        return cached

    from models.dummy import DummyModel
    from models.visual_world_model import VWorldModel

    # `VWorldModel.__init__` logs ~15 startup lines per instance, which says nothing
    # here (there is no encoder, no predictor and no straightening term).
    vwm_log = logging.getLogger("models.visual_world_model")
    previous_level = vwm_log.level
    vwm_log.setLevel(max(previous_level, logging.WARNING))
    try:
        model = VWorldModel(
            image_size=ACS_IMAGE_SIZE_STUB,
            num_hist=ACS_NUM_HIST,
            num_pred=ACS_NUM_PRED,
            encoder=DummyModel(emb_dim=1),
            proprio_encoder=DummyModel(emb_dim=1),
            action_encoder=DummyModel(emb_dim=1),
            decoder=None,
            predictor=None,
            straighten=False,          # no curvature term is evaluated here
            acs_action_reduce=action_reduce,
            acs_gate=gate,
        )
    finally:
        vwm_log.setLevel(previous_level)
    # Reuses the probe's existing helpers rather than adding a second copy of either
    # (ACS Requirement 9.15 / E15): `_freeze_for_probe` calls
    # `_plain_tensor_attrs_to_cpu`, which is the fix for `models/vit.py`'s
    # cuda-pinned mask. `_warm_dino_hub` is not called, and does not need to be: this
    # readout unpickles nothing and constructs no DINOv2 encoder.
    model = _freeze_for_probe(model)
    _ACS_GATE_MODELS[key] = model
    return model


def _slicer_action_tensor(dset):
    """
    The tensor `TrajSlicerDataset.__getitem__` slices its actions out of, and the
    index convention it uses -- without going anywhere near a video frame (Trap 2).

    `__getitem__` has three branches. The `load_visual_frames` branch reads
    `self.dataset.actions[i, start:end]`; the `get_frames` branch reads
    `self.dataset.get_frames(i, range(start, end))`, and every shipped `get_frames`
    body starts with `act = self.actions[idx, frames]`. Both therefore resolve to the
    *same* tensor under the *same* slice index `i`, so one expression covers both.
    (`i` is the index the slice list recorded; when `self.dataset` is a `TrajSubset`
    its `actions` comes through `TrajSubset.__getattr__`. That is the shipped
    behaviour, and reproducing it -- rather than improving it -- is what makes the
    32-window bitwise check against `dset[idx][1]` meaningful.)

    The third branch, `self.dataset[i]`, belongs to a dataset with neither method and
    would decode. None of the four Stage-0 environments takes it, so it raises here
    instead of being silently approximated.
    """
    inner = dset.dataset
    if not (hasattr(inner, "load_visual_frames") or hasattr(inner, "get_frames")):
        raise RuntimeError(
            f"{type(inner).__name__} has neither load_visual_frames nor get_frames, so "
            f"TrajSlicerDataset.__getitem__ takes its third branch and this action-only "
            f"loader would not reproduce it. Extend _slicer_action_tensor before "
            f"measuring this environment."
        )
    actions = getattr(inner, "actions", None)
    if actions is None:
        raise RuntimeError(
            f"{type(inner).__name__} exposes no `actions` tensor; the action-only "
            f"loader has nothing to read (every shipped dataset stores one)."
        )
    return actions


def load_action_windows(train_cfg, split, chunk_windows=ACS_CHUNK_WINDOWS):
    """
    Every window's action block for one split, with no video decoded (Trap 2).

    Returns `(dset, act, meta)`:

        dset  the `TrajSlicerDataset` itself, so the 32-window bitwise check of
              ACS Requirement 1.16 can compare against `dset[idx][1]`
        act   (n_windows, num_frames, frameskip * env_action_dim), in the dataset's
              own dtype -- never cast, because a cast would break that check
        meta  n_windows / num_frames / frameskip / env_action_dim / block_dim

    `load_windows` is left alone: its `state_dim` guard is correct for the readouts it
    protects and PushT-specific, so reusing it here would raise on Wall before
    measuring anything (ACS error case E14).
    """
    import hydra
    import numpy as np
    import torch
    from einops import rearrange

    split_key = ACS_SPLITS[split]
    datasets, _traj = hydra.utils.call(
        train_cfg.env.dataset,
        num_hist=train_cfg.num_hist,
        num_pred=train_cfg.num_pred,
        frameskip=train_cfg.frameskip,
    )
    dset = datasets[split_key]
    if len(dset) == 0:
        raise RuntimeError(
            f"The {split} split of env={train_cfg.env.name} produced no windows; check "
            f"env.dataset.data_path={train_cfg.env.dataset.data_path}."
        )

    num_frames = int(dset.num_frames)
    frameskip = int(dset.frameskip)
    block_dim = int(dset.action_dim)                    # frameskip * env_action_dim
    env_action_dim = int(dset.dataset.action_dim)       # the protocol value `d`
    span = num_frames * frameskip
    if block_dim != frameskip * env_action_dim:
        raise RuntimeError(
            f"env={train_cfg.env.name}: the slicer reports action_dim={block_dim} but "
            f"frameskip={frameskip} times the env action dim {env_action_dim} is "
            f"{frameskip * env_action_dim}; the substep packing is not what "
            f"reduce_action assumes."
        )

    actions = _slicer_action_tensor(dset)
    slices = np.asarray(dset.slices)
    if slices.ndim != 2 or slices.shape[1] != 3:
        raise RuntimeError(f"dset.slices has shape {slices.shape}, expected (n, 3).")
    spans = slices[:, 2] - slices[:, 1]
    if not np.all(spans == span):
        raise RuntimeError(
            f"env={train_cfg.env.name}: {int((spans != span).sum())} slice(s) span "
            f"something other than num_frames * frameskip = {span}; the window "
            f"arithmetic of this loader and of TrajSlicerDataset have diverged."
        )

    traj_idx = torch.as_tensor(slices[:, 0].astype(np.int64))
    starts = torch.as_tensor(slices[:, 1].astype(np.int64))
    offsets = torch.arange(span, dtype=torch.int64)
    blocks = []
    for lo in range(0, len(slices), int(chunk_windows)):
        hi = min(lo + int(chunk_windows), len(slices))
        # (chunk, span) absolute frame indices -> (chunk, span, d) gather. One
        # advanced-indexing op per chunk; no __getitem__, no VideoReader.
        frame_idx = starts[lo:hi, None] + offsets[None, :]
        block = actions[traj_idx[lo:hi, None], frame_idx]
        # The same rearrange TrajSlicerDataset.__getitem__ applies, one batch axis out.
        # Pure data movement, so it is value-identical to the per-window call.
        blocks.append(rearrange(block, "b (n f) d -> b n (f d)", n=num_frames))
    act = torch.cat(blocks, dim=0) if len(blocks) > 1 else blocks[0]

    # Layout only: `n_windows` is a property of the split and lives in the split's own
    # block, so no statistic can be read against the wrong denominator.
    meta = {
        "num_frames": num_frames,
        "frameskip": frameskip,
        "env_action_dim": env_action_dim,
        "block_dim": block_dim,
        "substeps": block_dim // env_action_dim,
        "triples_per_window": num_frames - 2,
        "dtype": str(act.dtype).replace("torch.", ""),
    }
    log.info("%s split of env=%s: %s window(s) of %s x %s actions, no video decoded.",
             split, train_cfg.env.name, int(act.shape[0]), num_frames, block_dim)
    return dset, act, meta


def cos_and_gate(act, action_reduce, gate, env_action_dim):
    """
    `cos(a_t, a_{t+1})` and the gate weight `w`, both from the shipped gate.

    `cos` is *recovered* rather than recomputed: `acs_gate="affine_cos"` is exactly
    `(1 + cos) / 2` by definition, so `2 * w_affine - 1` inverts it. That keeps the
    promise that no second cosine-of-actions implementation exists in this file
    (ACS Requirement 15.3, P19) -- the cosine is computed once, inside
    `VWorldModel.action_gate`, whichever statistic ends up reading it.

    The inversion is monotone and exact up to one float32 rounding of `1 + cos`
    (<= ~6e-8 absolute), which is well below any resolution the reported mean,
    median, 20-bin histogram or `frac(cos < 0)` claims.

    Returns two flat tensors of length `n_windows * (num_frames - 2)`.
    """
    import torch

    affine = acs_gate_model(action_reduce, "affine_cos")
    gated = acs_gate_model(action_reduce, gate)
    with torch.no_grad():
        cos = affine.action_gate(act, env_action_dim=env_action_dim) * 2.0 - 1.0
        w = gated.action_gate(act, env_action_dim=env_action_dim)
    return cos.reshape(-1).float(), w.reshape(-1).float()


def summarize_cos_and_gate(cos, w, n_windows):
    """
    The Stage-0 statistics for one environment, split and reduction.

    Every one of ACS Requirements 1.2-1.9, with `n_triples` and `n_windows` beside it
    (1.10) so no number is ever read without its denominator.

    `R = E|w - E[w]| / (2 * E[w])` is the reallocation statistic (design 11.3): the
    population form of the total-variation distance between the normalized weight
    vector and uniform, and the finite-batch form of training's `acs_gate_tv`. It is
    `mean(w)` that is *not* the right kill signal -- a gate that is constant at any
    positive level reproduces the baseline exactly, so what matters is the spread.
    `R` is `None` when `E[w] = 0`, which is a degenerate gate rather than a small one.
    """
    import numpy as np

    cos_np = np.asarray(cos.numpy(), dtype=np.float64)
    w_np = np.asarray(w.numpy(), dtype=np.float64)
    n_triples = int(cos_np.size)
    counts, edges = np.histogram(cos_np, bins=COS_HIST_BINS, range=COS_HIST_RANGE)

    w_mean = float(w_np.mean())
    reallocation = (
        float(np.abs(w_np - w_mean).mean() / (2.0 * w_mean)) if w_mean > 0 else None
    )
    return {
        "n_windows": int(n_windows),
        "n_triples": n_triples,
        "cos_mean": float(cos_np.mean()),
        "cos_median": float(np.median(cos_np)),
        "frac_cos_lt_0": float((cos_np < 0.0).mean()),
        "frac_cos_lt_0p5": float((cos_np < 0.5).mean()),
        "cos_histogram": {
            "bins": COS_HIST_BINS,
            "range": list(COS_HIST_RANGE),
            "edges": [float(e) for e in edges],
            "counts": [int(c) for c in counts],
        },
        "gate_mean": w_mean,
        "gate_zero_frac": float((w_np == 0.0).mean()),
        # Reported so gate check 1c can compare training's per-step `acs_gate_p*`
        # against this population distribution rather than against a single moment.
        "gate_p10": float(np.percentile(w_np, 10)),
        "gate_p50": float(np.percentile(w_np, 50)),
        "gate_p90": float(np.percentile(w_np, 90)),
        "reallocation_R": reallocation,
    }


def measure_action_similarity(train_cfg, split, reductions, gate):
    """One split of one environment, every requested reduction."""
    dset, act, meta = load_action_windows(train_cfg, split)
    per_reduction = {}
    for reduction in reductions:
        cos, w = cos_and_gate(act, reduction, gate, meta["env_action_dim"])
        per_reduction[reduction] = summarize_cos_and_gate(cos, w, int(act.shape[0]))
    n_windows = int(act.shape[0])
    return {
        "n_windows": n_windows,
        "n_triples": n_windows * meta["triples_per_window"],
        "reductions": per_reduction,
    }, dset, act, meta


def resolve_acs_out_path(raw_out, env_name, multi_env):
    """
    One machine-readable JSON per environment (ACS Requirement 1.18).

    Default `probe_outputs/acs_actions_<env>.json`. A directory, or anything without
    a `.json` suffix, gets that basename inside it; an explicit `.json` is honoured
    for a single environment and disambiguated per environment for several, so the
    loop in the design's Stage-0 recipe and a bare `--readout actions` both land one
    file per env instead of overwriting one file four times.
    """
    if raw_out is None:
        out = Path(os.getcwd()) / ACS_OUT_DIR / f"{ACS_OUT_PREFIX}_{env_name}.json"
        return Path(os.path.normpath(str(out)))
    out = Path(raw_out).expanduser()
    if not out.is_absolute():
        out = Path(os.getcwd()) / out
    out = Path(os.path.normpath(str(out)))
    if out.is_dir() or raw_out.endswith(("/", "\\")) or out.suffix.lower() != ".json":
        return out / f"{ACS_OUT_PREFIX}_{env_name}.json"
    if multi_env:
        return out.with_name(f"{out.stem}_{env_name}{out.suffix}")
    return out


ACS_REPORT_SCHEMA = "acs_actions/1"

# Written into every report, so the limits travel with the numbers rather than
# living only in PROGRESS_ACS.md (ACS Requirement 3.9).
ACS_REPORT_NOTES = (
    "n = 4 environments with no independent replicates: this readout can refute the "
    "mechanism ordering, not establish it.",
    "The four environments carry differently-typed action variables (PushT relative "
    "pusher displacements, PointMaze forces/velocity commands, Wall dot velocities), "
    "so cos(a_t, a_{t+1}) is not the same physical quantity across the four points.",
    f"frameskip={ACS_FRAMESKIP} may wash out reversals that occur inside a single "
    f"latent step; the `raw` reduction is reported beside `sum` for that reason.",
    "R, not mean(w), is the statistic that gates: a gate constant at any positive "
    "level reproduces the baseline exactly (design 11.3).",
    "cos is recovered from the shipped affine_cos gate as 2w-1; there is no second "
    "cosine-of-actions implementation in this file (ACS Requirement 15.3).",
)


def build_actions_report(env_name, train_cfg, args, reductions, splits, meta):
    """The per-environment Stage-0 report. Machine-readable first, printed second."""
    try:
        save_name = train_cfg.env.get("save_name", None) or env_name
    except Exception:  # noqa: BLE001 - a config without save_name is the common case
        save_name = env_name
    return {
        "schema": ACS_REPORT_SCHEMA,
        "readout": "actions",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "env": {"name": env_name, "save_name": str(save_name)},
        "protocol": {
            "num_hist": ACS_NUM_HIST,
            "num_pred": ACS_NUM_PRED,
            "frameskip": ACS_FRAMESKIP,
            "composed_from": "conf/train.yaml",
        },
        "gate": args.acs_gate,
        "action_reductions": list(reductions),
        "headline_split": args.split,
        "action_layout": meta,
        "splits": splits,
        "gpu_allocated": False,
        "video_decoded": False,
        "notes": list(ACS_REPORT_NOTES),
    }


def print_actions_report(report):
    """One table per split, one row per reduction, denominators attached."""
    env = report["env"]["name"]
    print()
    print(RULE)
    print(f"Stage-0 action similarity  --  env={env}  gate={report['gate']}  "
          f"(num_hist={ACS_NUM_HIST}, num_pred={ACS_NUM_PRED}, "
          f"frameskip={ACS_FRAMESKIP})")
    print(RULE)
    headline = report["headline_split"]
    for split in sorted(report["splits"], key=lambda s: (s != headline, s)):
        block = report["splits"][split]
        tag = "headline" if split == headline else "cross-check"
        print(f"\n{split} ({tag}): n_windows={block['n_windows']} "
              f"n_triples={block['n_triples']}")
        print(THIN)
        print(f"{'reduce':<8}{'mean':>10}{'median':>10}{'cos<0':>9}{'cos<0.5':>9}"
              f"{'mean(w)':>10}{'w=0':>9}{'R':>9}")
        print(THIN)
        for reduction, stats in block["reductions"].items():
            print(f"{reduction:<8}"
                  f"{_fmt(stats['cos_mean'], 10, 4)}"
                  f"{_fmt(stats['cos_median'], 10, 4)}"
                  f"{_fmt(stats['frac_cos_lt_0'], 9, 4)}"
                  f"{_fmt(stats['frac_cos_lt_0p5'], 9, 4)}"
                  f"{_fmt(stats['gate_mean'], 10, 4)}"
                  f"{_fmt(stats['gate_zero_frac'], 9, 4)}"
                  f"{_fmt(stats['reallocation_R'], 9, 4)}")
    print()
    print("R = E|w - E[w]| / (2 E[w]): the share of straightening pressure the gate "
          "relocates.")
    print("mean(w) is reported because it is interpretable, NOT because it gates: a "
          "constant gate")
    print("at any positive level reproduces the baseline exactly (design 11.3).")
    print(RULE)


def main_actions(args):
    """
    `--readout actions`: the Stage-0 premise test. No checkpoint, no model weights,
    no GPU, no video decode -- minutes of CPU, and it can kill the feature.
    """
    reductions = (list(ACS_ACTION_REDUCTIONS) if args.acs_action_reduce == "all"
                  else [args.acs_action_reduce])
    envs = list(dict.fromkeys(args.env))
    multi_env = len(envs) > 1
    require_dataset_dir()
    # The slicer permutes its own slice list with the global numpy RNG, so seeding here
    # is what makes two Stage-0 runs -- and the 32-window bitwise check -- reproducible.
    # Every statistic below is order-invariant regardless.
    _seed_everything(PROBE_SEED)

    written = []
    for env_name in envs:
        try:
            train_cfg = compose_env_cfg(env_name)
        except Exception as exc:  # noqa: BLE001 - a compose failure is a setup problem
            print(f"ERROR: could not compose env={env_name} from conf/train.yaml "
                  f"({type(exc).__name__}: {exc}). --readout actions needs the repo's "
                  f"conf/ tree and hydra installed; DATASET_DIR must be set because "
                  f"env.dataset.data_path interpolates it.", file=sys.stderr)
            return 1
        out_path = resolve_acs_out_path(args.out, env_name, multi_env)

        splits = {}
        meta = None
        for split in ACS_SPLITS:
            # Both splits every time: `--split` picks the headline, and the other one
            # is the cross-check (ACS Requirement 1.12), not an opt-in.
            try:
                block, _dset, _act, meta = measure_action_similarity(
                    train_cfg, split, reductions, args.acs_gate)
            except FileNotFoundError as exc:
                print(f"ERROR: could not read the {split} split of env={env_name} "
                      f"({exc}). Check DATASET_DIR and "
                      f"env.dataset.data_path.", file=sys.stderr)
                return 1
            splits[split] = block

        report = build_actions_report(env_name, train_cfg, args, reductions, splits,
                                      meta)
        write_report(out_path, report)
        print_actions_report(report)
        print(f"\nReport written to {out_path}")
        written.append(out_path)

    if multi_env:
        log.info("Wrote %s Stage-0 report(s): %s", len(written),
                 ", ".join(str(p) for p in written))
    return 0


# --------------------------------------------------------------------------
# 9. `--readout actions --summarize`: the Stage-0 verdict rules (ACS task 4.3)
# --------------------------------------------------------------------------
# Everything from here to `stage0_stats_from_reports` is a PURE FUNCTION over plain
# dicts of statistics: no file, no dataset, no tensor, no clock, no torch. That is
# deliberate. This is the one function in the feature that can kill it, so it has to
# be callable straight from a unit test at, just below and just above every
# pre-registered boundary (task 4.4), and a data-fitted adjustment has to be
# impossible to make without editing a named threshold constant above.
def parse_table1_gains(raw):
    """
    `"umaze=50.00,medium=10.67,wall=10.67,pusht=7.33"` -> `{"umaze": 50.0, ...}`.

    All four rule keys are required. A partial mapping would silently change which
    ordering rule A is reported against.
    """
    gains = {}
    for chunk in str(raw).split(","):
        item = chunk.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                f"--table1-gains entry {item!r} is not KEY=VALUE; expected e.g. "
                f"{ACS_TABLE1_GAINS_DEFAULT!r}")
        key, _, value = item.partition("=")
        key = key.strip()
        if key not in ACS_RULE_ENV_KEYS:
            raise ValueError(
                f"--table1-gains key {key!r} is not one of "
                f"{', '.join(ACS_RULE_ENV_KEYS)}")
        if key in gains:
            raise ValueError(f"--table1-gains names {key!r} twice")
        try:
            gains[key] = float(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"--table1-gains value for {key!r} is not a number: {value.strip()!r}")
    missing = [key for key in ACS_RULE_ENV_KEYS if key not in gains]
    if missing:
        raise ValueError(
            f"--table1-gains is missing {', '.join(missing)}; all four of "
            f"{', '.join(ACS_RULE_ENV_KEYS)} are required")
    return gains


def _ratio(numerator, denominator):
    """`num / den`, with both degenerate cases named instead of raising."""
    if denominator > 0:
        return numerator / denominator
    if numerator > 0:
        return math.inf          # a positive fraction against a zero one
    return None                  # 0 / 0: there is no ratio to report


def _fractions_for_rule_a(frac_cos_lt_0):
    """Validate the rule-A input: exactly the four keys, each a fraction in [0, 1]."""
    if not isinstance(frac_cos_lt_0, dict):
        raise ValueError(f"rule A takes a dict of frac(cos<0) per environment, got "
                         f"{type(frac_cos_lt_0).__name__}")
    missing = [key for key in ACS_RULE_ENV_KEYS if key not in frac_cos_lt_0]
    if missing:
        raise ValueError(
            f"rule A needs frac(cos<0) for all four environments; missing "
            f"{', '.join(missing)} (have {', '.join(sorted(frac_cos_lt_0)) or 'none'})")
    unknown = [key for key in frac_cos_lt_0 if key not in ACS_RULE_ENV_KEYS]
    if unknown:
        raise ValueError(
            f"rule A does not know the environment key(s) {', '.join(sorted(unknown))}; "
            f"the four keys are {', '.join(ACS_RULE_ENV_KEYS)}")
    values = {}
    for key in ACS_RULE_ENV_KEYS:
        try:
            value = float(frac_cos_lt_0[key])
        except (TypeError, ValueError):
            raise ValueError(f"frac(cos<0) for {key} is not a number: "
                             f"{frac_cos_lt_0[key]!r}")
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError(f"frac(cos<0) for {key} is {value}, which is not a "
                             f"fraction in [0, 1]")
        values[key] = value
    return values


def rule_a_verdict(frac_cos_lt_0):
    """
    Rule A -- the mechanism-ordering test (PROGRESS_ACS.md section 4.1, design 11.4).

    | PushT is the **highest** of the four AND exceeds each of Wall / UMaze / Medium
      by `>= 1.5x` AND UMaze is the **lowest**                        | **GO**     |
    | PushT is highest but the remaining ordering inverts (UMaze not lowest), or its
      margin over the largest of the other three is in `[1.1x, 1.5x)` | **MIDDLE** |
    | PushT is **not** the highest, or is within `1.1x` of the smoothest | **STOP** |

    The two STOP conditions are tested first, then GO, then MIDDLE, which makes the
    rule **total**: every assignment of four fractions lands on exactly one verdict,
    including the region the prose does not name explicitly (PushT highest, clearly
    above the smoothest, but under 1.1x of the largest other). That region is a
    MIDDLE: PushT *is* the highest and the ordering behind it is not the predicted
    one, which is the MIDDLE clause, and it is not one of the two STOP conditions.

    `reversal_structure` -- "this reduction shows reversal structure" -- is exactly
    the negation of the two STOP conditions, so it introduces no new threshold. It is
    what Requirement 3.6's `sum`-versus-`raw` comparison reads.

    Both comparisons are made on the **reported** ratios, so the printed number and
    the verdict can never disagree. Float64 being what it is, `0.30 / 0.20` is
    `1.4999999999999998` and lands on the MIDDLE side of `1.5x`: a boundary test
    should scale by a power of two, where the division is exact -- `0.25 -> 0.375`
    sits exactly on `1.5x`, and `0.5 -> 1.1 * 0.5` exactly on `1.1x`. A measured
    fraction with
    sixteen significant digits never sits on a boundary anyway. Where it does, the
    tie breaks toward the more cautious verdict, which is the direction to err in.
    """
    values = _fractions_for_rule_a(frac_cos_lt_0)
    pusht = values["pusht"]
    others = {key: value for key, value in values.items() if key != "pusht"}
    largest_other_key = max(others, key=lambda key: (others[key], key))
    largest_other = others[largest_other_key]
    smoothest_key = min(values, key=lambda key: (values[key], key))
    smoothest = values[smoothest_key]
    umaze_others = [value for key, value in values.items() if key != "umaze"]

    pusht_is_highest = pusht > largest_other
    umaze_is_lowest = values["umaze"] < min(umaze_others)
    margin = _ratio(pusht, largest_other)
    ratio_to_smoothest = _ratio(pusht, smoothest)
    indistinguishable = (ratio_to_smoothest is not None
                         and ratio_to_smoothest < RULE_A_INDISTINGUISHABLE)

    if not pusht_is_highest:
        verdict, clause = VERDICT_STOP, "2.6"
        reason = (f"PushT's frac(cos<0) = {pusht:.6g} is not the highest of the four "
                  f"({largest_other_key} = {largest_other:.6g}). The premise -- that "
                  f"PushT's control zigzags where the mazes' does not -- is false in "
                  f"the data, so the feature is not built.")
    elif indistinguishable:
        verdict, clause = VERDICT_STOP, "2.7"
        reason = (f"PushT's frac(cos<0) = {pusht:.6g} is within "
                  f"{RULE_A_INDISTINGUISHABLE}x of the smoothest environment "
                  f"({smoothest_key} = {smoothest:.6g}, ratio "
                  f"{ratio_to_smoothest:.4g}x), i.e. indistinguishable from it. The "
                  f"feature is not built.")
    elif margin is not None and margin >= RULE_A_CLEAR_MARGIN and umaze_is_lowest:
        verdict, clause = VERDICT_GO, "2.2"
        reason = (f"PushT's frac(cos<0) = {pusht:.6g} is the highest of the four and "
                  f"exceeds the largest of the other three ({largest_other_key} = "
                  f"{largest_other:.6g}) by {margin:.4g}x >= "
                  f"{RULE_A_CLEAR_MARGIN}x, and UMaze ({values['umaze']:.6g}) is the "
                  f"lowest. The ordering is consistent with the mechanism story; the "
                  f"claim it licenses is still weak (n = 4, no replicates).")
    else:
        verdict = VERDICT_MIDDLE
        if not umaze_is_lowest:
            clause = "2.3"
            reason = (f"PushT's frac(cos<0) = {pusht:.6g} is the highest of the four, "
                      f"but the remaining ordering inverts: UMaze "
                      f"({values['umaze']:.6g}) is not the lowest "
                      f"({smoothest_key} = {smoothest:.6g} is). Build ACS, but the "
                      f"mechanism claim is downgraded to \"the gate is a useful "
                      f"inductive bias\" and the writeup must not claim ACS explains "
                      f"the Table 1 gain ordering.")
        else:
            clause = "2.4"
            reason = (f"PushT's frac(cos<0) = {pusht:.6g} is the highest of the four "
                      f"and UMaze ({values['umaze']:.6g}) is the lowest, but the "
                      f"margin over the largest of the other three "
                      f"({largest_other_key} = {largest_other:.6g}) is "
                      f"{margin:.4g}x, short of {RULE_A_CLEAR_MARGIN}x. Build ACS, "
                      f"but the mechanism claim is downgraded to \"the gate is a "
                      f"useful inductive bias\" and the writeup must not claim ACS "
                      f"explains the Table 1 gain ordering.")

    # A ratio against a zero fraction is `inf`, and `inf` has no JSON literal, so the
    # report carries `null` plus an explicit `_unbounded` flag rather than emitting a
    # number no strict parser will read back.
    return {
        "rule": "A",
        "name": "mechanism ordering",
        "verdict": verdict,
        "clause": clause,
        "reason": reason,
        "frac_cos_lt_0": {key: _round(value) for key, value in values.items()},
        "pusht": _round(pusht),
        "pusht_is_highest": bool(pusht_is_highest),
        "largest_other": {"env": largest_other_key, "value": _round(largest_other)},
        "smoothest": {"env": smoothest_key, "value": _round(smoothest)},
        "umaze_is_lowest": bool(umaze_is_lowest),
        "margin_over_largest_other": (
            _round(margin) if margin is None or math.isfinite(margin) else None),
        "margin_over_largest_other_unbounded": margin == math.inf,
        "ratio_to_smoothest": (
            _round(ratio_to_smoothest)
            if ratio_to_smoothest is None or math.isfinite(ratio_to_smoothest)
            else None),
        "ratio_to_smoothest_unbounded": ratio_to_smoothest == math.inf,
        "reversal_structure": bool(pusht_is_highest and not indistinguishable),
        "thresholds": {"clear_margin": RULE_A_CLEAR_MARGIN,
                       "indistinguishable": RULE_A_INDISTINGUISHABLE},
        "caps_applied": [],
    }


def rule_b_verdict(reallocation_R):
    """
    Rule B -- the reallocation test (PROGRESS_ACS.md section 4.2, design 11.4).

    | `R >= 0.15`         | **GO**     |
    | `0.08 <= R < 0.15`  | **MIDDLE** -- small expected effect; `acs_gate=hard` or a
                            sharpened gate is the pre-declared remedy               |
    | `R < 0.08`          | **STOP**                                                |

    Independent of rule A, and it can STOP on its own. `R = None` -- the probe's
    encoding of `E[w] = 0` -- is a STOP: a gate that zeroes every triple reallocates
    nothing, it switches the term off.

    `mean(w)` is deliberately not consulted. Because ACS is a weighted *mean*, a gate
    constant at any positive level reproduces the baseline exactly, so the level says
    nothing and only the spread gates.
    """
    if reallocation_R is None:
        return {
            "rule": "B",
            "name": "reallocation",
            "verdict": VERDICT_STOP,
            "clause": "2.11",
            "reason": ("R is undefined because mean(w) = 0: the gate zeroes every "
                       "triple, so the term reallocates nothing and is identically "
                       "0 rather than weakly reweighted."),
            "R": None,
            "thresholds": {"go": RULE_B_GO, "middle": RULE_B_MIDDLE},
            "caps_applied": [],
        }
    try:
        value = float(reallocation_R)
    except (TypeError, ValueError):
        raise ValueError(f"R is not a number: {reallocation_R!r}")
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"R is {value}, which is not a non-negative finite number; "
                         f"R = E|w - E[w]| / (2 E[w]) lies in [0, 1)")

    if value >= RULE_B_GO:
        verdict, clause = VERDICT_GO, "2.8"
        reason = (f"PushT's R = {value:.6g} >= {RULE_B_GO}: the gate relocates "
                  f"{value * 100:.1f}% of the straightening pressure between triples.")
    elif value >= RULE_B_MIDDLE:
        verdict, clause = VERDICT_MIDDLE, "2.9"
        reason = (f"PushT's R = {value:.6g} is in [{RULE_B_MIDDLE}, {RULE_B_GO}): ACS "
                  f"may be built, the expected effect size is small, and "
                  f"acs_gate=hard or a sharpened gate is the pre-declared remedy "
                  f"(recorded before the data, not invented now).")
    else:
        verdict, clause = VERDICT_STOP, "2.11"
        reason = (f"PushT's R = {value:.6g} < {RULE_B_MIDDLE}: the term reallocates "
                  f"under {RULE_B_MIDDLE * 100:.0f}% of its mass and cannot plausibly "
                  f"produce a +4/+5 effect when the entire first-order straightening "
                  f"effect was +7.33 OL / +6.66 MPC.")
    return {
        "rule": "B",
        "name": "reallocation",
        "verdict": verdict,
        "clause": clause,
        "reason": reason,
        "R": _round(value),
        "thresholds": {"go": RULE_B_GO, "middle": RULE_B_MIDDLE},
        "caps_applied": [],
    }


def combine_rule_verdicts(rule_a, rule_b):
    """
    The combined verdict: **STOP if either rule is STOP** (PROGRESS_ACS.md 4.3).

    Stage 1 is permitted only when rule A is GO-or-MIDDLE AND rule B is GO-or-MIDDLE
    (Requirement 2.14). `GO` requires both rules to be GO; anything else that is not
    a STOP is a MIDDLE, so the more cautious of the two verdicts always survives.
    """
    for name, verdict in (("A", rule_a), ("B", rule_b)):
        if verdict not in VERDICTS:
            raise ValueError(f"rule {name} verdict {verdict!r} is not one of "
                             f"{', '.join(VERDICTS)}")
    if VERDICT_STOP in (rule_a, rule_b):
        verdict = VERDICT_STOP
        reason = ("At least one rule is STOP, so the combined verdict is STOP: the "
                  "ACS term, the gate and the action reducer are not implemented, "
                  "MCA_Fallback becomes the next arm, and the Stage-0 statistics are "
                  "written up as findings N1 and N2 regardless.")
    elif rule_a == VERDICT_GO and rule_b == VERDICT_GO:
        verdict = VERDICT_GO
        reason = ("Both rules are GO. A GO is permission to spend 0.8 GPU-h on the "
                  "Stage-1 arm, not evidence for the mechanism.")
    else:
        verdict = VERDICT_MIDDLE
        reason = ("Neither rule is STOP and at least one is MIDDLE: Stage 1 is "
                  "permitted with the downgraded claim recorded now, at the moment "
                  "the verdict is read, rather than retroactively.")
    return {
        "verdict": verdict,
        "rule_a": rule_a,
        "rule_b": rule_b,
        "stage1_permitted": rule_a != VERDICT_STOP and rule_b != VERDICT_STOP,
        "reason": reason,
    }


def _cap_verdict(block, cap, reason):
    """Lower a rule verdict to `cap` if it is more permissive; never raise it."""
    capped = dict(block)
    if VERDICT_SEVERITY[capped["verdict"]] > VERDICT_SEVERITY[cap]:
        capped["caps_applied"] = list(capped.get("caps_applied", ())) + [
            {"from": capped["verdict"], "to": cap, "reason": reason}]
        capped["verdict"] = cap
    return capped


def ordering_vs_table1_gains(frac_cos_lt_0, gains):
    """
    Rule A's story, counted: `frac(cos<0)` should order **inversely** to Table 1's
    straightening gains.

    Reported, never gating. Pairs whose gains are equal (Wall and Medium are both
    +10.67) carry no prediction and are skipped. And `n = 4` with no independent
    replicates, across environments carrying differently-typed action variables --
    PushT relative pusher displacements, PointMaze forces or velocity commands on a
    point mass, Wall dot velocities -- can refute this ordering but cannot establish
    it, which is why the verdict is read off rule A and not off this count.
    """
    values = _fractions_for_rule_a(frac_cos_lt_0)
    keys = list(ACS_RULE_ENV_KEYS)
    compared, concordant, tied = [], 0, []
    for i, first in enumerate(keys):
        for second in keys[i + 1:]:
            if gains[first] == gains[second]:
                tied.append([first, second])
                continue
            higher, lower = ((first, second) if gains[first] > gains[second]
                             else (second, first))
            # Inverse ordering: the environment with the LARGER gain should carry the
            # SMALLER reversal fraction.
            agrees = values[higher] < values[lower]
            concordant += int(agrees)
            compared.append({"higher_gain": higher, "lower_gain": lower,
                             "agrees": bool(agrees)})
    discordant = len(compared) - concordant
    return {
        "gains_open_loop": {key: _round(gains[key], 2) for key in keys},
        "observed_order_desc_frac": sorted(keys, key=lambda k: (-values[k], k)),
        "predicted_order_desc_frac": sorted(keys, key=lambda k: (gains[k], k)),
        "pairs_compared": len(compared),
        "pairs_concordant": concordant,
        "pairs_discordant": discordant,
        "tied_gain_pairs": tied,
        "matches_inverse_gains": discordant == 0,
        "pairs": compared,
        "note": ("Reported, not gating. n = 4, no independent replicates, and the "
                 "four environments carry differently-typed action variables, so "
                 "cos(a_t, a_{t+1}) is not the same physical quantity across the "
                 "four points being correlated."),
    }


def evaluate_stage0(stats_by_reduction, table1_gains=None):
    """
    Rules A and B on one split's statistics, plus Requirement 3.6's `sum`/`raw`
    comparison and the combined verdict.

    `stats_by_reduction` is `{reduction: {rule_env_key: {"frac_cos_lt_0": float,
    "reallocation_R": float | None}}}` -- exactly the fields
    `summarize_cos_and_gate` writes, and nothing more, so a unit test hands it a
    literal instead of a dataset.

    The verdict is read off the `sum` reduction, which is the shipped default
    `acs_action_reduce=sum` and therefore the gate training will actually apply.
    Every reduction present is evaluated as well and reported.
    """
    if ACS_HEADLINE_REDUCTION not in stats_by_reduction:
        raise ValueError(
            f"the verdict is read off the {ACS_HEADLINE_REDUCTION!r} reduction "
            f"(training's default acs_action_reduce), which is absent; have "
            f"{', '.join(sorted(stats_by_reduction)) or 'nothing'}")

    per_reduction = {}
    for reduction, per_env in sorted(stats_by_reduction.items()):
        missing = [key for key in ACS_RULE_ENV_KEYS if key not in per_env]
        if missing and reduction != ACS_HEADLINE_REDUCTION:
            per_reduction[reduction] = {
                "skipped": f"missing {', '.join(missing)}"}
            continue
        rule_a = rule_a_verdict(
            {key: per_env[key]["frac_cos_lt_0"] for key in ACS_RULE_ENV_KEYS})
        rule_b = rule_b_verdict(per_env["pusht"].get("reallocation_R"))
        per_reduction[reduction] = {"rule_a": rule_a, "rule_b": rule_b}

    rule_a = per_reduction[ACS_HEADLINE_REDUCTION]["rule_a"]
    rule_b = per_reduction[ACS_HEADLINE_REDUCTION]["rule_b"]

    # Requirement 3.6 / PROGRESS_ACS.md 4.1: frameskip=5 can wash out a reversal that
    # happens inside one latent step. If `sum` shows no reversal structure while `raw`
    # does, the reversals are inside the step -- where the latent velocity cannot see
    # them either -- so a verdict read off `raw` is capped at MIDDLE and can never be
    # GO. The cap is written as a cap, not as an upgrade: it can only lower a verdict,
    # so it never converts the headline STOP into permission to build.
    within = per_reduction.get(ACS_WITHIN_STEP_REDUCTION, {})
    raw_rule_a = within.get("rule_a")
    raw_structure = bool(raw_rule_a is not None and raw_rule_a["reversal_structure"])
    sum_structure = bool(rule_a["reversal_structure"])
    applies = raw_structure and not sum_structure
    cap_reason = ("Requirement 3.6: no reversal structure under `sum` but structure "
                  "under `raw` -- the reversals are inside a latent step, which the "
                  "latent velocity cannot see either. MIDDLE, never GO.")
    raw_capped = (_cap_verdict(raw_rule_a, VERDICT_MIDDLE, cap_reason)["verdict"]
                  if raw_rule_a is not None else None)
    if applies:
        rule_a = _cap_verdict(rule_a, VERDICT_MIDDLE, cap_reason)
    requirement_3_6 = {
        "measured": raw_rule_a is not None,
        "headline_reduction": ACS_HEADLINE_REDUCTION,
        "within_step_reduction": ACS_WITHIN_STEP_REDUCTION,
        "sum_has_reversal_structure": sum_structure,
        "raw_has_reversal_structure": raw_structure,
        "applies": bool(applies),
        "headline_rule_a_verdict": rule_a["verdict"],
        "raw_rule_a_verdict": raw_rule_a["verdict"] if raw_rule_a else None,
        "raw_rule_a_verdict_capped": raw_capped,
        "note": (cap_reason if applies else
                 ("`sum` shows reversal structure, so the downgrade does not fire; "
                  "`raw` is reported beside it because frameskip=5 can wash out "
                  "within-step reversals."
                  if sum_structure else
                  "Neither reduction shows reversal structure, so there is nothing "
                  "for the downgrade to rescue.")),
        "definition": ("\"reversal structure\" is the negation of rule A's two STOP "
                       "conditions -- PushT strictly the highest of the four and not "
                       "within 1.1x of the smoothest -- so this comparison introduces "
                       "no threshold that was not pre-registered."),
    }

    evaluation = {
        "headline_reduction": ACS_HEADLINE_REDUCTION,
        "rule_a": rule_a,
        "rule_b": rule_b,
        "requirement_3_6": requirement_3_6,
        "per_reduction": per_reduction,
        "combined": combine_rule_verdicts(rule_a["verdict"], rule_b["verdict"]),
    }
    if table1_gains is not None:
        evaluation["table1_ordering"] = ordering_vs_table1_gains(
            {key: stats_by_reduction[ACS_HEADLINE_REDUCTION][key]["frac_cos_lt_0"]
             for key in ACS_RULE_ENV_KEYS},
            table1_gains)
    return evaluation


# --- reading the per-environment reports back (I/O side) ----------------------
def expand_report_paths(raw_paths):
    """
    Resolve `--summarize` arguments, expanding globs here as well as in the shell.

    The design's Stage-0 recipe is a bash loop with `acs_actions_*.json`; on a shell
    that does not expand globs for a native command the pattern would arrive
    verbatim, and "no such file: probe_outputs/acs_actions_*.json" is a worse error
    than doing the expansion.
    """
    paths, seen = [], set()
    for raw in raw_paths:
        matches = ([Path(match) for match in sorted(globlib.glob(raw))]
                   if any(char in raw for char in "*?[") else [Path(raw)])
        if not matches:
            raise FileNotFoundError(f"--summarize pattern matched no file: {raw}")
        for match in matches:
            path = match.expanduser()
            if not path.is_absolute():
                path = Path(os.getcwd()) / path
            path = Path(os.path.normpath(str(path)))
            if str(path) not in seen:
                seen.add(str(path))
                paths.append(path)
    return paths


def load_actions_report(path):
    """One `acs_actions/1` report, validated hard enough that a wrong file is named."""
    if not path.is_file():
        raise FileNotFoundError(f"--summarize input does not exist: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        try:
            report = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} is not valid JSON ({exc})")
    if not isinstance(report, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    schema = report.get("schema")
    if schema != ACS_REPORT_SCHEMA:
        raise ValueError(
            f"{path} has schema {schema!r}, not {ACS_REPORT_SCHEMA!r}; --summarize "
            f"reads the per-environment reports written by --readout actions")
    return report


def report_rule_env_key(report, path):
    """`env=point_maze` -> the rule's `umaze`, so one naming runs through the rule."""
    env = report.get("env") or {}
    name = str(env.get("name", ""))
    if name in ACS_ENV_RULE_KEYS:
        return ACS_ENV_RULE_KEYS[name]
    save_name = str(env.get("save_name", ""))
    if save_name in ACS_RULE_ENV_KEYS:
        return save_name
    raise ValueError(
        f"{path} reports env={name!r} (save_name={save_name!r}), which is none of the "
        f"four environments the rule is written over "
        f"({', '.join(ACS_ENV_RULE_KEYS)})")


def stage0_stats_from_reports(reports, split):
    """
    `{reduction: {rule_env_key: statistics}}` for one split, plus the sources block.

    `reports` is a list of `(path, report)`. Two reports for the same environment, a
    report missing the requested split and a set of reports that disagree about the
    gate are all errors: each of them would produce a verdict whose inputs are not
    the ones it claims.
    """
    stats, sources, gates, protocols = {}, [], set(), set()
    for path, report in reports:
        key = report_rule_env_key(report, path)
        duplicates = [source for source in sources if source["env_key"] == key]
        if duplicates:
            raise ValueError(
                f"two --summarize inputs report the same environment {key!r}: "
                f"{duplicates[0]['path']} and {path}")
        splits = report.get("splits") or {}
        if split not in splits:
            raise ValueError(
                f"{path} has no {split!r} split (has "
                f"{', '.join(sorted(splits)) or 'none'})")
        block = splits[split]
        gates.add(str(report.get("gate")))
        protocol = report.get("protocol") or {}
        protocols.add((protocol.get("num_hist"), protocol.get("num_pred"),
                       protocol.get("frameskip")))
        reductions = block.get("reductions") or {}
        if not reductions:
            raise ValueError(f"{path}: the {split!r} split reports no reduction")
        for reduction, entry in reductions.items():
            for field in ("frac_cos_lt_0", "reallocation_R", "n_triples", "n_windows"):
                if field not in entry:
                    raise ValueError(
                        f"{path}: the {split!r}/{reduction} statistics have no "
                        f"{field!r}; the report was not written by --readout actions")
            stats.setdefault(reduction, {})[key] = {
                "frac_cos_lt_0": entry["frac_cos_lt_0"],
                "frac_cos_lt_0p5": entry.get("frac_cos_lt_0p5"),
                "reallocation_R": entry["reallocation_R"],
                "gate_mean": entry.get("gate_mean"),
                "gate_zero_frac": entry.get("gate_zero_frac"),
                "n_triples": entry["n_triples"],
                "n_windows": entry["n_windows"],
            }
        sources.append({
            "path": str(path),
            "env": str((report.get("env") or {}).get("name", "")),
            "env_key": key,
            "schema": report.get("schema"),
            "generated_at": report.get("generated_at"),
            "gate": report.get("gate"),
            "n_windows": block.get("n_windows"),
            "n_triples": block.get("n_triples"),
        })
    if len(gates) > 1:
        raise ValueError(
            f"the --summarize inputs were measured with different gates "
            f"({', '.join(sorted(gates))}); R is gate-dependent, so one verdict "
            f"cannot be read off a mixed set")
    if len(protocols) > 1:
        raise ValueError(
            f"the --summarize inputs were measured under different protocols "
            f"({sorted(protocols)}); num_hist/num_pred/frameskip must match")
    missing = [key for key in ACS_RULE_ENV_KEYS
               if key not in stats.get(ACS_HEADLINE_REDUCTION, {})]
    if missing:
        raise ValueError(
            f"the rule is written over all four environments and "
            f"{', '.join(missing)} {'is' if len(missing) == 1 else 'are'} missing from "
            f"the {ACS_HEADLINE_REDUCTION!r} reduction of the {split!r} split; run "
            f"--readout actions for {', '.join(missing)} first")
    return stats, sources, (gates.pop() if gates else None)


ACS_VERDICT_NOTES = (
    "Rules A and B, and every threshold in them (1.5x, 1.1x, 0.15, 0.08), were "
    "written into PROGRESS_ACS.md section 4 on 2026-08-08, before these statistics "
    "were collected. They are judgment calls, not derivations (Requirement 2.17).",
    "n = 4 environments with no independent replicates: rule A can refute the "
    "mechanism ordering, it cannot establish it.",
    "The four environments carry differently-typed action variables (PushT relative "
    "pusher displacements, PointMaze forces/velocity commands, Wall dot velocities), "
    "so cos(a_t, a_{t+1}) is not the same physical quantity across the four points. "
    "Structural, not a noise problem.",
    "A confirmed ordering is consistent with confounds other than the ACS mechanism: "
    "contact dynamics, the second movable object, rotational state, and 2 training "
    "epochs on PushT against 20 elsewhere.",
    "A GO is permission to spend 0.8 GPU-h on the Stage-1 arm, not evidence for the "
    "mechanism. A STOP is treated as decisive.",
    "R, not mean(w), is what gates rule B: a gate constant at any positive level "
    "reproduces the baseline exactly, so only the spread matters (design 11.3).",
    "On a rule-A MIDDLE the mechanism claim is downgraded to \"the gate is a useful "
    "inductive bias\" and the Table-1-ordering explanation is withheld. On a rule-B "
    "MIDDLE, acs_gate=hard or a sharpened gate is the pre-declared remedy and the "
    "expected effect size is small. Both are recorded when the verdict is read, not "
    "retroactively.",
)


def build_verdict_report(sources, split, gate, gains, evaluation, cross_check, stats):
    """The combined verdict report (ACS Requirement 1.18, task 4.3)."""
    return {
        "schema": ACS_VERDICT_SCHEMA,
        "readout": "actions",
        "summary_of": "stage0 pre-registered verdict rules A and B",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pre_registered_in": "PROGRESS_ACS.md section 4 (written 2026-08-08)",
        "headline_split": split,
        "headline_reduction": ACS_HEADLINE_REDUCTION,
        "gate": gate,
        "thresholds": {
            "rule_a_clear_margin": RULE_A_CLEAR_MARGIN,
            "rule_a_indistinguishable": RULE_A_INDISTINGUISHABLE,
            "rule_b_go": RULE_B_GO,
            "rule_b_middle": RULE_B_MIDDLE,
        },
        "sources": sources,
        "statistics": stats,
        "rule_a": evaluation["rule_a"],
        "rule_b": evaluation["rule_b"],
        "requirement_3_6": evaluation["requirement_3_6"],
        "per_reduction": evaluation["per_reduction"],
        "table1_ordering": evaluation.get("table1_ordering"),
        "combined": evaluation["combined"],
        "verdict": evaluation["combined"]["verdict"],
        "stage1_permitted": evaluation["combined"]["stage1_permitted"],
        "cross_check_split": cross_check,
        "table1_gains_open_loop": {key: gains[key] for key in ACS_RULE_ENV_KEYS},
        "notes": list(ACS_VERDICT_NOTES),
    }


def resolve_verdict_out_path(raw_out):
    """Default `probe_outputs/acs_stage0_verdict.json`; a directory gets that name."""
    if raw_out is None:
        out = Path(os.getcwd()) / ACS_OUT_DIR / ACS_VERDICT_BASENAME
        return Path(os.path.normpath(str(out)))
    out = Path(raw_out).expanduser()
    if not out.is_absolute():
        out = Path(os.getcwd()) / out
    out = Path(os.path.normpath(str(out)))
    if out.is_dir() or raw_out.endswith(("/", "\\")) or out.suffix.lower() != ".json":
        return out / ACS_VERDICT_BASENAME
    return out


def print_verdict_report(report):
    """The verdict, printed, with every driving number and its denominator."""
    stats = report["statistics"].get(report["headline_reduction"], {})
    gains = report["table1_gains_open_loop"]
    print()
    print(RULE)
    print(f"STAGE-0 VERDICT  --  pre-registered rules A and B "
          f"({report['pre_registered_in']})")
    print(RULE)
    print(f"  split / reduction     : {report['headline_split']} / "
          f"{report['headline_reduction']}    gate: {report['gate']}")
    print("  reports               : " + ", ".join(
        f"{source['env_key']}={Path(source['path']).name}"
        for source in report["sources"]))
    print()
    print(f"  {'env':<9}{'frac(cos<0)':>13}{'R':>9}{'mean(w)':>10}{'w=0':>9}"
          f"{'n_triples':>12}{'T1 gain':>10}")
    print(f"  {THIN[:70]}")
    for key in ACS_RULE_ENV_KEYS:
        entry = stats.get(key, {})
        print(f"  {key:<9}"
              f"{_fmt(entry.get('frac_cos_lt_0'), 13, 4)}"
              f"{_fmt(entry.get('reallocation_R'), 9, 4)}"
              f"{_fmt(entry.get('gate_mean'), 10, 4)}"
              f"{_fmt(entry.get('gate_zero_frac'), 9, 4)}"
              f"{str(entry.get('n_triples', 'n/a')):>12}"
              f"{_fmt(gains.get(key), 10, 2)}")

    for block in (report["rule_a"], report["rule_b"]):
        print()
        print(f"  RULE {block['rule']} ({block['name']}): {block['verdict']}  "
              f"[Requirement {block['clause']}]")
        print(f"    {block['reason']}")
        for cap in block.get("caps_applied", ()):
            print(f"    capped {cap['from']} -> {cap['to']}: {cap['reason']}")
    a = report["rule_a"]

    def _ratio_txt(value, unbounded):
        if unbounded:
            return "inf"
        return _fmt(value, 0, 4).strip()

    print()
    print(f"    rule A detail: PushT={_fmt(a['pusht'], 0, 4).strip()} highest="
          f"{a['pusht_is_highest']} margin over {a['largest_other']['env']}="
          f"{_ratio_txt(a['margin_over_largest_other'], a.get('margin_over_largest_other_unbounded'))}x "
          f"(GO needs >= {RULE_A_CLEAR_MARGIN}x) smoothest={a['smoothest']['env']} "
          f"ratio={_ratio_txt(a['ratio_to_smoothest'], a.get('ratio_to_smoothest_unbounded'))}x "
          f"(STOP below {RULE_A_INDISTINGUISHABLE}x) umaze_lowest="
          f"{a['umaze_is_lowest']}")

    r36 = report["requirement_3_6"]
    print()
    print(f"  REQUIREMENT 3.6 (sum vs raw): "
          f"{'DOWNGRADE APPLIES' if r36['applies'] else 'does not fire'}")
    print(f"    reversal structure: sum={r36['sum_has_reversal_structure']} "
          f"raw={r36['raw_has_reversal_structure']}; rule A on raw="
          f"{r36['raw_rule_a_verdict']} (capped at {r36['raw_rule_a_verdict_capped']})")
    print(f"    {r36['note']}")

    ordering = report.get("table1_ordering")
    if ordering:
        print()
        print(f"  TABLE-1 ORDERING (reported, not gating): "
              f"{'matches' if ordering['matches_inverse_gains'] else 'does NOT match'} "
              f"the inverse of the gains "
              f"({ordering['pairs_concordant']}/{ordering['pairs_compared']} "
              f"informative pairs)")
        print(f"    observed  desc frac(cos<0): "
              f"{' > '.join(ordering['observed_order_desc_frac'])}")
        print(f"    predicted desc frac(cos<0): "
              f"{' > '.join(ordering['predicted_order_desc_frac'])}")

    cross = report.get("cross_check_split") or {}
    if cross.get("evaluated"):
        print()
        print(f"  CROSS-CHECK ({cross['split']}): rule A={cross['rule_a']} "
              f"rule B={cross['rule_b']} combined={cross['verdict']} "
              f"(PushT R={_fmt(cross.get('pusht_R'), 0, 4).strip()})")
    elif cross.get("error"):
        print()
        print(f"  CROSS-CHECK ({cross.get('split')}): not evaluated -- "
              f"{cross['error']}")

    combined = report["combined"]
    print()
    print(RULE)
    print(f"COMBINED VERDICT: {combined['verdict']}   "
          f"(rule A={combined['rule_a']}, rule B={combined['rule_b']})")
    print(f"Stage 1 permitted: {'YES' if combined['stage1_permitted'] else 'NO'}")
    print(RULE)
    print(f"  {combined['reason']}")
    if combined["verdict"] == VERDICT_STOP:
        print("  Tasks 6.x onward are NOT executed: no compute_acs, no gate, no "
              "action reducer, no ACS code path.")
    print()
    for note in report["notes"]:
        print(f"  - {note}")
    print(RULE)


def main_summarize(args):
    """
    `--readout actions --summarize`: evaluate the pre-registered rules, emit one
    combined verdict JSON and print the verdict.

    Reads JSON only. No dataset, no DATASET_DIR, no model, no torch.
    """
    try:
        gains = parse_table1_gains(args.table1_gains)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    try:
        paths = expand_report_paths(args.summarize)
        reports = [(path, load_actions_report(path)) for path in paths]
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    split = args.split
    try:
        stats, sources, gate = stage0_stats_from_reports(reports, split)
        evaluation = evaluate_stage0(stats, gains)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # The other split, evaluated the same way and labelled a cross-check: the verdict
    # is read off the headline split, and a rule that flips between splits is a fact
    # about the measurement worth having on the record (Requirement 1.12).
    other = next((name for name in ACS_SPLITS if name != split), None)
    cross_check = {"split": other, "evaluated": False}
    if other is not None:
        try:
            other_stats, _sources, _gate = stage0_stats_from_reports(reports, other)
            other_eval = evaluate_stage0(other_stats, gains)
            cross_check = {
                "split": other,
                "evaluated": True,
                "rule_a": other_eval["rule_a"]["verdict"],
                "rule_b": other_eval["rule_b"]["verdict"],
                "verdict": other_eval["combined"]["verdict"],
                "pusht_frac_cos_lt_0": other_stats[ACS_HEADLINE_REDUCTION]["pusht"][
                    "frac_cos_lt_0"],
                "pusht_R": other_stats[ACS_HEADLINE_REDUCTION]["pusht"][
                    "reallocation_R"],
                "note": ("Cross-check only. The verdict is read off the "
                         f"{split!r} split, which is what training sees."),
            }
        except ValueError as exc:
            cross_check = {"split": other, "evaluated": False, "error": str(exc)}

    report = build_verdict_report(sources, split, gate, gains, evaluation,
                                  cross_check, stats)
    out_path = resolve_verdict_out_path(args.out)
    write_report(out_path, report)
    print_verdict_report(report)
    print(f"\nVerdict written to {out_path}")
    log.info("Stage-0 combined verdict: %s (rule A=%s, rule B=%s); Stage 1 %s.",
             report["verdict"], report["rule_a"]["verdict"],
             report["rule_b"]["verdict"],
             "permitted" if report["stage1_permitted"] else "NOT permitted")
    return 0


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.summarize is not None and args.readout != "actions":
        parser.error(f"--summarize evaluates the Stage-0 verdict rules on "
                     f"{ACS_OUT_PREFIX}_<env>.json reports and requires "
                     f"--readout actions, not --readout {args.readout}")

    if args.readout == "actions":
        # The Stage-0 readout takes no checkpoint, so the two path flags below are
        # neither required nor read.
        if args.summarize is not None:
            return main_summarize(args)
        return main_actions(args)

    # `--ckpt` / `--train-cfg` stopped being argparse-`required` when `--readout`
    # landed, because `actions` has nothing to point them at. Reinstated here with
    # the same message and the same exit code 2 argparse produced.
    missing = [flag for flag, value in (("--ckpt", args.ckpt),
                                        ("--train-cfg", args.train_cfg))
               if value is None]
    if missing:
        parser.error(f"the following arguments are required with "
                     f"--readout {args.readout}: {', '.join(missing)}")
    return main_curvature(args)


def main_curvature(args):
    # ---- 1. paths first, before any model or weight (Requirement 7.5) ----
    ckpt_path, cfg_path = validate_paths(args.ckpt, args.train_cfg)
    out_path = resolve_out_path(args.out if args.out is not None else DEFAULT_OUT,
                                ckpt_path)
    budget = Budget(args.max_minutes)

    # ---- 2. fingerprint the checkpoint (Requirement 7.4) ----
    before = file_fingerprint(ckpt_path)
    log.info("Checkpoint %s: sha256=%s size=%s bytes mtime=%s",
             ckpt_path, before["sha256"], before["size_bytes"], before["mtime"])

    from omegaconf import OmegaConf
    train_cfg = OmegaConf.load(cfg_path)
    num_frames = int(train_cfg.num_hist) + int(train_cfg.num_pred)
    synthesized = max(0, int(train_cfg.num_hist) + int(args.rollout_len) - 1 - num_frames)

    # ---- 3. load the model, read-only, CPU-only, no optimizer ----
    model, epoch = load_probe_model(ckpt_path, train_cfg)
    configure_ccr(model, args.rho, args.rollout_len, args.action_source)
    # Rejected here -- with training's own message -- before the dataset (the
    # expensive part) is touched.
    reject_infeasible_action_source(model, num_frames)
    log.info("Probing arm: rho=%s rollout_len=%s action_source=%s "
             "synthesized_action_frames=%s (num_hist=%s, num_frames=%s)",
             args.rho, args.rollout_len, args.action_source, synthesized,
             train_cfg.num_hist, num_frames)
    if args.rho == 0:
        log.warning("--rho 0 makes the perturbed and unperturbed passes the identical "
                    "computation, so curvature_gap is 0 by construction and the probe "
                    "gate cannot pass. This is the control arm, not a measurement of "
                    "the mechanism.")
    if args.action_source == "synthetic" and synthesized == 0:
        log.warning("--action-source synthetic with synthesized_action_frames=0 is "
                    "silently the `logged` arm: rollout_len=%s fits inside "
                    "num_frames=%s, so no action frame is synthesized.",
                    args.rollout_len, num_frames)

    # ---- 4. windows from the unmodified loader at a fixed seed ----
    try:
        windows, dims = load_windows(train_cfg, args.num_windows)
    except FileNotFoundError as exc:
        print(f"ERROR: could not read the dataset ({exc}). Check DATASET_DIR and "
              f"env.dataset.data_path in {cfg_path}.", file=sys.stderr)
        return 1

    # ---- 5. measure ----
    measurement = measure(model, windows, args.rho, args.draws, budget,
                          budget.deadline(MAIN_BUDGET_FRACTION), "checkpoint")
    if measurement["windows"] == 0:
        print("ERROR: no window produced a finite curvature value; there is nothing "
              "to report.", file=sys.stderr)
        return 1

    # ---- 6. readouts, disaggregated per state dimension ----
    readouts = readouts_from_measurement(measurement)

    # ---- 7. reference value for every readout ----
    ctx = {
        "train_cfg": train_cfg,
        "windows": windows,
        "dims": dims,
        "rho": args.rho,
        "rollout_len": args.rollout_len,
        "action_source": args.action_source,
        "draws": args.draws,
        "budget": budget,
        "reference_deadline": budget.deadline(REFERENCE_BUDGET_FRACTION),
        "run_dir": cfg_path.parent,
        "out_path": out_path,
        "ckpt_sha256": before["sha256"],
        "reference_details": {},
    }
    references = resolve_references(args.reference, ctx)

    # ---- 8. gate, re-hash, report ----
    gate = evaluate_gate(readouts["_curvature_detail"])
    after = file_fingerprint(ckpt_path)
    report = build_report(args, ckpt_path, cfg_path, before, after, train_cfg,
                          measurement, readouts, references, gate, budget,
                          synthesized, num_frames, ctx["reference_details"], epoch)
    write_report(out_path, report)
    print_report(report)
    print(f"\nReport written to {out_path}")

    if report["checkpoint_modified"]:
        # The report is still written, for forensics, and then this is fatal.
        raise RuntimeError(
            f"The checkpoint {ckpt_path} CHANGED while the probe was running: "
            f"sha256 {before['sha256']} -> {after['sha256']}. The probe is read-only, "
            f"so something else wrote to it; every number in {out_path} is suspect."
        )
    log.info("Checkpoint unchanged (sha256 %s); the probe wrote only %s.",
             after["sha256"], out_path)
    # `partial` is not a failure: the wall-clock guard exits 0 (Requirement 7.6).
    return 0


if __name__ == "__main__":
    sys.exit(main())
