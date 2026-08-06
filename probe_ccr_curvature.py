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
"""
import argparse
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
    ap.add_argument("--ckpt", required=True,
                    help="path to model_<epoch>.pth (read-only; never modified)")
    ap.add_argument("--train-cfg", required=True,
                    help="path to the run's resolved hydra.yaml")
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
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help=f"report path or directory (default {DEFAULT_OUT}); must not "
                         f"be inside the checkpoint directory")
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


def _freeze_for_probe(model):
    """
    eval() + requires_grad(False). Belt and braces: every measurement already runs
    under `torch.no_grad()`, so this only makes "no populated gradient" true by
    construction rather than by discipline (Property 14).
    """
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
# main
# --------------------------------------------------------------------------
def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # ---- 1. paths first, before any model or weight (Requirement 7.5) ----
    ckpt_path, cfg_path = validate_paths(args.ckpt, args.train_cfg)
    out_path = resolve_out_path(args.out, ckpt_path)
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
