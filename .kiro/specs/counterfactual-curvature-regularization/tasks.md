# Implementation Plan: Counterfactual Curvature Regularization (CCR)

## Overview

Python 3.10 / PyTorch / Hydra, matching the rest of the repo. The plan is ordered by what gates what:

1. **Test scaffolding first.** `tests/conftest.py` (tiny stub encoder) is a dependency of every property
   test, and `tests/reference_impl.py` (frozen pre-feature `forward` tail and `rollout` loop) must exist
   *before* the `_rollout_latents` refactor, since it is the model the refactor is compared against
   (Properties 1 and 7).
2. **Pure refactor before any new behaviour.** `_rollout_latents` is extracted with Property 7 passing
   before a line of CCR code is written on top of it.
3. **Config + run naming before any run-directory work.** The `ccr_tag` resolver and the `conf/train.yaml`
   keys gate the loss-signature guard and the byte-identical legacy run-name test (Requirement 3.4,
   Property 2), which must pass before any run is launched.
4. **Pilot infrastructure before the first pilot.** Iteration cap, checkpointed `global_iter`, run-directory
   loss-signature guard, JSONL telemetry and `summarize_training_log.py` must be complete and tested,
   because the pilot verdict is read out of that telemetry.
5. **Probe gates the ladder.** No Pilot_Run launches until `probe_ccr_curvature.py` passes its written gate
   (Requirement 11.3): aggregate `curvature_gap` positive and at least 20% of unperturbed curvature on at
   least 3 of the 5 disaggregated dimensions.
6. **Pilots are serial**, one job at a time — the `1g.45gb` MIG slice holds exactly one job
   (Requirement 9.7). The graph encodes this as one arm per wave.
7. **Full_Run is gated on the pilot gate**; the 3-seed evaluation and the Acceptance_Gate come last, and the
   Platform_Baseline (~75.3 / ~82.0) is **re-measured** under Evaluation_Protocol rather than taken from the
   recorded number.

Task labels:

- **[CODE]** — an agent can write and run this locally on CPU. No GPU, no dataset download, no network.
- **[GPU RUN]** / **[CPU RUN]** — an operator launches a job. Not agent-executable.
- **[HUMAN]** — a judgement, verdict or approval request. Not agent-executable.

## Tasks

- [x] 1. Test harness foundation
  - [x] 1.1 [CODE] Add `hypothesis` as a test-only dependency
    - Create `requirements-dev.txt` with pinned `hypothesis` and `pytest`; add a `pytest.ini` (or `[pytest]`
      section) registering the `tests/` rootdir and a minimum of 100 examples per property
    - **`hypothesis` MUST NOT be added to `requirements-train.txt` or `requirements-plan.txt`** — the
      training and planning images stay unchanged
    - _Requirements: 5.6_

  - [x] 1.2 [CODE] Create `tests/conftest.py` with the CPU test doubles
    - Stub encoder `nn.Module`: `name = "tiny"` (identity `encoder_transform` branch), `emb_dim = 4`,
      `latent_ndim = 2`, `patch_size = 2`, `forward` returning `(b*t, p=4, d=4)`, and an `agg` implementing
      the same `mean | flatten | mlp` contract as `models/dino.py`
    - Stub predictor as a `Linear` over tokens; proprio/action encoders from `models/proprio.py`;
      `decoder=None`; float32 on CPU
    - Shared fixtures/strategies: batch 1-4, `num_frames` 3-6, `num_hist` 2-3, `rho` in `[0, 1]` including
      exactly 0, `lambda_cf` in `[0, 10]`, `ccr_rollout_len` 1-6, `ccr_action_source` in
      `{logged, synthetic}`, `agg_type` in `{mean, flatten, mlp}`, `concat_dim` in `{0, 1}`, with
      `(ccr_rollout_len, num_frames, num_hist, ccr_action_source)` generated **jointly** so the
      feasible/infeasible boundary is covered under both sources
    - Every property test depends on this file, so it lands before any of them
    - _Requirements: 5.6_

  - [x] 1.3 [CODE] Create `tests/reference_impl.py` with the frozen pre-feature implementations
    - Verbatim copies of the current `VWorldModel.forward` loss tail and the current `rollout` predictor
      loop, with a comment recording the base commit SHA they were copied from
    - This is the model side of the model-based tests for Properties 1 and 7, so it **must exist before the
      `_rollout_latents` refactor** — otherwise there is nothing to compare the refactor against
    - _Requirements: 3.2, 3.3, 5.2_

- [x] 2. Rollout body extraction (pure refactor, behaviour preserved)
  - [x] 2.1 [CODE] Extract `_rollout_latents` from `rollout` in `models/visual_world_model.py`
    - Move the predictor loop verbatim (identical tensor ops in identical order) into
      `_rollout_latents(self, z, action)`; `rollout(obs_0, act)` delegates to it and keeps its signature,
      return type and numerics so `plan.py`, `planning/*` and `Trainer.openloop_rollout` are unaffected
    - No CCR code in this task — this is a pure refactor
    - _Requirements: 1.2, 1.7, 5.2_

  - [x] 2.2 [CODE] Write property test for the rollout refactor (`tests/test_rollout_refactor.py`)
    - **Property 7: The rollout refactor preserves rollout**
    - **Validates: Requirements 1.2, 1.7, 5.2**
    - Compares `rollout(obs_0, act)` element-for-element against the frozen loop in `tests/reference_impl.py`
    - **Gate, deliberately not optional:** this test must pass before any CCR code is written on top of
      `_rollout_latents`

- [x] 3. Configuration surface and run naming
  - [x] 3.1 [CODE] Add the CCR/MCA/pilot Hydra keys to `conf/train.yaml`
    - Under `training`: `lambda_cf: 0.0`, `ccr_rho: 0.0`, `ccr_rollout_len: 5`,
      `ccr_action_source: synthetic`, `mca_weight: 0.0`, `max_iterations: 0`,
      `telemetry_every_x_iterations: 200`; keep `save_every_x_iterations: 1000`
    - Forward the five loss knobs from `train.py` into the Hydra `instantiate` call for `VWorldModel`,
      alongside the existing `straighten` / `stop_grad` / `vcreg*` arguments — no Python literal fallback for
      any of them
    - _Requirements: 3.1, 3.5, 1.4, 6.9_

  - [x] 3.2 [CODE] Add the `ccr_tag` resolver and append it to the Run_Naming expression
    - `custom_resolvers.py`: `CCR_TAG_DEFAULTS = (0.0, 0.0, "synthetic", 0.0)`, `_fmt_num` (`.` → `p`), and
      `ccr_tag(lambda_cf, rho, action_source, mca_weight)` returning `""` at defaults and
      `_cf{}_rho{}_src{}_mca{}` otherwise; register with `OmegaConf.register_new_resolver`
    - `conf/train.yaml`: append one interpolation to the **end** of both `hydra.run.dir` and
      `hydra.sweep.dir`, leaving the existing expression untouched. `ccr_rollout_len` is deliberately not in
      the tag (it lives in `LOSS_SIGNATURE_KEYS` instead)
    - _Requirements: 3.4, 6.4, 6.5_

  - [x] 3.3 [CODE] Write property test for run naming (`tests/test_run_naming.py`)
    - **Property 2: Run naming is empty at defaults, complete otherwise, and injective**
    - **Validates: Requirements 3.4, 6.4, 6.5**
    - Includes the byte-identical legacy assertion: defaults resolve to
      `pusht_aggmlpcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05`, and two tuples differing only in
      `ccr_action_source` never contribute the same tag
    - **Gate, deliberately not optional:** must pass before any run is launched (Requirement 3.4)

  - [ ]* 3.4 [CODE] Write config-default unit tests (`tests/test_config_defaults.py`)
    - Resolved defaults: `lambda_cf == 0`, `mca_weight == 0`, `ccr_rollout_len == 5`,
      `ccr_action_source == "synthetic"`, `max_iterations == 0`, `save_every_x_iterations == 1000`
    - _Requirements: 3.1, 1.4, 6.9_

- [x] 4. CCR core in `models/visual_world_model.py`
  - [x] 4.1 [CODE] Add knob storage, eager validation, boolean gates and startup logging
    - Store `lambda_cf`, `ccr_rho`, `ccr_rollout_len`, `ccr_action_source`, `mca_weight`; reject negative
      weights and any `ccr_action_source` outside `("logged", "synthetic")` **even at `lambda_cf = 0`**;
      reject `ccr_rollout_len < 1`; set `self.ccr = lambda_cf > 0` and `self.mca = mca_weight > 0`
    - No `nn.Module` construction anywhere in the CCR path (no new parameter, buffer or submodule)
    - Emit the enabled/disabled startup lines naming term, weight, `rho`, `rollout_len`, `action_source`,
      `synthesized_action_frames = max(0, num_hist + L - 1 - num_frames)`, `curvature_mode` and the device
      from `next(self.parameters()).device`
    - _Requirements: 1.6, 3.5, 3.6, 4.3_

  - [ ]* 4.2 [CODE] Write property test for state neutrality (`tests/test_ccr_no_new_state.py`)
    - **Property 9: Enabling a term adds no state to the model**
    - **Validates: Requirements 1.6, 4.3**

  - [ ]* 4.3 [CODE] Write property test for startup announcement (`tests/test_ccr_startup_log.py`)
    - **Property 16: An enabled term announces itself with its device**
    - **Validates: Requirements 3.6**

  - [x] 4.4 [CODE] Implement `_sample_action_perturbation` and `_ccr_actions`
    - `_sample_action_perturbation`: `torch.empty_like(act).uniform_(-1, 1).mul_(rho)` — bounded by
      construction, no `if rho == 0` branch, device/dtype inherited from `act`, single RNG consumer in the
      whole CCR path
    - `_ccr_actions(act, required)`: perturbed recorded prefix, plus zero-padded synthesized frames past the
      window edge (reachable only under `synthetic`, since `logged` is rejected upstream); takes no
      `ccr_action_source` argument
    - Keep the sampler a separate method so Property 4 can monkeypatch it to zeros
    - _Requirements: 1.3, 1.8, 2.1, 2.2, 2.3_

  - [ ]* 4.5 [CODE] Write property test for the perturbation radius (`tests/test_ccr_perturbation.py`)
    - **Property 3: Perturbations respect the radius, for recorded and synthesized actions alike**
    - **Validates: Requirements 1.3, 2.1, 1.8**

  - [x] 4.6 [CODE] Implement `compute_ccr`
    - `required = num_hist + L - 1`; raise the Requirement 1.10 `ValueError` **only** under
      `ccr_action_source == "logged"` when `required > available`, with a message naming `lambda_cf`, the
      requested `L`, the required length, the available length, the maximum permitted `L`
      (`available - num_hist + 1`) and the `synthetic` escape hatch
    - Build `act_cf` via `_ccr_actions`, clone `z[:, :num_hist]` before `replace_actions_from_z` (in-place
      write would corrupt the baseline term's graph), roll with `_rollout_latents`, take
      `visual_only(z_imag[:, -(L + 2):])`, and return `total_curvature(feats, mode="aggcos")`
    - Zero additional encoder forward passes; visual-only channel selection matching the baseline curvature
      term (the `196 x 18` variant is unreachable with `agg_type: mlp`, recorded as a known deviation)
    - _Requirements: 1.2, 1.4, 1.5, 1.7, 1.10, 2.2_

  - [ ]* 4.7 [CODE] Write property test for geometry reuse (`tests/test_ccr_geometry_reuse.py`)
    - **Property 8: CCR reuses the existing geometry machinery**
    - **Validates: Requirements 1.2, 1.5, 1.7**

  - [ ]* 4.8 [CODE] Write property test for the horizon guard (`tests/test_ccr_horizon_errors.py`)
    - **Property 11: Rollout length beyond the available actions is an error under `logged` only**
    - **Validates: Requirements 1.10, 1.4**

  - [ ]* 4.9 [CODE] Write property test for action-source equivalence (`tests/test_ccr_action_sources.py`)
    - **Property 17: The two action sources coincide whenever the window suffices**
    - **Validates: Requirements 1.2, 1.4, 2.2, 1.10**

  - [ ]* 4.10 [CODE] Write property test for dataset-constant freedom (`tests/test_ccr_no_dataset_constant.py`)
    - **Property 5: CCR carries no dataset-specific constant**
    - **Validates: Requirements 1.8, 11.4**

  - [x] 4.11 [CODE] Wire CCR into `forward` behind the boolean gate
    - `if self.ccr:` add `lambda_cf * ccr_loss` to the loss and record `ccr_loss` / `ccr_loss_scaled` in
      `loss_components`; also add `curvature_loss_scaled` (keeping the existing unscaled key) so the baseline
      term is comparable in telemetry
    - Disabled path performs one attribute lookup and one comparison: no tensor work, no extra rollout, no
      extra encoder pass
    - _Requirements: 1.1, 1.9, 3.2, 3.3_

  - [ ]* 4.12 [CODE] Write property test for the disabled path (`tests/test_ccr_disabled_path.py`)
    - **Property 1: The disabled path is the baseline path**
    - **Validates: Requirements 3.2, 3.3, 1.1, 4.1**
    - Bitwise loss equality against `tests/reference_impl.py` plus equal `encode` / `encode_obs` / `predict`
      / `total_curvature` call counts

  - [ ]* 4.13 [CODE] Write property test for arm equivalence (`tests/test_ccr_arms_equivalence.py`)
    - **Property 4: The arms differ only by the perturbation**
    - **Validates: Requirements 2.1, 2.2, 2.3**
    - Monkeypatches `_sample_action_perturbation` to zeros and compares against a `rho = 0` run

- [x] 5. MCA term (pilot only, default off)
  - [x] 5.1 [CODE] Implement `compute_mca` and wire it behind `self.mca`
    - Scale-invariant ratio-to-batch-mean form over `visual_only` velocities in patch space versus after
      `encoder.agg`; reuses the existing `agg` module, adds no module and no parameter
    - Record `mca_loss` / `mca_loss_scaled` in `loss_components`
    - _Requirements: 4.1, 4.2, 4.3, 4.4_

  - [ ]* 5.2 [CODE] Write property test for MCA (`tests/test_mca.py`)
    - **Property 12: MCA measures scale-free aggregation distortion**
    - **Validates: Requirements 4.2**

- [ ] 6. Checkpoint - CCR and MCA core complete
  - Ensure all tests pass, ask the user if questions arise.
  - Property 7 and Property 2 (including the byte-identical legacy run-directory string) must be green
    before continuing.

- [x] 7. Pilot infrastructure in `train.py`
  - [x] 7.1 [CODE] Add `global_iter`, the iteration cap and the budget log
    - `self.global_iter = 0` next to `self.epoch`, appended to `self._keys_to_save` so `save_ckpt` persists
      and `load_ckpt` restores it; legacy checkpoints fall through the existing `Keys not found in ckpt`
      warning and start at 0
    - After the optimizer steps: increment, and when `0 < max_iterations <= global_iter` force a telemetry
      row, flush iteration logs, `save_ckpt()` and break mid-epoch; the epoch loop breaks on
      `_stop_requested`, skipping `val()` and `logs_flash` (which would `KeyError` without a validation pass)
    - Startup line: `Iteration budget: steps/epoch=... epochs=... total=... max_iterations=... (cap active)`
    - `max_iterations <= 0` reproduces current behaviour exactly
    - _Requirements: 6.1, 6.2, 6.3_

  - [ ]* 7.2 [CODE] Write property test for the iteration cap (`tests/test_iteration_cap.py`)
    - **Property 6: The iteration cap only ever shortens a run**
    - **Validates: Requirements 6.1, 6.2, 6.3**

  - [x] 7.3 [CODE] Implement the run-directory loss-signature guard
    - `LOSS_SIGNATURE_KEYS = ("straighten", "stop_grad", "vcreg", "vcreg_std_coeff", "vcreg_cov_coeff",
      "lambda_cf", "ccr_rho", "ccr_rollout_len", "ccr_action_source", "mca_weight")`
    - `_guard_run_dir()` called from `Trainer.__init__` right after `cfg["saved_folder"]` is set and
      **before** `wandb.init`, before `hydra.yaml` is written and before any checkpoint write: no
      `model_latest.pth` → write `loss_config.json` (main process only) and return; otherwise compare against
      `loss_config.json`, falling back to the previous run's resolved `hydra.yaml` with missing keys treated
      as defaults, warning and proceeding if neither exists; on mismatch raise `RuntimeError` naming the
      directory and the differing keys with both values, writing nothing
    - _Requirements: 6.6_

  - [ ]* 7.4 [CODE] Write property test for the loss-configuration guard (`tests/test_run_dir_guard.py`)
    - **Property 13: The loss-configuration guard aborts exactly on conflict**
    - **Validates: Requirements 6.6**

  - [x] 7.5 [CODE] Implement the JSONL telemetry sink
    - Append `training_log.jsonl` in the Hydra run directory (main process only, flushed on write), one
      object per logged iteration at `training.telemetry_every_x_iterations` (default 200, matching the
      reference run so step-200 rows are directly comparable) plus the final step of a capped run
    - Record per-term scaled value and share (`scaled / loss_components["loss"]`), `it_per_s` from
      `time.perf_counter()`, `enabled_terms`, and the self-describing `ccr` block (`raw`, `lambda_cf`,
      `rho`, `rollout_len`, `action_source`, `synthesized_action_frames`)
    - Term registry: `z_loss`→`prediction`, `curvature_loss_scaled`→`curvature`, `ccr_loss_scaled`→`ccr`,
      `mca_loss_scaled`→`mca`, `z_vcreg_loss_scaled`→`vcreg`,
      `decoder_loss_reconstructed`→`decoder`
    - _Requirements: 6.7, 6.8_

  - [ ]* 7.6 [CODE] Write property test for loss bookkeeping and telemetry (`tests/test_telemetry.py`)
    - **Property 10: Loss bookkeeping and shares are consistent**
    - **Validates: Requirements 1.9, 4.4, 6.7, 6.8**
    - Includes the JSON write/read round trip, since the pilot verdict is read out of this file

  - [x] 7.7 [CODE] Create `summarize_training_log.py`
    - `summarize_training_log.py <run_dir>` prints the term/scaled/share table, the step rate and the
      step-200 row; `--compare <reference_run_dir>` prints the row-by-row delta against a reference run;
      `--collapse-check` flags a term whose share falls below 0.1% within the first 1,000 iterations
    - _Requirements: 8.3, 8.4, 8.6_

  - [ ]* 7.8 [CODE] Write unit tests for the summarizer (`tests/test_summarize_log.py`)
    - Synthetic JSONL fixtures: share table arithmetic, `--compare` deltas, and `--collapse-check` firing on
      a term that decays below 0.1% before iteration 1,000
    - _Requirements: 8.3, 8.4, 8.6_

- [x] 8. Scope containment, protocol invariance and generality
  - [x] 8.1 [CODE] Write the changed-file guard test (`tests/test_scope_guard.py`)
    - Assert the feature branch's changed-file set is a subset of the Requirement 5.6 allowlist
      (`models/visual_world_model.py`, `conf/train.yaml`, `train.py`, `custom_resolvers.py`, new standalone
      scripts, `tests/`)
    - Assert `planning/*.py` and `datasets/*.py` file hashes are equal to the base revision
    - **Gate, deliberately not optional:** scope containment is a stated requirement, and this is the only
      automated check of it
    - _Requirements: 5.2, 5.4, 5.6_

  - [ ]* 8.2 [CODE] Write protocol-invariance tests (`tests/test_protocol_invariants.py`)
    - Resolved Full_Run config equality: batch 32, `num_hist` 3, `num_pred` 1, `num_frames` 4, `frameskip` 5,
      epochs 2, `encoder_lr` 1e-5, `straighten` `aggcos1e-1`, `stop_grad` True, `mixed_precision` bf16,
      `seed` 0
    - `conf/plan_gd.yaml` / `conf/plan_gd_mpc.yaml` hyperparameter equality: `max_iter` 1,
      `n_taken_actions` 25, `sub_planner.horizon` 25, `lr` 0.1, `sample_type` `zero`, `action_noise` 0,
      `opt_steps` 100; Evaluation_Protocol shape: 50 samples per seed, seeds 100/200/300, open-loop
      `mode=last, alpha=1`, MPC `mode=staged, alpha=1`
    - _Requirements: 5.1, 5.3, 5.5_

  - [ ]* 8.3 [CODE] Write the `env=point_maze` generality check (`tests/test_generality_point_maze.py`)
    - Resolve `conf/train.yaml` with `env=point_maze` and CCR enabled, construct the model and run one
      forward with **no code change** — this is what turns the no-hardcoding claim into a verified fact
      rather than an assertion
    - Also covers the default-coherence pair: shipped defaults with a positive `lambda_cf` run on the PushT
      target-cell shapes, while `ccr_action_source=logged` raises the Requirement 1.10 `ValueError`
    - _Requirements: 11.4, 1.8, 1.10_

- [x] 9. Offline probe
  - [x] 9.1 [CODE] Create `probe_ccr_curvature.py` (standalone, read-only, CPU-only)
    - Validate `--ckpt` / `--train-cfg` paths first and `sys.exit(1)` with the absolute missing path before
      constructing any model; hash (`sha256`), size and mtime the checkpoint; load with
      `map_location="cpu"`, `model.eval()`, everything under `torch.no_grad()`, no optimizer anywhere
    - Sample `--num-windows` validation windows through the unmodified loader at a fixed seed; per window
      encode once then evaluate `total_curvature(visual_only(z_imag[:, -(L+2):]), "aggcos")` unperturbed and
      over `--draws` perturbations, reusing the training `_ccr_actions` construction so the probe measures
      the arm that will be trained (and rejecting `logged` with the same message when infeasible)
    - Readouts `curvature_gap` and `state_readout_r2`, each with an aggregate plus the five per-dimension
      entries (`agent_x, agent_y, block_x, block_y, block_angle`, top-tercile motion subsets), and each with
      `reference_value` / `reference_source` from `{pristine, early_telemetry, control_run}`
    - `--max-minutes` wall-clock guard marks the report `partial: true` and exits 0; re-hash the checkpoint
      at the end and raise loudly if it changed; write reports to `probe_outputs/`, never into the
      checkpoint directory
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_

  - [ ]* 9.2 [CODE] Write property test for the probe (`tests/test_probe.py`)
    - **Property 14: The probe is read-only and fully disaggregated**
    - **Validates: Requirements 7.1, 7.2, 7.3, 7.4**

  - [ ]* 9.3 [CODE] Write probe CLI unit tests (`tests/test_probe_cli.py`)
    - Missing checkpoint or config path: non-zero exit, absolute path in the message, no model constructed
    - _Requirements: 7.5_

  - [ ]* 9.4 [CPU RUN] Timed probe budget integration test (`tests/test_probe_budget.py`)
    - One timed CPU run against the target PushT checkpoint, asserted under 30 minutes. Integration test, run
      once, not a property; needs the real checkpoint on disk
    - _Requirements: 7.6_

- [x] 10. Acceptance-gate predicate
  - [x] 10.1 [CODE] Implement `acceptance_gate` as a pure predicate (`ccr_acceptance_gate.py`)
    - `acceptance_gate(cand_ol, cand_mpc, base_ol, base_mpc, paper_ol=77.33, paper_mpc=85.33, se_pts=5.7,
      margin_pts=6.0)` returning `fail` / `inconclusive` / `pass`; one condition alone is a failure; margin
      at or below 6 points is inconclusive; the ~5.7-point binomial standard error is reported alongside
    - _Requirements: 10.1, 10.2, 10.4, 10.5, 10.6_

  - [ ]* 10.2 [CODE] Write property test for the acceptance gate (`tests/test_acceptance_gate.py`)
    - **Property 15: The acceptance gate is a dual, margin-aware predicate**
    - **Validates: Requirements 10.1, 10.2, 10.5, 10.6**

- [x] 11. Runtime environment driver
  - [x] 11.1 [CODE] Create `run_ccr_pilot.sh` (new standalone driver, no existing script modified)
    - `export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync`; `unset CUDA_VISIBLE_DEVICES`;
      `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=NUMEXPR_NUM_THREADS=8`;
      `PLAN_SERIAL_ENV=1` for evaluation launches
    - `ps -eo pid,stat,etime,cmd` pre-flight that refuses to start when a `train`/`plan`/`probe` python
      process is alive (the `1g.45gb` MIG slice holds exactly one job, and `nvidia-smi` does not enumerate
      processes on a MIG slice); chain jobs on the driver PID, one at a time
    - Must exist before the first GPU launch
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7_

  - [ ]* 11.2 [CODE] Write driver-contract tests (`tests/test_run_ccr_pilot_sh.py`)
    - Parse `run_ccr_pilot.sh` and assert each Requirement 9 environment variable is set to the required
      value, `CUDA_VISIBLE_DEVICES` is unset, and the `ps` pre-flight guard is present and fatal
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

- [ ] 12. Checkpoint - all local code and tests complete before any GPU time
  - Ensure all tests pass, ask the user if questions arise.
  - Property 7, Property 2 (byte-identical legacy run name) and the changed-file scope guard must be green.
    No run of any kind launches before this checkpoint.

- [ ] 13. Offline probe execution and probe gate (rung 1 of the ladder)
  - [x] 13.1 [CPU RUN] Run `probe_ccr_curvature.py` on the target PushT checkpoint
    - `--rho 0.5 --rollout-len 5 --action-source synthetic --num-windows 64 --draws 4
      --reference pristine --max-minutes 30 --out probe_outputs/ccr_pusht.json`
    - The probe fixes **two** numbers, not one: the selected `rho`, and `c` — the raw *perturbed imagined*
      curvature — which sets `lambda_cf` for tasks 15.1 and 15.5 via the corrected rule
      `lambda_cf = 0.00994 / c`. The earlier `0.024 / g` form used the wrong quantity (on-log curvature times
      a ratio) and is superseded. The pilot λ is not final until this reports
    - **Run and complete.** `rho = 0.05` as originally specified FAILED (aggregate gap `-0.001259`, 0 of 5
      dimensions) because it is 10-20x below the action scale `GDPlanner` explores. A widened criterion was
      declared before re-running, and a sweep at ~78 s per arm gave 0.25 → 1 of 5, **0.50 → 5 of 5**,
      1.0 → 5 of 5, 2.0 → 5 of 5. `rho = 0.5` selected; `c = 0.155470 + 0.073174 = 0.228644`
    - ~30 minutes, CPU, read-only. Not agent-executable: needs the real checkpoint and dataset
    - _Requirements: 7.1, 7.6, 11.3_

  - [x] 13.2 [HUMAN] Record the probe gate verdict
    - Written gate: aggregate `curvature_gap` positive **and** at least 20% of the unperturbed curvature
      magnitude on at least 3 of the 5 disaggregated dimensions. If it fails, **no Pilot_Run is launched**
    - **Verdict: PASS at `rho = 0.5`** (5 of 5 dimensions, ratios 0.276-0.733). Caveats recorded in
      `PROGRESS_CCR.md` §5: (a) the pass required re-calibrating `rho`, so it is a pass on the widened
      criterion declared before the sweep, not on the originally recorded value; (b) `state_readout_r2` for
      `block_angle` is 0.183, the worst of the five, and `block_angle` also has the weakest gap ratio at every
      `rho` — the dimension PushT is scored on is the one CCR has least purchase over
    - Human judgement call; record the verdict and its caveats in the project progress log
    - _Requirements: 8.1, 8.2, 11.3, 7.2, 7.3_

- [ ] 14. Compute allocation reconciliation
  - [ ] 14.1 [HUMAN] Surface the compute overrun and request approval
    - The recorded allocation (Requirement 11.5) is ≈23 GPU-hours around **three** pilot arms. Two things
      revise it. The sweep is now **four** arms and the treatment arm runs `L = 5` rather than `L = 2`: pilot
      subtotal ≈5.2-5.8 h instead of ≈3.75-4.25 h, everything else (probe, triage, Full_Run, 3-seed eval)
      unchanged at ≈19.3 h. And the recorded plan omitted the baseline train entirely, because it assumed a
      checkpoint was already on disk; there was none, so the baseline was trained as part of this work and has
      to be counted. At the measured 2.863 it/s a full-budget PushT run is `123,858 / 2.863 = 43,260 s
      ≈ 12.0 h`, so the baseline train is ≈12.0 h and its 3-seed eval ≈1.5 h. **Revised total ≈37
      GPU-hours**
    - The ≈14 GPU-hour overrun is reported, not absorbed: ≈13.5 h of it is the already-spent baseline, which
      is not recoverable, and ≈1.5-2 h is the fourth pilot arm. **If it is refused, the arm to drop is the
      `lambda_cf` variation (15.5), not the `logged` control (15.3)** — the control is what makes the
      `synthetic` extrapolation risk measurable
    - Note separately that additional training seeds under Requirement 10.5 would cost a further ≈26
      GPU-hours and need their own approval (Requirement 11.6)
    - _Requirements: 11.5, 11.6_

- [ ] 15. Pilot arms (rung 2; strictly serial - one job on the MIG slice at a time)
  - **Run task 18.1 (the baseline 3-seed evaluation) before this section**, not after. It is numbered with
    the acceptance-gate group because that is what consumes it, but it has no dependency on any pilot: the
    baseline checkpoint is already on disk. Executing it first turns it into the cheap stop-and-investigate
    check (~1.5 h) that validates the whole `plan.py` evaluation path *before* ~5.5 h of pilots and ~17 h of
    Full_Run are spent against a reference that has never been measured. If it lands outside ~72-82 open-loop,
    stop: there is no point comparing CCR to a baseline number that is itself wrong
  - **No matched-budget control arm.** A fifth arm with CCR off, capped at the same 8,000 steps, is not
    needed: the baseline's own first-8,000-step telemetry is already on disk and is exactly that control at
    zero cost, per `SHORT_BUDGET_PILOTS.md` §4. Every gate comparison below is against the recorded
    step-8,000 row rather than against a run we pay for
  - [ ] 15.1 [GPU RUN] Treatment arm: `synthetic`, `L = 5`, `rho = 0.5`
    - `run_ccr_pilot.sh` wrapping `python train.py --config-name train.yaml env=pusht encoder=dino_channel
      training.straighten=aggcos1e-1 training.encoder_lr=1e-5 training.stop_grad=True
      training.lambda_cf=0.04 training.ccr_rho=0.5 training.ccr_action_source=synthetic
      training.ccr_rollout_len=5 training.mca_weight=0 training.max_iterations=8000 training.epochs=3`
    - λ comes from the design's corrected rule `lambda_cf = 0.00994 / c`, where `c` is the probe's raw
      perturbed imagined curvature. Task 13.1 reported `c = 0.228644` at `rho = 0.5`, giving `0.043`; **`0.04`
      is the value to launch** (14.0% CCR share against the recorded step-8,000 total)
    - Two-minute smoke check: the `CCR enabled:` line names the right weight, `rho`, `action_source`,
      `synthesized_action_frames=3` and device, and a checkpoint exists on disk. A `synthetic` arm reporting
      `synthesized_action_frames=0` is silently a `logged` arm and the launch is wrong
    - ~85-95 min. Not agent-executable
    - _Requirements: 8.1, 8.2, 9.1-9.7, 11.3, 4.5, 6.1_

  - [ ] 15.2 [HUMAN] Step-rate and step-200 check, plus the Requirement 11.7 regression report
    - `summarize_training_log.py <run_dir> --compare <reference_run_dir>`: `it_per_s >= 1.91` — derived from
      the baseline's measured median of 2.863 it/s over 54 telemetry records (`2.863 / 1.5`), not from the
      rounded ~2.9 it/s of `REPRODUCTION.md`, which the measurement confirms to within 1.3% step time — and
      the step-200 row matching the reference for shared terms. Checked on this arm **first**, since `L = 5`
      has the least headroom
    - A CCR arm at the upper end of the estimated +30-50% step-time cost lands at `2.863 / 1.5 = 1.91` it/s
      and therefore grazes the floor rather than clearing it. Under Requirement 11.7 that is a **reporting
      event before the Full_Run, not an abort**
    - If step time regressed by more than 50%, write the regression report and revise the compute plan
      **before** the Full_Run, not after
    - _Requirements: 8.4, 11.7, 6.8_

  - [ ] 15.3 [GPU RUN] Horizon control arm: `ccr_action_source=logged ccr_rollout_len=2`
    - Isolates whether the gain needs the horizon past the window edge, and prices the `synthetic`
      extrapolation risk. ~75-85 min, serial after 15.1
    - _Requirements: 8.2, 9.7, 11.3_

  - [ ] 15.4 [GPU RUN] Perturbation control arm: `ccr_rho=0`
    - Separates "rollout space vs encoder space" from "off-log vs on-log". Same code path as the treatment
      arm by construction. ~75-85 min, serial after 15.3
    - _Requirements: 2.1, 2.2, 2.3, 9.7_

  - [ ] 15.5 [GPU RUN] Weight variation arm: `lambda_cf=0.08` at `rho = 0.5`
    - Sensitivity of the result to the term's share of the objective: `0.04` → 14.0%, `0.08` → 24.6%, both
      inside the `[2%, 30%]` window, with `0.08` still leaving headroom for the upward share drift the
      baseline exhibited (73.7% @8k → 82.7% @123.8k at fixed scale)
    - Two earlier pairs are superseded. `{0.1, 0.3}` is out because `0.3` drives the prediction share to
      ≈7.3%, below the 11.75% gate floor. `{0.02, 0.05}` is out because it was derived from
      `raw_ccr = g * 0.41421`, the wrong quantity: CCR is evaluated on the imagined off-log rollout, whose
      perturbed curvature the probe measures directly as `c = 0.228644` (`0.55x` the assumed value). At the
      measured `c`, `0.02` lands at only 7.5% — too weak to be an informative arm. Note that `lambda_cf = 0.1`
      lands at 28.9% and **is** admissible; the earlier "4-15x too strong" claim was wrong
    - ~75-85 min, serial after 15.4. **This is the arm to drop if the 14.1 compute overrun is refused**
    - _Requirements: 8.2, 8.3, 9.7, 11.5_

  - [ ] 15.6 [HUMAN] Record the pilot gate verdict
    - Written gate, all four checks: (1) startup line and checkpoint present within two minutes, with the
      primary confirmation that CCR is running being the `ccr` term appearing in the telemetry record's
      `enabled_terms` (equivalently `enabled: true` in the record's `ccr` block), since that is derived from
      the model's own gate firing rather than from config — `synthesized_action_frames == 3` is a **secondary**
      synthetic-vs-logged check, read only after CCR is confirmed enabled, because on the CCR-disabled
      baseline the old field still read 3 and so never confirmed CCR was running;
      (2) `it_per_s >= 1.91` and step-200 row matches the reference; (3) at `global_iter` 8,000 — the pilots'
      own budget and the step the recorded reference row was read at — the CCR **share** is in `[0.02, 0.30]`
      and the prediction share is at least **11.75%** (half of the reference's 23.493%), equivalently a scaled
      CCR contribution `X in [0.0011, 0.0241]` against the recorded total of 0.056171, with the 30% cap
      binding and the prediction floor slack by more than 2x — shares, never raw losses; (4) the raw CCR term
      does not fall below 1e-3 within the first 1,000 iterations, and if it does the pilot is recorded as
      **not** a success
    - Mid-run representation readouts are catastrophic-failure detectors only, never trends. If the MCA
      pilot was run and failed its own gate, MCA is excluded from the Full_Run. Append each arm's outcome and
      caveats to the project progress log
    - _Requirements: 8.1, 8.3, 8.5, 8.6, 8.8, 4.6, 11.3_

- [ ] 16. Triage evaluation (rung 3)
  - [ ] 16.1 [GPU RUN] Single-data-seed triage evaluation of the winning pilot arm
    - `plan.py` under the unmodified Evaluation_Protocol, one data seed, ~20 min, `PLAN_SERIAL_ENV=1`.
      Sanity only: a low number is not evidence against, since a pilot predictor is ~7x worse on `z_loss`.
      No success-rate difference is reported from a single seed
    - _Requirements: 8.7, 5.3, 5.5, 9.4, 9.7_

- [ ] 17. Full run (rung 4)
  - [ ] 17.1 [GPU RUN] Full_Run at the paper budget, gated on the pilot gate
    - 123,858 optimizer steps / 2 epochs, `mca_weight=0`, every Protocol_Invariant untouched,
      `max_iterations` back to 0. **Must not launch until 15.6 records a pass** (Requirement 11.3), and not
      until any 15.2 regression has been reported and the plan revised (Requirement 11.7). ~17 h
    - _Requirements: 11.3, 5.1, 4.5, 11.7_

- [ ] 18. Acceptance-gate evaluation (rung 5)
  - [ ] 18.1 [GPU RUN] Re-evaluate the Platform_Baseline under Evaluation_Protocol
    - The identical pre-feature config (`pusht_aggmlpcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05`),
      50 samples per seed, seeds 100/200/300, open-loop `mode=last, alpha=1`, MPC `mode=staged, alpha=1`.
      **Measured, not assumed from the recorded ~75.3 / ~82.0** — the recorded number is a cross-check, not
      the gate input
    - One command covers all six jobs: `bash run_ccr_pilot.sh eval <baseline_run_dir>` runs the three
      open-loop seeds then the three MPC seeds, serially in one driver, with `PLAN_SERIAL_ENV=1`. ~1.5 h
    - **Execute this FIRST, ahead of the section-15 pilots.** It is listed here because the Acceptance_Gate is
      what consumes it, but it depends on nothing except the baseline checkpoint, which is already on disk.
      Run early it is a ~1.5 h validation of the entire evaluation path; run late it is a number arriving
      after ~24 GPU-hours have already been committed to comparisons against it. Expected band ~75-78
      open-loop / ~82-85 MPC (paper 77.33±6.18 / 85.33±4.99; prior B200 reproduction ~75.3 / ~82.0). Outside
      ~72-82 open-loop is stop-and-investigate, in **either** direction — a number well above the paper's mean
      is as much a sign of a protocol discrepancy as one below it
    - _Requirements: 10.2, 10.3, 5.5, 9.4, 9.7_

  - [ ] 18.2 [GPU RUN] Three-seed evaluation of the Full_Run candidate
    - Same protocol, same seeds, open-loop and MPC, `mca_weight=0`. ~1.5 h, serial after 18.1
    - _Requirements: 5.5, 8.7, 4.5, 10.3_

  - [ ] 18.3 [HUMAN] Record the Acceptance_Gate verdict
    - Feed both measured pairs through `acceptance_gate`: pass requires beating 77.33 open-loop, 85.33 MPC,
      **and** both re-measured Platform_Baseline rates. One condition alone is a failure. Report the ~5.7
      percentage-point binomial standard error (n=50, p≈0.8) alongside every comparison; a margin of 6 points
      or less over the baseline is inconclusive
    - If the margin rule triggers, either report inconclusive or request approval for the ≈26 additional
      GPU-hours of extra training seeds (Requirement 11.6) before launching them
    - Record the verdict and caveats in the project progress log
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 11.6, 8.8_

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster path. Three test tasks are deliberately
  **not** marked optional because they are gates named in the requirements: 2.2 (Property 7, the refactor
  guard), 3.3 (Property 2 and the byte-identical legacy run name, Requirement 3.4), and 8.1 (the changed-file
  scope guard, Requirements 5.2/5.4/5.6). Property 1 (task 4.12) is marked optional by format convention but
  is strongly recommended, since it is the only bitwise check of default-off legacy equivalence.
- **[CODE]** tasks are agent-executable on CPU with no dataset or network access — the tiny stub encoder in
  `tests/conftest.py` exists exactly for that. **[GPU RUN]**, **[CPU RUN]** and **[HUMAN]** tasks are
  operator or judgement work and are listed for sequencing, not for agent execution.
- The design's `tests/` layout is expanded to one file per property test, so that independent property tasks
  never write the same file and can be scheduled in parallel. The set of tests is unchanged.
- `hypothesis` is test-only. It must not appear in `requirements-train.txt` or `requirements-plan.txt`.
- Every property test docstring carries the tag
  **Feature: counterfactual-curvature-regularization, Property N: <property text>**, minimum 100 examples.
- Requirement 9 (environment recipe, `ps` hygiene, serial execution) and Requirements 8 and 11 (pilot
  discipline, approval, escalation, compute accounting) are operator procedure. Their automatable fragments
  are tasks 11.2, 7.6 (Property 10) and 7.7's `--collapse-check`; the rest are the **[HUMAN]** tasks above.
- Serialization is real, not stylistic: the `1g.45gb` MIG slice holds one job, so 15.1, 15.3, 15.4, 15.5,
  16.1, 17.1, 18.1 and 18.2 each occupy their own wave.
- **18.1 is scheduled ahead of 15.1** in the wave graph even though it is numbered with the acceptance-gate
  group. It has no upstream dependency beyond the baseline checkpoint, and running it first makes it a ~1.5 h
  validation of the evaluation path rather than a number that arrives after ~24 GPU-hours of work has been
  staked on it. 18.2 (the candidate's 3-seed evaluation) stays late, because it needs the Full_Run.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2", "1.3", "3.1", "10.1", "11.1"] },
    { "id": 1, "tasks": ["2.1", "3.2", "3.4", "10.2", "11.2"] },
    { "id": 2, "tasks": ["2.2", "3.3"] },
    { "id": 3, "tasks": ["4.1"] },
    { "id": 4, "tasks": ["4.2", "4.3", "4.4"] },
    { "id": 5, "tasks": ["4.5", "4.6"] },
    { "id": 6, "tasks": ["4.7", "4.8", "4.9", "4.10", "4.11"] },
    { "id": 7, "tasks": ["4.12", "4.13", "5.1"] },
    { "id": 8, "tasks": ["5.2", "7.1"] },
    { "id": 9, "tasks": ["7.2", "7.3"] },
    { "id": 10, "tasks": ["7.4", "7.5"] },
    { "id": 11, "tasks": ["7.6", "7.7", "9.1"] },
    { "id": 12, "tasks": ["7.8", "8.1", "8.2", "8.3", "9.2", "9.3"] },
    { "id": 13, "tasks": ["9.4"] },
    { "id": 14, "tasks": ["13.1"] },
    { "id": 15, "tasks": ["13.2"] },
    { "id": 16, "tasks": ["14.1"] },
    { "id": 17, "tasks": ["18.1"] },
    { "id": 18, "tasks": ["15.1"] },
    { "id": 19, "tasks": ["15.2"] },
    { "id": 20, "tasks": ["15.3"] },
    { "id": 21, "tasks": ["15.4"] },
    { "id": 22, "tasks": ["15.5"] },
    { "id": 23, "tasks": ["15.6"] },
    { "id": 24, "tasks": ["16.1"] },
    { "id": 25, "tasks": ["17.1"] },
    { "id": 26, "tasks": ["18.2"] },
    { "id": 27, "tasks": ["18.3"] }
  ]
}
```
