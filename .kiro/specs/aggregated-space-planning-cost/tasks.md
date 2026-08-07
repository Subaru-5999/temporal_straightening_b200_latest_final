# Implementation Plan: Aggregated-Space Planning Cost

## Overview

Python 3.10 / PyTorch / Hydra, matching the rest of the repo. Two new root-level files carry the whole
feature (`agg_objectives.py`, `plan_agg.py`); nothing under `planning/`, nothing under `datasets/` and not
`plan.py` is edited. The plan is ordered by what gates what:

1. **Scope guard first.** `tests/test_scope_guard.py` is the check run before every launch, and it fails the
   moment `agg_objectives.py` and `plan_agg.py` appear in the working tree. Its two new allowlist entries,
   the `plan.py` byte-freeze addition (Requirement 4.4, Property 9) and the Scope_Amendment comment are one
   task, landing before any feature code, so the gate never reports a false violation.
2. **The bitwise-zero property gates everything downstream.** Property 1 is the reason the Baseline_Arm is a
   valid control rather than an approximation. It is not optional and it passes before any pod job runs.
3. **Coefficient reuse is proved locally, not assumed.** Property 3 (identity head, `alpha = 0`, L_agg equal
   to the unmodified `planning.objectives` value in `staged` mode) is the only check that the frozen stage
   dispatch and per-frame coefficients are reused *exactly*. It must be green before the MPC confirmation
   run, since `staged` is the mode that run uses.
4. **Run-directory separation is load-bearing, and it is checked without a GPU.** The shipped
   `hydra.run.dir` template omits the weight, so all seven sweep arms would write into one `logs.json` cell
   and `aggregate_results.py` would silently average seven weights into one number. Task 3.1 resolves the
   override template through Hydra compose and asserts seven distinct directories. Its own early task, its
   own gate.
5. **The paired zero-weight check gates the sweep, and it is also the sweep's zero arm.** One ~15 min pod job
   pair: `plan.py` versus `plan_agg.py --agg_weight=0` at seed 400, open-loop, identical per-episode success
   vectors (Requirement 3.3). This is the end-to-end confirmation that the design's bitwise-zero argument
   survives a real run. The `plan_agg.py` leg *is* the Baseline_Arm reference point the sweep curve is read
   against (Requirement 6.3), so it is not run twice. The `plan.py` leg carries a hazard: the shipped
   `hydra.run.dir` template carries neither seed nor weight, so without an explicit override it resolves to
   the very directory holding the already-recorded 75.33 +/- 6.11 open-loop Platform_Baseline and would append
   a seed-400 line, turning the Acceptance_Gate's own reference cell into a 4-seed mean. **Both** legs
   therefore pass an explicit `hydra.run.dir`; the `plan.py` leg goes to a scratch prefix.
6. **Sweep → selection → confirmation → verdict.** 6 non-zero open-loop arms at the Tuning_Seed (~35 min, plus
   the zero arm already run in task 11.1, so ~40 min against Requirement 9.4) gate weight selection; selection
   gates the 12-run confirmation (~3 h: `REPRODUCTION.md` records ~25 min per MPC seed on this eval path, and
   task 14 runs two arms of 6 runs each); confirmation gates the Acceptance_Gate. Total GPU across the plan is
   ~4 h.
7. **Runs are strictly serial.** The `1g.45gb` MIG slice holds exactly one job (Requirement 9.2), so the
   dependency graph gives every arm its own wave.

Task labels:

- **[CODE]** — an agent can write and run this locally on CPU. No GPU, no dataset, no checkpoint, no network.
- **[GPU RUN]** / **[CPU RUN]** — an operator launches a job. Not agent-executable.
- **[HUMAN]** — a judgement, verdict, interpretation or approval request. Not agent-executable.

## Tasks

- [ ] 1. Scope containment and the frozen-source gate
  - [ ] 1.1 [CODE] Extend `tests/test_scope_guard.py` for this feature
    - Add exactly two entries to `ALLOWED_FILES`: `agg_objectives.py` and `plan_agg.py`. `tests/` and
      `.kiro/specs/` are already covered by `ALLOWED_PREFIXES` and `run_ccr_pilot.sh` is already an
      allowlist member, so this feature adds no other entry (Requirement 4.5)
    - Add root-level `plan.py` to the byte-identity assertion, which today covers only
      `FROZEN_DIRS = ("planning", "datasets")`. Keep the newline-normalized sha256 comparison and the
      report-every-mismatch behaviour; `plan.py` is a single file rather than a directory, so extend the
      path collection rather than `FROZEN_DIRS` alone (Requirement 4.4)
    - Record the Scope_Amendment comment block above the two new entries, in the shape the `models/vit.py`
      precedent set (what was touched, why it was unavoidable, why it is safe, what guards it): `plan.py`
      builds its objective with `hydra.utils.call(cfg_dict["objective"])` and passes nothing else;
      `planning/objectives.py` closes over three scalars and receives no handle on the world model; so
      Agg_Head cannot reach the objective through any frozen argument channel and must be injected from
      outside the frozen paths. State that `plan_agg.py` rewrites `_target_` in its **own** `cfg_dict` and
      rebinds `plan.PlanEvaluator` in the wrapper's own process, and that both are runtime attribute rebinds
      which edit no file (Requirements 4.6, 4.7)
    - **Property 9: Frozen sources are byte-identical to the base revision**
    - **Validates: Requirements 4.3, 4.4**
    - **Gate, deliberately not optional:** this test runs before every launch and is the only automated check
      that Frozen_Paths still match Base_Revision `d73b9c6`. It lands before any feature code so the guard
      never reports a violation this feature did not cause
    - _Requirements: 4.3, 4.4, 4.5, 4.6_

  - [ ] 1.2 [CODE] Extend `tests/conftest.py` with the aggregated-space test doubles
    - Stand-in Agg_Head as `nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden),
      nn.ReLU(), nn.Linear(hidden, out_dim), nn.LayerNorm(out_dim))` mirroring `DinoV2Encoder.agg`'s `mlp`
      branch, plus an **identity-on-flattened-features** head variant, which Property 3 needs
    - Shared strategies: batch 1-4, `T` 2-6, patch count and channel width whose product matches the head's
      `in_dim` (and a deliberately mismatched pair for Property 6), `alpha` in `[0, 2]`, `base` in `[1, 4]`,
      mode in `{last, all, staged}`, `step` in `{None, 0 .. T}`, `agg_weight` in `[0, 3]` including exactly
      `0`, and a non-finite latent strategy covering `inf`, `-inf`, `nan` and denormals for Property 1
    - A stub encoder object exposing `agg_type`, `agg_mlp`, `agg_post_norm`, `_agg_mlp_in_dim` and
      `_agg_out_dim`, so `extract_agg_head` is testable with no checkpoint on disk
    - Every property test in this feature depends on this file, so it lands before any of them. Additive
      only: the existing CCR fixtures keep their current names and values
    - _Requirements: 1.1, 1.3, 1.8, 2.4_

- [ ] 2. `agg_objectives.py` foundations: constants, weight validation, head access
  - [ ] 2.1 [CODE] Create `agg_objectives.py` with the constants, the holder, and the head plumbing
    - Constants: `SWEEP_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)`, `TUNING_SEED = 400`,
      `REPORTING_SEEDS = (100, 200, 300)`, `AGG_WEIGHT_MAX = 3.0`, and `RUN_DIR_TEMPLATES`, a dict keyed by
      config name (`plan_gd`, `plan_gd_mpc`) holding the `hydra.run.dir` override strings that substitute
      `aggw${agg_weight}` for the `${ckpt_base_path}` component. One source of truth, so the shell driver and
      task 3.1's test read the same strings
    - `validate_agg_weight(value) -> float`: rejects negatives, `nan`, `inf`, non-numerics and values above
      `3`, naming the rejected value and the accepted closed interval `[0, 3]`
    - `_AggContext` dataclass and the module-level `AGG_CONTEXT` singleton: `agg_head`, `agg_weight`,
      `opt_steps`, `output_dir`, `instrumentation`; `publish(...)`, `require()` raising the actionable
      "launch `plan_agg.py` instead" message, and `clear()`
    - `extract_agg_head(encoder) -> (head, in_dim, out_dim)`: aborts naming the encountered `agg_type` when
      it is not `mlp`; keeps only `agg_mlp` and `agg_post_norm`; reads widths from `_agg_mlp_in_dim` /
      `_agg_out_dim` rather than parsing the checkpoint directory name (the `agg32` token is a run-dir
      literal, not a head width)
    - `_apply_head(z_visual, head)`: `(b, t, p, d)` → `reshape(b * t, p, d)` → head → `reshape(b, t, -1)`,
      mirroring `VWorldModel.total_curvature`'s `aggcos` reshape; resolves `head.to(device=z.device,
      dtype=z.dtype)` lazily on first call and caches it, calls `head.eval()` and
      `requires_grad_(False)` on its parameters; raises the Requirement 1.9 `ValueError` when
      `p * d != in_dim`, naming the received shape, the flattened width it implies and the required width,
      **before** the `nn.Linear` call so a bare mat1/mat2 message can never surface
    - Import `planning.objectives` read-only: call the existing factory, rebind nothing (Requirement 4.7)
    - _Requirements: 1.8, 1.9, 2.4, 2.6, 3.4, 3.5, 4.1, 4.7, 6.1_

  - [ ]* 2.2 [CODE] Write property test for weight validation (`tests/test_agg_weight_validation.py`)
    - **Property 13: Weight validation rejects out-of-domain values**
    - **Validates: Requirements 3.4, 3.5**

  - [ ]* 2.3 [CODE] Write property test for the shape error (`tests/test_agg_shape_errors.py`)
    - **Property 6: Shape mismatches are reported with both shapes**
    - **Validates: Requirements 1.9**

- [ ] 3. Run-directory separation (early gate, no GPU)
  - [ ] 3.1 [CODE] Write the run-directory separation check (`tests/test_agg_run_dir_separation.py`)
    - Resolve each `RUN_DIR_TEMPLATES` entry through Hydra `compose` against `conf/plan_gd.yaml` and
      `conf/plan_gd_mpc.yaml` (importing `custom_resolvers` so `replace_slash` is registered), once per
      weight in `(0,) + SWEEP_GRID`, and assert the seven resolved directories are **pairwise distinct**
    - Assert the shipped templates, resolved the same way, collapse all seven weights onto **one** directory.
      That is the failure being guarded against: `aggregate_results.py` forms one cell by appending seven
      lines to one `logs.json` and would average seven weights into one number without ever erroring
    - Assert the override preserves everything the aggregator parses: the `plan_outputs_gd` /
      `plan_outputs_gd_mpc` prefix, the `${replace_slash:${model_name}}_gH..._${goal_source}` component, and
      the trailing `obj${objective.mode}_init${planner.sub_planner.sample_type}` token; and assert the two
      settings resolve under **different** prefixes, so the MPC leg cannot land in the open-loop tree
    - Assert the templates are single-quoted in the driver, i.e. that `${...}` reaches Hydra rather than bash
    - **Gate, deliberately not optional:** without run-directory separation every sweep arm collides into one
      cell and the sweep curve is meaningless. This is checkable on CPU with no dataset and no checkpoint, so
      there is no reason to discover it on the pod
    - _Requirements: 2.7, 6.7, 7.3_

- [ ] 4. The combined objective `L_plan = L_spatial + w * L_agg`
  - [ ] 4.1 [CODE] Implement `create_agg_objective_fn` in `agg_objectives.py`
    - Build **two** callables from the frozen factory:
      `spatial_fn = create_objective_fn(alpha=alpha, base=base, mode=mode)` for L_spatial and
      `agg_fn = create_objective_fn(alpha=0, base=base, mode=mode)` for L_agg. Neither the coefficient
      vector nor the staged dispatch is copied — both terms go through the one implementation in
      `planning/objectives.py`, so they cannot drift from it (Requirements 1.2, 1.6)
    - `_agg_dicts`: apply the head frame-wise so `T` is preserved, and pass zero-valued proprio tensors with
      `alpha = 0`, which contributes exactly `0.0` to the sum. `last` mode needs no special case since the
      frozen `objective_fn_last` already slices `[:, -1:]` (Requirement 1.5)
    - Resolve `enabled = float(agg_weight) > 0.0` **once** at factory time. The disabled path performs no
      tensor operation on `loss_spatial` and returns the delegate's own tensor object, so bitwise equality is
      by identity and a non-finite L_agg cannot poison the sum (Requirement 3.2). No per-step `== 0.0`
      comparison on a tensor
    - Signature is a superset of `create_objective_fn`'s and takes `**kwargs`, so the unmodified objective
      block resolves against either factory and a future config key cannot become a `TypeError` inside frozen
      `hydra.utils.call`. `AGG_CONTEXT.require()` supplies Agg_Head
    - Return shape `(B,)`, on the device and in the dtype of `z_obs_pred["visual"]`; raw mean-squared L_agg
      and raw L_spatial, no rescaling of either term (Requirements 1.4, 1.8, 1.10)
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.10, 3.2_

  - [ ] 4.2 [CODE] Write property test for the bitwise-zero guarantee (`tests/test_agg_zero_bitwise.py`)
    - **Property 1: Zero weight is bitwise identity**
    - **Validates: Requirements 1.2, 3.1, 3.2, 5.5**
    - Compares **raw bytes** (`t.detach().cpu().numpy().tobytes()`), not `torch.equal`: `torch.equal` treats
      `nan` as unequal to itself and would mask exactly the failure this property exists to catch. Inputs
      include `inf`, `-inf`, `nan` and denormals, across all three modes, arbitrary `alpha`, `base` and
      `step`; also asserts a raw L_agg magnitude is still recorded at the recorded steps
    - **Gate, deliberately not optional:** this property is why the Baseline_Arm is a valid control rather
      than an approximation. Every downstream comparison — the paired zero-weight check, the sweep's
      same-seed reference point, the Paired_Comparison — rests on it, so it passes before any GPU job
    - _Requirements: 1.2, 3.1, 3.2, 5.5_

  - [ ] 4.3 [CODE] Write property test for stage and coefficient reuse (`tests/test_agg_staged_coeffs.py`)
    - **Property 3: Stage selection and coefficients are the frozen module's**
    - **Validates: Requirements 1.2, 1.6**
    - With the identity head on flattened patch features and `alpha = 0`, L_agg must equal the value the
      unmodified `planning.objectives.create_objective_fn` callable returns for the same mode, `base` and
      `step`; and in `staged` mode it must equal the frozen `last`-mode value when `step < T - 1` and the
      frozen `all`-mode value otherwise. Generated over `T`, `base`, `step` and arbitrary latent dictionaries
    - **Gate, deliberately not optional:** this is the only check that the coefficient reuse is **exact**
      rather than approximate. It must be green before the MPC confirmation run, because `staged` is the mode
      that run uses (Requirement 8.3) and a coefficient discrepancy there would be invisible in the success
      rate
    - _Requirements: 1.2, 1.6_

  - [ ]* 4.4 [CODE] Write property test for additive decomposition (`tests/test_agg_decomposition.py`)
    - **Property 2: Additive decomposition with no hidden normalization**
    - **Validates: Requirements 1.3, 1.4, 1.8, 1.10, 3.4**

  - [ ]* 4.5 [CODE] Write property test for `last`-mode locality (`tests/test_agg_last_mode.py`)
    - **Property 4: Last mode depends only on the final predicted frame**
    - **Validates: Requirements 1.5**

  - [ ]* 4.6 [CODE] Write property test for head freezing and differentiability (`tests/test_agg_head_frozen.py`)
    - **Property 5: Agg_Head is frozen and differentiable through its input**
    - **Validates: Requirements 1.7**
    - Asserts the head's parameter **bytes** are unchanged after a backward pass and that
      `z_obs_pred["visual"].grad` is populated

  - [ ]* 4.7 [CODE] Write property test for frozen-module immutability (`tests/test_agg_objectives_untouched.py`)
    - **Property 8: planning.objectives is left untouched**
    - **Validates: Requirements 4.7**

- [ ] 5. Instrumentation of both loss components
  - [ ] 5.1 [CODE] Implement `AggInstrumentation` in `agg_objectives.py`
    - Count objective invocations to recover the optimizer step index: `planning/gd.py` calls the objective
      exactly once per inner iteration in order, and `eval_every` is `-1` so the early `break` is
      unreachable, which makes the call index the step index. `should_record()` fires at `step_index == 0`
      and `step_index == opt_steps - 1`; `advance()` rolls over to the next `plan_call`
    - Each record carries `plan_call`, `mpc_step_arg` (the frozen `step` argument as received), `step_index`,
      `updates_applied`, `l_spatial`, `l_agg` and `ratio`. Write `step_100_semantics` into the file stating
      plainly that with `opt_steps: 100` the indices are 0-99, so Requirement 5.2's "step 100" is the 100th
      evaluation (`step_index 99`, formed after 99 Adam updates), and that no evaluation exists after the
      100th update
    - `ratio` is `agg_weight * l_agg / l_spatial`, or the string `"undefined"` exactly when `l_spatial` is
      `0.0` (Requirement 5.6). At `agg_weight == 0` the ratio is `0.0`, not `"undefined"`, and the raw L_agg
      is still recorded (Requirement 5.5) — computed under `torch.no_grad()` so it never joins the autograd
      graph
    - Self-check: assert the received `step` argument is constant within a counted plan call and record
      `step_boundary_mismatch: true` rather than silently mislabelling if it is not. Write failures are
      counted in `record_failures`, never raised — a bad write must not lose a 15-minute evaluation
    - Emit `agg_instrumentation.json` with a `headline` block for `plan_call == 0` plus every record
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6_

  - [ ] 5.2 [CODE] Pin the step-counting scheme against the real planner (`tests/test_agg_step_counting.py`)
    - The whole instrumentation index rests on a **read of frozen code**: `planning/gd.py` calls the objective
      exactly once per inner iteration, `eval_every` is `-1` so the early `break` is unreachable, and nothing
      calls the objective outside the loop. Nothing in the plan currently pins that read, and the recorder's
      own `step` self-check cannot detect desync in the open-loop setting, where `step` is always `None`
    - Drive the **real** `planning.gd.GDPlanner` on CPU against a stub world model and a counting objective:
      assert the objective is invoked exactly `opt_steps` times per `plan()` call, in order, with no call
      before the first update and none after the last, across generated `opt_steps` and horizons
    - **Gate, deliberately not optional:** if this read is wrong the Instrumentation_Record silently
      mislabels which optimizer step it describes, and the term-magnitude interpretation that Requirement 5
      exists to support — and that decides whether the Sweep_Grid brackets anything useful — is read off the
      wrong step. A frozen-code read that no test pins is the failure mode this closes
    - _Requirements: 5.1, 5.2, 5.3_

  - [ ]* 5.3 [CODE] Write property test for instrumentation (`tests/test_agg_instrumentation.py`)
    - **Property 7: Instrumentation is complete, correctly indexed, and round-trips**
    - **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.6**
    - Includes the write/read round trip, since the sweep curve and the term-magnitude interpretation are
      read out of this file

- [ ] 6. Per-episode outcome capture (design decision: the `plan.PlanEvaluator` rebind)
  - [ ] 6.1 [CODE] Implement `RecordingPlanEvaluator` and the outcome sink in `agg_objectives.py`
    - This is a **decision the design took, not a derivation**, and it is recorded here so it is not lost:
      `plan.py` persists only means (`PlanEvaluator._compute_rollout_metrics` reduces `successes` into
      `logs["success_rate"]` and the per-episode array is dropped), and the per-episode videos that would
      encode the outcome are written for `n_plot_samples = 10` only and only when `decode_for_viz` is true,
      which the launcher sets false. Requirements 7.4 and 11.4 need the vectors, so the wrapper rebinds
      `plan.PlanEvaluator` — the module-level name `plan.py` imports and constructs directly — to a subclass
      defined in the new module
    - `RecordingPlanEvaluator.eval_actions` delegates to `super()`, records `result[1]` (the `successes`
      vector) through `AGG_CONTEXT.record_episodes(filename, ...)`, catches `OSError` into
      `note_record_failure`, and returns the object `super()` returned. Read-only observer: adds no state any
      computation reads, consumes no RNG, performs no tensor work
    - `AGG_CONTEXT.record_episodes` / `flush_and_clear` append `agg_episode_outcomes.jsonl`, one line per
      `eval_actions` call: `{"filename", "plan_call", "n_evals", "successes"}`. The reported vector is the
      `filename == "output_final"` row, the eval `PlanWorkspace.perform_planning` uses for
      `final_eval/success_rate`; MPC's intermediate `plan{iter}` rows are recorded but are not the reported
      result
    - No file under `planning/` is edited and no name inside `planning/` is rebound, so `plan.py`'s bytes and
      the task 1.1 assertion both still hold
    - _Requirements: 7.4, 11.4, 4.4_

  - [ ]* 6.2 [CODE] Write property test for the recording evaluator (`tests/test_agg_recording_evaluator.py`)
    - **Property 11: The recording evaluator is transparent**
    - **Validates: Requirements 7.4**
    - This is the guard on the rebind decision above: it must return the delegate's **identical** object and
      append exactly one outcome row whose success vector equals the tuple's success element. Strongly
      recommended despite the `*`, since it is the only automated evidence that the rebind is observational

- [ ] 7. Sweep selection and paired counting
  - [ ] 7.1 [CODE] Implement `select_agg_weight` and `paired_counts` in `agg_objectives.py`
    - `select_agg_weight(rows)`: highest open-loop success rate at the Tuning_Seed wins; on a tie the
      **smallest** tied weight is selected and the tie is recorded; rows at Reporting_Seeds contribute
      nothing, so seeds 100/200/300 cannot influence the choice (Requirements 6.4, 6.5, 6.6)
    - `paired_counts(candidate, baseline)`: candidate-only, baseline-only and matching counts over two equal
      length boolean vectors, summing to the vector length (Requirement 11.4)
    - _Requirements: 6.4, 6.5, 6.6, 11.4_

  - [ ]* 7.2 [CODE] Write property test for weight selection (`tests/test_agg_weight_selection.py`)
    - **Property 10: Weight selection uses only the Tuning_Seed**
    - **Validates: Requirements 6.4, 6.5, 6.6, 6.7**

  - [ ]* 7.3 [CODE] Write property test for paired counts (`tests/test_agg_paired_counts.py`)
    - **Property 14: Paired counts partition the episodes**
    - **Validates: Requirements 11.4**

- [ ] 8. Wrapper entry point `plan_agg.py`
  - [ ] 8.1 [CODE] Create `plan_agg.py` with the Hydra entry and the protocol checker
    - `@hydra.main(config_path="conf", config_name="plan_gd")`, written **without** `version_base` exactly as
      `plan.main` is, because `plan.planning_main` depends on the cwd being the run directory, which is the
      Hydra-version-dependent `job.chdir` default `plan.main` already relies on
    - `resolve_protocol(config_name, cfg)` holds one expected table **per setting**, keyed off
      `HydraConfig.get().job.config_name`: open-loop (`plan_gd`) expects `max_iter 1` and
      `n_taken_actions 25`; MPC (`plan_gd_mpc`) expects `max_iter 20` and `n_taken_actions 5`. `n_evals 50`,
      `objective.mode` (`last` / `staged`), `objective.alpha 1`, `sub_planner.horizon 25`,
      `sub_planner.lr 0.1`, `sub_planner.sample_type zero`, `sub_planner.action_noise 0` and
      `sub_planner.opt_steps 100` are common. Task 13.1 settles this per-setting reading of the
      Evaluation_Protocol **before** this task is written, which is why it sits in wave 0
    - Deviation aborts before any load, naming the field, the expected value and the resolved value
      (Requirement 8.7). All ten resolved values go into the manifest either way (Requirement 8.6), so if the
      literal reading of 8.4 turns out to be intended, the record shows exactly which two fields differ
    - Validate the weight through `validate_agg_weight` first, before any load, so a bad weight costs seconds
      rather than a dataset load
    - _Requirements: 3.1, 3.5, 4.2, 8.1, 8.2, 8.3, 8.4, 8.6, 8.7_

  - [ ] 8.2 [CODE] Complete `plan_agg.py`: head load, publication, objective rewrite, delegation
    - Load Agg_Head with the frozen helper `plan.load_ckpt(model_ckpt, device="cpu")`, take
      `payload["encoder"]`, pass it through `extract_agg_head`, and abort if the `encoder` key is absent
      since Requirement 2.4 cannot then be met. Warn and record both widths if they differ from 1568 / 128
    - This load happens **before** `planning_main` calls `utils.seed(cfg_dict["seed"])`, which reseeds
      `random`, `torch`, `numpy` and every CUDA generator. Every RNG state inside `planning_main` is
      therefore exactly what `plan.py` would have had, which is what keeps Requirement 3.3 and the
      Paired_Comparison exact. Do not move it later
    - `AGG_CONTEXT.publish(head, w, opt_steps=resolved sub_planner.opt_steps, output_dir=abs saved_folder)`;
      rebind `plan.PlanEvaluator = agg_objectives.RecordingPlanEvaluator`; rewrite
      `cfg_dict["objective"]["_target_"] = "agg_objectives.create_agg_objective_fn"` and
      `cfg_dict["objective"]["agg_weight"] = w` in the wrapper's **own** dict, so `_target_` cannot be
      forgotten and the only override a user types is `+agg_weight=<w>`
    - Write `agg_run_manifest.json` (feature, config name, setting, resolved weight, seed, checkpoint, head
      widths and `agg_type`, all ten resolved protocol fields, `protocol_ok`, git rev), then call
      `plan.planning_main(cfg_dict)` unchanged (Requirement 2.1). `finally`: restore `plan.PlanEvaluator` and
      `AGG_CONTEXT.flush_and_clear()`
    - Requirement 2.7 is met by construction — every result file (`logs.json`, `plan_targets.pkl`, videos,
      PNGs) is written by frozen code into a `plan_outputs_*` directory and the wrapper only **adds** files
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 5.4, 8.5_

  - [ ]* 8.3 [CODE] Write property test for protocol enforcement (`tests/test_agg_protocol.py`)
    - **Property 12: Protocol deviations are named**
    - **Validates: Requirements 8.6, 8.7**
    - Runs against both per-setting expected columns, so the MPC column is covered as well as the open-loop one

  - [ ]* 8.4 [CODE] Write the integration tests (`tests/test_agg_integration.py`)
    - Example-based, 1-3 cases each, not property tests: `aggregate_results.py` parses a wrapper-shaped
      output tree unmodified (Requirement 2.7); a tiny synthetic checkpoint yields the expected head through
      `extract_agg_head` (Requirement 2.4); encoder parameter hashes are unchanged across a short run
      (Requirement 8.5)
    - _Requirements: 2.4, 2.7, 8.5_

- [ ] 9. Launcher integration
  - [ ] 9.1 [CODE] Add the two env-gated hooks to `run_ccr_pilot.sh`
    - `PLAN_ENTRY="${PLAN_ENTRY:-plan.py}"` replaces the two literal `plan.py` tokens in `run_eval_jobs`;
      `SETTINGS="${SETTINGS:-both}"` (`ol` | `mpc` | `both`) guards the two seed loops. Both default to
      today's behaviour, so the CCR evaluation path stays byte-behaviour-identical
    - `run_eval_jobs` must pass the **per-setting** `hydra.run.dir` override from `RUN_DIR_TEMPLATES` — the
      MPC leg needs the `plan_outputs_gd_mpc` prefix, so one string for both settings is wrong
    - Everything else is reused untouched: the MIG preflight refusal, the environment recipe,
      `PLAN_SERIAL_ENV=1`, the chaining protocol, the one-job-at-a-time driver (Requirements 9.1-9.3).
      `run_ccr_pilot.sh` is already an `ALLOWED_FILES` member, so this edit adds no allowlist entry
    - _Requirements: 9.1, 9.2, 9.3_

  - [ ]* 9.2 [CODE] Extend the driver-contract test for the new hooks (`tests/test_run_ccr_pilot_sh.py`)
    - Assert `PLAN_ENTRY` and `SETTINGS` default to `plan.py` and `both`, that no literal `plan.py` token
      remains in `run_eval_jobs`, that each `SETTINGS` value selects the matching loop, that the two settings
      receive different `hydra.run.dir` prefixes, and that the existing `ps` preflight guard and environment
      recipe are unchanged
    - _Requirements: 9.1, 9.2, 9.3_

- [ ] 10. Checkpoint - all local code and tests green before any pod job
  - Ensure all tests pass, ask the user if questions arise.
  - Five gates must be green before anything is launched: 1.1 (scope guard, including the `plan.py`
    byte-freeze), 3.1 (run-directory separation), 4.2 (Property 1, bitwise zero), 4.3 (Property 3, staged
    coefficient equality) and 5.2 (the step-counting scheme against the real `GDPlanner`). Task 13.1's
    interpretation is settled at wave 0, before task 8.1 encodes it. No pod job of any kind starts before this
    checkpoint.

- [ ] 11. Paired zero-weight end-to-end check (gates the sweep; also the sweep's zero arm)
  - [ ] 11.1 [GPU RUN] Run `plan.py` and `plan_agg.py --agg_weight=0` at seed 400, open-loop
    - Both through the Job_Launcher (`bash run_ccr_pilot.sh eval <ckpt>`, `SETTINGS=ol SEEDS=400`), serially,
      one job at a time on the `1g.45gb` MIG slice; the second with `PLAN_ENTRY=plan_agg.py
      "+agg_weight=0"`. ~15 min for the pair
    - **Both legs pass an explicit `hydra.run.dir`. This is the hazard, stated here so it is not silently
      reintroduced:** the shipped template carries neither the seed nor the weight, so a `plan.py` leg without
      an override resolves to the *same* directory that already holds the recorded 75.33 +/- 6.11 open-loop
      Platform_Baseline. It would append a seed-400 line to that `logs.json` and `aggregate_results.py` would
      turn the cell into a 4-seed mean — and that cell is the reference the entire Acceptance_Gate is measured
      against. The `plan.py` leg therefore goes to a scratch prefix that cannot collide with any reported
      cell, e.g. a `plan_outputs_gd/..._paircheck/` component
    - **The `plan_agg.py` leg does NOT go to scratch.** It writes to the real `aggw0` cell from
      `RUN_DIR_TEMPLATES`, because this leg is also the Baseline_Arm the sweep curve is read against
      (Requirement 6.3), so it must land where task 12.7's aggregation reads it. Scratch for the frozen-entry
      leg, the real weight-keyed cell for the wrapper leg
    - Compare the two `output_final` per-episode success vectors element for element. `plan.py` writes only a
      mean, so the comparison is against the wrapper's `agg_episode_outcomes.jsonl` on one side and the
      frozen `logs.json` success rate plus the run's own per-episode record on the other; if `plan.py`'s
      vector is not recoverable, the mean equality over 50 episodes at identical seed is the check, and that
      is what gets recorded
    - Also confirm `agg_instrumentation.json` carries a raw L_agg magnitude at both recorded steps with
      `ratio` `0.0`, which is Requirement 5.5 observed on real tensors rather than synthetic ones
    - This job pair replaces a separate zero-weight sweep arm: the `plan_agg.py` leg is the same GPU job — 
      `plan_agg.py` at `agg_weight=0`, seed 400, open-loop, same checkpoint — so it is run once and read twice
      (Requirement 6.3). Not the recorded Platform_Baseline, which is a different seed set and a mean only
    - Not agent-executable: needs the pod, the dataset and the Target_Cell checkpoint
    - _Requirements: 3.3, 5.5, 6.3, 9.1, 9.2, 9.4, 9.6_

  - [ ] 11.2 [HUMAN] Record the paired zero-weight verdict
    - Identical per-episode vectors is the pass condition. Anything else means the bitwise-zero design does
      not hold through a real run — most likely an RNG perturbation from the wrapper's extra checkpoint load
      or the evaluator rebind — and the **sweep does not launch** until it is explained
    - Record the verdict, both success rates, and the instrumentation magnitudes in the project progress log.
      The step-0 magnitudes of L_spatial and L_agg are the first real evidence of the two terms' relative
      scale, which is what decides whether the Sweep_Grid brackets the useful range at all
    - _Requirements: 3.3, 5.1, 5.2, 5.3_

- [ ] 12. Weight sweep on the Tuning_Seed (6 non-zero open-loop arms, strictly serial, ~35 min total)
  - Every arm: `FOREGROUND=1 PLAN_ENTRY=plan_agg.py SETTINGS=ol SEEDS=400 bash run_ccr_pilot.sh eval "$CKPT"
    "+agg_weight=$W" "$RUNDIR"` with the open-loop `hydra.run.dir` override. Open-loop only, Tuning_Seed
    only, so the Reporting_Seeds contribute nothing to weight selection (Requirements 6.2, 6.6). One arm per
    wave: the MIG slice holds exactly one job
  - The zero-weight arm is **not** repeated here: it is the `plan_agg.py` leg of task 11.1, the same GPU job at
    the same seed, setting and checkpoint, writing the real `aggw0` cell (Requirement 6.3). With it the sweep
    is 7 arms and ~40 min against Requirement 9.4
  - [ ] 12.1 [GPU RUN] Sweep arm `agg_weight=0.01`
    - Bottom of the accepted interval; see task 12.8 on a boundary selection
    - _Requirements: 6.1, 6.2, 6.7, 9.4_

  - [ ] 12.2 [GPU RUN] Sweep arm `agg_weight=0.03`
    - _Requirements: 6.1, 6.2, 6.7, 9.4_

  - [ ] 12.3 [GPU RUN] Sweep arm `agg_weight=0.1`
    - The paper-literal value: `tab:long_horizon` reports exactly `L_spatial + 0.1 * L_agg`. It is one arm of
      seven, not a privileged one, since the paper applies the term only at the 50-step horizon
    - _Requirements: 6.1, 6.2, 6.7, 9.4_

  - [ ] 12.4 [GPU RUN] Sweep arm `agg_weight=0.3`
    - _Requirements: 6.1, 6.2, 6.7, 9.4_

  - [ ] 12.5 [GPU RUN] Sweep arm `agg_weight=1`
    - _Requirements: 6.1, 6.2, 6.7, 9.4_

  - [ ] 12.6 [GPU RUN] Sweep arm `agg_weight=3`
    - Top of the accepted interval (Requirement 3.4); `validate_agg_weight` rejects anything above it, so the
      grid cannot be extended upward without a spec change. See task 12.8
    - _Requirements: 6.1, 6.2, 6.7, 9.4_

  - [ ] 12.7 [CPU RUN] Aggregate the sweep and select the weight
    - `python aggregate_results.py`, then `select_agg_weight` over the seven rows — the six arms above plus the
      `aggw0` row written by task 11.1's `plan_agg.py` leg. Confirm the seven arms landed in seven distinct
      directories on disk — task 3.1 checks the template, this checks the outcome
    - Assemble the sweep curve: open-loop success rate at the Tuning_Seed plus the Instrumentation_Record for
      every Sweep_Grid value and for the Baseline_Arm (Requirement 6.7)
    - _Requirements: 6.4, 6.5, 6.7_

  - [ ] 12.8 [HUMAN] Record the sweep curve, the selected weight, and any boundary selection
    - Record `W_STAR`, any tie and how it was broken (smallest tied weight), and the effective ratio
      `Agg_Weight * L_agg / L_spatial` at steps 0 and 100 for each arm. A curve that is flat within the ~5.7
      point binomial standard error at n=50 is itself a finding and belongs in the record
    - **Boundary selection has a defined branch, decided here rather than improvised on the pod.** If
      `select_agg_weight` returns `0.01` or `3.0`, the optimum is **unbracketed** — the sweep cannot tell an
      interior peak from a curve still rising at the edge of the grid. The decision: record the selection *as a
      boundary selection*, carry it into the confirmation run **as-is**, and report it as-is. Do not extend the
      grid on the spot. Downward is outside Sweep_Grid and upward is refused by `validate_agg_weight`, which
      rejects anything above `3` (Requirement 3.4), so **any grid extension is a spec change and requires the
      Requirement 11.7 recorded approval before a further job is launched**
    - Note explicitly that `W_STAR` was chosen on seed 400 alone, so the confirmation run is the first time
      the Reporting_Seeds see this weight
    - _Requirements: 6.4, 6.5, 6.6, 6.7, 3.4, 11.7_

- [ ] 13. The Requirement 8.4 reading (interpretation; settle it first, at wave 0)
  - [ ] 13.1 [HUMAN] Confirm the per-setting protocol reading against `conf/plan_gd_mpc.yaml`
    - **Sequenced at wave 0, before any code.** This is a zero-cost human judgement, and it governs the
      per-setting expected table that task 8.1 writes at wave 6. Settling it afterwards would mean confirming
      an interpretation the code already encodes
    - Requirement 8.4's field list — `max_iter 1`, `n_taken_actions 25`, `sub_planner.horizon 25`,
      `sub_planner.lr 0.1`, `sub_planner.sample_type zero`, `sub_planner.action_noise 0`,
      `sub_planner.opt_steps 100` — matches the shipped `planner` block of `conf/plan_gd.yaml` **only**.
      `conf/plan_gd_mpc.yaml`, which Requirement 8.3 mandates for the MPC setting, ships `max_iter: 20` and
      `n_taken_actions: 5` and is otherwise identical in the `sub_planner` block
    - Taken literally for MPC, 8.4 would force `max_iter 1`, which makes the MPC setting
      open-loop-with-a-staged-objective and could not reproduce the 82.00 Platform_Baseline MPC number that
      Requirement 8's own user story exists to stay comparable with
    - **The framing to use, corrected:** the Evaluation_Protocol is the ten-field per-setting table, which
      *combines shipped config defaults with the overrides `run_ccr_pilot.sh` already applies*. It is **not**
      "no override of any protocol field relative to the mandated config file" — that reading is factually
      wrong. `conf/plan_gd.yaml` ships `objective.alpha: 0` and `conf/plan_gd_mpc.yaml` ships
      `objective.mode: all`, while `run_ccr_pilot.sh` already overrides both to `alpha=1` and
      `mode=last`/`staged` per Requirements 8.2 and 8.3. The protocol therefore demonstrably **does** require
      overrides in the `objective` block; Requirement 8.4's list happens to coincide with `plan_gd.yaml`'s
      `planner` block alone. Anywhere the old "no override relative to the mandated config file" phrasing
      survives — here or in the design's section 7 reasoning — it is to be replaced with this framing
    - The per-setting expected table in task 8.1 is unchanged by this correction. Only the justification changes
    - **This is an interpretation of a requirement, not an implementation detail**, which is why it is a
      human confirmation rather than a code comment. All ten resolved values are in every manifest, so if the
      literal reading of 8.4 is intended the record already shows which two fields differ and the re-run is a
      two-flag change
    - _Requirements: 8.2, 8.3, 8.4, 8.6, 8.7_

- [ ] 14. Three-seed confirmation run (12 runs, both settings, ~3 h, strictly serial)
  - **Budget, corrected:** `REPRODUCTION.md` records ~25 min per MPC seed on this eval path, so one arm of 3
    open-loop plus 3 MPC seeds is ~1.5 h. Task 14 runs **two** arms (14.1 baseline, 14.2 candidate), 6 runs
    each, so the confirmation is **~3 h**, not ~1.5 h. Requirement 9.5 carries the same understatement and
    should be read as ~3 h for the confirmation
  - Run **once**, for the selected `W_STAR` only (Requirement 7.2). Any later weight is an exploratory
    follow-up and does not replace this as the reported result (Requirement 7.5). The w=0 arm is re-run
    rather than reusing the recorded Platform_Baseline, because the recorded numbers are means and the
    Paired_Comparison needs per-episode vectors from the wrapper's own outcome file
  - [ ] 14.1 [GPU RUN] Baseline_Arm: `agg_weight=0`, seeds 100/200/300, open-loop and MPC
    - `FOREGROUND=1 PLAN_ENTRY=plan_agg.py SETTINGS=both SEEDS="100 200 300"` with the per-setting
      `hydra.run.dir` overrides. 6 runs, ~1.5 h for this arm alone (3 MPC seeds at ~25 min each dominate)
    - _Requirements: 7.1, 7.4, 8.2, 8.3, 9.1, 9.2, 9.5_

  - [ ] 14.2 [GPU RUN] Candidate_Arm: `agg_weight=$W_STAR`, seeds 100/200/300, open-loop and MPC
    - 6 runs, ~1.5 h, serial after 14.1, for ~3 h across the two arms. Requires task 4.3 green (Property 3) —
      the MPC leg runs `objective.mode=staged`, and the staged coefficient reuse is only proved exact by that
      property — and task 13.1 confirmed
    - The 12 runs are why Requirement 9.5's ~1.5 h figure is roughly 2x understated; read it as ~3 h. **If it
      overruns, the overrun is recorded, not traded against the protocol**: no reduction of `n_evals`, no
      dropped seed, no relaxed planner hyperparameter
    - _Requirements: 7.1, 7.2, 7.4, 8.1, 8.2, 8.3, 8.4, 9.5_

  - [ ] 14.3 [CPU RUN] Aggregate the confirmation run and compute the Paired_Comparison
    - `python aggregate_results.py` for open-loop and MPC means and standard deviations over the
      Reporting_Seeds (Requirement 7.3)
    - `paired_counts` per Reporting_Seed over the two `output_final` vectors from
      `agg_episode_outcomes.jsonl`: candidate-only wins, baseline-only wins, matching outcomes. The
      comparison is exact because both arms draw identical episodes at the same `cfg.seed`
    - _Requirements: 7.3, 7.4, 9.6, 11.4_

- [ ] 15. Acceptance gate
  - [ ] 15.1 [HUMAN] Record the Acceptance_Gate verdict
    - `python ccr_acceptance_gate.py --cand-ol-seeds <...> --cand-mpc-seeds <...> --base-ol 75.33
      --base-mpc 82.00`. Candidate means and the Platform_Baseline only, no threshold arguments: the
      predicate is reused as it stands (Requirement 10.7)
    - Pass requires beating Paper_Target (77.33 open-loop, 85.33 MPC) **and** Platform_Baseline by more than
      6 points on both settings, which resolves to 81.33 open-loop and 88.00 MPC. One condition alone is a
      `fail`; both conditions with the weaker margin at or below 6 points is `inconclusive`. Report the ~5.7
      point binomial standard error (n=50, p≈0.8) alongside every comparison
    - Record the verdict, both margins and the Paired_Comparison counts in the project progress log
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7_

- [ ] 16. Negative_Result_Record
  - [ ] 16.1 [HUMAN] Write the Negative_Result_Record on a `fail` or `inconclusive` verdict
    - All four parts: (1) the sweep curve — open-loop success rate at the Tuning_Seed for every Sweep_Grid
      value and for the Baseline_Arm (Requirement 11.2); (2) the Candidate_Arm 3-seed open-loop and MPC means
      and standard deviations over the Reporting_Seeds (Requirement 11.3); (3) the Paired_Comparison against
      the Baseline_Arm, per Reporting_Seed, giving candidate-only, baseline-only and matching counts
      (Requirement 11.4); (4) the conclusion that the paper's long-horizon aggregated term does not transfer
      to the short-horizon Target_Cell (Requirement 11.5)
    - Include the Instrumentation_Record magnitudes: a raw sum of a 1568-dimensional and a 128-dimensional
      squared distance is only interpretable with both terms' scales on the page, and the effective ratio at
      steps 0 and 100 is what distinguishes "the term was too weak to matter" from "the term dominated and
      broke the planner"
    - **The investigation then ends.** No further evaluation or training job is launched (Requirement 11.6),
      and Requirement 11.7 forbids progressing to the sketched follow-on phases — the learned
      temporal-distance cost (Phase 2) and the 2x2 straightening-by-cost study (Phase 3) — **without
      explicit approval recorded before any further job starts**
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7_

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster path. **Six test tasks are deliberately
  not marked optional**, because they are gates rather than checks: 1.1 (the scope guard, including the
  `plan.py` byte-freeze, Property 9, Requirements 4.3-4.6), 3.1 (run-directory separation), 4.2 (Property 1,
  the bitwise-zero guarantee that makes the Baseline_Arm a valid control), 4.3 (Property 3, the only proof
  that the staged coefficient reuse is exact rather than approximate), 5.2 (the step-counting scheme pinned
  against the real `GDPlanner`, since the instrumentation index otherwise rests on an unpinned read of frozen
  code and the recorder's own self-check is blind in open-loop) and, in the process sense, task 10's
  checkpoint. Task 6.2 (Property 11) is marked optional by format convention but is the only automated
  evidence that the `plan.PlanEvaluator` rebind is observational, so skipping it is not recommended.
- **[CODE]** tasks are agent-executable on CPU with no GPU, dataset, checkpoint or network — the stand-in head
  and stub encoder in `tests/conftest.py` exist exactly for that, and Properties 1-8 and 10-14 all run against
  small synthetic tensors. Task 5.2 is CPU-only too: it drives the real `planning.gd.GDPlanner` against a stub
  world model, which needs no checkpoint and no dataset. **[GPU RUN]**, **[CPU RUN]** and **[HUMAN]** tasks are operator or judgement work,
  listed for sequencing rather than agent execution.
- `agg_objectives.py` is built across tasks 2.1, 4.1, 5.1, 6.1 and 7.1, and `plan_agg.py` across 8.1 and 8.2.
  Those tasks write the same file, so the dependency graph places each in its own wave. The property tests are
  one file per property so they never collide and can be scheduled in parallel.
- Every property test docstring carries the tag
  **Feature: aggregated-space-planning-cost, Property N: <property text>**, minimum 100 Hypothesis examples.
  Property 1 compares raw bytes rather than using `torch.equal`, which treats `nan` as unequal to itself.
- Serialization is real, not stylistic: the `1g.45gb` MIG slice holds one job (Requirement 9.2), so 11.1,
  12.1-12.6, 14.1 and 14.2 each occupy their own wave.
- Wall-clock budgets, corrected against `REPRODUCTION.md`'s ~25 min per MPC seed: paired zero-weight check
  ~15 min (its `plan_agg.py` leg doubles as the sweep's zero arm); sweep ~35 min across the 6 non-zero arms,
  ~40 min counting the zero arm; confirmation **~3 h** across 12 runs, since task 14 runs two arms of 6 runs
  each and one arm alone is ~1.5 h. Total GPU across the plan is **~4 h**, not ~2.5 h. Requirement 9.5's
  ~1.5 h confirmation figure carries the same understatement and should be read as ~3 h. The budget is tight;
  an overrun is recorded, not traded against the protocol.
- The zero-weight GPU job exists once, in task 11.1, and is read twice: as the paired-check wrapper leg and as
  the sweep's Baseline_Arm reference point (Requirement 6.3). Its `plan.py` counterpart is the one job in the
  plan that must be steered into a scratch run directory, because the shipped template would otherwise append
  a seed-400 line to the recorded Platform_Baseline cell the Acceptance_Gate is measured against.
- Two design **decisions** rather than derivations have their own tasks so they are not lost in
  implementation: the `plan.PlanEvaluator` rebind for per-episode outcome capture (task 6.1, guarded by
  Property 11 in task 6.2) and the Requirement 8.4 versus `conf/plan_gd_mpc.yaml` reading (task 13.1, at
  wave 0, implemented as the per-setting expected table in task 8.1). A third now has one too: the
  boundary-weight branch, in task 12.8 — a selection of `0.01` or `3.0` is unbracketed, is recorded and
  reported as such, and any grid extension needs the Requirement 11.7 recorded approval.
- Requirements 7.2, 7.5, 9.1-9.5, 11.1, 11.6 and 11.7 are process rules recorded in the result document
  rather than automated tests. Their automatable fragments are tasks 9.2, 8.3 (Property 12) and 3.1.
- The rejected alternatives stay rejected: no monkeypatching of `planning.objectives.create_objective_fn`
  (Requirement 4.7 forbids it, and it would recurse since L_spatial is produced by calling that factory), and
  no edit to `conf/plan_gd.yaml` or `conf/plan_gd_mpc.yaml` (not in the allowlist, and Requirements 8.2/8.3
  name those files as the protocol).

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2", "13.1"] },
    { "id": 1, "tasks": ["2.1"] },
    { "id": 2, "tasks": ["2.2", "2.3", "3.1", "4.1"] },
    { "id": 3, "tasks": ["4.2", "4.3", "5.1"] },
    { "id": 4, "tasks": ["4.4", "4.5", "4.6", "4.7", "5.2", "5.3", "6.1"] },
    { "id": 5, "tasks": ["6.2", "7.1"] },
    { "id": 6, "tasks": ["7.2", "7.3", "8.1"] },
    { "id": 7, "tasks": ["8.2", "9.1"] },
    { "id": 8, "tasks": ["8.3", "8.4", "9.2"] },
    { "id": 9, "tasks": ["11.1"] },
    { "id": 10, "tasks": ["11.2"] },
    { "id": 11, "tasks": ["12.1"] },
    { "id": 12, "tasks": ["12.2"] },
    { "id": 13, "tasks": ["12.3"] },
    { "id": 14, "tasks": ["12.4"] },
    { "id": 15, "tasks": ["12.5"] },
    { "id": 16, "tasks": ["12.6"] },
    { "id": 17, "tasks": ["12.7"] },
    { "id": 18, "tasks": ["12.8"] },
    { "id": 19, "tasks": ["14.1"] },
    { "id": 20, "tasks": ["14.2"] },
    { "id": 21, "tasks": ["14.3"] },
    { "id": 22, "tasks": ["15.1"] },
    { "id": 23, "tasks": ["16.1"] }
  ]
}
```
