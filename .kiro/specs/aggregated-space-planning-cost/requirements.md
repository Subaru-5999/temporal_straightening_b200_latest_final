# Requirements Document

## Introduction

This feature adds an aggregated-space goal cost to latent planning, so the planning objective becomes
`L_plan = L_spatial + w * L_agg`. `L_spatial` is the existing squared goal distance over patch features
computed by `planning/objectives.py`; `L_agg` is the squared goal distance measured after the encoder's
aggregation head (`DinoV2Encoder.agg`, `agg_type: mlp`, `196 * 8 = 1568 -> 512 -> 512 -> 128` followed by
`LayerNorm`). The term is config-gated by a single weight `w`, whose default of `0` reproduces current
planning behaviour bitwise. No retraining is involved: the term is evaluated against an existing trained
checkpoint.

Evidence motivating the feature, established before this spec:

- The target paper is *Temporal Straightening for Latent Planning*, arXiv 2603.12231, with LaTeX sources in
  `paper_tex/`.
- The paper's long-horizon table `tab:long_horizon` reports exactly `L_plan = L_spatial + 0.1 * L_agg`. On
  PushT with `L_curv` enabled it reports 20.00 open-loop / 33.33 MPC for that objective versus 13.33 / 24.00
  for the spatial-only objective, i.e. gains of +6.67 and +9.33 percentage points.
- The paper applies the aggregated term only to the 50-step long-horizon setting. It never applies the term
  to the short-horizon Table 1 cell, which is the cell targeted here, so transfer to the short horizon is an
  open question rather than a restatement of a published result.
- The paper's limitations section flags the same idea as future work: the world model can learn dynamics in
  one space while the planner optimizes in a projected space.
- The term is absent from the released code. `planning/objectives.py` computes only
  `loss_visual + alpha * loss_proprio` over patch features.
- Mechanistic motivation: `agg` maps 1568 patch dimensions to 128, so a purely patch-space planning cost
  leaves most of the space that the curvature regularizer acted on unconstrained during planning.

Scope resolution, term scaling, run sequencing, and gate-miss behaviour were settled during clarification and
are encoded below as requirements rather than left as open design choices. The two central constraints are
that `planning/*.py`, `datasets/*.py`, and `plan.py` stay byte-identical to base revision `d73b9c6`, and that
the weight `w` is tuned only on a held-out seed that is disjoint from the reporting seeds.

## Glossary

- **Agg_Head**: The aggregation head of `models/dino.py::DinoV2Encoder`, reached through the `agg` method with
  `agg_type: mlp`. For the Target_Cell it maps `196 * 8 = 1568` flattened patch dimensions through
  `1568 -> 512 -> 512 -> 128` with `ReLU` activations, followed by `LayerNorm` over the 128 output dimensions.
- **L_spatial**: The scalar-per-episode planning loss produced by the unmodified
  `planning.objectives.create_objective_fn` for the configured `objective.mode` and `objective.alpha`, namely
  `loss_visual + alpha * loss_proprio` over patch features.
- **L_agg**: The mean squared difference between Agg_Head applied to predicted visual patch features and
  Agg_Head applied to the goal visual patch features, reduced to one scalar per episode.
- **L_plan**: The combined planning loss `L_spatial + Agg_Weight * L_agg`.
- **Agg_Weight**: The configuration value `w` that scales L_agg in L_plan. Its default value is `0`.
- **Agg_Objective_Module**: The new root-level Python module `agg_objectives.py`, which computes L_agg and
  L_plan.
- **Plan_Wrapper**: The new root-level entry script `plan_agg.py`, which runs a planning evaluation by reusing
  `plan.py` without modifying it, and which injects the Agg_Objective_Module objective and the checkpoint's
  encoder into the planner.
- **Frozen_Paths**: Every `*.py` file under `planning/` and `datasets/`, plus root-level `plan.py`.
- **Base_Revision**: Git revision `d73b9c6`, the pre-feature state of this repository, which the Scope_Guard
  measures Frozen_Paths against.
- **Scope_Guard**: The existing test module `tests/test_scope_guard.py`, containing an allowlist assertion and
  a byte-identity assertion over `planning/*.py` and `datasets/*.py`.
- **Scope_Amendment**: A comment block recorded inside the Scope_Guard allowlist that names each newly
  allowlisted file and states why the file is required.
- **Target_Cell**: PushT, encoder configuration `DINOv2 (patch) + proj, 14x14x8`, with `L_curv` enabled;
  evaluated from the existing checkpoint
  `checkpoints/test/pusht_aggmlpcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05` at 2 epochs and 123,858
  training steps.
- **Evaluation_Protocol**: 50 evaluation samples per data seed; open-loop measured through `conf/plan_gd.yaml`
  with `objective.mode=last` and `objective.alpha=1`; MPC measured through `conf/plan_gd_mpc.yaml` with
  `objective.mode=staged` and `objective.alpha=1`; planner hyperparameters fixed at `max_iter 1`,
  `n_taken_actions 25`, `sub_planner.horizon 25`, `sub_planner.lr 0.1`, `sub_planner.sample_type zero`,
  `sub_planner.action_noise 0`, `sub_planner.opt_steps 100`.
- **Tuning_Seed**: Data-sampling seed 400, held out from reporting and used only to select Agg_Weight.
- **Reporting_Seeds**: Data-sampling seeds 100, 200, and 300, used only for the confirmation run and the
  Acceptance_Gate.
- **Sweep_Grid**: The set of Agg_Weight values evaluated on the Tuning_Seed, spanning 0.01 to 3.
- **Baseline_Arm**: A planning evaluation run through the Plan_Wrapper with `Agg_Weight = 0`.
- **Candidate_Arm**: A planning evaluation run through the Plan_Wrapper with the Agg_Weight selected from the
  Sweep_Grid.
- **Platform_Baseline**: The success rates already measured on this pod for the Target_Cell: 75.33 +/- 6.11
  open-loop and 82.00 +/- 2.00 MPC.
- **Paper_Target**: The paper's published Table 1 numbers for the Target_Cell: 77.33 +/- 6.18 open-loop and
  85.33 +/- 4.99 MPC.
- **Acceptance_Gate**: The dual, margin-aware predicate implemented by `ccr_acceptance_gate.py::acceptance_gate`,
  which requires the candidate to beat both Paper_Target and Platform_Baseline on both settings, and treats a
  margin over Platform_Baseline of 6 percentage points or less as inconclusive. For the Target_Cell this
  resolves to thresholds of 81.33 open-loop and 88.00 MPC.
- **Instrumentation_Record**: The recorded magnitudes of L_spatial and L_agg at planner optimizer step 0 and
  step 100, together with the effective ratio `Agg_Weight * L_agg / L_spatial` at each of those two steps.
- **Paired_Comparison**: A per-episode comparison between the Candidate_Arm and the Baseline_Arm at the same
  data seed, exact because training on this pod is bitwise deterministic and `plan.py` seeds episode sampling
  from `cfg.seed`, so both arms draw identical initial states and goals.
- **Job_Launcher**: The existing script invocation `run_ccr_pilot.sh eval <run_dir>`, which applies the pod
  environment recipe and refuses to start while the MIG slice is occupied.
- **Negative_Result_Record**: The written deliverable produced when the Acceptance_Gate does not return
  `pass`, containing the sweep curve, the 3-seed numbers, the Paired_Comparison, and the conclusion.

## Requirements

### Requirement 1: Aggregated-space cost computation

**User Story:** As a researcher, I want the goal distance measured after the encoder's aggregation head, so
that the planner optimizes in the same 128-dimensional space the curvature regularizer shaped.

#### Acceptance Criteria

1. THE Agg_Objective_Module SHALL expose a factory function that returns a planning objective callable with
   the same call signature as the callable returned by `planning.objectives.create_objective_fn`, accepting
   `z_obs_pred`, `z_obs_tgt`, and `step`.
2. THE Agg_Objective_Module SHALL compute L_spatial by delegating to the unmodified
   `planning.objectives.create_objective_fn` for the configured `objective.mode` and `objective.alpha`.
3. THE Agg_Objective_Module SHALL compute L_agg as the mean of the squared elementwise difference between
   Agg_Head applied to the predicted visual patch features and Agg_Head applied to the goal visual patch
   features, reduced over all non-batch dimensions to one scalar per episode.
4. THE Agg_Objective_Module SHALL return `L_plan = L_spatial + Agg_Weight * L_agg` as a tensor of shape
   `(B,)`, matching the shape returned by the unmodified objective.
5. WHERE `objective.mode` is `last`, THE Agg_Objective_Module SHALL compute L_agg from the final predicted
   frame only.
6. WHERE `objective.mode` is `staged`, THE Agg_Objective_Module SHALL apply to L_agg the same stage selection
   and the same per-frame coefficients that `planning.objectives` applies to L_spatial for the same `step`
   value.
7. THE Agg_Objective_Module SHALL evaluate Agg_Head with its parameters held at their checkpoint values and
   with gradients propagating to the predicted latent features.
8. THE Agg_Objective_Module SHALL evaluate Agg_Head on the device and in the dtype of the predicted visual
   patch features.
9. IF the predicted visual patch features have a patch-token count or channel width that Agg_Head does not
   accept, THEN THE Agg_Objective_Module SHALL raise an error naming the received shape and the shape
   Agg_Head requires.
10. THE Agg_Objective_Module SHALL use raw mean-squared L_agg and raw L_spatial without rescaling or
    normalizing either term relative to the other.

### Requirement 2: Wrapper entry script

**User Story:** As a researcher, I want the new cost injected from outside the planning package, so that the
frozen planning code and `plan.py` keep working untouched.

#### Acceptance Criteria

1. THE Plan_Wrapper SHALL run a planning evaluation by calling into `plan.py` as imported, without editing
   `plan.py`.
2. THE Plan_Wrapper SHALL accept the same Hydra configuration names and command-line overrides that `plan.py`
   accepts, including `--config-name`, `ckpt_base_path`, `model_name`, `seed`, `objective.mode`, and
   `objective.alpha`.
3. THE Plan_Wrapper SHALL supply the Agg_Objective_Module objective to the planner in place of the objective
   that `hydra.utils.call(cfg_dict["objective"])` would otherwise produce.
4. THE Plan_Wrapper SHALL obtain Agg_Head from the encoder of the loaded checkpoint model and pass that
   Agg_Head to the Agg_Objective_Module.
5. THE Plan_Wrapper SHALL read Agg_Weight from configuration and SHALL record the resolved Agg_Weight value in
   the run output directory.
6. IF the loaded checkpoint's encoder reports an `agg_type` other than `mlp`, THEN THE Plan_Wrapper SHALL
   terminate with an error naming the encountered `agg_type`.
7. THE Plan_Wrapper SHALL write its results in the same file layout that `plan.py` writes, so that
   `aggregate_results.py` reads Plan_Wrapper output without modification.

### Requirement 3: Config-gated weight with a bitwise-zero default

**User Story:** As a researcher, I want `Agg_Weight = 0` to reproduce current behaviour bitwise, so that the
Baseline_Arm is a valid control rather than an approximation.

#### Acceptance Criteria

1. THE Plan_Wrapper SHALL default Agg_Weight to `0`.
2. WHERE Agg_Weight equals `0`, THE Agg_Objective_Module SHALL return a loss tensor that is bitwise equal to
   the tensor returned by the unmodified `planning.objectives.create_objective_fn` callable for the same
   inputs, mode, and alpha.
3. WHERE Agg_Weight equals `0`, THE Plan_Wrapper SHALL produce per-episode success outcomes that are equal to
   the per-episode success outcomes produced by `plan.py` at the same seed, configuration name, and
   checkpoint.
4. THE Agg_Objective_Module SHALL accept any finite non-negative float Agg_Weight value in the closed interval
   from 0 to 3.
5. IF Agg_Weight is negative or is not a finite number, THEN THE Plan_Wrapper SHALL terminate with an error
   naming the rejected value.

### Requirement 4: Byte-freeze of planning code and scope containment

**User Story:** As a researcher, I want the planning, dataset, and `plan.py` sources frozen, so that the
already-measured Platform_Baseline stays valid without re-measurement.

#### Acceptance Criteria

1. THE Agg_Objective_Module SHALL be a new root-level file `agg_objectives.py`.
2. THE Plan_Wrapper SHALL be a new root-level file `plan_agg.py`.
3. AFTER this feature is implemented, THE Scope_Guard byte-identity assertion over `planning/*.py` and
   `datasets/*.py` against Base_Revision SHALL pass.
4. THE Scope_Guard SHALL report root-level `plan.py` as byte-identical to its content at Base_Revision.
5. THE Scope_Guard allowlist SHALL contain exactly the two new paths `agg_objectives.py` and `plan_agg.py` as
   additions attributable to this feature, alongside the spec documents under `.kiro/specs/` and the tests
   under `tests/`, which the existing allowlist prefixes already cover.
6. THE Scope_Guard SHALL carry a Scope_Amendment stating that `plan.py` builds its objective through
   `hydra.utils.call(cfg_dict["objective"])` and passes no further arguments, and that
   `planning/objectives.py` receives no handle on the world model, so Agg_Head must be injected from outside
   the frozen paths.
7. THE Agg_Objective_Module SHALL import from `planning.objectives` in read-only fashion, calling the existing
   factory and leaving module-level names in `planning.objectives` at their original values.

### Requirement 5: Instrumentation of both loss components

**User Story:** As a researcher, I want both loss components reported, so that a raw unnormalized sum whose
terms live in 1568-dimensional and 128-dimensional spaces can be interpreted rather than guessed at.

#### Acceptance Criteria

1. WHEN the planner reaches optimizer step 0, THE Plan_Wrapper SHALL record the batch-mean magnitude of
   L_spatial and the batch-mean magnitude of L_agg.
2. WHEN the planner reaches optimizer step 100, THE Plan_Wrapper SHALL record the batch-mean magnitude of
   L_spatial and the batch-mean magnitude of L_agg.
3. THE Plan_Wrapper SHALL record the effective ratio `Agg_Weight * L_agg / L_spatial` at optimizer step 0 and
   at optimizer step 100.
4. THE Plan_Wrapper SHALL write the Instrumentation_Record to a machine-readable file in the run output
   directory.
5. WHERE Agg_Weight equals `0`, THE Plan_Wrapper SHALL record the raw L_agg magnitude for reference while
   leaving L_plan equal to L_spatial.
6. IF L_spatial at a recorded step is `0`, THEN THE Plan_Wrapper SHALL record the ratio field as the string
   `undefined` and SHALL record both raw magnitudes.

### Requirement 6: Weight sweep on the held-out tuning seed

**User Story:** As a researcher, I want Agg_Weight selected on a held-out seed, so that the reported result is
not a product of tuning against the acceptance gate.

#### Acceptance Criteria

1. THE Sweep_Grid SHALL contain the Agg_Weight values 0.01, 0.03, 0.1, 0.3, 1, and 3, which span the range
   0.01 to 3 and include the paper-literal value 0.1.
2. THE Plan_Wrapper SHALL evaluate every Sweep_Grid value at the Tuning_Seed in the open-loop setting only.
3. THE Plan_Wrapper SHALL evaluate the Baseline_Arm at the Tuning_Seed in the open-loop setting, giving a
   same-seed reference point for the sweep curve.
4. THE sweep SHALL select as the Candidate_Arm weight the Sweep_Grid value with the highest open-loop success
   rate at the Tuning_Seed.
5. IF two or more Sweep_Grid values tie on the highest open-loop success rate at the Tuning_Seed, THEN THE
   sweep SHALL select the smallest tied Agg_Weight value and SHALL record the tie.
6. THE sweep SHALL use only the Tuning_Seed, so that Reporting_Seeds contribute no information to Agg_Weight
   selection.
7. THE sweep SHALL record, for every Sweep_Grid value, the open-loop success rate and the
   Instrumentation_Record, forming the sweep curve.

### Requirement 7: Three-seed confirmation run

**User Story:** As a researcher, I want a single confirmation run on the reporting seeds, so that the
Acceptance_Gate is evaluated once on data the weight was not chosen against.

#### Acceptance Criteria

1. AFTER the Candidate_Arm weight is selected, THE Plan_Wrapper SHALL evaluate the Candidate_Arm at each of
   the Reporting_Seeds in the open-loop setting and in the MPC setting.
2. THE confirmation run SHALL be executed once for the selected Agg_Weight.
3. THE confirmation run SHALL use `aggregate_results.py` to produce mean and standard deviation over the
   Reporting_Seeds for the open-loop setting and for the MPC setting.
4. THE confirmation run SHALL record per-episode outcomes for the Candidate_Arm and for the Baseline_Arm at
   each Reporting_Seed, so that the Paired_Comparison is computable.
5. WHERE additional Agg_Weight values are considered after the confirmation run, THE confirmation run for the
   originally selected Agg_Weight SHALL remain the reported result, and any further value SHALL be recorded as
   an exploratory follow-up.

### Requirement 8: Evaluation protocol invariance

**User Story:** As a researcher, I want the evaluation protocol held fixed, so that Candidate_Arm numbers are
comparable to Platform_Baseline and Paper_Target.

#### Acceptance Criteria

1. THE Plan_Wrapper SHALL evaluate 50 samples per data seed.
2. WHERE the open-loop setting is measured, THE Plan_Wrapper SHALL use `conf/plan_gd.yaml` with
   `objective.mode=last` and `objective.alpha=1`.
3. WHERE the MPC setting is measured, THE Plan_Wrapper SHALL use `conf/plan_gd_mpc.yaml` with
   `objective.mode=staged` and `objective.alpha=1`.
4. THE Plan_Wrapper SHALL hold the planner hyperparameters at `max_iter 1`, `n_taken_actions 25`,
   `sub_planner.horizon 25`, `sub_planner.lr 0.1`, `sub_planner.sample_type zero`,
   `sub_planner.action_noise 0`, and `sub_planner.opt_steps 100`.
5. THE Plan_Wrapper SHALL load the Target_Cell checkpoint
   `checkpoints/test/pusht_aggmlpcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05` and SHALL leave the
   checkpoint weights unchanged, performing no training.
6. THE Plan_Wrapper SHALL record the resolved values of every Evaluation_Protocol field in the run output
   directory, so that protocol invariance is checkable after the fact.
7. IF any resolved Evaluation_Protocol field differs from the value listed in this requirement, THEN THE
   Plan_Wrapper SHALL terminate with an error naming the field, the expected value, and the resolved value.

### Requirement 9: Execution environment and job sequencing

**User Story:** As a researcher, I want runs launched through the existing pod recipe, so that the single MIG
slice is never contended and results stay reproducible.

#### Acceptance Criteria

1. THE Job_Launcher SHALL be the only mechanism used to start Candidate_Arm, Baseline_Arm, and sweep
   evaluations on the pod.
2. THE Job_Launcher SHALL run at most one evaluation job at a time on the NVIDIA B200 MIG `1g.45gb` slice.
3. WHEN the MIG slice is occupied, THE Job_Launcher SHALL refuse to start a further evaluation job.
4. THE sweep SHALL be budgeted at approximately 40 minutes of wall-clock time in total.
5. THE confirmation run SHALL be budgeted at approximately 1.5 hours of wall-clock time in total.
6. THE Plan_Wrapper SHALL rely on the pod's bitwise-deterministic execution and on `plan.py` seeding episode
   sampling from `cfg.seed`, so that two arms evaluated at the same seed draw identical episodes and the
   Paired_Comparison is exact.

### Requirement 10: Dual acceptance gate

**User Story:** As a researcher, I want the result judged by the existing dual gate, so that a small
within-noise gain is not reported as a success.

#### Acceptance Criteria

1. THE Acceptance_Gate SHALL be evaluated by calling `ccr_acceptance_gate.py::acceptance_gate` with the
   Candidate_Arm means over the Reporting_Seeds and the Platform_Baseline values.
2. THE Acceptance_Gate SHALL require the Candidate_Arm to exceed Paper_Target on both the open-loop setting
   and the MPC setting.
3. THE Acceptance_Gate SHALL require the Candidate_Arm to exceed Platform_Baseline by more than 6 percentage
   points on both the open-loop setting and the MPC setting, which resolves to thresholds of 81.33 open-loop
   and 88.00 MPC.
4. IF exactly one of the Paper_Target condition and the Platform_Baseline condition holds, THEN THE
   Acceptance_Gate SHALL return `fail`.
5. IF both conditions hold and the weaker per-setting margin over Platform_Baseline is 6 percentage points or
   less, THEN THE Acceptance_Gate SHALL return `inconclusive`.
6. THE Acceptance_Gate SHALL report the binomial standard error of approximately 5.7 percentage points at 50
   samples near a success rate of 0.8 alongside every comparison.
7. THE Acceptance_Gate predicate SHALL be reused as it stands, with the Candidate_Arm supplying new inputs
   rather than new thresholds.

### Requirement 11: Stop on gate miss and record the negative result

**User Story:** As a researcher, I want a missed gate to end the investigation with a recorded negative
result, so that no further compute is spent on a term that does not transfer.

#### Acceptance Criteria

1. IF the Acceptance_Gate returns `fail` or `inconclusive`, THEN THE investigation SHALL stop after producing
   the Negative_Result_Record.
2. THE Negative_Result_Record SHALL contain the sweep curve: open-loop success rate at the Tuning_Seed for
   every Sweep_Grid value and for the Baseline_Arm.
3. THE Negative_Result_Record SHALL contain the Candidate_Arm 3-seed open-loop and MPC means and standard
   deviations over the Reporting_Seeds.
4. THE Negative_Result_Record SHALL contain the Paired_Comparison against the Baseline_Arm, reporting for each
   Reporting_Seed the count of episodes the Candidate_Arm solved and the Baseline_Arm did not, the count of
   episodes the Baseline_Arm solved and the Candidate_Arm did not, and the count of episodes with matching
   outcomes.
5. THE Negative_Result_Record SHALL state the conclusion that the paper's long-horizon aggregated term does
   not transfer to the short-horizon Target_Cell.
6. WHEN the Negative_Result_Record is complete, THE investigation SHALL end without launching further
   evaluation or training jobs and without progressing to follow-on work.
7. WHERE follow-on work is desired after a missed gate, THE investigation SHALL require explicit approval
   recorded before any further job is launched.
