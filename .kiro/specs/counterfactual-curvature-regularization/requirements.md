# Requirements Document

## Introduction

This feature adds **Counterfactual Curvature Regularization (CCR)** to the training objective of the
*Temporal Straightening for Latent Planning* world model (arXiv 2603.12231), together with the pilot
infrastructure needed to evaluate it under the project's short-budget-pilot discipline.

The paper's existing curvature regularizer penalises `1 - cos` between consecutive latent velocities of
**dataset (on-log)** trajectories, aggregated through `encoder.agg`. Proposition `app_cos` in
`paper_tex/sec/2_appendix.tex` proves this bounds `(A - I)` only along **visited** velocity directions, and
Remark `app_dir_vs_spec` states that upgrading to the spectral bound of Theorem 1 requires an unproven
coverage condition. At plan time, `GDPlanner` starts from a zero action sequence and takes 100 Adam steps,
so it traverses **off-log** action sequences, i.e. exactly the directions the bound does not cover. CCR
closes that gap by applying the same curvature penalty to *imagined* latent trajectories produced by rolling
the predictor forward from a real encoded state under **perturbed** actions.

A second, cheaper term, **Metric-Consistent Aggregation (MCA)**, is piloted in parallel. MCA encourages
`encoder.agg` to approximately preserve distances between velocity vectors, so that straightness enforced in
aggregated space transfers into the `196 x 8` patch metric that `planning/objectives.py` actually measures.
MCA is explicitly a pilot-only side experiment and is not part of the primary claim.

Prior work grounding the direction:

- Train-plan gap premise: "Closing the Train-Test Gap in World Models for Gradient-Based Planning"
  (arXiv 2512.09929, ICLR 2026).
- Action perturbation inside a learned world model: Dream2Fix (arXiv 2603.13528).
- Straightening as a measure and as a transferable regulariser: Henaff et al., perceptual straightening
  (Nature Neuroscience, 2019); Niu et al., NeurIPS 2024 (arXiv 2411.01777).

Target cell: Table 1 row `DINOv2 (patch) + proj, 14x14x8, L_curv ✓`, environment **PushT**
(run dir `pusht_aggmlpcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05`). Paper values 77.33 open-loop /
85.33 MPC; same-platform B200 reproduction of the identical config ~75.3 / ~82.0
(`AGENT_MEMORY_2.0.md` section 8).

## Glossary

- **CCR_Term**: the Counterfactual Curvature Regularization loss term added inside
  `VWorldModel.forward()` in `models/visual_world_model.py`.
- **MCA_Term**: the Metric-Consistent Aggregation loss term, a pilot-only second loss term added in the
  same location.
- **World_Model**: the `VWorldModel` class, including `encode`, `predict`, `rollout`,
  `replace_actions_from_z`, `total_curvature` and `_cos_curvature`.
- **Baseline_Objective**: the training objective as it exists before this feature, i.e.
  `L_pred + straighten_scale * curvature_loss` with `vcreg` disabled and the decoder stop-gradded.
- **Curvature_Function**: the existing `World_Model.total_curvature(features, mode)` method, used in
  `aggcos` mode.
- **Rollout_Function**: the existing `World_Model.rollout(obs_0, act)` method.
- **rho**: the CCR action-perturbation radius, expressed in units of the action normalizer's standard
  deviation. Because `normalize_action: True` makes training actions unit-variance per dimension, `rho` is
  dimensionless and carries no dataset-specific constant.
- **lambda_cf**: the scalar weight applied to `CCR_Term` in the total loss.
- **Planner_Horizon**: the predictor-step horizon used by the evaluation planner, equal to
  `sub_planner.horizon / frameskip = 25 / 5 = 5`.
- **Training_Pipeline**: `train.py` together with `conf/train.yaml` and the Hydra composition it drives.
- **Run_Naming**: the Hydra `hydra.run.dir` / `hydra.sweep.dir` expression in `conf/train.yaml` that derives
  a run directory (and therefore `model_name`) from configuration values.
- **Telemetry_Logger**: the per-iteration loss-component logging path in `Training_Pipeline` that emits
  `loss_components` to the JSONL training log.
- **Loss_Share**: a loss term's scaled contribution divided by the total loss for the same iteration.
- **Iteration_Cap**: the configuration value `training.max_iterations`, a mid-epoch stopping bound on the
  number of optimizer steps.
- **Offline_Probe**: a read-only analysis tool that loads an existing checkpoint and measures a readout
  without training.
- **Readout**: a measured quantity used to judge a pilot, reported per state dimension
  (agent x, agent y, block x, block y, block angle).
- **Pilot_Run**: a training run bounded by `Iteration_Cap` to 8,000 steps, used to decide whether to spend
  the full budget.
- **Full_Run**: a training run at the paper's full PushT budget of 123,858 optimizer steps (2 epochs).
- **Evaluation_Protocol**: `plan.py` driven by `conf/plan_gd.yaml` / `conf/plan_gd_mpc.yaml` with 50 test
  samples per data seed, data seeds 100/200/300, PushT objectives (open-loop `mode=last, alpha=1`;
  MPC `mode=staged, alpha=1`).
- **Paper_Target**: the Table 1 PushT ✓ success rates, 77.33 open-loop and 85.33 MPC.
- **Platform_Baseline**: the same-platform B200 reproduction of the identical config, ~75.3 open-loop and
  ~82.0 MPC, re-evaluated under **Evaluation_Protocol** for the comparison.
- **Acceptance_Gate**: the dual criterion that a candidate must exceed both **Paper_Target** and
  **Platform_Baseline**.
- **Runtime_Environment**: the Blackwell/MIG environment variable recipe and GPU pre-flight hygiene recorded
  in `AGENT_MEMORY_2.0.md` sections 4 and 5 and `.kiro/skills/ts-repro/SKILL.md`.
- **Protocol_Invariants**: batch size 32, `num_hist` 3, `num_pred` 1, `num_frames` 4, `frameskip` 5,
  PushT 2 epochs, `training.encoder_lr` 1e-5, `training.straighten` `aggcos1e-1`,
  `training.stop_grad` True, `mixed_precision` bf16, `training.seed` 0.

## Requirements

### Requirement 1: CCR loss term

**User Story:** As a researcher, I want a counterfactual curvature penalty on imagined off-log rollouts, so
that latent straightness is enforced along the action directions the gradient-based planner actually
explores.

#### Acceptance Criteria

1. WHERE `lambda_cf` is greater than zero, THE World_Model SHALL add `lambda_cf * CCR_Term` to the training
   loss returned by `forward()`.
2. WHERE `lambda_cf` is greater than zero, THE World_Model SHALL compute CCR_Term by encoding a real
   observation window, rolling the predictor forward with Rollout_Function under an action sequence formed by
   adding a perturbation to the recorded normalized actions, and applying Curvature_Function in `aggcos`
   mode to the resulting imagined latent sequence.
3. WHERE `lambda_cf` is greater than zero, THE World_Model SHALL bound every element of the action
   perturbation to the closed interval `[-rho, +rho]` in normalized-action units.
4. WHERE `lambda_cf` is greater than zero, THE World_Model SHALL roll the imagined trajectory forward for a
   number of predictor steps equal to a configured value whose default equals Planner_Horizon (5).
5. WHERE `lambda_cf` is greater than zero, THE World_Model SHALL compute CCR_Term from the visual-and-proprio
   channels of the imagined latents, excluding the action channels, matching the channel selection used by
   the Baseline_Objective curvature term.
6. THE World_Model SHALL compute CCR_Term without instantiating any new `nn.Module` and without introducing
   any new trainable parameter.
7. THE World_Model SHALL compute CCR_Term using only the existing Curvature_Function, Rollout_Function,
   `replace_actions_from_z`, `predict` and `encode` methods for its latent geometry, adding no second
   curvature definition.
8. THE World_Model SHALL derive the perturbation scale from `rho` and the normalized action space alone,
   using no environment-specific or dataset-specific numeric constant.
9. WHERE `lambda_cf` is greater than zero, THE World_Model SHALL record CCR_Term and its scaled value in the
   `loss_components` dictionary returned by `forward()`.
10. IF `lambda_cf` is greater than zero AND the configured rollout length would require more predictor steps
    than the available action sequence supports, THEN THE World_Model SHALL raise an error naming
    `lambda_cf`, the requested rollout length and the available action length.

### Requirement 2: rho = 0 control arm

**User Story:** As a researcher, I want a built-in control that keeps the rollout-space penalty but removes
the perturbation, so that "rollout space versus encoder space" is separated from "off-log versus on-log".

#### Acceptance Criteria

1. WHERE `lambda_cf` is greater than zero AND `rho` equals zero, THE World_Model SHALL compute CCR_Term on
   imagined rollouts driven by the unperturbed recorded normalized actions.
2. WHERE `rho` equals zero, THE World_Model SHALL apply the same rollout length, channel selection and
   Curvature_Function call used when `rho` is greater than zero, so that the only difference between the arms
   is the perturbation.
3. THE Training_Pipeline SHALL treat `rho` equal to zero as a valid configuration rather than an error.

### Requirement 3: Default-off gating and byte-identical legacy behaviour

**User Story:** As a researcher, I want every new term disabled by default, so that existing checkpoints,
run directory names and reproduced Table 1 numbers stay valid.

#### Acceptance Criteria

1. THE Training_Pipeline SHALL default `lambda_cf` to zero and default the MCA_Term weight to zero in
   `conf/train.yaml`.
2. WHILE `lambda_cf` equals zero AND the MCA_Term weight equals zero, THE World_Model SHALL produce a total
   loss numerically equal to the Baseline_Objective for the same inputs and weights.
3. WHILE `lambda_cf` equals zero AND the MCA_Term weight equals zero, THE World_Model SHALL execute no
   additional predictor rollout call and no additional encoder forward pass beyond those of the
   Baseline_Objective.
4. WHILE every new configuration key holds its default value, THE Run_Naming SHALL produce run directory
   strings byte-identical to those produced before this feature, including
   `pusht_aggmlpcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05`.
5. THE Training_Pipeline SHALL expose `lambda_cf`, `rho`, the CCR rollout length, the MCA_Term weight and
   the Iteration_Cap as Hydra configuration keys, using no hardcoded literal in Python for any of them.
6. WHEN a new term is enabled, THE Training_Pipeline SHALL log a startup line naming the term, its weight,
   its `rho` where applicable, and the device its tensors are computed on.

### Requirement 4: MCA pilot-only term

**User Story:** As a researcher, I want an optional term that makes the aggregation map approximately
distance-preserving on velocity vectors, so that straightness enforced in aggregated space transfers into
the patch metric the planner scores.

#### Acceptance Criteria

1. WHERE the MCA_Term weight is greater than zero, THE World_Model SHALL add the weighted MCA_Term to the
   training loss returned by `forward()`.
2. WHERE the MCA_Term weight is greater than zero, THE World_Model SHALL compute MCA_Term from the
   discrepancy between the norm of a latent velocity measured in the `196 x 8` patch space and the norm of
   the same velocity after passing through `encoder.agg`.
3. THE World_Model SHALL compute MCA_Term without instantiating any new `nn.Module` and without introducing
   any new trainable parameter.
4. WHERE the MCA_Term weight is greater than zero, THE World_Model SHALL record MCA_Term and its scaled
   value in the `loss_components` dictionary returned by `forward()`.
5. THE Acceptance_Gate SHALL be evaluated against a configuration in which the MCA_Term weight equals zero,
   so that the primary claim rests on CCR_Term alone.
6. WHERE the MCA_Term pilot fails its written pass/fail gate, THE researcher SHALL exclude MCA_Term from the
   Full_Run configuration.

### Requirement 5: Protocol invariance

**User Story:** As a researcher, I want the paper's training and evaluation protocol untouched, so that any
success-rate difference is attributable to the new loss term.

#### Acceptance Criteria

1. THE feature SHALL leave every value in Protocol_Invariants unchanged for the Full_Run.
2. THE feature SHALL leave `planning/gd.py`, `planning/cem.py`, `planning/mpc.py`, `planning/objectives.py`
   and `planning/evaluator.py` unmodified.
3. THE feature SHALL leave `conf/plan_gd.yaml` and `conf/plan_gd_mpc.yaml` planner hyperparameters
   unchanged, including `max_iter` 1, `n_taken_actions` 25, `sub_planner.horizon` 25, `lr` 0.1,
   `sample_type` `zero`, `action_noise` 0 and `opt_steps` 100.
4. THE feature SHALL leave `datasets/traj_dset.py`, `datasets/pusht_dset.py` and the data loading path
   unmodified.
5. THE Evaluation_Protocol SHALL use 50 test samples per data seed with data seeds 100, 200 and 300, PushT
   open-loop objective `mode=last, alpha=1` and PushT MPC objective `mode=staged, alpha=1`.
6. THE feature SHALL confine training-side code changes to `models/visual_world_model.py`, `conf/train.yaml`,
   `train.py`, the Run_Naming expression, and new standalone probe or summarisation scripts.
7. THE feature SHALL apply to the PushT `DINOv2 (patch) + proj, 14x14x8, L_curv ✓` cell only, treating other
   Table 1 configurations, other environments and Direction E (action-Gramian conditioning) as out of scope.

### Requirement 6: Pilot infrastructure

**User Story:** As a researcher, I want a mid-epoch iteration cap, isolated run directories and per-term loss
shares, so that an 8,000-step pilot is possible, safe and readable.

#### Acceptance Criteria

1. WHERE Iteration_Cap is set to a positive integer, THE Training_Pipeline SHALL stop training after that
   many optimizer steps, including mid-epoch.
2. WHERE Iteration_Cap is unset or non-positive, THE Training_Pipeline SHALL run for the configured number
   of epochs, matching current behaviour.
3. THE Training_Pipeline SHALL count Iteration_Cap against the checkpointed global iteration counter, so
   that a resumed run honours the same total bound.
4. THE Run_Naming SHALL include `lambda_cf`, `rho` and the MCA_Term weight in the derived run directory
   whenever any of them differs from its default value.
5. WHILE `lambda_cf`, `rho` and the MCA_Term weight all hold their default values, THE Run_Naming SHALL
   contribute an empty string for them, preserving the criterion in Requirement 3.4.
6. IF a launched run resolves to a run directory that already contains a checkpoint produced under a
   different loss configuration, THEN THE Training_Pipeline SHALL abort before writing and report the
   conflicting directory path.
7. THE Telemetry_Logger SHALL write, for each logged iteration, each loss term's scaled value and its
   Loss_Share.
8. THE Telemetry_Logger SHALL write the observed optimizer-step rate in iterations per second, so that a
   pilot's step rate is comparable against a reference run.
9. THE Training_Pipeline SHALL retain `training.save_every_x_iterations` at 1000 so that a checkpoint exists
   at iteration 0 and an empty checkpoint directory one minute into a run is diagnosable as a crash.

### Requirement 7: Offline probe and readout tooling

**User Story:** As a researcher, I want a read-only probe that measures the mechanism on an existing
checkpoint, so that I can confirm the mechanism exists before spending GPU hours on training.

#### Acceptance Criteria

1. THE Offline_Probe SHALL load an existing checkpoint and measure curvature of imagined rollouts under both
   unperturbed and perturbed action sequences, without performing any optimizer step.
2. THE Offline_Probe SHALL report every Readout disaggregated per state dimension, namely agent x, agent y,
   block x, block y and block angle, in addition to any aggregate value.
3. THE Offline_Probe SHALL report a reference value for each Readout, drawn from a pretrained or untrained
   reference model, from the reference run's own early telemetry, or from a matched-budget control run.
4. THE Offline_Probe SHALL leave every loaded checkpoint file unmodified.
5. IF a requested checkpoint path does not exist, THEN THE Offline_Probe SHALL report the missing path and
   exit without loading a model.
6. THE Offline_Probe SHALL run on CPU within 30 minutes for the target checkpoint.

### Requirement 8: Pilot discipline

**User Story:** As a researcher, I want the project's short-budget-pilot rules enforced as explicit steps, so
that pilot conclusions are not rationalised after the fact.

#### Acceptance Criteria

1. WHILE a pass/fail gate for a Pilot_Run has not been written down, THE researcher SHALL NOT launch that
   Pilot_Run.
2. THE researcher SHALL choose each pilot Readout to be causally upstream of success rate, to have a named
   reference value, and to move within the pilot's 8,000-step budget.
3. THE researcher SHALL read Loss_Share values rather than raw loss values when judging whether a new term
   dominates the objective.
4. WHEN a Pilot_Run completes, THE researcher SHALL compare its step rate and its step-200 loss values
   against the reference run before interpreting any other result.
5. THE researcher SHALL treat mid-run representation Readouts as catastrophic-failure detectors only, and
   SHALL NOT interpret them as trends.
6. IF a Pilot_Run's new term falls to approximately zero within the first 1,000 iterations, THEN THE
   researcher SHALL record the term as having absorbed the task without pressuring the encoder, and SHALL
   NOT report the Pilot_Run as a success.
7. THE researcher SHALL evaluate a single data seed for triage and all three data seeds before reporting any
   success-rate difference as real.
8. WHEN a Pilot_Run finishes, THE researcher SHALL record its outcome together with its caveats in the
   project progress log.

### Requirement 9: Runtime environment and GPU hygiene

**User Story:** As a researcher, I want the Blackwell/MIG recipe applied before every run, so that runs fail
for scientific reasons rather than environmental ones.

#### Acceptance Criteria

1. WHEN a training or evaluation run is launched, THE Runtime_Environment SHALL set
   `PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync`.
2. WHEN a training or evaluation run is launched, THE Runtime_Environment SHALL leave `CUDA_VISIBLE_DEVICES`
   unset.
3. WHEN a training or evaluation run is launched, THE Runtime_Environment SHALL set `OMP_NUM_THREADS`,
   `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `NUMEXPR_NUM_THREADS` to 8.
4. WHEN an evaluation run is launched, THE Runtime_Environment SHALL set `PLAN_SERIAL_ENV=1`.
5. WHEN a run is launched, THE researcher SHALL first list running python processes with `ps` and terminate
   stray or stopped python processes, because the `1g.45gb` MIG slice holds exactly one job.
6. THE researcher SHALL determine GPU memory holders from `ps` output rather than from the `nvidia-smi`
   process table, which does not enumerate processes on a MIG slice.
7. THE researcher SHALL run training, pilots and evaluations serially, one job at a time.

### Requirement 10: Acceptance gate

**User Story:** As a researcher, I want a dual, pre-declared success bar with an explicit noise floor, so
that a reported improvement is credible.

#### Acceptance Criteria

1. THE Acceptance_Gate SHALL require the candidate open-loop success rate to exceed 77.33 and the candidate
   MPC success rate to exceed 85.33.
2. THE Acceptance_Gate SHALL additionally require the candidate open-loop success rate to exceed the
   Platform_Baseline open-loop rate and the candidate MPC success rate to exceed the Platform_Baseline MPC
   rate, both measured under Evaluation_Protocol.
3. THE Acceptance_Gate SHALL evaluate both the candidate and the Platform_Baseline under an unmodified
   Evaluation_Protocol.
4. THE researcher SHALL report the binomial standard error at 50 samples near p equal to 0.8 as
   approximately 5.7 percentage points alongside every success-rate comparison.
5. WHERE the candidate's margin over Platform_Baseline is 6 percentage points or less across the three data
   seeds, THE researcher SHALL either report the result as inconclusive or extend the evidence with
   additional training seeds.
6. IF only one of the two Acceptance_Gate conditions holds, THEN THE researcher SHALL report the outcome as a
   gate failure.

### Requirement 11: Research process governance

**User Story:** As a researcher, I want approval and citation discipline encoded, so that GPU time is only
spent on directions with a stated prior-work basis and explicit sign-off.

#### Acceptance Criteria

1. WHILE no research direction is approved, THE researcher SHALL NOT modify training or evaluation code.
2. THE researcher SHALL cite at least one supporting prior-work reference for a candidate direction before
   implementing that direction.
3. THE researcher SHALL follow the escalation ladder of Offline_Probe, then Pilot_Run, then Full_Run, and
   SHALL NOT launch a Full_Run before a Pilot_Run for the same configuration has passed its written gate.
4. THE implementation SHALL generalise to other Table 1 cells and other environments without code change,
   requiring only configuration overrides.
5. THE researcher SHALL record the planned compute allocation of approximately 23 GPU-hours, comprising a
   30-minute Offline_Probe, three Pilot_Runs of roughly 75 to 85 minutes sweeping `rho` and `lambda_cf`, a
   20-minute single-seed triage evaluation, a Full_Run of roughly 17 hours, and a three-seed evaluation of
   roughly 1.5 hours.
6. WHERE additional training seeds are required by Requirement 10.5, THE researcher SHALL request approval
   for the approximately 26 additional GPU-hours before launching them.
7. WHEN a change increases measured step time by more than 50 percent relative to the reference run, THE
   researcher SHALL report the regression and revise the compute plan before launching the Full_Run.
