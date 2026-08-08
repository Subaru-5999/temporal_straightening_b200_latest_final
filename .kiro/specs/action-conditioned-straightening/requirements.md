# Requirements Document

## Introduction

Action-Conditioned Straightening (ACS) is a training-time modification of the paper's curvature
regularizer in `models/visual_world_model.py`. The paper penalizes `1 - cos(v_t, v_{t+1})` uniformly
over every latent-velocity triple; ACS replaces that uniform mean with a gate-weighted mean over the
**same** per-triple values, where the gate measures how similar the controlling actions were:

```
L_acs = Σ_t w_t · c_t / Σ_t w_t,   c_t = 1 - cos(v_t, v_{t+1}),   w_t = relu(cos(a_t, a_{t+1})).detach()
```

The curvature values live in the paper's 128-d aggregated space, computed by the same shared code the
baseline uses; the space, `λ = 0.1`, and the static-velocity mask are unchanged.

These requirements are derived from `design.md`, which is authoritative. Every numeric threshold below
appears in the design and is reproduced **verbatim**; none is invented, relaxed, or rounded here. The
thresholds are pre-registered — written before the data — because a rule invented after the data gets
fitted to it, which is the documented CCR failure mode.

The work is staged. Stage 0 is a CPU-only premise test that costs zero GPU-hours and **can kill the
feature**. Stage 1 is an 8,000-step arm read against a bitwise-matched control. Stage 2 is a full run
plus 3-seed evaluation, reachable only through the early-read gate.

## Glossary

- **ACS**: Action-Conditioned Straightening, the feature specified by this document.
- **ACS_Project**: the human-plus-agent process that decides which stages are executed and which
  artifacts are written. Requirements addressed to the ACS_Project are process obligations.
- **World_Model**: `VWorldModel` in `models/visual_world_model.py`.
- **ACS_Term**: `VWorldModel.compute_acs`, which returns the scalar `L_acs` and a telemetry dict.
- **Gate**: `VWorldModel.action_gate`, which returns the detached per-triple weights `w`.
- **Action_Reducer**: `VWorldModel.reduce_action`, which maps `act (b, t, f·d)` to the per-latent-step
  action vector used by the Gate.
- **Straighten_Parser**: the branch in `VWorldModel.__init__` that maps `training.straighten` to
  `curvature_mode` and `straighten_scale`.
- **Geometry_Helpers**: `VWorldModel._agg_velocities` and `VWorldModel._cos_curvature_terms`, the
  shared implementations of the aggregated velocities and the per-triple cosine.
- **Triple**: one curvature triple `(z_t, z_{t+1}, z_{t+2})`, contributing one `c` value and one `w`
  value. At the target cell there are 2 per sample and 64 per batch.
- **Unmasked_Triple**: a Triple that survives the existing `step_thresh = 1e-6` velocity-norm mask.
- **Reversing_Triple**: an Unmasked_Triple whose consecutive reduced action pair has `cos <= 0`.
- **Telemetry_Writer**: the telemetry path in `train.py` that writes `training_log.jsonl`, including
  `TELEMETRY_TERMS`, `TELEMETRY_ACS_KEY` and `_acs_telemetry_block`.
- **Stage0_Probe**: `probe_ccr_curvature.py --readout actions`.
- **GateSplit_Probe**: `probe_ccr_curvature.py --readout gatesplit`.
- **State_Probe**: `probe_ccr_curvature.py --readout state`, reused unchanged.
- **Log_Summarizer**: `summarize_training_log.py`, including the new `--prediction-gate`,
  `--prediction-gate-direction` and `--acs-gate-check` flags.
- **Acceptance_Gate_Tool**: `ccr_acceptance_gate.py`, reused unchanged.
- **Scope_Guard**: `tests/test_scope_guard.py`.
- **Run_Directory_Resolver**: `custom_resolvers.acs_tag` together with the `hydra.run.dir` and
  `hydra.sweep.dir` interpolations in `conf/train.yaml`.
- **Control_8k**: the bitwise reproduction of the baseline's first 8,000 steps in
  `checkpoints_ctrl8k`, whose 40 telemetry rows agree with the baseline to `+0.000000`.
- **Arm_Run**: an 8,000-step ACS run launched with `training.straighten=acsaggcos1e-1`.
- **Baseline_Run**: the existing full `aggcos1e-1` run (`model_2.pth`, 123,858 steps, measured
  75.33 ± 6.11 open-loop / 82.00 ± 2.00 MPC).
- **R**: the Stage-0 reallocation statistic `E|w − E[w]| / (2 · E[w])`, the population form of the
  total-variation distance between the normalized weight vector and uniform.
- **acs_gate_tv**: the finite-batch form of **R**, logged during training.
- **Early_Read_Gate**: checks 0, 1, 1b, 1c, 2a, 2b and 3 of `design.md` §13.
- **Progress_Record**: `PROGRESS_ACS.md`.
- **MCA_Fallback**: `VWorldModel.compute_mca`, already written and never run, the named fallback arm.

## Requirements

### Requirement 1: Stage-0 Action-Similarity Measurement

**User Story:** As a researcher, I want the consecutive-action-similarity distribution measured on all
four datasets before any loss code is written, so that the premise of ACS is tested for zero
GPU-hours instead of assumed.

#### Acceptance Criteria

1. THE Stage0_Probe SHALL compute `cos(a_t, a_{t+1})` over the `train` split of `pusht`, `wall`,
   `point_maze` and `point_maze_medium`.
2. THE Stage0_Probe SHALL report the mean of `cos(a_t, a_{t+1})` per environment per action reduction.
3. THE Stage0_Probe SHALL report the median of `cos(a_t, a_{t+1})` per environment per action reduction.
4. THE Stage0_Probe SHALL report `frac(cos < 0)` per environment per action reduction.
5. THE Stage0_Probe SHALL report `frac(cos < 0.5)` per environment per action reduction.
6. THE Stage0_Probe SHALL report a 20-bin histogram of `cos(a_t, a_{t+1})` over `[-1, 1]` per
   environment per action reduction.
7. THE Stage0_Probe SHALL report `mean(w)` for `w = relu(cos)` per environment per action reduction.
8. THE Stage0_Probe SHALL report `frac(w = 0)` per environment per action reduction.
9. THE Stage0_Probe SHALL report `R = E|w − E[w]| / (2 · E[w])` per environment per action reduction.
10. THE Stage0_Probe SHALL report `n_triples` and `n_windows` alongside every statistic in criteria
    1.2 through 1.9.
11. THE Stage0_Probe SHALL measure all three action reductions `sum`, `raw` and `first` in a single
    invocation.
12. THE Stage0_Probe SHALL report the `validation` split statistics as a cross-check of the `train`
    split statistics.
13. THE Stage0_Probe SHALL compose each environment configuration from `conf/train.yaml` with
    `env=<name>`, `num_hist=3`, `num_pred=1` and `frameskip=5`.
14. THE Stage0_Probe SHALL read actions through a dedicated action-only loader that reads the
    underlying dataset action tensor together with `dset.slices`, `dset.frameskip` and
    `dset.num_frames` and applies the same `rearrange("(n f) d -> n (f d)")`.
15. THE Stage0_Probe SHALL leave `probe_ccr_curvature.load_windows` and its `state_dim` guard
    unchanged.
16. THE Stage0_Probe SHALL verify on 32 randomly selected windows that the action tensor produced by
    the action-only loader is bitwise equal to `dset[idx][1]`.
17. THE Stage0_Probe SHALL complete without allocating a GPU and without decoding video.
18. THE Stage0_Probe SHALL write its statistics to a machine-readable JSON report per environment and
    a combined verdict report.

### Requirement 2: Stage-0 Pre-Registered Verdict Rules

**User Story:** As a researcher, I want the Stage-0 verdict rule written down before the data is
collected and mechanically evaluated afterwards, so that a GO cannot be manufactured by fitting the
rule to the numbers.

#### Acceptance Criteria

1. THE ACS_Project SHALL record the complete text of rules A and B, including every threshold in this
   requirement, in the repository before the Stage-0 statistics are collected.
2. WHERE PushT's `frac(cos < 0)` is the highest of the four environments AND exceeds each of the Wall,
   UMaze and Medium values by at least 1.5x AND UMaze's value is the lowest of the four, THE
   Stage0_Probe SHALL report the rule A verdict `GO`.
3. WHERE PushT's `frac(cos < 0)` is the highest of the four environments AND the remaining ordering
   inverts, THE Stage0_Probe SHALL report the rule A verdict `MIDDLE`.
4. WHERE PushT's `frac(cos < 0)` is the highest of the four environments AND its margin over the
   largest of the other three values is at least 1.1x and less than 1.5x, THE Stage0_Probe SHALL
   report the rule A verdict `MIDDLE`.
5. WHERE the rule A verdict is `MIDDLE`, THE ACS_Project SHALL record that the mechanism claim is
   downgraded to "the gate is a useful inductive bias" and that the claim of explaining the Table 1
   gain ordering is withheld.
6. WHERE PushT's `frac(cos < 0)` is not the highest of the four environments, THE Stage0_Probe SHALL
   report the rule A verdict `STOP`.
7. WHERE PushT's `frac(cos < 0)` is within 1.1x of the smoothest environment's value, THE
   Stage0_Probe SHALL report the rule A verdict `STOP`.
8. WHERE PushT's `R >= 0.15`, THE Stage0_Probe SHALL report the rule B verdict `GO`.
9. WHERE PushT's `R` is at least 0.08 and less than 0.15, THE Stage0_Probe SHALL report the rule B
   verdict `MIDDLE`.
10. WHERE the rule B verdict is `MIDDLE`, THE ACS_Project SHALL record `acs_gate=hard` or a sharpened
    gate as the pre-declared remedy and SHALL record that the expected effect size is small.
11. WHERE PushT's `R < 0.08`, THE Stage0_Probe SHALL report the rule B verdict `STOP`.
12. IF the rule A verdict is `STOP`, THEN THE ACS_Project SHALL NOT implement the ACS_Term, the Gate,
    the Action_Reducer or any other ACS code path.
13. IF the rule B verdict is `STOP`, THEN THE ACS_Project SHALL NOT implement the ACS_Term, the Gate,
    the Action_Reducer or any other ACS code path.
14. THE ACS_Project SHALL proceed to Stage 1 only when the rule A verdict is `GO` or `MIDDLE` AND the
    rule B verdict is `GO` or `MIDDLE`.
15. IF either rule returns `STOP`, THEN THE ACS_Project SHALL select MCA_Fallback as the next arm.
16. IF either rule returns `STOP`, THEN THE ACS_Project SHALL write up the Stage-0 statistics as
    finding N1 in the Progress_Record.
17. THE ACS_Project SHALL record that the rule A and rule B thresholds are judgment calls rather than
    derivations.

### Requirement 3: Recorded Limitations of the Premise Test and the Gate

**User Story:** As a reviewer, I want each known weakness of the ACS argument recorded next to the
conclusion it limits, so that a positive result cannot be read as stronger than the evidence supports.

#### Acceptance Criteria

1. THE ACS_Project SHALL record that the Stage-0 correlation has `n = 4` points with no independent
   replicates and therefore can refute but cannot establish the mechanism.
2. THE ACS_Project SHALL record that the four environments carry differently-typed action variables —
   PushT relative pusher displacements, PointMaze forces or velocity commands on a point mass, Wall
   dot velocities — so `cos(a_t, a_{t+1})` is not the same physical quantity across the four points.
3. THE ACS_Project SHALL record that a confirmed Stage-0 ordering is consistent with confounds other
   than the ACS mechanism, naming contact dynamics, the second movable object, rotational state, and
   2 training epochs on PushT against 20 elsewhere.
4. THE ACS_Project SHALL record that a Stage-0 `GO` is treated as permission to spend 0.8 GPU-hours
   rather than as evidence for the mechanism.
5. THE ACS_Project SHALL record that the `frameskip=5` reduction may wash out reversals occurring
   inside a single latent step.
6. IF the `sum` reduction shows no reversal structure while the `raw` reduction does, THEN THE
   ACS_Project SHALL record a rule A verdict of `MIDDLE` rather than `GO`.
7. THE ACS_Project SHALL record that at `num_hist=3, num_pred=1` there are only 2 Triples per sample,
   so zeroing Reversing_Triples raises curvature-gradient variance, and that no Early_Read_Gate check
   measures that cost directly.
8. THE ACS_Project SHALL record that the Gate measures whether the *controlled object* reversed
   direction rather than whether the latent velocity's direction change is action-explained.
9. THE ACS_Project SHALL state the limitations named in this requirement in the same paragraph as any
   conclusion they limit.

### Requirement 4: The ACS Loss Term

**User Story:** As a researcher, I want the gated curvature term computed as a weighted mean over the
same per-triple values the baseline averages, so that a win is attributable to reallocation and cannot
be restated as a smaller `λ`.

#### Acceptance Criteria

1. THE ACS_Term SHALL return `Σ w · c / clamp_min(Σ w, 1e-3)`, with both sums taken over the
   Unmasked_Triples of the batch.
2. THE ACS_Term SHALL obtain `c_t = 1 - cos(v_t, v_{t+1})` from the Geometry_Helpers in the 128-d
   aggregated space produced by `encoder.agg` over `visual_only(z)`.
3. THE ACS_Term SHALL normalize over all Unmasked_Triples of the batch rather than per sample.
4. THE ACS_Term SHALL clamp the denominator at the hardcoded `WEIGHT_SUM_FLOOR = 1e-3`.
5. THE ACS_Term SHALL select `c` and `w` with the same velocity-norm mask before forming the
   numerator and the denominator.
6. WHERE `w` is constant at any value greater than zero, THE ACS_Term SHALL return a value equal to
   `total_curvature(visual_only(z), "aggcos")` within fp32 tolerance.
7. THE ACS_Term SHALL return a finite scalar in `[0, 2]` for all finite `z` and `act`.
8. WHERE `z` is scaled by any `α > 0`, THE ACS_Term SHALL return a value equal to the unscaled result
   within fp32 tolerance.
9. WHERE the batch axis of `z` and `act` is permuted, THE ACS_Term SHALL return a value equal to the
   unpermuted result within fp32 tolerance.
10. IF every Unmasked_Triple in the batch is a Reversing_Triple, THEN THE ACS_Term SHALL return
    exactly 0 with a defined gradient.
11. THE ACS_Term SHALL satisfy `sign(∂L_acs/∂w_t) = sign(c_t − L_acs)` for every Unmasked_Triple `t`.
12. THE ACS_Term SHALL be scaled by `straighten_scale = 0.1`, selected by the mode string
    `acsaggcos1e-1`, with no calibration ladder and no share target.
13. WHILE `curvature_mode == "acsaggcos"`, THE World_Model SHALL add exactly one curvature
    contribution to the loss, namely the scaled ACS_Term.
14. THE ACS_Term SHALL leave `z` and `act` unmutated.
15. THE ACS_Term SHALL read `EPS = 1e-6`, `STEP_THRESH = 1e-6` and `WEIGHT_SUM_FLOOR = 1e-3` from
    hardcoded constants rather than from configuration.
16. THE ACS_Term SHALL compute its value without any additional encoder pass and without any
    additional predictor call.

### Requirement 5: The Action Gate and the Action Reduction

**User Story:** As a researcher, I want the gate computed from raw recorded actions and detached, so
that the only descent direction the term offers is the trajectory geometry.

#### Acceptance Criteria

1. THE Gate SHALL compute `w_t = relu(cos(a_t, a_{t+1}))` when `acs_gate == "relu_cos"`.
2. THE Gate SHALL compute `w` from the raw `act` tensor of the batch rather than from the output of
   `action_encoder`.
3. THE Gate SHALL return a tensor whose `requires_grad` is `False` and whose `grad_fn` is `None`.
4. THE Gate SHALL return values satisfying `0 <= w <= 1` elementwise for every member of the
   `acs_gate` enum.
5. WHEN the two reduced action vectors of a Triple are positively parallel, THE Gate SHALL return
   `w = 1` for that Triple.
6. WHERE `acs_gate == "relu_cos"`, THE Gate SHALL return `w = 0` for every Triple whose reduced
   action pair has `cos <= 0`.
7. THE Gate SHALL return a tensor of shape `(b, t-2)`, matching `c` and the mask elementwise.
8. WHERE the reduced action block of a Triple has zero norm, THE Gate SHALL return `w = 0` for that
   Triple.
9. WHERE `act` is scaled by any `α > 0`, THE Gate SHALL return values equal to the unscaled result
   within fp32 tolerance.
10. THE Action_Reducer SHALL accept `acs_action_reduce` values from the closed enum
    `{sum, raw, first}` with default `sum`.
11. THE World_Model SHALL accept `acs_gate` values from the closed enum
    `{relu_cos, affine_cos, hard, permuted}` with default `relu_cos`.
12. WHERE `acs_action_reduce == "sum"`, THE Action_Reducer SHALL return
    `out[..., j] = Σ_s act[..., s·d + j]`, the net commanded displacement over the latent step.
13. WHERE `acs_action_reduce == "first"`, THE Action_Reducer SHALL return `act[..., :d]`.
14. WHERE `acs_action_reduce == "raw"`, THE Action_Reducer SHALL return `act` itself.
15. THE Action_Reducer SHALL resolve the substep count from `act.shape[-1]` and the environment
    action dimension of the batch rather than from a configuration constant.
16. THE Action_Reducer SHALL leave `act` unmutated for `sum` and `first`.
17. THE World_Model SHALL expose no continuous gate exponent, threshold or sharpness constant.

### Requirement 6: Configuration Surface, Mode-String Parsing, Run Naming and Resume

**User Story:** As a researcher launching an arm, I want the ACS arm selected by a mode string that
gets its own run directory and loss signature, so that it cannot silently resume or overwrite the
baseline.

#### Acceptance Criteria

1. WHEN `training.straighten` matches the prefix `acsaggcos`, THE Straighten_Parser SHALL set
   `curvature_mode = "acsaggcos"` and read `straighten_scale` from the remaining suffix.
2. IF `training.straighten` is a non-empty string matching none of `acsaggcos`, `aggcos` or `cos`,
   THEN THE Straighten_Parser SHALL raise a `ValueError` naming the accepted forms `False`,
   `cos<scale>`, `aggcos<scale>` and `acsaggcos<scale>`.
3. IF the suffix of an `acsaggcos` mode string is non-numeric, THEN THE Straighten_Parser SHALL raise
   a `ValueError` during `__init__`.
4. IF the parsed `straighten_scale` is less than or equal to zero, THEN THE Straighten_Parser SHALL
   raise a `ValueError` during `__init__`.
5. IF `acs_action_reduce` is outside its enum, THEN THE World_Model SHALL raise a `ValueError` during
   `__init__` even when `training.straighten == "aggcos1e-1"`.
6. IF `acs_gate` is outside its enum, THEN THE World_Model SHALL raise a `ValueError` during
   `__init__` even when `training.straighten == "aggcos1e-1"`.
7. THE `conf/train.yaml` file SHALL define `training.acs_action_reduce: sum` and
   `training.acs_gate: relu_cos`.
8. THE Trainer SHALL forward `acs_action_reduce` and `acs_gate` into the World_Model constructor via
   `self.cfg.training.get(key)`, so that an absent key arrives as `None` and selects the default.
9. THE Run_Directory_Resolver SHALL append `${acs_tag:${training.acs_action_reduce},${training.acs_gate}}`
   after `${ccr_tag:...}` in both `hydra.run.dir` and `hydra.sweep.dir`.
10. WHERE `acs_action_reduce` and `acs_gate` are at their defaults, THE Run_Directory_Resolver SHALL
    resolve `acs_tag` to the empty string.
11. WHERE all knobs are at their defaults and `training.straighten == "aggcos1e-1"`, THE
    Run_Directory_Resolver SHALL produce the byte-identical legacy directory name
    `pusht_aggmlpcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05`.
12. WHEN `training.straighten == "acsaggcos1e-1"`, THE Run_Directory_Resolver SHALL produce a
    directory name distinct from the baseline's.
13. THE Run_Directory_Resolver SHALL leave `ccr_tag`'s arity, defaults and output unchanged for every
    input.
14. THE Trainer SHALL add `"acs_action_reduce"` and `"acs_gate"` to `LOSS_SIGNATURE_KEYS` and their
    defaults `"sum"` and `"relu_cos"` to `LOSS_SIGNATURE_DEFAULTS`.
15. WHEN a `loss_config.json` written before the `acs_*` keys existed is compared against a
    default launch, THE Trainer SHALL treat the missing keys as their defaults and report the
    signatures as equal.
16. THE ACS_Project SHALL verify that relaunching into an Arm_Run directory resumes rather than
    restarting or raising, before any 12-hour run is started.
17. THE `tests/test_run_naming.py` module SHALL strip both the `ccr_tag` and the `acs_tag`
    interpolations to recover the pre-feature template and SHALL assert that the pair is appended at
    the end.

### Requirement 7: Default-Off Bitwise Reproduction of the Baseline

**User Story:** As a researcher, I want the disabled path to be the unmodified path, so that the
measured 75.33 / 82.00 baseline stands without a retrain.

#### Acceptance Criteria

1. WHERE `training.straighten == "aggcos1e-1"`, THE World_Model SHALL produce a `loss` bitwise equal
   to the pre-feature reference implementation on the same inputs.
2. WHERE `training.straighten == "aggcos1e-1"`, THE World_Model SHALL produce every
   `loss_components` value bitwise equal to the pre-feature reference implementation.
3. WHERE `curvature_mode != "acsaggcos"`, THE World_Model SHALL emit no `loss_components` key with
   the prefix `acs_`.
4. WHERE `curvature_mode != "acsaggcos"`, THE World_Model SHALL emit no
   `curvature_loss_unweighted` key.
5. THE World_Model SHALL compute `_cos_curvature` and `total_curvature` through the Geometry_Helpers
   with the same operations, order and dtypes as the pre-feature code, producing bitwise identical
   results.
6. WHERE `curvature_mode != "acsaggcos"`, THE World_Model SHALL execute no ACS tensor operation.
7. THE World_Model SHALL create no new module, no new parameter and no new buffer in `__init__` for
   ACS.
8. THE World_Model SHALL store every ACS configuration value as a plain Python scalar, string or
   bool.
9. THE default configuration `training.straighten: False` SHALL leave the loss and every
   `loss_components` value bitwise equal to the pre-feature reference implementation.

### Requirement 8: Telemetry

**User Story:** As a researcher reading the Early_Read_Gate, I want the ACS arm's curvature row
comparable to the control's and the gate statistics logged per step, so that the gate verdict is read
from numbers rather than eyeballed.

#### Acceptance Criteria

1. WHILE `curvature_mode == "acsaggcos"`, THE World_Model SHALL report the ACS_Term value under the
   existing key `curvature_loss_used_for_training`.
2. WHILE `curvature_mode == "acsaggcos"`, THE World_Model SHALL report the ACS_Term value times
   `straighten_scale` under the existing key `curvature_loss_scaled`.
3. WHILE `curvature_mode == "acsaggcos"`, THE World_Model SHALL report `c[mask].mean()` under the key
   `curvature_loss_unweighted`, detached and never added to the loss.
4. THE `curvature_loss_unweighted` value SHALL be bitwise equal to
   `total_curvature(visual_only(z), "aggcos")` on the same tensor.
5. THE Telemetry_Writer SHALL exclude `curvature_loss_unweighted` from `TELEMETRY_TERMS`.
6. THE Telemetry_Writer SHALL produce telemetry records whose `terms[*].share` values sum to 1.0
   within 0.01 under ACS.
7. THE Telemetry_Writer SHALL leave `TELEMETRY_TERMS` unchanged.
8. WHILE `curvature_mode == "acsaggcos"`, THE World_Model SHALL report `acs_gate_mean` as
   `w[mask].mean()`.
9. WHILE `curvature_mode == "acsaggcos"`, THE World_Model SHALL report `acs_gate_tv` as
   `0.5 · Σ |w/Σw − 1/N|` over the Unmasked_Triples.
10. WHILE `curvature_mode == "acsaggcos"`, THE World_Model SHALL report `acs_gate_zero_frac` as the
    fraction of Unmasked_Triples with `w == 0`.
11. WHILE `curvature_mode == "acsaggcos"`, THE World_Model SHALL report `acs_gate_p10`,
    `acs_gate_p50` and `acs_gate_p90`.
12. WHILE `curvature_mode == "acsaggcos"`, THE World_Model SHALL report `acs_denom_clamped_frac` as
    the fraction of steps at which `WEIGHT_SUM_FLOOR` bound the denominator.
13. WHILE `curvature_mode == "acsaggcos"`, THE World_Model SHALL report `acs_masked_frac` as the
    fraction of Triples dropped by the velocity-norm mask.
14. THE Telemetry_Writer SHALL emit an `acs` block containing `enabled`, `gate_mean`, `gate_tv`,
    `gate_zero_frac`, `gate_p10`, `gate_p50`, `gate_p90`, `denom_clamped_frac`, `masked_frac`,
    `curvature_unweighted`, `action_reduce` and `gate`.
15. THE Telemetry_Writer SHALL derive the `acs` block's `enabled` field from the presence of
    `acs_gate_mean` in `loss_components` rather than from the configuration.
16. IF the `acs` block's derived `enabled` disagrees with the configuration, THEN THE
    Telemetry_Writer SHALL log a warning.
17. WHERE the ACS path did not run, THE Telemetry_Writer SHALL omit the `acs` block.
18. THE ACS_Term SHALL return telemetry values that are all detached scalars.

### Requirement 9: Error Handling and Eager Validation

**User Story:** As a researcher, I want every misconfiguration to raise and name itself before the
first training step, so that no 12-hour run produces a silently wrong objective.

#### Acceptance Criteria

1. IF `acs_action_reduce` or `acs_gate` is outside its enum, THEN THE World_Model SHALL raise a
   `ValueError` naming the offending key and the accepted values (E1).
2. IF `training.straighten` is a non-empty string matching no known prefix, THEN THE World_Model
   SHALL raise a `ValueError` rather than disabling straightening and continuing to train (E2).
3. IF `curvature_mode == "acsaggcos"` and the encoder has no `agg` attribute, THEN THE ACS_Term SHALL
   raise a `ValueError` shaped like `total_curvature`'s existing `aggcos` check (E4).
4. IF `act.shape[-1]` is not divisible by the environment action dimension under `sum` or `first`,
   THEN THE Action_Reducer SHALL raise a `ValueError` naming both numbers (E5).
5. IF `z.shape[1] < 3`, THEN THE ACS_Term SHALL raise a `ValueError` naming the frame count and the
   requirement (E6).
6. IF `z.shape[1] != act.shape[1]`, THEN THE ACS_Term SHALL raise a `ValueError` naming both values
   (E7).
7. IF every Unmasked_Triple in a batch is a Reversing_Triple, THEN THE ACS_Term SHALL report
   `acs_denom_clamped_frac = 1.0` for that step and continue training (E8).
8. IF `acs_denom_clamped_frac` is sustained above zero across steady-state rows, THEN THE
   ACS_Project SHALL record it as a dataset finding and as an Early_Read_Gate red flag.
9. IF every Triple in a batch is masked as static, THEN THE ACS_Term SHALL return 0 and report
   `acs_masked_frac = 1.0` (E9).
10. IF a reduced action block has zero norm, THEN THE Gate SHALL rely on `cosine_similarity`'s `eps`
    and return `w = 0` without raising (E10).
11. IF a non-finite ACS_Term value is observed, THEN THE ACS_Project SHALL record it as an upstream
    numerics defect rather than an ACS defect, because every denominator is clamped or `eps`-guarded
    (E11).
12. IF an Arm_Run's run directory collides with a run of a different loss signature, THEN
    `_guard_run_dir` SHALL raise before any artifact is written, naming the differing signature keys
    (E12).
13. IF a telemetry write fails, THEN THE Telemetry_Writer SHALL warn once, disable telemetry and
    continue training (E13).
14. THE Stage0_Probe SHALL avoid `load_windows`'s PushT `state_dim` guard by using its own
    action-only loader (E14).
15. THE Stage0_Probe and GateSplit_Probe SHALL reuse the existing `_warm_dino_hub` and
    `_plain_tensor_attrs_to_cpu` helpers for DINOv2 hub resolution and the cuda-pinned mask in
    `models/vit.py` (E15).

### Requirement 10: Early-Read Gate — Checks 0, 1, 1b and 1c

**User Story:** As a researcher, I want the 8,000-step arm judged against the bitwise control by
pre-registered numeric rules that a tool evaluates, so that the verdict is not a matter of discretion.

#### Acceptance Criteria

1. THE ACS_Project SHALL record the complete text of checks 0, 1, 1b, 1c, 2a, 2b and 3, including
   every threshold in this requirement and Requirement 11, before the Arm_Run is launched.
2. THE Log_Summarizer SHALL read `it_per_s` from steady-state telemetry rows past row 400 and SHALL
   compare it against the reference `2.862 it/s`.
3. IF steady-state `it_per_s` is less than `2.72`, THEN THE ACS_Project SHALL treat the result as an
   implementation defect, fix the code, and hold the arm rather than accepting the cost.
4. THE Log_Summarizer SHALL provide `--prediction-gate REFERENCE_RUN_DIR` and
   `--prediction-gate-direction {improve,guard}` with default `guard`.
5. THE ACS_Project SHALL run check 1 with `--prediction-gate-direction improve`, reading the
   prediction term as a positive directional prediction rather than as a guard.
6. WHERE the Arm_Run's scaled `prediction` at `global_iter` 8000 is at most `0.013196` AND at least
   15 of the last 20 matched rows are better than the control's, THE Log_Summarizer SHALL report
   check 1 as `GO`.
7. WHERE the Arm_Run's scaled `prediction` at `global_iter` 8000 is at most `0.012536`, THE
   Log_Summarizer SHALL additionally report check 1 as `STRONG GO` and SHALL record that verdict
   separately.
8. IF the Arm_Run's scaled `prediction` at `global_iter` 8000 exceeds `0.014516`, THEN THE
   Log_Summarizer SHALL report check 1 as `STOP`.
9. IF at least 15 of the last 20 matched rows are worse than the control's, THEN THE Log_Summarizer
   SHALL report check 1 as `STOP`.
10. WHERE neither the `GO` nor the `STOP` condition of check 1 holds, THE Log_Summarizer SHALL report
    check 1 as `MIDDLE`, decided by checks 1b, 1c and 2 with no discretion.
11. THE Log_Summarizer SHALL read every check-1 quantity at matched `global_iter` against
    Control_8k's own rows.
12. THE Log_Summarizer SHALL report the Arm_Run's curvature share at `global_iter` 200 and at
    `global_iter` 8000 against the band `[65%, 80%]`, whose control value is `73.741%`.
13. IF the curvature share falls outside `[65%, 80%]`, THEN THE ACS_Project SHALL investigate before
    treating the arm as believable.
14. THE Log_Summarizer SHALL report the Arm_Run's prediction share against the floor `11.75%`.
15. THE Log_Summarizer SHALL report, via `--collapse-check`, that no term falls below the collapse
    threshold within the first 1,000 iterations.
16. THE Log_Summarizer SHALL report `curvature_loss_used_for_training` at `global_iter` 200 and at
    `global_iter` 8000 together with their ratio.
17. THE Log_Summarizer SHALL provide `--acs-gate-check`, which reads the `acs` telemetry block and
    prints `gate_mean`, `gate_tv`, `gate_zero_frac` and `denom_clamped_frac` against the Stage-0
    population estimate.
18. THE Log_Summarizer SHALL report `acs_gate_tv >= 0.08` as a pass condition of check 1c.
19. THE Log_Summarizer SHALL report whether `acs_gate_tv` lies within a factor of 1.5 of the Stage-0
    `R` estimate for PushT.
20. THE Log_Summarizer SHALL report `acs_denom_clamped_frac < 0.01` as a pass condition of check 1c.
21. THE Log_Summarizer SHALL report `acs_gate_mean`, `acs_gate_p10`, `acs_gate_p50` and
    `acs_gate_p90` for comparison against the Stage-0 distribution.
22. THE Log_Summarizer SHALL report `acs_gate_zero_frac` against Stage-0's `frac(cos < 0)`.
23. THE Log_Summarizer SHALL report `acs_masked_frac`.
24. IF `acs_gate_tv` is approximately zero, THEN THE ACS_Project SHALL report check 1c as `STOP` on
    the grounds that the term is the baseline and nothing is attributable to it.
25. IF `acs_gate_tv` differs from the Stage-0 `R` estimate by more than a factor of 1.5, THEN THE
    ACS_Project SHALL record a suspected wiring defect in the substep reduction or the
    triple-to-action-pair alignment.
26. THE ACS_Project SHALL compare the Arm_Run's `curvature_loss_unweighted` against Control_8k's
    curvature at `global_iter` 200 within `rtol = 0.05`, using the unweighted quantity rather than
    the weighted one.

### Requirement 11: Early-Read Gate — Checks 2a, 2b and 3

**User Story:** As a researcher, I want a held-out measurement that distinguishes reallocated pressure
from a changed average, so that a loss that fell for the wrong reason cannot pass the gate.

#### Acceptance Criteria

1. THE GateSplit_Probe SHALL measure held-out per-triple curvature at `--num-windows 192` for the arm
   checkpoint and the control checkpoint with identical flags and seed.
2. THE GateSplit_Probe SHALL split held-out Triples into a `w = 0` bucket and a `w >= 0.5` bucket.
3. THE GateSplit_Probe SHALL compare **unweighted** per-triple curvature between the arm and the
   control within each bucket.
4. THE GateSplit_Probe SHALL report whether the arm's `w = 0` bucket curvature is higher than the
   control's, which is the pre-registered ACS prediction for that bucket.
5. THE GateSplit_Probe SHALL report whether the arm's `w >= 0.5` bucket curvature is equal to or
   lower than the control's, which is the pre-registered ACS prediction for that bucket.
6. THE GateSplit_Probe SHALL report the overall unweighted mean without a pre-registered direction.
7. IF both directional rows of check 2a fail, THEN THE ACS_Project SHALL report check 2a as `STOP` on
   the grounds that no reallocation is measurable on held-out data.
8. THE GateSplit_Probe SHALL compute the aggregated latents and per-triple cosines through the
   existing `_aggregate_latent` helper and the Geometry_Helpers.
9. THE State_Probe SHALL measure `block_angle` readout R² at `--num-windows 192` for the arm and the
   control, using `state_readout_r2` unchanged.
10. THE ACS_Project SHALL treat check 2b as passed when `block_angle` R² does not degrade beyond
    noise, and SHALL record an improvement as supporting evidence that is not required for `GO`.
11. THE ACS_Project SHALL evaluate check 3 on the 8,000-step checkpoints with 1 seed under the
    unmodified evaluation protocol in both open-loop and MPC settings.
12. THE ACS_Project SHALL read check 3 against the measured Control_8k values `16.0` open-loop and
    `18.0` MPC.
13. IF the check 3 difference is at most `-10` points in either setting, THEN THE ACS_Project SHALL
    record a red flag worth acting on.
14. WHERE the check 3 difference lies within `±10` points, THE ACS_Project SHALL record that the
    result carries no information and SHALL report it as neither support nor refutation.
15. THE ACS_Project SHALL record that check 3 is a catastrophe detector only, with a per-arm binomial
    standard error of about 5.2 points at `p ≈ 0.17`.

### Requirement 12: Acceptance Gate and Protocol Invariants for a Full Run

**User Story:** As a researcher, I want the full-run acceptance bar and the evaluation protocol fixed
in advance, so that a positive result is measured under the paper's settings and against a
pre-declared number.

#### Acceptance Criteria

1. THE ACS_Project SHALL launch Stage 2 only for an arm that cleared the Early_Read_Gate.
2. THE Acceptance_Gate_Tool SHALL require an open-loop success rate of at least `79.33`.
3. THE Acceptance_Gate_Tool SHALL require an MPC success rate of at least `87.00`.
4. THE Acceptance_Gate_Tool SHALL require both settings to clear their bars for a pass.
5. THE Acceptance_Gate_Tool SHALL evaluate the mean over the 3 data-sampling seeds `100`, `200` and
   `300`.
6. THE Acceptance_Gate_Tool SHALL evaluate at `n_evals=50`.
7. THE ACS_Project SHALL report the per-seed success rates alongside the seed mean.
8. THE ACS_Project SHALL train the Arm_Run with encoder lr `1e-5`, 2 epochs on PushT, batch 32,
   `num_hist=3`, `num_pred=1`, `frameskip=5`, bf16 and `stop_grad=True`.
9. THE ACS_Project SHALL keep `λ = 0.1` unchanged in every ACS arm.
10. THE ACS_Project SHALL evaluate open-loop with the GD planner at `objective.mode=last`,
    `alpha=1`, `max_iter=1` and `n_taken_actions=25`.
11. THE ACS_Project SHALL evaluate MPC with the GD planner at `objective.mode=staged`, `alpha=1`,
    `max_iter=20` and `n_taken_actions=5`.
12. THE ACS_Project SHALL configure the sub-planner with horizon 25, lr 0.1, `sample_type=zero`,
    `action_noise=0` and `opt_steps=100`.
13. THE ACS_Project SHALL run PushT before any other environment.
14. WHERE a positive PushT result is obtained and another environment is attempted, THE ACS_Project
    SHALL restrict the claim for that environment to the open-loop setting.
15. THE ACS_Project SHALL record that `+4` open-loop on a 3-seed mean is roughly 1.3 standard errors
    and that the single-checkpoint per-seed spread `74 / 82 / 70` is the noise reality.

### Requirement 13: Control Arms

**User Story:** As a researcher, I want the λ-reduction objection answered at zero cost and the
attribution objection answered by a permuted-gate arm, so that a win can be attributed to action
conditioning rather than to reweighting in general.

#### Acceptance Criteria

1. THE ACS_Project SHALL use the existing Baseline_Run as the λ-matched plain-`L_curv` control at
   `λ = 0.1` without training a new control arm.
2. THE ACS_Project SHALL cite the weighted mean's invariance to uniform gate rescaling as the reason
   the λ-reduction confound is absent by construction rather than merely controlled.
3. THE ACS_Project SHALL read every Early_Read_Gate comparison against Control_8k, whose 8,000-step
   prefix reproduces the baseline bitwise.
4. THE World_Model SHALL implement the permuted-gate arm as the `acs_gate=permuted` enum member,
   which permutes `w` across the batch's Unmasked_Triples before the weighted mean.
5. THE `permuted` gate SHALL preserve `mean(w)`, the full weight distribution and `acs_gate_tv`
   exactly, changing only the correspondence between a weight and its own Triple.
6. THE ACS_Project SHALL launch the permuted-gate arm only after the Arm_Run clears check 1 with a
   confirmed directional prediction.
7. THE ACS_Project SHALL require the permuted-gate arm's `acs_gate_mean` and `acs_gate_tv` to match
   the Arm_Run's within batch noise, and SHALL treat a mismatch as evidence that the permutation is
   not behaving as specified.
8. THE ACS_Project SHALL record that no arm controls for PushT-specific effects and that a
   single-environment result remains a single-environment result.

### Requirement 14: Scope, Frozen Sources and Exclusions

**User Story:** As a maintainer, I want the change confined to an audited additive file list with the
planning, dataset and encoder sources frozen, so that the reproduction of the paper cannot be
perturbed by this feature.

#### Acceptance Criteria

1. THE ACS_Project SHALL restrict changes to `models/visual_world_model.py`, `train.py`,
   `conf/train.yaml`, `custom_resolvers.py`, `probe_ccr_curvature.py`,
   `summarize_training_log.py`, `run_ccr_pilot.sh`, `tests/*` and the new `PROGRESS_ACS.md`.
2. THE Scope_Guard SHALL assert that every file under `planning/` with a `.py` extension hashes equal
   to the base revision.
3. THE Scope_Guard SHALL assert that every file under `datasets/` with a `.py` extension hashes equal
   to the base revision.
4. THE Scope_Guard SHALL assert that `plan.py` hashes equal to the base revision.
5. THE Scope_Guard SHALL assert that `models/vit.py` and `models/dino.py` hash equal to the base
   revision.
6. THE Scope_Guard SHALL assert that every changed path is in its allowlist, which gains
   `PROGRESS_ACS.md`.
7. THE ACS_Project SHALL add `PROGRESS_ACS.md` as the only new non-test file.
8. THE ACS_Project SHALL keep CCR disabled in every arm with `lambda_cf=0` and `ccr_rho=0`.
9. THE ACS_Project SHALL exclude the on-hold TMR arm, its share ladder and its patch-space question
   from this feature.
10. THE ACS_Project SHALL leave `plan_agg.py` and `agg_objectives.py` untouched and unimported.
11. THE ACS_Term SHALL compute curvature in the paper's aggregated space, leaving the straightening
    space unchanged.
12. THE ACS_Project SHALL introduce no new runtime dependency.
13. THE ACS_Project SHALL configure each property-based test with a minimum of 100 examples.
14. THE ACS_Project SHALL keep the existing test suite passing after every ACS change.

### Requirement 15: One Implementation of the Gate, Shared by Probe and Training

**User Story:** As a researcher, I want Stage 0 and training to compute the gate with the same code,
so that the Stage-0 prediction and the training-time measurement are the same number and the CCR
calibration error cannot recur.

#### Acceptance Criteria

1. THE Stage0_Probe SHALL compute `a_t` by calling `VWorldModel.reduce_action`.
2. THE Stage0_Probe SHALL compute `w_t` by calling `VWorldModel.action_gate`.
3. THE test suite SHALL assert that no independent cosine-of-actions computation exists in
   `probe_ccr_curvature.py`.
4. THE ACS_Term and the Stage0_Probe SHALL compute `acs_gate_tv` and `R` as finite-batch and
   population forms of the same quantity.
5. WHERE a future variant adopts the plain-sum reduction, THE ACS_Project SHALL obtain the
   calibration constant by calling the shipped `compute_acs` on `model_2.pth` and the unmodified
   validation loader, and SHALL resolve the weight against the measured step-8000 total
   `B = 0.056171` as `X = σ/(1−σ) · B`.

### Requirement 16: Negative-Result Record

**User Story:** As a researcher, I want every prediction, measurement, verdict and error recorded in
`PROGRESS_ACS.md` as it happens, so that the paper-facing findings survive whatever the outcome is.

#### Acceptance Criteria

1. THE ACS_Project SHALL create the Progress_Record at Stage 0.
2. THE ACS_Project SHALL update the Progress_Record at every decision point.
3. THE ACS_Project SHALL write each prediction into the Progress_Record before the corresponding
   measurement is taken.
4. THE ACS_Project SHALL record which Stage-0 rule fired and its exact numbers before the Arm_Run is
   launched.
5. WHERE the Stage-0 verdict is `MIDDLE`, THE ACS_Project SHALL record the downgraded mechanism claim
   at the moment the verdict is read rather than retroactively.
6. THE ACS_Project SHALL record the training-time `acs_gate_tv` against the Stage-0 `R` estimate,
   including when the Arm_Run succeeds.
7. THE ACS_Project SHALL record whether the check-1b scale-preservation prediction held.
8. THE ACS_Project SHALL record whether the check-1 directional prediction on the prediction-loss
   channel held.
9. THE ACS_Project SHALL record every error made, including errors that cost only minutes.
10. THE ACS_Project SHALL record the novelty positioning as written before the outcome, with a date.
11. THE ACS_Project SHALL record finding N1: the per-environment action-similarity distributions set
    against Table 1's straightening gains `+50.00 / +10.67 / +10.67 / +7.33`.
12. THE ACS_Project SHALL record finding N2: the reallocation statistic `R` and `frac(w = 0)` per
    environment.
13. THE ACS_Project SHALL record finding N3: the measured direction of the `block_angle` R² change
    against the control.
14. THE ACS_Project SHALL record the curvature-share drift across multiple iterations rather than a
    single row.
15. THE ACS_Project SHALL record per-seed evaluation values rather than seed means alone.
16. THE ACS_Project SHALL state the limits of each conclusion in the same paragraph as the
    conclusion, including that 8,000 steps is 6.5% of the budget, that both arms sit near the
    success-rate floor, that the matched-budget test is structurally biased against any new term, and
    that one seed does not establish generalization.
