# Implementation Plan: Action-Conditioned Straightening (ACS)

## Overview

Python 3.10 / PyTorch / Hydra, matching the rest of the repo. The design specifies Python signatures for
every interface, so there is no language choice to make.

The ordering is unusual for this spec, and the reason is Requirement 2.12/2.13: **a Stage-0 STOP means the
ACS term is never implemented.** The premise test costs zero GPU-hours and runs before the loss exists, so
everything that would be wasted by a STOP is scheduled after it.

1. **Scope guard first** (section 1). `PROGRESS_ACS.md` joins `ALLOWED_FILES` and the frozen-source
   assertions are extended, before any feature code exists, so the guard never reports a violation this
   feature did not cause.
2. **Shared helpers and the parser fix** (section 2). The `_agg_velocities` / `_cos_curvature_terms`
   extraction is bitwise-neutral (Requirement 7.5) and the `straighten` parser gains its `else: raise`
   (Requirement 9.2), closing a live landmine where a typo trains 12 h with no curvature term while logging
   `"Straightening disabled"`. Both Stage 0 and the term depend on this section.
3. **`reduce_action` and `action_gate`** (section 3). Stage 0 must call the *shipped* implementations
   (Requirements 15.1, 15.2), so they exist before Stage 0 runs. That is the structural fix for the CCR
   calibration error, applied to the gate.
4. **Stage 0: the premise test** (sections 4-5). CPU-only, zero GPU-hours, and a **hard gate**. Everything
   below is conditional on its verdict.
5. **Then** the ACS term (6), the config/run-dir/telemetry surface (8) and the Stage-1 gate tooling (9).
6. **Then** the GPU stages (11-13), strictly serial: the `1g.45gb` MIG slice holds exactly one job.

Task labels:

- **[CODE]** — an agent can write and run this locally on CPU. No GPU, no dataset, no checkpoint, no
  network.
- **[CPU RUN]** — an operator runs a measurement that needs the dataset or a checkpoint but no GPU. Stage 0
  is this.
- **[GPU RUN]** — an operator launches a pod job. Not agent-executable.
- **[HUMAN]** — a judgement, verdict or approval. Not agent-executable.

Wall-clock budget, recorded honestly: **Stage 0 minutes / 0 GPU-h**; Stage 1 arm **0.8 GPU-h**; matched 8k
eval 0.4 GPU-h; permuted-gate arm 0.8 GPU-h; Stage 2 full run + 3-seed eval 13.6 GPU-h. **Best case
(Stage-0 STOP) 0 GPU-h; typical case ~0.8-1.2 GPU-h; worst case ~16 GPU-h.**

## Tasks

- [x] 1. Scope guard and test scaffolding
  - [x] 1.1 [CODE] Extend `tests/test_scope_guard.py` for the ACS file set
    - Add `PROGRESS_ACS.md` to `ALLOWED_FILES` — the only new non-test file this feature adds
    - Keep the frozen-source assertions: every `planning/*.py`, every `datasets/*.py`, root-level `plan.py`,
      and extend `FROZEN_FILES` with `models/vit.py` and `models/dino.py` (Requirement 14.5) so ACS cannot
      touch the encoder or predictor sources
    - **Record the tension rather than hiding it:** `models/vit.py` is already in `ALLOWED_FILES` under the
      CCR SDPA scope amendment, so a naive base-revision hash comparison fails on a change ACS did not
      cause. Freeze it against its *current* content with a comment naming the CCR amendment and
      `tests/test_vit_sdpa_equivalence.py` as its guard, exactly as the module already does for
      `PREEXISTING_FILES`. `models/dino.py` freezes against the base revision
    - **Gate, deliberately not optional:** scope containment is a stated requirement and this is its only
      automated check. It lands before any feature code
    - _Requirements: 14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 14.7_

  - [x] 1.2 [CODE] Add the ACS test doubles and hypothesis strategies to `tests/conftest.py`
    - **Additive only.** Every existing CCR and aggregated-space fixture keeps its name, its signature and
      its values: `build_stub_world_model`, `make_stub_batch`, `ccr_windows`, `target_cell_model`,
      `target_cell_batch`, `make_agg_head`, `stub_agg_encoder` and the rest are untouched. The model factory
      already carries new ctor kwargs through its `**extra_model_kwargs` passthrough
    - New shared strategies: latents `(b, t, p, d)` float32 with `b ∈ [1, 6]`, `t ∈ [3, 6]`, small `p, d`,
      values bounded away from overflow; actions `(b, t, f·d_env)` generated jointly with `t` so the frame
      axes always agree; `acs_action_reduce ∈ {sum, raw, first}`; `acs_gate ∈ {relu_cos, affine_cos, hard,
      permuted}`; a log-uniform positive scalar strategy over `[1e-3, 1e3]` for the rescaling properties
    - New explicit degenerate cases, each its own named strategy or fixture so a property can request it
      directly: all-equal frames (every velocity below `step_thresh`), exactly one non-static sample among
      static ones, `b = 1`, all-antiparallel actions, all-parallel actions, zero-norm action blocks
    - Every ACS property test depends on this file, so it lands in the first wave
    - _Requirements: 14.13, 14.14_

  - [x] 1.3 [CODE] Freeze the pre-feature curvature path in `tests/reference_impl.py`
    - Append verbatim copies of the current `_cos_curvature` and `total_curvature` (both `cos` and `aggcos`
      branches) and the current `forward` straightening tail, with a comment recording the commit SHA they
      were copied from
    - **This must exist before the section-2 refactor**, otherwise Property 1 and Property 12 have nothing
      to compare against. Same ordering rationale as the CCR plan's `_rollout_latents` extraction
    - Additive: the existing frozen `forward` loss tail and `rollout` loop keep their names, since
      `tests/test_rollout_refactor.py` and `tests/test_agg_zero_bitwise.py` read them
    - _Requirements: 7.1, 7.2, 7.5_

- [x] 2. Shared geometry helpers and the `straighten` parser landmine
  - [x] 2.1 [CODE] Extract `_agg_velocities` and `_cos_curvature_terms` in `models/visual_world_model.py`
    - `_agg_velocities(features) -> (v1, v2)`, each `(b, t-2, agg_dim)`, via `encoder.agg` over
      `visual_only(z)`; `_cos_curvature_terms(v1, v2, eps=1e-6, step_thresh=1e-6) -> (loss (b,t-2), mask
      (b,t-2))`
    - `_cos_curvature` becomes `loss, mask = self._cos_curvature_terms(v1, v2); return loss[mask].mean()`,
      and `total_curvature(mode="aggcos")` routes through `_agg_velocities`. Same operations, same order,
      same dtypes — **bitwise identical** on the baseline path
    - Pure refactor: no ACS code in this task. One implementation of the aggregated velocities and one of
      the per-triple cosine is the structural guard against CCR's two-implementations-drifting failure
    - _Requirements: 7.5, 4.2, 14.11_

  - [x] 2.2 [CODE] Add the `acsaggcos` parser branch, the `else: raise`, and the two ctor kwargs
    - `acs_action_reduce: str = "sum"` and `acs_gate: str = "relu_cos"`, stored as plain Python strings — no
      module, no parameter, no buffer, because `VWorldModel` is built after `accelerator.prepare()` and is
      never itself prepared
    - Parser order `acsaggcos` → `aggcos` → `cos`, then **`else: raise ValueError`** naming the accepted
      forms `False`, `cos<scale>`, `aggcos<scale>`, `acsaggcos<scale>`. Today this falls through and trains a
      full run with no curvature term at all; a non-numeric suffix and a `scale <= 0` raise here too
    - Validate both enums eagerly in `__init__` **even when `straighten="aggcos1e-1"`**, so a typo in an
      unused knob cannot survive until the run that enables it. String comparisons only: the off path gains
      no tensor work
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 7.7, 7.8, 9.1, 9.2, 5.11, 5.17_

  - [x] 2.3 [CODE] Write property test for the disabled path (`tests/test_acs_off_bitwise.py`)
    - **Property 1: The disabled path is bitwise the baseline**
    - **Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5, 7.7, 7.9**
    - With `straighten="aggcos1e-1"` and with the default `straighten=False`, `loss` and every
      `loss_components` value are **bitwise** equal to `tests/reference_impl.py`; no key with prefix `acs_`
      and no `curvature_loss_unweighted` key exists. Also asserts the section-2.1 refactor left
      `_cos_curvature` and `total_curvature` bitwise, and that no new module, parameter or buffer appeared
    - **Gate, deliberately not optional:** this is what lets the measured 75.33 OL / 82.00 MPC baseline stand
      without a 12 h retrain

  - [x] 2.4 [CODE] Write property test for eager validation (`tests/test_acs_validation.py`)
    - **Property 13: Enum and mode-string validation is eager**
    - **Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 9.1, 9.2, 9.4**
    - Every invalid `acs_action_reduce`, every invalid `acs_gate`, every non-empty `straighten` string
      matching no known prefix (`acsagcos1e-1` is the motivating typo), a non-numeric suffix and a
      non-positive scale raise in `__init__` **while ACS is not selected**
    - **Gate, deliberately not optional:** it closes the silent-disable hole, which is the difference between
      a `ValueError` at second zero and a 12-hour null run

- [x] 3. The action reduction and the gate
  - [x] 3.1 [CODE] Implement `reduce_action` in `models/visual_world_model.py`
    - `sum`: `out[..., j] = Σ_s act[..., s·d + j]`, the net commanded displacement over the latent step;
      `first`: `act[..., :d]`; `raw`: `act` itself, documented as an identity so no caller assumes a copy
    - Resolve the substep count from `act.shape[-1]` and the batch's env action dim, never from a config
      constant; raise a `ValueError` naming both numbers when it does not divide (E5). Reshaping silently
      would corrupt every gate value
    - No mutation of `act` for `sum` or `first`
    - _Requirements: 5.10, 5.12, 5.13, 5.14, 5.15, 5.16, 9.4_

  - [x] 3.2 [CODE] Implement `action_gate` with the four-member enum dispatch
    - `a = reduce_action(act)`, `cos_a = cosine_similarity(a[:, :-2], a[:, 1:-1], dim=-1)`, then
      `relu_cos` → `relu(cos)`, `affine_cos` → `(1 + cos)/2`, `hard` → `1[cos > 0]`, `permuted` → `relu(cos)`
      permuted across the batch's unmasked triples (Requirement 13.4)
    - Computed from the **raw `act` tensor**, never from `action_encoder(act)`, and `.detach()`ed anyway as
      an executable contract. Returns `(b, t-2)`, matching `c` and the mask elementwise, values in `[0, 1]`
    - Zero-norm action blocks fall out as `w = 0` through `cosine_similarity`'s `eps` — no raise (E10)
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 5.11, 9.10, 13.4_

  - [x]* 3.3 [CODE] Write property test for gate range and the parallel identity (`tests/test_acs_gate_range.py`)
    - **Property 3: Gate range and parallel-action identity**
    - **Validates: Requirements 5.1, 5.4, 5.5, 5.6, 5.7, 13.4**
    - All four gate modes: `0 <= w <= 1` elementwise, `w = 1` for positively parallel reduced actions, and
      for `relu_cos` exactly `w = 0` on the whole `cos <= 0` half-space

  - [x] 3.4 [CODE] Write property test for gate detachment (`tests/test_acs_gate_detached.py`)
    - **Property 4: The gate carries no gradient**
    - **Validates: Requirements 5.2, 5.3, 8.18, 13.4**
    - `w.requires_grad is False` and `w.grad_fn is None`; the gradient w.r.t. `z` equals the gradient of the
      same expression with `w` substituted by its numeric values, so neither the encoder nor the trained
      `action_encoder` can lower the loss by driving `w → 0`
    - **Gate, deliberately not optional:** an attached gate re-introduces the λ-reduction confound the whole
      design is built to eliminate, invisibly and adaptively

  - [x]* 3.5 [CODE] Write property test for rescaling invariance (`tests/test_acs_gate_rescaling.py`)
    - **Property 5: Gate invariance to positive rescaling, and sum ≡ mean**
    - **Validates: Requirements 5.9, 5.12, 5.13, 5.16**
    - `α` log-uniform over `[1e-3, 1e3]`; also asserts `cos(sum(u), sum(v)) == cos(mean(u), mean(v))`, which
      is why `mean` is not offered as a fourth reduction

- [x] 4. Stage-0 probe, verdict rules, and the pre-registered record
  - [x] 4.1 [CODE] Add `--readout actions` and the action-only loader to `probe_ccr_curvature.py`
    - Introduce a `--readout` selector whose default reproduces today's behaviour exactly (both existing
      readouts), then add `actions`. `load_windows` and its PushT `state_dim` guard are **left unchanged** —
      Wall's `state_dim` is 4, so reusing it raises before measuring anything (E14)
    - The action-only loader composes the env config from `conf/train.yaml` with `env=<name>`, `num_hist=3`,
      `num_pred=1`, `frameskip=5`, then reads the underlying dataset's action tensor plus `dset.slices`,
      `dset.frameskip` and `dset.num_frames` and applies the same `rearrange("(n f) d -> n (f d)")`.
      **It must not go through `dset[idx]`**: that routes PushT through `PushTDataset.get_frames`, opens a
      `VideoReader` and decodes 20 frames per window, turning minutes into hours
    - `a_t` and `w_t` come from the **shipped** `VWorldModel.reduce_action` / `action_gate` — no second
      implementation of the gate anywhere in the repository
    - Per env, per reduction (`--acs-action-reduce all` measures `sum`, `raw` and `first` in one invocation):
      mean, median, `frac(cos<0)`, `frac(cos<0.5)`, a 20-bin histogram over `[-1, 1]`, `mean(w)`,
      `frac(w=0)`, `R = E|w − E[w]| / (2·E[w])`, and `n_triples` / `n_windows` beside every statistic.
      `--split train` is the headline, `validation` reported as a cross-check
    - No GPU allocation, no video decode; reuse `_warm_dino_hub` and `_plain_tensor_attrs_to_cpu`; write one
      machine-readable JSON per env into `probe_outputs/`
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 1.10, 1.11, 1.12, 1.13, 1.14, 1.15, 1.17, 1.18, 9.14, 9.15, 15.1, 15.2_

  - [x] 4.2 [CODE] Write property test for one gate implementation (`tests/test_acs_single_gate_impl.py`)
    - **Property 19: One implementation of the gate, shared by probe and training**
    - **Validates: Requirements 1.16, 15.1, 15.2, 15.3**
    - Assert `probe_ccr_curvature.py` contains no independent cosine-of-actions computation (source scan),
      and that on 32 randomly selected windows the action-only loader's tensor is **bitwise** equal to
      `dset[idx][1]` — so skipping the video decode cannot silently change what is measured
    - **Gate, deliberately not optional:** this is the structural fix for the CCR calibration error applied
      to the gate. The 32-window bitwise check needs the real dataset on disk; skip cleanly when
      `DATASET_DIR` is unset so the CPU suite stays green, and run it as part of task 5.1

  - [x] 4.3 [CODE] Implement the Stage-0 verdict rules in `probe_ccr_curvature.py --readout actions --summarize`
    - `--summarize <reports...> --table1-gains "umaze=50.00,medium=10.67,wall=10.67,pusht=7.33"` reads the
      per-env reports and emits one combined verdict JSON plus a printed verdict
    - **Rule A (mechanism ordering).** `GO` when PushT's `frac(cos<0)` is the highest of the four AND exceeds
      each of Wall / UMaze / Medium by `>= 1.5x` AND UMaze is the lowest. `MIDDLE` when PushT is highest but
      the remaining ordering inverts, or its margin over the largest of the other three is in `[1.1x, 1.5x)`.
      `STOP` when PushT is not the highest, or is within `1.1x` of the smoothest
    - **Rule B (reallocation).** `GO` at `R >= 0.15`; `MIDDLE` at `0.08 <= R < 0.15`; `STOP` at `R < 0.08`
    - Both rules evaluated independently; the combined verdict is `STOP` if either is `STOP`, and Stage 1 is
      permitted only when both are `GO` or `MIDDLE`. Emit the `sum`-vs-`raw` comparison flagged by
      Requirement 3.6 (no reversal structure under `sum` but structure under `raw` ⟹ `MIDDLE`, not `GO`)
    - _Requirements: 2.2, 2.3, 2.4, 2.6, 2.7, 2.8, 2.9, 2.11, 2.14, 3.6, 1.9, 1.18_

  - [x]* 4.4 [CODE] Write unit tests for the Stage-0 verdict rules (`tests/test_acs_stage0_verdict.py`)
    - The design gives the model-layer properties P1-P19 but no coverage of the **decision-rule layer**, and
      the Stage-0 verdict is the function that can kill the feature. Tests, on synthetic statistic dicts:
    - **Totality**: every input over the space of four `frac(cos<0)` values and a PushT `R` maps to exactly
      one of `GO` / `MIDDLE` / `STOP` per rule — no gap, no overlap, no unhandled combination
    - **Exact boundaries**: `1.5x` and `1.1x` on rule A, `0.15` and `0.08` on rule B, each tested at, just
      below and just above the value, so an inclusive/exclusive slip is caught
    - **Combination**: either `STOP` ⟹ combined `STOP`; both at least `MIDDLE` ⟹ Stage 1 permitted
    - _Requirements: 2.2, 2.3, 2.4, 2.6, 2.7, 2.8, 2.9, 2.11, 2.14_

  - [x] 4.5 [CODE] Create `PROGRESS_ACS.md` with everything that must be written before the data
    - Rules A and B in full, including every threshold, **before** the Stage-0 statistics are collected, with
      the explicit statement that the thresholds are judgment calls rather than derivations
    - Checks 0, 1, 1b, 1c, 2a, 2b and 3 in full with every threshold (`2.72 it/s`; `0.013196` / `0.012536` /
      `0.014516`; 15-of-20; curvature share `[65%, 80%]` against the control's `73.741%`; prediction share
      floor `11.75%`; `acs_gate_tv >= 0.08`; `acs_denom_clamped_frac < 0.01`; `±10` points on check 3), plus
      the acceptance bars `79.33` OL / `87.00` MPC
    - The recorded limitations, each stated next to the conclusion it limits: `n = 4` with no replicates;
      differently-typed action variables across the four envs; the confounds (contact dynamics, second
      movable object, rotational state, 2 epochs vs 20); a `GO` is permission to spend 0.8 GPU-h and not
      evidence for the mechanism; `frameskip=5` may wash out within-step reversals; only 2 triples per sample
      so zeroing reversals raises gradient variance and no check measures that; the gate proxies "the
      controlled object reversed", not "the latent velocity's direction change is action-explained"
    - The novelty positioning **dated**, written before the outcome
    - _Requirements: 16.1, 16.3, 16.10, 2.1, 2.17, 10.1, 3.1, 3.2, 3.3, 3.4, 3.5, 3.7, 3.8, 3.9, 14.7_

- [ ] 5. Stage 0 execution and verdict — the hard gate
  - [~] 5.1 [CPU RUN] Run the Stage-0 action-similarity measurement on all four datasets
    - ```bash
      for ENV in pusht wall point_maze point_maze_medium; do
        python probe_ccr_curvature.py --readout actions --env "$ENV" \
          --acs-action-reduce all --split train \
          --out "probe_outputs/acs_actions_${ENV}.json"
      done
      python probe_ccr_curvature.py --readout actions --summarize probe_outputs/acs_actions_*.json \
        --table1-gains "umaze=50.00,medium=10.67,wall=10.67,pusht=7.33" \
        --out probe_outputs/acs_stage0_verdict.json
      ```
    - Also run task 4.2's 32-window bitwise check here, where the real dataset is present
    - **Minutes, CPU only, 0 GPU-h.** Needs `DATASET_DIR` and the four datasets, so it is not
      agent-executable. No checkpoint, no model weights, no video decode
    - _Requirements: 1.1, 1.11, 1.12, 1.13, 1.17, 1.18, 1.16_

  - [~] 5.2 [HUMAN] Record the Stage-0 verdict and decide whether ACS is built at all
    - Read rules A and B off `acs_stage0_verdict.json` and write the firing rule and its exact numbers into
      `PROGRESS_ACS.md` **before** anything downstream is launched
    - **A `STOP` on either rule ends the feature.** Tasks 6.x onward are not executed, `MCA_Fallback`
      (`compute_mca`, already written, never run, zero new code, 0.8 GPU-h to a verdict) is selected as the
      next arm, and the Stage-0 statistics are written up as findings N1 and N2 regardless
    - On `MIDDLE`, record the downgraded mechanism claim — "the gate is a useful inductive bias", with the
      Table-1-ordering explanation explicitly withheld — **at the moment the verdict is read**, not
      retroactively. On a rule-B `MIDDLE`, record `acs_gate=hard` or a sharpened gate as the pre-declared
      remedy and that the expected effect size is small
    - _Requirements: 2.5, 2.10, 2.12, 2.13, 2.14, 2.15, 2.16, 2.17, 16.2, 16.4, 16.5, 16.11, 16.12_

- [ ] 6. The ACS term (only on a Stage-0 GO or MIDDLE)
  - [~] 6.1 [CODE] Implement `compute_acs` and wire it into `forward`
    - Geometry half through the **shared** helpers from 2.1: `feats = visual_only(z)`,
      `v1, v2 = _agg_velocities(feats)`, `c, mask = _cos_curvature_terms(v1, v2)`. Gate half from 3.2. Then
      `num = (w[mask] * c[mask]).sum()`, `den = w[mask].sum().clamp_min(1e-3)`, `loss = num / den`
    - `mask` is applied to `c` **and** `w` before both sums — the same mask, or "flat gate ⟹ baseline" stops
      holding on real data and the term is silently scaled down by the static fraction
    - `EPS = 1e-6`, `STEP_THRESH = 1e-6`, `WEIGHT_SUM_FLOOR = 1e-3` hardcoded, following `step_thresh`'s own
      precedent. No extra encoder pass, no extra predictor call, no new module/parameter/buffer
    - Telemetry dict, all detached scalars: `curvature_unweighted = c[mask].mean()`, `gate_mean`,
      `gate_tv = 0.5·Σ|w/Σw − 1/N|`, `gate_zero_frac`, `gate_p10/p50/p90`, `denom_clamped`, `masked_frac`
    - `forward` gains one gated block: `curvature_mode == "acsaggcos"` reports the ACS value under the
      **existing** `curvature_loss_used_for_training` / `curvature_loss_scaled` keys plus the `acs_*`
      diagnostics; every other mode takes the unchanged branch byte for byte. Exactly one curvature term
      contributes either way — ACS replaces, never adds
    - Runtime guards: `t >= 3` (E6), `z.shape[1] == act.shape[1]` (E7), encoder has `agg` with a message
      shaped like `total_curvature`'s existing check (E4)
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.12, 4.13, 4.14, 4.15, 4.16, 8.1, 8.2, 8.3, 8.8, 8.9, 8.10, 8.11, 8.12, 8.13, 8.18, 9.3, 9.5, 9.6, 9.7, 9.9, 14.11_

  - [~] 6.2 [CODE] Write property test for reduction to `L_curv` (`tests/test_acs_reduces_to_curv.py`)
    - **Property 2: Reduction to `L_curv` at a constant gate**
    - **Validates: Requirements 4.1, 4.3, 4.6, 13.2**
    - `ŵ` drawn log-uniform over `[1e-3, 1]`, including `ŵ = 1`: `compute_acs` equals
      `total_curvature(visual_only(z), "aggcos")` to fp32 tolerance
    - **Gate, deliberately not optional:** this is the executable form of the no-λ-reduction argument. It is
      the reason a win cannot be restated as "you found a better λ", and therefore the reason the λ-matched
      control arm is free rather than a 13.6 GPU-h run

  - [~] 6.3 [CODE] Write property test for the unweighted diagnostic (`tests/test_acs_unweighted_bitwise.py`)
    - **Property 12: The unweighted diagnostic is the baseline's number, bitwise**
    - **Validates: Requirements 4.2, 8.3, 8.4, 14.11**
    - `curvature_loss_unweighted` is **bitwise** equal to `total_curvature(visual_only(z), "aggcos")` on the
      same tensor, detached, and never added to the loss
    - **Gate, deliberately not optional:** under ACS the `curvature` row is a *w-weighted* average while the
      control's is a *uniform* average of the same quantity, so it reads lower even with identical geometry.
      Without this key the gate has a false positive waiting at exactly the moment it matters

  - [~] 6.4 [CODE] Write property test for the shared static mask (`tests/test_acs_static_mask.py`)
    - **Property 9: The static mask is applied to `c` and `w` identically**
    - **Validates: Requirements 4.5, 8.13**
    - Append `k` zero-motion samples: `compute_acs` changes by less than fp32 tolerance and
      `acs_masked_frac` rises accordingly
    - **Gate, deliberately not optional:** this test fails if `w` and `c` use different masks, which is the
      one wiring error that silently rescales the whole term

  - [ ]* 6.5 [CODE] Write property test for bounds and finiteness (`tests/test_acs_bounds.py`)
    - **Property 7: Non-negativity, boundedness, finiteness**
    - **Validates: Requirements 4.7, 4.14, 9.9**
    - Full strategy including all-equal frames, exactly one non-static sample, `b = 1`, every gate mode

  - [ ]* 6.6 [CODE] Write property test for an all-reversing batch (`tests/test_acs_all_reversing.py`)
    - **Property 10: An all-reversing batch yields exactly zero, not a NaN**
    - **Validates: Requirements 4.4, 4.10, 8.12, 9.7**
    - Actions antiparallel by construction: `compute_acs == 0` exactly, finite, gradient defined,
      `acs_denom_clamped_frac == 1.0`. Intended semantics, not an exception path

  - [ ]* 6.7 [CODE] Write property test for latent scale invariance (`tests/test_acs_scale_invariance.py`)
    - **Property 6: Scale invariance in the latent**
    - **Validates: Requirements 4.8**

  - [ ]* 6.8 [CODE] Write property test for batch-permutation invariance (`tests/test_acs_batch_permutation.py`)
    - **Property 8: Batch-permutation invariance of the reduction**
    - **Validates: Requirements 4.3, 4.9**

  - [ ]* 6.9 [CODE] Write property test for monotone reallocation (`tests/test_acs_monotone_reallocation.py`)
    - **Property 11: Monotone reallocation**
    - **Validates: Requirements 4.1, 4.11**
    - Finite-difference in `w_t`, sign compared to `(c_t − L)`. This is what makes "reallocates pressure" a
      true description rather than a slogan

  - [ ]* 6.10 [CODE] Write unit tests for the runtime error paths (`tests/test_acs_errors.py`)
    - E4 (encoder without `agg`), E5 (`act.shape[-1]` not divisible, naming both numbers), E6 (`t < 3`),
      E7 (`z.shape[1] != act.shape[1]`), E10 (zero-norm action block does not raise)
    - _Requirements: 9.3, 9.4, 9.5, 9.6, 9.10_

- [~] 7. Checkpoint - ACS term complete
  - Ensure all tests pass, ask the user if questions arise.
  - Property 1 (2.3), Property 13 (2.4), Property 4 (3.4), Property 19 (4.2), Property 2 (6.2),
    Property 12 (6.3), Property 9 (6.4) and the scope guard (1.1) must all be green, and the pre-existing
    suite must still pass unchanged.

- [ ] 8. Configuration surface, run naming, loss signature and telemetry
  - [~] 8.1 [CODE] Add the `acs_tag` resolver to `custom_resolvers.py`
    - `ACS_TAG_DEFAULTS = ("sum", "relu_cos")` and `acs_tag(action_reduce, gate)` returning `""` at defaults
      and `_ar{}_g{}` otherwise, registered as a **sibling** of `ccr_tag`
    - `ccr_tag`'s arity, defaults and output are **untouched for every input**. That is what keeps
      `pusht_aggmlpcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05` byte-identical and the existing
      `model_2.pth` and its telemetry addressable
    - _Requirements: 6.10, 6.13_

  - [~] 8.2 [CODE] Add the two keys and the interpolation to `conf/train.yaml`
    - `training.acs_action_reduce: sum` and `training.acs_gate: relu_cos`, each commented as a closed enum
      with a pre-registered default rather than a continuous constant
    - Append `${acs_tag:${training.acs_action_reduce},${training.acs_gate}}` **after** `${ccr_tag:...}` in
      both `hydra.run.dir` and `hydra.sweep.dir`, leaving the rest of the expression untouched
    - _Requirements: 6.7, 6.9, 6.11, 6.12_

  - [~] 8.3 [CODE] Forward the kwargs, extend the loss signature, and add the `acs` telemetry block in `train.py`
    - Forward `acs_action_reduce=self.cfg.training.get("acs_action_reduce")` and
      `acs_gate=self.cfg.training.get("acs_gate")` into the model constructor beside the existing
      `mca_weight` / `ccr_*` forwards, so an absent yaml key arrives as `None` and selects the default
    - `LOSS_SIGNATURE_KEYS += ("acs_action_reduce", "acs_gate")`;
      `LOSS_SIGNATURE_DEFAULTS += {"acs_action_reduce": "sum", "acs_gate": "relu_cos"}`. `straighten` is
      already there, so `aggcos1e-1` vs `acsaggcos1e-1` already differ and `_guard_run_dir` already refuses
      a silent cross-resume
    - `TELEMETRY_ACS_KEY = "acs_gate_mean"` and `_acs_telemetry_block()` shaped like the existing CCR block:
      `enabled`, `gate_mean`, `gate_tv`, `gate_zero_frac`, `gate_p10/p50/p90`, `denom_clamped_frac`,
      `masked_frac`, `curvature_unweighted`, `action_reduce`, `gate`. `enabled` is derived from
      `"acs_gate_mean" in loss_components`, never from config; a disagreement with the config logs a
      warning; the block is omitted entirely when the ACS path did not run
    - **`TELEMETRY_TERMS` is unchanged.** `curvature_loss_unweighted` is deliberately not a term — that is
      what keeps `Σ share ≈ 1.0` and what keeps `--compare` diffing the arm's `curvature` row against the
      control's
    - _Requirements: 6.8, 6.14, 8.5, 8.6, 8.7, 8.14, 8.15, 8.16, 8.17, 9.13_

  - [~] 8.4 [CODE] Update `tests/test_run_naming.py` for the appended tag
    - **This is an edit to an existing test, not a new module and not a replacement.** The file currently
      derives its pre-feature template by stripping the `ccr_tag` interpolation and asserts `ccr_tag` is
      appended at the very end; appending `acs_tag` breaks that assertion
    - Strip **both** interpolations to recover the pre-feature template, and assert the **pair** is appended
      at the end in the order `ccr_tag` then `acs_tag`. `tests/*` is in the scope allowlist, so this is an
      expected in-scope edit and not a scope violation
    - Not marked optional: without it the existing suite fails, and Requirement 14.14 requires the suite to
      stay green
    - _Requirements: 6.17, 14.14_

  - [ ]* 8.5 [CODE] Write property test for run-directory names (`tests/test_acs_run_dir.py`)
    - **Property 17: Run-directory names**
    - **Validates: Requirements 6.10, 6.11, 6.12, 6.13**
    - `${ccr_tag:...}${acs_tag:...}` is empty at defaults, so the baseline resolves byte-identically to
      `pusht_aggmlpcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05`; `straighten=acsaggcos1e-1` resolves
      to a distinct directory; `ccr_tag` is unchanged for all inputs

  - [ ]* 8.6 [CODE] Write property test for legacy resume (`tests/test_acs_legacy_resume.py`)
    - **Property 18: Legacy resume survives the loss-signature change**
    - **Validates: Requirements 6.15**
    - A `loss_config.json` written before `acs_*` existed compares equal to a current default launch, so the
      baseline checkpoint stays resumable

  - [ ]* 8.7 [CODE] Write property test for telemetry `enabled` (`tests/test_acs_telemetry_enabled.py`)
    - **Property 14: Telemetry `enabled` reflects what ran, not what was configured**
    - **Validates: Requirements 8.8, 8.10, 8.11, 8.15, 8.16, 8.17**
    - Mirrors the CCR fix, where a config-derived `enabled` read `3` on a CCR-disabled baseline and therefore
      confirmed nothing

  - [ ]* 8.8 [CODE] Write property test for term shares (`tests/test_acs_term_shares.py`)
    - **Property 15: Term shares still sum to ~100%**
    - **Validates: Requirements 8.5, 8.6**
    - `Σ terms[*].share ≈ 1.0` within 0.01 under ACS; catches an accidental addition of
      `curvature_loss_unweighted` to `TELEMETRY_TERMS`

- [ ] 9. Early-read gate tooling
  - [~] 9.1 [CODE] Build `--prediction-gate` and `--prediction-gate-direction` in `summarize_training_log.py`
    - **`--prediction-gate` does not exist yet.** The TMR spec proposed it and TMR was never built, so it is
      built here. `--prediction-gate REFERENCE_RUN_DIR` plus
      `--prediction-gate-direction {improve,guard}`, **default `guard`** so nothing existing changes
      behaviour; ACS runs it with `improve`
    - Verdict function, read at matched `global_iter` against the control's own (exact) rows: `GO` when
      scaled `prediction` at 8000 is `<= 0.013196` **and** at least 15 of the last 20 matched rows are
      better; additionally `STRONG GO` at `<= 0.012536`, recorded separately; `STOP` at `> 0.014516` **or**
      when at least 15 of the last 20 rows are worse; `MIDDLE` otherwise
    - Also print check 0 (`it_per_s` from steady-state rows past row 400 against `--reference-it-per-s`, with
      the `2.72` floor) and check 1b (curvature share at `global_iter` 200 and 8000 against `[65%, 80%]`,
      prediction share against the `11.75%` floor, and `curvature_loss_used_for_training` at 200 and 8000
      with their ratio). `--collapse-check` and the term table already generalize
    - _Requirements: 10.2, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 10.10, 10.11, 10.12, 10.14, 10.15, 10.16_

  - [~] 9.2 [CODE] Add `--acs-gate-check` to `summarize_training_log.py`
    - Read the `acs` telemetry block and print `gate_mean`, `gate_tv`, `gate_zero_frac`, `gate_p10/p50/p90`,
      `denom_clamped_frac` and `masked_frac` against the Stage-0 population estimate, with the pass
      conditions evaluated mechanically: `acs_gate_tv >= 0.08`, `acs_gate_tv` within a factor 1.5 of Stage-0
      `R` for PushT, `acs_denom_clamped_frac < 0.01`
    - Making it a flag is what stops check 1c from being an eyeball. A large Stage-0-vs-training mismatch
      means the training-time `a_t` is not the one Stage 0 measured — a wrong axis in the substep reduction
      or an off-by-one in the triple-to-action-pair alignment
    - _Requirements: 10.17, 10.18, 10.19, 10.20, 10.21, 10.22, 10.23_

  - [ ]* 9.3 [CODE] Write unit tests for the early-read verdict function (`tests/test_acs_prediction_gate_verdict.py`)
    - Second uncovered decision-rule layer: P1-P19 cover the model, the verdict functions cover the numbers
      the project is read against. On synthetic JSONL fixtures:
    - The check-1 bands at, just below and just above `0.013196`, `0.012536` and `0.014516`; the 15-of-20
      sign-test boundary at 14, 15 and 16 rows in both directions
    - **Partition**: `GO` / `STOP` / `MIDDLE` cover the whole `(prediction, rows_better, rows_worse)` space
      with no gap and no overlap, and `STRONG GO` is recorded as an annotation on `GO` rather than a fourth
      disjoint verdict
    - Direction handling: the same input yields opposite verdicts under `guard` and `improve`, and `guard`
      reproduces the pre-ACS semantics exactly
    - _Requirements: 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 10.10_

  - [ ]* 9.4 [CODE] Write unit tests for the gate-statistic identities (`tests/test_acs_gate_statistics.py`)
    - Third uncovered decision-rule layer: `acs_gate_tv` (finite-batch) and `R` (population) must be forms of
      the same quantity, or Stage 0's prediction and Stage 1's measurement are not comparable and check 1c
      is meaningless. Assert they agree to a stated tolerance as the batch grows, on synthetic weight
      populations including a flat gate (`tv = R = 0` at any `mean(w)`) and a fully bimodal one
    - Assert the `permuted` gate preserves `mean(w)`, the full weight distribution (multiset equality) and
      `gate_tv` **exactly**, changing only the correspondence between a weight and its own triple — that
      exactness is what makes the attribution arm interpretable and is itself a check on the arm
    - _Requirements: 15.4, 13.5, 13.7, 8.9_

  - [~] 9.5 [CODE] Add `--readout gatesplit` to `probe_ccr_curvature.py`
    - Held-out per-triple curvature under an arm and a control checkpoint, bucketed by `w` into `w = 0` and
      `w >= 0.5`, comparing the **unweighted** per-triple curvature within each bucket. Reuses
      `_aggregate_latent` and the shared `_cos_curvature_terms`; the overall unweighted mean is reported
      without a pre-registered direction
    - Read-only, CPU-only, `--num-windows 192`. 64 windows is noise: the CCR round read a `block_angle`
      delta of −28% at 64 that collapsed to −9% at 192
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.8_

  - [ ]* 9.6 [CODE] Write unit tests for the gate-split bucketing (`tests/test_acs_gatesplit.py`)
    - Bucket boundaries (`w = 0` exactly, `w = 0.5` inclusive), empty-bucket handling, and that the reported
      per-bucket quantity is the unweighted curvature rather than the weighted one
    - _Requirements: 11.2, 11.3, 11.6_

- [~] 10. Checkpoint - all local gates green before any pod job
  - Ensure all tests pass, ask the user if questions arise.
  - No GPU job launches before this checkpoint. Required green: the scope guard (1.1), Property 1 (2.3),
    Property 13 (2.4), Property 4 (3.4), Property 19 (4.2), Property 2 (6.2), Property 12 (6.3),
    Property 9 (6.4), the updated `tests/test_run_naming.py` (8.4), and the full pre-existing suite.
  - `PROGRESS_ACS.md` must already carry the Stage-0 verdict and the full pre-registered text of checks 0
    through 3, written before any arm data exists.

- [ ] 11. Stage 1 - the 8,000-step arm and the early-read gate (~0.8 GPU-h)
  - [~] 11.1 [GPU RUN] Launch the ACS arm against the bitwise-matched control
    - ```bash
      CKPT_BASE=$PWD/checkpoints_acs8k bash run_ccr_pilot.sh pilot \
        training.straighten=acsaggcos1e-1
      ```
    - **No launcher edit is needed.** `run_ccr_pilot.sh`'s `add_default training.straighten=aggcos1e-1` goes
      through `_user_overrides_key`, so a user-supplied `training.straighten=acsaggcos1e-1` suppresses the
      default and the whole protocol block (env, encoder, lr `1e-5`, `stop_grad=True`, iteration cap,
      epochs) applies unchanged. `λ` stays at 0.1 — there is nothing to calibrate
    - The control is free and exact:
      `CTRL=$PWD/checkpoints_ctrl8k/test/pusht_aggmlpcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05`.
      The arm lands in
      `checkpoints_acs8k/test/pusht_acsaggmlpcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05`
    - Two-minute smoke check: the startup line names `curvature_mode=acsaggcos`, and the first telemetry row
      carries `acs_gate_mean` — presence of that key, not the config, is what proves the ACS path ran
    - ~47 min, one job on the `1g.45gb` slice. Operational traps: `nvidia-smi` does **not** enumerate MIG
      processes, use `ps -eo pid,stat,etime,cmd | grep '[p]ython train'`; `kill <pid>` does **not** stop a
      run, use `kill -- -<pid>` on the driver's process group; `kill -0` **succeeds on zombies** because
      PID 1 does not reap in this container, so read `ps -p <pid> -o stat=` and treat `Z` or empty as
      finished; the PID counter has wrapped, so match on `cmd` as well as PID; **never Ctrl-Z a GPU job**
    - _Requirements: 12.8, 12.9, 12.13, 13.1, 13.3_

  - [~] 11.2 [CPU RUN] Run checks 0, 1, 1b and 1c against the control
    - ```bash
      ARM=$PWD/checkpoints_acs8k/test/pusht_acsaggmlpcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05
      python summarize_training_log.py "$ARM" --compare "$CTRL" --collapse-check \
        --reference-it-per-s 2.862 --iter 8000 \
        --prediction-gate "$CTRL" --prediction-gate-direction improve \
        --acs-gate-check
      ```
    - Check 1 is read with `improve`: prediction loss is a **positive directional prediction** here, not a
      guard. It is the only quantity measured to be causally linked to success on this codebase
    - Also record the shared-term smoke comparison at `global_iter` 200 within `rtol = 0.05`, using
      `curvature_loss_unweighted` against the control's `curvature` — the *unweighted* quantity, never the
      weighted one
    - A `it_per_s` below `2.72` is a **bug**, not a cost: predicted ACS overhead is order `1e-8` of the step.
      Fix the code and hold the arm
    - _Requirements: 10.2, 10.3, 10.5, 10.11, 10.12, 10.13, 10.14, 10.15, 10.16, 10.17, 10.24, 10.25, 10.26, 9.8_

  - [~] 11.3 [CPU RUN] Run the gate-split probe on the arm and the control (check 2a)
    - ```bash
      python probe_ccr_curvature.py --readout gatesplit --num-windows 192 \
        --ckpt "$ARM/checkpoints/model_latest.pth" --train-cfg "$ARM/hydra.yaml" \
        --out probe_outputs/acs_gatesplit_arm.json
      # then the identical command against the control checkpoint, then diff the two reports
      ```
    - Pre-registered directions: the arm's `w = 0` bucket curvature **higher** than the control's, the
      `w >= 0.5` bucket **equal or lower**. Failing both directional rows is a `STOP` — nothing downstream is
      attributable to the gate. This is sharper than "did the loss go down", which is exactly what CCR
      delivered (−96% on its own objective, none of it converting)
    - Read-only, CPU. Serial with every other probe/train job on the slice
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7_

  - [~] 11.4 [CPU RUN] Run the state readout on the arm and the control (check 2b)
    - ```bash
      python probe_ccr_curvature.py --readout state --num-windows 192 \
        --ckpt "$ARM/checkpoints/model_latest.pth" --train-cfg "$ARM/hydra.yaml" \
        --out probe_outputs/acs_state_arm.json
      ```
    - `state_readout_r2` reused unchanged. `block_angle` R² must not degrade beyond noise; an improvement is
      recorded as supporting evidence and is **not required** for `GO`. Rotation is curvature, so this is the
      prediction that would turn `PROGRESS_CCR.md` §6f from a limitation into a general statement about
      curvature-family regularizers (finding N3)
    - _Requirements: 11.9, 11.10, 16.13_

  - [~] 11.5 [GPU RUN] Matched-budget evaluation of the 8k checkpoints (check 3, ~0.4 GPU-h)
    - 1 data seed, unmodified evaluation protocol, open-loop and MPC, `PLAN_SERIAL_ENV=1`, read against the
      measured Control_8k values `16.0` OL / `18.0` MPC
    - **Catastrophe detector only.** Both arms sit near the floor; at `p ≈ 0.17` the per-arm binomial SE is
      ~5.2 points, so `Δ` inside `±10` carries no information and must be reported as neither support nor
      refutation. `Δ <= -10` on either setting is a red flag worth acting on
    - _Requirements: 11.11, 11.12, 11.13, 11.14, 11.15_

  - [~] 11.6 [HUMAN] Record the early-read gate verdict
    - Walk checks 0 → 1 → 1b → 1c → 2a → 2b → 3 in order and write each measured number against its
      pre-registered threshold into `PROGRESS_ACS.md`. Specifically record: whether the directional
      prediction on the causal channel held; whether the check-1b scale-preservation prediction held; and
      training `acs_gate_tv` against the Stage-0 `R` estimate, **including if the arm succeeded**
    - A `gate_tv ≈ 0` verdict is a `STOP` on the grounds that the term *is* the baseline — regardless of what
      `mean(w)` reads, since a flat gate at any level reproduces the baseline exactly
    - Record the curvature-share **drift** across iterations rather than a single row: calling shares
      converged off two points was a documented CCR error (31.4% @200 → 65.4% @3000 → 73.7% @8000 →
      80.5% @35.6k → 82.7% @123.9k)
    - State the limits in the same paragraph as the conclusion: 8,000 steps is 6.5% of the budget, both arms
      sit near the success-rate floor, the matched-budget test is structurally biased against any new term,
      and one seed does not establish generalization
    - _Requirements: 10.13, 10.24, 10.25, 16.2, 16.6, 16.7, 16.8, 16.9, 16.14, 16.16_

- [ ] 12. Attribution arm - permuted gate (conditional, ~0.8 GPU-h)
  - [~] 12.1 [GPU RUN] Launch the permuted-gate control arm
    - ```bash
      CKPT_BASE=$PWD/checkpoints_acsperm8k bash run_ccr_pilot.sh pilot \
        training.straighten=acsaggcos1e-1 training.acs_gate=permuted
      ```
    - `acs_tag` appends `_arsum_gpermuted`, so this arm gets its own run directory and its own loss signature
      with no further work
    - **Launched only after the arm cleared check 1 with a confirmed directional prediction.** It answers the
      one objection the free baseline cannot: "any reweighting that takes pressure off the most-curved
      triples would help". It is uninformative if the primary arm produced no signal to attribute
    - Same operational traps as 11.1. Serial: one job per MIG slice
    - _Requirements: 13.4, 13.6_

  - [~] 12.2 [CPU RUN] Verify the permutation behaved as specified
    - `summarize_training_log.py <perm_run_dir> --compare "$CTRL" --acs-gate-check`: the permuted arm's
      `acs_gate_mean` and `acs_gate_tv` must match the ACS arm's within batch noise. A mismatch is evidence
      that the permutation is not doing what it claims, not a finding about attribution
    - _Requirements: 13.5, 13.7_

  - [~] 12.3 [HUMAN] Record the attribution verdict
    - If ACS beats the permuted arm, the effect is attributable to **action conditioning**; if it does not,
      it is attributable to **reweighting**, which is a weaker and differently-framed claim. Write whichever
      it is into `PROGRESS_ACS.md`, together with the standing statement that no arm controls for
      PushT-specific effects and a single-environment result remains a single-environment result
    - _Requirements: 13.7, 13.8, 16.2_

- [ ] 13. Stage 2 - full run and acceptance (~13.6 GPU-h)
  - [~] 13.1 [GPU RUN] Verify resume before committing to a 12-hour run
    - Relaunch into the Stage-1 arm's directory and confirm it **resumes** from `model_latest.pth` rather
      than restarting from scratch or raising: `global_iter` continues, the loss-signature guard passes, and
      the telemetry log appends rather than truncating
    - This is not a formality. `train.py` resume was silently broken for DINOv2 runs once and nobody noticed
      because every run had started fresh. Minutes; do it before 13.2, not after
    - _Requirements: 6.16, 6.15, 9.12_

  - [~] 13.2 [GPU RUN] Full run at the paper budget (~12.1 GPU-h)
    - ```bash
      bash run_ccr_pilot.sh full training.straighten=acsaggcos1e-1
      ```
    - Launched **only** for an arm that cleared the early-read gate. Every protocol invariant untouched:
      encoder lr `1e-5`, 2 epochs PushT, batch 32, `num_hist=3`, `num_pred=1`, `frameskip=5`, bf16,
      `stop_grad=True`, `λ = 0.1`, CCR off (`lambda_cf=0`, `ccr_rho=0`), `max_iterations` back to 0
    - Same operational traps as 11.1, and they matter more here: a naive wait loop that trusts `kill -0`
      already burned 2 h 39 m of idle GPU on a slice whose job had finished
    - _Requirements: 12.1, 12.8, 12.9, 12.13, 14.8_

  - [~] 13.3 [GPU RUN] Three-seed evaluation of the full-run candidate (~1.5 GPU-h)
    - ```bash
      bash run_ccr_pilot.sh eval <full_run_dir>
      ```
    - Seeds 100 / 200 / 300, `n_evals=50`, `PLAN_SERIAL_ENV=1`, both settings: open-loop GD at
      `objective.mode=last, alpha=1, max_iter=1, n_taken_actions=25`; MPC at `mode=staged, alpha=1,
      max_iter=20, n_taken_actions=5`; sub-planner horizon 25, lr 0.1, `sample_type=zero`,
      `action_noise=0`, `opt_steps=100`. Six jobs, serial in one driver
    - _Requirements: 12.5, 12.6, 12.10, 12.11, 12.12_

  - [~] 13.4 [HUMAN] Record the acceptance verdict
    - ```bash
      python ccr_acceptance_gate.py <eval_outputs...>
      ```
    - Bar: **79.33 open-loop and 87.00 MPC**, both settings, on the 3-seed mean. One setting alone is a
      failure. Report **per-seed** values beside the mean, not the mean alone
    - Record in the same paragraph that `+4` open-loop on a 3-seed mean is roughly 1.3 SE even with exact
      pairing, and that the single-checkpoint spread `74 / 82 / 70` is the noise reality this number lives in
    - If a positive PushT result leads to another environment, the claim there is restricted to open-loop:
      paper MPC is 100.00 Wall, 100.00 UMaze, 98.67 Medium, so a `+5` MPC margin is arithmetically impossible
    - _Requirements: 12.2, 12.3, 12.4, 12.7, 12.14, 12.15, 16.15, 16.16_

  - [~] 13.5 [HUMAN] Finalize the Negative_Result_Record
    - Complete `PROGRESS_ACS.md` whatever the outcome: which Stage-0 rule fired and its numbers; which gate
      stopped it, if any; training `acs_gate_tv` against Stage-0 `R`; whether the scale-preservation
      prediction held; whether the directional prediction on the causal channel held; every error made,
      including those that cost only minutes; and the dated novelty positioning as written before the outcome
    - Findings N1 (per-environment action-similarity distributions against Table 1's `+50.00 / +10.67 /
      +10.67 / +7.33`), N2 (`R` and `frac(w = 0)` per environment) and N3 (the measured `block_angle` R²
      direction) are written up **regardless** — N1 and N2 cost zero GPU-hours and stand whether ACS is built
      or not
    - _Requirements: 16.2, 16.4, 16.6, 16.7, 16.8, 16.9, 16.10, 16.11, 16.12, 16.13, 16.14, 16.15, 16.16, 3.9_

## Notes

- **A Stage-0 STOP means tasks 6.x through 13.x are not executed.** Requirements 2.12 and 2.13 state that a
  `STOP` on either rule forbids implementing the ACS term, the gate or any other ACS code path, and
  Requirement 2.15 selects `MCA_Fallback` instead. This is a real branch with a stated ~35-45% chance of
  firing, not a formality: `compute_mca` is already written in `models/visual_world_model.py`, has never been
  run, needs zero new code, costs `<0.1%` overhead and 0.8 GPU-h to a verdict, and targets a different gap
  (the regularization-space versus planning-space mismatch). If Stage 0 stops, the deliverable is findings N1
  and N2 for zero GPU-hours, and the next arm is MCA.
- Tasks marked with `*` are optional and can be skipped for a faster path. Eight test tasks are deliberately
  **not** optional because they are gates rather than checks: 1.1 (the scope guard, the only automated check
  of scope containment), 2.3 (Property 1, default-off bitwise, which is what lets 75.33 / 82.00 stand without
  a retrain), 2.4 (Property 13, the silent-disable hole), 3.4 (Property 4, gate detachment), 4.2
  (Property 19, one implementation of the gate), 6.2 (Property 2, reduction to `L_curv` at a constant gate —
  the executable form of the no-λ-reduction argument and the reason the control arm is free), 6.3
  (Property 12, the unweighted diagnostic bitwise equal to the baseline's number, without which the gate has
  a false positive waiting), and 6.4 (Property 9, the shared mask). Task 8.4 is also unmarked, but for a
  different reason: it is an **edit** to an existing test that would otherwise fail, and Requirement 14.14
  requires the suite to stay green.
- **The three decision-rule test modules (4.4, 9.3, 9.4) are marked optional by format convention but are
  strongly recommended.** P1-P19 cover the model layer; the verdict functions — Stage-0 rules A/B, the
  check-1 bands and sign test, and the `acs_gate_tv` / `R` identity — are otherwise untested, and they are
  the code that decides whether the feature lives. They are CPU-only and cheap. Skipping them means trusting
  the numbers the whole project is read against to code nothing checks.
- **[CODE]** tasks are agent-executable on CPU with no dataset, checkpoint or network access — the stub
  encoder in `tests/conftest.py` exists exactly for that. **[CPU RUN]**, **[GPU RUN]** and **[HUMAN]** tasks
  are operator or judgement work, listed for sequencing rather than agent execution.
- `tests/conftest.py` (1.2), `tests/reference_impl.py` (1.3), `tests/test_scope_guard.py` (1.1) and
  `tests/test_run_naming.py` (8.4) are **edits to existing files**, additive in every case. Existing CCR and
  aggregated-space fixtures keep their names and values; `ccr_tag` keeps its arity, defaults and output.
  These edits are in-scope (`tests/*` is in the allowlist) and must not be mistaken for scope violations.
- One property per test module, so independent property tasks never write the same file and can be scheduled
  in parallel. The set of properties is exactly the design's P1-P19. Every property test docstring carries
  **Feature: action-conditioned-straightening, Property N: <property text>**, minimum 100 examples via
  `@settings(max_examples=100)`.
- `TELEMETRY_TERMS` is unchanged by this feature. `curvature_loss_unweighted` is deliberately not a term:
  that is what keeps `Σ share ≈ 1.0` (Property 15) and what keeps `--compare` able to diff the arm's
  `curvature` row against the control's at matched `global_iter`.
- Serialization is real, not stylistic: the `1g.45gb` MIG slice holds exactly one job, so 11.1, 11.5, 12.1,
  13.1, 13.2 and 13.3 each occupy their own wave, and the CPU probe runs (5.1, 11.2, 11.3, 11.4, 12.2) are
  serialized with them by `run_ccr_pilot.sh`'s `ps` pre-flight guard.
- Operational traps carried into every GPU task note because they cost real time: `nvidia-smi` does not
  enumerate MIG processes (use `ps`); `kill <pid>` does not stop a run (use `kill -- -<pid>` on the driver's
  process group, which holds `train.py` and ~16 dataloader workers); `kill -0` succeeds on zombies because
  PID 1 does not reap in this container (read `ps -o stat=` and treat `Z` or empty as finished); the PID
  counter has wrapped, so match on `cmd` too; never Ctrl-Z a GPU job.
- The λ-matched control costs **zero**: the weighted-mean form is invariant to uniform gate rescaling, so the
  λ-matched plain-`L_curv` arm at `λ = 0.1` *is* the existing baseline, already trained and evaluated, with a
  bitwise 8k prefix in `checkpoints_ctrl8k` (40/40 telemetry rows agreeing to `+0.000000`). There is no
  calibration ladder anywhere in this plan, because there is no new magnitude to derive.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2", "1.3", "4.5"] },
    { "id": 1, "tasks": ["2.1"] },
    { "id": 2, "tasks": ["2.2"] },
    { "id": 3, "tasks": ["2.3", "2.4", "3.1"] },
    { "id": 4, "tasks": ["3.2"] },
    { "id": 5, "tasks": ["3.3", "3.4", "3.5", "4.1"] },
    { "id": 6, "tasks": ["4.2", "4.3"] },
    { "id": 7, "tasks": ["4.4"] },
    { "id": 8, "tasks": ["5.1"] },
    { "id": 9, "tasks": ["5.2"] },
    { "id": 10, "tasks": ["6.1"] },
    { "id": 11, "tasks": ["6.2", "6.3", "6.4", "6.5", "6.6", "6.7", "6.8", "6.9", "6.10", "8.1"] },
    { "id": 12, "tasks": ["8.2", "8.3"] },
    { "id": 13, "tasks": ["8.4", "8.5", "8.6", "8.7", "8.8", "9.1", "9.5"] },
    { "id": 14, "tasks": ["9.2", "9.3", "9.6"] },
    { "id": 15, "tasks": ["9.4"] },
    { "id": 16, "tasks": ["11.1"] },
    { "id": 17, "tasks": ["11.2"] },
    { "id": 18, "tasks": ["11.3"] },
    { "id": 19, "tasks": ["11.4"] },
    { "id": 20, "tasks": ["11.5"] },
    { "id": 21, "tasks": ["11.6"] },
    { "id": 22, "tasks": ["12.1"] },
    { "id": 23, "tasks": ["12.2"] },
    { "id": 24, "tasks": ["12.3"] },
    { "id": 25, "tasks": ["13.1"] },
    { "id": 26, "tasks": ["13.2"] },
    { "id": 27, "tasks": ["13.3"] },
    { "id": 28, "tasks": ["13.4"] },
    { "id": 29, "tasks": ["13.5"] }
  ]
}
```
