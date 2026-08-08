# PROGRESS — Aggregated-Space Planning Cost

Live state of the aggregated-space arm (`L_plan = L_spatial + w · L_agg`). Written so the work can be
resumed cold, without the conversation.

Requirements and design live in `.kiro/specs/aggregated-space-planning-cost/`. This file records what
has been **measured**, and nothing else. Selected as the next arm after `PROGRESS_MCA.md` §6.2 returned
`STOP` at rung 1.

---

## 1. What this arm is, and its honest novelty position

`L_plan = L_spatial + w · L_agg`, applied at planning time only. **No new loss mathematics exists in
this feature:** both terms come from calling the frozen `planning.objectives.create_objective_fn`, and
the only new computation is a frame-wise reshape through the checkpoint's own `agg_mlp` /
`agg_post_norm`. Nothing trains.

**It is the paper's own formula in a new regime, not a new method.** The paper introduces it only at
50-step targets, claims it only under MPC, and rests it on PushT baselines of 13.33 OL / 24.00 MPC.
This spec applies it at **25-step** targets against a **75.33 / 82.00** baseline and gates on **both**
settings. Of the paper's eight combined-cost cells: one clears 2 SE (+9.33 MPC), two are marginal
(+6.67 OL, and **−7.33** OL on Medium), five sit inside noise, and **two of four open-loop cells are
worse**. So the short-horizon dual gate asks for evidence the paper's own table does not contain.

**Why it is nonetheless better motivated than MCA was.** Both close the same gap — straightening acts
in the 128-d aggregated space while `planning/objectives.py` scores MSE in the 1568-d patch space. MCA
tried to force `agg` toward a similarity; `PROGRESS_MCA.md` M1 then measured that the trained `agg` is
strongly non-metric **because training made it so** (`CV(r)` 0.094 → 0.589, `ρ = +0.487`, the two
spaces disagreeing on 18.6% of motion pairs), which suggests the distortion may be useful rather than a
defect. This arm makes the planner score **in the space the regularizer already acts in**, so it
renders the distortion irrelevant instead of correcting it, and needs no claim about whether the
distortion is good.

---

## 2. Status

| item | state |
|---|---|
| Objective wrapper, protocol guard, instrumentation, sweep tooling | complete (commit `d3c3ce5`); CPU property tests green |
| **Task 11.1 — paired zero-weight check** | **PASS 2026-08-08 — see §4. Numerically identical, not merely equal in the mean** |
| Task 11.2 — record the paired verdict | **this file, §4** |
| Task 11.3 — long-horizon protocol column | **complete** — `PROTOCOL_EXPECTED` keyed on `(config_name, goal_H)`; short columns unweakened; `goal_H` now pinned. See §6 |
| Task 9.2 — driver-contract test | **complete, promoted out of optional** — §5's recommendation, acted on |
| Task 11.4 — Positive_Control (~1.5 GPU-h) | _not launched_ — **reading recorded in §6 before launch** |
| Task 11.5 — Positive_Control verdict | _not read_ |
| Section 12 — the 6-arm weight sweep (~4 GPU-h) | **not launched; gated on BOTH 11.1 and 11.4** |
| Acceptance gate | not reached |

**GPU-hours spent on this arm so far: ~0.05** (two 85-second evaluations).

**Seeds, stated because it is the first thing a reader queries.** `Tuning_Seed` = **400**, held out from
reporting, used only to select `Agg_Weight`. `Reporting_Seeds` = **100, 200, 300**, used only for the
confirmation run and the Acceptance_Gate (Requirement 6). Seed 400 is **not** part of the paper's
evaluation protocol and is not meant to be — it exists so the weight is not tuned against the gate the
result is judged by, which is stricter than the paper, whose `w = 0.1` has no stated tuning provenance.
**No seed-400 number is comparable to the 75.33 ± 6.11 baseline**, which is a 3-seed mean over
100/200/300.

---

## 3. The reference row (measured, do not re-derive)

Platform_Baseline, `model_2.pth` @124k, `n_evals=50`, seeds 100/200/300 — per-seed values, never means
alone:

| setting | measured | per-seed | paper |
|---|---|---|---|
| open-loop | **75.33 ± 6.11** | 74, 82, 70 | 77.33 ± 6.18 |
| MPC | **82.00 ± 2.00** | 82, 80, 84 | 85.33 ± 4.99 |

A 12-point open-loop spread across three seeds **on the same checkpoint** is the noise floor every claim
here lives in. Binomial SE at `n = 50`, `p ≈ 0.8` is ~5.7 points on a single seed.

The baseline cell's `logs.json` is backed up outside its run directory at
`~/baseline_backups/platform_baseline_logs_20260808T152313Z.json` (pod-local). It is the reference all
five arms attempted so far are measured against, and the `plan.py` leg of task 11.1 resolves to that
exact directory unless steered away — see §5.1.

---

## 4. Task 11.1 — paired zero-weight check: PASS (2026-08-08)

Open-loop, seed 400, `n_evals=50`, one job at a time on the `1g.45gb` MIG slice. Both legs on
commit `8e27e31`.

| | leg 1 — `plan_agg.py` `+agg_weight=0` | leg 2 — `plan.py`, frozen entry |
|---|---|---|
| success rate | **0.64** | **0.64** |
| per-episode `success` (50) | recorded | **identical element for element** |
| `state_dist` (50, float32) | `47.410824, 11.765438, 39.83232, 181.32492, …` | **identical to 8 significant figures** |
| `perform_planning_s` | 76.299 | 76.193 |
| run directory | real `aggw0` cell (Baseline_Arm for task 12.7) | `plan_outputs_gd_scratch/..._seed400` |
| baseline guard | — | **BASELINE INTACT** |

**Stronger than the task required.** The task allowed falling back to mean equality over 50 episodes if
`plan.py`'s per-episode vector could not be recovered. Both the boolean vector **and** the float32
`state_dist` array match exactly, so the wrapper at `w = 0` is **numerically identical** to the frozen
entry point rather than statistically indistinguishable from it. Concretely: `create_agg_objective_fn`
at `w = 0` reduces to `create_objective_fn`, and `RecordingPlanEvaluator` is behaviourally
`PlanEvaluator`. **Any difference at nonzero weight is therefore attributable to `L_agg` and not to a
wrapper artifact** — which is the entire purpose of this check, and what licenses reading the sweep.

Corroborating detail: the 0.1 s planning-time delta is consistent with the `<0.1%` overhead claim.
`Agg_Head` loaded as `agg_type=mlp in_dim=1568 out_dim=128 checkpoint_epoch=2`, `protocol_ok=True`,
`enabled=False`, records flushed with **0 record failures**.

**What this does NOT establish.** It is an identity check, not a measurement of the method. 0.64 at seed
400 is below all three baseline seeds (74/82/70) and says nothing whatever about whether `L_agg` helps;
at `n = 50` its binomial SE is ~6.8 points. The sweep is still gated on task 11.4's Positive_Control,
because a flat short-horizon sweep on its own cannot distinguish "the term does not transfer out of the
long-horizon regime" from "the wrapper is subtly wrong somewhere the CPU property tests cannot reach" —
and those tests check the objective's algebra, never that it improves anything.

---

## 5. Errors and hazards found

| # | date | error | cost | how it was caught |
|---|---|---|---|---|
| 1 | 2026-08-08 | **`HYDRA_RUN_DIR` had never worked.** `agg_objectives.run_dir_override` emitted `hydra.run.dir=<template>` **unquoted**, and Hydra parses an override's right-hand side with its own ANTLR grammar before OmegaConf sees it — that grammar rejects an unquoted `}`. Both call sites were broken: the `agg` branch **and** `run_ccr_pilot.sh`'s caller-supplied branch, which is the one the `plan.py` leg takes, so both legs of task 11.1 would have failed | 1 second, wrote nothing, 0 GPU-h | Task 11.1's first invocation. **Why it survived review:** there were two quoting requirements at two layers and only one was guarded — `test_run_dir_overrides_are_single_quoted_in_shell_drivers` checks protection from *bash* expansion, and nothing handed the token to *Hydra*. Worse, `test_templates_require_single_quoting_and_survive_it` asserted `token == f"hydra.run.dir={template}"`, so the suite was **pinning the bug in place**. Fixed in `e737eb3` with `test_run_dir_override_parses_under_hydra_grammar`, which parses the emitted token with the real `OverridesParser` and asserts the value round-trips unchanged |
| 2 | 2026-08-08 | **`plan_agg.py` never imported `custom_resolvers`**, so `replace_slash` was unregistered and job start-up died with `UnsupportedInterpolationType` — one layer past error 1, with an unrelated message. `plan.py` imports it at module level, which is why the shipped template resolves there | 1 second, 0 GPU-h | The next invocation, after error 1 was fixed. Fixed in `8e27e31`; the import went **inside** the guarded hydra block, because `custom_resolvers` pulls in hydra/omegaconf and this file's docstring promises the protocol layer stays importable without them. Guarded by a **static source** check so it runs where hydra is absent — deliberately, since that is the environment where the mistake is invisible to any runtime check |

**The common cause, worth naming once:** both bugs were in code that had **never been executed**. Task
11.1 is `plan_agg.py`'s first launch and the `HYDRA_RUN_DIR` hook's first use. The CPU property tests
check the objective's algebra thoroughly and never check that the entry point can start. Task **9.2 —
"Extend the driver-contract test for the new hooks"** was left optional and unwritten; a contract test
on the emitted token would have caught error 1 with no GPU. **Promoting 9.2 out of optional is the
cheapest remaining insurance before the six sweep arms of section 12.**

### 5.1 Hazards (live, not yet triggered)

- **The `plan.py` leg resolves to the recorded baseline cell unless steered away.** The shipped
  `hydra.run.dir` template carries neither the seed nor the weight, so an unoverridden `plan.py` leg
  appends its seed line to the `logs.json` holding 75.33 ± 6.11, and `aggregate_results.py` would turn
  the reference cell into a 4-seed mean **without erroring**. Task 11.1's leg 2 was steered to
  `plan_outputs_gd_scratch/` and verified with a `diff` against the backup.
- **`_agg32_` in `conf/train.yaml`'s `hydra.run.dir` is a hardcoded literal, not an interpolation.**
  The real `agg_out_dim` is **128**, confirmed twice on the pod (`in_dim=1568 out_dim=128`). Every run
  directory claims `agg32` regardless, so two runs differing only in `agg_out_dim` would collide on one
  directory. Not blocking here — nothing sweeps it — recorded because it is live for anything that does.
- **`plan_outputs_gd_scratch/` matches `collect_results.py`'s `plan_outputs_*` glob**, so the paircheck
  cell is visible to the collector as a distinct cell. It cannot collide with a reported one, which is
  what the requirement asked, but **it must not be reported**.
- **The `mpc/` key prefix in `logs.json` is unconditional and does not indicate the setting.** The
  open-loop cell's records are keyed `mpc/success_rate`, and both task-11.1 legs printed
  `MPC iter 0 Eval` while running open-loop with the GD planner. Anyone reading these files directly
  would take open-loop numbers for MPC ones.

---

## 6. The long-horizon reading — RECORDED 2026-08-08, BEFORE TASK 11.4 RUNS

Task 11.4 requires the reading to be written into **both** the manifest and this log *before* the job
runs, not reconstructed afterwards. The manifest half is `PROTOCOL_EXPECTED_SOURCE[(config_name, 50)]`,
which carries the text below into `agg_run_manifest.json` for every long-horizon run. This is the log
half.

**The paper does not state its long-horizon planner settings anywhere.** `paper_tex/sec/1_main.tex`
introduces `L_plan = L_spatial + 0.1 · L_agg` only for "a longer-horizon setting where the target is 50
steps away", and scopes the claim to MPC — "this combined cost improves over using the spatial cost
alone across all models **under MPC**". No open-loop claim is made. The appendix protocol table
(`Subplanner horizon 25`, `# Executed actions 25`, footnoted 5 for MPC) is the **short**-horizon
protocol.

**Reading taken — (a), scale the horizon with the goal distance:**

| field | open-loop | MPC |
|---|---|---|
| `goal_H` | 50 | 50 |
| `planner.sub_planner.horizon` | 50 | 50 |
| `planner.n_taken_actions` | **50** | **5** |

Everything else is the short column unchanged: `n_evals 50`, `objective.alpha 1`, `objective.mode`
last/staged, `max_iter` 1/20, `sub_planner.lr 0.1`, `opt_steps 100`, `sample_type zero`,
`action_noise 0`. `frameskip` stays 5 and every horizon is divisible by it.

**Reading rejected — (b)**, hold `horizon` at 25 and let open-loop cover half the distance. Rejected
because (a) is the only reading under which open-loop is even attempting the task. Recorded because (b)
would **by itself** explain the paper's open-loop collapse to 13.33 against MPC's 24.00, so if the
control reproduces MPC but not open-loop, (b) becomes the live hypothesis and this note is the thing
that stops that from being an after-the-fact rationalisation.

**MPC keeps 5 executed actions at both horizons.** The appendix footnotes 5 independently of the
horizon, and MPC is the setting the paper's claim is actually scoped to. If 11.4 lands on 50 instead,
that is a one-value edit plus its test literal — and it must be recorded here as a changed reading, not
quietly swapped.

**This is a guess either way.** It is written down so it is auditable, not because it is known.

### 6.1 Reference cells to compare against (paper, `+ Proj` row with `L_curv` ✓)

| long-horizon PushT | open-loop | MPC |
|---|---|---|
| spatial only | 13.33 ± 3.77 | 24.00 ± 6.53 |
| combined cost | 20.00 ± 0.00 | 33.33 ± 4.16 |
| **delta to look for** | **+6.67** | **+9.33** |

**What the control can and cannot decide.** A *failure* to reproduce means the wrapper is wrong and the
sweep is unreadable. A *success* only licenses interpreting a flat short-horizon sweep — **it is not
evidence for this arm**, because it reproduces the paper's own cell with the paper's own weight. At one
seed and `n_evals = 50` the binomial SE is ~5–7 points, so a `+9.33` MPC delta is roughly 1.5 SE on a
single seed: the control can detect a gross wiring failure, not a small effect. Task 11.4 runs one seed
first for exactly that reason.

### 6.2 Task 11.3, as shipped

`PROTOCOL_EXPECTED` and `PROTOCOL_EXPECTED_SOURCE` are now keyed on the `(config_name, goal_H)` **pair**,
and `resolve_protocol` resolves `goal_H` out of the config *first*, then selects the column. Selection is
a lookup over `HORIZON_REGIMES = {25: "short", 50: "long"}` rather than a `goal_H != 25` test, so a third
horizon aborts naming the field and the two that exist instead of silently falling into the long column.

**The short-horizon gate is not weakened**, and that is asserted against a literal copy of the pre-task
table in `tests/test_agg_protocol_horizon.py` rather than checked by eye. `goal_H` is now a pinned field,
which **strengthens** it: previously `goal_H` was unconstrained, so a 50-step run satisfied the short
columns on `sub_planner.horizon` alone, and a manifest could not tell a 25-step run from a 50-step one —
which matters now that the tree will hold both.

Also added: a `FRAMESKIP = 5` divisibility self-check on every pinned horizon, run at import as a `raise`
rather than an `assert` so `python -O` cannot strip it. Justification — `PlanEvaluator.__init__`
integer-divides `goal_H`, `n_taken_actions` and `sub_planner.horizon` by frameskip and rejects nothing,
so a non-multiple is silently **truncated** and a column could pin a number the planner never runs.

**Signature change worth a reviewer's eye:** `expected_table` now takes a required second argument and
the two dicts changed key shape. Nothing outside `plan_agg.py` referenced them, and no short-horizon
*value* moved, but this is not purely additive.

Suite after 11.3 and 9.2: **369 passed, 12 skipped, 3 failed** — up from 309 passed; the 3 failures are
the pre-existing CUDA-only `tests/test_vit_sdpa_equivalence.py` cases.

---

## 7. Task 11.4 blocked twice, and the two fixes (2026-08-08)

The Positive_Control's first launch failed for two unrelated reasons. Neither is a result; both are recorded
because they changed how the job is run.

### 7.1 The MPC leg: a command shape the driver cannot express

`SETTINGS=both ... planner.n_taken_actions=50` aborted with a `ProtocolError` on the MPC leg. A positional
override reaches **both** seed loops, but the long-horizon columns pin `n_taken_actions` 50 open-loop and **5**
MPC. The gate did exactly what it exists for: it named the field, the expected value and the resolved one, and
aborted before any load. Task 11.4's own text already authorised the fix — run the settings as separate
`SETTINGS=ol` / `SETTINGS=mpc` invocations — so the control is **four** jobs, and the two MPC legs pass no
`n_taken_actions` override at all and take the shipped 5.

### 7.2 The open-loop leg: reading (a) does not fit the slice

`torch.OutOfMemoryError` at `models/vit.py:100`, the materialised `q @ k.T`. **Confirmed not a leak** before
anything else: the slice showed **16 MiB of 45312 MiB** in use, no live processes, no stray python.

The arithmetic, done before recommending anything, because two unverified arguments already failed on this
project today (`PROGRESS_ACS.md` N-series and finding M2 below):

`VWorldModel._rollout_latents` makes exactly `sub_planner.horizon // frameskip` predictor calls, and
`planning/gd.py` runs one `total_loss.backward()` per optimizer step over the **whole** rollout, so every
call's activations are retained in a single graph and the total scales linearly in the call count — **10 calls
at `goal_H 50` against 5 at `goal_H 25`**. At the Target_Cell shapes (`n_evals 50`, `heads 16`, `depth 6`,
`T = num_frames * num_patches = 3 * 196 = 588`, bf16) the softmax output alone is
`50 * 16 * 588**2 * 2 B = 553 MB` per layer per call. The first two calls are shorter because the context
window is still filling (`T` 196, 392, then 588):

| | retained score matrices | rest (qkv, projection inputs, FFN GELU) | total |
|---|---|---|---|
| `goal_H 25`, 5 calls | `(0.061 + 0.246 + 3*0.553) * 6` = **11.8 GB** | ~9.5 GB | **~21 GB** — fits, and did |
| `goal_H 50`, 10 calls | `(0.061 + 0.246 + 8*0.553) * 6` = **28.4 GB** | ~22 GB | **~50 GB** — exceeds 45312 MiB |

The score matrices are **57%** of what a `T=588` call retains. Removing them puts the long-horizon run at
~22 GB. The margin is not tight, which is why this is the fix rather than a coin flip against reading (b).

### 7.3 The fix, and the deviation it is

`plan_agg.attention_scope(regime)` wraps the single `plan.planning_main` call. `short` yields `materialized`
and **touches nothing at all** — no import, no class-attribute write. `long` enters the existing
`models.vit.sdpa_attention` context manager. **No file is edited:** `models/vit.py` already shipped the
`Attention.use_sdpa` switch and its scoping context manager for the CCR rollout, so `plan.py`'s bytes,
`planning/`, `datasets/` and the Scope_Guard assertion all still hold, and no allowlist entry is added.

**It is keyed on the horizon regime for one reason.** Every recorded number on this platform was measured on
the materialised path — the Platform_Baseline in §3 and §4's paired check, which found the wrapper at `w = 0`
*numerically identical* to frozen `plan.py`. Enabling SDPA everywhere would move the **reported**
short-horizon result onto a path that identity was never established on, with no error and no visible symptom,
because SDPA computes the same function. So the reported result keeps the measured path and only the control
deviates.

Both control arms share the deviation, since both launch through this wrapper at `goal_H 50`, so the
`w=0` → `w=0.1` delta §6.1 asks for is like-for-like. Only its absolute level could carry an implementation
effect, and `tests/test_vit_sdpa_equivalence.py` pins the two branches as agreeing to bf16 rounding.
`attention_impl` and `attention_impl_reason` reach `agg_run_manifest.json` on every run, so the deviation
travels with the numbers rather than living only here.

**Stated before the run, not after: what is not verified.** `scaled_dot_product_attention` is called with an
explicit `attn_mask`, which rules out the flash backend. The saving depends on PyTorch selecting the
memory-efficient backend rather than falling back to `math`, which materialises the same matrix. The CCR arm
ran this path on this pod at *training* shapes, so a fallback is not the expected outcome, but it has **not**
been checked at `n_evals 50`. **A repeat OOM is that diagnosis**, and the next move is then §6's reading (b) —
recorded as **hardware-forced**, not as a re-reading of the paper, since §6 pre-registered (a) and already
noted that (b) would by itself explain the paper's open-loop collapse.

Suite after 11.3b: **383 passed, 12 skipped, 3 failed** (the 3 are the pre-existing CUDA-only SDPA equivalence
cases). `tests/test_agg_long_horizon_attention.py`: 14 passed.
