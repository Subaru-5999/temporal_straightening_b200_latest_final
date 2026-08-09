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
| Task 11.3b — long-horizon attention deviation | **complete** — gated on the horizon regime; the inner-scope override that defeated it is fixed. See §7.3, §7.4 |
| Task 11.4 — Positive_Control (~2.5 GPU-h) | **complete, all 4 runs** — 16.00 / 16.00 open-loop, 16.00 / 16.00 MPC (§8, §8.3, §8.5, §9) |
| Task 11.5 — Positive_Control verdict | **read: FAILED on the decisive MPC leg.** Delta **+0.00** against the paper's +9.33; McNemar p = 1.000 on 4 discordant pairs. See §9 |
| Section 12 — the 6-arm weight sweep (~4 GPU-h) | **BLOCKED by 11.5's pre-registered branch — does not launch.** §9.2 |
| Rung-1 offline checks | **done, 0 GPU-h (§9.4)** — explanation 2 (wrong checkpoint row) **ruled out**; explanation 3 (protocol) **unresolvable from the paper**; and the control is shown to have been **underpowered by design** against the paper's mean MPC effect of +4.00 |
| Next decision | §9.4/§9.5 — the wrapper is now validated by four things other than the control, but 11.5's branch was pre-registered. **Awaiting an explicit recorded choice (Requirement 11.7)** |
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

### 7.4 The scope was defeated by an inner scope — error 3, and a prediction of mine that was wrong

**The run OOM'd again, and the "not verified" clause above did not explain it.** Recorded plainly because the
clause was a pre-registered diagnosis and it was the wrong one.

The second attempt printed `[plan_agg] attention implementation: sdpa (long horizon, goal_H=50)`,
`protocol_ok=True`, `horizon=long`, wrote its manifest, and then died in **13 seconds** at
`models/vit.py:100` — `dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale`, which is the **`else`**
branch. So `Attention.use_sdpa` was `False` at the predictor call. Not a backend fallback: the fast branch was
never entered at all.

**Cause.** `VWorldModel._predict_maybe_checkpointed` wraps every `predict` in
`with sdpa_attention(fast_attention)`, and `_rollout_latents` defaulted `fast_attention=False`. So each of the
ten predictor calls opened an **inner** scope that set the switch to an absolute `False`, overriding the outer
scope `plan_agg.attention_scope` had just opened. The general shape: *a context manager that sets an absolute
value cannot be nested by a caller that only wants to opt in.* The inner scope was written for a good reason —
the CCR docstring explains it must sit inside the checkpointed callable, because a checkpointed segment is
recomputed in backward long after an enclosing `with` has exited — and that reason is untouched by the fix.

**Fix.** `VWorldModel.resolve_fast_attention`: `fast_attention=None` now means **inherit** the ambient
`Attention.use_sdpa`, and it is the default for `_predict_maybe_checkpointed` and `_rollout_latents`. The
ambient value is read at *forward* time and passed on as an explicit bool, so the forward/backward pinning the
CCR docstring depends on still holds. `compute_ccr` is the only caller that passes an explicit value
(`fast_attention=self.ccr_fast_attention`) and is unaffected; the class switch still ships `False`, so
`rollout`, `plan.py`, `planning/*` and `Trainer.openloop_rollout` take the materialised path exactly as before.
`models/visual_world_model.py` was already an allowlist member, so the Scope_Guard needs no new entry, and
`models/vit.py`, `plan.py`, `planning/` and `datasets/` are still untouched.

**Why the 11.3b tests did not catch it.** They pinned the gating — short stays `materialized`, long enters the
scope, the scope restores on exception, the manifest records the choice — and every one of those statements was
true. None of them pinned the **composition**: that the switch the scope sets is the switch the predictor
*reads*. The 14 tests tested my half of the mechanism and stopped at the seam. Now closed by six more, the
load-bearing ones being `test_predict_observes_the_ambient_switch_when_not_asked` (drives the real
`_predict_maybe_checkpointed` against a `predict` spy and asserts the observed branch equals the ambient one)
and `test_checkpointed_recomputation_is_pinned_to_the_forward_branch` (asserts the backward recomputation runs
on the branch the forward took, after the scope has exited). The spy's `predict` returns `h * h` over an
intermediate on purpose: with `z * 1` nothing inside the segment is saved, no recomputation runs, and the test
would have passed vacuously.

| # | date | error | cost | how it was caught |
|---|---|---|---|---|
| 3 | 2026-08-08 | **An inner `sdpa_attention(False)` scope silently overrode the outer one**, so the long-horizon deviation had no effect and the run OOM'd in the materialised branch anyway. My §7.3 "a repeat OOM is a `math`-backend fallback" prediction was **wrong** — the traceback line number (`vit.py:100`, the `else` branch) is what ruled it out immediately | 13 s, ~0.004 GPU-h | The second task-11.4 launch. **Why 11.3b's own tests missed it:** they covered the gating end of the mechanism and never that the flag the scope writes is the flag the predictor reads. Fixed with `resolve_fast_attention` and six composition tests |

Suite after the fix: **389 passed, 12 skipped, 3 failed** (the same pre-existing CUDA-only cases).
`tests/test_agg_long_horizon_attention.py`: 20 passed.

**The §7.2 arithmetic is still unconfirmed.** It predicted ~50 GB against a 45312 MiB slice on the materialised
path, and both OOMs are consistent with it, but no run has yet demonstrated that removing the score matrices is
sufficient — because no run has yet actually removed them. The next launch is the first real test of both the
estimate and the fix, and the §7.3 backend-fallback clause remains live and unverified rather than refuted.

---

## 8. Task 11.4 — Positive_Control, arm A open-loop: RAN (2026-08-08)

First long-horizon measurement on this platform. Arm A is the **spatial-only** leg (`w = 0`), so this is the
control's reference point, not a result about `L_agg`.

| field | value |
|---|---|
| arm / setting / seed | `agg_weight=0` / open-loop / 100 |
| protocol | `protocol_ok=True`, `horizon_regime=long`, `goal_H 50`, `sub_planner.horizon 50`, `n_taken_actions 50` |
| attention | `attention_impl=sdpa` (the §7.3 deviation, now actually in effect) |
| **success rate** | **0.16** = 8/50, binomial SE **~5.2 points** |
| successful episodes (0-indexed) | 6, 28, 29, 31, 37, 39, 41, 48 |
| `perform_planning_s` | 210.03 (1.98 s per GD step) |
| records | `outcome_rows=2` (`plan0`, `output_final`), `record_failures=0` |

**Against the paper's cell.** `+ Proj` with `L_curv` ✓, long-horizon PushT, spatial only: **13.33 ± 3.77**
open-loop. Measured **16.00**. That is inside 1 SE of the paper and inside the tolerance task 11.5
pre-registered before any of this ran ("a spatial-only baseline of, say, 16.00 is not a failure to reproduce").
It is also the collapse the paper reports: 16.00 at `goal_H 50` against this platform's 75.33 ± 6.11 at
`goal_H 25`, on the same checkpoint and the same planner. **Reading (a) reproduces the regime**, which is the
one thing the spatial-only arm could tell us and the reason §6 recorded the reading in advance.

**What it does not tell us.** Nothing about `L_agg`: that is the `w=0` → `w=0.1` delta, and three of the four
control runs are still outstanding. A single arm cannot be read as reproduction of the paper's *claim*.

### 8.1 The §7.2 memory estimate, now supported

The run fit and completed. The estimate said ~50 GB materialised against a 45312 MiB slice, and ~22 GB with the
score matrices removed; the materialised path OOM'd twice and the SDPA path completed. **That is directional
support, not a measurement** — nothing sampled peak allocation, so the ~22 GB figure itself is still unverified
and the true peak could be anywhere below the slice. The §7.3 backend-fallback clause is now **refuted for
these shapes**: had PyTorch fallen back to the `math` backend, the run would have OOM'd exactly as before.

Timing corroborates the mechanism independently: 1.98 s per GD step against 0.69 s at `goal_H 25` is 2.87x for
2x the predictor calls, which is the shape expected when the per-call cost is no longer dominated by
materialising and re-reading a 553 MB score matrix per layer.

### 8.2 Two corrections carried forward

- **My pre-registered diagnosis was wrong** and is now settled: the second OOM was the inner-scope override of
  §7.4, not a backend fallback. Recorded there; repeated here because §7.3 stated the wrong prediction with
  confidence and a reader arriving at §8 should not have to infer that it was superseded.
- **The §7.2 arithmetic was right about the ordering** (25 fits, 50 does not, attention is the dominant term)
  but is still not confirmed numerically. It is being carried as supported-not-verified.

### 8.3 Arm A MPC — RAN (2026-08-08)

| field | value |
|---|---|
| arm / setting / seed | `agg_weight=0` / MPC / 100 |
| protocol | `protocol_ok=True`, `horizon_regime=long`, `goal_H 50`, `sub_planner.horizon 50`, `n_taken_actions 5`, `max_iter 20`, `objective.mode staged` |
| attention | `attention_impl=sdpa` |
| **success rate** | **0.16** = 8/50, binomial SE **~5.2 points** |
| successful episodes (0-indexed) | 1, 10, 28, 29, 37, 41, 45, 48 |
| `perform_planning_s` | 4096.8 (68 min; 20 replans at ~197 s each) |
| records | `outcome_rows=21` (`plan0`-`plan19` + `output_final`), `record_failures=0` |

Paper cell: **24.00 ± 6.53**. Measured **16.00** — 8 points low, ~1.2 SE at one seed and inside the paper's own
3-seed spread. Per §6.1 and task 11.5 the pass condition is the *delta*, not the absolute, so this is the
control's reference point and not a failure to reproduce. The `n_taken_actions 5` in the manifest is the
confirmation that the separate `SETTINGS=mpc` invocation used the shipped MPC value; the combined
`SETTINGS=both` form is what the §7.1 `ProtocolError` was protecting against.

**Two details worth keeping.** MPC and open-loop landed on the *same rate* from **different episodes**: 5 of
the 8 overlap (28, 29, 37, 41, 48), open-loop won 6/31/39 alone, MPC won 1/10/45 alone. And MPC's trajectory
was flat at 0.00 through iteration 7, first success at iteration 8, then **plateaued at 0.16 from iteration 15
to 19** — the last five replans bought nothing. Both are recorded because they are the kind of structure a
single scalar hides, and disaggregating is what caught the sign errors in `PROGRESS_ACS.md` and
`PROGRESS_MCA.md`.

### 8.4 The term-magnitude ratio — the number the paper never reports

From arm A open-loop's `agg_instrumentation.json` (`agg_weight=0`, so L_agg is *measured but not in the sum*):

| | `l_spatial` | `l_agg` | raw `l_agg / l_spatial` | contribution at the paper's `w = 0.1` |
|---|---|---|---|---|
| step 0 (0 updates) | 0.95589 | 0.14759 | 0.1544 | **1.5%** |
| step 99 (99 updates) | 0.12196 | 0.023866 | 0.1957 | **2.0%** |

`step_boundary_mismatch: false`, `record_failures: 0`, and the ratio field is `0.0` at `w = 0` with the raw
L_agg still recorded — Requirement 5.5 observed on real tensors.

**What this predicts, written down before arm B runs.** At the paper-literal `w = 0.1` the aggregated term is
**1.5-2.0% of the spatial term**. If arm B moves the success rate materially, a ~2% perturbation of the
objective did it, which would be a strong and slightly surprising claim about the *conditioning* of the
landscape rather than about the size of the term. If arm B is flat, "the term was too weak to matter at this
weight" is the reading the instrumentation already supports, and that is exactly the distinction task 11.5 says
this number exists to make.

**What it says about the pre-registered Sweep_Grid, which is not what I would have guessed.**
`SWEEP_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)` spans contributions of roughly **0.15% to 59%** of L_spatial
(using the step-99 ratio). So the grid brackets *negligible* through *comparable* — it **never reaches the
regime where L_agg dominates**, which would need `w ≈ 5-6.5`. The grid is therefore one-sided with respect to
the failure mode task 11.5 names ("the term dominated and broke the planner"): a flat sweep across all six
weights could not distinguish "no effect anywhere" from "the useful weight is above 3.0".

**This is recorded, not acted on.** The grid and `AGG_WEIGHT_MAX = 3.0` were pre-registered in task 2.1 before
any of this was measured, and changing them now — after seeing the ratio — is precisely the §0 failure mode
that `PROGRESS_MCA.md` §7.2 caught me at. Any extension needs the Requirement 11.7 recorded approval **before**
a job runs, and task 12.8 already handles the boundary-selection case. Caveat on the number itself: it comes
from the `w = 0` run, so the optimizer trajectory is the spatial-only one; at nonzero weight the ratio can
drift because the actions being optimized differ.

### 8.5 Arm B open-loop — RAN (2026-08-09): delta exactly 0.00, and the reason is visible

| field | value |
|---|---|
| arm / setting / seed | `agg_weight=0.1` / open-loop / 100 |
| objective rewrite | `agg_weight=0.1, enabled=True` — `L_agg` **is** in the sum |
| protocol / attention | `protocol_ok=True`, `long`, `goal_H 50`, `n_taken_actions 50` / `sdpa` |
| **success rate** | **0.16** = 8/50 |
| successful episodes | 6, 28, 29, 31, 37, 39, 41, 48 |
| `perform_planning_s` | 209.60 (arm A: 210.03 — the <0.1% overhead claim holds at long horizon too) |

**The success vector is identical to arm A's, element for element.** Paired counts (Requirement 11.4):
candidate-only **0**, baseline-only **0**, matching **50**. Open-loop delta = **+0.00** against the paper's
**+6.67**.

**This is not a no-op, and that distinction is the finding.** `state_dist` differs on **50 of 50** episodes, so
the term genuinely changed the optimized actions everywhere:

| | arm A (`w=0`) | arm B (`w=0.1`) | delta |
|---|---|---|---|
| mean `state_dist` | 80.656 | 82.188 | **+1.532** (worse) |
| mean over the 8 successes | 33.132 | 33.904 | +0.772 |
| mean over the 42 failures | 89.708 | 91.385 | +1.676 |
| episodes B ended closer / farther | — | — | **19 / 31** |
| median \|delta\| / max \|delta\| | — | — | 0.628 / 55.691 (episode 3) |

So `L_agg` perturbs every plan, usually by very little (median 0.63 on distances averaging ~81), occasionally a
lot (episode 3 moved 55.7 — a failure in both arms either way), and **never by enough to cross a success
threshold**. The direction is slightly unfavourable: 31 of 50 episodes ended farther from the goal, which at
n=50 is p≈0.14 two-sided and therefore **not** a significant degradation — it is noise around zero, recorded so
the "slightly worse" is not later upgraded into a claim.

**The §8.4 prediction held.** It was written before this run: at `w = 0.1` the term contributes 1.5-2.0% of
L_spatial, so "if arm B is flat, *the term was too weak to matter at this weight* is the reading the
instrumentation already supports." Arm B is flat to the episode. That is one pre-registered prediction
confirmed, against the two of mine that failed today (§7.3's backend diagnosis, §7.2's still-unverified
arithmetic).

**What this does and does not decide.** It does **not** decide the Positive_Control. The paper makes **no**
open-loop claim at long horizon — its claim is scoped to MPC — and two of its four open-loop combined-cost
cells are worse than spatial-only. The decisive leg is arm B MPC, which is the only one of the four that tests
what the paper actually asserts.

**A prediction for section 12, recorded now so it is not hindsight.** If `w = 0.1` cannot flip a single episode
out of 50 at long horizon, the sweep's two smallest weights (0.01, 0.03 — contributions of ~0.2% and ~0.6%)
are very likely to return the Baseline_Arm's vector exactly, making 2 of the 6 arms uninformative by
construction. Caveat: that is an extrapolation *across regimes*, and the short horizon has a different loss
scale and a far higher baseline, so it is a prediction to check rather than a reason to change the grid. The
grid stays as pre-registered.

---

## 9. Task 11.5 — Positive_Control verdict: **FAILED on the decisive leg. Section 12 does not launch.**

All four runs complete, seed 100, `n_evals = 50`, `goal_H = 50`, reading (a), `attention_impl=sdpa`,
`protocol_ok=True` on every run. ~2.5 GPU-h.

| long-horizon PushT, seed 100 | spatial only (`w=0`) | combined (`w=0.1`) | **our delta** | paper's delta |
|---|---|---|---|---|
| open-loop | 16.00 | 16.00 | **+0.00** | +6.67 |
| **MPC** | **16.00** | **16.00** | **+0.00** | **+9.33** |

Binomial SE ~5.2 points per cell. Paper reference: 13.33 ± 3.77 / 24.00 ± 6.53 spatial-only,
20.00 ± 0.00 / 33.33 ± 4.16 combined.

### 9.1 The paired structure, which says more than the equal rates

| | open-loop | MPC |
|---|---|---|
| candidate-only wins | **0** | **2** (episodes 3, 42) |
| baseline-only wins | **0** | **2** (episodes 41, 48) |
| matching outcomes | **50** | **46** |
| McNemar exact, two-sided | — | **p = 1.000** |
| mean `state_dist` | 80.66 → 82.19 (+1.53) | 120.58 → 133.43 (+12.85) |
| episodes ending closer / farther | 19 / 31 | 21 / 29 |

**Open-loop is a bit-identical outcome vector.** MPC does move outcomes — but only **4 of 50 episodes changed
at all**, and they split exactly 2-2. That is the informative part: a true +9.33 effect is ~4.7 net episodes out
of 50, which would require roughly 5-6 candidate-only wins against ~1 the other way. Observing a total of four
discordant pairs, evenly split, means the term barely reaches the decision boundary in either direction. So
this is not "a noisy null" — it is a null with a visible mechanism.

**Direction, stated with its uncertainty.** Both settings drift slightly *worse* in mean final distance
(+1.53 open-loop, +12.85 MPC) and slightly more episodes end farther from the goal (31/50 and 29/50, p≈0.14 and
p≈0.32 two-sided). Neither is significant. This is noise around zero and must not be reported as a
degradation.

### 9.2 The verdict, per the branch pre-registered in task 11.5 before any of this ran

Task 11.5's text: *"MPC delta near zero or negative: the wrapper does not reproduce the paper's own result, so a
null at short horizon would be uninterpretable — it could be our plumbing. **Do not proceed to task 12.**"*

**Honored. Section 12 is blocked and no sweep job runs.** The whole reason the Positive_Control was worth
~2.5 GPU-h is that it makes the short-horizon sweep readable, and it has come back saying the sweep would not
be. Reading the sweep anyway would be the §0 failure mode — deciding what the gate means after seeing which way
it fell.

**What this does NOT establish.** It is not evidence that the paper is wrong. One seed, `n = 50`, a checkpoint
that is our own reproduction rather than the paper's artifact, and a long-horizon planner protocol the paper
never states (§6, reading (a), recorded in advance as a guess). Any of those can account for a missing +9.33.

### 9.3 The four candidate explanations, and what each would cost to check

Ranked by what the evidence already points at, not by convenience.

1. **The term is too weak at `w = 0.1` — the leading hypothesis, and already measured.** §8.4: `0.1 · L_agg` is
   **1.5-2.0%** of `L_spatial`. A 2% perturbation moved every plan slightly and flipped 4 of 50 outcomes
   symmetrically, which is exactly what a too-weak term looks like. Cost: **already paid**. Note the
   uncomfortable implication — if this is the explanation, the paper's own `w = 0.1` should not have produced
   +9.33 on their platform either, unless their `L_agg`/`L_spatial` scale ratio differs materially from ours.
   That is a checkable claim about their setup, not about ours.
2. **Wrong `agg` head / wrong checkpoint row.** Whether `pusht_aggmlpcos1e-1_agg32_projchannel_dim8_hw14` is
   the cell the paper's `+ Proj` with `L_curv` ✓ row was measured on. Cost: **offline, minutes** —
   `REPRODUCTION.md` plus the run-directory naming.
3. **Wrong long-horizon protocol.** §6 rejected reading (b) in advance. Our spatial-only open-loop of 16.00
   against the paper's 13.33 is a good match, which *supports* (a) for open-loop; but our MPC spatial-only is
   16.00 against 24.00, **8 points low**, which is the one place our reproduction of the paper's own
   spatial-only row is weakest. Cost: **offline to re-read, ~2.4 GPU-h to re-run a variant.**
4. **A real one-seed miss.** The paper's MPC delta is ~1.5 SE at one seed, so a single seed cannot exclude it.
   Cost: **~4.8 GPU-h** for seeds 200 and 300 across all four runs.

Task 11.5 also pre-registered that an ambiguous control is resolved by adding the two remaining Reporting_Seeds
**or** by recording it as ambiguous — never by reading a one-seed delta as confirmation. Per
`SHORT_BUDGET_PILOTS.md` §1, options 2 and 3 are rung-1 offline checks and come before either GPU option.

**Nothing further runs until the branch is chosen and recorded (Requirement 11.7).**

### 9.4 Rung-1 offline checks (2026-08-09, 0 GPU-h) — explanation 2 ruled out, explanation 3 unresolvable, and a power problem in our own control

**Explanation 2 — wrong `agg` head / wrong checkpoint row: RULED OUT.** `REPRODUCTION.md` §2a and §3 use
`pusht_aggmlpcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05` as *the* PushT **✓ run** — `encoder=dino_channel`
(DINOv2 patch + proj 14×14×8), `training.straighten=aggcos1e-1`, `encoder_lr=1e-5`, 2 epochs — whose paper
target is 77.33 / 85.33. That is the `+ Proj` with `L_curv` ✓ row. Right checkpoint, right row, right training
recipe. No further doubt here.

**Explanation 3 — the long-horizon protocol: confirmed unresolvable from the paper.** `paper_tex/sec/1_main.tex`
Long-horizon paragraph read in full: it says only *"a longer-horizon setting where the target is 50 steps away"*
and gives **no planner value at all**. `paper_tex/sec/2_appendix.tex` Table (`Subplanner horizon 25`,
`# Executed actions 25`, footnoted *"This is for open-loop. If using MPC, we execute the first 5 actions"*) is
the short-horizon protocol, exactly as §6 recorded. So reading (a) stays a recorded guess; it cannot be
promoted or refuted by reading, only by spending GPU on reading (b). §6's pre-registration stands unamended.

**The finding that actually matters — our paired SE, and the paper's own effect-size distribution.**

I quoted ~5.2 points per cell earlier. That is the *marginal* SE and it is the wrong statistic here: both arms
draw identical episodes at the same seed, so the delta is a **paired** quantity and its SE comes from the
discordant pairs alone.

| our delta | discordant | SE(delta) | 95% CI | paper's value | verdict |
|---|---|---|---|---|---|
| open-loop **+0.00** | 0 of 50 | — | ~[−6.00, +6.00] (rule of three) | +6.67 | just **outside** |
| MPC **+0.00** | 4 of 50 (2-2) | **4.00** | **[−7.84, +7.84]** | +9.33 | **2.33 SE, p = 0.020** |

So the control is *sharper* than I said: it gives moderate evidence against a `+9.33` effect on this
checkpoint, not merely "a null".

**But now the paper's own table, all four combined-cost cells, with SE of the 3-seed mean:**

| cell | spatial → combined | delta | SE | |delta|/SE |
|---|---|---|---|---|
| PushT `+ Proj` ✓ **MPC** | 24.00 → 33.33 | **+9.33** | 4.47 | **2.09** |
| PushT ResNet ✓ MPC | 33.33 → 36.00 | +2.67 | 4.20 | 0.64 |
| Medium `+ Proj` ✓ MPC | 88.00 → 92.00 | +4.00 | 3.59 | 1.11 |
| Medium ResNet ✓ MPC | 98.67 → 98.67 | **+0.00** | 1.28 | 0.00 |
| PushT `+ Proj` ✓ OL | 13.33 → 20.00 | +6.67 | 2.18 | 3.06 |
| PushT ResNet ✓ OL | 10.67 → 13.33 | +2.67 | 3.93 | 0.68 |
| Medium `+ Proj` ✓ OL | 68.00 → 66.67 | **−1.33** | 6.63 | 0.20 |
| Medium ResNet ✓ OL | 76.00 → 68.67 | **−7.33** | 4.47 | 1.64 |

**MPC deltas: mean +4.00**, range +0.00 to +9.33 — **one** cell clears 2 SE and three sit inside noise, one of
them an *exact tie at ceiling* (98.67 → 98.67), which the paper's phrase "improves over using the spatial cost
alone across all models under MPC" does not describe. **Open-loop deltas: mean +0.17**, two of four negative.

**Three consequences, and the third is about our own design.**

1. Our MPC CI [−7.84, +7.84] **contains the paper's mean MPC effect of +4.00**. So our result is inconsistent
   with the single largest cell in their table and perfectly consistent with their typical one. We happened to
   target the outlier.
2. Our open-loop **+0.00** is dead-centre of the paper's own open-loop distribution (mean +0.17). On open-loop
   our result **agrees** with their evidence taken as a whole, and disagrees only with the one PushT cell.
3. **Recorded against myself:** task 11.4 designed a control whose only detectable target was the paper's
   largest and least typical cell. At a paired SE of 4.00, `+4.00` — their mean effect — is a 1.0 SE
   quantity, i.e. undetectable. Adding seeds 200 and 300 (the option 11.5 offers) cuts SE to ~2.3, making +4.00
   a ~1.7 SE quantity: **still underpowered**, for ~4.8 GPU-h. Reaching 80% power on +4.00 needs SE ≈ 1.4, so
   roughly 9 seeds or `n_evals ≈ 400` — on the order of **15 GPU-h of MPC legs**. The control as specified could
   never have validated the wrapper against the effect size the paper actually reports, and I did not notice
   that when writing §6.1's "the control can detect a gross wiring failure, not a small effect" — which said the
   right thing without following it to its conclusion.

### 9.5 What the wrapper is now validated by, independent of the control

The concern 11.5 raised was "the wrapper is subtly wrong somewhere the CPU property tests cannot reach". Four
pieces of evidence now bear on that directly, none of which is the control:

1. **Task 11.1**: at `w = 0` the wrapper is *numerically identical* to frozen `plan.py` — same 50-element
   boolean vector, `state_dist` equal to 8 significant figures. The plumbing around the objective is exact.
2. **`L_agg` is minimised by the optimizer**: 0.1476 → 0.0239 across the 100 GD steps, an **84% reduction**,
   against L_spatial's 0.956 → 0.122 (87%). A malformed or wrongly-targeted term would not descend comparably
   to the term the planner is built around. This is the check that most directly rules out a sign or
   wrong-goal error, and it comes free from `agg_instrumentation.json`.
3. **The term reaches the actions**: at `w = 0.1`, 50 of 50 episodes have different `state_dist`, and 4 of 50
   MPC outcomes flip. It is live, differentiable and consequential.
4. **Properties 1 and 3** (tasks 4.2, 4.3): bitwise-zero identity over `inf`/`nan`/denormals, and exact reuse
   of the frozen staged dispatch and per-frame coefficients.

So "our plumbing is broken" is now a **narrow** residual risk rather than the leading hypothesis. The leading
hypothesis is §9.3 item 1 — the term contributes 1.5-2.0% of the objective at `w = 0.1` — combined with the
newly-quantified fact that the paper's own typical effect is +4.00 and our control could not have seen it.

**This is a post-hoc re-argument and is marked as such.** It was assembled after seeing the control fail. It
does **not** by itself unblock section 12: 11.5's branch was pre-registered and only an explicit recorded
approval (Requirement 11.7) may override it. Recording the argument is legitimate; acting on it silently would
be the §0 failure mode.
