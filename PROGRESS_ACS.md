# PROGRESS — Action-Conditioned Straightening (ACS)

Live state of the ACS effort. Written so the work can be resumed cold, without the conversation
that produced it. Update this file at every decision point.

Spec: `.kiro/specs/action-conditioned-straightening/` (`requirements.md`, `design.md`, `tasks.md`)
Repo: https://github.com/Subaru-5999/temporal_straightening_b200_latest_final (branch `main`)
Sibling records: `PROGRESS_CCR.md` (the completed negative round this file's conventions come from),
`.kiro/specs/temporal-metric-regularization/design.md` (TMR, **on hold**; ACS supersedes it)

---

## 0. What this file is, and why it exists in this order

**This is a pre-registration.** Sections 4 (Stage-0 rules A and B), 5 (early-read gate checks 0, 1,
1b, 1c, 2a, 2b, 3), 6 (the acceptance bars), 7 (the recorded limitations), 8 (the novelty
positioning) and 9 (the probability estimate) are written **before any ACS measurement is taken** —
before the Stage-0 probe runs, before the loss term exists in the code, and before any GPU time is
spent. Section 10 holds the empty slots the measurements go into.

The reason is a documented failure mode from the previous round, not a stylistic preference. In CCR,
`rho = 0.05` was **derived rather than measured**, the probe failed against it, and the criterion was
then widened (`PROGRESS_CCR.md` §5a); the λ-selection rule had to be corrected after the fact because
it named the wrong quantity (§6a); and the loss shares were called "converged" off two data points and
kept moving for another 120,000 steps (§4). Each of those is a rule meeting its data and losing. The
countermeasure is mechanical: write the rule down first, in a file the scope guard tracks, and let a
tool evaluate it afterwards.

Created 2026-08-08, at repo revision `d3c3ce5`, **before Stage 0 was run.**

Requirements this file discharges: 16.1, 16.3, 16.10, 2.1, 2.17, 10.1, 3.1, 3.2, 3.3, 3.4, 3.5, 3.7,
3.8, 3.9, 14.7.

---

## 1. What ACS is, in one paragraph

A change to the **reduction** inside the paper's existing curvature term — not a new loss term. The
paper penalizes `c_t = 1 - cos(v_t, v_{t+1})` uniformly over every latent-velocity triple; ACS
replaces the uniform mean with a **gate-weighted mean over the same per-triple values**, where the
gate measures how similar the controlling actions were:

```
L_acs = Σ_t w_t · c_t / clamp_min(Σ_t w_t, 1e-3),   c_t = 1 - cos(v_t, v_{t+1}),
w_t   = relu(cos(a_t, a_{t+1})).detach() ∈ [0, 1],  a_t = Σ_{s=0}^{4} act[:, t, 2s:2s+2]
```

The space (the paper's 128-d aggregated space via `encoder.agg`), `λ = 0.1`, and the
`step_thresh = 1e-6` static-velocity mask are all **unchanged**. The gate is computed from the **raw
`act` tensor** and detached, so nothing the encoder or the trained `action_encoder` learns can move it.

The gap targeted is in the paper's **premise**, not its formula: perceptual straightening is a
hypothesis about *passive natural video*, where there is no controller and smoothness is the right
prior. The paper transplants it to an actively controlled agent trained on random, suboptimal
rollouts, where a direction change in the latent velocity is often the *correct* representation of a
change in action. Selected by the mode string `training.straighten=acsaggcos1e-1`; the default path
(`aggcos1e-1`, and `False`) stays **bitwise** the pre-feature code.

Because the weighted mean is invariant to any uniform rescaling of the gate (`w ≡ ŵ > 0` gives
`L_acs = L_curv` exactly), ACS can only **reallocate** straightening pressure, never reduce it in
aggregate. That is what makes the λ-matched control the **existing baseline** at zero cost, and what
makes "you just found a smaller λ" unavailable as an objection.

---

## 2. Status

| item | state |
|---|---|
> **THIS FEATURE IS CLOSED. Stage 0 returned STOP on rule A (§10.2). `compute_acs` was never written
> and will not be. Total GPU-hours spent: 0. The return is findings N1 and N2 (§10.4). Next arm:
> `MCA_Fallback` (§12).**

| item | state |
|---|---|
| Pre-registration (this file, §4-§9) | **complete — written 2026-08-08 before any measurement** |
| Scope guard extended for the ACS file set (task 1.1) | complete — `PROGRESS_ACS.md` allowlisted, `models/vit.py` + `models/dino.py` frozen |
| Shared geometry helpers + `straighten` parser `else: raise` (section 2) | complete — bitwise-neutral refactor; **kept**, the parser fix closes a landmine independent of ACS |
| `reduce_action` / `action_gate` (section 3) | complete, with properties 3, 4, 5 green; **kept** as the probe's instrument |
| Stage-0 probe readout `--readout actions` + verdict rules (tasks 4.1-4.5) | complete; property 19 verified bitwise against the real datasets |
| **Stage 0 measurement (task 5.1) — CPU, minutes, 0 GPU-h** | **RUN 2026-08-08 — see §10.1.** 3 of 4 envs; `point_maze_medium` dataset absent from the pod |
| **Stage-0 verdict (rules A and B, task 5.2) — CAN KILL THE FEATURE** | **READ 2026-08-08: rule A STOP (clause 2.6), rule B GO, combined STOP — see §10.2. It killed the feature.** |
| ACS term `compute_acs` (section 6) | **NOT WRITTEN and will not be** — Requirement 2.12/2.13 |
| Stage 1 arm (8,000 steps, 0.8 GPU-h) | **not launched** — Stage 1 not permitted |
| Early-read gate verdict | **not reachable** — no arm exists; §10.3 stays empty |
| Stage 2 full run + 3-seed eval (13.6 GPU-h) | **not launched** |
| Findings N1 / N2 / N3 | **N1 and N2 written — see §10.4.** N3 not reachable without a trained arm |

**Budget, recorded honestly.** Stage 0 minutes / **0 GPU-h**. Stage 1 arm 0.8 GPU-h; matched 8k eval
0.4; permuted-gate arm 0.8; Stage 2 full run + 3-seed eval 13.6. **Best case (Stage-0 STOP) 0 GPU-h;
typical case ~0.8-1.2 GPU-h; worst case ~16 GPU-h.** CCR spent ~26 GPU-h to reach a negative result;
the asymmetry here is the design.

---

## 3. The control reference row (measured, do not re-derive)

Every Stage-1 comparison is read against these numbers. They come from the completed baseline run
(`model_2.pth`, 123,858 steps, 12.04 h) and its **bitwise** 8,000-step prefix in `checkpoints_ctrl8k`
(40/40 telemetry rows agree to `+0.000000` on re-run — training on this pod is bitwise deterministic,
so the matched control is *exact* and there is no run-to-run variance to subtract).

**`global_iter` 8000, the row the gate is judged against:**

| term | scaled | share |
|---|---|---|
| curvature | 0.041421 | **73.741%** |
| prediction | **0.013196** | 23.493% |
| decoder | 0.001554 | 2.767% |
| **total** | **0.056171** | 100% |

**Step rate:** median **2.862** it/s over 619 telemetry records (a 50-step smoke read 1.890 it/s, so
warmup rows are artifacts and must not be used).

**Success rates.** Baseline @124k, 3 data-sampling seeds (100/200/300), `n_evals=50`:

| setting | measured | per-seed | paper |
|---|---|---|---|
| open-loop | **75.33 ± 6.11** | 74, 82, 70 | 77.33 ± 6.18 |
| MPC | **82.00 ± 2.00** | 82, 80, 84 | 85.33 ± 4.99 |

Control **@8k**: **16.0** OL / **18.0** MPC.

Read the open-loop per-seed spread — **74, 82, 70**, a 12-point range over three seeds on the *same
checkpoint* — as the noise reality every claim in this file lives in. Table 1's straightening gains
(`L_curv` ✗ → ✓, open-loop) that rule A is set against: UMaze **+50.00**, Medium **+10.67**, Wall
**+10.67**, PushT **+7.33**.

**Share drift, recorded because calling shares "converged" off two points was a documented CCR
error:** curvature share 31.4% @200 → 65.4% @3000 → **73.7% @8000** → 80.5% @35.6k → 79.5% @84.4k →
82.7% @123.9k. Not monotone, plateaus near 80%. The reference is read at 8,000 and nowhere else.

---

## 4. Stage 0 — the pre-registered verdict rules. WRITTEN 2026-08-08, BEFORE THE DATA

Stage 0 measures the distribution of consecutive-action similarity `cos(a_t, a_{t+1})` across **all
four** datasets (PushT, Wall, PointMaze-UMaze, PointMaze-Medium), on the **train** split (validation
reported as a cross-check), for **all three** action reductions (`sum` ≡ `mean`, `raw`, `first`).
CPU only, minutes, **no GPU, no checkpoint, no model weights, no video decode**.

Reported per environment per reduction: mean, median, `frac(cos<0)`, `frac(cos<0.5)`, a 20-bin
histogram over `[-1, 1]`, `mean(w)` for `w = relu(cos)`, `frac(w=0)`,
**`R = E|w − E[w]| / (2·E[w])`**, and `n_triples` / `n_windows` beside every statistic so no number
appears without its denominator.

`mean(w)` is deliberately **not** a gate statistic. Because ACS uses a weighted mean, a flat gate at
*any* level reproduces the baseline exactly — `w ≡ 0.5` everywhere gives `L_acs = L_curv`. `R` is the
weight mass moved relative to uniform (the population form of the total-variation distance between
`w/Σw` and `1/N`), and it is the quantity that gates. `R = 0` means ACS *is* the baseline regardless
of `mean(w)`.

### 4.1 Rule A — the mechanism-ordering test

If straightening helps most where the control is smooth, `frac(cos<0)` should order **inversely** to
Table 1's gains: UMaze lowest, Medium ≈ Wall in the middle, **PushT highest**.

| outcome | verdict |
|---|---|
| PushT's `frac(cos<0)` is the **highest** of the four **AND** exceeds each of the Wall / UMaze / Medium values by **>= 1.5x** **AND** UMaze's value is the **lowest** of the four | **GO** — the ordering is consistent with the mechanism story; build ACS and make the (weak) mechanism claim |
| PushT is highest **but** the remaining ordering inverts (e.g. Wall > Medium, or UMaze not lowest), **or** PushT's margin over the largest of the other three is in **`[1.1x, 1.5x)`** | **MIDDLE** — build ACS, but the mechanism claim is **downgraded** to "the gate is a useful inductive bias", and the writeup must **not** claim ACS explains the Table 1 gain ordering |
| PushT's `frac(cos<0)` is **not the highest** of the four, **or** is within **1.1x** of the smoothest environment's value | **STOP** — the premise is dead, the mechanism story is wrong, **the feature is not built** |

**Additional pre-declared downgrade (Requirement 3.6).** If the `sum` reduction shows **no** reversal
structure while `raw` **does**, the reversals are happening *inside* a latent step — which the latent
velocity cannot see either — and the rule A verdict is recorded as **MIDDLE**, not GO.

### 4.2 Rule B — the reallocation test (independent, and it can STOP on its own)

| PushT `R` | verdict |
|---|---|
| **`R >= 0.15`** | **GO** on rule B |
| **`0.08 <= R < 0.15`** | **MIDDLE** — the expected effect size is small; ACS may be built, and **`acs_gate=hard` or a sharpened gate is the pre-declared remedy**, recorded now rather than invented later |
| **`R < 0.08`** | **STOP** — the term reallocates under 8% of its mass and cannot plausibly produce a +4/+5 effect when the entire first-order straightening effect was +7.33 OL / +6.66 MPC |

### 4.3 Combination, and what a STOP actually means

Both rules are evaluated **independently** and mechanically, by
`probe_ccr_curvature.py --readout actions --summarize`. The combined verdict is **STOP if either rule
is STOP**. Stage 1 is permitted **only** when rule A is GO-or-MIDDLE **and** rule B is GO-or-MIDDLE.

**A STOP ends the feature.** Tasks 6.x onward are not executed: no `compute_acs`, no gate, no action
reducer, no ACS code path at all. `MCA_Fallback` (`compute_mca` — already written, reviewed, never
run, zero new code, `<0.1%` overhead, 0.8 GPU-h to a verdict, targeting the orthogonal
regularization-space-versus-planning-space gap) becomes the next arm, and the Stage-0 statistics are
written up as findings **N1 and N2 regardless** (§10.4). There is no salvage path that keeps the
story: gating on a signal that does not vary the way the story requires is not an inductive bias, it
is noise.

On a **MIDDLE**, the downgraded claim is recorded **at the moment the verdict is read**, not
retroactively.

### 4.4 The thresholds are judgment calls, not derivations (Requirement 2.17)

`1.5x`, `1.1x`, `0.15` and `0.08` are **judgment calls.** None is derived from a model, a power
calculation or a measurement. `1.5x` is "clearly separated rather than marginally separated"; `1.1x`
is "indistinguishable"; `0.08` is the point below which the reallocated mass is too small to
plausibly move a +4/+5 bar given that the whole first-order effect was +7.33; `0.15` is roughly twice
that. They are written down before the data **precisely because they are judgment calls** — an
arbitrary threshold fixed in advance is a test, and the same threshold chosen afterwards is a fit.
This is the CCR failure mode (`PROGRESS_CCR.md` §5a, §6a) being blocked structurally.

### 4.5 What Stage 0 can and cannot establish — attached here, not in a footnote

**`n = 4` with no independent replicates. It can refute the mechanism; it cannot establish it.** Four
points, and the "gains" it is correlated against are themselves 3-seed means with per-seed spreads as
wide as 74/82/70 on a single checkpoint. Two further limitations matter more than the sample size:

1. **The four environments carry differently-typed action variables.** PushT's actions are *relative
   pusher displacements* (`rel_actions.pth`, `/100.0`, normalized by the hardcoded near-isotropic
   `ACTION_STD = [0.2019, 0.2002]`); PointMaze's are forces / velocity commands on a point mass;
   Wall's are dot velocities — the latter two normalized by *data-computed* per-dim mean/std. So
   `cos(a_t, a_{t+1})` is **not the same physical quantity** across the four points being correlated.
   This is a *structural* limitation of the comparison, not a noise problem, and no amount of data
   fixes it.
2. **A confirmed ordering is consistent with many mechanisms other than ACS's.** PushT differs from
   PointMaze in **contact dynamics**, in having a **second movable object**, in having **rotational
   state** (`block_angle` readout R² 0.183 against 0.50-0.80 for the positional dims,
   `PROGRESS_CCR.md` §5c), and in being trained for **2 epochs instead of 20**. Any of those could
   produce the same gain ordering.
3. **`frameskip=5` may wash out the reversals that motivate the whole idea.** The gate sees the *net*
   displacement over 5 env steps. A pusher that reverses *within* a latent step has a small-norm sum
   whose direction is dominated by whichever half of the motion was larger, and two consecutive
   latent steps could both have near-zero net displacement and an essentially random relative angle.
   This is measured directly, and it is why `raw` and `first` are measured too (§4.1's downgrade
   rule).

Therefore the Stage-0 result is used **asymmetrically, on purpose**: a STOP is decisive, because the
premise is a necessary condition and it failed; a **GO is permission to spend 0.8 GPU-h, not evidence
for the mechanism.**

Stage 0 is worth running even if ACS is never built: the per-environment statistics answer *when does
temporal straightening help?* with a measurable dataset property rather than a post-hoc narrative, for
zero GPU-hours (N1, N2).

---

## 5. The early-read gate — checks 0, 1, 1b, 1c, 2a, 2b, 3. WRITTEN BEFORE THE ARM IS LAUNCHED

Pre-registered here in full, with every threshold, before the Stage-1 arm exists. Checks 0-3 and their
mechanization through `summarize_training_log.py --prediction-gate` are reused from the on-hold TMR
design; **check 1's interpretation is inverted, and that inversion is the most important content in
the whole spec.** Every quantity is read at **matched `global_iter`** against `checkpoints_ctrl8k`'s
own rows, which are exact. Cost of the whole gate: **~0.8 GPU-h** (the control is free, and a lost
prefix can be regenerated bitwise).

### 5.1 Check 0 — step rate, as a bug detector

**`it_per_s >= 2.72`** at steady state, read from telemetry rows **past row 400**, against the
reference **2.862** it/s.

Predicted ACS cost is order **`1e-8`** of the step (one 5-term sum over `(32,4,10)`, one cosine over
`(32,2,2)`, one relu, two masked sums — against 128 DINOv2 ViT-S/14 image passes at ~4.6 GFLOPs each).
So a 5% breach is **not a cost to accept**: it means the implementation is doing work it was not
designed to do. **Fix the code and hold the arm.** This is a bug detector, not a budget check.

### 5.2 Check 1 — prediction loss, INVERTED: a positive directional prediction, not a guard

For CCR and TMR, prediction loss was a **guard** — the thing that must not degrade. For ACS it is a
**positive prediction.** ACS stops forcing differently-acted transitions to look collinear, which is
exactly the information the predictor needs to know where a given action takes the latent. If the
mechanism is real, the predictor should get *better*.

This matters because prediction loss is **the only quantity measured to be causally linked to success
on this codebase**: CCR degraded it by +16.9% in 8 of 8 consecutive matched rows (one-sided sign test
p ≈ 0.004) and success fell (−2.0 OL, −8.0 MPC at matched budget). Every prior intervention here
pushed that channel the wrong way and lost.

Run with `--prediction-gate <CONTROL_RUN_DIR> --prediction-gate-direction improve` (default is
`guard`; check 1 must be run in `improve` mode).

| condition at `global_iter` 8000 (scaled `prediction`; control **0.013196**) | verdict |
|---|---|
| `prediction <= 0.013196` (at or better than control) **AND** **>= 15 of the last 20** matched rows better (one-sided sign test p ≈ 0.021) | **GO** — the directional prediction is confirmed on the causal channel; the strongest early signal available in this project |
| additionally `prediction <= 0.012536` (−5% or better) | **STRONG GO** — recorded **separately**, because effect size matters for whether +4/+5 is reachable |
| `prediction > 0.014516` (+10%) **OR** **>= 15 of the last 20** matched rows **worse** | **STOP** — the directional prediction was **refuted** on the one channel measured to be causal, and ACS's whole mechanism story runs through it |
| anything else | **MIDDLE** — decided by checks 1b, 1c and 2, **with no discretion** |

The STOP bound tightens from TMR's +25% to **+10%** because for ACS a degradation is not a cost to
tolerate, it is a **refutation**: the mechanism claim is precisely that removing pressure from
action-reversing transitions preserves action-discriminability.

**Limit of this check, stated with it:** it is measured at 8,000 steps, which is **6.5% of the
budget**, on a single arm, on one channel. A confirmed direction at 8k is far more informative than
check 3 (a continuous quantity on 40 exact matched rows on a bitwise-deterministic platform, versus a
near-floor binomial), but it is a *proxy* for success, not success.

### 5.3 Check 1b — scale preservation (the λ prediction, made falsifiable)

The weighted mean is scale-preserving, so the curvature share should land where the baseline's did.
Read at **`global_iter` 200 and 8000**:

- **curvature share within `[65%, 80%]`** — the control's value is **73.741%** at 8k. Outside that
  band, either the reallocation is far more consequential than the algebra suggests or there is a bug;
  **both require investigation before the arm is believed.**
- **prediction share `>= 11.75%`** — the CCR floor, retained (predicted ~23.5%, so ~2x slack).
- **no term below the collapse threshold inside the first 1,000 iterations** (`--collapse-check`).
- record `curvature_loss_used_for_training` at 200 and at 8000 **and report the ratio**. A term ~80%
  satisfied by step 8000 exerts little pressure over the remaining ~116,000 steps while the cost is
  paid for the full distance; CCR's raw term fell 79% and that was **measured cost with vanishing
  benefit** — a STOP even though it looks like the mechanism working.
- compare the arm's **`curvature_loss_unweighted`** against the control's curvature at
  `global_iter` 200 within **`rtol = 0.05`**, using the *unweighted* quantity.

**Why the unweighted key exists at all, recorded here so the gate is not misread:** under ACS the
`curvature` row is a **w-weighted** average while the control's is a **uniform** average of the same
per-triple values, and ACS downweights exactly the triples it says are most curved. So the arm's
curvature row will read **lower than the control's even with identical geometry.** Comparing those two
rows as if they measured geometry is a false positive waiting at exactly the moment it matters.
`curvature_loss_unweighted` is the geometry number, bitwise equal to
`total_curvature(visual_only(z), "aggcos")`, detached, never added to the loss, and deliberately
**not** in `TELEMETRY_TERMS` so `Σ share ≈ 1.0` still holds.

### 5.4 Check 1c — did the gate actually gate?

ACS-specific, and it is the check that stops an unattributable result. Read from the `acs` telemetry
block via `summarize_training_log.py --acs-gate-check`:

| quantity | rule |
|---|---|
| **`acs_gate_tv`** (the finite-batch form of `R`) | must be **`>= 0.08`** **AND** within a factor **1.5** of the Stage-0 population `R` for PushT |
| **`acs_denom_clamped_frac`** | must be **`< 0.01`** |
| `acs_gate_mean`, `acs_gate_p10` / `p50` / `p90` | reported; must be consistent with the Stage-0 distribution |
| `acs_gate_zero_frac` | reported; should match Stage-0's `frac(cos<0)` |
| `acs_masked_frac` | reported; a high value means the windows are mostly static and the whole term is thin |

**If `acs_gate_tv ≈ 0`, the term IS the baseline and nothing can be attributed to it — regardless of
what `mean(w)` reads. That is a STOP.** A mismatch beyond 1.5x against Stage-0's `R` means the
training-time `a_t` is not the one Stage 0 measured, i.e. a **wiring defect** in the substep reduction
or in the triple-to-action-pair alignment. That class of error is caught by a mechanical check and
missed by an eyeball, which is why Stage 0 and training call the **same shipped `reduce_action` and
`action_gate`** — the structural fix for CCR's calibration error, applied to the gate.

### 5.5 Check 2a — the gate-split curvature signature (held-out)

ACS's target is not "less curvature", it is a **reallocation**. Measured on **held-out** windows at
**`--num-windows 192`** (the CCR round established that 64 windows is noise: a −28% `block_angle`
delta at n=64 collapsed to −9% at n=192), arm checkpoint versus control checkpoint, **identical flags
and seed**, through the existing `_aggregate_latent` helper and the shared geometry helpers. Split
held-out triples by gate value and compare **unweighted** per-triple curvature:

| bucket | pre-registered ACS prediction |
|---|---|
| **`w = 0`** (reversing) | curvature **higher** than the control's — pressure was removed here, so the geometry is allowed to bend |
| **`w >= 0.5`** (near-constant) | curvature **equal or lower** than the control's — pressure was concentrated here |
| overall unweighted mean | **reported, direction not pre-registered** — it is a mixture of the two |

**Failing both directional rows = STOP:** the reduction did not reallocate anything measurable on
held-out data, so nothing downstream is attributable to the gate. This is sharper than "did the loss go
down", because a loss that goes down for the wrong reason is exactly what CCR delivered (−96% on its
own objective, none of it converted).

### 5.6 Check 2b — the rotational-state prediction (a known limitation turned into a test)

`PROGRESS_CCR.md` §6f established that curvature regularization **suppresses rotational state**:
`block_angle` readout R² is 0.183 in the paper's own trained model against 0.50-0.80 for the four
positional dimensions, it *degrades with training* (0.278 @8k → 0.183 @124k), and Table 1's gains are
largest on the pure-position tasks and smallest on PushT — the only task with rotational state.
Rotation *is* curvature: a rotating object traces an arc, so its velocity direction changes by
construction.

ACS removes straightening pressure precisely where the latent velocity turns, so it predicts the
**opposite** direction: `block_angle` R² should **improve** versus the matched control. Measured with
`state_readout_r2` **unchanged**, at **`--num-windows 192`**, on `--readout state` (`block_angle`).

**Gated leniently and deliberately:** check 2b passes when `block_angle` R² **does not degrade beyond
noise**. An improvement is recorded as **supporting evidence and is not required for GO** — it is a
bonus prediction. A confirmation would convert §6f from a limitation into a general statement about
curvature-family regularizers; a refutation bounds §6f to unconditional penalties. Either is finding
N3.

### 5.7 Check 3 — matched-budget success rate, CATASTROPHE DETECTOR ONLY

8,000-step checkpoints, **1 seed**, unmodified evaluation protocol, open-loop and MPC. Training is
bitwise deterministic and `plan.py` seeds episodes from `seed` with a deterministic planner
(`sample_type=zero`, `action_noise=0`), so this is an **exact paired difference** — counts of
episodes, 2 percentage points each. Control @8k: **16.0 OL / 18.0 MPC**.

| condition | reading |
|---|---|
| difference **`<= -10`** points in either setting | **red flag worth acting on** |
| difference within **`±10`** points | **carries no information** — must be reported as **neither support nor refutation** |

**Honest statement of its power, in the same paragraph as its rule: it is nearly uninformative.** Both
arms sit near the floor; at `p ≈ 0.17` the per-arm binomial SE is **~5.2 points**, so distinguishing
arms at 2 SE needs `Δ >= ~11` points — 5 to 6 episodes out of 50. It is a catastrophe detector and
nothing else. The matched-budget test is also **structurally biased against any new term**, since a
new term pays its cost from step 1 and 8,000 steps is 6.5% of the budget, and **one seed does not
establish generalization** to other episode sets.

---

## 6. Acceptance bars for a full run (Stage 2) — PRE-REGISTERED

| setting | our baseline | paper | **operational bar** |
|---|---|---|---|
| open-loop | 75.33 ± 6.11 (74, 82, 70) | 77.33 ± 6.18 | **79.33** (+4.0) |
| MPC | 82.00 ± 2.00 (82, 80, 84) | 85.33 ± 4.99 | **87.00** (+5.0) |

**Both settings must clear their bar.** Mean over the **3 data-sampling seeds 100 / 200 / 300** at
**`n_evals=50`**, evaluated by `ccr_acceptance_gate.py`. **Per-seed values are reported alongside the
mean, never a mean in isolation.**

Protocol invariants, unchanged from the paper: encoder lr `1e-5`, **2 epochs** on PushT, batch 32,
`num_hist=3`, `num_pred=1`, `frameskip=5`, bf16, `stop_grad=True`, **`λ = 0.1` in every ACS arm**,
CCR off (`lambda_cf=0`, `ccr_rho=0`). Open-loop: GD planner, `objective.mode=last`, `alpha=1`,
`max_iter=1`, `n_taken_actions=25`. MPC: GD planner, `objective.mode=staged`, `alpha=1`,
`max_iter=20`, `n_taken_actions=5`. Sub-planner: horizon 25, lr 0.1, `sample_type=zero`,
`action_noise=0`, `opt_steps=100`. **PushT runs before any other environment**, and if another
environment is attempted the claim there is **open-loop only** (paper MPC is 100.00 Wall / 100.00
UMaze / 98.67 Medium — a +5 MPC margin is arithmetically impossible).

**Limit of this bar, stated with it:** **+4 open-loop on a 3-seed mean is roughly 1.3 standard
errors**, even with exact pairing, and the single-checkpoint per-seed spread of **74 / 82 / 70** is the
noise reality it lives in. A positive result at this bar is real but **thin**, and would need the
per-seed values and the paired per-episode vectors reported alongside it.

---

## 7. Recorded limitations — consolidated, each already stated next to its conclusion

Requirement 3.9 requires every limitation to appear in the same paragraph as the conclusion it limits,
so each of these is stated **inline** above as well. This section is the index, not the only place they
appear.

| # | limitation | the conclusion it limits | stated inline at |
|---|---|---|---|
| L1 | **`n = 4`, no independent replicates.** Can refute the mechanism, cannot establish it | the Stage-0 rule A verdict | §4.5(1) |
| L2 | **Differently-typed action variables** across the four environments: PushT relative pusher displacements, PointMaze forces / velocity commands on a point mass, Wall dot velocities. `cos(a_t, a_{t+1})` is not the same physical quantity across the four correlated points. *Structural, not a noise problem — no amount of data fixes it* | the Stage-0 cross-environment correlation, and any mechanism claim built on it | §4.5(1), §8.5 |
| L3 | **Confounds:** contact dynamics, a second movable object, rotational state (`block_angle` R² 0.183 vs 0.50-0.80), and 2 training epochs on PushT against 20 elsewhere. Any of these could produce the same gain ordering | a *confirmed* Stage-0 ordering | §4.5(2) |
| L4 | **A GO is permission to spend 0.8 GPU-h, not evidence for the mechanism.** The Stage-0 result is used asymmetrically on purpose | the Stage-0 GO verdict | §4.5 (closing) |
| L5 | **`frameskip=5` may wash out within-step reversals.** The gate sees only the net displacement over 5 env steps; a within-step reversal has a small-norm sum whose direction is set by whichever half was larger | the premise itself, and rule A — hence the `raw`/`first` cross-measurement and the pre-declared MIDDLE downgrade | §4.5(3), §4.1 |
| L6 | **Only 2 triples per sample at `num_hist=3, num_pred=1`.** Zeroing reversing triples can leave a sample contributing one triple or none, so the *effective* number of constrained triples falls by `frac(w=0)` and the curvature gradient gets noisier over ~123,858 steps. **No early-read check measures this cost directly** — the batch-level pooling of 64 triples mitigates it, and mitigation is not measurement | any Stage-1 GO, and the cost side of the whole term | §7.1 below |
| L7 | **The gate proxies "the *controlled object* reversed", not "the latent velocity's direction change is action-explained".** PushT actions command the pusher; the latent velocity is dominated by the whole scene including the T-block. High `cos` during a non-contact repositioning move coexists with a latent velocity that is almost pure pusher translation; low `cos` during a re-approach coexists with a static block. Those coincide often on PushT (the pusher is the only actuated object) and **are not the same statement** | the mechanism claim in *every* verdict, GO or otherwise | §7.2 below |
| L8 | **Batch coupling.** The weighted mean normalizes across the batch, so a sample's contribution depends on the others drawn with it; when `Σ w` is small the gradient is dominated by a few triples. `acs_denom_clamped_frac` and `gate_p10` are logged and the `< 0.01` clamp rule guards the extreme, but the intermediate small-but-unclamped regime is **monitored rather than bounded** | check 1c's pass, and any attribution of a Stage-1 effect | §5.4 |
| L9 | **Nothing controls for "PushT-specific".** A single-environment result on the one Table 1 cell with headroom in both settings is a single-environment result | any Stage-2 pass | §6, §8.5 |
| L10 | **The last two interventions on this codebase were negative.** CCR reached a measured negative result at ~26 GPU-h; TMR was shelved before launch on evidence in the paper's own appendix. Weak evidence about the search space, and it should move the prior **down** | the probability estimate in §9 | §9 |

### 7.1 L6, stated where it bites (Requirement 3.7)

At `num_hist=3, num_pred=1` there are exactly **2 curvature triples per sample** and 64 per batch.
`relu(cos)` zeroes the entire reversing half-space, so a sample can be left contributing a single
triple, or none. The batch-level weighted mean pools 64 triples and mitigates this, but the
**effective** number of constrained triples falls by `frac(w=0)`, and a noisier curvature gradient
sustained over ~123,858 steps is a **real cost that none of checks 0, 1, 1b, 1c, 2a, 2b or 3 measures
directly.** So any Stage-1 GO is a statement about the measured channels only; the variance cost is
argued, not measured, and that asymmetry is recorded here rather than discovered later.

This is also the load-bearing risk on the benefit side: **downweighting action-reversing transitions
removes *some* straightening pressure, and straightening demonstrably works** — the paper's own
ablation reports that *every* cosine variant beats no-straightening. ACS removes pressure from a
subset of transitions on a theory about which subset deserves it. If that theory is wrong in detail —
if the encoder needs uniform pressure to reach a straight solution at all, or if the reversing triples
are where the most useful gradient lives — then ACS is simply **less straightening, and less
straightening is worse.** The reallocation partially offsets this (surviving triples get *more*
pressure than baseline, not the same), but reallocation is not replacement, and there is no argument
that the reallocated pressure is as useful as what was removed.

### 7.2 L7, stated where it bites (Requirement 3.8)

The gate is `relu(cos(a_t, a_{t+1}))` on the **net commanded pusher displacement**. What the
hypothesis is about is whether the *latent velocity's* direction change is explained by the action.
Those are different statements, and the gate implements the first. On PushT they coincide often enough
for the mechanism story to be plausible — the pusher is the only actuated object — and "often enough
to be plausible" is the exact strength of the claim, in every verdict. A Stage-1 or Stage-2 pass
therefore supports "gating on controlled-object reversal helps", **not** "the latent velocity's
direction change is action-explained".

---

## 8. Novelty positioning — WRITTEN 2026-08-08, BEFORE ANY OUTCOME

Dated so that a win shows the prior art was **disclosed in advance** rather than found by a reviewer,
and written against our own interest where the evidence goes that way. Everything below is a
paraphrase with an inline link; no source is quoted at length.

### 8.1 The target paper

*Temporal Straightening for Latent Planning* — [arXiv 2603.12231](https://arxiv.org/abs/2603.12231),
an **[accepted ICML 2026 poster](https://icml.cc/virtual/2026/poster/64904)** (NYU / Brown / Toronto,
**Yann LeCun a coauthor**). This is the paper ACS modifies, and its acceptance status is recorded here
so nobody later treats it as a preprint of unknown standing. Its cell we target is the one **it
reports as its weakest straightening gain** (PushT, +7.33 OL).

### 8.2 Iso-FM — the closest prior art, and it lands on the *sibling*, not on ACS

[**Iso-FM**](https://arxiv.org/abs/2604.04491) (ICML 2026) publishes the mathematical object of the
**on-hold TMR sibling spec**: penalizing acceleration / enforcing constant speed along the trajectory.
[**OAT-FM**](https://arxiv.org/html/2509.24936) goes further and treats constant-velocity enforcement
as an *existing baseline it improves on*. TMR's mathematical novelty was therefore limited **before it
was ever run**, which is why TMR is on hold and ACS supersedes it.

**Why ACS does not collide with it, for one structural reason:** the straightening / flow-matching
literature is **passive**. Its regularizers are unconditional **by necessity — there is no control
signal in the setting to condition on**. Iso-FM constrains the *speed profile* of an uncontrolled
trajectory; ACS changes *which transitions a curvature penalty applies to*, using a signal Iso-FM's
setting does not contain. There is no published gated form for ACS to collide with. This is recorded
as a *structural* argument rather than a "we searched and found nothing" argument, because the latter
is what TMR relied on and it failed.

The passive line, for the record:
[Hénaff et al. 2019](https://link.springer.com/10.1038/s41593-019-0377-4),
[V1 straightens natural movie trajectories](https://link.springer.com/10.1038/s41467-021-25939-z),
[AI-generated video detection via representational straightness](https://arxiv.org/abs/2507.00583),
[LLM representational curvature](https://arxiv.org/abs/2604.23985),
[Chirality in Action](https://arxiv.org/html/2509.08502v1) — all treat trajectory straightness as an
**unconditional** property of a representation.

### 8.3 Temporal-Distance-JEPA — shares the framing, different instrument

[**Temporal-Distance-JEPA**](https://arxiv.org/abs/2607.25337) states our framing directly: JEPA-style
planners inherit their ranking from embedding geometry (typically latent Euclidean distance), which is
a byproduct of representation learning rather than a cost mined from logged experience. The related
[TRM](https://arxiv.org/abs/2605.22164) and the quasimetric GCRL line do the same.

**The instrument is different: they add a learned cost head; ACS adds no head, no module, no
parameter and no buffer.** ACS changes a reduction inside an existing regularizer and touches nothing
downstream — `plan.py`, `planning/*` and `datasets/*` are frozen by the scope guard. Shared intuition,
different object, different failure modes. Their independent arrival at the framing is weak positive
evidence that reviewers will recognize the framing; it is **not** evidence about the success rate.

### 8.4 Action-conditioned representation learning, and the Koopman / equivariant line

[CAPE](https://arxiv.org/abs/2606.07304),
[action-conditional self-predictive RL](https://arxiv.org/html/2406.02035v1),
[SCAR](https://arxiv.org/pdf/2605.16412) and
[latent-action world models](https://arxiv.org/html/2512.10016) all condition representation learning
on actions — but every one does it by **predicting or discriminating action outcomes**: the action
enters as an *input to a predictive objective*. **None gates a geometric regularizer on the action.**
In ACS the action enters no prediction at all; it is a **weight on a geometric penalty**, and it
carries **no gradient**.

[KEEC](https://arxiv.org/abs/2312.01544),
[Koopman operators for interactive dynamics](https://arxiv.org/html/2306.11941v4) and
[Koopman Dreamer](https://arxiv.org/html/2607.19719) share the intuition that the action induces a
transformation on the latent state, but they **parameterize the dynamics**: the action selects or
indexes a linear operator the *predictor* applies. ACS leaves the dynamics model entirely alone — no
predictor call is added, `models/vit.py` and `models/dino.py` are frozen — and uses the action to
modulate a **regularizer on the encoder's trajectory geometry**. A dynamics parameterization and a
geometric regularizer are different contributions; this paragraph exists so the distinction is on
record before a reviewer draws it.

### 8.5 The defensible claim, stated conservatively — and what it does NOT include

> The novelty is **conditioning a straightening prior on the control signal**, motivated by the
> observation that the hypothesis the prior derives from was formulated for *passive observation* and
> is applied here to an *actively controlled* agent.

What that claim does **not** include:

- It does **not** claim to explain the Table 1 gain ordering. The correlation is `n = 4`, across
  environments carrying differently-typed action variables (L2), with confounds (L3), and cannot
  establish a mechanism. Under a **MIDDLE** Stage-0 verdict the claim is downgraded further, to "the
  gate is a useful inductive bias", and the writeup **must not** claim the explanation.
- It does **not** claim novelty in "using actions in representation learning" — §8.4 is crowded.
- It does **not** claim the gate function is new mathematics. `relu(cos)` is the simplest object
  satisfying the stated requirements; the contribution is *what it weights and why*, not the weight.
- It does **not** claim the latent velocity's direction change is action-explained — only that the
  *controlled object* reversed (L7).
- It does **not** generalize beyond PushT. Nothing in the plan controls for PushT-specific effects
  (L9), and an extension to Wall / UMaze / Medium would necessarily be open-loop-only.

**The bar, recorded as a difficulty statement.** We propose to beat an accepted ICML 2026 paper's own
reported cell by **+4 OL / +5 MPC** by changing **one reduction** inside **one existing loss term**, on
the cell the authors themselves report as their weakest straightening gain. That is hard, and nothing
in §8 makes it easier.

**Novelty and beating the number are separate axes.** If ACS clears the acceptance gate it publishes
on the success-rate result plus the mechanism finding, whatever the related work says. If it does not,
novelty is moot and the **Stage-0 measurements are the deliverable** (N1, N2) at zero GPU-hours. The
literature search changed the *framing*, not the experimental plan.

---

## 9. Honest probability assessment — recorded before any measurement

**Probability of clearing the operational bar (+4 OL *and* +5 MPC on 3-seed means): 25-35%.**

**Why that number, specifically.** It is higher than the on-hold TMR arm (8-13%) and the MCA arm
(12-18%) **for one reason, not a general feeling: ACS is the first intervention on this codebase whose
predicted effect on the *causal channel* — prediction loss — is positive rather than negative.** Every
prior intervention pushed that channel the wrong way and lost: CCR degraded prediction by +16.9% (8/8
consecutive matched rows, sign test p ≈ 0.004) and success fell −2.0 OL / −8.0 MPC at matched budget;
TMR's most likely failure mechanism was the same channel, which is why its check 1 was a *guard*. ACS's
check 1 is a *prediction*, and a confirmed directional result there at 8k is the strongest early signal
available in this project.

**And why it is not higher.** +4 / +5 is still most of what the entire *first-order* effect delivered —
straightening itself bought +7.33 OL / +6.66 MPC on this cell. Asking a second-order refinement for
~60% of the first-order effect is a large ask. L10 (two consecutive negative rounds on this codebase)
moves the prior **down**, not sideways.

| bar | probability |
|---|---|
| **Stage 0 returns GO or MIDDLE (the premise is not refuted)** | **~55-65%** |
| Given GO/MIDDLE, ACS *improves* prediction loss vs the matched control at 8k (check 1) | ~50% |
| Given GO/MIDDLE, ACS clears the whole early-read gate | ~40% |
| ACS beats our baseline on open-loop (point estimate, full run) | ~45% |
| ACS beats our baseline on MPC (point estimate, full run) | ~40% |
| **ACS clears +4 OL and +5 MPC** | **25-35%** |
| ACS yields a defensible open-loop-only improvement | ~40% |
| The Stage-0 measurements (N1, N2) are obtained regardless of outcome | ~98% |
| The rotational-state prediction (check 2b) is confirmed | ~35% |

**Why the "GO or MIDDLE" row is only ~55-65%.** Stage 0 is a genuine test and the motivating
observation is currently an **argument, not a measurement**. PushT's control could well be smooth at
the 5-substep aggregation level even though the task requires circling and re-approaching: the pusher
may turn *gradually* over several latent steps rather than reversing within one (L5). That is measured
directly, and it is why ~40% of the mass sits on a STOP.

**The risk the design has closed rather than guarded:** "the relaxation could be mimicked by lowering
λ." The weighted mean is invariant to uniform gate rescaling, so there is no λ-reduction component to
control for, and the λ-matched control is the existing baseline at **zero** cost. The risk the design
has **not** closed is L6/§7.1 — less straightening may simply be worse.

---

## 10. Placeholders — filled in as the measurements land, not before

Each subsection below is empty by design. Filling one in **before** its measurement exists would
defeat the purpose of this file.

### 10.1 Stage-0 measured statistics — MEASURED 2026-08-08

Run on the B200 pod, `/workspace/arun/ccr`, commit `23b25db`, `DATASET_DIR=/workspace/arun/data`.
**CPU only, minutes, 0 GPU-h, no video decoded, no checkpoint read.** Reports in
`probe_outputs/acs_actions_{pusht,wall,point_maze}.json`.

**`point_maze_medium` was not measured: the dataset is not on this pod**
(`/workspace/arun/data/point_maze_medium/states.pth` does not exist). `--summarize` therefore refused
to emit a verdict JSON, correctly — the rule is pre-registered over four environments. **The rule-A
STOP does not depend on it**, and §10.2 records why.

Train split (headline), all three reductions:

| env | reduction | mean cos | median cos | `frac(cos<0)` | `frac(cos<0.5)` | `mean(w)` | `frac(w=0)` | **`R`** | `n_triples` | `n_windows` |
|---|---|---|---|---|---|---|---|---|---|---|
| pusht | sum | 0.5528 | 0.7733 | **0.1504** | 0.3255 | 0.6292 | 0.1504 | **0.2538** | 3,963,442 | 1,981,721 |
| pusht | raw | 0.4446 | 0.5649 | 0.1864 | 0.4547 | 0.4997 | 0.1864 | 0.3117 | 3,963,442 | 1,981,721 |
| pusht | first | 0.4796 | 0.7307 | 0.2019 | 0.3680 | 0.5877 | 0.2019 | 0.2932 | 3,963,442 | 1,981,721 |
| wall | sum | 0.6777 | 0.8331 | **0.0785** | 0.2312 | 0.7042 | 0.0785 | 0.1857 | 107,136 | 53,568 |
| wall | raw | 0.5173 | 0.6060 | 0.0837 | 0.3725 | 0.5378 | 0.0837 | 0.2095 | 107,136 | 53,568 |
| wall | first | 0.5715 | 0.7602 | 0.1386 | 0.3199 | 0.6278 | 0.1386 | 0.2466 | 107,136 | 53,568 |
| point_maze (umaze) | sum | 0.0027 | 0.0067 | **0.4983** | 0.6649 | 0.3203 | 0.4983 | 0.5496 | 291,600 | 145,800 |
| point_maze (umaze) | raw | 0.0003 | −0.0004 | 0.5005 | 0.9392 | 0.1293 | 0.5005 | 0.5764 | 291,600 | 145,800 |
| point_maze (umaze) | first | −0.0009 | 0.0012 | 0.4996 | 0.6691 | 0.3175 | 0.4996 | 0.5523 | 291,600 | 145,800 |
| point_maze_medium | — | _not measured_ | | | | | | | | |

Validation split (cross-check):

| env | reduction | mean cos | median cos | `frac(cos<0)` | `mean(w)` | **`R`** | `n_triples` | `n_windows` |
|---|---|---|---|---|---|---|---|---|
| pusht | sum | 0.5900 | 0.8069 | 0.1331 | 0.6553 | 0.2326 | 4,230 | 2,115 |
| pusht | raw | 0.5068 | 0.6577 | 0.1565 | 0.5543 | 0.2796 | 4,230 | 2,115 |
| pusht | first | 0.5528 | 0.7965 | 0.1579 | 0.6362 | 0.2543 | 4,230 | 2,115 |
| wall | sum | 0.6758 | 0.8321 | 0.0796 | 0.7031 | 0.1876 | 11,904 | 5,952 |
| wall | raw | 0.5109 | 0.5997 | 0.0880 | 0.5321 | 0.2136 | 11,904 | 5,952 |
| wall | first | 0.5643 | 0.7610 | 0.1440 | 0.6252 | 0.2499 | 11,904 | 5,952 |
| point_maze (umaze) | sum | −0.0121 | −0.0257 | 0.5088 | 0.3129 | 0.5580 | 32,400 | 16,200 |
| point_maze (umaze) | raw | −0.0061 | −0.0058 | 0.5075 | 0.1258 | 0.5794 | 32,400 | 16,200 |
| point_maze (umaze) | first | −0.0097 | −0.0222 | 0.5072 | 0.3142 | 0.5580 | 32,400 | 16,200 |

- **20-bin histograms over `[-1, 1]`, transcribed 2026-08-08.** The JSON reports are **pod-local and
  gitignored** (`.gitignore:14` matches `*outputs*`; the pod is pull-only, `AGENT_MEMORY_2.0.md` §5.1),
  so these counts are the version-controlled copy. Edges are deterministic: 20 equal bins of width 0.1
  over `[-1, 1]`, bin 0 = `[-1.0, -0.9]`, bin 19 = `[0.9, 1.0]`.

| env | split | red | counts, bin 0 → 19 |
|---|---|---|---|
| point_maze | train | sum | 41902, 17641, 14192, 12372, 10951, 10367, 9637, 9583, 9302, 9357, 9101, 9497, 9773, 9955, 10263, 10863, 12138, 14301, 18046, 42359 |
| point_maze | train | raw | 39, 621, 2179, 5347, 9527, 14712, 20559, 27006, 31386, 34575, 34284, 31160, 26784, 20970, 14735, 9345, 5232, 2471, 635, 33 |
| point_maze | train | first | 43053, 17488, 13642, 11726, 10954, 10177, 9902, 9604, 9705, 9435, 9548, 9865, 9708, 9847, 10443, 11023, 11940, 13541, 17441, 42558 |
| point_maze | valid | sum | 4751, 2054, 1595, 1409, 1218, 1117, 1146, 1113, 1086, 995, 997, 1029, 1095, 1095, 1126, 1184, 1298, 1548, 1908, 4636 |
| point_maze | valid | raw | 4, 98, 243, 635, 1096, 1707, 2166, 3052, 3614, 3827, 3727, 3435, 3073, 2363, 1614, 898, 545, 239, 62, 2 |
| point_maze | valid | first | 4850, 1971, 1569, 1353, 1216, 1148, 1162, 1032, 1063, 1068, 1034, 1080, 1024, 1053, 1165, 1198, 1333, 1412, 1964, 4705 |
| pusht | train | sum | 122214, 49510, 39632, 37785, 37796, 46213, 50834, 57995, 69704, 84392, 99454, 114891, 134884, 156083, 188740, 213277, 255923, 315557, 466883, 1421675 |
| pusht | train | raw | 2108, 12046, 25410, 42195, 62320, 76380, 96613, 116884, 140414, 164426, 183929, 193967, 208867, 224688, 251809, 287340, 336312, 410832, 565047, 561855 |
| pusht | train | first | 162405, 72821, 63691, 63791, 60545, 62716, 67819, 74566, 80174, 91686, 100437, 115046, 131614, 142979, 168331, 200675, 237315, 310379, 451281, 1305171 |
| pusht | valid | sum | 125, 25, 27, 32, 52, 38, 51, 60, 79, 74, 86, 106, 149, 153, 186, 214, 291, 332, 546, 1604 |
| pusht | valid | raw | 0, 14, 20, 40, 66, 74, 80, 106, 132, 130, 164, 188, 196, 200, 238, 257, 374, 409, 725, 817 |
| pusht | valid | first | 132, 54, 66, 47, 49, 55, 52, 58, 68, 87, 106, 107, 115, 185, 165, 208, 220, 358, 527, 1571 |
| wall | train | sum | 311, 303, 413, 518, 597, 818, 949, 1241, 1475, 1787, 2080, 2653, 3019, 3916, 4693, 6077, 7754, 10699, 15311, 42522 |
| wall | train | raw | 4, 62, 165, 372, 522, 822, 1035, 1399, 1992, 2596, 3357, 4367, 5623, 7388, 10208, 12768, 16871, 19711, 15040, 2834 |
| wall | train | first | 1162, 849, 881, 1083, 1222, 1544, 1550, 1898, 2181, 2483, 2884, 3183, 3848, 4555, 4945, 6190, 7585, 9849, 13826, 35418 |
| wall | valid | sum | 29, 21, 50, 57, 108, 96, 89, 149, 169, 179, 278, 288, 317, 425, 540, 608, 839, 1199, 1763, 4700 |
| wall | valid | raw | 0, 7, 7, 34, 83, 114, 97, 173, 206, 326, 396, 505, 636, 794, 1076, 1499, 1930, 2103, 1580, 338 |
| wall | valid | first | 114, 114, 137, 134, 144, 182, 186, 216, 246, 241, 316, 364, 406, 497, 526, 716, 820, 1041, 1604, 3900 |

- **The histogram overturns the reading the two moments invited, and this is the substantive result.**
  UMaze's `mean cos = 0.0027` is **not** a narrow distribution around zero and **not** a broad unimodal
  one either. It is **U-shaped, with spikes at both extremes**: 14.37% of triples in `[-1, -0.9]` and
  14.53% in `[0.9, 1.0]`, over a flat ~3.2%-per-bin floor in between.
- **That U-shape is, to a close approximation, the arcsine law — exactly what *zero* directional
  autocorrelation looks like in two dimensions.** For a uniformly random direction in 2-D, `cos θ` has
  density `1/(π√(1−c²))`, which diverges at ±1; the predicted bin-0 and bin-19 masses are both 14.36%
  and the central bins 3.19%. Measured against that law, over all 20 bins: worst deviation **2.8σ**,
  `χ² = 45.5` on 19 df, every bin within ~3% of prediction. Formally a rejection at `n = 291,600` — the
  deviations are mild but statistically real — while the *shape*, including both divergent tails, is
  reproduced. **So the ±1 spikes are not a bang-bang controller signature; they are the geometry of a
  2-D cosine under no directional correlation at all.** PushT and Wall are far from the law in the
  opposite direction: `[0.9, 1.0]` holds 35.87% (PushT) and 39.69% (Wall) against arcsine's 14.36%,
  and `[-1, -0.9]` holds 3.08% and 0.29% against 14.36% — strong directional persistence, Wall the
  strongest.
- **Methodological caution the histograms expose: `frac(cos<0.5)` is not comparable across reductions,
  because they have different dimensionality.** `sum` reduces to a 2-D net displacement; `raw` compares
  the full 10-D block. Under uniform directions `frac(cos<0)` is 0.5 in *any* dimension, but the mass
  near zero grows with dimension (density `∝ (1−c²)^((d−3)/2)`), so UMaze's `raw` value of
  `frac(cos<0.5) = 0.9392` against `sum`'s 0.6649 is largely a dimensional artifact rather than a fact
  about the data — its `raw` histogram is the expected 10-D bell centred on 0. Rule A reads `frac(cos<0)`
  only, so the verdict is unaffected; any future arm comparing the `raw` and `sum` columns on
  `frac(cos<0.5)` would be comparing two different geometries.
- **Validation cross-check: agrees on every ordering.** `frac(cos<0)` train vs validation is
  0.1504/0.1331 (pusht), 0.0785/0.0796 (wall), 0.4983/0.5088 (umaze). The rule-A ordering is identical
  on both splits, so the verdict below is not a split artifact.
- **32-window bitwise check of the action-only loader against `dset[idx][1]` (task 4.2, Requirement 1.16):
  PASSED.** `pytest tests/test_acs_single_gate_impl.py` → 23 passed, 1 skipped. The skip is
  `point_maze_medium`, for the missing dataset. Skipping the `VideoReader` decode did not change what
  was measured, on the real data, for the three environments that produced numbers.
- **`sum` vs `raw` for the Requirement 3.6 downgrade: the downgrade does not fire and has nothing to
  rescue.** Under `raw` the ordering is umaze 0.5005 > pusht 0.1864 > wall 0.0837 — PushT is not the
  highest there either, so neither reduction shows reversal structure in rule A's sense. `frameskip=5`
  washing out within-step reversals (L5) is *not* the explanation for this STOP.
- **Internal consistency, unplanned but worth recording:** `frac(w=0)` equals `frac(cos<0)` to four
  decimals in all nine rows. That is Property 3's zero-set claim — `relu_cos` zeroes exactly the
  `cos <= 0` half-space — confirmed on 4.4 M real triples rather than only on generated cases.

### 10.2 Which Stage-0 rule fired, and the exact numbers — READ 2026-08-08

Written **before** anything downstream is launched (Requirement 16.4). Nothing downstream was launched:
this section records the end of the feature.

## COMBINED VERDICT: STOP. ACS IS NOT BUILT.

- **Rule A verdict: STOP, clause 2.6 — "PushT is not the highest of the four."**
  Driving numbers, train split / `sum` reduction: PushT `frac(cos<0)` = **0.1504**, against
  **UMaze 0.4983** and Wall 0.0785. PushT is not the maximum; UMaze exceeds it by **3.31x**. PushT's
  "margin over the largest of the other three" is **0.30x** — below 1, so the `>= 1.5x` GO condition is
  not merely unmet, it points the wrong way. UMaze lowest? **No — UMaze is the highest**, which is the
  opposite extreme from the one the mechanism story predicts.
- **Rule B verdict: GO** — PushT `R` = **0.2538**, clearing the `>= 0.15` bar. The gate would have
  reallocated about a quarter of the straightening pressure between triples on PushT. Recorded because
  it is true and because it is the half of the pre-registration that ACS passed; it does not rescue
  anything, since §4.3 makes either rule sufficient to STOP.
- **Combined verdict: STOP** (either rule STOP ⟹ STOP). Stage 1 not permitted.

**Why the missing `point_maze_medium` does not change this.** Clause 2.6 fires on PushT failing to be
the maximum, and UMaze already exceeds it by 3.31x. A fourth value can only sit above or below PushT;
neither makes PushT the maximum. The STOP is determined by the three environments that were measured.
What `medium`'s absence *does* limit is the completeness of N1's four-point ordering claim, and that is
recorded in §10.4 rather than glossed.

**The premise is not merely unconfirmed — it is inverted, and this is the finding.** The mechanism story
(§1, §4.1) was that straightening helps most where the control is smooth. Rule A predicted
`frac(cos<0)` ordered inversely to Table 1's gains: pusht > wall ≈ medium > umaze. Measured:
**umaze > pusht > wall.** UMaze carries the paper's largest straightening gain by a wide margin
(+50.00 OL) and has the *most* action-reversing transitions of the three — `frac(cos<0) = 0.4983`,
`mean cos = 0.0027`, `median 0.0067`. Consecutive commanded actions there are close to directionally
independent: a coin flip. PushT, the weakest gain (+7.33), is the second *smoothest*. Both splits agree
and all three reductions agree.

**The most likely reason is L2, the confound this file pre-registered as a limitation rather than
discovered afterwards.** The four environments carry differently-typed action variables: PointMaze
actions are forces on a point mass, PushT actions are relative pusher displacements. A point mass under
near-random *force* commands still traces a smooth path, so `cos(a_t, a_{t+1})` on an acceleration is
not the same physical quantity as on a displacement. This is exactly what §4.5 says Stage 0 can do —
refute the ordering, never establish it — and refutation is what happened. It also means the shipped
gate would have been reading acceleration reversals on PointMaze and displacement reversals on PushT:
not one inductive bias applied to four environments, but four different ones.

**Not re-instrumented after the fact.** A type-comparable statistic (velocity reversals derived from
`state` rather than `act`, say) might well order differently. Choosing it *now*, having seen that
`frac(cos<0)` refuted the premise, is precisely the CCR failure mode §4.4 was written to block, so it
was not done. The STOP stands as the pre-registered rule read it. If a future arm wants that statistic,
it pre-registers it first, before looking.

**Consequences, executed:**

- Tasks **6.x onward are NOT executed.** `compute_acs` does not exist and will not be written. No ACS
  loss term, no `acs_tag` resolver, no `conf/train.yaml` keys, no telemetry block, no Stage-1 arm, no
  permuted-gate attribution arm, no Stage-2 run. Sections 1-4 of the plan (the shared geometry helpers,
  the `straighten` parser `else: raise`, `reduce_action` / `action_gate`, and the Stage-0 probe) are
  **kept**: the parser fix closes a live landmine independent of ACS, the helpers are a bitwise-neutral
  refactor, and the probe is the instrument that produced N1 and N2.
- **`MCA_Fallback` is selected as the next arm** (§12): `VWorldModel.compute_mca`, already written and
  reviewed, never run. Zero new code, 0.8 GPU-h to a verdict.
- **N1 and N2 are written up regardless** — see §10.4. They are the return on this work.
- **GPU-hours spent on ACS: 0.** The best case in the §2 budget table is the one that happened. CCR
  spent ~26 GPU-h to reach a negative result; the ordering of this plan is why this one cost none.

### 10.3 Gate verdict — _NOT YET READ_

| check | threshold (from §5) | measured | verdict |
|---|---|---|---|
| 0 — step rate | `it_per_s >= 2.72` (ref 2.862) | _tbd_ | _tbd_ |
| 1 — prediction (INVERTED) | GO `<= 0.013196` + 15/20; STRONG GO `<= 0.012536`; STOP `> 0.014516` or 15/20 worse | _tbd_ | _tbd_ |
| 1b — curvature share | `[65%, 80%]`, control 73.741% | _tbd_ @200, _tbd_ @8000 | _tbd_ |
| 1b — prediction share | `>= 11.75%` | _tbd_ | _tbd_ |
| 1b — collapse check | no term below threshold in first 1,000 iters | _tbd_ | _tbd_ |
| 1b — `curvature_loss_unweighted` vs control @200 | `rtol = 0.05` | _tbd_ | _tbd_ |
| 1c — `acs_gate_tv` | `>= 0.08` **and** within 1.5x of Stage-0 `R` | _tbd_ | _tbd_ |
| 1c — `acs_denom_clamped_frac` | `< 0.01` | _tbd_ | _tbd_ |
| 1c — gate distribution | `gate_mean`, `p10/p50/p90`, `zero_frac` vs Stage 0 | _tbd_ | _tbd_ |
| 2a — `w = 0` bucket | arm curvature **higher** than control (held-out, n=192) | _tbd_ | _tbd_ |
| 2a — `w >= 0.5` bucket | arm curvature **equal or lower** than control | _tbd_ | _tbd_ |
| 2b — `block_angle` R² | must not degrade beyond noise (n=192) | _tbd_ | _tbd_ |
| 3 — matched-budget eval | catastrophe detector only, `±10` (control 16.0 OL / 18.0 MPC) | _tbd_ | _tbd_ |

Also to be recorded here: the curvature-share **drift** across multiple iterations rather than a single
row (Requirement 16.14); whether the check-1b scale-preservation prediction held (16.7); whether the
check-1 directional prediction on the causal channel held (16.8); and the training-time `acs_gate_tv`
against the Stage-0 `R` estimate **including when the arm succeeds** (16.6).

### 10.4 Findings N1 / N2 / N3 — N1 and N2 WRITTEN 2026-08-08; N3 UNREACHABLE

**N1 — when does temporal straightening help? A dataset property, measured, and it does *not* track
control smoothness.** Across three of the paper's four goal-reaching datasets, the fraction of
consecutive-action direction reversals `frac(cos(a_t, a_{t+1}) < 0)` on the net commanded displacement
per latent step is **UMaze 0.4983, PushT 0.1504, Wall 0.0785** (train split, `sum` reduction,
291,600 / 3,963,442 / 107,136 triples; the validation split and the `raw` and `first` reductions give
the same ordering). Set against the paper's own open-loop straightening gains — **UMaze +50.00,
Wall +10.67, PushT +7.33** — the ordering is the **inverse of what the smoothness story predicts**: the
environment that gains most from temporal straightening is the one whose consecutive commanded actions
are closest to directionally independent (`mean cos = 0.0027`, `median 0.0067`), and the environment
that gains least is the second smoothest. Zero GPU-hours; the statistic is a property of the released
datasets and stands whether ACS is built or not.
**The histograms sharpen this from a fraction into a distributional statement (§10.1).** UMaze's
consecutive net commanded displacements are, to a close approximation, **directionally uniform**: the
measured `cos` distribution tracks the 2-D arcsine law `1/(π√(1−c²))` across all 20 bins — 14.37% and
14.53% in the two extreme bins against a predicted 14.36%, worst per-bin deviation 2.8σ, `χ² = 45.6` on
19 df, so mildly but really rejected at `n = 291,600` while reproducing the shape including both
divergent tails. PushT and Wall sit far from that law in the opposite direction, with 35.87% and 39.69%
of triples in `[0.9, 1.0]` and 3.08% and 0.29% in `[-1, -0.9]`. Ranked by directional persistence the
three are **Wall > PushT > UMaze (≈ none)**, and against gains of **+10.67 / +7.33 / +50.00** that is
**not a monotone inverse relation either**: Wall is the *most* persistent yet gains more than PushT. So
the defensible version of the pattern is about the extreme and not the ordering — the one environment
with essentially no directional autocorrelation in its control is the one straightening helps most, by a
factor of five over the other two — and with `n = 3` even that is an observation, not a law.
**Limitations, attached here because they bound this exact claim.** **L1:** `n = 3` measured of 4
intended, with no independent replicates — this can refute the smoothness ordering, it cannot establish
the inverse one as a law, and `point_maze_medium` is missing (dataset absent from the pod), so the
four-point ordering the finding was designed around is incomplete. **L2:** the environments carry
differently-typed action variables — PointMaze forces on a point mass, PushT relative pusher
displacements, Wall dot velocities — so `cos(a_t, a_{t+1})` is not the same physical quantity at the
three points being correlated, and a point mass under near-random force commands still traces a smooth
path; this is the single most likely explanation of the inversion and it is a structural objection, not
a noise problem. **L3:** the gain differences are confounded with contact dynamics, a second movable
object, rotational state, and 2 training epochs on PushT against 20 elsewhere, so even a *concordant*
ordering would not have isolated the mechanism. The defensible form of N1 is therefore negative and
narrow: **action-reversal frequency, as measured on the recorded action variable, does not explain the
Table 1 gain ordering** — and any future claim that straightening helps where control is smooth needs a
type-comparable instrument, pre-registered before it is looked at.

**N2 — how much of the paper's curvature penalty falls on action-reversing transitions.** A
quantitative statement about the target paper's own objective, from its own data, for zero GPU time.
Under a `relu_cos` gate on the net commanded displacement, the share of curvature triples that would
receive **exactly zero** weight is **UMaze 49.83%, PushT 15.04%, Wall 7.85%** (`frac(w = 0)`, train
split, `sum`; identical to `frac(cos<0)` to four decimals, which is the gate's zero-set identity holding
on real data). The corresponding reallocation statistic `R = E|w − E[w]| / (2·E[w])` is **UMaze 0.5496,
PushT 0.2538, Wall 0.1857**. Read plainly: **roughly half of the paper's aggregated curvature penalty on
UMaze, and about one triple in seven on PushT, is applied to transitions where the commanded action
reversed direction** — transitions where a straight latent path is arguably the wrong target. This is
the one place the ACS premise survives measurement: the quantity is large enough to matter on all three
environments, which is why rule B returned GO. What N1 refutes is that its *size* tracks where
straightening helps. Same L2 caveat: "the commanded action reversed" means something different in a
force-controlled environment than in a displacement-controlled one.

**N3 — not reachable.** It required the `block_angle` R² of a trained ACS arm against the matched
control, and no arm was trained. The question — whether removing straightening pressure at direction
changes recovers the rotational state that `PROGRESS_CCR.md` §6f found curvature regularization
suppresses — remains open, and `MCA_Fallback` does not answer it either. Recorded as unanswered rather
than quietly dropped.

### 10.5 Stage-2 evaluation — _NOT YET RUN_

Per-seed values, never means alone (Requirement 16.15).

| setting | seed 100 | seed 200 | seed 300 | mean ± std | bar | verdict |
|---|---|---|---|---|---|---|
| open-loop | _tbd_ | _tbd_ | _tbd_ | _tbd_ | 79.33 | _tbd_ |
| MPC | _tbd_ | _tbd_ | _tbd_ | _tbd_ | 87.00 | _tbd_ |

---

## 11. Errors made — every one, including the ones that cost only minutes

CCR's convention (Requirement 16.9). An empty list here after the work is done would mean the log was
not kept, not that no mistakes were made.

| # | date | error | cost | how it was caught |
|---|---|---|---|---|
| 1 | 2026-08-08 | **Property 4 as specified in `design.md` was nearly vacuous.** The design states the gate-detachment property over `d(L_acs)/dz` only. Because `w` is a function of `act`, deleting `.detach()` from `action_gate` leaves `d(L_acs)/dz` **bitwise unchanged** — the test would have passed on precisely the attached gate Requirement 5.3 forbids, and the λ-reduction confound the whole design exists to eliminate would have shipped invisibly | minutes; caught before the term existed | Deliberate mutation check while writing task 3.4: `.detach()` was removed from the shipped source and the test still passed. Fixed by extending the substitution comparison to `act` and the encoder / `action_encoder` / `proprio_encoder` parameters; re-verified that 6 of 11 tests then fail on the mutant, and the source was restored |
| 2 | 2026-08-08 | **A relative float tolerance was the wrong instrument for the `sum` reduction** in task 3.5. When substeps cancel, the reduced vector is small while the addends are not, so the relative error is unbounded while the absolute error stays at `eps · Σ|x_s|` | minutes | hypothesis falsified the first draft immediately. Replaced with a derived per-triple bound `8·eps·(n + f·(κ₁+κ₂))` carrying the cancellation amplification `κ`, measured to have ~13x headroom over 6,000 draws |
| 3 | 2026-08-08 | **Stage-0 instructions handed to the operator contained an unresolved placeholder** — `export DATASET_DIR=/path/to/datasets` — and named the git remote `latest`, which exists only on the authoring machine and not on the pod (where it is `origin`) | one round trip, ~minutes, 0 GPU-h | The pod's `git pull` failed, so every command after it silently ran against the pre-`--readout` code and produced a misleading "the following arguments are required: --ckpt" instead of an obvious staleness error. A `git log --oneline -1` assertion was added to the reissued instructions; the real path `/workspace/arun/data` is recorded in four places in this repo and should have been read rather than placeheld |
| 5 | 2026-08-08 | **Handed the pod a `git commit` + `git push` sequence, which it has never been able to run.** The pod is a pull-only consumer of the repo with no commit identity and no credentials. `commit` died with "Author identity unknown" and the trailing `push` then blocked on an interactive `Username for 'https://github.com':` prompt that looks like a hang | one round trip, ~minutes, 0 GPU-h | The operator pasted the prompt back. Third instance of one root cause — assuming the pod's setup matches the authoring machine's — after the `latest`-vs-`origin` remote name and the `/path/to/datasets` placeholder. Written up as a standing protocol in `AGENT_MEMORY_2.0.md` §5.1 rather than left as three separate incidents: the pod pulls, results return by terminal paste |
| 4 | 2026-08-08 | **`point_maze_medium` is not on the pod**, so Stage 0 measured 3 of the 4 environments its rule is pre-registered over. Not checked before issuing the run | none to the verdict — clause 2.6 fires on PushT already being beaten 3.31x by UMaze, which no fourth value can change. Real cost is to N1's four-point ordering claim (§10.4, L1) | The fourth invocation raised `FileNotFoundError` on `states.pth`, and `--summarize` then refused to emit a verdict JSON rather than evaluating a four-environment rule on three — the guard behaved as designed |

Inherited errors worth not repeating, carried from `PROGRESS_CCR.md`:

1. **A derived constant instead of a measured one.** `rho = 0.05` was reasoned to, not measured, and
   the probe failed against it (§5a). ACS's structural answer: Stage 0 calls the **shipped**
   `reduce_action` and `action_gate`, so the Stage-0 `R` and the training-time `acs_gate_tv` are the
   same quantity computed by the same code, and check 1c compares them mechanically.
2. **A rule corrected after meeting its data.** The λ-selection rule named the wrong quantity and was
   fixed afterwards (§6a). ACS's answer is this file, written first.
3. **Calling loss shares "converged" off two data points.** They moved for another 120,000 steps (§4).
   ACS records **drift**, never a single row (§10.3).
4. **A wait loop that missed a zombie driver.** `ps -p <pid>` succeeds on a `Z` / `<defunct>` process;
   any wait loop must check `ps -p <pid> -o stat=` for `Z`.

---

## 12. If ACS stops

`MCA_Fallback` — `VWorldModel.compute_mca`, already written, reviewed, **never run**. No new module, no
new parameter, `<0.1%` overhead, **0.8 GPU-h to a verdict, zero new code.** It targets an orthogonal
gap: straightening is applied in the 128-d aggregated space while `planning/objectives.py` scores MSE
in the 1568-d patch space, and `encoder.agg` (1568→512→512→128 MLP with a terminal LayerNorm) is
neither an isometry nor injective. It is also **rotation-neutral**, so the §6f rotational-state
objection does not reach it. That is the named fallback on a Stage-0 STOP or an early-read-gate STOP.

Longer list: `PLAN_B_ALTERNATIVES.md`.

---

## 13. Implementation deviations from `design.md`

Appended during implementation. Nothing above this heading is edited: the pre-registered thresholds,
rules, limitations and the dated novelty positioning are evidence and stay as written. This section
records only where the shipped code differs from the design's literal text, and why.

### 13.1 `reduce_action` / `action_gate` take an explicit `env_action_dim` (tasks 3.1, 3.2)

Design §7.1/§7.2 write `reduce_action(self, act)` and `action_gate(self, act)`. **That signature is
not implementable.** `act` is `(b, t, f·d)` — `(b, t, 10)` at the PushT target cell — and `10` cannot
distinguish 5 substeps × 2 dims from 2 × 5. The environment action dimension never reaches the model:
`train.py` forwards `action_dim=action_emb_dim`, and `action_encoder.in_chans` is the already-packed
`f·d`. Guessing `d` would return a plausibly-shaped tensor with every gate weight wrong, which is the
F4 failure mode one level down.

Shipped signatures:

```python
def reduce_action(self, act, env_action_dim=None)
def action_gate(self, act, mask=None, env_action_dim=None)
```

Resolution order, in `_resolve_env_action_dim`: the explicit `env_action_dim` argument → an optional
`self.acs_env_action_dim` attribute a caller sets once from the dataset → `ValueError` naming
`act.shape[-1]` and the accepted remedy. **It never guesses.** `acs_action_reduce='raw'` needs no
dimension and short-circuits before the resolution.

Carried forward: **task 6.1 (`compute_acs`) and task 8.3 must thread the same argument**, and 8.3
supplies the value from the dataset's own `action_dim` (`dset.dataset.action_dim`, 2 on PushT) rather
than from a config constant. Requirement 5.15 ("resolve the substep count from `act.shape[-1]` and the
environment action dimension of the batch rather than from a configuration constant") is satisfied by
this threading, not weakened by it.

### 13.2 `action_gate` takes the static mask, because `permuted` needs it

The `permuted` null control (Requirement 13.4) shuffles `w` across the batch's **unmasked** triples,
so it has to see the mask `_cos_curvature_terms` returns. `mask` is therefore a parameter of
`action_gate`, read *only* by `permuted`; every other gate is elementwise and ignores it.
`mask=None` shuffles across the whole `(b, t-2)` tensor, which is the same thing when nothing is
masked. Masked entries keep their own values, so no dead position can absorb weight belonging to a
live triple.

### 13.3 `cos` is clamped to `[-1, 1]` before the gate dispatch

`F.cosine_similarity` can return values a few ulps outside `[-1, 1]` in float32, which would put
`relu_cos` and `affine_cos` marginally above `1` on near-parallel actions and make Requirement 5.4
(`0 <= w <= 1` elementwise) hold only approximately. One `clamp(-1, 1)` makes it exact. This is a
numerical guard on a mathematically-bounded quantity, not a threshold, exponent or sharpness constant,
so Requirement 5.17 is untouched. Zero-norm handling is unchanged: `cosine_similarity`'s own `eps`
still yields `cos = 0` (E10).

### 13.4 What `permuted` preserves *exactly*, and what it preserves to an ulp

Requirement 13.5 and task 9.4 ask for exact preservation of `mean(w)`, the weight distribution and
`gate_tv`. What the implementation guarantees exactly is the **multiset of unmasked weights** — the
permutation is a gather-shuffle-scatter over exactly the unmasked index set. Scalars *derived* from
that multiset by a float reduction are exact only up to summation order, because reordering the same
addends is not bitwise-neutral in float32. Measured at implementation time, 50 draws at
`b=16, t=5` (64–80 unmasked triples):

| quantity | max `|Δ|` vs `relu_cos` | exact on sorted values |
|---|---|---|
| multiset of `w[mask]` | 0 (bitwise) | — |
| `mean(w[mask])` | `8.94e-08` | yes |
| `gate_tv` | `1.19e-07` | yes |

So task 9.4 must assert multiset equality on **sorted** values (bit-exact) and compare `mean(w)` /
`gate_tv` either on sorted inputs or within a stated fp32 tolerance. This is a statement about float
arithmetic, not a weakening of the null control: no weight is created, destroyed or rescaled, and
Requirement 13.7's "match within batch noise" check at the telemetry level is unaffected — the
discrepancy is ~7 orders of magnitude below batch noise.
