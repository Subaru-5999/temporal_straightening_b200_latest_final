# PROGRESS — Counterfactual Curvature Regularization (CCR)

Live state of the CCR effort. Written so the work can be resumed cold, without the
conversation that produced it. Update this file at every decision point.

Spec: `.kiro/specs/counterfactual-curvature-regularization/`
Repo: https://github.com/Subaru-5999/temporal_straightening_b200_latest_final (branch `main`)
Pod checkout: `/workspace/arun/ccr`

---

## 1. What CCR is, in one paragraph

A new **loss term**, config-gated and **off by default**. The paper's `L_curv` penalises
curvature of latent trajectories the *dataset* visited. CCR applies the same curvature
penalty to trajectories the *predictor imagines* under **perturbed** actions. It targets an
unproven step in the paper's own argument: Proposition `app_cos` bounds `(A − I)` only along
*visited* velocity directions, and `Remark app_dir_vs_spec` then needs an unproven coverage
condition to reach Theorem 1's spectral bound. `GDPlanner` starts from a zero action sequence
and takes 100 Adam steps, so every iterate it evaluates is **off-log** — precisely the
directions the bound does not cover.

Approved direction: **D**, with **F (MCA)** piloted alongside, PushT as the target cell, a
dual acceptance gate, and the iteration cap accepted as in-scope infrastructure.

---

## 2. Status

| item | state |
|---|---|
| Implementation | **complete** |
| Test suite | **16/16 passing on the pod** |
| Baseline train (paper's method, CCR off) | **COMPLETE** — 123,858 steps, 12.04 h |
| Offline probe (rung 1) | **COMPLETE — gate PASS at `rho = 0.5`** |
| Baseline 3-seed eval (task 18.1) | **COMPLETE — 75.33 ± 6.11 OL / 82.00 ± 2.00 MPC, inside band** |
| Pilot arms (rung 2) | **not started — first launch stalled on a zombie chain, see §7a** |
| Full CCR run | not started — blocked on pilot gate |
| Acceptance gate verdict | not started |

Commits: `c86654c` implementation → `89c7df1` test fixes → `70fe2ee` telemetry `enabled`
flag → `150583a` measured gate recorded → `e8f6bd1` progress log + spec revision.

---

## 3. The Platform_Baseline run (done)

```bash
RUN_DIR=/workspace/arun/ccr/checkpoints/test/pusht_aggmlpcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05
LOG=ccr_full_20260806_050537.log        # driver pid 63736, exited cleanly
```

Launched with:
```bash
bash run_ccr_pilot.sh full training.lambda_cf=0 training.ccr_rho=0 training.mca_weight=0
```

This is **the paper's method, unmodified** — `L_pred + 0.1·L_curv`, 2 epochs, encoder lr
1e-5, batch 32, num_hist 3, frameskip 5. The three zeros switch the whole CCR path off.
Confirmed by: `Straightening enabled: mode=aggcos, scale=0.1`, `CCR disabled
(lambda_cf=0.0); MCA disabled (mca_weight=0.0)`, no `ccr` term in telemetry, and a run
directory byte-identical to the legacy name.

**Outcome:** 123,858 steps / 2 epochs in **12.04 h**. Train loss 0.0289, val loss 0.0289.
`model_2.pth` written 17:08 (sha256 `4d68b528…`, 265,381,955 bytes). Also validates the
default-off contract at runtime, and its first 8,000 steps are the **free matched-budget
control** (`SHORT_BUDGET_PILOTS.md` §4), which is why no control pilot arm is needed.

### 3a. Its measured evaluation (task 18.1, COMPLETE — 1 h 26 m)

`bash run_ccr_pilot.sh eval "$RUN_DIR"`, log `ccr_eval_baseline.log`, all six jobs OK.

| setting | measured | paper | prior B200 |
|---|---|---|---|
| open-loop | **75.33 ± 6.11** (74, 82, 70) | 77.33 ± 6.18 | ~75.3 |
| MPC | **82.00 ± 2.00** (82, 80, 84) | 85.33 ± 4.99 | ~82.0 |

**Verdict: inside the pre-registered band** (~75-78 OL / ~82-85 MPC), and within 1 SE of the
paper on both settings (−2.00 OL, −3.33 MPC against a ~5.7 pt binomial SE at n=50). It lands
essentially on top of the prior B200 reproduction, and the open-loop std of 6.11 against the
paper's 6.18 says even the noise structure matches. The reproduction is sound; the
Platform_Baseline is now measured, not assumed.

Read the open-loop per-seed values as the noise reality of this whole comparison: **74, 82,
70** — a 12-point spread over three seeds on the *same checkpoint*.

---

## 4. Measured facts (do not re-derive)

**Step rate:** median **2.862-2.865 it/s** over 619 telemetry records for the full run.
That is +1.2% step time vs the ~2.9 it/s in `REPRODUCTION.md`, so the documented figure is
valid on this pod. Requirement 11.7 floor = `2.862 / 1.5` = **1.91 it/s**.

**Matched-budget reference, `global_iter` 8000** — the row the pilot gate is judged against:

| term | scaled | share |
|---|---|---|
| curvature | 0.041421 | 73.741% |
| prediction | 0.013196 | 23.493% |
| decoder | 0.001554 | 2.767% |
| **total** | **0.056171** | 100% |

Raw on-log aggregated curvature = `0.041421 / 0.1` = **0.41421**. Note this is *not* the
quantity CCR's magnitude is set from — see §5.

**Share drift** (why the reference is read at 8,000 and nowhere else): curvature share
31.4% @200 → 65.4% @3000 → **73.7% @8000** → 80.5% @35,600 → 79.5% @84,400 → 82.7%
@123,858. Not monotone, plateaus around 80%. Driven by prediction falling 0.1585 → 0.0061
while curvature fell only 0.0770 → 0.0290. The paper's own configuration ends up ~80%
curvature. **It was wrong to call the shares "converged" off two data points; they kept
moving.**

**Probe cost:** ~78 s per arm on CPU, 64 windows × 4 draws. Cheap enough to sweep.

---

## 5. Rung 1 — the offline probe (COMPLETE, gate PASS)

### 5a. `rho = 0.05` failed, and that was a design calibration error

First run, at the originally specified `rho = 0.05`: aggregate `curvature_gap`
**−0.001259** against an unperturbed magnitude of 0.155470, i.e. **−0.8%**, with **0 of 5**
dimensions passing. Gate verdict FAIL.

Diagnosis, made before re-running: `normalize_action: True` makes actions unit-variance per
dimension, and `GDPlanner` takes 100 Adam steps at `lr = 0.1` from a zero initialisation, so
its iterates reach `O(0.5-1.0)` in normalized action units. **`rho = 0.05` is 10-20× smaller
than the region the planner actually explores** — it probed a neighbourhood where the
imagined trajectory is locally linear by construction. A widened criterion was declared
*before* the sweep was run.

### 5b. The sweep resolved it — premise CONFIRMED

| `rho` | dimensions passing | gap / unperturbed magnitude |
|---|---|---|
| 0.05 | 0 of 5 | −0.018 to 0.009 |
| 0.25 | 1 of 5 | — |
| **0.50** | **5 of 5** | 0.276 - 0.733 |
| 1.00 | 5 of 5 | 0.944 - 1.474 |
| 2.00 | 5 of 5 | 1.961 - 2.704 |

**`rho = 0.5` is selected**: the smallest value clearing all five dimensions, so the least
extrapolation away from the recorded action distribution while still inside the planner's own
operating range. Off-log trajectories *are* measurably more curved than on-log ones once the
perturbation is scaled to the planner.

Reports: `probe_outputs/ccr_pusht.json`, `probe_outputs/ccr_pusht_rho*.json`.

### 5c. Independent finding: `state_readout_r2`

Ridge readout R² per dimension, from the same probe:

| dimension | R² |
|---|---|
| block_x | 0.800 |
| block_y | 0.735 |
| agent_x | 0.728 |
| agent_y | 0.502 |
| **block_angle** | **0.183** |

PushT success is an orientation-sensitive criterion. `block_angle` is simultaneously the
**worst-encoded** dimension and the one with the **weakest `curvature_gap` ratio at every
`rho`**. Whatever CCR does to the latent geometry, it does not obviously fix the dimension
the task is scored on. This is the strongest single piece of evidence *against* the
direction, and it came free.

### 5d. Known probe limitation, now fixed in code

The `pristine` reference source failed with `expected self and mask to be on the same device,
but got mask on cuda:0 and self on cpu`, so every readout came back
`reference_value: n/a`. Root cause: `models/vit.py:58` sets
`self.bias = generate_mask_matrix(...).to('cuda')` as a **plain attribute, not a registered
buffer**, so `nn.Module.to("cpu")` never moves it. A checkpoint-loaded model escapes this
because the predictor is pickled whole and `map_location="cpu"` rewrites the attribute; a
freshly instantiated predictor does not.

Fixed in `probe_ccr_curvature.py` via `_plain_tensor_attrs_to_cpu`, called from
`_freeze_for_probe`, rather than in `models/vit.py` — that file is outside the Requirement
5.6 changed-file allowlist and its `cuda` default is what every training and planning run
relies on. **Not yet re-run with `--reference pristine`**; the readouts above stand on their
own (the gate criterion is relative to the *unperturbed* curvature, which is measured in the
same pass), the reference was only ever a "not yet degraded" sanity anchor.

---

## 6. The gate, recorded before the probe (Requirement 8.1)

**Probe gate.** Mechanism present iff aggregate `curvature_gap` is positive **and** ≥20% of
the unperturbed curvature magnitude, on **≥3 of the 5** disaggregated dimensions
(`agent_x, agent_y, block_x, block_y, block_angle`). → **PASS at `rho = 0.5`.**

**λ selection — CORRECTED.** The rule originally read `lambda_cf = 0.024 / g` with raw CCR
assumed to be `g × 0.41421` (the on-log curvature times the perturbed/unperturbed ratio).
**That is the wrong quantity.** CCR is evaluated on the imagined *off-log* rollout, and its
magnitude is the perturbed *level*, not a ratio. The probe measures it directly:

```
c = raw CCR term = perturbed imagined curvature
  = 0.155470 (unperturbed) + 0.073174 (gap)
  = 0.228644            at rho = 0.5      # 0.55x the value the old rule assumed

lambda_cf = 0.00994 / c   ->  ~15% CCR share (target)
lambda_cf = 0.02407 / c   ->  30% CCR share (hard ceiling, do not exceed)
```

Resolved against the measured `c` and the step-8,000 total of 0.056171:

| `lambda_cf` | scaled CCR `X` | CCR share | verdict |
|---|---|---|---|
| 0.02 | 0.00457 | 7.5% | inside the window but too weak to be informative |
| **0.04** | **0.00915** | **14.0%** | **treatment arm** |
| 0.08 | 0.01829 | 24.6% | weight-variation arm; leaves drift headroom |
| 0.10 | 0.02286 | 28.9% | admissible, but no headroom for the upward drift |
| 0.30 | 0.06859 | 55% | out — past the ceiling, prediction share collapses |

**Sweep pair: `{0.04, 0.08}`.** Two earlier pairs are superseded: `{0.1, 0.3}` (no basis)
and `{0.02, 0.05}` (wrong denominator). **The earlier claim that `λ = 0.1` was "4-15× too
strong" was wrong** — at the measured `c` it lands at 29% and is admissible.

**Pilot gate, all four checks:**
1. `ccr` appears in the telemetry record's `enabled_terms` (equivalently `enabled: true` in
   the `ccr` block). **This is the primary confirmation** — it derives from the model's gate
   firing, not from config. `synthesized_action_frames == 3` is a *secondary*
   synthetic-vs-logged check, read only after CCR is confirmed enabled. (The old field read
   `3` on a CCR-disabled baseline, so it never confirmed CCR was running.) Checkpoint on disk
   within seconds; an empty dir a minute in means a crash.
2. `it_per_s >= 1.91`. A CCR arm at the upper end of the estimated +30-50% step-time cost
   lands at ~1.91 and **grazes** the floor — under Requirement 11.7 that is a reporting event
   before the Full_Run, **not an abort**.
3. At `global_iter` **8000**: CCR share in `[2%, 30%]` **and** prediction share `>= 11.75%`.
   Equivalently scaled CCR contribution `X in [0.0011, 0.0241]`. The 30% cap binds; the
   prediction floor is slack by >2×.
4. Raw CCR term does **not** fall below `1e-3` in the first 1,000 iterations. If it does, the
   term absorbed the task without pressuring the encoder → **not a success**.

**Acceptance gate (dual) — now resolvable to absolute numbers.** Pass requires beating
**77.33 OL and 85.33 MPC** (paper) *and* both measured Platform_Baseline rates
(**75.33 OL / 82.00 MPC**), with a margin over the baseline above 6 pts. One condition alone =
failure. Taking the max of each pair of constraints:

| setting | paper leg | baseline + 6 pt margin | **binding target** | delta CCR must add |
|---|---|---|---|---|
| open-loop | 77.33 | 75.33 + 6 = 81.33 | **81.33** | **+6.0** |
| MPC | 85.33 | 82.00 + 6 = 88.00 | **88.00** | **+6.0** |

The margin rule binds on both settings, so the gate is asking for exactly the ~6.6 pt effect
the noise floor says is the minimum detectable one. Worth noting that the baseline landing at
the *low* end of its band (82.00 MPC rather than 85) **lowered** the absolute bar: had it come
in at 85, the MPC target would have been 91.00. Use `python ccr_acceptance_gate.py`.

---

## 6a. Go/no-go rule for the Full_Run — PRE-REGISTERED 2026-08-07 04:15, before the iter-8000 row

**The 8,000-step row can veto but cannot endorse.** It is a safety check, not an efficacy check: a
pilot predictor is ~7x worse on `z_loss` than a finished one, so nothing here predicts the final
success rate. A clean pass leaves the estimate at ~10-12%. Written down before the data to stop a
story being fitted to it afterwards.

The informative quantity is the **prediction loss against the baseline's own row at the same step**
(0.013196), because that is the causal channel by which CCR would hurt: planning descends on latent
distance, so a degraded predictor loses success rate whatever the geometry does.

| condition at `global_iter` 8000 | verdict |
|---|---|
| prediction ≤ **0.014516** (within 10%), CCR share in [2%, 30%], prediction share ≥ 11.75%, no collapse | **GO** — controls + Full_Run; estimate stays ~10-12% |
| prediction > **0.016495** (>25% worse) | **STOP** — CCR is materially degrading the predictor; switch to B1 |
| in between, or any criterion marginal | **PROBE, then decide** (below) |

**Tiebreaker: re-probe the pilot checkpoint** (~78 s, CPU, read-only). Baseline reference is gap
ratios 0.276-0.733 and `block_x 0.800, block_y 0.735, agent_x 0.728, agent_y 0.502,
block_angle 0.183`.

| probe outcome | meaning | estimate |
|---|---|---|
| off-log curvature gap materially reduced | the term does what it was built to do | supports GO |
| `block_angle` still last by a wide margin (~0.18) | the §8(b)/§5c objection stands | ~10%, argue for B1 |
| `block_angle` improved **relative to the other four** | CCR is touching the dimension PushT is scored on | ~20%, argue for the Full_Run |

Caveat that must be applied to the last two rows: the pilot has seen 8,000 steps against the
baseline's 123,858, so a lower absolute R² is uninformative. **Only the ranking among the five
dimensions is comparable.**

## 6b. Pilot result at `global_iter` 8000 — VERDICT: middle band, probe then decide

Run `checkpoints_fast/test/..._cf0p04_rho0p5_srcsynthetic_mca0`, driver 4077128, clean finish
05:09:27, 8,000 steps in 6,690 s at a steady **1.198 it/s** (40 records, min 1.191 max 1.200).

| term | scaled | share | baseline @8000 | delta |
|---|---|---|---|---|
| curvature | 0.043296 | 68.505% | 0.041421 | +4.5% |
| prediction | **0.015428** | 24.411% | **0.013196** | **+16.9%** |
| ccr | 0.002848 | 4.506% | — | — |
| decoder | 0.001630 | 2.578% | 0.001554 | +4.9% |
| total | 0.063202 | | 0.056171 | +12.5% |

Four of five gate criteria PASS (term enabled with `synthesized_action_frames=3`; step-200 row
matches; shares inside `[2%, 30%]` and prediction share 24.4% >> 11.75% floor; no collapse). The
step-rate floor FAILS at 1.198 vs 1.933 — the known Requirement 11.7 reporting event.

Applying §6a: prediction 0.015428 sits between the GO bound (0.014516) and the STOP bound
(0.016495). **Middle band → probe the pilot checkpoint, then decide.**

### The two findings that matter more than the gate

**(1) The prediction-loss cost is a trend, not noise.** I had dismissed this off a single row. Per-row
delta of our prediction loss against the reference at matched `global_iter`:

```
200  400  600  800 1000 | 1200 1400 1600 1800 2000 2200 2400 | 8000
 −    −    −    −    −  |  +3%  +9% +14% +22% +18%  +7% +18% | +17%
 better than baseline    | consistently worse, 8 of 8 rows
```

Eight consecutive same-direction rows is p ≈ 0.004 under a sign test. **CCR costs ~15% on the
prediction loss from ~1,200 steps onward**, and that is the quantity planning descends on.

**(2) CCR has largely solved itself.** Raw CCR: 0.3395 @200 → 0.3021 @400 → **0.0712 @8000**, a
**79% reduction**, against the on-log curvature term's 43% over the same span. Share down to 4.51%
and still falling. The off-log curvature is far *easier* to reduce than the on-log curvature.

Reads well on its face — the mechanism works, the encoder does straighten off-log rollouts. But for
the Full_Run it cuts the other way: a penalty 79% satisfied by step 8,000 exerts little pressure
over the remaining 116,000 steps, so the run would pay the 15% prediction cost for the full distance
while the geometric benefit is already banked and fading. **Measured cost, vanishing benefit.**

Estimate revised **~10% → ~7%**.

### Operational note

The driver went `Zs` / `[bash] <defunct>` on exit and the ad-hoc monitor loop I supplied used the
naive `kill -0`, so it span for 2h39m past a job that finished at 05:09 — **2h39m of idle GPU on a
free slice.** This is the second time the same zombie trap cost time; `run_ccr_pilot.sh` was already
fixed for it and the throwaway loop was not. Any wait loop must check `ps -p <pid> -o stat=` for `Z`.

## 6c. Probe of the pilot's 8k checkpoint — "GATE FAIL" means CCR SUCCEEDED

`probe_outputs/ccr_pilot8k.json`, 68.6 s, `rho=0.5 L=5 synthetic`, checkpoint sha256
`0ea15e21…` unchanged. The `pristine` reference now resolves (the device fix works: "Moved 6
non-buffer tensor attribute(s) to CPU").

**Do not read the headline verdict at face value.** The gate asks "is there excess off-log
curvature available to fix?" That is the right question for a *baseline* checkpoint and the wrong
one for a *CCR-trained* checkpoint, where a FAIL means the target was eliminated:

| quantity | baseline ckpt | CCR ckpt @8k | change |
|---|---|---|---|
| unperturbed off-log curvature | 0.155470 | **0.032888** | **−79%** |
| perturbation-induced gap | 0.073174 | **0.005352** | **−93%** |
| dimensions passing the gate | 5 of 5 | 1 of 5 | target removed |

**CCR is maximally effective at its stated objective**, achieved in 6.5% of the training budget.
Links 1 and 2 of the mechanism chain are now both confirmed: off-log trajectories are more curved,
and CCR straightens them almost completely. Only link 3 — does that raise success rate — is open.

### `state_readout_r2`: better on all five, but confounded

| dimension | baseline (124k steps) | CCR (8k steps) | delta |
|---|---|---|---|
| aggregate | 0.675878 | **0.733137** | +0.057 |
| agent_x | 0.727565 | 0.769019 | +0.041 |
| agent_y | 0.502438 | 0.618116 | +0.116 |
| block_x | 0.799777 | 0.837943 | +0.038 |
| block_y | 0.734649 | 0.741352 | +0.007 |
| block_angle | 0.183020 | 0.200866 | +0.018 |

(`pristine` reference for context: aggregate 0.504, `block_angle` −0.995.)

**Unattributable as it stands.** 8,000 steps versus 123,858 — the confound flagged in §6a, running
in the favourable direction. The baseline's own 8,000-step checkpoint does not exist, because
`save_ckpt` overwrites `model_latest.pth` and only epoch-boundary files survive.

**On the §6a criterion this is the "argue for B1" branch.** `block_angle` did not improve relative to
the other four: its gap to fourth place *widened*, 0.319 → 0.417. Still last by a wide margin.

## 6d. NEXT: matched 8,000-step control — ~47 min, resolves the confound

The pre-registered rule did not anticipate a uniform readout improvement. Worth 47 minutes to
resolve rather than dismissing on a technicality. Train CCR-off for the same 8,000 steps, probe with
identical flags, compare the five readouts:

```bash
CKPT_BASE=$PWD/checkpoints_ctrl8k bash run_ccr_pilot.sh pilot \
  training.lambda_cf=0 training.ccr_rho=0 training.mca_weight=0
```

| control aggregate R² | interpretation | estimate |
|---|---|---|
| ~0.73 | the gain was training length, CCR did nothing to the representation | ~7-10%, stop and write up the negative |
| ~0.60-0.65 | **CCR genuinely improves state readout** — a different and more direct mechanism than gradient conditioning | **~25-30%, argue for the Full_Run** |

Why the second branch would matter: "better representation → better planning" is a far shorter causal
chain than "better conditioning → better gradient descent → better planning", and it would sidestep
the §8(b) objection entirely, since it does not depend on PushT's conditioning being the bottleneck.

## 6e. MATCHED 8k CONTROL — the decisive result. RECOMMENDATION: STOP.

Control: `checkpoints_ctrl8k/test/pusht_..._lr1e-05` (legacy name, all knobs default), 8,000 steps in
46:47 at 2.85-2.88 it/s — confirming the CCR arm's 2.4x slowdown (2.87 → 1.198). Probes
`probe_outputs/ctrl8k.json` and `ccr_pilot8k.json`, identical flags, seed, and 64 windows.

### CCR works, confirmed causally

| | control @8k | CCR @8k | change |
|---|---|---|---|
| off-log curvature | 0.196010 | 0.032888 | **−83%** |
| perturbation gap | 0.068658 | 0.005352 | **−92%** |
| gate | 5/5 PASS | 1/5 | target eliminated |

Matched on everything but CCR, so this is caused by the term, not by training length.

### And it degrades the dimension PushT is scored on

| dimension | control @8k | CCR @8k | delta |
|---|---|---|---|
| aggregate | 0.701269 | 0.733137 | +0.032 |
| agent_x | 0.746992 | 0.769019 | +0.022 |
| agent_y | 0.534881 | 0.618116 | +0.083 |
| block_x | 0.868711 | 0.837943 | −0.031 |
| block_y | 0.712489 | 0.741352 | +0.029 |
| **block_angle** | **0.278188** | **0.200866** | **−0.077 (−28%)** |

The §6a decider was `block_angle` against a matched control. **CCR makes it worse.**

## 6f. The finding worth keeping: curvature regularization suppresses rotational state

**Rotation *is* curvature.** A rotating object traces an arc in feature space; its latent velocity
direction changes continuously by construction. `L_curv` minimizes `1 − cos(v_t, v_{t+1})`, which is
zero only when consecutive velocities are parallel. Rotation cannot be straight, so a curvature
penalty is in direct tension with encoding orientation.

Four independent pieces of evidence:

1. `block_angle` is the worst-encoded dimension in the paper's own trained model — 0.183 against
   0.50-0.80 for the four positional dimensions — and the paper's method *is* a curvature penalty.
2. It **degrades with training**: 0.278 @8k → 0.183 @124k in CCR-off runs. The longer the penalty is
   applied, the more orientation information is lost.
3. CCR, a second curvature penalty, reaches 0.201 by step 8,000 — most of the baseline's final
   degradation, in 6.5% of the budget.
4. Table 1's GD improvements from straightening: **+50.00** UMaze, **+10.67** Medium, **+10.67** Wall
   — all pure-position state — versus **+7.33** PushT, the only task with rotational state and the
   smallest gain of the four.

This explains a pattern in the paper's own results that the paper does not address, and it is a
sharper contribution than "we beat Table 1" would have been. Cost: ~24 GPU-h, not ~52.

**Estimate: ~5%.** No longer inferred from the paper's numbers — measured against a matched control
on the quantity the task is scored on. **Recommendation: do not launch the Full_Run.**

### Robustness check before writing it up (~8 min, CPU)

Single probes with 22 windows per per-dimension subset. Re-probe both checkpoints at
`--num-windows 192` (subsets grow 22 → 64). If `block_angle` is still clearly lower under CCR, the
finding holds.

## 6g. TRAINING ON THIS POD IS BITWISE DETERMINISTIC — verified, and it changes the statistics

`summarize_training_log.py "$CTRL_DIR" --compare "$RUN_DIR"`: **all 40 matched rows agree to
`+0.000000`** on `prediction`, `curvature`, `decoder` and `loss`, at `global_iter` 200 through 8000.
The `checkpoints_ctrl8k` run is an exact reproduction of the original baseline's first 8,000 steps.
Only `it_per_s` differs, which is wall-clock, not numerics.

Two consequences, both important:

**1. Zero confound.** Any CCR@8k vs control@8k difference is attributable *entirely* to CCR. There is
no run-to-run variation to subtract. This was an assumption I had been making without checking; it is
now verified.

**2. My eval power analysis was wrong and too pessimistic.** I computed `SE(Δ) ≈ 9.8` points by
treating the two arms as independent binomials. They are **paired**: `plan.py` seeds episode sampling
from `seed`, and the planner is deterministic (`sample_type: zero`, `action_noise: 0`), so both arms
see the *identical* 50 episodes. With deterministic training the pairing is exact.

For a paired binary comparison the noise comes only from discordant episodes: if the models differ on
`k` of 50, `SE(Δ) ≈ sqrt(k)/50` — 6.3 pts at k=10, 4.5 pts at k=5. More usefully: **the single-seed
result is an exact measurement of the difference on that episode set**, with no measurement error at
all. The only open question is generalization to other episodes, which is what extra seeds buy.

So a difference of even 3-4 episodes (6-8 points) is real on that set. Extend to 3 seeds before
believing the magnitude, but the sign is trustworthy from one seed.

**Reusable consequence:** determinism means any future matched comparison needs only one run per arm
to isolate an effect exactly. It also means the "free matched-budget control" idea in
`SHORT_BUDGET_PILOTS.md` §4 extends from telemetry to *checkpoints* — a capped rerun reproduces any
earlier prefix exactly, so a lost intermediate checkpoint can always be regenerated for its cost in
steps.

## 6h. FINAL: matched 8k success-rate comparison — CCR LOSES ON BOTH. STOP.

`SEEDS=100`, one seed, both arms under the unmodified Evaluation_Protocol. Training is bitwise
deterministic (§6g) and episodes are seeded identically, so these are **exact paired differences**,
not estimates: counts of episodes, 2 percentage points each.

| arm | open-loop | MPC |
|---|---|---|
| CTRL @8k | 16.0 | 18.0 |
| CCR @8k | 14.0 | 10.0 |
| **Δ (CCR − control)** | **−2.0 (1 episode)** | **−8.0 (4 episodes)** |
| BASE @124k (the gate comparator) | 75.33 ± 6.11 | 82.00 ± 2.00 |

### Every downstream measurement, collected

| measurement | result |
|---|---|
| off-log curvature gap | **−96%** — CCR works perfectly at its stated objective |
| prediction loss vs matched control | **+16.9%** worse, 8/8 consecutive rows, p ≈ 0.004 |
| state readout, aggregate, matched, n=192 | +0.005 — nothing |
| state readout, `block_angle`, matched, n=192 | −0.035 (−9%; was −28% at n=64, so mostly noise) |
| success rate, open-loop, matched | **−2.0 pts** |
| success rate, MPC, matched | **−8.0 pts** |
| step time | **2.4x** slower → 29 h Full_Run |

**CCR does exactly what it was designed to do and none of it converts.** Dual gate estimate:
**~3-4%**. **Do not launch the Full_Run.**

### Limits of this conclusion, stated fairly

8,000 steps is 6.5% of the budget; both arms sit near the floor (10-18%); the test is structurally
biased against CCR because its cost is immediate while any benefit may need longer to convert; and
one seed does not establish generalization to other episode sets. This is **not proof that CCR fails
at full budget.** It is the absence of any positive signal across five downstream measurements,
against four negative ones, with the burden of proof on CCR to justify 29 GPU-hours.

### What the CCR effort established (the writeup)

1. **A clean platform reproduction** of the paper's PushT target cell: 75.33 ± 6.11 OL / 82.00 ± 2.00
   MPC against the paper's 77.33 ± 6.18 / 85.33 ± 4.99, within 1 SE on both.
2. **The paper's unproven coverage condition, measured.** Proposition `app_cos` bounds `(A − I)` only
   along visited velocity directions. Off-log trajectories *are* measurably more curved — 5/5 state
   dimensions at `rho = 0.5`, ratios 0.29-0.42 — so the gap the bound leaves open is real, not
   hypothetical.
3. **Off-log curvature is almost entirely removable, cheaply.** 96% of the perturbation-induced gap
   eliminated in 6.5% of a training budget.
4. **And removing it does not help planning.** −2.0 OL / −8.0 MPC at matched budget, with a 16.9%
   prediction-loss cost. So latent-geometry conditioning off the data manifold is *not* the binding
   constraint on PushT.

Total cost ~26 GPU-h. That is a coherent negative result about a specific, previously unexamined step
in the paper's argument.

## 7. Next actions, in order

Common preamble for every command below:
```bash
cd /workspace/arun/ccr && git pull origin main
export DATASET_DIR=/workspace/arun/data          # run_ccr_pilot.sh dies without it
RUN_DIR=$PWD/checkpoints/test/pusht_aggmlpcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05
```

### Step 0c — Requirement 11.7 REGRESSION REPORT (preliminary, step-200 row only)

Pilot `..._cf0p04_rho0p5_srcsynthetic_mca0`, driver 4035595, launched 02:32:20 on 2026-08-07.

**Three of four pilot gate checks pass.** CCR is confirmed running from the model's own gate:
`enabled True`, `raw 0.339185`, `action_source synthetic`, `synthesized_action_frames 3`,
`rho 0.5`, `rollout_len 5`, `grad_checkpoint True`. Shared terms match the reference at step 200
inside `rtol=0.05` (curvature −0.0012, prediction −0.0027, decoder +0.0002), so the arm is a
clean twin of the baseline apart from CCR. No collapse.

Shares at `global_iter` 200: prediction 61.101%, curvature 29.726%, **ccr 5.322%**, decoder
3.851%, total 0.254924. The 5.3% is *not* a miss — λ was calibrated against the
iteration-8,000 total of 0.056171, and the total at step 200 is 4.5x that. Raw CCR is
**0.339185** against the probe's predicted `c = 0.228644` (1.48x higher), so if raw holds the
share at 8,000 lands near **19-20%** rather than the 14% target — above target, inside the
`[2%, 30%]` window, no action needed.

**Check 2 FAILS: 0.573 it/s against the 1.93 floor, +406% step time.**

Fitting `T_step = E + n·p`, with `p` one predictor pass and `E` everything else:

```
baseline   E +  2p = 0.3925 s     (1 forward + 1 backward of predict)
CCR L=5    E + 17p = 1.745  s     (+5 calls x 3 passes each under checkpointing)
        -> p = 0.090 s,  E = 0.212 s
```

**One predictor forward is 0.090 s — about 30% of an entire baseline step.** The design
asserted the encoder pass dominates and the extra predictor calls are "a small fraction" of it.
That is wrong, and it is the *same* unquantified assumption that produced the OOM. Both
failures come from one gap: the predictor's cost, in time and in memory, was never measured.
`models/vit.py` runs `depth=6`, `heads=16` over `T = num_hist * num_patches = 588` tokens with
a materialised `(b, h, T, T)` attention matrix.

Projected Full_Run at 0.573 it/s: `123,858 / 0.573 = 216,157 s` = **60.0 h**, against the 17 h
in the recorded plan.

**CONFIRMED, not warmup.** Six rows: 0.5725 @200, then 0.5892, 0.5890, 0.5892, 0.5896, 0.5887
through iteration 1,200. Sustained rate **0.589 it/s**, reference 2.84-2.88 at the same
iterations. Full_Run on this path = `123,858 / 0.589` = **58.4 h**.

**CCR is learnable — a real positive signal.** Raw CCR fell 0.339185 @200 → 0.199388 @1200, a
41% reduction, while its share *rose* 5.32% → 6.67% because the total loss fell faster. So the
encoder is genuinely straightening the off-log rollouts rather than the term being trivially
absorbed. Gate check 4 (raw not below 1e-3 inside 1,000 iterations) passes with four orders of
magnitude to spare.

**Watch, do not yet conclude:** at iteration 1,200 our prediction loss is 0.055630 against the
reference's 0.048998, i.e. 13.5% worse — the "CCR squeezes the prediction loss" risk. But at
1,000 it was 3.3% *better* (0.052731 vs 0.054557), so this is row-to-row noise, not a trend.
Mid-run readings are failure detectors, not trajectories.

**RESOLVED by `ccr_fast_attention` (measured 03:16-03:22 in `checkpoints_fast/`).** SDPA gives
**1.196 it/s** against the materialised path's 0.589 — a **2.03x** speedup, steady over rows 200
(1.135, warmup-depressed) and 400 (1.196). Refitting: `E + 2p_slow + 10p_fast = 0.836 s` with
`E = 0.212`, `p_slow = 0.090` gives **`p_fast = 0.0444 s`**, i.e. SDPA halved the predictor pass.

| path | it/s | Full_Run |
|---|---|---|
| materialised attention | 0.589 | 58.4 h |
| **SDPA on the CCR rollout (shipped default)** | **1.196** | **28.8 h** |
| SDPA everywhere (needs a baseline retrain) | ~1.34 | ~25.7 h |
| baseline reference | 2.86 | 12.0 h |

The third row is why the default-off design stands: 3 h saved for a 12 h retrain plus a 1.5 h
re-eval, and it would invalidate the measured 75.33/82.00. Closed.

Still FAILS the 1.93 it/s floor at +142% step time. That floor came from the same wrong cost
model — no configuration adding five predictor rollouts can clear a 50% step-time bound — so it
is doing its job as an alarm, not as an achievable target.

**Options, with modelled rates (superseded above for the fast-attention row).** All of them are
decisions for the user, not defaults to apply:

| option | rate | Full_Run | cost |
|---|---|---|---|
| as-is (`L=5`, checkpointing) | 0.57 | **60 h** | none, but unaffordable |
| CCR every 4th step | ~1.37 | ~25 h | weaker effective pressure; raise λ to compensate |
| CCR every 8th step | ~1.78 | ~19 h | weaker still |
| CCR on 8 of 32 samples | ~1.37 | ~25 h | noisier estimate, same expectation |
| `L=2` + checkpointing | ~1.07 | ~32 h | loses the horizon argument, and is the `logged` control |
| `models/vit.py` -> SDPA | fast | ~12-15 h | **out of scope**, changes baseline numerics, invalidates the 12 h baseline train and its measured 75.33/82.00 |

Note the 1.93 it/s floor was itself derived from a wrong cost model: no configuration that adds
five predictor rollouts can clear a 50%-step-time bound. The floor is doing its job as an alarm,
but it was never an achievable target for this term.

### Step 0b — the relaunched pilot OOM'd; fixed with gradient checkpointing

`torch.OutOfMemoryError` inside `F.softmax` in the predictor's attention, ~2 min into the run.
**Not a tuning problem — the overrun is ~40 GB, and it is a design error of mine.**

The arithmetic I never did. `models/vit.py` materialises the full attention matrix:
`dots` has shape `(b, heads, T, T)` where `T = num_hist * num_patches = 3 * 196 = 588`, so
`(32, 16, 588, 588)` = 177 M elements = **354 MB in bf16**. The softmax output and the
`dropout=0.1` output are each the same size again, for each of `depth=6` layers. So **one
`predict` call stores ~8 GB of activations for backward.**

The baseline objective calls `predict` once. CCR at `L = 5` calls it **five** more times →
**~40 GB of extra activation memory** on a 45 GB slice. The design said the extra predictor
calls were "a small fraction" of the encoder pass; that was true of *compute time* and simply
wrong about *memory*, which I never analysed. Note the `logged` control arm at `L = 2` would
have been marginal too (~16 GB extra), so this was going to bite either way.

**Fix:** `training.ccr_grad_checkpoint` (new, default `true`) recomputes the CCR rollout's
predictor calls in backward instead of storing them, via
`torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`. Peak drops from ~40 GB extra to
~8 GB extra. Costs one extra forward on the CCR path only, so expect the step rate to sit
lower — the `it_per_s >= 1.91` floor is now the check to watch, and under Requirement 11.7 a
breach is a reporting event before the Full_Run, not an abort.

Numerically neutral: the default `preserve_rng_state=True` saves and restores the RNG around
the recomputation, which matters because the predictor runs `dropout=0.1` — without it the
recomputed forward would draw a different mask and the gradient would be wrong.
`_rollout_latents` takes `checkpoint=False` by default, so `rollout`, `plan.py`, `planning/*`
and `Trainer.openloop_rollout` are untouched and Property 7 still holds bitwise. The knob is
in neither `ccr_tag` nor `LOSS_SIGNATURE_KEYS`, because toggling it must not rename a run or
block a resume.

**Also seen:** `tail: inotify cannot be used ... Too many open files`. Separate from the OOM
(it is the interactive shell, not the trainer), but it points at leaked file descriptors from
the accumulated runs. Check `ulimit -n` and clear stray processes if it recurs.

### Step 0 — the earlier chained pilot launch FAILED to start; relaunch unchained

**What happened (2026-08-06/07).** The baseline eval (driver 4032390) finished at 19:47:55.
The pilot (driver 4032433) was chained on it with `CHAIN_ON_PID` and was still sitting in
`wait_for_driver_pid` **7 h 46 m later**, having never launched `train.py`: no
`Iteration budget` line, no `CCR enabled` line, no run directory. ~6 h of idle GPU.

**Cause.** `while kill -0 "$pid"; do sleep 30; done` is not a sufficient exit condition.
`setsid` detaches the driver, so on exit its parent is PID 1 — and in this container PID 1 does
not reap. The finished eval driver lingered as a **zombie**, its pid stayed in the process
table, `kill -0` kept succeeding, and the loop waited on a job that had already finished.

**Fixed** in `run_ccr_pilot.sh`: the loop now reads `ps -o stat=` and treats `Z*` or an empty
state as gone, and prints a heartbeat every 30 min so a silent multi-hour wait is
distinguishable from a hang. Confirm with `ps -p <pid> -o pid,stat,etime,cmd` — a `Z` or
`<defunct>` is a finished job, not a running one.

**Recovery.** The slice is free, so chain nothing:
```bash
kill 4032433                                  # the stuck driver; it holds no GPU memory
bash run_ccr_pilot.sh pilot \
  training.lambda_cf=0.04 training.ccr_rho=0.5 \
  training.ccr_action_source=synthetic training.ccr_rollout_len=5 \
  training.mca_weight=0
```

### Step 1 — baseline 3-seed eval (task 18.1) — DONE, see §3a
**This goes before the pilots**, even though the spec numbers it with the acceptance-gate
group. It depends on nothing but the baseline checkpoint, which is on disk, and it is the only
check that the whole `plan.py` evaluation path produces a number in the paper's band. Run
early it costs 1.5 h and de-risks everything downstream; run late it is a number that arrives
after ~24 GPU-h have already been staked on comparisons against it.

```bash
LOG=ccr_eval_baseline.log bash run_ccr_pilot.sh eval "$RUN_DIR"
BASE_PID=$(cat ccr_eval_baseline.pid)
```

One command, six jobs, serial in one driver: open-loop (`plan_gd.yaml`, `mode=last`,
`alpha=1`) for seeds 100/200/300, then MPC (`plan_gd_mpc.yaml`, `mode=staged`, `alpha=1`) for
the same three. `PLAN_SERIAL_ENV=1` is applied automatically in `eval` mode.

```bash
grep -ah success_rate ccr_eval_baseline.log | tail -n 20
python aggregate_results.py
```

Expect **~75-78 OL / ~82-85 MPC**. Outside ~72-82 open-loop is **stop-and-investigate in
either direction** — a number well above the paper's mean is as much a sign of a protocol
discrepancy as one below it. Report the ~5.7 pt binomial SE alongside.

### Step 2 — treatment pilot arm, chained behind the eval (task 15.1, ~85-95 min GPU)
```bash
CHAIN_ON_PID=$BASE_PID bash run_ccr_pilot.sh pilot \
  training.lambda_cf=0.04 training.ccr_rho=0.5 \
  training.ccr_action_source=synthetic training.ccr_rollout_len=5 \
  training.mca_weight=0
PILOT_PID=$(cat ccr_pilot_*.pid | tail -1)
```
Expected run dir:
`checkpoints/test/pusht_..._sgTrue_lr1e-05_cf0p04_rho0p5_srcsynthetic_mca0`

**Two-minute check** — `ccr` in `enabled_terms` (primary), `synthesized_action_frames=3`
(secondary), a checkpoint on disk. A `synthetic` arm reporting `synthesized_action_frames=0`
is silently a `logged` arm and the launch is wrong.

### Step 3 — remaining pilot arms, serial, chained on the driver PID
```bash
# horizon control — does the gain need the steps past the window edge?
CHAIN_ON_PID=$PILOT_PID bash run_ccr_pilot.sh pilot training.lambda_cf=0.04 training.ccr_rho=0.5 \
  training.ccr_action_source=logged training.ccr_rollout_len=2

# perturbation control — rollout-space vs encoder-space, isolated from off-log vs on-log
CHAIN_ON_PID=<prev> bash run_ccr_pilot.sh pilot training.lambda_cf=0.04 training.ccr_rho=0

# λ variation (DROP THIS ARM if the compute overrun is refused)
CHAIN_ON_PID=<prev> bash run_ccr_pilot.sh pilot training.lambda_cf=0.08 training.ccr_rho=0.5
```
Judge each with:
```bash
python summarize_training_log.py <arm_dir> --compare "$RUN_DIR" --collapse-check \
  --reference-it-per-s 2.862 --iter 8000
```

### Step 4 — triage eval (1 seed, ~20 min)
Sanity only. A pilot predictor is ~7× worse on `z_loss`, so **a low number is not evidence
against**. A high one is strong evidence for.

### Step 5 — full CCR run (~16-18 h), gated on the pilot gate
```bash
bash run_ccr_pilot.sh full training.lambda_cf=<chosen> training.ccr_rho=0.5 \
  training.ccr_action_source=synthetic training.ccr_rollout_len=5 training.mca_weight=0
```
**Watch:** the CCR share will drift upward over the full run even at fixed λ, because the
total loss shrinks while curvature-family terms fall more slowly. Check the share partway
through, not only at the start.

### Step 6 — acceptance gate
```bash
bash run_ccr_pilot.sh eval <ccr_full_run_dir>
python ccr_acceptance_gate.py --cand-ol-seeds ... --cand-mpc-seeds ... \
                              --base-ol-seeds ... --base-mpc-seeds ...
```

---

## 8. Honest probability assessment

| bar | probability |
|---|---|
| beat same-platform baseline on open-loop (point estimate) | ~45% |
| beat it on MPC | ~28% |
| beat it on both | ~20% |
| beat the paper's 77.33 / 85.33 on both | ~13% |
| **full dual gate** (both, plus >6 pt margin) | **~10%** |

**Revised DOWN from ~15% on 2026-08-07 by two readings of the paper's own Table 1 that should
have been made at proposal time.**

**(a) The gate asks CCR for as much as straightening itself delivered.** Target cell
`DINOv2 patch + proj, 14x14x8`, PushT:

| `L_curv` | open-loop | MPC |
|---|---|---|
| ✗ | 70.00 ± 1.63 | 78.67 ± 0.94 |
| ✓ | 77.33 ± 6.18 | 85.33 ± 4.99 |
| **delta** | **+7.33** | **+6.66** |

The dual gate requires +6.0 on both. A refinement of straightening must therefore add nearly
what straightening itself added.

**(b) PushT is the cell where the conditioning mechanism appears NOT to operate.** CCR exists to
improve *gradient* conditioning, so straightening should help GD more than CEM, which uses no
gradients. From the appendix GD/CEM table, delta from ✗ to ✓:

| env | GD | CEM |
|---|---|---|
| PointMaze UMaze | **+50.00** | +18.67 |
| PointMaze Medium | **+10.67** | −6.00 |
| Wall | +10.67 | +8.00 |
| **PushT** | **+7.33** | **+8.67** |

UMaze and Medium show a large GD-specific advantage — the conditioning story working. On PushT
CEM gains *more* than GD, i.e. the effect is planner-agnostic there. Coherent with §5c: PushT's
bottleneck looks like **representation quality on block orientation** (readout R² 0.183, worst of
five, and the weakest curvature gap at every `rho`), not trajectory straightness. CCR is a pure
conditioning intervention aimed at the cell where conditioning matters least.

**What would move it back up:** PushT's GD std is 6.18, so the 1.34-point GD/CEM difference sits
well inside its own noise. If (b) is an artifact, the mechanism argument survives and the estimate
returns to ~15%. The paper's data cannot distinguish these.

**(c) PushT is the ONLY cell where the dual gate is achievable — decision confirmed by the user
2026-08-07, and it corrects a bad suggestion of mine.** I proposed retargeting to PointMaze Medium
because the GD-versus-CEM split there (+10.67 / −6.00) shows the conditioning mechanism working
where PushT's does not. That proposal was wrong, because the `L_curv` ✓ MPC rates are:

| cell | MPC ✓ | headroom to 100 | can clear a +6 pt margin? |
|---|---|---|---|
| Wall | 100.00 | 0.00 | **no** |
| PointMaze UMaze | 100.00 | 0.00 | **no** |
| PointMaze Medium | 98.67 | 1.33 | **no** |
| **PushT** | **85.33** | **14.67** | **yes** |

The Acceptance_Gate requires beating the paper on **both** open-loop and MPC by more than 6 points.
Three of four cells are saturated on MPC, so retargeting to Medium would silently have downgraded
the claim to open-loop only. PushT is **forced** by the gate structure, not merely preferred.

The two findings therefore coexist rather than conflict: PushT is the only cell where a win is
possible *and* the cell where conditioning looks least operative. ~10% is low because the only
worthwhile target is the hardest one, not because the direction is incoherent. Requirement 5.7
stands unchanged.

**Counter-evidence that is real and not dismissed:** the probe confirmed off-log trajectories are
measurably more curved (5/5 dimensions at `rho=0.5`), and the pilot shows CCR is genuinely
learnable — raw 0.339 → 0.199 while its share rose, so the encoder straightens rather than gaming
the scale-invariance. That eliminates one of three failure modes. It establishes that CCR does
something, not that the something is what PushT needs.

Belief about the **true** effect: ~20% meaningfully positive, ~45% approximately zero,
**~35% negative (CCR makes it worse)**. If the true effect is zero, P(observed beat) = 50%,
so a point-estimate "beat" is mostly noise.

**Noise floor:** binomial SE at n=50 near p=0.8 is 5.7 pts; over 3 seeds the SE of the mean
is `5.7/√3 ≈ 3.3` pts, so a credible improvement needs `> ~6.6` pts. The gate's 6-pt margin
lands on that threshold — it is the minimum delta at which the number means anything.

**How CCR could actively hurt:** (a) under `synthetic`, 3 of 5 imagined steps are rolled past
any real observation, so it may straighten trajectories the predictor has wrong; (b) the
objective is already ~80% curvature, so a second geometric term squeezes the prediction loss
that planning depends on; (c) `_cos_curvature` is scale-invariant with a `1e-6` velocity
threshold, so shrinking imagined velocities reduces CCR without straightening anything (gate
check 4 watches this); (d) `rho = 0.5` is a large perturbation in unit-variance action
space — the imagined rollouts it produces are genuinely off-distribution for the predictor,
which is the point, but also the risk.

---

## 9. Open items

1. **Compute approval — still outstanding.** Recorded plan ~23 GPU-h; revised **~37 GPU-h**.
   Of the ~14 h overrun, ~13.5 h is the baseline train the recorded plan omitted by assuming
   a checkpoint on disk (already spent, unrecoverable) and ~1.5-2 h is the fourth pilot arm.
   **If refused, drop the λ-variation arm, not the `logged` control** — the control is what
   prices the `synthetic` extrapolation risk. Task 14.1.
2. **`roll4g0.9` prior work — UNRESOLVED, and the only free information that could change the
   plan.** `/workspace/arun/temporal_straightening_old/checkpoints_rollout/test/pusht_..._roll4g0.9_ep3`
   looks like a prior rollout-based straightening attempt, and `checkpoints_ms_4scale` /
   `_ms1-4_lam0.1-0.2` look like multi-scale straightening (candidate B). If `roll4g0.9`
   already straightened **off-log** rollouts and did not win, D overlaps prior work and the
   estimate drops to ~5%. One grep settles it:
   ```bash
   grep -roh '"final_eval/success_rate": [0-9.]*' \
     /workspace/arun/temporal_straightening_old/plan_outputs_gd/pusht_*roll4g0.9*/ 2>/dev/null
   ```
   The distinction that matters: **logged** (on-distribution) vs **off-log** rollouts.
3. **Extra training seeds** under Requirement 10.5 cost a further ~26 GPU-h and need separate
   approval (Requirement 11.6).
4. **Probe `pristine` reference** never actually produced numbers. The device bug is fixed but
   the probe has not been re-run with `--reference pristine`. Low priority — it is an anchor,
   not a gate input.

---

## 10. Known limitations, already documented

- Under `logged`, the imagined horizon is **2**, not 5 (`num_frames=4`, `num_hist=3`). Only
  the `logged` arm; `synthetic` exists to reach L=5 without changing any protocol invariant.
- Under `synthetic`, 3 of 5 steps are **extrapolations** past any real observation. Mitigation:
  it is the regime `GDPlanner` operates in anyway, and the `logged` arm isolates the risk.
- Curvature window includes real context frames — 2 of 5 triples at L=5 touch a real frame.
- The `aggcos` path can only be fed **visual** channels: `agg_mlp` input width is fixed at
  `196 × emb_dim`. Matches the baseline curvature term's selection.
- `global_iter` is absent from pre-feature checkpoints, so a resumed legacy run starts the cap
  count at 0.
- `models/vit.py` hardcodes its causal mask to `cuda` as a plain attribute. Worked around in
  the probe; untouched in the training path.
- **One `predict` call costs ~8 GB of activation memory** at the target-cell shapes, because
  `models/vit.py` materialises the `(32, 16, 588, 588)` attention matrix rather than using
  `scaled_dot_product_attention`. CCR at `L = 5` therefore requires
  `training.ccr_grad_checkpoint=true` on a 45 GB slice. Switching `models/vit.py` to SDPA would
  remove the cost entirely and is the better engineering fix, but that file is outside the
  Requirement 5.6 allowlist and the change would alter the baseline's numerics, so it is
  deliberately **not** done here.

---

## 11. Operational reminders

- **One job per MIG slice.** `1g.45gb` holds exactly one. Chain on the **driver's PID**
  (`CHAIN_ON_PID`), never on the absence of its children. `setsid` exits immediately so `$!`
  is not the driver's pid.
- **`kill -0` succeeds on a zombie.** PID 1 does not reap in this container, so a finished
  detached driver lingers as `stat Z` / `<defunct>` and a naive `kill -0` chain waits on it
  forever. Always check `ps -p <pid> -o stat=`: `Z` means finished. Fixed in
  `run_ccr_pilot.sh`, but the same trap applies to any hand-rolled wait loop.
- **A chained launch that prints "No 'Model saved dir:' line after 240s" is normal** — the
  driver has not started `train.py` yet. `report_launch` now says so instead of warning.
- **`nvidia-smi` does not enumerate processes on MIG.** Use `ps`. Kill stopped (`T`/`Tl`)
  pythons — a suspended job keeps its CUDA context and its memory.
- **Never Ctrl-Z a GPU job.**
- **`kill <driver_pid>` does NOT stop a run.** `setsid` puts the driver, `train.py` and its
  ~16 dataloader workers in one process group; killing the driver orphans the python child,
  which keeps running and keeps the whole MIG slice. Use `kill -- -<driver_pid>` and then
  verify with `ps -eo pid,stat,etime,cmd | grep '[p]ython train'`. The script printed the
  wrong form until commit after `b4055a1`.
- **The scope guard flags runtime artifacts as violations.** `*.out`, `results_per_seed.csv`
  and `results_cells.csv` are generated on the pod; they are gitignored now. A scope test that
  cries wolf is worse than no scope test.
- **`train.py` resume was broken for DINOv2 runs** and nobody noticed, because every run so far
  started fresh. `init_models` calls `load_ckpt` *before* building the encoder, so `torch.load`
  hits `ModuleNotFoundError: No module named 'dinov2'` while unpickling whole `nn.Module`
  objects whose classes live in the `torch.hub` cache. Fixed by `Trainer._warm_dino_hub`,
  mirroring what `plan.load_ckpt` and `probe_ccr_curvature.py` already did. **A multi-hour
  Full_Run depends on resume working, so verify it before launching one.**
- **A relaunch into the same run directory RESUMES.** `ccr_fast_attention` and
  `ccr_grad_checkpoint` are deliberately outside `LOSS_SIGNATURE_KEYS`, so the guard allows it.
  That is correct behaviour but it appends to the existing `training_log.jsonl`, mixing step
  rates from two configurations in one file. For a clean measurement use a fresh
  `CKPT_BASE=$PWD/checkpoints_<tag>`.
- `run_ccr_pilot.sh` applies the whole Blackwell/MIG env recipe and refuses to start if the
  slice is busy.
- **Read loss shares, not loss values** (`SHORT_BUDGET_PILOTS.md` §6).
- **Mid-run representation readings are not a trajectory** (§7b) — treat them only as
  catastrophic-failure detectors.
- **3 eval seeds before believing any difference.**
- **A short smoke run's step rate is a warmup artifact.** A 50-step test read 1.890 it/s on a
  pod whose sustained rate is 2.862. Do not size a plan off it.

---

## 12. If Plan A fails

See `PLAN_B_ALTERNATIVES.md`. Four candidates with the mathematics worked through, a routing
table mapping *how* Plan A fails to which candidate that selects, and three premises each
falsifiable for under an hour of CPU. B1 (Straightened-Geometry Auxiliary Cost) is the
recommended first fallback: no retraining, ~3 GPU-h, and it exploits a different gap — the
paper's Theorem 1 bounds conditioning in the **agg** space the regularizer acted on, while
`planning/objectives.py` measures **patch** space, of which `agg`'s rank-128 map leaves
1,440 of 1,568 dimensions (91.8%) unconstrained.

**Fold in from this round:** the `block_angle` R² = 0.183 finding is itself a candidate
direction (the encoder barely represents the quantity PushT is scored on), and the `rho`
sweep is evidence for the B1 premise check.
