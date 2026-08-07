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

## 7. Next actions, in order

Common preamble for every command below:
```bash
cd /workspace/arun/ccr && git pull origin main
export DATASET_DIR=/workspace/arun/data          # run_ccr_pilot.sh dies without it
RUN_DIR=$PWD/checkpoints/test/pusht_aggmlpcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05
```

### Step 0 — the chained pilot launch FAILED to start; relaunch unchained

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
| beat same-platform baseline on open-loop (point estimate) | ~50% |
| beat it on MPC | ~33% |
| beat it on both | ~25% |
| beat the paper's 77.33 / 85.33 on both | ~18% |
| **full dual gate** (both, plus >6 pt margin) | **~15%** |

Revised up from ~10% by the probe: the mechanism the whole direction rests on is confirmed
present, and it took a 10× `rho` correction rather than a change of premise. Held back from
higher by §5c — the confirmed gap is weakest exactly on `block_angle`, the dimension PushT is
scored on and the one the encoder represents worst (R² 0.183).

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
