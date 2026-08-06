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
| Baseline train (paper's method, CCR off) | **IN PROGRESS**, ETA ~17:10 on 2026-08-06 |
| Offline probe (rung 1) | not started — blocked on baseline |
| Pilot arms (rung 2) | not started — blocked on probe gate |
| Full CCR run | not started — blocked on pilot gate |
| Acceptance gate verdict | not started |

Commits: `c86654c` implementation → `89c7df1` test fixes → `70fe2ee` telemetry `enabled`
flag → `150583a` measured gate recorded.

---

## 3. What is running right now

```bash
LOG=ccr_full_20260806_050537.log        # driver pid 63736
RUN_DIR=/workspace/arun/ccr/checkpoints/test/pusht_aggmlpcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05
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

Why it matters: it is the **Platform_Baseline** the dual gate compares against, it validates
the default-off contract at runtime, and its first 8,000 steps are the **free matched-budget
control** (`SHORT_BUDGET_PILOTS.md` §4), which is why no control pilot arm is needed.

Expected eval result: **~75-78 OL / ~82-85 MPC**. Paper prints 77.33±6.18 / 85.33±4.99; the
prior B200 reproduction of this cell was ~75.3 / ~82.0 (`AGENT_MEMORY_2.0.md` §8). Below ~72
OL or above ~82 OL are both reasons to **stop and investigate** before touching CCR.

---

## 4. Measured facts (do not re-derive)

**Step rate:** median **2.862 it/s** over 178 records, min 2.360 / max 2.880. That is +1.3%
step time vs the ~2.9 it/s in `REPRODUCTION.md`, so the documented figure is valid on this
pod. Requirement 11.7 floor = `2.862 / 1.5` = **1.91 it/s**. Full budget 123,858 steps ≈
**12.0 h**.

**Matched-budget reference, `global_iter` 8000** — the row the pilot gate is judged against:

| term | scaled | share |
|---|---|---|
| curvature | 0.041421 | 73.741% |
| prediction | 0.013196 | 23.493% |
| decoder | 0.001554 | 2.767% |
| **total** | **0.056171** | 100% |

Raw on-log aggregated curvature = `0.041421 / 0.1` = **0.41421**.

**Share drift** (why the reference is read at 8,000 and nowhere else): curvature share
31.4% @200 → 65.4% @3000 → 64.7% @4200 → **73.7% @8000** → 74.0% @10400 → 80.5% @35600.
Driven by prediction falling 0.1585 → 0.0061 while curvature fell only 0.0770 → 0.0290. The
paper's own configuration ends up ~80% curvature.

---

## 5. The gate, recorded before the probe (Requirement 8.1)

**Probe gate.** Mechanism present iff aggregate `curvature_gap` is positive **and** ≥20% of
the unperturbed curvature magnitude, on **≥3 of the 5** disaggregated dimensions
(`agent_x, agent_y, block_x, block_y, block_angle`). If it fails, **no pilot launches**.

**λ selection.** With `g` = probe's perturbed/unperturbed curvature ratio:
```
lambda_cf = 0.024 / g   ->  ~15% CCR share (target)
lambda_cf = 0.058 / g   ->  30% CCR share (hard ceiling, do not exceed)
```
| `g` | target | ceiling |
|---|---|---|
| 1.0 | 0.024 | 0.058 |
| 1.5 | 0.016 | 0.039 |
| 2.0 | 0.012 | 0.029 |
| 3.0 | 0.008 | 0.019 |

The originally recorded `{0.1, 0.3}` was **4-15× too strong** — at `g≈1` it puts CCR at
42-69% of the objective, past the 30% cap, and at 0.3 the prediction share falls to ~7.3%,
below the 11.75% floor.

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

**Acceptance gate (dual).** Pass requires beating **77.33 OL and 85.33 MPC** *and* both
re-measured Platform_Baseline rates. One condition alone = failure. Margin ≤6 pts =
inconclusive. Use `python ccr_acceptance_gate.py`.

---

## 6. Next actions, in order

### Step 1 — baseline completes (~17:10)
```bash
RUN_DIR=/workspace/arun/ccr/checkpoints/test/pusht_aggmlpcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05
grep -aE "Epoch 2|Training loss|Saved model" ccr_full_20260806_050537.log | tail -5
ls -l "$RUN_DIR/checkpoints/"          # want model_2.pth
python summarize_training_log.py "$RUN_DIR" --reference-it-per-s 2.862
```

### Step 2 — offline probe, rung 1 (~30 min, CPU, read-only, no GPU)
```bash
cd /workspace/arun/ccr && git pull origin main
export DATASET_DIR=/workspace/arun/data D4RL_SUPPRESS_IMPORT_ERROR=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
python probe_ccr_curvature.py \
  --ckpt "$RUN_DIR/checkpoints/model_2.pth" --train-cfg "$RUN_DIR/hydra.yaml" \
  --rho 0.05 --rollout-len 5 --action-source synthetic \
  --num-windows 64 --draws 4 --reference pristine \
  --max-minutes 30 --out probe_outputs/ccr_pusht.json 2>&1 | tee probe_ccr_pusht.out
```
Read: `synthesized_action_frames` must be **3**; the **per-dimension** `curvature_gap` table
(not the aggregate); `PROBE GATE : PASS/FAIL`. Compute `g` from the gap, pick λ from §5.

### Step 3 — pilot arms, rung 2 (serial, one job per MIG slice, ~60-95 min each)
```bash
# treatment (λ from the rule; 0.02 shown for g≈1)
bash run_ccr_pilot.sh pilot training.lambda_cf=0.02 training.ccr_rho=0.05 \
  training.ccr_action_source=synthetic training.ccr_rollout_len=5
PID=$(cat ccr_pilot_<ts>.pid)

# horizon control — does the gain need the steps past the window edge?
CHAIN_ON_PID=$PID bash run_ccr_pilot.sh pilot training.lambda_cf=0.02 \
  training.ccr_action_source=logged training.ccr_rollout_len=2

# perturbation control — rollout-space vs encoder-space, isolated from off-log vs on-log
CHAIN_ON_PID=<prev> bash run_ccr_pilot.sh pilot training.lambda_cf=0.02 training.ccr_rho=0

# λ variation (DROP THIS ARM if the compute overrun is refused)
CKPT_BASE=$PWD/checkpoints_cf05 CHAIN_ON_PID=<prev> \
  bash run_ccr_pilot.sh pilot training.lambda_cf=0.05 training.ccr_rho=0.05
```
Judge each with:
```bash
python summarize_training_log.py <arm_dir> --compare "$RUN_DIR" --collapse-check \
  --reference-it-per-s 2.862 --iter 8000
```

### Step 4 — triage eval (1 seed, ~20 min)
Sanity only. A pilot predictor is ~7× worse on `z_loss`, so **a low number is not evidence
against**. A high one is strong evidence for.

### Step 5 — full CCR run (~16-18 h), gated on the pilot gate passing
```bash
bash run_ccr_pilot.sh full training.lambda_cf=<chosen> training.ccr_rho=0.05 \
  training.ccr_action_source=synthetic training.ccr_rollout_len=5 training.mca_weight=0
```
**Watch:** the CCR share will drift upward over the full run even at fixed λ, because total
loss shrinks ~36% between 8k and 35.6k while curvature-family terms fall more slowly. Check
the share partway through, not only at the start.

### Step 6 — acceptance gate
```bash
bash run_ccr_pilot.sh eval "$RUN_DIR"              # re-measure Platform_Baseline
bash run_ccr_pilot.sh eval <ccr_full_run_dir>      # candidate
python ccr_acceptance_gate.py --cand-ol-seeds ... --cand-mpc-seeds ... \
                              --base-ol-seeds ... --base-mpc-seeds ...
```

---

## 7. Honest probability assessment

| bar | probability |
|---|---|
| beat same-platform baseline on open-loop (point estimate) | ~45% |
| beat it on MPC | ~30% |
| beat it on both | ~22% |
| beat the paper's 77.33 / 85.33 on both | ~15% |
| **full dual gate** (both, plus >6 pt margin) | **~10%** |

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
check 4 watches this).

**How the probe moves it:** `g ≈ 1` → **~3%**, premise falsified, stop and switch to
candidate E. `g ≈ 1.5-2` → ~12%. `g > 2.5` and strong on `block_*` → **~25%**, because the
on-log term has been driven to 80% of the objective while off-log stays badly curved, which is
exactly the unproven coverage gap.

---

## 8. Open items

1. **Compute approval.** Recorded plan ~23 GPU-h; revised **~37 GPU-h**. Of the ~14 h
   overrun, ~13.5 h is the baseline train the recorded plan omitted by assuming a checkpoint
   on disk (already spent, unrecoverable) and ~1.5-2 h is the fourth pilot arm. **If refused,
   drop the λ-variation arm, not the `logged` control** — the control is what prices the
   `synthetic` extrapolation risk. Task 14.1.
2. **`roll4g0.9` prior work — UNRESOLVED, and the only free information that could change the
   plan.** `/workspace/arun/temporal_straightening_old/checkpoints_rollout/test/pusht_..._roll4g0.9_ep3`
   looks like a prior rollout-based straightening attempt, and `checkpoints_ms_4scale` /
   `_ms1-4_lam0.1-0.2` look like multi-scale straightening (candidate B). If `roll4g0.9`
   already straightened **off-log** rollouts and did not win, D overlaps prior work and the
   estimate drops to ~5%. Check:
   ```bash
   grep -roh '"final_eval/success_rate": [0-9.]*' \
     /workspace/arun/temporal_straightening_old/plan_outputs_gd/pusht_*roll4g0.9*/ 2>/dev/null
   ```
   The distinction that matters: **logged** (on-distribution) vs **off-log** rollouts.
3. **Extra training seeds** under Requirement 10.5 cost a further ~26 GPU-h and need separate
   approval (Requirement 11.6).

---

## 9. Known limitations, already documented

- Under `logged`, the imagined horizon is **2**, not 5 (`num_frames=4`, `num_hist=3`). Only
  the `logged` arm; `synthetic` exists to reach L=5 without changing any protocol invariant.
- Under `synthetic`, 3 of 5 steps are **extrapolations** past any real observation. Mitigation:
  it is the regime `GDPlanner` operates in anyway, and the `logged` arm isolates the risk.
- Curvature window includes real context frames — 2 of 5 triples at L=5 touch a real frame.
- The `aggcos` path can only be fed **visual** channels: `agg_mlp` input width is fixed at
  `196 × emb_dim`. Matches the baseline curvature term's selection.
- `global_iter` is absent from pre-feature checkpoints, so a resumed legacy run starts the cap
  count at 0.

---

## 10. Operational reminders

- **One job per MIG slice.** `1g.45gb` holds exactly one. Chain on the **driver's PID**
  (`CHAIN_ON_PID`), never on the absence of its children. `setsid` exits immediately so `$!`
  is not the driver's pid.
- **`nvidia-smi` does not enumerate processes on MIG.** Use `ps`. Kill stopped (`T`/`Tl`)
  pythons — a suspended job keeps its CUDA context and its memory.
- **Never Ctrl-Z a GPU job.**
- `run_ccr_pilot.sh` applies the whole Blackwell/MIG env recipe and refuses to start if the
  slice is busy.
- **Read loss shares, not loss values** (`SHORT_BUDGET_PILOTS.md` §6).
- **Mid-run representation readings are not a trajectory** (§7b) — treat them only as
  catastrophic-failure detectors.
- **3 eval seeds before believing any difference.**
