# PROGRESS — "Straightening trades away rotation" (ROT)

The ICLR direction selected 2026-08-09. Objective and gate: `RESEARCH_GOAL.md`.

---

## 0. Discipline

Same rule as `PROGRESS_MCA.md` §0: **every gate is written before the measurement it judges, and is
never revised after the number is seen.** A threshold fixed in advance is a test; the same threshold
revised afterwards is a fit.

This arm starts with a correction against itself, so the standard is set from the first line.

---

## 1. The claim

Temporal straightening penalises direction change in latent trajectories. **Rotation is direction
change** — a rotating object traces an arc — so the regulariser should discard orientation
information, and its benefit should collapse on tasks requiring rotational manipulation.

Supporting evidence available before this arm started:

- The paper's own open-loop straightening gains: UMaze **+50.00**, Medium **+10.67**, Wall
  **+10.67**, PushT **+7.33**. PushT is the only task in the suite with rotational state and gains
  least by ~7x.
- `PROGRESS_CCR.md` §5c: `block_angle` readout R² **0.183** in the paper's trained model against
  `block_x` 0.800, `block_y` 0.735, `agent_x` 0.728, `agent_y` 0.502.
- `PROGRESS_CCR.md` §6f: it degrades with training, 0.278 @8k → 0.183 @124k.
- The paper's `app:theory_cos` argues cosine similarity proxies driving the transition operator
  toward the **identity**, whereas Euclidean-distance-as-geodesic-proxy needs only an **isometry**.
  `A ≈ orthogonal` permits rotation; the excess strength of `A ≈ I` is what would destroy it.

## 1.1 Two corrections recorded before any GPU is spent

**(a) The causal part of the §6f claim did not survive its own robustness check.** `PROGRESS_CCR.md`
§6e re-probed at `--num-windows 192`: `block_angle` matched-control delta went **−0.077 (−28%) at
n=64 → −0.035 (−9%), "mostly noise"**. So "a second curvature penalty degrades orientation" is not
established. It was presented as established when this direction was proposed; that was a failure to
check whether a cited finding survived its own follow-up.

**(b) `block_angle` is periodic and `state_readout` is a linear ridge.** A linear map cannot recover
`t` from a `(cos t, sin t)` encoding, so a low value is ambiguous between *"the representation
discards orientation"* and *"the probe cannot read orientation"*. Until that is separated, the
0.183 supports nothing.

## 1.2 The tool, and what it measures — built and validated on CPU first

`probe_ccr_curvature.circular_state_readout` adds a wrap-aware reading of the angular dimensions:
predict `(cos t, sin t)` from the same latent on the same window split, then score
`1 − Σ(1 − cos(t − t̂)) / Σ(1 − cos(t − t̄_circ))`. `state_readout` is untouched, so every recorded
`state_readout_r2` stays reproducible.

**Neither readout dominates, and that is measured rather than assumed**
(`tests/test_circular_state_readout.py`, 15 tests, synthetic latents with known ground truth):

| synthetic world | linear R² | circular R² |
|---|---|---|
| orientation stored exactly as `(cos t, sin t)` | **~0.60** | >0.95 |
| orientation stored as the raw angle `t` | >0.9 | **~0.50** |
| orientation absent | <0.3 | <0.3 |

Each readout is blind to the other's encoding, so the decision quantity is
**`best_r2 = max(linear, circular)`** — reported as `orientation_readable`. Only *both* being low is
evidence of orientation loss.

**The calibration that matters: a perfectly `(cos, sin)`-encoded angle read linearly scores ~0.60,
which lands inside the 0.50–0.80 band the four positional dimensions occupy.** The observed 0.183 is
far below that, so correction (b) alone does **not** explain it. This weakens my own artifact
hypothesis and is recorded as such.

---

## 2. RUNG 1 GATE — WRITTEN 2026-08-09 BEFORE THE PROBE RUNS

Run `--readout curvature` on the paper's own ✓ checkpoint
(`checkpoints/test/pusht_aggmlpcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05`, `model_2.pth`,
123,858 steps) at **`--num-windows 192`** — the sampling at which §6e's robustness check was run and
at which the earlier −28% became −9%. Read `orientation_readable["block_angle"]["best_r2"]` and the
four positional dimensions' `state_readout_r2` from the same run.

| `best_r2(block_angle)` | verdict |
|---|---|
| **≥ 0.50** | **VOID.** Orientation is readable and lands in the positional band. The "worst-encoded dimension" finding was a readout artifact, the mechanism has no foundation, and **this direction is abandoned with no GPU spent.** |
| **≤ 0.30** | **SURVIVES.** Orientation is genuinely poorly encoded under both specifications, ~2x below the positional floor of 0.50. Proceed to rung 2, the matched-8k ✗/✓ causal test. |
| 0.30 – 0.50 | **MIDDLE.** Partially readable. The claim must then be restated as "under-represented relative to position", not "barely represented", and rung 2 is still required with a smaller expected effect. Recorded as middle and decided explicitly, not resolved by preference. |

**Secondary, recorded but not gating:** whether `which` is `linear` or `circular` (it tells us how
the encoding is organised), and the positional dimensions' own values at n=192 — the comparison is
relative, so if the positional band itself moves at higher sampling the bands above move with it and
that must be stated rather than ignored.

**What this gate cannot do.** It is a single trained model. Even a `SURVIVES` verdict shows only that
orientation is poorly encoded in a model trained *with* straightening; it cannot attribute that to
straightening. Attribution requires rung 2's matched control. A `VOID` verdict, by contrast, is
decisive on its own, which is the asymmetry that makes this worth doing first.

---

## 3. Rung 2, specified now so it cannot be reshaped later

**No `pusht_False_*` checkpoint exists on this pod.** Inventory 2026-08-09 found only
`aggmlpcos1e-1` PushT runs; `REPRODUCTION.md`'s ✗ arm lived under
`/workspace/arun/temporal-straightening/checkpoints/repro/`, a different tree, and its weights are
gone. Its recorded numbers survive (76.00 ± 3.27 OL / 82.00 ± 4.32 MPC).

So the causal test is **two 8,000-step runs, ~47 min each at 2.86 it/s**, against the existing
bitwise 8k control in `checkpoints_ctrl8k` (straightening on, `block_angle` = 0.278188 at n=64):

- **arm B: `straighten=False`, `encoder_lr=1e-5`** — matched lr, so straightening is the *only*
  variable. This is the scientifically clean comparison.
- **arm C: `straighten=False`, `encoder_lr=1e-6`** — the paper's own ✗ protocol (Table 3 footnote),
  for the protocol-faithful reading. `REPRODUCTION.md` pitfall 1 records that ✗ at lr 1e-5 collapses
  open-loop, which is why both are needed: B isolates the variable, C matches the paper.

Splitting the lr confound is deliberate. The paper changes two things at once between its ✗ and ✓
rows, and a single ✗ run cannot tell which one moved orientation.

~1.6 GPU-h total, against 12 h for a full ✗ run. The gate for rung 2 will be written before those
runs launch, not now — writing it before rung 1's verdict is known would mean guessing which
comparison matters.

---

## 4. Status

| item | state |
|---|---|
| Circular readout implemented + CPU-validated | **done** — 15 tests; suite 404 passed, 12 skipped, 3 pre-existing CUDA-only failures |
| Rung-1 gate pre-registered | **done — §2, before the probe** |
| Rung-1 probe run | **done — SURVIVES.** `best_r2(block_angle) = 0.166` (linear 0.166, circular **−0.021**) against positional 0.506-0.709. Artifact hypothesis dead. §5 |
| Rung-2 gate | **written before the ✗ arms exist — §6** |
| Rung-2 control **C** measured | _not run_ — free, CPU, next step |
| Rung-2 arms B / C trained | _not run_ — ~1.6 GPU-h total |
| GPU-hours spent on this arm | **0** |

---

## 5. RUNG 1 RESULT — **SURVIVES** (2026-08-09, 0 GPU-h, 117 s CPU)

`probe_outputs/rot_rung1_pusht.json`, paper's ✓ checkpoint (`model_2.pth`, 123,858 steps,
sha256 `4d68b528…`, UNCHANGED after the run), `--num-windows 192` → 64 windows per
per-dimension subset, the sampling §6e of `PROGRESS_CCR.md` used for its robustness check.

| dimension | `state_readout_r2` (linear) |
|---|---|
| agent_x | 0.709061 |
| agent_y | 0.506135 |
| block_x | 0.694847 |
| block_y | 0.685037 |
| **block_angle** | **0.166436** |
| aggregate | 0.627166 |

| angular dim | linear | circular | **best** | which |
|---|---|---|---|---|
| `block_angle` | 0.166436 | **−0.021162** | **0.166436** | linear |

**Verdict against the §2 gate, which was written before this ran: `best_r2 = 0.166 ≤ 0.30`
→ SURVIVES.** Proceed to rung 2.

### 5.1 The artifact hypothesis is dead, and it was mine

§1.1(b) raised the possibility that `block_angle` = 0.183 was an artifact of reading a periodic
variable with a linear ridge. The circular readout settles it: **−0.021**, i.e. *worse than
predicting the circular mean*. Against the calibration in `tests/test_circular_state_readout.py`:

- a `(cos t, sin t)` encoding read circularly scores **>0.95** — ruled out;
- a raw-angle encoding read circularly scores **~0.50** — ruled out;
- orientation absent scores **<0.3** on both readouts — **this is what we observe**.

And the linear 0.166 is far below the **~0.60** that an exactly `(cos, sin)`-encoded angle produces
under a linear readout, so the linear number is not merely under-reporting a well-encoded quantity.

**Both readouts low, at robustness sampling, is the evidence §1.2 said was required.** Orientation is
not represented in the aggregated latent in any linearly-recoverable form, while the four positional
dimensions read at 0.506-0.709 — a factor of 3-4 higher. The deficit is a property of the
representation, not of the probe.

### 5.2 What this does NOT establish — unchanged from §2, restated because it is the whole risk

This is **one trained model, trained with straightening**. It shows orientation is absent; it does
**not** show straightening caused that. Competing explanations that this measurement cannot separate:

1. **Straightening causes it** — the arm's hypothesis.
2. **The `agg` pooling causes it.** A 1568→128 MLP pooling patch tokens may simply discard
   orientation regardless of any regulariser.
3. **The task/data cause it.** PushT's T-block may rotate little in the logged trajectories, leaving
   little orientation variance to encode. *Cheap to check and currently unchecked.*
4. **The 8-channel projector causes it.** `dim8` may be too narrow to carry orientation alongside
   position.

Rung 2's matched control separates (1) from (2) and (4). Explanation (3) is separable offline from
the dataset alone and should be checked before any GPU is spent, because if the logged block angle
barely moves, "the representation discards orientation" is uninteresting — there was nothing to
discard.

**The `PROBE GATE: FAIL` block in the output is not this arm's gate.** It is CCR's curvature gate at
the probe's default `rho=0.05`, which `PROGRESS_CCR.md` §5a already established is 10-20x smaller
than the region the planner explores (CCR needed `rho=0.5`). Faithfully reproduced, irrelevant here.

---

## 6. RUNG 2 GATE — WRITTEN 2026-08-09, BEFORE THE ✗ ARMS ARE TRAINED

Rung 1's verdict is known, so §3's deferral is discharged. Definitions:

- **C** = `orientation_readable.block_angle.best_r2` for `checkpoints_ctrl8k` — straightening **ON**,
  `lr 1e-5`, 8,000 steps, the existing bitwise control. Measured at `--num-windows 192` **before**
  either treatment arm is trained, so it is a reference and not an outcome.
- **B** = the same quantity for **arm B**: `straighten=False`, `encoder_lr=1e-5`, 8,000 steps —
  matched lr, so straightening is the only variable.
- **B_C** = the same for **arm C**: `straighten=False`, `encoder_lr=1e-6` — the paper's own ✗
  protocol (Table 3 footnote).
- **P** = mean `state_readout_r2` over the four positional dimensions in arm B.

| condition | verdict |
|---|---|
| **B − C ≥ 0.15** AND **B ≥ 0.5·P** AND `sign(B_C − C) = sign(B − C)` | **CAUSAL CONFIRMED.** Turning straightening off recovers orientation to at least half the positional level, by a margin no plausible probe noise covers, and the paper's own ✗ protocol agrees in direction. The mechanism claim is established and the direction proceeds to the method |
| **B − C ≤ 0.05** | **REFUTED.** Straightening is not the cause; the deficit is architectural or in the data. The finding degrades to "PushT orientation is poorly encoded", which is **not** an ICLR story on its own, and this arm closes |
| 0.05 < B − C < 0.15, or the two ✗ arms disagree in sign | **MIDDLE.** Partial or lr-confounded. Recorded as such; a full-budget ✗ run (12 h) becomes the only way to resolve it, and that spend requires explicit approval |

**Why arm C exists.** The paper changes *two* things between its ✗ and ✓ rows — straightening and the
encoder lr (1e-5 → 1e-6, Table 3 footnote). A single ✗ run cannot say which moved orientation. B
isolates straightening; C matches the paper. `REPRODUCTION.md` pitfall 1 records that ✗ at `lr 1e-5`
collapses open-loop planning, which is irrelevant here: rung 2 measures a *readout*, not success.

**Cost:** 2 × ~47 min training (8,000 steps at 2.86 it/s) + 3 × ~2 min CPU probes ≈ **1.6 GPU-h**.

**Order, and it matters:** measure **C** first (free, CPU, the checkpoint already exists), then train
B and C. Measuring the control before the treatments exist is what keeps the reference honest.

**Pre-registered prediction, so it can fail.** If straightening causes the deficit, B should land in
or near the positional band (≳0.35 absolute) while C stays near 0.28. If B comes back near C, the
mechanism is refuted and I will record that as the arm's outcome rather than looking for a third
explanation.

---

## 7. Control C measured, and a design correction — 2026-08-09, 0 GPU-h

`probe_outputs/rot_rung2_ctrl8k.json`, `checkpoints_ctrl8k` (straightening **ON**, `lr 1e-5`, 8,000
steps, sha256 `2c7b7cf5…` UNCHANGED), `--num-windows 192`:

| `block_angle` | linear | circular | **best** |
|---|---|---|---|
| **ON @ 8k** (control C) | 0.394319 | 0.343252 | **0.394319** |
| **ON @ 124k** (§5) | 0.166436 | **−0.021162** | **0.166436** |

**Orientation readability falls 58% across the same training run** (0.394 → 0.166 linear), and the
circular reading collapses from moderately readable to *worse than a constant predictor*
(0.343 → −0.021). `PROGRESS_CCR.md` §6f had this as 0.278 → 0.183 on the linear readout at lower
sampling; measured with both readouts at n=192 it is far sharper. Note the two readouts **agree** at
8k (0.394 / 0.343) and **diverge** at 124k (0.166 / −0.021), which is itself informative: whatever
training removes, it removes the circularly-structured part of the orientation code first.

### 7.1 Two problems with the §6 gate, recorded not repaired

**(a) The `B ≥ 0.5·P` clause is non-binding.** At 8k the four positional dimensions read ~0.72
(`PROGRESS_CCR.md` §6e: 0.747 / 0.535 / 0.869 / 0.712), so `0.5·P ≈ 0.36` and the control's 0.394
already clears it. The operative threshold is therefore `B − C ≥ 0.15` alone, i.e. `B ≥ 0.544`.
The §6 numbers stand as written; this records what they actually bind on.

**(b) The 8k-only design is probably underpowered, and this is the task-11.4 error again.** The
0.394 → 0.166 trajectory says the mechanism is **cumulative over 124k steps**. At 8k straightening
has acted for 6.5% of the budget, so ON-vs-OFF at 8k can be small even if the mechanism is real —
exactly the mistake logged in `PROGRESS_AGG.md` §10.3, where a control was powered against the
largest cell instead of the effect actually expected. Recorded one rung after writing that lesson
into `RESEARCH_GOAL.md`.

### 7.2 The 2x2, pre-registered before either OFF arm is trained

Half of it already exists at zero cost:

| | 8,000 steps | 123,858 steps |
|---|---|---|
| straightening **ON** | **0.394** (have) | **0.166** (have) |
| straightening **OFF** | needed, ~47 min | needed, ~12 h |

**All four cells at `encoder_lr = 1e-5`, so straightening is the only variable.** The paper's ✗
protocol uses `lr 1e-6` (Table 3 footnote) and that run is a *separate* need — it is the missing ✗
baseline for the results table, since `REPRODUCTION.md`'s ✗ checkpoint lived in a tree that no longer
exists on this pod. It is not part of this causal test, and conflating the two is what §6's arm C was
guarding against.

Let `ON_8k = 0.394`, `ON_full = 0.166`, and `OFF_8k`, `OFF_full` be the measured
`orientation_readable.block_angle.best_r2` of the new arms.

| condition | verdict |
|---|---|
| **`(OFF_full − ON_full) − (OFF_8k − ON_8k) ≥ 0.15`** — the interaction, i.e. the ON/OFF gap *widens* with training | **CAUSAL CONFIRMED.** Straightening progressively destroys orientation. This is the paper's central claim and its first figure |
| **`OFF_full` also falls to ≈ `ON_full`** (within 0.08) while `OFF_8k ≈ ON_8k` | **REFUTED — prolonged training destroys orientation regardless of straightening.** Still a real and unreported finding about the architecture, but **not** about straightening, and not the ICLR story. The arm closes and the finding is written up as an analysis note |
| interaction between 0.05 and 0.15, or signs inconsistent | **MIDDLE.** Record; the decision on further spend is explicit and requires approval |

**Pre-registered prediction, so it can fail:** if straightening is the cause, `OFF_full` should stay
near or above 0.35 while `ON_full` sits at 0.166, and the 8k cells should differ far less. If
`OFF_full` lands near 0.166, the mechanism is refuted and that is the recorded outcome — I will not
go looking for a fourth explanation.

**Cost: ~13 GPU-h** (47 min + 12 h) plus two ~2 min CPU probes. It also yields the full-budget
straightening-OFF checkpoint this project has never had, which every future comparison needs.

**Still unchecked and still cheap: §5.2 explanation (3).** Whether the logged PushT T-block actually
rotates enough for orientation to be worth encoding. If its angular variance is negligible, "the
representation discards orientation" is uninteresting. That is a dataset-only measurement, no GPU, and
it should be done while the training runs.

---

## 8. DATA GATE — WRITTEN 2026-08-09, BEFORE `probe_pusht_rotation.py` RUNS

Closes §5.2 explanation (3), the last competing explanation that costs nothing to test. Dataset only:
`states.pth`, no checkpoint, no model, no GPU, no hydra, no video decode.

**Why it can void the arm.** §5 and §7 measure that the trained model barely encodes `block_angle`
and that readability falls 58% over training. **Both are uninteresting if the block does not rotate.**
An encoder that reallocates capacity away from a dimension carrying no signal is behaving correctly,
not destroying information, and the 0.394 → 0.166 trajectory would then be a *success* of training
rather than a pathology.

Read at the **25-step** horizon, the paper's short goal distance (`goal_H` in `conf/plan_gd.yaml`).
`m` = median `|Δblock_angle|` in degrees over 25 env steps; `f15` = fraction of 25-step spans with
`|Δblock_angle| > 15°`.

| condition | verdict |
|---|---|
| **`m ≥ 10°` AND `f15 ≥ 0.20`** | **SUBSTANTIAL.** The block rotates materially over a planning horizon, so orientation is information the representation could have kept and did not. §5/§7 stand and the arm proceeds |
| **`m < 3°` OR `f15 < 0.05`** | **NEGLIGIBLE — the direction is VOID.** There was nothing to discard. §5 and §7 are then facts about a dimension with no signal, the mechanism story collapses, and the arm closes with the 13 GPU-h of §7.2 unspent |
| otherwise | **MARGINAL.** Rotation present but modest. Recorded; the claim weakens to "orientation carries less signal than position *and* is encoded worse than that gap explains", which needs the variance comparison to carry it, and the decision on further spend is explicit |

The thresholds are also in `probe_pusht_rotation.verdict()` as executable code, so the log and the
script cannot drift apart.

**Reported alongside, not gating:** block translation over the same spans, so rotation is comparable
against the motion the model *does* encode; the 5- and 50-step horizons; and the fraction exceeding
5°/15°/30°/45°/90°. That last distribution is the axis `RESEARCH_GOAL.md` §2.3 needs for the
within-PushT test — per-episode required rotation against per-episode benefit, n≈150 paired episodes
instead of n=4 environments — so this run also selects the split point for that test, and selecting
it from the *data distribution* before any success rate is looked at is what keeps it honest.

**Wrap-awareness is not optional and is tested.** Every angle difference goes through
`atan2(sin d, cos d)`. A naive difference across the period boundary yields a spurious near-full
rotation, which would inflate `m` and `f15` and manufacture exactly the `SUBSTANTIAL` verdict the
arm wants. `tests/test_pusht_rotation_probe.py` pins this: 15 tests including
`test_naive_difference_would_have_been_wrong`, which records that the naive form reads 6.18 rad where
the truth is 0.10, and `test_gate_says_negligible_for_a_static_block`, which proves the VOID verdict
is reachable rather than decorative. Suite: **419 passed**, 12 skipped, 3 pre-existing CUDA-only
failures.

**Status:** gate written, probe implemented and CPU-validated, **not yet run**. The
straightening-OFF full run is training concurrently (`checkpoints_off_full`, `pusht_False_..._lr1e-05`,
~11 h remaining at 2.90 it/s); this probe is CPU-only and reads a different file, so it does not
contend for the MIG slice.

---

## 9. DATA GATE RESULT — **SUBSTANTIAL** (2026-08-09, 0 GPU-h, seconds)

`probe_outputs/rot_data_pusht.json`. Train split: 18,685 rollouts x 246 frames = 2,336,736 frames.
`seq_lengths.pkl` used (not inferred). Angle in radians, period 2π. Circular std 66.52°.

| horizon (env steps) | rot median | rot p90 | rot max | >15° | >30° | block \|Δxy\| median |
|---|---|---|---|---|---|---|
| **5** (= 1 latent step, frameskip 5) | **0.09°** | 14.80° | 101.57° | 9.8% | 1.0% | 0.136 |
| **25** (short `goal_H`) | **11.43°** | 61.65° | 179.81° | **44.3%** | 28.8% | 25.353 |
| **50** (long `goal_H`) | **32.14°** | 98.91° | 179.97° | 67.5% | 51.8% | 67.786 |

Val split (21 rollouts) agrees and runs slightly higher: 13.85° median and 49.0% over 15° at horizon
25. **Verdict against the §8 gate: median 11.43 ≥ 10 AND f15 0.443 ≥ 0.20 → SUBSTANTIAL.**
§5.2 explanation (3) is closed; the arm proceeds.

### 9.1 Two findings sharper than the gate they came from

**(a) Rotation is heavy-tailed and intermittent, and this refines the mechanism.** At **one latent
step** — the unit the curvature penalty actually operates on — the median rotation is **0.09°** while
p90 is **14.80°** and the max is 101°. Most transitions barely rotate; roughly a tenth rotate a lot.

The penalty `1 − cos(v_t, v_{t+1})` is applied **uniformly to every transition**, so it bears
disproportionately on that informative minority. The mechanism is therefore not the generic "rotation
is direction change" I first wrote, but the sharper: *a uniform curvature penalty preferentially
suppresses a heavy-tailed minority of transitions, and in PushT those are exactly the transitions
carrying orientation change.* It also explains how orientation can be destroyed without visible harm
to aggregate prediction loss — the affected transitions are ~10% of the data.

**(b) Rotation is not a low-signal dimension, so the deficit is not correct capacity allocation.**
Over 25 steps the block translates a median 25.35 units and rotates a median 11.43° — comparable
quantities of motion. Yet the trained model reads translation at **0.685-0.695** and rotation at
**0.166** (§5). The competing explanation "the encoder correctly ignores a variable that does not
move" is refuted by the data, not argued away.

### 9.2 The within-PushT split point — PRE-REGISTERED NOW, before any success rate is examined

`RESEARCH_GOAL.md` §2.3 requires the rotation axis to live *inside* PushT (n≈150 paired episodes)
rather than across n=4 environments. The split has to come from the data distribution, chosen before
any outcome is looked at, or it is fitted.

At the short horizon the distribution supports two clean splits, and both are fixed here:

- **Primary: 15°.** 44.3% of 25-step spans exceed it, so both sides have ~equal support — the
  best-powered split available.
- **Secondary: 30°.** 28.8% exceed it; a sharper contrast with weaker power, reported as the
  robustness check rather than the headline.

**Pre-registered prediction:** if straightening trades away rotation, its per-episode benefit
(straightening-ON minus straightening-OFF, paired on identical episodes) should be **smaller on
high-rotation episodes than on low-rotation ones**, split at 15°. A null or reversed ordering
contradicts the mechanism even if §5/§7/§9 all hold, and will be recorded as such.

Note this test needs the straightening-**OFF** checkpoint to be evaluated for planning success, not
just probed for readouts — so it depends on `checkpoints_off_full` finishing.

### 9.3 Status

| cell of the §7.2 2x2 | value |
|---|---|
| ON @ 8k | **0.394** |
| ON @ 124k | **0.166** |
| OFF @ 8k | measurable now — `checkpoints_off8k` exists, CPU probe, ~2 min |
| OFF @ 124k | training, ~10 h remaining (`checkpoints_off_full`, `pusht_False_..._lr1e-05`, 2.90 it/s) |

GPU-hours spent on this arm: **0** for every measurement so far. The only spend is the OFF training
pair (~13 GPU-h), of which the 8k half is already done.

---

## 10. OFF @ 8k — the matched control. `best_r2 = 0.871` against ON's 0.394

`probe_outputs/rot_rung2_off8k.json`, `checkpoints_off8k/test/pusht_False_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05`
(sha256 `c2ad85d3…` UNCHANGED), `--num-windows 192`.

| arm | linear | circular | **best** | which |
|---|---|---|---|---|
| **OFF @ 8k** (`straighten=False`, lr 1e-5) | 0.663520 | **0.870666** | **0.870666** | circular |
| ON @ 8k (`aggcos1e-1`, lr 1e-5) | 0.394319 | 0.343252 | 0.394319 | linear |
| ON @ 124k (`aggcos1e-1`, lr 1e-5) | 0.166436 | **−0.021162** | 0.166436 | linear |

**Matched on everything but the straightening term**: same env, encoder, `lr 1e-5`, batch, 8,000 steps,
CCR and MCA off in both. The ON arm is the bitwise 8k prefix of the baseline run. So the
**+0.477** difference is caused by the term, not by budget, lr or data — the same matched-control
logic `PROGRESS_CCR.md` §6e used.

### 10.1 The mechanism, read off the `which` column

Without straightening the model encodes orientation the natural way — as `(cos θ, sin θ)` — and the
circular readout recovers it at **0.871**, while the linear readout sees only 0.664, exactly the
under-reporting the calibration predicts for that encoding. Turn straightening on and the circular
reading **collapses**: 0.871 → 0.343 → **−0.021**.

**Straightening destroys the circular structure of the orientation code specifically.** That is a
sharper and more mechanistic statement than "orientation is poorly encoded", and it is visible only
because the probe reads both specifications — the linear-only view would have shown 0.664 → 0.394 →
0.166 and missed that the *periodic* component is what dies.

Combined with §9.1(a) — rotation concentrated in ~10% of latent steps — the account is now: a uniform
`1 − cos(v_t, v_{t+1})` penalty bears on the heavy-tailed minority of transitions that rotate, and
what it removes is the circular code those transitions carry.

### 10.2 My §7.2 gate asks the wrong question, and I am not rewriting it

§7.2's CONFIRMED branch tests the **interaction** `(OFF_full − ON_full) − (OFF_8k − ON_8k) ≥ 0.15`,
written on the assumption — from the 0.394 → 0.166 trajectory — that the effect would be *small at 8k
and accumulate*. It is not small at 8k. It is **0.477** at 8k.

Consequences, stated plainly:

- The **REFUTED** branch required `OFF_8k ≈ ON_8k`. That is now decisively false, so refutation
  cannot fire.
- The **interaction** may well come out below 0.15 even though the causal claim is strongly supported,
  because the main effect is nearly saturated at 8k. Reaching interaction ≥ 0.15 needs
  `OFF_full ≥ 0.793`.
- **The causal claim is carried by the matched 8k pair, not by the interaction.** At matched budget
  and lr with only straightening differing, 0.871 vs 0.394 *is* the causal test.

**This observation was made after seeing the data, so it does not get to relabel the gate.** When
`OFF_full` lands I will report the §7.2 interaction verdict as whatever it computes to, and report the
matched-8k main effect separately as the quantity that actually bears on causation. The lesson for the
*next* pre-registration: when a mechanism's time-course is unknown, gate on the **main effect at
matched budget** and treat the interaction as a secondary, descriptive question.

### 10.3 Required before this means what it looks like: is the effect specific to `block_angle`?

The console output was truncated before the `state_readout_r2` table, so the four positional
dimensions for OFF @ 8k are unread. **If OFF reads *every* dimension far better, straightening
degrades state readability in general and `block_angle` is not special** — which would still be a
finding, but a different and much weaker one than the arm's claim.

Reference points for that comparison, ON @ 8k at n=192 (`probe_outputs/rot_rung2_ctrl8k.json`) and at
n=64 (`PROGRESS_CCR.md` §6e: agent_x 0.747, agent_y 0.535, block_x 0.869, block_y 0.712,
aggregate 0.701). Both JSONs are already on disk; the comparison costs nothing.

**Pre-registered reading, written before the positional rows are looked at:** the claim is
*specificity*. If `block_angle` improves by ≥0.30 when straightening is removed while **no positional
dimension improves by more than 0.15**, the effect is orientation-specific and the arm's mechanism
stands. If the positional dimensions improve comparably, the finding degrades to "straightening
reduces linear state decodability across the board" and the rotation story is not supported.

---

## 11. SPECIFICITY TEST — **FAILED** (2026-08-09, 0 GPU-h). The mechanism claim is not supported.

Matched 8k pair, `--num-windows 192`, `state_readout_r2` per dimension:

| dimension | ON @ 8k | OFF @ 8k | delta |
|---|---|---|---|
| agent_x | 0.8257 | 0.8393 | +0.0136 |
| agent_y | 0.6696 | 0.8160 | +0.1464 |
| block_x | 0.8386 | 0.9122 | +0.0736 |
| block_y | 0.7869 | 0.9662 | **+0.1793** |
| block_angle (linear) | 0.3943 | 0.6635 | +0.2692 |
| **block_angle (best_r2)** | 0.3943 | 0.8707 | **+0.4763** |

§10.3 pre-registered: **specific iff `block_angle` improves ≥0.30 AND no positional dimension
improves >0.15.** First clause passes (+0.4763). **Second clause fails: `block_y` +0.1793.**

**Verdict: NOT SPECIFIC. Recorded as written, missing by 0.029.** A threshold that is relaxed because
the result landed just outside it is not a threshold. §0 of `PROGRESS_MCA.md` and rule 1 of
`RESEARCH_GOAL.md` both apply.

### 11.1 What the numbers do support, stated without inflation

Straightening degrades **every** state readout at matched budget. Mean positional improvement when it
is removed: **+0.1032**. `block_angle`: **+0.4763** — 4.6x the positional mean and 2.7x the worst
single positional dimension. So there is a **gradient**, not a clean dissociation.

The supportable claim is therefore: *temporal straightening reduces the linearly-decodable state
content of the aggregated representation across all measured dimensions, most severely for
orientation.* That is real, unreported, and much weaker than the arm's premise. It is an analysis note,
not the ICLR mechanism paper described in `RESEARCH_GOAL.md` §2.

### 11.2 The deeper objection, which the near-miss exposed

**Reduced decodability may be the method succeeding.** The paper's abstract argues pretrained encoders
carry information *"irrelevant -- or even detrimental -- to planning"*, and straightening lifts UMaze
open-loop 44 -> 94. If it removes state information **and** improves planning, then everything
measured in §5, §7, §10 and §11 is the mechanism **working as intended**, and "orientation is
destroyed" is a description of the method's operation rather than a defect.

Nothing measured so far distinguishes those two readings, because **no measurement in this arm has
touched planning success.** The entire arm is readout probes. A finding about decodability cannot
become a finding about a *deficiency* without showing that the lost information was needed — which
requires the paired per-episode success comparison of §9.2, on episodes split by required rotation.

That objection was available from the paper's own abstract on day one and I did not raise it. It is a
worse miss than the specificity threshold.

### 11.3 Status and what remains defensible

| item | state |
|---|---|
| §2 rung-1 gate | PASSED (`best_r2` 0.166 ≤ 0.30) |
| §8 data gate | SUBSTANTIAL (block rotates 11.43 deg median / 25 steps) |
| §10.3 specificity gate | **FAILED** — the mechanism claim is not supported |
| §7.2 interaction gate | undecided; `OFF_full` ~9 h out |
| GPU-hours spent | ~0.8 (the 8k OFF pilot); every probe was free |

**`checkpoints_off_full` is left to finish.** Not to rescue the mechanism claim, but because it is the
**straightening-OFF full-budget checkpoint this project has never had** — `REPRODUCTION.md`'s ✗ arm
lived in a deleted tree — and every future comparison, including the paper's own ✗ row, needs it. It
also completes the 2x2 and will be reported against §7.2 as written.

**What is genuinely publishable from this arm, if anything:** the 2x2 itself, as a measurement of what
straightening does to state content, together with the circular-code observation (0.871 -> 0.343 ->
-0.021, a *kind* of degradation unique to the periodic dimension since positional dimensions have no
circular structure to lose). That is a workshop-scale analysis note. It is not a novel method and it
does not beat the baseline.

---

## 12. OFF@124k landed: the §7.2 interaction gate is CAUSAL CONFIRMED (2026-08-09)

`checkpoints_off_full` finished training 19:34 UTC (epoch 2/2, 123,858 steps) and the watcher ran the
probe automatically at 19:37 UTC: `probe_outputs/rot_rung2_off_full.json`, rc=0, checkpoint sha256
`bb7718aaea867f41...` verified unchanged before and after, 192/192 windows, 4 draws.

Run identity (from the live process cmdline, verified before arming): `env=pusht encoder=dino_channel
training.straighten=False training.encoder_lr=1e-5 training.lambda_cf=0 training.ccr_rho=0
training.mca_weight=0`, run dir
`checkpoints_off_full/test/pusht_False_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05`. The probe log
confirms `Straightening disabled` and `CCR disabled (lambda_cf=0.0)` on load; the probe's own 8.1
curvature gate reports FAIL, which is the correct, expected reading for a straightening-OFF checkpoint
and independently certifies the run was clean.

### 12.1 The completed 2x2 (orientation_readable.block_angle.best_r2, --num-windows 192)

| | ON | OFF | OFF - ON |
|---|---|---|---|
| 8k | 0.394 | 0.871 | +0.477 |
| 124k (full) | 0.166 | 0.978203 | +0.812203 |

OFF@full detail: linear readout 0.755696, circular readout 0.978203, best = circular. The OFF arm
preserves a genuine (cos,sin)-style orientation code at full budget -- the strongest decodability in
the entire 2x2.

### 12.2 Verdict, judged against §7.2 exactly as written

Interaction = (OFF_full - ON_full) - (OFF_8k - ON_8k) = 0.812203 - 0.477 = **+0.335203 >= 0.15
-> CAUSAL CONFIRMED.** Matched-8k main effect, reported separately as the pre-registration requires:
**+0.477.**

The pre-data prior was MIDDLE on the grounds that the 8k main effect already looked saturated. It was
wrong in the informative direction: the OFF side did not saturate (0.871 -> 0.978) while the ON side
kept collapsing (0.394 -> 0.166). The causal effect of the straightening term on destroying the
orientation code **widens with training budget** rather than being a transient initialization artifact.

### 12.3 What this does and does not change

It does not resurrect the mechanism claim of §2 -- the specificity gate of §10.3 still stands FAILED
(block_y +0.1793 breaches the 0.15 clause), and the §11.2 objection still stands: none of this touches
planning success. What it upgrades is the measurement: a budget-growing, causally-attributed,
orientation-worst destruction of decodable state content, on a circular code, is a solid empirical
core. The next and final step for this arm is HANDOFF §10.2 -- the paired per-episode selection
experiment on this exact ON/OFF pair -- which is the only measurement that can convert "straightening
destroys orientation content" into anything about behaviour.

