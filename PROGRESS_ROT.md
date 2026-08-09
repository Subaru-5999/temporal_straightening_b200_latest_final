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
