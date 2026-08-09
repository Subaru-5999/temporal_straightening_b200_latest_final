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
| Rung-1 probe run | _not run_ |
| Rung-2 gate | not written (deliberately — see §3) |
| GPU-hours spent on this arm | **0** |
