# PROGRESS — Metric-Consistent Aggregation (MCA)

Live state of the MCA arm. Written so the work can be resumed cold, without the conversation.

Selected as the next arm by `PROGRESS_ACS.md` §10.2 / §12 after Stage 0 returned `STOP` on ACS rule A.

---

## 0. What this file is, and why §4 exists before any measurement

**This is a pre-registration.** §4 (the rung-1 gate) is written **before** the offline probe is run, and
§5 (the rung-2 gate) must be written **before** the pilot is launched. Nothing above §10 is edited once a
measurement exists: the thresholds are evidence, and a threshold chosen after seeing the data it judges is
a fit, not a test. That is the documented CCR failure mode (`PROGRESS_CCR.md` §5a, §6a) and the reason ACS
cost 0 GPU-h instead of 26.

**The ladder is the plan** (`SHORT_BUDGET_PILOTS.md` §1; CCR requirement 11.3 makes it mandatory):

| rung | cost | answers | state |
|---|---|---|---|
| 1 — offline probe on the existing checkpoint | 2–5 min, CPU | is the mechanism actually present? | **gate written, §4. NOT YET RUN** |
| 2 — short-budget pilot (8,000 steps) | ~0.8 GPU-h | does the change move the mechanism? | gate NOT YET WRITTEN, §5 |
| 3 — full-budget run | ~14 h | what is the reportable number? | not reachable yet |

**Never jump a rung.** And note the calibration that matters most: **CCR's rung 1 gate PASSED** at
`rho = 0.5` and CCR still lost on both settings at matched budget after ~26 GPU-h. **Rung 1 is a cheap
veto, not evidence of success.**

---

## 1. What MCA is, in one paragraph

`VWorldModel.compute_mca` — **already written, reviewed, never run.** No new module, no new parameter,
`<0.1%` overhead, and `training.mca_weight` is already a `conf/train.yaml` key (default `0.0`) that every
CCR pilot carried at zero. It penalises `encoder.agg` for distorting velocity norms:

```
feats   = visual_only(z)                             # (b, t, 196, 8)  -> 1568-d flattened
agg     = encoder.agg(feats)                         # (b, t, 128)
v_patch = ‖feats[:,1:] - feats[:,:-1]‖               # (b, t-1), 1568-d
v_agg   = ‖agg[:,1:]   - agg[:,:-1]‖                 # (b, t-1), 128-d
r       = v_agg / (v_patch + eps)
L_mca   = E[(r / E[r] − 1)²]
```

**The gap it targets.** `training.straighten=aggcos1e-1` enforces straightness in the **128-d aggregated
space**, while `planning/objectives.py::objective_fn_last` scores `nn.MSELoss` on
`z_obs_pred["visual"]` — the **1568-d patch space**, no `agg` anywhere. For straightness to transfer to
the space the planner actually measures, `agg` needs to be a *similarity* (distance-preserving up to one
global constant). `encoder.agg` is a `1568 → 512 → 512 → 128` MLP with a terminal `nn.LayerNorm(128)`;
it is neither an isometry nor injective. MCA does not ask it to be an isometry — only a similarity, which
is why the penalty compares each `r` against the batch mean `r̄` and is therefore scale-invariant.

**Why it is rotation-neutral**, and therefore why `PROGRESS_CCR.md` §6f's rotational-state objection does
not reach it: the penalty is a function of velocity *norms* only. It has no preferred direction in either
space, so it cannot suppress a particular state channel the way an unconditional curvature penalty was
shown to suppress `block_angle`.

---

## 2. Status

| item | state |
|---|---|
| Arm selected | **yes — `PROGRESS_ACS.md` §12, on the ACS Stage-0 STOP** |
| `compute_mca` implemented | **yes, pre-existing.** Reviewed, never run, `training.mca_weight` already wired |
| Rung-1 gate pre-registered (§4) | **written 2026-08-08, before any measurement** |
| Rung-1 readout `--readout aggmetric` | complete — `_mca_terms` extracted bitwise-neutral, probe calls the shipped code |
| Rung-1 probe run | **RUN 2026-08-08, 53.1 s CPU — see §6.1** |
| Rung-1 verdict | **STOP, clause `4.3-stop-positive` — see §6.2. `ρ = +0.487`, the sign my §4.2 argument said was impossible** |
| Rung-2 gate pre-registered (§5) | **not written, and will not be** |
| Rung-2 pilot (~0.8 GPU-h) | **NOT LAUNCHED — rung 2 not permitted** |
| Rung 3 | not reached |

> **REOPENED 2026-08-09 as an openly post-hoc decision — see §8.** The rung-1 STOP below stands as written
> and §4.3 is unedited; the prior is revised down to under 15% per §6.2. A full 123,858-step run at
> `mca_weight=0.1` is in progress (~12 h), to be followed by the 3-seed evaluation. The banner below records
> the state at rung 1 and is deliberately left intact.
>
> **THIS ARM IS CLOSED AT RUNG 1. `ρ(r, ‖Δpatch‖) = +0.487` — `encoder.agg` *expands* large motions
> rather than compressing them, which §4.3 pre-registered as a STOP at any magnitude. MCA is not
> piloted. Total GPU-hours spent: 0. Total CPU: 53 seconds. The return is findings M1 and M2 (§6.4).**

**GPU-hours spent on MCA: 0. Wall-clock spent: 53.1 s of CPU.**

**Pre-registered probability of clearing the operational bar: 12–18%**, recorded in `PROGRESS_ACS.md` §9
before this arm was selected — *below* ACS's 25–35% and above TMR's 8–13%. Carried here unchanged rather
than revised upward now that it is the arm in hand. The bars are `79.33` OL / `87.00` MPC against the
measured baseline of `75.33 ± 6.11` OL / `82.00 ± 2.00` MPC.

---

## 3. Hazards found while writing this file (before any measurement)

- **`_agg32_` in the run-directory template is a hardcoded literal, not an interpolation.**
  `conf/train.yaml` `hydra.run.dir` contains the fixed string `_agg32_`, while `conf/encoder/dino_channel.yaml`
  sets `agg_out_dim: 128`. So every run directory claims `agg32` regardless of the real aggregation width,
  and **two runs differing only in `agg_out_dim` would collide on one directory** — a silent cross-resume of
  the kind `ccr_tag` and the ACS plan's `acs_tag` exist to prevent. MCA does not sweep `agg_out_dim`, so this
  is not blocking; recorded because it is a live trap for anything that does.
- **The probe must read the aggregation width from the checkpoint's own `hydra.yaml`, never from the
  directory name.** For the target cell the real value is 128.

---

## 4. RUNG 1 — the pre-registered offline probe gate. WRITTEN 2026-08-08, BEFORE THE DATA

Run on the existing trained PushT checkpoint (`model_2.pth`, the control reference run). Read-only,
CPU-only, no training, no GPU. 2–5 minutes.

### 4.1 The statistic is the loss, and there is only one implementation of it

`compute_mca` returns `E[(r/r̄ − 1)²]`, which is exactly `Var(r)/E[r]² = CV(r)²`. **The rung-1 headline
statistic and the training penalty are therefore the same number.** The probe calls the shipped code; it
does not re-derive `r`. Following the ACS precedent that paid off (`_cos_curvature_terms`), `compute_mca`
is refactored into a bitwise-neutral `_mca_terms(z) -> (r, v_patch, v_agg)` plus a reduction, and the
probe calls `_mca_terms`. A test asserts the probe's `CV²` equals `compute_mca`'s return **bitwise**.
This is the structural fix for the CCR calibration error, applied here.

### 4.2 Check A — distortion magnitude. REPORTED, NEVER GATING.

`CV(r)` on the trained checkpoint, and on the `pristine` reference (freshly initialised projector /
predictor / heads, DINOv2 from the hub cache — `build_pristine_model` already exists and is free).

**Why this cannot be the gate, stated plainly rather than discovered later.** The terminal
`nn.LayerNorm(128)` pins each aggregated vector to approximately zero mean and unit variance across its
128 dimensions, so `‖agg(x)‖ ≈ √128·|γ|` and `‖Δagg‖` is bounded by roughly twice that shell radius —
while `‖Δpatch‖` in 1568-d is unbounded. **A bounded numerator over an unbounded denominator must produce
spread in `r`.** So `CV(r) > 0` is structurally guaranteed, it is not a discovery, and it cannot veto
anything. Check A exists to size the headroom and to say whether training moved `agg` toward or away from
similarity relative to the untrained reference. **Rung 1 for MCA is calibration, not a veto** — with one
exception, check B.

### 4.3 Check B — is the distortion SYSTEMATIC? THIS IS THE GATE.

The mechanism claim is not "`r` has spread". It is "the distortion systematically misdirects the penalty".
The LayerNorm shell makes a specific, falsifiable prediction: **`r` should *decrease* with `‖Δpatch‖`**,
because large patch-space motions are compressed onto a bounded shell. If instead `r` is spread but
*uncorrelated* with `‖Δpatch‖`, then `agg` is a similarity plus noise, and `L_mca` would spend gradient
shrinking noise rather than correcting a bias.

**Instrument, fixed before looking: Spearman rank correlation `ρ(r, ‖Δpatch‖)`.** Rank, not Pearson,
because both quantities are heavy-tailed and the predicted relation is monotone-but-nonlinear
(saturating). Choosing the estimator after seeing the scatter would be the §0 failure mode.

| `ρ(r, ‖Δpatch‖)` | verdict |
|---|---|
| **`ρ ≤ −0.30`** | **GO** — saturation is present and substantial; `L_mca` has a systematic bias to correct. Proceed to write §5 and launch rung 2 |
| **`−0.30 < ρ ≤ −0.10`** | **MIDDLE** — a systematic component exists but is weak. Rung 2 permitted, expected effect size small, and the writeup must not claim `agg` is badly non-metric |
| **`ρ > −0.10`** | **STOP** — no systematic saturation. The spread in `r` is noise, `L_mca` would penalise noise, and **MCA is not piloted.** Next arm becomes `aggregated-space-planning-cost` (which closes the same gap from the planner's side) |

**These thresholds are judgment calls, not derivations.** `0.30` is "an effect large enough to be worth
0.8 GPU-h"; `0.10` is "below this, *systematic* stops being an honest word, since the monotone component
explains ~1% of rank variance". Neither is a significance threshold — at `n` in the thousands the standard
error of `ρ` is ~0.01, so significance is trivially achieved and says nothing. **Do not tune these against
the measured value.**

A positive `ρ` at any magnitude is also a `STOP`, and a more interesting one: it would mean `agg`
*expands* large motions, contradicting the LayerNorm argument in §4.2 and indicating the architecture is
not understood. Record it as a finding rather than retrying with a different statistic.

### 4.4 Disaggregation is mandatory, not optional

`SHORT_BUDGET_PILOTS.md` §4 is the reason: an aggregate `probe_r2` improved 0.244 → 0.280 while `agent_x`
collapsed 0.943 → −0.011. Same data, opposite conclusions. Today's ACS Stage 0 repeated the lesson from
the other direction — UMaze's `mean cos = 0.0027` read as "no structure" until the 20-bin histogram showed
a 2-D arcsine shape. **A scalar can point the wrong way.** So rung 1 reports:

- `r` by **decile of `‖Δpatch‖`** — the direct picture of saturation, and the thing check B compresses to one number
- a **20-bin histogram of `r/r̄`**, so its shape is on the record and not just its variance
- `CV(r)` and `ρ` **per state-dimension tercile**, reusing the probe's existing `_top_tercile_mask` and
  `STATE_DIM_NAMES`, so a distortion confined to one channel cannot hide inside the aggregate
- `n_pairs` and `n_windows` beside every statistic

### 4.5 Also reported, free, and the most communicable number: rank-disagreement rate

For velocity pairs `(i, j)`, the fraction where the two spaces **order motions differently** —
`‖Δagg‖ᵢ > ‖Δagg‖ⱼ` while `‖Δpatch‖ᵢ < ‖Δpatch‖ⱼ`. One sort, no threshold, and it states the gap in
plain language: *"straightening ranks motions differently than the planner does, on X% of pairs."*
Reported, never gating.

### 4.6 What rung 1 can and cannot establish — attached here, not in a footnote

- **It cannot predict success.** CCR passed rung 1 and lost. A `GO` buys permission to spend 0.8 GPU-h.
- **`‖Δpatch‖` between consecutive frames is a proxy for what the planner measures, not the thing
  itself.** `objective_fn_last` scores MSE between a *predicted* and a *target* latent, which is
  `‖·‖²/1568` — monotone in the same norm, so the connection is real, but the planner's pairs are not
  consecutive-frame pairs. Rung 1 measures the geometry of the map, not the planner's loss surface.
- **One checkpoint, one environment, one epoch count.** The target cell is PushT at 2 epochs. Nothing here
  generalises to UMaze / Wall / Medium without re-running, and the ACS Stage 0 just demonstrated that
  cross-environment intuitions about this codebase can invert.
- **A near-isometric result would not clear `agg`.** `agg` is also **not injective** (1568 → 128), and
  MCA does not address injectivity at all. Straightness could fail to transfer through information loss
  even if every norm were preserved perfectly. MCA is one of two failure modes, and rung 1 measures only
  that one.

---

## 5. RUNG 2 — the pilot gate. _NOT YET WRITTEN. MUST PRECEDE THE LAUNCH._

To be written in full — every threshold, the control reference, the disaggregated readouts, the step-rate
bug detector, the collapse check and the matched-budget catastrophe detector — **before** any pilot is
launched, on the model of `PROGRESS_ACS.md` §5. Launching first and writing the gate afterwards is the
failure this project has already paid for twice.

Known now: the pilot is `MAX_ITERS=8000` against the bitwise-matched control
(`training.lambda_cf=0 training.ccr_rho=0 training.mca_weight=0`), and `training.mca_weight=<w>` is the
only knob that changes. `ccr_tag` already includes `mca_weight`, so the arm gets its own run directory
automatically — **zero new code**, as §12 promised.

---

## 6. Placeholders — filled in as the measurements land, not before

### 6.1 Rung-1 measured statistics — MEASURED 2026-08-08

Pod `/workspace/arun/ccr`, commit `b5a852c`, checkpoint
`pusht_aggmlpcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05/checkpoints/model_2.pth`
(sha256 `4d68b528…`, epoch 2, **UNCHANGED** after the run), `agg_out_dim = 128` read from `hydra.yaml`.
512 validation windows, **1,536 velocity pairs** (3 per window), seed 0. **53.1 s, CPU, 0 GPU-h.**
Report: `probe_outputs/mca_aggmetric_pusht.json` (pod-local, gitignored — `AGENT_MEMORY_2.0.md` §5.1).

**Check A — distortion magnitude. Reported, never gating.**

| statistic | trained | pristine |
|---|---|---|
| `CV(r)` | **0.588598** | 0.093670 |
| `compute_mca = CV(r)²` | 0.346447 | 0.008774 |
| `mean(r)` | 0.085255 | 0.242782 |
| `n_pairs` / `n_windows` | 1536 / 512 | 1536 / 512 |

`CV²` vs the shipped `compute_mca`: relative residual **1.27e-07** against a 1e-4 tolerance — the
§4.1 single-implementation guarantee **HOLDS**, verified per window. Training moved `agg`
**away** from similarity by 6.3× in `CV(r)`, and made it compress everything 2.85× more overall.

**`r` by decile of `‖Δpatch‖` — the direct saturation picture, and it runs the wrong way.**

| decile | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| `‖Δpatch‖` mean | 2.94 | 3.95 | 4.65 | 5.30 | 5.88 | 6.50 | 7.36 | 8.66 | 10.82 | 14.68 |
| mean `r` | 0.0535 | 0.0577 | 0.0672 | 0.0753 | 0.0741 | 0.0888 | 0.1057 | 0.1038 | 0.1097 | **0.1173** |

**Monotone increasing across all ten deciles** (one flat step, 4→5), total **2.19×**. §4.2 predicted a
decrease.

**Histogram of `r/r̄`**, 20 bins over the fixed `[0, 4]`, `r̄ = 0.085255`, nothing outside the range:
`30, 129, 251, 257, 257, 171, 139, 93, 53, 34, 43, 32, 13, 10, 9, 9, 1, 3, 2, 0`. Unimodal, right-skewed,
mode at `r/r̄ ∈ [0.4, 1.0]`, a tail to ~3.8× the mean.

**Per state-dimension tercile** (top third of windows by `|state_d[last] − state_d[first]|`, 513 pairs
and 171 windows each, `SE(ρ) = 0.0442`):

| dimension | `agent_x` | `agent_y` | `block_x` | `block_y` | `block_angle` |
|---|---|---|---|---|---|
| `CV(r)` | 0.5725 | 0.5721 | 0.5283 | 0.5303 | 0.5058 |
| `ρ` | +0.357 | **+0.541** | +0.365 | +0.395 | +0.380 |

**Every channel is positive**, 8–12σ each. The sign is not an artifact of one dimension, which is exactly
what §4.4's disaggregation existed to test.

**Rank-disagreement rate (§4.5, reported, never gating): 0.1856 — EXHAUSTIVE** over all 1,178,880 pairs,
351 tied. *"Straightening ranks motions differently than the planner does, on 18.6% of pairs."*

### 6.2 Rung-1 verdict — READ 2026-08-08

## RUNG-1 VERDICT: STOP, clause `4.3-stop-positive`. RUNG 2 NOT PERMITTED. MCA IS NOT PILOTED.

- **`ρ(r, ‖Δpatch‖) = +0.486982`**, `n_pairs = 1536`, `SE(ρ) = 0.0255` — **19σ from zero, and positive.**
- **Pristine reference `ρ = −0.019829`** — indistinguishable from zero. Reported, not gating.
- `ρ > 0` was pre-registered as a STOP **at any magnitude**, on the stated ground that it would mean the
  architecture is not understood. That ground is now demonstrated, not hypothetical: see §6.4 M2 and the
  error logged as §7.1.

**Honesty about the gate itself, recorded because it is a flaw in my rule and not only in my model.**
`L_mca` penalises *any* deviation of `r` from constant; it is **sign-agnostic**. So a large positive `ρ`
is just as much "a systematic bias to correct" as a large negative one would have been, and on that
reading the `ρ > 0` branch **over-fires**: it conflates two distinct questions — *is the distortion
systematic?* (which `|ρ|` answers, and the answer is emphatically yes) and *does my architectural story
hold?* (which the sign answers, and it does not). A two-sided gate would have returned GO.

**The STOP stands anyway, and is honored.** Rewriting §4.3 now, having seen `ρ`, is precisely the §0
failure mode — an arbitrary threshold fixed in advance is a test; the same threshold revised afterwards is
a fit. The lesson belongs to the *next* pre-registration, which should ask "is the distortion systematic"
two-sidedly and keep "does my mechanism story hold" as a separate, explicitly labelled sanity check
rather than fusing them into one gate. **If MCA is ever piloted it must be as an openly post-hoc decision
with the 12–18% prior revised downward, not by editing this section.**

### 6.3 Novelty positioning — _NOT YET WRITTEN, AND REQUIRED BEFORE ANY WRITEUP_

**This is the section most likely to sink the arm, and it is empty on purpose.** Penalising a map for
distorting norms is well-trodden — isometric autoencoders, isometry-regularised representation learning,
and the Iso-FM / OAT-FM line already noted in `PROGRESS_ACS.md` §8.2 (which lands on TMR's object, not
this one, but establishes that the neighbourhood is populated). **TMR's mathematical novelty was found to
be already published only after its spec was written.** MCA's defensible claim is therefore narrower than
"novel regularizer": it is the *placement* — closing a specific space mismatch between a published
method's regularizer and that same method's planner. A real prior-art search must be done **and dated**
before any claim is made, and if it lands on an existing paper, that is a `STOP` and it is cheaper to
discover now than after 14 GPU-hours.

### 6.4 Findings M1 and M2 — WRITTEN 2026-08-08

**M1 — the aggregation head this paper straightens in is strongly non-metric, and *training made it
so*.** On the paper's own trained PushT checkpoint, the velocity-norm ratio `r = ‖Δagg‖/‖Δpatch‖` has
`CV = 0.589` against **0.094** for an otherwise identical untrained head — training increased the
distortion **6.3×**. It is not noise: `ρ(r, ‖Δpatch‖) = +0.487` at `SE = 0.0255` (19σ), and mean `r`
rises **monotonically across all ten deciles** of `‖Δpatch‖`, `0.0535 → 0.1173`. So the trained `agg`
**expands large motions relative to small ones** while compressing everything 2.85× more overall than the
untrained head does. The untrained reference is metric-neutral in this respect (`ρ = −0.020`), so the
expansion is **learned, not architectural**. Consequence, stated in one line: the two spaces order motions
differently on **18.6% of pairs** (exhaustive over 1,178,880), so straightness enforced in the 128-d
aggregated space demonstrably does not transfer to the 1568-d patch space `planning/objectives.py` scores
in. Zero GPU-hours, 53 s of CPU, from a checkpoint that already existed.
**Limitations, attached here.** One checkpoint, one environment (PushT), one epoch count (2), one
`agg_out_dim` (128); `‖Δpatch‖` between consecutive frames is a monotone proxy for what the planner
measures and not the planner's own pairs; and `agg` is also **non-injective** (1568 → 128), a second
failure mode this measurement says nothing about.
**The interpretive turn, and it cuts against MCA.** A distortion that training *created* and made
monotone in motion size looks less like a defect and more like something the objective found useful —
plausibly contrast enhancement, spending aggregated dynamic range on the large motions. If so, forcing
`agg` toward a similarity would **destroy** that, and MCA was attacking the wrong side of the gap. The
better-motivated direction is to make the planner score in the space the regularizer already acts in,
which is what `aggregated-space-planning-cost` does — it renders the distortion irrelevant instead of
correcting it, and requires no claim about whether the distortion is good or bad.

**M2 — a methodological finding, and the reason to publish the negative result rather than bury it: an
asymptotic bound that never binds explains nothing.** §4.2 argued that `encoder.agg`'s terminal
`nn.LayerNorm(128)` pins outputs to a shell of radius `√128 ≈ 11.31`, capping `‖Δagg‖` at `≈ 22.63`, and
concluded that a bounded numerator over an unbounded 1568-d denominator **must** make `r` decrease with
`‖Δpatch‖`. The measurement falsified the sign. The diagnosis is exact: the typical `‖Δagg‖` is
`r̄ · ‖Δpatch‖ ≈ 0.085 × 7.07 ≈ 0.603`, which is **38× below** the ceiling the bound describes. The
constraint is real and simply never engages, so it governs nothing and the learned MLP behaviour dominates
completely. **The error was invoking a limiting constraint without checking which regime the data occupies
—** and it is the same shape of error as the one ACS Stage 0 exposed a few hours earlier, where UMaze's
`mean cos = 0.0027` "obviously" meant no structure until the 20-bin histogram showed a 2-D arcsine law.
In both cases a plausible one-line argument about a scalar was wrong, and in both cases the
**disaggregated** view settled it — the histogram there, the decile table here. That is now two
independent confirmations of `SHORT_BUDGET_PILOTS.md` §4 within one day, at a combined cost of 0 GPU-hours.

---

## 7. Errors made

| # | date | error | cost | how it was caught |
|---|---|---|---|---|
| 1 | 2026-08-08 | **§4.2's LayerNorm saturation argument was wrong, and it was the argument the whole gate was built around.** It reasoned that a terminal `nn.LayerNorm(128)` bounds `‖Δagg‖` at `≈ 22.63` while `‖Δpatch‖` is unbounded, so `r` must *decrease* with `‖Δpatch‖`. Measured `ρ = +0.487` — the opposite sign, at 19σ, monotone across all ten deciles and positive in all five state channels. The bound is real but sits **38× above** the typical `‖Δagg‖ ≈ 0.603`, so it never engages and constrains nothing | 53 s of CPU and one probe implementation; **0 GPU-h.** The pre-registered gate converted a wrong prediction into a cheap, recorded STOP instead of a 0.8 GPU-h pilot chasing a mechanism that does not exist in the claimed direction | The rung-1 probe itself, on its first run. §4.3 had pre-registered `ρ > 0` as a distinct STOP whose stated meaning was "the architecture is not understood" — written before the data, and it turned out to be the branch that fired. The disaggregated decile table (§4.4, mandatory) made the direction unmistakable rather than arguable |
| 2 | 2026-08-08 | **The `ρ > 0` branch of my own gate conflates two questions and therefore over-fires.** `L_mca` is sign-agnostic, so a large positive `ρ` is as much a systematic bias to correct as a large negative one; a two-sided gate would have returned GO. The rule fused "is the distortion systematic?" with "does my architectural story hold?" | none yet — the STOP is honored as written rather than revised, so the cost is a possibly-forgone arm, not a wrong action | Noticed while writing §6.2 up, immediately after reading the verdict. Recorded there rather than acted on: editing §4.3 after seeing `ρ` is the §0 failure mode. The fix belongs to the next pre-registration — ask the systematic question two-sidedly, keep the mechanism-story check separate and explicitly labelled |

---

## 8. ARM REOPENED 2026-08-09 — openly post-hoc, at the user's direction

**This is the post-hoc reopening §6.2 anticipated and constrained.** §4.3 is **not edited**: the rung-1 STOP
at `ρ = +0.487` stands exactly as written, and the reopening is recorded here as a separate, later,
explicitly non-pre-registered decision. Per §6.2 the 12–18% prior is revised **downward, to under 15%**.

**Grounds, such as they are.** §7.2 recorded against myself that the `ρ > 0` branch over-fires: it fused
*"is the distortion systematic?"* — which `|ρ| = 0.487` at 19σ answers emphatically yes — with *"does my
architectural story hold?"*, which the sign answers no. A two-sided gate would have returned GO. That was
recorded before this reopening was contemplated, which is the only reason it can be leaned on now.
It remains a *forgone-arm* argument, not evidence that MCA works.

**Why MCA rather than CCR or ACS**, all three checked before launching:

- **ACS is not runnable.** `total_curvature` accepts only `"aggcos"` and `"cos"`; `mode="acsaggcos"` raises
  `ValueError`. The parser, the enums and the validation landed (ACS tasks 3.3–4.4) and the loss never did,
  so `training.straighten=acsaggcos1e-1` would crash. ACS closed at 21/70 tasks.
- **CCR costs ~29 h**, not 12: its pilot ran at 1.198 it/s against the baseline's 2.86, so 123,858 steps is
  ~103,000 s. And §6e measured it degrading `block_angle` against a matched control, on the one Table-1 task
  that has rotational state.
- **MCA is `<0.1%` overhead and ~12 h**, and its STOP is the one I have on record as mis-specified.

### 8.1 Configuration, and why it is comparable to the paper

`checkpoints_mca/test/pusht_aggmlpcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05_cf0_rho0_srcsynthetic_mca0p1`

Only `training.mca_weight` differs from the paper's ✓ recipe. Confirmed from the run header:
`Straightening enabled: mode=aggcos, scale=0.1`, `MCA enabled: weight=0.1`, `CCR disabled (lambda_cf=0.0)`,
`Iteration budget: steps/epoch=61929 epochs=2 total=123858 max_iterations=0 (no cap)` — the baseline's exact
step budget — `env=pusht encoder=dino_channel training.encoder_lr=1e-5 stop_grad=True`, bf16, batch 32.
`encoder.agg_mlp.{0,2,4}` and `agg_post_norm` are in the trainable list, which is the necessary condition
for MCA's penalty to act on anything.

**`mca_weight = 0.1` is a guess, not a calibrated value.** §5's rung-2 gate was never written, so no weight
was pre-registered. 0.1 matches the project's λ convention and nothing more. This is the same exposure that
sank `aggregated-space-planning-cost`, where a plausible-looking weight put the term at 1.5–2.0% of the
objective and it could not flip one episode in fifty.

### 8.2 Smoke test on never-run code — PASS

`compute_mca` had never executed. Both bugs in the aggregated-space arm were in never-executed code, so a
200-iteration smoke ran first into `checkpoints_mca_smoke/`, ~80 s. It completed at **2.88 it/s**, matching
the baseline's 2.862–2.865 and confirming the `<0.1%` overhead claim on real hardware.

### 8.3 Early veto checks at global_iter 600 — neither fired

Rules carried over from CCR (§6a's `[2%, 30%]` share window and §7(1)'s prediction-degradation pattern)
rather than invented for this arm.

| global_iter | `mca` scaled | `mca` share | `prediction` ours | ref | delta |
|---|---|---|---|---|---|
| 200 | 0.003484 | **1.410%** | 0.157518 | 0.158464 | **−0.6%** |
| 400 | 0.003692 | 2.026% | 0.098745 | 0.098584 | +0.16% |
| 600 | 0.004198 | **3.023%** | 0.072052 | 0.072869 | **−1.1%** |

**The raw MCA term is not self-solving** — 0.003484 → 0.004198, slightly *up*, while total loss falls
0.247 → 0.139. The share rises because the denominator shrinks. That is the **opposite** of CCR's killer
signature, whose raw term fell 79% by step 8,000 ("measured cost with vanishing benefit", §7(2) there).

**Prediction loss is not degraded**: 2 of 3 rows better than the matched baseline, one tied at +0.16%. CCR's
tell was 8 of 8 rows worse from ~1,200 steps at +3% to +22%. The falling prediction *share* (−0.0091,
−0.0242, −0.0267) is MCA entering the denominator, not the predictor getting worse.

Step rate 2.855 vs 2.900 reference (+1.6% step time), floor 1.933 — PASS. Step-200 shared-term match against
the reference — PASS at rtol 0.05 on curvature, decoder and prediction.

**A specification gap in my own rule, recorded rather than resolved conveniently.** I wrote "share in
[2%, 30%]" without naming the row, and the answer depends on it: 1.410% at step 200 is *below* the floor,
3.023% at 600 is inside. CCR calibrated that window at `global_iter 8000` and `PROGRESS_ACS.md` states the
reference is read "at 8,000 and nowhere else", so **8,000 is the row that governs** and it is the read that
counts. Written down before that row exists.

### 8.4 What these checks cannot do

CCR's §6a said its 8,000-step row "can veto but cannot endorse", because a pilot predictor is ~7x worse on
`z_loss` than a finished one. At 600 steps that applies far more strongly. Nothing here predicts a success
rate. The only thing that will is the 3-seed evaluation at the end, against the paper's 77.33 / 85.33 and the
platform's 75.33 / 82.00, and the `ccr_acceptance_gate.py` predicate requires **+6.0 on both settings** for a
`pass` rather than the 79.33 / 87.00 quoted in the ACS and TMR design prose.

### 8.5 The stated mechanism is wrong, and the defensible one is narrower — found 2026-08-09, mid-run

`compute_mca`'s docstring says `agg` "only needs to be a *similarity* (distance-preserving up to one global
constant) for straightness to transfer". The first half is sound: a true similarity preserves angles, and the
paper's curvature term is a cosine, so straightness would transfer exactly. **But `CV(r) = 0` does not make
`agg` a similarity.**

`r = ‖Δagg‖ / ‖Δpatch‖` is the norm ratio along **one direction per frame pair** — the consecutive-frame
velocity. Equalising those ratios constrains the map's stretch along sampled trajectory directions only.
Angle preservation requires the Jacobian to be a constant multiple of an orthonormal frame in *every*
direction, which is what [Rate-Distortion Optimization Guided Autoencoder, arXiv 1910.04329] enforces and
what MCA does not. Two velocities can each be scaled by exactly `r̄` while the angle between them is changed
arbitrarily. **MCA is a necessary but far from sufficient condition for the property its motivation invokes.**

This is the same shape of error as §7.1 — invoking a geometric constraint without checking that it delivers
what is claimed — and as `PROGRESS_AGG.md` §7.3, where a pre-registered diagnosis was confidently wrong.
Third instance today, same failure mode: reasoning from a geometric property's *name* rather than its
*content*.

**The defensible mechanism, which survives.** Uniform `r` reduces **distance-ranking distortion**: §6.4
measured 18.6% of motion pairs ranked differently by the two spaces, and a planner whose objective is a
distance cares about ranking. That is a real and measured harm, and MCA plausibly reduces it. So the arm is
not void — but its claim must be "reduces distance-ranking distortion", never "makes straightness transfer".

**A counter-argument from the paper's own ablation, which I had not weighed.** `paper_tex/sec/2_appendix.tex`
`app:straightening` reports that the learnable aggregation head **beats** patch-wise `cos` (`λ=0.1` agg vs
`λ=0.01` patch-wise, `img:ablation_str`), with the stated reasoning that straightening should act on global
trajectory representations. If the metric distortion were simply harmful that result is hard to explain; the
likelier reading is that the distortion is the *price of a benefit* — agg wins because it is free to reshape
the space. MCA pushes it back toward patch-space fidelity and therefore risks undoing the very thing that
made agg the best variant. This strengthens §6.4 M1 with the paper's own evidence rather than a hunch.

**Consequence for reading the run.** A null is now the *expected* outcome on two independent grounds: the
mechanism is weaker than stated, and the paper's ablation suggests the distortion is load-bearing. The prior
of "under 15%" from §8 should be read at the low end of that. The run continues because the empirical question
is still unanswered and because it is this project's first completed full-budget run, not because the
mechanism story is strong.

**Novelty position, checked against the literature rather than asserted.** The loss family is not novel —
scaled-isometry and distance-preserving regularizers are established prior art ([Isometric Autoencoders,
arXiv 2006.09289], [arXiv 1910.04329], [low bending and low distortion embeddings, arXiv 2208.10193], [Neural
Isometries, arXiv 2405.19296]). The *question* is not asked in the paper, which diagrams the space split
(`img:agg`) and defends it but never examines metric distortion. The *combination* appears unpublished, but is
an obvious pairing, and [arXiv 2603.03238] reports encoder geometry regularizers making latent-dynamics
training harder, especially for long-horizon rollouts — prior art pointing against the premise. **The
transferable contribution is the diagnosis (`CV(r)` 0.094 → 0.589, `ρ = +0.487` at 19σ, 18.6% rank
disagreement), not the regularizer.**
