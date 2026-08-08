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
| Rung-1 readout `--readout aggmetric` | _not started_ |
| Rung-1 probe run | **_NOT RUN — see §10.1_** |
| Rung-1 verdict | **_NOT READ — see §10.2_** |
| Rung-2 gate pre-registered (§5) | **_NOT WRITTEN — must precede the pilot launch_** |
| Rung-2 pilot (~0.8 GPU-h) | _not launched_ |
| Rung 3 | _not reachable_ |

**GPU-hours spent on MCA so far: 0.**

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

### 6.1 Rung-1 measured statistics — _NOT YET MEASURED_

Per §4.4: `CV(r)` trained and pristine; `ρ(r, ‖Δpatch‖)`; `r` by `‖Δpatch‖` decile; 20-bin histogram of
`r/r̄`; per-state-dimension-tercile breakdown; rank-disagreement rate; `n_pairs` / `n_windows` throughout.

### 6.2 Rung-1 verdict — _NOT YET READ_

Check B's `ρ` and the firing row of §4.3, written **before** rung 2 is launched.

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

### 6.4 Findings — _NOT YET WRITTEN_

Rung 1 produces a reportable statement whether or not MCA is piloted: **how far the aggregation head this
paper straightens in is from being a similarity, and whether that distortion is systematic.** Zero
GPU-hours, from the paper's own trained checkpoint.

---

## 7. Errors made

| # | date | error | cost | how it was caught |
|---|---|---|---|---|
| — | — | _none yet; this file was created before any MCA measurement_ | — | — |
