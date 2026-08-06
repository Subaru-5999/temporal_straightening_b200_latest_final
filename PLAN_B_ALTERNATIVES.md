# PLAN B — Alternative routes to beating the PushT baseline

Backup for `PROGRESS_CCR.md`. **Only executed if Plan A (Counterfactual Curvature
Regularization) fails its gate.** Same target cell throughout: PushT,
`DINOv2 (patch) + proj, 14×14×8, L_curv ✓`, paper values **77.33 OL / 85.33 MPC**,
same-platform baseline ~75.3 / ~82.0.

Same rules as Plan A: novel, grounded in prior work, scoped to the loss function **or** the
planning component, evaluated under the unmodified protocol (50 samples, seeds 100/200/300,
OL `mode=last α=1`, MPC `mode=staged α=1`).

---

## 0. Facts these candidates are built on

All measured on this pod, not assumed. Sources: the baseline run in
`checkpoints/test/pusht_aggmlpcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05`, the
paper LaTeX in `paper_tex/`, and `conf/encoder/dino_channel.yaml`.

| fact | value | why it matters |
|---|---|---|
| Curvature share of the objective | 31.4% @200 → 73.7% @8k → **80.5% @35.6k** | fixed λ means the intended tradeoff is never actually held |
| Prediction scaled loss | 0.1585 → 0.0132 → **0.0061** | prediction is being squeezed out over training |
| Raw on-log agg curvature @8k | 0.41421 | the scale any new term must be balanced against |
| Straightening is applied to | `encoder.agg(z)`, i.e. **R^1568 → R^128** | *not* the space the planner measures distance in |
| Planning cost is measured in | patch space, **196 × 8 = 1568 dims**, isotropic MSE | |
| Prediction term horizon | **1 step** (`num_pred=1`) | but planning rolls **5** steps |
| PushT ✗→✓ lift | 70 → 77.33 (+7.3) | smallest of the four environments (UMaze +50) |
| Step rate | 2.862 it/s → 12.0 h per full run | |

**The observation that generates candidate B1.** Theorem 1 bounds the planning Hessian's
conditioning via `ε = ‖A − I‖`, and the cosine regularizer drives `ε → 0` **in aggregated
space** (`total_curvature(..., mode="aggcos")` pools through `encoder.agg` before taking
velocities). But `planning/objectives.py` measures the terminal cost in **patch space**. So
the paper proves a conditioning guarantee for a geometry that is *not* the geometry gradient
descent optimizes over. The authors' own App B.6 ablation says the agg head wins precisely
*because* "the underlying spatial features are not forced to be overly straightened" — which
is the mismatch stated out loud, and then never carried through to the cost.

---

## 1. Routing: which candidate the Plan A failure mode selects

| how Plan A failed | what it implies | go to |
|---|---|---|
| Probe `g ≈ 1` — off-log trajectories are already straight | the coverage gap isn't the problem; geometry is fine on-log *and* off-log | **B1**, then B2 |
| Pilot gate failed on prediction share / CCR share | the objective cannot absorb another geometric term | **B2** (rebalance instead of add), then B1 |
| Pilot passed but full run scored ≈ baseline | landscape wasn't the binding constraint | **B4**, then B5 |
| Full run scored **below** baseline | added geometry actively hurts; stop adding to the loss | **B1** or **B5** (both planner-side, no retraining) |
| OL improved but MPC did not | mechanism is real but only helps open-loop | keep CCR, add **B1** on top |

---

## 2. B1 — Straightened-Geometry Auxiliary Cost (SGAC) *(recommended first)*

**Type:** planning component. **No retraining.** **Novelty: high.**

### The idea

Add a term to the planning cost that measures distance in the geometry the straightening
regularizer actually conditioned:

```
J(a) = ‖ẑ_H − z_g‖²_patch  +  γ · ‖agg(ẑ_H) − agg(z_g)‖²
```

`γ = 0` recovers the paper exactly. `γ > 0` injects gradient signal through the map whose
trajectory geometry was made straight during training.

### Mathematics, and whether it can work

Write `Φ(a) = ẑ_H` for the 5-step latent rollout, `J_Φ = ∂Φ/∂a ∈ R^{1568 × Kd_a}`, and
`P = ∂agg/∂z ∈ R^{128 × 1568}`. The two Gauss-Newton Hessians are

```
H_patch = 2 J_Φᵀ J_Φ                    H_agg = 2 J_Φᵀ Pᵀ P J_Φ
H_total = H_patch + γ H_agg
```

Theorem 1's `ε`-straightness was enforced on `A_agg = ∂agg(z_{t+1})/∂agg(z_t)`, so the bound
`κ_eff ≤ κ(B)²((1+ε)/(1−ε))^{2(K−1)}` applies to `H_agg`, **not** to `H_patch`. Adding
`γ H_agg` is therefore the only way to put a term with a *proven* conditioning bound into the
objective GD actually descends.

Does adding it improve conditioning? For PSD matrices,
`λ_min(H_total) ≥ λ_min(H_patch) + γ λ_min(H_agg)` and
`λ_max(H_total) ≤ λ_max(H_patch) + γ λ_max(H_agg)`. So `κ` improves **iff** `H_agg` puts mass
into directions where `H_patch` is nearly flat. That is exactly the plausible case: straightened
directions are the ones with well-behaved, non-vanishing sensitivity. It is **not guaranteed** —
if `H_agg`'s mass lands where `H_patch` is already strong, `λ_max` grows and `κ` worsens. This
is the honest crux, and a `γ` sweep resolves it empirically.

**Why the additive form, not a replacement.** `agg` is `Linear(1568→512) → ReLU →
Linear(512→512) → ReLU → Linear(512→128) → LayerNorm`. Its Jacobian has rank ≤ 128, so
`dim ker(P) ≥ 1568 − 128 = 1440` — **91.8% of patch space is invisible to the agg cost**, and
the trailing `LayerNorm` removes scale plus one more degree of freedom. A pure agg-space cost
(`γ = ∞`) would be badly degenerate: many distinct block poses collapse to the same target, and
GD would have no gradient to correct them. Keeping `J_patch` at full weight fixes the rank
deficiency; `γ` only adds. **The paper's setting is nested at `γ = 0`**, so with enough sweep
budget the family cannot do worse than the baseline except by noise.

### Prior work

- Temporal Straightening itself (arXiv 2603.12231, ICML 2026) — Theorem 1, and App B.6's agg-head ablation which states the mismatch.
- [TRM](https://arxiv.org/abs/2605.22164), [RC-aux](https://arxiv.org/html/2605.07278), [TD-JEPA](https://arxiv.org/abs/2607.25337v1) — all *replace* the terminal cost with a learned head. SGAC learns nothing new; it reuses a head the training already produced. That is the differentiator.
- [C-JEPA](https://arxiv.org/html/2602.11389) — plans on ~1% of latent features for efficiency. Feature *selection*, not alignment with the regularized geometry.

Searches for "planning cost in the pooled/aggregated space the regularizer acted on" returned
nothing. Novelty claim: the observation is unclaimed, and it is a gap in a published proof.

### Cost — the reason this goes first

**No retraining.** Reuses the existing checkpoint.

| step | cost |
|---|---|
| implement an `agg` objective mode in `planning/objectives.py` | ~2 h, CPU |
| triage sweep `γ ∈ {0.1, 1, 10, 100}`, 1 seed OL each | ~1.3 h GPU |
| best `γ`, 3 seeds OL + MPC | ~1.5 h GPU |
| **total** | **~3 h GPU** |

That is **1/8th** of Plan A's cost, which is why it should arguably run *before* Plan A's full
run regardless of the probe result.

### Kill criterion

If no `γ` in `{0.1, 1, 10, 100}` beats `γ = 0` by more than 2 points on single-seed OL, stop.
Two points is inside single-seed noise, so it means the term is inert.

### Scope caveat

Touching `planning/objectives.py` violates Plan A's Requirement 5.2. A Plan B spec must
re-scope deliberately — and must re-run the ✗ cell with the same `γ` so the ✗→✓ comparison
stays honest (+1.5 h).

---

## 3. B2 — Share-Targeted Curvature Control (STC)

**Type:** loss function. **Retraining required.** **Novelty: moderate.**

### The idea

Replace the fixed `λ = 0.1` with a controller that holds the curvature term at a **target share
of the objective**:

```
λ_{t+1} = λ_t · exp( η ( s_t − s* ) ),      s_t = λ_t C_t / L_total,t
```

`s*` is the target share (e.g. 0.40), `η` a small gain, `λ` clipped to a safe range. This is a
multiplicative dual ascent on the constraint `s_t = s*`.

### Mathematics, and whether it can work

Fixed `λ` does **not** fix the tradeoff. Measured: the curvature share ran 31.4% → 80.5%,
because prediction fell 26× (0.1585 → 0.0061) while raw curvature fell only ~1.4×. The
effective weighting the model trains under is therefore wildly non-stationary, and `λ = 0.1`
describes the intended balance at *no point* during training.

What the controller would actually have done, at step 35,600 with `s* = 0.40`. Raw curvature
`C = 0.029/0.1 = 0.29`, prediction `0.0061`, decoder `0.00093`:

```
s* = λC / (λC + 0.00703) = 0.40   ⟹   λC = 0.00469   ⟹   λ = 0.00469 / 0.29 ≈ 0.016
```

So it would have driven `λ` from 0.1 down to **~0.016, a 6× reduction**. That is a materially
different training trajectory, not a tweak — which is the first thing to check about any
proposed control law, and it passes.

**Why that direction might help PushT specifically.** PushT's open-loop number is the weakest ✓
cell, and open-loop executes 25 env steps from one plan, so it is limited by 5-step rollout
fidelity. Rollout fidelity is what the *prediction* term buys. A controller that stops
prediction from being squeezed to 17% of the objective directly buys the thing PushT's OL
number is short of.

**Why it might not.** (a) It may be equivalent to a well-chosen fixed lower `λ` — the claim is
that the *right* `λ` is non-stationary, and that must be tested against a tuned fixed-`λ` arm,
which doubles the cost. (b) The paper reports `λ = 0.1` as ablated-best (App B.6), so lower `λ`
may simply be worse for reasons the share view doesn't capture. (c) Dual ascent on a
non-stationary quantity can oscillate; needs EMA smoothing on `s_t` and clipping.

### Prior work

- [GECO / Taming VAEs](https://arxiv.org/abs/1810.00597) — Lagrangian constraint optimization to balance loss terms instead of tuning weights. The mechanism.
- GradNorm; Kendall et al. uncertainty weighting — adaptive multi-task loss balancing.
- Temporal Straightening App B.6 — the fixed `λ = 0.1` this replaces.

Honest novelty: the *mechanism* is established. Novel are the target (a **loss share**, not a
loss value), the application (a geometric regularizer in latent planning), and the motivating
measurement (the 31%→80% drift, which is ours and is not in the paper).

### Cost

| step | cost |
|---|---|
| implement controller + telemetry | ~3 h CPU |
| pilot, 8k steps, `s* ∈ {0.3, 0.5}` | ~2 h GPU |
| fixed-`λ` control arm at the controller's mean `λ` | ~1 h GPU |
| full run + 3-seed eval | ~13.5 h GPU |
| **total** | **~17.5 h GPU** |

### Kill criterion

If at 8k steps the controller's prediction scaled loss is not at least 1.5× the baseline's
0.0132, it isn't buying prediction capacity and the premise fails.

---

## 4. B4 — Reachability-Weighted Diagonal Metric (RWDM)

**Type:** planning component, with an offline-RL-trained weight vector. **No world-model
retraining.** **Novelty: moderate.**

### The idea

The terminal cost weights all 1568 latent dimensions equally. Learn a **diagonal** metric
`w ∈ R^{1568}_{≥0}` from the offline dataset so that latent distance predicts *time-to-goal*,
then plan with

```
J(a) = Σ_i w_i (ẑ_H,i − z_g,i)²
```

`w` is fit by offline goal-conditioned regression on **frozen** latents: sample pairs
`(z_t, z_{t+k})` from the 18,685 logged PushT trajectories, regress `k` on the weighted squared
differences, i.e. solve `min_w Σ (Σ_i w_i Δ_i² − k)²` with `w ≥ 0` — a non-negative least
squares in 1568 unknowns on cached latents. Minutes on CPU, no gradient through the encoder.

### Mathematics, and whether it can work

The key property, and the reason this is not TRM: **the cost stays a positive-definite
quadratic form in `Δ = ẑ_H − z_g`.** Its Hessian in action space is
`H = 2 J_Φᵀ diag(w) J_Φ`, still Gauss-Newton, still PSD, and Theorem 1's argument structure
survives with `W_K` replaced by `J_Φᵀ diag(w) J_Φ`. A learned pairwise head (TRM, TD-JEPA) is an
arbitrary nonlinear function of `Δ`, so it can be non-convex in `a` and destroy the gradient
structure GD depends on. TRM is built for **CEM-style ranking**, where non-convexity is free.
GD cannot afford it. That distinction is the novelty.

Conditioning effect: `diag(w)` rescales the row space of `J_Φ`. If `w` down-weights dimensions
that carry no reachability information (background patches), it removes their contribution to
`λ_max` without touching `λ_min` on the informative subspace, so `κ` falls. Quantifiable in
advance from the fit: if the top-100 weights carry >90% of the mass, the effective dimension
drops from 1568 to ~100 and the conditioning gain is real.

**Why it might not.** (a) Time-to-goal `k` is a *quasimetric* — asymmetric — while `diag(w)` is
symmetric, so the fit is misspecified by construction and its residual bounds how much it can
help. Check `R²` of the fit before spending GPU time; `R² < 0.3` means stop. (b) Logged PushT
data is noise-driven, not expert, so `k` overestimates true time-to-go. (c) A near-uniform `w`
would mean the isotropic cost was already right.

### Prior work

- [QRL](https://arxiv.org/abs/2304.01203) — optimal goal-reaching value functions have quasimetric structure. The premise, and the source of the asymmetry caveat.
- [TRM](https://arxiv.org/abs/2605.22164), [TD-JEPA](https://arxiv.org/abs/2607.25337v1) — learn a replacement cost. RWDM deliberately restricts to a diagonal form to preserve GD-friendliness.
- Mahalanobis metric learning (ITML, LMNN) — the fitting machinery, long established.

### Cost

| step | cost |
|---|---|
| cache latents for ~2k windows | ~20 min GPU |
| NNLS fit + `R²` check | minutes, CPU |
| triage 1 seed, then 3 seeds OL + MPC | ~1.8 h GPU |
| **total** | **~2.5 h GPU** |

### Kill criterion

Fit `R² < 0.3`, or the fitted `w` is within 20% of uniform. Both mean there is no
reachability signal for a diagonal metric to exploit.

---

## 5. B5 — Amortized Warm Start for GD (AWS)

**Type:** planning component. **No world-model retraining.** **Novelty: low-moderate.**

### The idea

`GDPlanner` starts from `sample_type: zero` and takes 100 Adam steps at `lr 0.1`. In a
non-convex landscape that finds whichever basin contains the origin. Train a small
goal-conditioned action-sequence proposal `π(a_{0:H} | z_0, z_g)` by behaviour cloning on the
logged trajectories (frozen encoder, latents cached), and initialise GD from `π` instead of
zero. GD still does all 100 steps, so the planner is unchanged apart from its starting point.

### Mathematics, and whether it can work

For a quadratic with Hessian `H`, GD error after `n` steps is
`‖a_n − a*‖ ≤ (1 − 1/κ)^n ‖a_0 − a*‖`. Warm starting shrinks `‖a_0 − a*‖`; it does **not**
change `κ`. So it buys a constant factor, and only matters when 100 steps are not enough to
converge — or when the landscape is non-convex and the initial basin decides the outcome. The
paper's own action-space loss-landscape figure shows PushT remains non-convex after
straightening, so basin selection is plausibly the binding issue for the weakest ✓ cell.

**Why it might not.** If 100 Adam steps already converge within the basin containing the
optimum, warm starting changes nothing. Worse, a BC policy trained on *noise-driven* PushT data
may initialise in a systematically poor basin, making things worse than the neutral zero init.
Both are cheap to detect.

### Prior work

- [Amortizing Planning in World Models](https://arxiv.org/html/2605.08732v1) — amortizes planning into a latent inverse-dynamics map under a smooth latent geometry.
- [Dream-MPC](https://arxiv.org/html/2605.04568v2) — generates GD candidates from a rolled-out policy.
- [Closing the Train-Test Gap](https://arxiv.org/abs/2512.09929) (ICLR 2026) — GD planning matching CEM after train-time changes.

**Honest novelty: this is the weakest of the four.** Warm-starting trajectory optimizers is
established practice. Include it as a cheap complement, not as a headline claim.

### Cost

~1 h to train the proposal on cached latents, ~1.8 h to evaluate. **~3 h GPU.**

---

## 6. Conditional candidate — horizon-matched prediction

`num_pred = 1` while planning rolls 5 steps, so compounding rollout error is unconstrained.
A multi-step prediction loss is the obvious fix.

**Do not pursue until the `roll4g0.9` question is answered.**
`temporal_straightening_old/checkpoints_rollout/test/pusht_..._roll4g0.9_ep3` looks exactly like
a 4-step rollout loss with discount 0.9, and `checkpoints_ms_4scale` /
`_ms1-4_lam0.1-0.2` look like multi-scale straightening. If those were run and did not win, this
whole direction is already closed. One grep settles it (`PROGRESS_CCR.md` §8).

---

## 7. Recommended order

| # | candidate | GPU-h | retrain? | novelty | run when |
|---|---|---|---|---|---|
| 1 | **B1 SGAC** | ~3 | no | high | immediately — arguably before Plan A's full run |
| 2 | **B4 RWDM** | ~2.5 | no | moderate | after B1, or in parallel on CPU |
| 3 | **B5 AWS** | ~3 | no | low-moderate | cheap complement |
| 4 | **B2 STC** | ~17.5 | yes | moderate | only if all planner-side routes fail |

**~8.5 GPU-hours buys all three planner-side candidates** — less than one full training run,
and none of them touches the world model. That is the argument for this ordering: exhaust the
no-retraining options before paying for another 12-18 h run.

A serious thought about sequencing: **B1 is cheap enough that it should probably run before Plan
A's full CCR run**, not after it. It uses the baseline checkpoint that will exist in a few
hours, costs ~3 h, and tests a gap in a published proof. Spending 16-18 h on CCR before
spending 3 h on B1 is the wrong order if the goal is a result rather than a specific result.

---

## 8. Honest assessment

| candidate | beats same-platform baseline | passes the full dual gate |
|---|---|---|
| B1 SGAC | ~35% | ~12% |
| B4 RWDM | ~25% | ~8% |
| B5 AWS | ~20% | ~5% |
| B2 STC | ~30% | ~10% |
| **any one of the four** | ~60% | **~25%** |

The last row is why a portfolio is worth having: four weakly-correlated ~10% shots give a
materially better chance than one, and three of them cost ~3 h each.

Unchanged constraints that apply to every candidate:

- **The MPC leg is the hard one.** MPC re-plans every 5 steps, so it is inherently less
  sensitive to landscape geometry, and the gate needs >6 points over 85.33 — roughly 91.3%,
  near ceiling. Every candidate here is stronger on OL than MPC. B4 is the only one with a
  mechanism that should help MPC too, since it changes *what* is measured rather than how
  smoothly it can be descended.
- **Noise floor.** SE at n=50 near p=0.8 is 5.7 points; over 3 seeds, 3.3 points. Credible
  improvement needs > ~6.6 points. Nothing under 6 points is reportable as a win.
- **Our baseline is below the paper** (~75.3 vs 77.33), so beating the printed number costs an
  extra ~2 points before the margin is even counted.

---

## 9. Infrastructure already built and reusable

Nothing here needs new tooling. From Plan A:

- `run_ccr_pilot.sh` — Blackwell/MIG env recipe, `ps` pre-flight, PID chaining, pilot/full/eval modes.
- `summarize_training_log.py` — loss shares, step rate, `--compare`, `--collapse-check`.
- `ccr_acceptance_gate.py` — the dual gate as a pure predicate.
- `training.max_iterations` — mid-epoch cap for ~1 h pilots.
- `_guard_run_dir` + `ccr_tag` — arm isolation, so no two configurations can auto-resume each other.
- `training_log.jsonl` telemetry with per-term shares.
- `tests/conftest.py` — CPU stub encoder, so property tests need no GPU or dataset.
- The baseline checkpoint and its step-8,000 reference row — the free matched-budget control.

A Plan B spec should reuse all of it. Only `planning/objectives.py` and a new config group are
genuinely new surface for B1, B4 and B5.

---

## 10. What to do before committing to any of these

1. **Answer the `roll4g0.9` question.** Closes or opens §6, and recalibrates how much of this
   neighbourhood is already explored.
2. **Verify `agg` is actually well-conditioned on real data.** B1's whole premise is that
   `H_agg` is better conditioned than `H_patch`. Measurable on the baseline checkpoint with a
   handful of JVPs, ~30 min CPU, no GPU. **Do this before writing any B1 code** — it is the
   same probe-before-pilot discipline that caught the λ error in Plan A.
3. **Check the `w` fit `R²` for B4** before any GPU time. Minutes on cached latents.

Each of those is a cheap upstream check on a premise, in the spirit of
`SHORT_BUDGET_PILOTS.md` §1. Two of the four candidates can be falsified for under an hour of
CPU, before a single GPU-hour is spent.
