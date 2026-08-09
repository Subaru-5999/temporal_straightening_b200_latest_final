# RESEARCH GOAL — what we are chasing, and the plan

Written 2026-08-09 so the objective survives context loss. This is the north-star document. The compact
always-loaded version is `.kiro/steering/research-goal.md`.

---

## 1. The objective, stated exactly

**Two things, both required:**

1. A **novel method** that beats the paper's own PushT baseline (arXiv 2603.12231, `+ Proj` with `L_curv` ✓).
2. A paper accepted at the **ICLR main conference**.

Either alone is not the goal. A +4 on one task is not an ICLR paper, and an ICLR paper that does not beat the
baseline is not what was asked for.

### 1.1 The numbers, and the gate that actually decides

| setting | our reproduction | per-seed | paper |
|---|---|---|---|
| open-loop | **75.33 ± 6.11** | 74, 82, 70 | 77.33 ± 6.18 |
| MPC | **82.00 ± 2.00** | 82, 80, 84 | 85.33 ± 4.99 |

`ccr_acceptance_gate.py::acceptance_gate` is the authority, and it is stricter than the design prose:

```
fail          unless cand_ol > 77.33 AND cand_mpc > 85.33          (the paper)
inconclusive  if min(cand_ol - 75.33, cand_mpc - 82.00) <= 6.0
PASS requires cand_ol > 81.33 AND cand_mpc > 88.00                  (+6.0 on BOTH)
```

**The "79.33 / 87.00 operational bar" in the ACS and TMR design prose is not the predicate.** It does not
appear in the code. A conclusive pass needs +6.0 on both settings.

The gate takes four means and performs **no statistical test** — `se_pts` is documented as not entering the
verdict. The ±6 "inconclusive" band is standing in for a confidence interval.

### 1.2 The measurement is currently too weak, and that is fixable

Comparing 3-seed means, a true +4.0 open-loop effect has roughly **26%** power (paired SE ≈ 3.08 by the ACS
design's own estimate). Training is bitwise deterministic and `plan.py` seeds episodes deterministically, so
the comparison is genuinely **paired** and should be analysed as such — McNemar / paired intervals over
per-episode outcomes, not a difference of means.

Open-loop evaluation costs 76 s per 50 episodes, so `n_evals=200` is ~5 min per seed and lifts power
substantially. MPC is ~25 min per 50 and cannot be scaled the same way. Report `n_evals=50` for
paper-comparability **and** the higher-n paired result for the claim. `agg_objectives.paired_counts` and
`RecordingPlanEvaluator` already capture the per-episode vectors and are not yet wired into the gate.

---

## 2. The current direction — "straightening trades away rotation"

Selected 2026-08-09 after five arms produced no win. This is the first direction motivated by a **measured
defect** rather than a plausible story.

### 2.1 The claim

Temporal straightening penalizes direction change in latent trajectories. **Rotation is direction change** — a
rotating object traces an arc — so the regularizer systematically discards orientation information, and its
benefit collapses on tasks requiring rotational manipulation.

### 2.2 The four contributions, in the order a reviewer weighs them

1. **The paper's own gain pattern is explained.** Open-loop straightening gains: UMaze **+50.00**, Medium
   **+10.67**, Wall **+10.67**, PushT **+7.33** (encoded in `probe_ccr_curvature.py` as
   `ACS_TABLE1_GAINS_DEFAULT`). PushT is the only task with rotational state and gains least by ~7x. The paper
   does not address this.
2. **The mechanism is measured, causally.** `block_angle` readout R² is **0.183** in the paper's own trained
   model against **0.50–0.80** for the four positional dimensions — the worst-encoded dimension — and it
   *degrades with training*, 0.278 @8k → 0.183 @124k (`PROGRESS_CCR.md` §6f). A matched no-straightening
   control turns this from correlation into causation.
3. **The theoretical condition is unnecessarily strong.** `paper_tex/sec/2_appendix.tex` `app:theory_cos` /
   `rem:app_dir_vs_spec` argue cosine similarity proxies driving the transition operator toward the
   **identity**. But Euclidean-distance-as-geodesic-proxy only needs an **isometry**; orthogonal operators
   preserve distance while permitting rotation. The excess of `A ≈ I` over `A ≈ orthogonal` is exactly what
   destroys orientation. This sharpens their proposition rather than contradicting it.
4. **The fix follows from the mechanism.** Penalize deviation from **constant** curvature rather than from
   **zero** curvature. Straight lines and uniform arcs both score zero; erratic turning is still penalized.
   Rotation is exempted by construction, not by a hyperparameter.

### 2.3 The weakness, and the fix for it

The gain-ordering claim rests on PushT being the only rotational task — **n = 1 on one side of the axis**.
`PROGRESS_ACS.md` L1/L2 already logged this exact failure: a cross-environment ordering with n = 4 cannot
carry a mechanism claim, and differently-typed variables across environments make it structurally unfixable
by more data.

**Move the axis inside PushT.** Every episode's goal requires some amount of T-block rotation. The prediction
becomes: *straightening's per-episode benefit decreases as the goal's required rotation increases.* That is
n = 150 paired episodes over three seeds instead of n = 4 environments, it reuses `RecordingPlanEvaluator` and
`paired_counts`, and it is nearly free. **This must be pre-registered before the correlation is computed.**

### 2.4 Prior art, checked before committing

- The isometry/orthogonality framing is **not free**: [LeJEPA world models, arXiv 2605.26379] connects linear
  orthogonal identifiability to optimal latent planning; [arXiv 2607.22430] recovers dynamics up to orthogonal
  transformation; [RLDP, arXiv 2603.15857] adds orthogonality regularization. Positioning must be explicit.
- No instance found of replacing a straightening penalty with a constant-curvature one, motivated by measured
  rotational information loss. That is the gap.
- [arXiv 2603.03238] reports encoder geometry regularizers making latent-dynamics training harder, especially
  for long-horizon rollouts — prior art pointing *against* geometry regularization generally. Must be cited and
  addressed, not ignored.

### 2.5 Critical path

| rung | work | cost |
|---|---|---|
| **1** | Does a no-straightening (`pusht_False_*`, lr 1e-06) checkpoint still exist? If yes the ✗/✓ `block_angle` comparison is free. Derive the isometry-vs-identity argument against `app:theory_cos`. Confirm per-episode required rotation is recoverable from the PushT dataset. **Pre-register the within-PushT correlation gate.** | **0 GPU** |
| **2** | Retrain ✗ if missing (~12 h). Implement constant-curvature straightening, smoke it, then 8k pilot against the bitwise matched control in `checkpoints_ctrl8k`, **gated on whether `block_angle` R² recovers** — the mechanism, not a success proxy | ~24 h |
| **3** | Full runs, 4 environments, both arms, with the paired evaluation fixed (`n_evals` raised open-loop, paired intervals reported) | 60–120 h serial |

### 2.6 Honest odds

- ✗/✓ orientation degradation confirmed: **60–70%**. The within-✓ trajectory and the gain ordering both
  already point that way.
- Method clears the gate's +6.0 on both settings: **20–30%**. Better than anything attempted so far because the
  fix targets a measured cause.
- ICLR main acceptance given a positive result: unquantified. Needs breadth on Medium and Wall, where the
  mechanism predicts *small* gains — a risk, but a falsifiable prediction, which is what reviewers want.

### 2.7 What kills it

- ✗/✓ comes back flat: straightening does not degrade orientation and the gain pattern has another cause.
- The within-PushT correlation is null: mechanism unsupported at high n, which is worse than not having tested.
- Constant-curvature straightening recovers orientation but loses the position-task gains: the tension is real
  and unavoidable rather than an artifact of an over-strong condition. Still a publishable negative with a
  clean mechanism.

### 2.8 Timeline

ICLR main deadlines land around late September. **Verify the exact date rather than trusting this line.** That
is roughly seven weeks from 2026-08-09, and rung 3 alone is one to two weeks of serial pod time, so rungs 1–2
must be decisive inside about ten days.

---

## 3. Arms closed, so they are not re-proposed

| arm | outcome | cost |
|---|---|---|
| **CCR** counterfactual curvature | STOP at rung 2 (8k pilot = 6.5% of budget). Degrades `block_angle` vs a matched control. **Full run never done** — §6a said the 8k row "can veto but cannot endorse" | ~3.7 GPU-h training |
| **TMR** temporal metric | Shelved on prior art (Iso-FM published its object) | 0 |
| **ACS** action-conditioned | STOP at Stage 0: premise inverted (`frac(cos<0)` PushT is *not* highest). **The trainable regularizer was never written** — `total_curvature` accepts only `cos`/`aggcos`, so `straighten=acsaggcos1e-1` raises | 0 |
| **MCA** metric-consistent aggregation | STOP at rung 1 (`ρ = +0.487`, 53 s CPU). Reopened post-hoc 2026-08-09, then **stopped mid-run** for this direction. Mechanism was overstated: uniform norm ratio is not a similarity | ~1 GPU-h |
| **Aggregated-space planning cost** | Closed. Control returned **+0.00** vs the paper's +9.33 MPC. Term is 1.5–2.0% of the objective; open-loop success vector bit-identical | ~2.5 GPU-h |

**Never trained to completion: any of them.** Total novel-method training across all five is under 4 GPU-h. The
12 h run was the shared baseline reproduction.

---

## 4. Rules earned the hard way — violating these has cost real time

1. **Pre-register every gate before measuring, and never revise after seeing data.** Editing a threshold once
   the number is in hand converts a test into a fit (`PROGRESS_MCA.md` §0).
2. **Disaggregate.** Scalars pointed the wrong way three times: ACS's mean cosine, MCA's `ρ` sign, and the
   aggregated arm's equal success rates hiding 2-2 discordant flips.
3. **Search prior art before proposing a method, not after.** TMR died on this.
4. **Power the control against the effect actually reported**, not the largest cell in the table. Task 11.4
   targeted the paper's one 2.1-SE cell while its mean MPC effect was +4.00, undetectable at that design's SE.
5. **Verify the code path exists before committing GPU.** ACS's regularizer was never written; the parser
   accepted the string and the loss raised.
6. **Never-run code gets a smoke test first.** Both bugs in the aggregated-space arm were in code that had
   never executed.
7. **A context manager that sets an absolute value cannot be nested by a caller that only wants to opt in.**
   `sdpa_attention(False)` inside the rollout silently defeated the outer scope; 14 passing tests covered the
   gating and none checked that the flag written is the flag read.
8. **Reason from a geometric property's content, not its name.** Three instances in one day: MCA's LayerNorm
   bound that never binds, the SDPA backend diagnosis, and "uniform norm ratio ⇒ similarity".
9. **Test the composition, not just the ends.** Passing tests on both halves of a mechanism prove nothing about
   the seam between them.
10. **One command at a time.** The pod is pull-only with no commit identity; results return by terminal paste.
    Assert the expected SHA on every pull. One job at a time on the `1g.45gb` MIG slice.
