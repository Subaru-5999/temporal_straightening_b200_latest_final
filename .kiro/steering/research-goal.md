# What we are chasing (always in loop)

Full plan: `RESEARCH_GOAL.md`. This file is the compact version, auto-loaded every session.

## The objective — both halves are required

1. A **novel method** beating the paper's own PushT baseline (arXiv 2603.12231, `+ Proj` / `L_curv` ✓).
2. Acceptance at the **ICLR main conference**.

A +4 on one task is not an ICLR paper. An ICLR paper that does not beat the baseline is not the goal either.

## The gate that actually decides — read the code, not the design prose

`ccr_acceptance_gate.py::acceptance_gate`:

```
fail          unless cand_ol > 77.33 AND cand_mpc > 85.33      (the paper)
inconclusive  if min(cand_ol - 75.33, cand_mpc - 82.00) <= 6.0
PASS requires cand_ol > 81.33 AND cand_mpc > 88.00              (+6.0 on BOTH)
```

The **"79.33 / 87.00 operational bar" in the ACS and TMR design docs is NOT the predicate** and is not in the
code. Our reproduction: **75.33 ± 6.11** OL (74/82/70), **82.00 ± 2.00** MPC (82/80/84). Paper: 77.33 / 85.33.

The gate takes four means and runs **no statistical test**. Comparing 3-seed means gives ~**26%** power against
a +4 effect. The comparison is genuinely **paired** (bitwise-deterministic training, deterministic episode
seeding) so analyse it that way — McNemar / paired intervals over per-episode vectors. Open-loop eval is 76 s
per 50 episodes, so `n_evals=200` is ~5 min/seed; MPC is ~25 min per 50 and cannot scale the same way.
`agg_objectives.paired_counts` and `RecordingPlanEvaluator` already capture the vectors and are **not** wired
into the gate.

## Current direction: "straightening trades away rotation"

Straightening penalizes direction change; **rotation is direction change**, so it discards orientation, and its
benefit collapses on rotational tasks. Evidence already in hand:

- Paper's open-loop gains: UMaze **+50.00**, Medium **+10.67**, Wall **+10.67**, PushT **+7.33**. PushT is the
  only rotational task and gains least by ~7x.
- `block_angle` readout R² **0.183** vs **0.50–0.80** for positional dims, and it *degrades with training*
  (0.278 @8k → 0.183 @124k), `PROGRESS_CCR.md` §6f.
- The paper's `app:theory_cos` argues cosine proxies `A ≈ I`, but distance-as-geodesic-proxy only needs an
  **isometry**. `A ≈ orthogonal` suffices and permits rotation; the excess is what destroys orientation.
- Fix: penalize deviation from **constant** curvature, not from **zero** curvature.

**Known weakness:** the gain ordering is n=1 on the rotational side. Fix by moving the axis **inside** PushT —
per-episode required rotation vs per-episode benefit, n=150 paired episodes. **Pre-register that gate before
computing the correlation.**

**Prior art is not free:** orthogonal-dynamics-for-planning theory exists (arXiv 2605.26379, 2607.22430,
2603.15857) and arXiv 2603.03238 reports geometry regularizers *hurting* latent dynamics. Cite and address.

Path: rung 1 (0 GPU — does a `pusht_False_*` lr-1e-06 checkpoint exist for the ✗/✓ comparison?) → rung 2
(~24 h, 8k pilot gated on `block_angle` **recovery**, i.e. the mechanism, not a success proxy) → rung 3
(60–120 h, 4 envs, both arms). Odds: mechanism confirmed 60–70%, gate cleared 20–30%.

## Closed arms — do not re-propose

**CCR** stopped at an 8k pilot (degrades `block_angle` vs matched control; full run never ran). **TMR** shelved
on Iso-FM prior art. **ACS** stopped at Stage 0, premise inverted — and **its regularizer was never written**
(`total_curvature` accepts only `cos`/`aggcos`). **MCA** stopped at rung 1 (`ρ = +0.487`), briefly reopened,
then halted; its mechanism was overstated. **Aggregated-space cost** closed: control gave **+0.00** vs the
paper's +9.33.

**No arm has ever been trained to completion.** Total novel-method training is under 4 GPU-h.

## Rules earned the hard way

1. Pre-register gates before measuring; never revise after seeing data.
2. **Disaggregate** — scalars pointed the wrong way three separate times.
3. Search prior art **before** proposing a method.
4. Power a control against the effect actually reported, not the table's largest cell.
5. Verify the code path exists before committing GPU.
6. Never-run code gets a smoke test first.
7. A context manager setting an absolute value cannot be nested by a caller that only wants to opt in.
8. Reason from a property's content, not its name.
9. Test the **composition**, not just both ends of a mechanism.
10. **One command at a time.** Pod is pull-only, no commit identity, results return by paste, assert the SHA on
    every pull, one job at a time on the `1g.45gb` MIG slice.
