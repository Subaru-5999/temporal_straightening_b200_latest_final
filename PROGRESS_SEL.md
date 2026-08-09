# PROGRESS_SEL — Selection experiment: pre-registration and log

Status: **PRE-REGISTERED 2026-08-10, approved by the user verbatim. No number in §1 may change
after any selection-eval episode has been measured.**

Authority hierarchy per HANDOFF.md: RESEARCH_GOAL.md > .kiro/steering > PROGRESS_*.md. Venue: ICRA.

## 1. Pre-registration (frozen)

**Question.** Does the straightening term's destruction of orientation content (PROGRESS_ROT §12,
interaction +0.335, CAUSAL CONFIRMED) have any behavioural consequence? The only measurement that
can answer is HANDOFF §10.2: evaluate the straightening-ON and straightening-OFF full-budget
checkpoints on *identical* episodes and split the paired per-episode outcomes by covariates
computable free from the eval's own artifacts.

**Arms (fixed).**
- ON: `checkpoints/test/pusht_aggmlpcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05/checkpoints/model_2.pth`
  — the checkpoint behind the recorded baseline 75.33 ± 6.11 OL.
- OFF: `checkpoints_off_full/test/pusht_False_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05/checkpoints/model_2.pth`
  — straighten=False, encoder_lr=1e-5, lambda_cf=0, ccr_rho=0, mca_weight=0; probed 2026-08-09
  (`probe_outputs/rot_rung2_off_full.json`, best_r2 0.978203).

**Episodes (fixed).** PushT, `goal_source=dset`, `goal_H=25`, `n_evals=50`, seeds 100 / 200 / 300,
both settings (open-loop and MPC). Identical episodes across arms follow from the frozen `plan.py`
target sampler at equal seed; pairing is verified by asserting identical `state_0`/`state_g` before
any statistic is computed (`select_failure_mode.py` refuses otherwise).

**Covariates (fixed, computed from `plan_targets.pkl` only):**
1. required rotation — wrap-aware absolute angular change of `block_angle` from `state_0` to
   `state_g` (**primary**);
2. block translation — Euclidean displacement of the block xy from `state_0` to `state_g`;
3. state-space goal distance — Euclidean distance between full agent/block state vectors at 0 and g;
4. initial agent–block distance at `state_0`.

**Splits (fixed).** Required rotation at **15°** and **30°** (the thresholds pre-registered in
PROGRESS_ROT §9.2 from the §8 data gate: 44.3% of dataset segments exceed 15°, 28.8% exceed 30°).
Every other covariate split at its **median, computed from covariates alone** (outcome-blind).

**Paired statistic.** Per side: n, ON rate, OFF rate, paired delta (points), paired SE
(McNemar/Discordant-pair variance), exact two-sided binomial p on discordant pairs.

**Qualifying gap (fixed).** A split qualifies iff its high-minus-low paired-delta gap is
**≥ 8 points AND ≥ 2× its paired SE** (SEs of the two sides combined in quadrature).

**Rotation-direction prediction (fixed).** The ROT mechanism predicts the ON-minus-OFF benefit
GROWS with required rotation: the qualifying gap must be positive (high side better).

**Stopping rule (fixed).** If no covariate split qualifies: STOP. The selection experiment found no
behavioural signature; write the analysis paper on the measurement core (PROGRESS_ROT §12) and do
not launch further selection-related evals.

**Reversal clause (fixed).** If a rotation split's gap is significantly negative (≤ −8 points AND
≤ −2× its paired SE): the ROT mechanism's behavioural prediction is CONTRADICTED and is recorded
and reported as such — it is not absorbed into the generic STOP.

**Reporting.** Whatever the outcome, report: overall paired delta per setting; every split with both
sides' statistics; the rotation verdict (CONFIRMED / CONTRADICTED / NULL); the stopping-rule
decision. No gate in this file may be revised after data is seen.

## 2. Execution machinery (already built, tested)

- Eval path: frozen `plan.py` under `plan_agg.py` at `+agg_weight=0` (bitwise-identical to plan.py,
  Task 11.1), launched via `run_ccr_pilot.sh eval` with `PLAN_ENTRY=plan_agg.py`, an explicit
  single-quoted `HYDRA_RUN_DIR` carrying `${seed}` (so no leg touches the recorded baseline
  `logs.json` cell, HANDOFF §7.5), `MODEL_EPOCH=2`, one arm/setting/seed job at a time.
- Per-run artifacts: `plan_targets.pkl` (covariates) + `agg_episode_outcomes.jsonl` (per-episode
  vectors; the reported vector is the `output_final` row).
- Analysis: `select_failure_mode.py` (24 synthetic-fixture tests, CPU-only). Gates of §1 are its
  CLI defaults (`--gap-min-pts 8 --gap-min-se 2`, rotation splits `--rotation-splits 15 30`);
  changing them on the command line after data is seen would violate §1.

## 3. Log

- 2026-08-10: pre-registration approved; first eval leg = ON arm, open-loop, seed 100, as a smoke
  leg to verify the artifact pair before the remaining battery.
- 2026-08-10: smoke leg PASSED — ON OL seed 100 = 0.74, exactly the recorded per-seed value; the
  full ON 3-seed mean reproduces 75.33 with per-seed 74/82/70, bitwise-identical to the recorded
  baseline cell. Battery: 12 legs (2 arms × 3 seeds × 2 settings) chained serially on the MIG slice.
- 2026-08-10: **OPEN-LOOP ANALYSIS** (`sel_outputs/sel_report_ol.json`, 150 paired episodes):
  overall ON 75.33 vs OFF 70.00, paired delta +5.33 ± 3.75 pts, McNemar exact p = 0.2153
  (on_only 20, off_only 12). Splits (gap = high − low, gate ≥ 8 pts AND ≥ 2×SE):
  - rot ≤ 15°: low +9.09, high +1.37, gap −7.72 ± 7.52 → not qualify, not reversed;
  - rot ≤ 30°: low +10.64 (p=0.021), high −3.57, gap −14.21 ± 8.23 → not qualify, not reversed
    (reversal would need ≤ −16.46);
  - block_trans @ 26.28: gap +13.33 ± 7.41 → short of 2SE (needs ≥ 14.82);
  - goal_dist @ 147.92: gap +8.00 ± 7.46 → short of 2SE (needs ≥ 14.92);
  - agent_block_dist @ 101.95: gap −2.67 ± 7.49 → nothing.
  **Rotation verdict: NULL. Stopping rule: TRIGGERED for the OL setting** — no qualifying split.
  Descriptive (not gate-relevant) note: every rotation point estimate trends OPPOSITE to the ROT
  prediction; the ON benefit sits on LOW-rotation episodes. MPC legs were already launched before
  this result and will complete and be judged identically; no further selection evals will be
  launched (stopping rule, §1).
- 2026-08-10: **MPC ANALYSIS** (`sel_outputs/sel_report_mpc.json`, 150 paired episodes):
  overall ON 82.00 (matches the recorded platform baseline exactly) vs OFF 75.33, paired delta
  +6.67 ± 4.68 pts, McNemar exact p = 0.2026 (on_only 30, off_only 20). Splits: rot ≤ 15° gap
  −2.31 ± 9.36; rot ≤ 30° gap +0.76 ± 9.61; block_trans +5.33 ± 9.35; goal_dist +5.33 ± 9.35;
  agent_block_dist −2.67 ± 9.36. All below gate. **Rotation verdict: NULL. Stopping rule:
  TRIGGERED for the MPC setting.**
- 2026-08-10: **SELECTION EXPERIMENT CLOSED.** Both settings judged by the frozen gates; neither
  produced a qualifying split; rotation NULL in both. Final reading: straightening confers a
  roughly UNIFORM +5.3 (OL) / +6.7 (MPC) point advantage, individually non-significant on paired
  tests, with no behavioural concentration on required rotation or any other pre-registered
  covariate. The ROT mechanism has no behavioural signature. Per §1: no further selection evals;
  the measurement core stands as recorded (PROGRESS_ROT §12 + this file). Next: the method-first
  program (Paper B), designed from these measurements, with its own pre-registered pilot gates.


