#!/usr/bin/env python3
"""
ccr_acceptance_gate.py  --  the CCR Acceptance_Gate as a pure predicate, plus a thin CLI.

The gate (Requirement 10) is dual and margin-aware:

  * the candidate must beat the Paper_Target on BOTH open-loop and MPC (10.1),
  * AND it must beat the re-measured Platform_Baseline on BOTH open-loop and MPC (10.2),
  * one condition holding alone is a gate FAILURE, not a partial pass (10.6),
  * a margin over the Platform_Baseline of <= 6 percentage points is INCONCLUSIVE (10.5),
  * the ~5.7 percentage-point binomial standard error at n=50 near p=0.8 is reported
    alongside every comparison (10.4).

`acceptance_gate` is a pure function with no import-time side effects, so the property
test for Property 15 can call it directly with generated values.

Standard library only (no numpy, no torch): this runs on the B200 pod and on a dev box.

Usage:
    # single aggregate numbers
    python ccr_acceptance_gate.py --cand-ol 79.3 --cand-mpc 88.7 --base-ol 75.3 --base-mpc 82.0

    # per-seed numbers (mean over seeds is used as the point estimate, std is reported)
    python ccr_acceptance_gate.py \
        --cand-ol-seeds 78.0,80.0,80.0 --cand-mpc-seeds 88.0,90.0,88.0 \
        --base-ol-seeds 74.0,76.0,76.0 --base-mpc-seeds 82.0,82.0,82.0
"""

import argparse
import math
import statistics
import sys

# Paper_Target: Table 1 of arXiv 2603.12231v2, PushT row
# `DINOv2 (patch) + proj, 14x14x8, L_curv checkmark`
# (run dir pusht_aggmlpcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05):
# 77.33% goal-reaching success open-loop, 85.33% under MPC, GD planner,
# mean over 3 data-sampling seeds.
PAPER_OL = 77.33
PAPER_MPC = 85.33

# Requirement 10.4: binomial standard error at n = 50 test samples near p = 0.8,
# 100 * sqrt(0.8 * 0.2 / 50) = 5.657 -> reported as ~5.7 percentage points.
SE_PTS = 5.7

# Requirement 10.5: a margin over Platform_Baseline at or below this is inconclusive.
MARGIN_PTS = 6.0

# Evaluation_Protocol shape, used only to label the reported standard error.
EVAL_SAMPLES_PER_SEED = 50
EVAL_P_NEAR = 0.8

VERDICTS = ("pass", "inconclusive", "fail")


def acceptance_gate(cand_ol, cand_mpc, base_ol, base_mpc,
                    paper_ol=PAPER_OL, paper_mpc=PAPER_MPC,
                    se_pts=SE_PTS, margin_pts=MARGIN_PTS):
    """Return "pass", "inconclusive" or "fail" for one candidate/baseline pair.

    Pure: no I/O, no globals mutated. `se_pts` is carried for reporting symmetry with
    the design signature and does not enter the verdict.
    """
    beats_paper = cand_ol > paper_ol and cand_mpc > paper_mpc
    beats_platform = cand_ol > base_ol and cand_mpc > base_mpc
    margin = min(cand_ol - base_ol, cand_mpc - base_mpc)
    if not (beats_paper and beats_platform):
        return "fail"                                             # 10.6
    return "inconclusive" if margin <= margin_pts else "pass"     # 10.5


def platform_margin(cand_ol, cand_mpc, base_ol, base_mpc):
    """The gate's margin: the weaker of the two per-setting margins over the baseline."""
    return min(cand_ol - base_ol, cand_mpc - base_mpc)


def binomial_se_pts(p=EVAL_P_NEAR, n=EVAL_SAMPLES_PER_SEED):
    """Binomial standard error in percentage points. At p=0.8, n=50 this is ~5.66."""
    return 100.0 * math.sqrt(p * (1.0 - p) / n)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def _parse_seed_list(text):
    """"78, 80,80" -> [78.0, 80.0, 80.0]. Raises argparse.ArgumentTypeError on junk."""
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("expected a comma-separated list of success rates")
    try:
        return [float(p) for p in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not a number in {text!r}: {exc}")


def _resolve(name, scalar, seeds):
    """Fold a scalar or a per-seed list into (value, std_or_None, seeds_or_None)."""
    if seeds:
        mean = statistics.fmean(seeds)
        std = statistics.stdev(seeds) if len(seeds) > 1 else 0.0
        if scalar is not None and abs(scalar - mean) > 1e-9:
            print(f"note: --{name} ({scalar:g}) ignored; using the mean of "
                  f"--{name}-seeds ({mean:.2f})")
        return mean, std, seeds
    if scalar is None:
        return None, None, None
    return scalar, None, None


def _fmt(value, std, seeds):
    out = f"{value:6.2f}"
    if std is not None:
        out += f" +/- {std:.2f} (n_seeds={len(seeds)}: " + ", ".join(f"{s:g}" for s in seeds) + ")"
    return out


def _explain(label, cand, ref, ref_label):
    ok = cand > ref
    sign = ">" if ok else ("==" if cand == ref else "<")
    return (f"    {label:<10} {cand:6.2f} {sign} {ref_label} {ref:6.2f}   "
            f"delta {cand - ref:+6.2f} pts   [{'PASS' if ok else 'FAIL'}]")


def build_parser():
    p = argparse.ArgumentParser(
        description="Evaluate the CCR Acceptance_Gate (Requirement 10) on PushT success rates.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("Both candidate and baseline must be measured under the unmodified "
                "Evaluation_Protocol: 50 test samples per data seed, seeds 100/200/300, "
                "open-loop mode=last alpha=1, MPC mode=staged alpha=1. The recorded B200 "
                "reproduction of this cell was ~75.3 open-loop / ~82.0 MPC, but the baseline "
                "is re-measured rather than taken from that number."))
    p.add_argument("--cand-ol", type=float, help="candidate open-loop success rate (%%)")
    p.add_argument("--cand-mpc", type=float, help="candidate MPC success rate (%%)")
    p.add_argument("--base-ol", type=float, help="Platform_Baseline open-loop success rate (%%)")
    p.add_argument("--base-mpc", type=float, help="Platform_Baseline MPC success rate (%%)")
    p.add_argument("--cand-ol-seeds", type=_parse_seed_list,
                   help="per-seed candidate open-loop rates, comma separated")
    p.add_argument("--cand-mpc-seeds", type=_parse_seed_list,
                   help="per-seed candidate MPC rates, comma separated")
    p.add_argument("--base-ol-seeds", type=_parse_seed_list,
                   help="per-seed baseline open-loop rates, comma separated")
    p.add_argument("--base-mpc-seeds", type=_parse_seed_list,
                   help="per-seed baseline MPC rates, comma separated")
    p.add_argument("--paper-ol", type=float, default=PAPER_OL,
                   help=f"paper open-loop target (default {PAPER_OL})")
    p.add_argument("--paper-mpc", type=float, default=PAPER_MPC,
                   help=f"paper MPC target (default {PAPER_MPC})")
    p.add_argument("--margin-pts", type=float, default=MARGIN_PTS,
                   help=f"inconclusive-at-or-below margin in points (default {MARGIN_PTS})")
    p.add_argument("--se-pts", type=float, default=SE_PTS,
                   help=f"reported binomial standard error in points (default {SE_PTS})")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    cand_ol, cand_ol_std, cand_ol_seeds = _resolve("cand-ol", args.cand_ol, args.cand_ol_seeds)
    cand_mpc, cand_mpc_std, cand_mpc_seeds = _resolve("cand-mpc", args.cand_mpc, args.cand_mpc_seeds)
    base_ol, base_ol_std, base_ol_seeds = _resolve("base-ol", args.base_ol, args.base_ol_seeds)
    base_mpc, base_mpc_std, base_mpc_seeds = _resolve("base-mpc", args.base_mpc, args.base_mpc_seeds)

    missing = [n for n, v in (("cand-ol", cand_ol), ("cand-mpc", cand_mpc),
                              ("base-ol", base_ol), ("base-mpc", base_mpc)) if v is None]
    if missing:
        print("error: missing success rates: "
              + ", ".join(f"--{m} (or --{m}-seeds)" for m in missing), file=sys.stderr)
        return 2

    verdict = acceptance_gate(cand_ol, cand_mpc, base_ol, base_mpc,
                              paper_ol=args.paper_ol, paper_mpc=args.paper_mpc,
                              se_pts=args.se_pts, margin_pts=args.margin_pts)
    margin = platform_margin(cand_ol, cand_mpc, base_ol, base_mpc)
    beats_paper = cand_ol > args.paper_ol and cand_mpc > args.paper_mpc
    beats_platform = cand_ol > base_ol and cand_mpc > base_mpc

    print("CCR Acceptance_Gate (Requirement 10) -- PushT, DINOv2 (patch) + proj 14x14x8, L_curv on")
    print(f"  reported noise floor: binomial standard error at n={EVAL_SAMPLES_PER_SEED} near "
          f"p={EVAL_P_NEAR} = ~{args.se_pts:.1f} percentage points "
          f"(exact {binomial_se_pts():.2f})   [Requirement 10.4]")
    print()
    print("  success rates (%):")
    print(f"    candidate  open-loop {_fmt(cand_ol, cand_ol_std, cand_ol_seeds)}")
    print(f"    candidate  MPC       {_fmt(cand_mpc, cand_mpc_std, cand_mpc_seeds)}")
    print(f"    baseline   open-loop {_fmt(base_ol, base_ol_std, base_ol_seeds)}")
    print(f"    baseline   MPC       {_fmt(base_mpc, base_mpc_std, base_mpc_seeds)}")
    print()
    print(f"  condition 1 -- beats Paper_Target ({args.paper_ol} / {args.paper_mpc})"
          f"  [Requirement 10.1]: {'PASS' if beats_paper else 'FAIL'}")
    print(_explain("open-loop", cand_ol, args.paper_ol, "paper"))
    print(_explain("MPC", cand_mpc, args.paper_mpc, "paper"))
    print()
    print(f"  condition 2 -- beats Platform_Baseline  [Requirement 10.2]: "
          f"{'PASS' if beats_platform else 'FAIL'}")
    print(_explain("open-loop", cand_ol, base_ol, "base "))
    print(_explain("MPC", cand_mpc, base_mpc, "base "))
    print()
    print(f"  margin over Platform_Baseline (weaker of the two): {margin:+.2f} pts "
          f"vs the {args.margin_pts:g}-pt inconclusive threshold and the "
          f"~{args.se_pts:.1f}-pt standard error")
    print()

    if not (beats_paper and beats_platform):
        which = []
        if not beats_paper:
            which.append("Paper_Target")
        if not beats_platform:
            which.append("Platform_Baseline")
        print(f"  VERDICT: {verdict.upper()} -- did not beat {' and '.join(which)} on both settings. "
              "One condition alone is a gate failure (Requirement 10.6).")
    elif verdict == "inconclusive":
        print(f"  VERDICT: {verdict.upper()} -- both conditions hold, but the margin "
              f"({margin:+.2f} pts) is at or below {args.margin_pts:g} pts and inside the "
              f"~{args.se_pts:.1f}-pt standard error. Report as inconclusive or extend the "
              "evidence with additional training seeds (Requirement 10.5, needs approval "
              "under Requirement 11.6).")
    else:
        print(f"  VERDICT: {verdict.upper()} -- both conditions hold and the margin "
              f"({margin:+.2f} pts) exceeds {args.margin_pts:g} pts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
