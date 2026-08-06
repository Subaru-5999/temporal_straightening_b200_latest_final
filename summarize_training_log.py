#!/usr/bin/env python3
"""
summarize_training_log.py  --  read a run's training_log.jsonl and say whether the
pilot is readable, comparable and not silently broken.

Why this exists (SHORT_BUDGET_PILOTS.md section 6): a falling total loss says
nothing. What decides a pilot is each term's *share* of the objective, the step
rate against a known run, and whether a new term collapsed to ~0 before it ever
pressured the encoder. This tool prints exactly those three things.

Usage:
    python summarize_training_log.py <run_dir>
    python summarize_training_log.py <run_dir> --latest --metrics loss
    python summarize_training_log.py <run_dir> --compare <reference_run_dir>
    python summarize_training_log.py <run_dir> --collapse-check

Record schema (one JSON object per line, see design.md section 9):
    {"global_iter": 4000, "epoch": 1, "iter_in_epoch": 4000, "wall_time_s": 2216.4,
     "it_per_s": 1.81, "loss": 0.2166,
     "terms": {"prediction": {"scaled": 0.0118, "share": 0.0545}, ...},
     "enabled_terms": ["prediction", "curvature", "ccr", "decoder"],
     "ccr": {"enabled": true, "raw": 0.412, "lambda_cf": 0.1, "rho": 0.05,
             "rollout_len": 5, "action_source": "synthetic",
             "synthesized_action_frames": 3}}

The `ccr` block's `enabled` flag says whether the CCR term actually contributed to
that iteration; when it is false the block carries only {"enabled": false,
"lambda_cf": ...}. Records written before that flag existed have no `enabled` key --
they are read as "unknown", never as enabled.

Stdlib only (json, argparse, pathlib, statistics) so it runs on the pod and on a
dev box with no torch. Read-only: it never writes into the run directory.

Exit codes: 0 = nothing to report; 1 = a requested check FAILED
(--collapse-check fired, or --strict with a step-rate/compare failure);
2 = the log could not be read at all.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

LOG_BASENAME = "training_log.jsonl"

# REPRODUCTION.md: PushT trains at ~2.9 it/s on the B200 MIG slice.
REFERENCE_IT_PER_S = 2.9
# Requirement 11.7: a >50% step-time regression must be reported before the Full_Run.
# step_time * 1.5  <=>  rate / 1.5  ->  2.9 / 1.5 = 1.93 it/s.
REGRESSION_FACTOR = 1.5

# Requirement 8.6 / SHORT_BUDGET_PILOTS.md section 6: a term whose share falls to
# ~0 early absorbed the task without pressuring the encoder. That looks like
# success and is not.
COLLAPSE_SHARE = 0.001   # 0.1%
COLLAPSE_WINDOW = 1000   # iterations

STEP_ROW_ITER = 200      # Requirement 8.4: the row the arms are compared on

METRIC_GROUPS = ("loss", "terms", "rate", "ccr", "all")

RULE = "=" * 78
THIN = "-" * 78


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def resolve_log_path(run_dir):
    """Accept either a run directory or the JSONL file itself."""
    p = Path(run_dir)
    if p.is_dir():
        return p / LOG_BASENAME
    if p.suffix == ".jsonl" or p.is_file():
        return p
    return p / LOG_BASENAME


def load_records(path):
    """
    Parse a JSONL telemetry log leniently.

    A killed job can leave a truncated final line even though the writer flushes
    per record, and this tool is read on the pod precisely to diagnose crashed
    runs. So malformed lines are counted and skipped, never fatal.

    Returns (records, report) where report holds the skip counters.
    """
    report = {"lines": 0, "malformed": 0, "not_object": 0, "no_iter": 0,
              "first_bad_line": None}
    records = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            report["lines"] += 1
            try:
                obj = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                report["malformed"] += 1
                if report["first_bad_line"] is None:
                    report["first_bad_line"] = lineno
                continue
            if not isinstance(obj, dict):
                report["not_object"] += 1
                if report["first_bad_line"] is None:
                    report["first_bad_line"] = lineno
                continue
            if as_int(obj.get("global_iter")) is None:
                report["no_iter"] += 1
            records.append(obj)
    records.sort(key=lambda r: (as_int(r.get("global_iter")) is None,
                                as_int(r.get("global_iter")) or 0))
    return records, report


def as_float(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None  # drop NaN


def as_int(value):
    f = as_float(value)
    return int(f) if f is not None else None


# --------------------------------------------------------------------------
# record accessors (every one tolerates missing keys)
# --------------------------------------------------------------------------
def term_entries(rec):
    """
    name -> {"scaled": float|None, "share": float|None, "share_calc": float|None}

    `share_calc` is scaled / loss recomputed here, so the printed table can be
    checked against the writer's own arithmetic instead of trusted.
    """
    terms = rec.get("terms")
    if not isinstance(terms, dict):
        return {}
    total = as_float(rec.get("loss"))
    out = {}
    for name, entry in terms.items():
        if isinstance(entry, dict):
            scaled = as_float(entry.get("scaled"))
            share = as_float(entry.get("share"))
        else:
            # tolerate a bare number written instead of the {scaled, share} block
            scaled, share = as_float(entry), None
        calc = None
        if scaled is not None and total not in (None, 0.0):
            calc = scaled / total
        out[str(name)] = {"scaled": scaled, "share": share, "share_calc": calc}
    return out


def enabled_terms(rec):
    val = rec.get("enabled_terms")
    if isinstance(val, list):
        return [str(v) for v in val]
    return []


def iter_of(rec):
    return as_int(rec.get("global_iter"))


def find_record(records, target_iter):
    for rec in records:
        if iter_of(rec) == target_iter:
            return rec
    return None


def nearest_record(records, target_iter):
    cand = [r for r in records if iter_of(r) is not None]
    if not cand:
        return None
    return min(cand, key=lambda r: abs(iter_of(r) - target_iter))


def fmt(value, width=10, places=6):
    if value is None:
        return f"{'n/a':>{width}}"
    return f"{value:>{width}.{places}f}"


def fmt_pct(value, width=8):
    if value is None:
        return f"{'n/a':>{width}}"
    return f"{100.0 * value:>{width - 1}.3f}%"


def fmt_signed(value, width=11, places=6):
    if value is None:
        return f"{'n/a':>{width}}"
    return f"{value:>+{width}.{places}f}"


# --------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------
def print_header(path, records, report):
    print(RULE)
    print(f"TRAINING LOG  {path}")
    print(RULE)
    iters = [iter_of(r) for r in records if iter_of(r) is not None]
    span = f"{min(iters)} .. {max(iters)}" if iters else "none"
    print(f"  records parsed        : {len(records)}  (global_iter {span})")
    skipped = report["malformed"] + report["not_object"]
    if skipped:
        print(f"  WARNING: skipped {skipped} unreadable line(s) "
              f"({report['malformed']} malformed JSON, {report['not_object']} non-object); "
              f"first at line {report['first_bad_line']}")
        print("           a truncated final line is the normal signature of a killed job")
    if report["no_iter"]:
        print(f"  WARNING: {report['no_iter']} record(s) carry no global_iter "
              f"(excluded from --compare matching)")
    if records:
        union = []
        for rec in records:
            for name in enabled_terms(rec) or term_entries(rec).keys():
                if name not in union:
                    union.append(name)
        print(f"  terms seen            : {', '.join(union) if union else 'none'}")


def print_term_table(rec, label):
    it = iter_of(rec)
    total = as_float(rec.get("loss"))
    print()
    print(f"{label}  (global_iter {it if it is not None else 'n/a'}"
          f", epoch {rec.get('epoch', 'n/a')}"
          f", wall_time_s {rec.get('wall_time_s', 'n/a')})")
    print(THIN)
    print(f"  {'term':<14}{'scaled':>12}{'share':>10}{'share=scaled/loss':>22}")
    entries = term_entries(rec)
    if not entries:
        print("  (no `terms` block in this record)")
    shares = []
    mismatched = []
    for name, entry in sorted(entries.items(),
                              key=lambda kv: -(kv[1]["share"] or kv[1]["share_calc"] or 0.0)):
        print(f"  {name:<14}{fmt(entry['scaled'], 12)}{fmt_pct(entry['share'], 10)}"
              f"{fmt_pct(entry['share_calc'], 22)}")
        share = entry["share"] if entry["share"] is not None else entry["share_calc"]
        if share is not None:
            shares.append(share)
        if entry["share"] is not None and entry["share_calc"] is not None:
            if abs(entry["share"] - entry["share_calc"]) > 1e-3:
                mismatched.append(name)
    share_sum = sum(shares) if shares else None
    print(f"  {'TOTAL':<14}{fmt(total, 12)}{fmt_pct(share_sum, 10)}")
    if mismatched:
        print(f"  WARNING: recorded share disagrees with scaled/loss for: "
              f"{', '.join(mismatched)}")
    if share_sum is not None and abs(share_sum - 1.0) > 0.01:
        print(f"  NOTE: shares sum to {100.0 * share_sum:.2f}%, not ~100% -- "
              f"a term is missing from the record or `loss` includes something untracked")


CCR_KEY_ORDER = ("enabled", "raw", "lambda_cf", "rho", "rollout_len", "action_source",
                 "synthesized_action_frames")


def ccr_enabled(block):
    """
    True / False from the block's own `enabled` flag, or None when the key is absent.

    None means "unknown, this record predates the flag" -- not "enabled". Records
    written before the flag existed are still being appended to by an in-flight run,
    so they must stay readable.
    """
    if not isinstance(block, dict) or "enabled" not in block:
        return None
    value = block["enabled"]
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        low = value.strip().lower()
        if low in ("true", "yes", "1"):
            return True
        if low in ("false", "no", "0"):
            return False
        return None
    num = as_float(value)
    return None if num is None else num != 0.0


def print_ccr_block(rec):
    block = rec.get("ccr")
    if not isinstance(block, dict):
        return
    enabled = ccr_enabled(block)
    print()
    print("CCR ARM (self-describing block)")
    print(THIN)

    if enabled is False:
        # Nothing ran, so there is no arm to table: rho / rollout_len /
        # action_source / synthesized_action_frames would describe a rollout that
        # never happened. lambda_cf is the reason it is off.
        lam = block.get("lambda_cf", "n/a")
        print(f"  CCR disabled (lambda_cf={lam}) -- no CCR term contributed to this "
              f"iteration; arm fields are not reported")
        return

    for key in CCR_KEY_ORDER:
        if key in block:
            print(f"  {key:<26}{block[key]}")
    for key in sorted(k for k in block if k not in CCR_KEY_ORDER):
        print(f"  {key:<26}{block[key]}")

    if enabled is None:
        print("  NOTE: this record predates the `enabled` field, so whether the CCR "
              "term actually ran cannot be confirmed from this block")
        print("        use `enabled_terms` (a `ccr` entry) as the authoritative signal")
        return

    # enabled is True: the arm fields describe something that really ran, so the
    # synthetic-vs-logged distinction is meaningful here and only here.
    if block.get("action_source") == "synthetic" and \
            as_int(block.get("synthesized_action_frames")) == 0:
        print("  WARNING: action_source=synthetic with synthesized_action_frames=0 is "
              "silently a `logged` arm -- the launch is wrong")


def print_step_rate(records, reference, floor):
    """Requirement 11.7 step-rate regression check, printed as a PASS/FAIL with numbers."""
    rates = [as_float(r.get("it_per_s")) for r in records]
    rates = [x for x in rates if x is not None and x > 0]
    print()
    print("STEP RATE")
    print(THIN)
    if not rates:
        print("  no it_per_s values in the log -- step-rate regression check SKIPPED")
        return None
    # the first record covers process warmup (dataset workers, DINOv2 load), so it
    # is reported but excluded from the verdict when there is anything else to use
    body = rates[1:] if len(rates) > 1 else rates
    median = statistics.median(body)
    print(f"  records with it_per_s : {len(rates)}")
    print(f"  first / latest        : {rates[0]:.3f} / {rates[-1]:.3f} it/s")
    print(f"  median (excl. first)  : {median:.3f} it/s")
    print(f"  min / max             : {min(body):.3f} / {max(body):.3f} it/s")
    print(f"  reference run         : {reference:.3f} it/s")
    print(f"  regression floor      : {floor:.3f} it/s "
          f"(= {reference:.2f} / {REGRESSION_FACTOR}, i.e. >50% step-time regression)")
    ok = median >= floor
    slowdown = (reference / median - 1.0) * 100.0 if median > 0 else float("inf")
    verdict = "PASS" if ok else "FAIL"
    print(f"  step-rate check       : {verdict}  "
          f"({median:.3f} it/s vs floor {floor:.3f} it/s; "
          f"step time {slowdown:+.1f}% vs reference)")
    if not ok:
        print("  Requirement 11.7: report this regression and revise the compute plan "
              "BEFORE launching the Full_Run")
    return ok


def print_step_row(records, target_iter):
    """Requirement 8.4: the step-N row is what makes two arms comparable."""
    rec = find_record(records, target_iter)
    if rec is not None:
        print_term_table(rec, f"STEP-{target_iter} ROW")
        return rec
    near = nearest_record(records, target_iter)
    print()
    print(f"STEP-{target_iter} ROW")
    print(THIN)
    if near is None:
        print(f"  no record at global_iter {target_iter} and no usable record to fall back on")
        return None
    print(f"  no record at global_iter {target_iter} "
          f"(telemetry cadence may differ); nearest is {iter_of(near)}")
    print_term_table(near, f"NEAREST ROW TO STEP-{target_iter}")
    return near


def print_compare(records, ref_records, target_iter, rtol, max_rows=12):
    """
    Row-by-row delta against a reference run, matched on global_iter, for the
    shared terms (Requirement 8.4).
    """
    print()
    print(RULE)
    print("COMPARE AGAINST REFERENCE RUN (matched on global_iter, shared terms only)")
    print(RULE)
    ours = {iter_of(r): r for r in records if iter_of(r) is not None}
    theirs = {iter_of(r): r for r in ref_records if iter_of(r) is not None}
    shared_iters = sorted(set(ours) & set(theirs))
    if not shared_iters:
        print("  no global_iter values in common -- the two runs cannot be compared "
              "row by row (different telemetry cadence?)")
        print(f"  ours   : {sorted(ours)[:8]}{' ...' if len(ours) > 8 else ''}")
        print(f"  ref    : {sorted(theirs)[:8]}{' ...' if len(theirs) > 8 else ''}")
        return None
    print(f"  matched rows: {len(shared_iters)} "
          f"(ours {len(ours)}, reference {len(theirs)})")
    only_ours = sorted(set(ours) - set(theirs))
    only_ref = sorted(set(theirs) - set(ours))
    if only_ours:
        print(f"  unmatched in ours     : {len(only_ours)} row(s), "
              f"e.g. {only_ours[:5]}")
    if only_ref:
        print(f"  unmatched in reference: {len(only_ref)} row(s), "
              f"e.g. {only_ref[:5]}")

    shown = shared_iters
    if 0 < max_rows < len(shared_iters):
        shown = shared_iters[:max_rows]
        if target_iter in shared_iters and target_iter not in shown:
            shown = sorted(set(shown[:max_rows - 1]) | {target_iter})
        print(f"  showing the first {len(shown)} matched row(s) of "
              f"{len(shared_iters)} -- pass --compare-rows 0 for all")

    for it in shown:
        a, b = ours[it], theirs[it]
        ea, eb = term_entries(a), term_entries(b)
        shared_terms = sorted(set(ea) & set(eb))
        print()
        print(f"  global_iter {it}")
        print(f"    {'term':<12}{'ours':>10}{'ref':>10}{'delta':>11}"
              f"{'sh ours':>9}{'sh ref':>9}{'d share':>9}")
        for name in shared_terms:
            sa, sb = ea[name]["scaled"], eb[name]["scaled"]
            ha = ea[name]["share"] if ea[name]["share"] is not None else ea[name]["share_calc"]
            hb = eb[name]["share"] if eb[name]["share"] is not None else eb[name]["share_calc"]
            d = sa - sb if (sa is not None and sb is not None) else None
            dh = ha - hb if (ha is not None and hb is not None) else None
            print(f"    {name:<12}{fmt(sa, 10)}{fmt(sb, 10)}{fmt_signed(d, 11)}"
                  f"{fmt_pct(ha, 9)}{fmt_pct(hb, 9)}{fmt_signed(dh, 9, 4)}")
        for name in sorted(set(ea) ^ set(eb)):
            side = "ours only" if name in ea else "reference only"
            print(f"    {name:<12}({side})")
        la, lb = as_float(a.get("loss")), as_float(b.get("loss"))
        dl = la - lb if (la is not None and lb is not None) else None
        print(f"    {'loss':<12}{fmt(la, 10)}{fmt(lb, 10)}{fmt_signed(dl, 11)}")
        ra, rb = as_float(a.get("it_per_s")), as_float(b.get("it_per_s"))
        dr = ra - rb if (ra is not None and rb is not None) else None
        print(f"    {'it_per_s':<12}{fmt(ra, 10, 3)}{fmt(rb, 10, 3)}{fmt_signed(dr, 11, 3)}")

    # Requirement 8.4 verdict, stated on the step-N row only.
    print()
    print(THIN)
    print(f"  Requirement 8.4 verdict on the step-{target_iter} row "
          f"(shared terms within rtol={rtol:g})")
    a, b = ours.get(target_iter), theirs.get(target_iter)
    if a is None or b is None:
        print(f"    SKIPPED: step-{target_iter} row missing in "
              f"{'ours' if a is None else 'the reference'}")
        return None
    ea, eb = term_entries(a), term_entries(b)
    shared_terms = sorted(set(ea) & set(eb))
    if not shared_terms:
        print("    SKIPPED: the two step rows share no term")
        return None
    ok = True
    for name in shared_terms:
        sa, sb = ea[name]["scaled"], eb[name]["scaled"]
        if sa is None or sb is None:
            print(f"    {name:<14}n/a (missing scaled value) -- cannot compare")
            ok = False
            continue
        tol = rtol * max(abs(sa), abs(sb), 1e-12)
        hit = abs(sa - sb) <= tol
        ok = ok and hit
        print(f"    {name:<14}{'MATCH' if hit else 'DIFFER'}  "
              f"ours {sa:.6f} vs ref {sb:.6f} (delta {sa - sb:+.6f}, tol {tol:.6f})")
    print(f"    step-{target_iter} loss match: {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("    the arms are not directly comparable on this row -- check seed, "
              "data order and Protocol_Invariants before interpreting anything else")
    return ok


def print_collapse_check(records, window, threshold):
    """
    Requirement 8.6: flag any term whose share falls below `threshold` within the
    first `window` iterations. A new term that collapses to ~0 early absorbed the
    task without pressuring the encoder; that is a failure, not a success.
    """
    print()
    print(RULE)
    print(f"COLLAPSE CHECK (share < {100.0 * threshold:g}% within the first "
          f"{window} iterations)")
    print(RULE)
    in_window = [r for r in records
                 if iter_of(r) is not None and iter_of(r) <= window]
    if not in_window:
        print(f"  no records at global_iter <= {window} -- check SKIPPED "
              f"(the run has no early telemetry to judge)")
        return None
    print(f"  records in window: {len(in_window)} "
          f"(global_iter {iter_of(in_window[0])} .. {iter_of(in_window[-1])})")
    print()
    print(f"  {'term':<14}{'min share':>12}{'at iter':>10}{'last share':>12}"
          f"{'first below':>13}  verdict")
    names = []
    for rec in in_window:
        for name in term_entries(rec):
            if name not in names:
                names.append(name)
    collapsed = []
    for name in sorted(names):
        obs = []
        for rec in in_window:
            entry = term_entries(rec).get(name)
            if not entry:
                continue
            share = entry["share"] if entry["share"] is not None else entry["share_calc"]
            if share is not None:
                obs.append((iter_of(rec), share))
        if not obs:
            print(f"  {name:<14}{'n/a':>12}{'-':>10}{'n/a':>12}{'-':>13}  NO DATA")
            continue
        s_min = min(s for _, s in obs)
        it_min = next(i for i, s in obs if s == s_min)
        last = obs[-1][1]
        first_below = next((i for i, s in obs if s < threshold), None)
        bad = first_below is not None
        if bad:
            collapsed.append((name, first_below, it_min, s_min))
        print(f"  {name:<14}{fmt_pct(s_min, 12)}{it_min:>10}{fmt_pct(last, 12)}"
              f"{(str(first_below) if bad else '-'):>13}  "
              f"{'COLLAPSED' if bad else 'ok'}")
    print()
    if collapsed:
        print("  FAIL (Requirement 8.6): "
              + "; ".join(f"`{n}` first fell below the threshold at iter {fb} "
                          f"(min {100.0 * s:.4f}% at iter {im})"
                          for n, fb, im, s in collapsed))
        print("  Record the term as having absorbed the task without pressuring the "
              "encoder. Do NOT report this Pilot_Run as a success.")
        return False
    print("  PASS: no term fell below the collapse threshold inside the window")
    return True


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_metrics(raw):
    if raw is None:
        return set(METRIC_GROUPS)
    wanted = {tok.strip().lower() for tok in raw.replace(",", " ").split() if tok.strip()}
    unknown = wanted - set(METRIC_GROUPS)
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown metric group(s): {', '.join(sorted(unknown))}. "
            f"choose from: {', '.join(METRIC_GROUPS)}")
    if "all" in wanted or not wanted:
        return set(METRIC_GROUPS)
    # `loss` is the shape used in SHORT_BUDGET_PILOTS.md section 6: the term /
    # scaled / share table plus the total.
    if "loss" in wanted:
        wanted.add("terms")
    return wanted


def build_parser():
    ap = argparse.ArgumentParser(
        description="Summarize a run's training_log.jsonl: term/scaled/share table, "
                    "step rate, step-200 row, reference deltas and collapse check.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="run directory containing training_log.jsonl "
                                    "(the JSONL file itself is also accepted)")
    ap.add_argument("--compare", metavar="REFERENCE_RUN_DIR", default=None,
                    help="print the row-by-row delta against a reference run's JSONL, "
                         "matched on global_iter, for the shared terms (Requirement 8.4)")
    ap.add_argument("--collapse-check", action="store_true",
                    help=f"flag any term whose share falls below "
                         f"{100.0 * COLLAPSE_SHARE:g}%% within the first "
                         f"{COLLAPSE_WINDOW} iterations (Requirement 8.6)")
    ap.add_argument("--latest", action="store_true",
                    help="show the last logged iteration (the default)")
    ap.add_argument("--iter", type=int, default=None, metavar="N",
                    help="show the record at global_iter N instead of the latest")
    ap.add_argument("--metrics", default=None, metavar="GROUPS",
                    help=f"comma-separated subset of {{{', '.join(METRIC_GROUPS)}}} "
                         f"to print (default: all)")
    ap.add_argument("--step-row", type=int, default=STEP_ROW_ITER, metavar="N",
                    help=f"iteration of the comparability row (default {STEP_ROW_ITER})")
    ap.add_argument("--reference-it-per-s", type=float, default=REFERENCE_IT_PER_S,
                    metavar="R", help=f"reference step rate (default {REFERENCE_IT_PER_S}, "
                                      f"the PushT reference run)")
    ap.add_argument("--it-per-s-floor", type=float, default=None, metavar="F",
                    help="step-rate floor (default: reference / 1.5, i.e. 1.93 it/s "
                         "for the PushT reference)")
    ap.add_argument("--collapse-share", type=float, default=COLLAPSE_SHARE, metavar="S",
                    help=f"collapse threshold as a fraction (default {COLLAPSE_SHARE})")
    ap.add_argument("--collapse-window", type=int, default=COLLAPSE_WINDOW, metavar="N",
                    help=f"collapse window in iterations (default {COLLAPSE_WINDOW})")
    ap.add_argument("--match-rtol", type=float, default=0.05, metavar="T",
                    help="relative tolerance for the step-row match verdict (default 0.05)")
    ap.add_argument("--compare-rows", type=int, default=12, metavar="N",
                    help="how many matched rows --compare prints in full (default 12, "
                         "0 for all); the step row is always included")
    ap.add_argument("--strict", action="store_true",
                    help="also exit non-zero when the step-rate or step-row match check fails")
    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    try:
        metrics = parse_metrics(args.metrics)
    except argparse.ArgumentTypeError as exc:
        ap.error(str(exc))
        return 2

    log_path = resolve_log_path(args.run_dir)
    if not log_path.is_file():
        print(f"ERROR: no telemetry log at {log_path.resolve()}", file=sys.stderr)
        print("       an empty run directory one minute into a run means the job crashed "
              "(Requirement 6.9)", file=sys.stderr)
        return 2

    records, report = load_records(log_path)
    print_header(log_path, records, report)
    if not records:
        print("\nERROR: no readable records -- nothing to summarize", file=sys.stderr)
        return 2

    # ---- selected row -------------------------------------------------
    if args.iter is not None:
        rec = find_record(records, args.iter)
        if rec is None:
            near = nearest_record(records, args.iter)
            print(f"\nNOTE: no record at global_iter {args.iter}; "
                  f"showing nearest ({iter_of(near)})")
            rec = near
        label = f"ROW AT global_iter {iter_of(rec)}"
    else:
        rec = records[-1]
        label = "LATEST ROW"

    if "terms" in metrics or "loss" in metrics:
        print_term_table(rec, label)
    if "ccr" in metrics:
        print_ccr_block(rec)

    rate_ok = None
    if "rate" in metrics:
        floor = (args.it_per_s_floor if args.it_per_s_floor is not None
                 else args.reference_it_per_s / REGRESSION_FACTOR)
        rate_ok = print_step_rate(records, args.reference_it_per_s, floor)

    step_rec = None
    if "terms" in metrics or "loss" in metrics:
        if iter_of(rec) != args.step_row:
            step_rec = print_step_row(records, args.step_row)
        else:
            print(f"\n(the row above IS the step-{args.step_row} row)")
            step_rec = rec

    match_ok = None
    if args.compare:
        ref_path = resolve_log_path(args.compare)
        if not ref_path.is_file():
            print(f"\nERROR: no reference telemetry log at {ref_path.resolve()}",
                  file=sys.stderr)
            return 2
        ref_records, ref_report = load_records(ref_path)
        print()
        print(f"reference log: {ref_path}  "
              f"({len(ref_records)} record(s), "
              f"{ref_report['malformed'] + ref_report['not_object']} skipped)")
        if not ref_records:
            print("ERROR: the reference log has no readable records", file=sys.stderr)
            return 2
        match_ok = print_compare(records, ref_records, args.step_row,
                                 args.match_rtol, args.compare_rows)

    collapse_ok = None
    if args.collapse_check:
        collapse_ok = print_collapse_check(records, args.collapse_window,
                                           args.collapse_share)

    # ---- verdict roll-up ---------------------------------------------
    print()
    print(RULE)
    print("VERDICTS")
    print(RULE)
    def show(name, value):
        state = {True: "PASS", False: "FAIL", None: "not run"}[value]
        print(f"  {name:<34}{state}")
    show("step-rate floor (Req 11.7)", rate_ok)
    show(f"step-{args.step_row} match (Req 8.4)", match_ok)
    show("collapse check (Req 8.6)", collapse_ok)
    print("  loss shares, not loss values, decide a pilot (Req 8.3)")

    failed = collapse_ok is False
    if args.strict:
        failed = failed or rate_ok is False or match_ok is False
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
