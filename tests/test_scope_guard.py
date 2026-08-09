"""Scope-containment guard for the CCR, aggregated-space and ACS feature branches.

**Validates: Requirements 5.2, 5.4, 5.6** (counterfactual-curvature-regularization)

**Property 9: Frozen sources are byte-identical to the base revision**
**Validates: Requirements 4.3, 4.4** (aggregated-space-planning-cost)

**Property 16: Frozen sources are byte-identical to the base revision**
**Validates: Requirements 14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 14.7**
(action-conditioned-straightening)

Requirement 5.6 confines training-side changes to a small allowlist; Requirements 5.2 and 5.4 additionally
forbid touching the planning and dataset paths at all. The aggregated-space feature adds root-level
``plan.py`` to that frozen set (Requirement 4.4) and two allowlist entries (Requirement 4.5). ACS adds
``PROGRESS_ACS.md`` to the allowlist (ACS Requirements 14.6, 14.7 — the only new non-test file it
introduces) and adds ``models/vit.py`` and ``models/dino.py`` to the frozen set (ACS Requirement 14.5), so
ACS cannot touch the encoder or the predictor source. This module is the only automated check of any of
those statements, so it is a non-optional gate rather than an optional property test.

Three assertions:

1. Every path the feature branch changed (committed, staged, unstaged or untracked) is in the allowlist.
2. Every ``planning/*.py`` and ``datasets/*.py`` file, plus root-level ``plan.py`` and ``models/dino.py``,
   hashes equal to its content at the base revision.
3. ``models/vit.py`` hashes equal to its **current** content, pinned as a literal digest below.

Assertion 3 records a tension rather than papering over it. ``models/vit.py`` is already in
``ALLOWED_FILES`` under the CCR SDPA scope amendment, so it legitimately differs from the base revision:
comparing it against ``BASE_REV`` like the other frozen files would fail on a change ACS did not cause,
and dropping it from the frozen set would leave ACS Requirement 14.5 unchecked for that file. It is
therefore frozen against the CCR-amended content that is in the tree today, exactly the way
``PREEXISTING_FILES`` exempts pre-ACS additions from the allowlist assertion. The equivalence of that
amended content to the original attention math is *not* this module's job; it is guarded by
``tests/test_vit_sdpa_equivalence.py``, which checks forward and gradient agreement in float64 and that the
block-causal mask is still enforced on the fast branch.

Implementation notes:

* The base revision defaults to the repository's initial commit and is overridable through ``CCR_BASE_REV``
  so the guard keeps working once feature commits land on top.
* The whole module is stdlib only (``subprocess``, ``hashlib``, ``pathlib``, ``os``) and skips cleanly when
  git is unavailable, so a source-tarball checkout does not fail the suite.
* Hashes are taken over newline-normalized bytes. The repository is checked out with ``core.autocrlf=true``
  on Windows, so a working-tree file legitimately holds CRLF where the stored blob holds LF; comparing raw
  bytes would report every file as modified on Windows and none on Linux.
"""

import hashlib
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Base revision the feature branch is measured against. ``d73b9c6`` ("Initial commit") is the pre-feature
#: state of this repository; override via the environment once feature commits exist.
BASE_REV = os.environ.get("CCR_BASE_REV", "d73b9c6")

#: Requirement 5.6 allowlist: the four in-scope source files, plus the new standalone scripts.
ALLOWED_FILES = frozenset(
    {
        # Requirement 5.6 in-scope source files.
        "models/visual_world_model.py",
        "conf/train.yaml",
        "train.py",
        "custom_resolvers.py",
        # New standalone scripts (design "Scope" section).
        "probe_ccr_curvature.py",
        "summarize_training_log.py",
        "ccr_acceptance_gate.py",
        "run_ccr_pilot.sh",
        # Test-only configuration (task 1.1). Deliberately NOT requirements-train.txt/-plan.txt: the
        # training and planning images stay unchanged.
        "requirements-dev.txt",
        "pytest.ini",
        # Project-direction document, added 2026-08-09. The north-star record of what the project is chasing
        # (the ICLR objective, the real acceptance predicate, the closed arms, and the rules earned by the
        # failures logged in the PROGRESS_*.md files). Documentation only: imported by nothing, executed by
        # nothing, and read by no source file, so it cannot influence a measured number. It is allowlisted
        # because losing it to a scope violation is exactly the context loss it exists to prevent.
        "RESEARCH_GOAL.md",
        # Progress log for the rotation direction, added 2026-08-09. Same category as the other
        # PROGRESS_*.md files: measurements, pre-registered gates and errors, read by no source file.
        # It holds the rung-1 gate that must exist before the probe runs, so it lands with the code
        # rather than after it.
        "PROGRESS_ROT.md",
        # Required addition, not in the Requirement 5.6 prose: the repo's .gitignore ignores "*.sh" with an
        # explicit per-script negation allowlist, so `run_ccr_pilot.sh` could not be tracked at all without
        # adding a `!run_ccr_pilot.sh` line. That single line is enabling infrastructure for a file the
        # design mandates, and it touches no training, planning or dataset behaviour.
        ".gitignore",
        # Mandated project artifact: `SHORT_BUDGET_PILOTS.md` section 10 requires results to be recorded
        # together with their caveats in a `PROGRESS_*.md`, so this file is required rather than an ad-hoc
        # addition. It is prose only and contains no training, planning or dataset code.
        "PROGRESS_CCR.md",
        # Contingency plan documenting the alternative routes to take if Plan A fails its acceptance gate.
        # Same rationale as `PROGRESS_CCR.md` above: prose only, no training, planning or dataset code.
        "PLAN_B_ALTERNATIVES.md",
        # SCOPE AMENDMENT, recorded rather than quietly taken. `models/vit.py` is not in the
        # Requirement 5.6 list, and it is edited here for one reason: it materialises a
        # (batch, heads, 588, 588) attention score matrix, which made CCR at L=5 OOM a 45 GB MIG slice
        # and then run 4.5x slower than the baseline (0.573 vs 2.548 it/s). The projected Full_Run was
        # 60 h against 17 h planned.
        #
        # The amendment is admissible because the change is strictly ADDITIVE AND DEFAULT-OFF:
        # `Attention.use_sdpa` is False unless the `sdpa_attention` context manager turns it on, and only
        # `VWorldModel.compute_ccr` ever does. Every pre-existing caller -- the baseline prediction loss,
        # `rollout`, `plan.py`, `planning/*`, `Trainer.openloop_rollout` -- takes the original ops in the
        # original order. That is what lets the measured Platform_Baseline (75.33 OL / 82.00 MPC) stand
        # without a 12 h retrain, and it is the whole reason the change was made this way rather than by
        # switching the file's attention outright.
        #
        # Guarded by `tests/test_vit_sdpa_equivalence.py`, which checks forward AND gradient agreement in
        # float64 and that the block-causal mask is still enforced on the fast branch.
        "models/vit.py",
        # SCOPE AMENDMENT for the aggregated-space planning cost.
        #
        # `plan.py` builds its planning objective with `hydra.utils.call(cfg_dict["objective"])` and
        # passes NOTHING else -- no model, no encoder, no planner handle. `planning/objectives.py`
        # correspondingly receives no handle on the world model: `create_objective_fn(alpha, base,
        # mode)` closes over three scalars. So Agg_Head, which lives on the checkpoint's
        # `DinoV2Encoder`, cannot reach the objective through any frozen argument channel, and the
        # aggregated-space term MUST be injected from outside the frozen paths. That is the entire
        # reason this feature is two new root-level files rather than an edit to
        # `planning/objectives.py`.
        #
        # `agg_objectives.py` computes L_agg and L_plan. It IMPORTS `planning.objectives` and CALLS
        # `create_objective_fn` twice -- once with the configured alpha for L_spatial, once with
        # alpha=0 on aggregated-space features for L_agg -- and rebinds nothing in that module, so
        # the frozen coefficient and stage-dispatch logic is reused rather than copied and cannot
        # drift from it.
        #
        # `plan_agg.py` is the entry point. It calls `plan.planning_main` as imported. It rewrites
        # `_target_` in its OWN cfg_dict, and it rebinds `plan.PlanEvaluator` to a subclass that
        # delegates to `super().eval_actions` and records the per-episode success vector the frozen
        # evaluator reduces to a mean -- the paired comparison needs those vectors and nothing
        # persists them. Both are runtime attribute rebinds in the wrapper's own process: no file
        # under `planning/`, no file under `datasets/` and not `plan.py` is edited, and the
        # byte-identity assertion below now covers `plan.py` as well to keep that honest.
        #
        # Guarded by `tests/test_agg_zero_bitwise.py`, which checks that at Agg_Weight 0 the returned
        # tensor is BITWISE equal to the unmodified objective's for arbitrary inputs including
        # non-finite ones; by `tests/test_agg_recording_evaluator.py`, which checks the recording
        # evaluator returns its delegate's result unchanged; and by
        # `tests/test_agg_objectives_untouched.py`, which checks every attribute of
        # `planning.objectives` keeps its original identity after use.
        "agg_objectives.py",
        "plan_agg.py",
        # ACS Requirements 14.6 and 14.7: the only new non-test file the action-conditioned-straightening
        # feature adds. Same rationale as `PROGRESS_CCR.md` above -- prose only, no training, planning or
        # dataset code -- and it is a required artifact rather than an ad-hoc addition, because ACS
        # Requirement 2.1 obliges the Stage-0 verdict rules to be written down *before* the Stage-0
        # statistics are collected. Allowlisted here, in the first ACS task, so the guard never reports a
        # violation for a file the feature is required to create; task 4.5 creates it.
        "PROGRESS_ACS.md",
        # The MCA arm's pre-registration, selected by `PROGRESS_ACS.md` §12 after ACS's Stage 0 returned
        # STOP. Same rationale as `PROGRESS_CCR.md` and `PROGRESS_ACS.md` above -- prose only, no
        # training, planning or dataset code -- and required rather than ad-hoc: the escalation ladder
        # (`SHORT_BUDGET_PILOTS.md` §1, CCR Requirement 11.3) obliges the rung-1 gate to be written down
        # before the offline probe runs, and the rung-2 gate before the pilot launches.
        "PROGRESS_MCA.md",
        # The pod operating protocol (§5.1: pull-only, results return by terminal paste). Added after
        # three separate round trips lost minutes to the same root cause -- assuming the pod's git
        # remotes, credentials, env vars and dataset inventory match the authoring machine's. Prose only.
        # Recorded in the allowlist rather than left as an untracked violation because the alternative is
        # that the lesson lives only in a chat log.
        "AGENT_MEMORY_2.0.md",
        # The aggregated-space arm's measurement log, on the model of `PROGRESS_CCR.md` above: prose
        # only, no training, planning or dataset code. Required rather than ad-hoc for two reasons.
        # Task 11.2 is "[HUMAN] Record the paired zero-weight verdict" and there was nowhere to record
        # it; and Requirement 10 obliges every success rate to be reported with its binomial SE and
        # every claim with its caveats, which is a document, not a log line.
        "PROGRESS_AGG.md",
    }
)

#: Directory prefixes whose entire subtree is allowed.
ALLOWED_PREFIXES = (
    "tests/",  # Requirement 5.6.
    ".kiro/specs/",  # Spec documents for this feature.
    # Project-direction documents, added 2026-08-09. `.kiro/steering/` holds the always-loaded objective
    # summary (`research-goal.md`), whose whole purpose is to survive context loss between sessions; the
    # existing `product.md` / `structure.md` / `tech.md` already live here and predate any arm. Documentation
    # only: no steering file is imported, executed or read by any source file, so nothing here can change a
    # measured number. Kept as a prefix rather than a file entry because steering is inherently a growing set.
    ".kiro/steering/",
    # Pre-existing additions that predate this feature branch and are not part of it: the extracted paper
    # sources (cited by the requirements introduction) and the short-budget-pilot notes. They were already
    # untracked in the working tree before any CCR work started, so failing the guard on them would report a
    # scope violation that this feature did not cause.
    "paper_tex/",
)

#: Pre-existing untracked single files, same rationale as ``paper_tex/``.
PREEXISTING_FILES = frozenset({"SHORT_BUDGET_PILOTS.md"})

#: Directories whose ``*.py`` contents must be byte-identical to the base revision.
FROZEN_DIRS = ("planning", "datasets")

#: Individual files that must be byte-identical to the base revision. Requirement 4.4 of the
#: aggregated-space feature adds root-level ``plan.py``: the wrapper `plan_agg.py` imports it and rebinds
#: `plan.PlanEvaluator` at runtime in its own process, which edits no file, and this is what keeps that
#: honest. ACS Requirement 14.5 adds ``models/dino.py``: ACS touches the encoder only through the existing
#: ``encoder.agg`` call, so the encoder source must not move. These are files rather than directories, hence
#: a separate tuple; both are handed to git as pathspecs by ``FROZEN_PATHSPECS`` below.
#:
#: ``models/vit.py`` is deliberately NOT here -- see ``FROZEN_CURRENT_DIGESTS``.
FROZEN_FILES = ("plan.py", "models/dino.py")

#: Pathspecs passed to ``git ls-tree`` / ``git ls-files`` when collecting the frozen set.
FROZEN_PATHSPECS = FROZEN_DIRS + FROZEN_FILES

#: ACS Requirement 14.5 for ``models/vit.py``, frozen against its **current** content rather than the base
#: revision, with the digest pinned here so the check is a real check and not a tautology.
#:
#: The tension, recorded rather than hidden: ``models/vit.py`` is in ``ALLOWED_FILES`` above under the CCR
#: SDPA scope amendment (the additive, default-off ``Attention.use_sdpa`` path that only
#: ``VWorldModel.compute_ccr`` enables). It therefore differs from ``BASE_REV`` by design, and a naive
#: base-revision hash comparison would fail on a change ACS did not cause. Freezing it against today's
#: CCR-amended bytes gives ACS Requirement 14.5 what it actually asks for -- ACS must not touch the
#: predictor source -- while leaving the CCR amendment where it already lives.
#:
#: Guard of record for the amendment's *correctness* (as opposed to its immutability):
#: ``tests/test_vit_sdpa_equivalence.py``.
#:
#: sha256 over newline-normalized bytes, matching ``_digest`` (see the module docstring on core.autocrlf).
#: If a later, separately argued amendment to ``models/vit.py`` is admitted, update this digest in the same
#: commit and say why in the commit message -- that is the point of pinning it.
FROZEN_CURRENT_DIGESTS = {
    "models/vit.py": "f831b8a942d27ec2c874ad317acbec516b6b20556dcca3d34b7c9cbb229dc364",
}


def _git(*args, binary=False):
    """Run git in the repository root. Returns ``(returncode, stdout)``."""
    completed = subprocess.run(
        ["git", *args],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out = completed.stdout if binary else completed.stdout.decode("utf-8", "replace")
    return completed.returncode, out


def _require_git():
    """Skip rather than fail when there is no usable git work tree or base revision."""
    try:
        code, out = _git("rev-parse", "--is-inside-work-tree")
    except (OSError, FileNotFoundError) as exc:  # git not installed
        pytest.skip(f"git is not available, cannot check scope containment: {exc}")
    if code != 0 or out.strip() != "true":
        pytest.skip(f"{REPO_ROOT} is not a git work tree, cannot check scope containment")
    code, _ = _git("cat-file", "-e", f"{BASE_REV}^{{commit}}")
    if code != 0:
        pytest.skip(
            f"base revision {BASE_REV!r} is not present in this checkout "
            f"(set CCR_BASE_REV to a revision that is)"
        )


def _norm(path):
    """Normalize to forward slashes so the test behaves identically on Windows and Linux."""
    return path.replace("\\", "/").strip()


def _status_paths():
    """Paths reported by ``git status``: staged, unstaged and untracked (ignored files excluded)."""
    code, out = _git("status", "--porcelain", "-uall", "-z", binary=True)
    assert code == 0, "git status failed"
    entries = out.split(b"\0")
    paths = set()
    i = 0
    while i < len(entries):
        entry = entries[i]
        i += 1
        if len(entry) < 4:
            continue
        status = entry[:2].decode("utf-8", "replace")
        paths.add(_norm(entry[3:].decode("utf-8", "surrogateescape")))
        if "R" in status or "C" in status:
            # Rename/copy: the following NUL-separated field is the original path.
            if i < len(entries) and entries[i]:
                paths.add(_norm(entries[i].decode("utf-8", "surrogateescape")))
                i += 1
    return paths


def _diff_paths():
    """Paths differing between the base revision and the current working tree (tracked files only)."""
    code, out = _git("diff", "--name-only", BASE_REV)
    assert code == 0, f"git diff against {BASE_REV} failed"
    return {_norm(line) for line in out.splitlines() if line.strip()}


def changed_paths():
    """The feature branch's full changed-file set: committed, staged, unstaged and untracked."""
    return _diff_paths() | _status_paths()


def _is_allowed(path):
    return (
        path in ALLOWED_FILES
        or path in PREEXISTING_FILES
        or path.startswith(ALLOWED_PREFIXES)
    )


def _digest(data):
    """sha256 over newline-normalized bytes (see module docstring on core.autocrlf)."""
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def _frozen_paths_at_base():
    code, out = _git("ls-tree", "-r", "--name-only", BASE_REV, "--", *FROZEN_PATHSPECS)
    assert code == 0, f"git ls-tree of {FROZEN_PATHSPECS} at {BASE_REV} failed"
    return {p for p in (_norm(line) for line in out.splitlines()) if p.endswith(".py")}


def _frozen_paths_now():
    # cached + untracked, honouring .gitignore, so __pycache__ and checkpoint scratch never register as
    # newly added files.
    code, out = _git(
        "ls-files", "--cached", "--others", "--exclude-standard", "--", *FROZEN_PATHSPECS
    )
    assert code == 0, f"git ls-files of {FROZEN_PATHSPECS} failed"
    return {p for p in (_norm(line) for line in out.splitlines()) if p.endswith(".py")}


def test_changed_files_are_within_the_requirement_5_6_allowlist():
    """Requirement 5.6: training-side changes are confined to the allowlist."""
    _require_git()
    violations = sorted(p for p in changed_paths() if not _is_allowed(p))
    assert not violations, (
        "Requirement 5.6 scope violation: the feature branch changed files outside the allowlist.\n"
        "Out-of-scope paths:\n  " + "\n  ".join(violations) + "\n"
        "Allowed files:\n  " + "\n  ".join(sorted(ALLOWED_FILES)) + "\n"
        "Allowed subtrees:\n  " + "\n  ".join(sorted(ALLOWED_PREFIXES))
    )


def test_frozen_sources_are_unchanged_from_the_base_revision():
    """The frozen paths are untouched.

    Requirements 5.2, 5.4 (CCR), 4.3, 4.4 (aggregated-space) and 14.2, 14.3, 14.4, 14.5 (ACS).

    **Property 9 / Property 16: Frozen sources are byte-identical to the base revision**
    **Validates: Requirements 4.3, 4.4, 14.2, 14.3, 14.4, 14.5**
    """
    _require_git()
    at_base = _frozen_paths_at_base()
    now = _frozen_paths_now()
    assert at_base, (
        f"no planning/*.py, datasets/*.py, plan.py or models/dino.py files found at base revision "
        f"{BASE_REV}; the guard would silently pass"
    )
    # Requirement 4.4 (plan.py) and ACS Requirement 14.5 (models/dino.py): each named file must actually be
    # in the compared set, so a path-collection regression cannot silently drop it and leave the assertion
    # vacuously true.
    for frozen_file in FROZEN_FILES:
        assert frozen_file in at_base, (
            f"{frozen_file} is not in the frozen set collected at {BASE_REV}; Requirement 4.4 / ACS "
            "Requirement 14.5 would not be checked"
        )

    mismatches = []
    for path in sorted(at_base | now):
        if path not in now:
            mismatches.append(f"{path}: deleted (present at {BASE_REV})")
            continue
        if path not in at_base:
            mismatches.append(f"{path}: added (absent at {BASE_REV})")
            continue
        code, base_bytes = _git("show", f"{BASE_REV}:{path}", binary=True)
        if code != 0:
            mismatches.append(f"{path}: could not read {BASE_REV}:{path}")
            continue
        current_bytes = (REPO_ROOT / path).read_bytes()
        base_hash = _digest(base_bytes)
        current_hash = _digest(current_bytes)
        if base_hash != current_hash:
            mismatches.append(f"{path}: {base_hash[:12]} at {BASE_REV} -> {current_hash[:12]} now")

    # Report every mismatch, not just the first.
    assert not mismatches, (
        "Frozen-source violation (CCR Requirements 5.2/5.4, aggregated-space Requirements 4.3/4.4, ACS "
        f"Requirements 14.2/14.3/14.4/14.5): planning/*.py, datasets/*.py, plan.py or models/dino.py "
        f"differ from base revision {BASE_REV}.\n  " + "\n  ".join(mismatches)
    )


def test_ccr_amended_vit_is_frozen_at_its_current_content():
    """ACS Requirement 14.5 for ``models/vit.py``, which cannot be compared against the base revision.

    **Property 16: Frozen sources are byte-identical to the base revision**
    **Validates: Requirements 14.1, 14.5**

    ``models/vit.py`` carries the CCR SDPA scope amendment, so it is both in ``ALLOWED_FILES`` and required
    by ACS to stay put. Those two facts do not fit in one base-revision comparison, so this test pins the
    amended content's digest instead. The amendment's numerical equivalence to the original attention is
    ``tests/test_vit_sdpa_equivalence.py``'s job, not this one's.
    """
    mismatches = []
    for path, expected in sorted(FROZEN_CURRENT_DIGESTS.items()):
        target = REPO_ROOT / path
        if not target.is_file():
            mismatches.append(f"{path}: missing from the working tree")
            continue
        actual = _digest(target.read_bytes())
        if actual != expected:
            mismatches.append(f"{path}: pinned {expected[:12]} -> now {actual[:12]}")

    assert not mismatches, (
        "Frozen-source violation (ACS Requirements 14.1/14.5): a file frozen at its CCR-amended content "
        "changed.\n  " + "\n  ".join(mismatches) + "\n"
        "ACS is confined to models/visual_world_model.py, train.py, conf/train.yaml, custom_resolvers.py, "
        "probe_ccr_curvature.py, summarize_training_log.py, run_ccr_pilot.sh, tests/* and PROGRESS_ACS.md; "
        "models/vit.py is not in that list. If this change is a separately argued amendment rather than ACS "
        "scope creep, update FROZEN_CURRENT_DIGESTS in the same commit and record the reason."
    )
