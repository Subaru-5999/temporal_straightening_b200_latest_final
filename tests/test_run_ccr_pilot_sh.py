"""Task 9.2 - driver contract for ``run_ccr_pilot.sh``'s eval hooks, ``HYDRA_RUN_DIR`` first.

**Why this is no longer optional.** Both bugs task 11.1 hit were in code that had never been
executed (``PROGRESS_AGG.md`` section 5): ``run_dir_override`` emitted its token *unquoted*, which
Hydra's override grammar rejects, and ``plan_agg.py`` never imported ``custom_resolvers``. The CPU
property tests check the objective's algebra thoroughly and never check that the entry point can
start. A contract test on the emitted override token would have caught the first for free, before
any GPU time -- and worse, the suite at the time asserted the *unquoted* form, so it was pinning the
bug in place. This module is that contract test.

What it covers, in the order the hook is exercised:

1. ``PLAN_ENTRY`` and ``SETTINGS`` default to ``plan.py`` and ``both``, no literal ``plan.py``
   token survives in ``run_eval_jobs``, and each ``SETTINGS`` value selects the matching loop;
2. **``HYDRA_RUN_DIR`` unset emits no ``hydra.run.dir`` override at all** -- which is what keeps the
   CCR evaluation path byte-identical to what it was before the hook existed;
3. ``HYDRA_RUN_DIR=agg`` emits the **per-setting** token from
   ``agg_objectives.run_dir_override``, and the open-loop and MPC settings get **different**
   templates under different ``plan_outputs_*`` prefixes. One string for both settings would put
   MPC results in the open-loop tree, where ``aggregate_results.py`` would read them back as
   open-loop cells;
4. every emitted ``hydra.run.dir`` value is **single-quoted**, at both layers that need it -- bash
   (an unquoted ``${...}`` is expanded to empty and truncates the directory) and Hydra's override
   grammar (an unquoted ``}`` is a parse error). The bash-layer scan is the helper from
   ``tests/test_agg_run_dir_separation.py``, imported rather than reimplemented;
5. a caller-supplied ``HYDRA_RUN_DIR`` containing a single quote is **rejected**, not silently
   mangled into a different override;
6. the emitted token is **one shell word**;
7. the ``ps`` pre-flight refusal and the Blackwell/MIG environment recipe are unchanged.

Everything here is source-level except the ``bash`` harness at the bottom, which re-uses the real
function definitions extracted from the driver. The static checks are the ones that must hold
everywhere, because this is authored on a Windows box with no bash, no hydra, no torch and no CUDA;
the harness skips cleanly when ``bash`` cannot run a command.

_Requirements: 9.1, 9.2, 9.3_
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agg_objectives import RUN_DIR_TEMPLATES, run_dir_override  # noqa: E402

# Reused, not reimplemented: task 3.1 already owns the "is this ${...} protected from bash"
# question, and two copies of that scan would drift.
from tests.test_agg_run_dir_separation import _single_quoted_spans  # noqa: E402

DRIVER = _REPO_ROOT / "run_ccr_pilot.sh"
DRIVER_TEXT = DRIVER.read_text(encoding="utf-8", errors="replace")

#: The Hydra config name each eval loop launches, and the ``plan_outputs_*`` prefix it must own.
LOOP_CONFIG_NAMES = {"ol": "plan_gd", "mpc": "plan_gd_mpc"}


# ---------------------------------------------------------------------------
# Source extraction
# ---------------------------------------------------------------------------


def _function_source(name: str) -> str:
    """The full text of shell function ``name``, ``name() {`` through its closing ``}``.

    The driver writes every function as ``name() {`` on one line and closes it with a ``}`` in
    column 0, so a line scan is exact here and needs no shell parser -- which matters, because this
    has to run where there is no shell.
    """
    lines = DRIVER_TEXT.splitlines()
    opener = re.compile(rf"^{re.escape(name)}\(\)\s*\{{\s*$")
    for index, line in enumerate(lines):
        if opener.match(line):
            for end in range(index + 1, len(lines)):
                if lines[end] == "}":
                    return "\n".join(lines[index : end + 1])
            raise AssertionError(f"{name}() in {DRIVER.name} is never closed by a '}}' in column 0")
    raise AssertionError(f"{DRIVER.name} defines no function {name}()")


def _eval_loop_source(setting: str) -> str:
    """The body of one ``run_eval_jobs`` loop, from its ``setting_selected`` guard to the ``fi``."""
    body = _function_source("run_eval_jobs")
    start = body.index(f"if setting_selected {setting}; then")
    end = body.index("\n  fi", start)
    return body[start:end]


# ---------------------------------------------------------------------------
# 1. The two eval hooks and their defaults
# ---------------------------------------------------------------------------


def test_plan_entry_and_settings_default_to_todays_behaviour():
    """Both hooks default to what the launcher already did, so the CCR path is unchanged."""
    assert 'PLAN_ENTRY="${PLAN_ENTRY:-plan.py}"' in DRIVER_TEXT
    assert 'SETTINGS="${SETTINGS:-both}"' in DRIVER_TEXT


def test_no_literal_plan_py_token_survives_in_run_eval_jobs():
    """``PLAN_ENTRY`` has to replace *both* literals, or one setting ignores the hook silently."""
    body = _function_source("run_eval_jobs")
    offenders = [
        line.strip()
        for line in body.splitlines()
        if re.search(r"(?<![\w./-])plan\.py", line) and not line.lstrip().startswith("#")
    ]
    assert not offenders, (
        "run_eval_jobs still names plan.py literally, so that job ignores PLAN_ENTRY and a "
        f"plan_agg.py launch would silently evaluate the spatial-only objective:\n  "
        + "\n  ".join(offenders)
    )
    assert body.count('python "$PLAN_ENTRY" --config-name plan_gd.yaml') == 1
    assert body.count('python "$PLAN_ENTRY" --config-name plan_gd_mpc.yaml') == 1


def test_settings_selects_the_matching_loop():
    """``ol`` / ``mpc`` / ``both``, validated up front, one guard per loop."""
    selector = _function_source("setting_selected")
    assert '[[ "$SETTINGS" == "both" || "$SETTINGS" == "$1" ]]' in selector

    validator = _function_source("validate_eval_hooks")
    assert "ol|mpc|both)" in validator, "an unknown SETTINGS value must be refused, not ignored"

    body = _function_source("run_eval_jobs")
    assert body.count("if setting_selected ol; then") == 1
    assert body.count("if setting_selected mpc; then") == 1
    assert "validate_eval_hooks" in body, "the hooks are validated before any job is launched"
    # The open-loop guard wraps the plan_gd loop and the MPC guard the plan_gd_mpc loop, not the
    # other way round: a swap would run MPC configs under the open-loop selection.
    assert "--config-name plan_gd.yaml" in _eval_loop_source("ol")
    assert "--config-name plan_gd_mpc.yaml" in _eval_loop_source("mpc")


# ---------------------------------------------------------------------------
# 2-3. The HYDRA_RUN_DIR hook: unset means nothing, "agg" means per-setting
# ---------------------------------------------------------------------------


def test_unset_hydra_run_dir_emits_no_override_at_all():
    """The CCR path stays byte-identical: no override, not an empty one, not a default one.

    Asserted on the structure of ``add_run_dir_default``: the emptiness test comes first and its
    branch is a bare ``return 0``, before any ``add_default`` call in the function.
    """
    source = _function_source("add_run_dir_default")
    assert 'value="${HYDRA_RUN_DIR:-}"' in source, (
        "the hook must read HYDRA_RUN_DIR with an empty default, or `set -u` aborts every eval "
        "launch that does not set it"
    )
    empty_test = source.index('if [[ -z "$value" ]]; then')
    assert "add_default" in source, "add_run_dir_default emits nothing at all"
    first_emit = source.index("add_default")
    assert empty_test < first_emit, (
        "add_run_dir_default emits a hydra.run.dir override before testing HYDRA_RUN_DIR for "
        "emptiness, so an unset variable would still change the command line"
    )
    tail = source[empty_test : source.index("fi", empty_test)]
    assert "return 0" in tail and "add_default" not in tail, (
        "the empty-HYDRA_RUN_DIR branch must return without emitting anything; today it is:\n"
        + tail
    )
    # And the driver says so where an operator reads it.
    assert "no override at all" in DRIVER_TEXT


def test_agg_resolves_the_per_setting_template_from_one_source_of_truth():
    """``HYDRA_RUN_DIR=agg`` reads ``agg_objectives.run_dir_override``, and the driver retypes nothing."""
    source = _function_source("add_run_dir_default")
    assert 'if [[ "$value" == "agg" ]]; then' in source
    assert "agg_objectives.run_dir_override(sys.argv[1])" in source, (
        "the agg branch must resolve the template through agg_objectives.run_dir_override, so the "
        "driver and tests/test_agg_run_dir_separation.py read the same strings"
    )
    # The template text itself never appears in the driver: a copy would drift, and a drifted run
    # directory is one aggregate_results.py parses as some other cell.
    for template in RUN_DIR_TEMPLATES.values():
        assert template not in DRIVER_TEXT
    # A resolution failure or an empty token is a hard error, not a silently missing override.
    assert source.count("die") >= 2
    assert 'printed nothing' in source


def test_the_two_loops_ask_for_different_templates():
    """Per-setting, positionally: the open-loop loop passes ``plan_gd``, the MPC loop ``plan_gd_mpc``.

    One string for both settings is the failure being guarded: the MPC leg needs the
    ``plan_outputs_gd_mpc`` prefix, and without it its ``logs.json`` lands in the open-loop tree
    where ``aggregate_results.parse_meta`` reads it back as an open-loop cell.
    """
    for setting, config_name in LOOP_CONFIG_NAMES.items():
        loop = _eval_loop_source(setting)
        assert f"add_run_dir_default {config_name}" in loop, (
            f"the {setting} loop does not pass {config_name!r} to add_run_dir_default"
        )
    # The two config names are not interchangeable: they resolve to different tokens under
    # different prefixes.
    ol_token = run_dir_override("plan_gd")
    mpc_token = run_dir_override("plan_gd_mpc")
    assert ol_token != mpc_token
    assert "plan_outputs_gd/" in ol_token and "plan_outputs_gd_mpc/" in mpc_token
    assert "plan_outputs_gd_mpc" not in ol_token
    # And the open-loop loop must not be the one asking for the MPC template.
    assert "add_run_dir_default plan_gd_mpc" not in _eval_loop_source("ol")
    assert "add_run_dir_default plan_gd\n" not in _eval_loop_source("mpc")


# ---------------------------------------------------------------------------
# 4-6. Quoting, rejection, one word
# ---------------------------------------------------------------------------


def _unprotected_interpolations(line: str):
    """``${`` occurrences in a ``hydra.run.dir=`` token that bash would expand.

    Same question as ``tests/test_agg_run_dir_separation.py`` asks of every driver; the span
    helper is imported from there rather than copied.
    """
    spans = _single_quoted_spans(line)
    positions = []
    for token in re.finditer(r"hydra\.run\.dir=", line):
        tail = line[token.start() :]
        for interpolation in re.finditer(r"\$\{", tail):
            position = token.start() + interpolation.start()
            if not any(begin <= position < end for begin, end in spans):
                positions.append(position)
    return positions


def test_every_emitted_hydra_run_dir_value_is_single_quoted():
    """Two independent layers need the quotes, and this asserts both on the emitted forms.

    bash: an unquoted ``${...}`` is expanded to empty, and the run lands in a truncated directory
    that ``aggregate_results.py`` parses as some other cell. Hydra: its override grammar rejects an
    unquoted ``}`` with a parse error before OmegaConf sees anything -- the bug that made task 11.1
    fail on its first invocation.
    """
    emitters = [
        line
        for line in _function_source("add_run_dir_default").splitlines()
        if "hydra.run.dir=" in line and not line.lstrip().startswith("#")
    ]
    assert emitters, "add_run_dir_default emits no hydra.run.dir token at all"
    for line in emitters:
        assert re.search(r"hydra\.run\.dir='[^']*'", line), (
            f"an emitted hydra.run.dir value is not single-quoted: {line.strip()!r}"
        )
        assert not _unprotected_interpolations(line), (
            f"an emitted hydra.run.dir token carries a ${{...}} bash would expand: {line.strip()!r}"
        )
    # The agg branch's quoting lives in run_dir_override, which emits the token whole.
    for config_name, template in RUN_DIR_TEMPLATES.items():
        assert run_dir_override(config_name) == f"hydra.run.dir='{template}'"


def test_a_caller_supplied_single_quote_is_rejected():
    """It cannot be escaped through bash single quotes or Hydra's quoted-value grammar."""
    source = _function_source("add_run_dir_default")
    assert '[[ "$value" != *"\'"* ]] || die' in source, (
        "a HYDRA_RUN_DIR containing a single quote must be refused: single-quoting it would "
        "terminate the value early and hand Hydra a different override than the caller typed"
    )
    guard = source.index('[[ "$value" != *"\'"* ]]')
    emit = source.index("hydra.run.dir='$value'")
    assert guard < emit, "the single-quote guard runs after the emission it is supposed to guard"

    # The same rule one layer down, on the templates themselves: `run_dir_override` refuses a
    # template it cannot quote rather than emitting a half-quoted override. Driven with a poisoned
    # template, because the real ones are quote-free -- which is what makes the refusal unreachable
    # in production and worth pinning here.
    original = dict(RUN_DIR_TEMPLATES)
    try:
        RUN_DIR_TEMPLATES["plan_gd"] = "plan_outputs_gd/it's_here"
        with pytest.raises(ValueError, match="single quote"):
            run_dir_override("plan_gd")
    finally:
        RUN_DIR_TEMPLATES.clear()
        RUN_DIR_TEMPLATES.update(original)
    assert run_dir_override("plan_gd") == f"hydra.run.dir='{original['plan_gd']}'"


def test_the_emitted_token_is_one_shell_word():
    """``add_default`` appends one array element; whitespace in it would split the command line."""
    for config_name in RUN_DIR_TEMPLATES:
        token = run_dir_override(config_name)
        assert not re.search(r"\s", token), f"{config_name}: {token!r} is more than one word"
    # The driver passes the resolved value as one quoted word on both branches.
    source = _function_source("add_run_dir_default")
    assert 'add_default "$token"' in source
    assert "add_default \"hydra.run.dir='$value'\"" in source


# ---------------------------------------------------------------------------
# 7. The parts of the driver this feature must not have touched
# ---------------------------------------------------------------------------


def test_the_ps_preflight_guard_is_unchanged():
    """Requirements 9.5/9.6: it refuses, it does not warn, and it does not pipe into ``grep -q``."""
    source = _function_source("preflight_or_die")
    assert 'snapshot="$(ps -eo pid,stat,etime,cmd)"' in source
    for pattern in ('grep -qE "$JOB_PATTERN" <<<"$snapshot"', 'grep -qE "$STOPPED_PATTERN" <<<"$snapshot"'):
        assert f"if {pattern}; then" in source, (
            f"the pre-flight no longer matches with `if {pattern}`. The tempting pipeline forms are "
            f"broken: `| head` succeeds on empty input so the `||` branch prints unconditionally, "
            f"and `grep -q`'s early exit turns ps's SIGPIPE status into the test's status under "
            f"pipefail -- i.e. a match can read as 'slice free'."
        )
    assert source.count("REFUSING TO START") == 2
    assert source.count("return 1") == 2, "a detected holder must stop the launch, not warn"
    # The guard has to cover plan_agg.py exactly as it covers plan.py, or a wrapper job does not
    # hold the slice against the next launch.
    assert "(train|plan|probe)[A-Za-z0-9_]*\\.py" in DRIVER_TEXT
    assert re.search(
        r"^JOB_PATTERN='\[p\]ython", DRIVER_TEXT, re.MULTILINE
    ), "the [p] trick keeps a stale grep line from matching itself"


def test_the_blackwell_mig_environment_recipe_is_unchanged():
    """Requirements 9.1-9.4: the four fixes that make a job survive the MIG slice at all."""
    source = _function_source("apply_env")
    for line in (
        "unset CUDA_VISIBLE_DEVICES",                            # 9.2 mujoco-py int() on a MIG UUID
        "export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync",  # 9.1 NVML assert in the allocator
        "export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8",
        "export PLAN_SERIAL_ENV=1",                             # 9.4 no env fork after CUDA init
        "export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl",
        "export D4RL_SUPPRESS_IMPORT_ERROR=1",
    ):
        assert line in source, f"the environment recipe no longer applies: {line}"
    assert 'if [[ -z "${DATASET_DIR:-}" ]]; then' in source, "a guessed dataset root is a different experiment"
    assert 'if [[ "$mode" == "eval" ]]; then' in source, "PLAN_SERIAL_ENV is eval-only"


# ---------------------------------------------------------------------------
# The bash harness: the same assertions against the real function definitions.
# Skipped where bash cannot run, which includes this Windows dev box (the
# WindowsApps `bash.exe` stub exits non-zero with no WSL distribution installed).
# ---------------------------------------------------------------------------


#: Resolved once. ``None`` means "no bash on this box", which is a skip, not a failure.
_BASH_CACHE: "list[str | None]" = []

#: Candidates in preference order. ``PATH`` first, then Git-for-Windows, which is a *real* bash
#: even on a box where ``PATH``'s ``bash`` is the WindowsApps stub that exits 1 with no WSL
#: distribution installed. Trying the candidates rather than trusting ``which`` is what lets the
#: harness actually run on the dev box instead of always skipping there.
_BASH_CANDIDATES = (
    "C:/Program Files/Git/bin/bash.exe",
    "C:/Program Files/Git/usr/bin/bash.exe",
    "/bin/bash",
    "/usr/bin/bash",
)


def _find_bash():
    if _BASH_CACHE:
        return _BASH_CACHE[0]
    candidates = [shutil.which("bash"), *_BASH_CANDIDATES]
    for candidate in candidates:
        if not candidate or not Path(candidate).exists():
            continue
        try:
            probe = subprocess.run(
                [candidate, "-c", "echo harness-ok"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=60,
            )
        except (OSError, subprocess.SubprocessError):  # pragma: no cover - environment specific
            continue
        if probe.returncode == 0 and b"harness-ok" in probe.stdout:
            _BASH_CACHE.append(candidate)
            return candidate
    _BASH_CACHE.append(None)
    return None


def _bash() -> str:
    bash = _find_bash()
    if bash is None:
        pytest.skip(
            "no working bash on this box (on Windows, PATH's bash.exe is the WindowsApps stub and "
            "exits non-zero with no WSL distribution installed); the source-level checks in this "
            "module cover the same contract and always run"
        )
    return bash


def _harness(hydra_run_dir, config_name: str, user_args=()):
    """Run the driver's real ``add_run_dir_default`` and return ``(rc, cmd_words, stderr)``.

    The function definitions are extracted from ``run_ccr_pilot.sh`` and ``eval``-ed, so the
    harness exercises the shipped source rather than a paraphrase of it. Only the collaborators the
    hook needs are supplied: ``die``, ``USER_ARGS``, ``_user_overrides_key``, ``add_default`` (the
    real one) and a ``python`` on ``PATH``.
    """
    bash = _bash()
    # `python` is supplied as a shell FUNCTION rather than through PATH: the hook calls it inside a
    # command substitution, which is a subshell of this same bash and therefore sees the function.
    # No PATH surgery (which is its own mess on Git-for-Windows, where ':' is the separator and
    # 'C:' starts a drive) and no temporary file inside the repo, which would show up as an
    # untracked path and fail tests/test_scope_guard.py.
    interpreter = sys.executable.replace("\\", "/")
    script = "\n".join(
        [
            "set -euo pipefail",
            'die() { echo "ERROR: $*" >&2; exit 2; }',
            'python() { "%s" "$@"; }' % interpreter,
            "CMD=()",
            "USER_ARGS=(%s)" % " ".join(f"'{arg}'" for arg in user_args),
            _function_source("_user_overrides_key"),
            _function_source("add_default"),
            _function_source("add_run_dir_default"),
            f'add_run_dir_default "{config_name}"',
            'printf "%s\\0" ${CMD[@]+"${CMD[@]}"}',
        ]
    )
    env = dict(os.environ)
    env.pop("HYDRA_RUN_DIR", None)
    if hydra_run_dir is not None:
        env["HYDRA_RUN_DIR"] = hydra_run_dir
    completed = subprocess.run(
        [bash, "-c", script],
        cwd=str(_REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        timeout=180,
    )
    raw = completed.stdout.decode("utf-8", "replace")
    words = [word for word in raw.split("\0") if word]
    return completed.returncode, words, completed.stderr.decode("utf-8", "replace")


def test_driver_syntax_is_valid():
    """``bash -n`` on the shipped file: the cheapest possible check, and it never ran."""
    bash = _bash()
    completed = subprocess.run(
        [bash, "-n", DRIVER.name],
        cwd=str(_REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")


@pytest.mark.parametrize("config_name", sorted(RUN_DIR_TEMPLATES))
def test_harness_unset_emits_nothing(config_name):
    rc, words, stderr = _harness(None, config_name)
    assert rc == 0, stderr
    assert words == [], (
        f"HYDRA_RUN_DIR unset emitted {words!r}; the CCR eval path must stay byte-identical, so "
        f"the hook emits no override at all rather than a default one"
    )
    rc, words, stderr = _harness("", config_name)
    assert rc == 0 and words == [], stderr


@pytest.mark.parametrize("config_name", sorted(RUN_DIR_TEMPLATES))
def test_harness_agg_emits_the_per_setting_token_as_one_word(config_name):
    rc, words, stderr = _harness("agg", config_name)
    assert rc == 0, stderr
    assert words == [run_dir_override(config_name)], (
        f"{config_name}: emitted {words!r}, expected exactly one word "
        f"{run_dir_override(config_name)!r}"
    )


def test_harness_agg_separates_the_two_settings():
    _, open_loop, _ = _harness("agg", "plan_gd")
    _, mpc, _ = _harness("agg", "plan_gd_mpc")
    assert open_loop != mpc
    assert "plan_outputs_gd_mpc/" in mpc[0] and "plan_outputs_gd_mpc" not in open_loop[0]


def test_harness_verbatim_value_is_single_quoted():
    rc, words, stderr = _harness("plan_outputs_gd_scratch/${replace_slash:${model_name}}", "plan_gd")
    assert rc == 0, stderr
    assert words == [
        "hydra.run.dir='plan_outputs_gd_scratch/${replace_slash:${model_name}}'"
    ], words


def test_harness_rejects_a_single_quote_in_the_value():
    rc, words, stderr = _harness("plan_outputs_gd/it's_here", "plan_gd")
    assert rc == 2, f"a single-quoted value was accepted: rc={rc} words={words!r}"
    assert "single quote" in stderr
    assert not words


def test_harness_respects_a_caller_supplied_override():
    """Hydra rejects a duplicated key, so ``add_default`` must yield to the caller's own value."""
    rc, words, stderr = _harness(
        "agg", "plan_gd", user_args=["hydra.run.dir=plan_outputs_gd_scratch/mine"]
    )
    assert rc == 0, stderr
    assert words == [], (
        f"the hook emitted {words!r} alongside a caller-supplied hydra.run.dir; Hydra rejects the "
        f"same key twice on one command line"
    )
