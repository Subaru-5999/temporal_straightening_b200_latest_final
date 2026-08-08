"""Task 3.1 - run-directory separation for the aggregated-space sweep.

The failure this module guards is silent. ``aggregate_results.py`` forms one Table-1 cell out of
the directory that holds a ``logs.json``, and every ``plan.py`` run APPENDS one line to that file.
The shipped ``hydra.run.dir`` template carries neither the seed nor the weight, so all seven sweep
arms would land as seven lines in ONE ``logs.json``, and the aggregator would report their mean as
a single number without ever erroring. The sweep curve would be meaningless and nothing would say
so.

So the checks here are, in order:

1. the override templates in :data:`agg_objectives.RUN_DIR_TEMPLATES` resolve to seven **pairwise
   distinct** directories, once per weight in ``(0,) + SWEEP_GRID``, for both settings;
2. the shipped templates, resolved the same way, collapse all seven weights onto **one**
   directory - i.e. the hazard is real and not hypothetical;
3. the override preserves everything the aggregator parses - the ``plan_outputs_gd`` /
   ``plan_outputs_gd_mpc`` prefix, the ``${replace_slash:${model_name}}_gH..._${goal_source}``
   component and the trailing ``obj${objective.mode}_init${planner.sub_planner.sample_type}``
   token - checked by running the aggregator's own ``parse_meta`` over the resolved paths, and the
   two settings resolve under **different** prefixes so the MPC leg cannot land in the open-loop
   tree;
4. the templates are **single-quoted** wherever a shell driver passes them, so ``${...}`` reaches
   Hydra instead of being expanded (to nothing) by bash;
5. the shipped-template mirror in ``agg_objectives.py`` still matches the ``hydra.run.dir`` text
   in ``conf/plan_gd.yaml`` / ``conf/plan_gd_mpc.yaml``, so the override cannot be a
   one-component substitution of a template the configs no longer use.

Checks 1-3 resolve through the real Hydra ``compose`` API and are skipped where ``hydra`` /
``omegaconf`` are absent (the Windows dev environment). Checks 4 and 5 are pure text and run
everywhere - which is deliberate, since drift in either is exactly the kind of thing that should
not wait for a pod.

Validates: Requirements 2.7, 6.7, 7.3
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import agg_objectives  # noqa: E402
from agg_objectives import (  # noqa: E402
    AGG_WEIGHT_RUN_DIR_COMPONENT,
    RUN_DIR_TEMPLATES,
    SHIPPED_RUN_DIR_TEMPLATES,
    SWEEP_GRID,
    run_dir_override,
)

CONF_DIR = _REPO_ROOT / "conf"

#: The two settings of the Evaluation_Protocol, keyed by Hydra config name.
CONFIG_YAMLS = {
    "plan_gd": CONF_DIR / "plan_gd.yaml",
    "plan_gd_mpc": CONF_DIR / "plan_gd_mpc.yaml",
}

#: Requirement 8.2 / 8.3: the objective mode each setting is measured with.
PROTOCOL_MODES = {"plan_gd": "last", "plan_gd_mpc": "staged"}

#: ``plan_outputs_<planner>`` is how ``aggregate_results.parse_meta`` recovers planner and setting.
EXPECTED_PREFIXES = {"plan_gd": "plan_outputs_gd", "plan_gd_mpc": "plan_outputs_gd_mpc"}

#: The Baseline_Arm weight plus the six Sweep_Grid values: seven arms, seven directories.
WEIGHTS = (0,) + tuple(SWEEP_GRID)

#: Target_Cell identity. `model_name` carries no slash, so `replace_slash` is a no-op on it; that
#: is fine, the point of resolving it is that the component survives the override at all.
MODEL_NAME = "pusht_aggmlpcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05"
CKPT_BASE_PATH = "./checkpoints/test"


# ---------------------------------------------------------------------------
# Hydra composition (checks 1-3)
# ---------------------------------------------------------------------------


def _require_hydra():
    """Skip unless the real Hydra stack is importable, and register ``replace_slash``."""
    pytest.importorskip("hydra", reason="hydra is not installed in this environment")
    pytest.importorskip("omegaconf", reason="omegaconf is not installed in this environment")
    # Importing this module is what registers `replace_slash` with OmegaConf. Without it the
    # `${replace_slash:${model_name}}` component is unresolvable and every resolution below fails.
    import custom_resolvers  # noqa: F401


def _compose(config_name: str, overrides: list):
    """Compose ``config_name`` out of ``conf/`` with the Hydra node included.

    ``conf/plan_gd.yaml`` overrides ``hydra/launcher`` to ``submitit_local`` in its defaults list.
    If that plugin is not installed, composition fails for a reason that has nothing to do with
    ``hydra.run.dir``, so the launcher is swapped for ``basic`` on retry. The run-directory
    expression is untouched either way.
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    try:
        ctx = initialize_config_dir(config_dir=str(CONF_DIR), version_base=None)
    except TypeError:  # hydra < 1.2 has no version_base
        ctx = initialize_config_dir(config_dir=str(CONF_DIR))

    with ctx:
        try:
            return compose(config_name=config_name, overrides=overrides, return_hydra_config=True)
        except Exception as exc:  # noqa: BLE001 - re-raised unless it is the launcher plugin
            if "submitit" not in str(exc).lower():
                raise
            return compose(
                config_name=config_name,
                overrides=["hydra/launcher=basic", *overrides],
                return_hydra_config=True,
            )


def _resolve_run_dir(config_name: str, weight, template: str | None) -> str:
    """Resolve ``template`` (or the shipped expression, when ``None``) at ``weight``.

    ``+agg_weight`` is added for both forms, exactly as ``plan_agg.py`` receives it, so the shipped
    and override resolutions differ in nothing but the template text.
    """
    from omegaconf import open_dict

    overrides = [
        f"model_name={MODEL_NAME}",
        f"ckpt_base_path={CKPT_BASE_PATH}",
        f"objective.mode={PROTOCOL_MODES[config_name]}",
        f"+agg_weight={weight}",
    ]
    cfg = _compose(config_name, overrides)
    if template is not None:
        # Set in-memory rather than through the override grammar: the template is full of
        # `${...}` and `:` and has no business being parsed by the override parser here.
        with open_dict(cfg):
            cfg.hydra.run.dir = template
    return str(cfg.hydra.run.dir)  # attribute access resolves the interpolations


@pytest.mark.parametrize("config_name", sorted(CONFIG_YAMLS))
def test_override_templates_give_every_weight_its_own_directory(config_name):
    """Check 1: the seven arms resolve to seven pairwise-distinct directories."""
    _require_hydra()
    resolved = {w: _resolve_run_dir(config_name, w, RUN_DIR_TEMPLATES[config_name]) for w in WEIGHTS}

    assert len(WEIGHTS) == 7, f"expected the Baseline_Arm plus 6 Sweep_Grid values, got {WEIGHTS}"
    distinct = set(resolved.values())
    assert len(distinct) == len(WEIGHTS), (
        f"{config_name}: the {len(WEIGHTS)} sweep arms resolved to only {len(distinct)} "
        f"directories, so at least two arms would append their line to one logs.json and "
        f"aggregate_results.py would average them into a single cell. Resolved: {resolved}"
    )
    for weight, run_dir in resolved.items():
        assert f"aggw{weight}" in run_dir, (
            f"{config_name}: agg_weight={weight!r} resolved to {run_dir!r}, which carries no "
            f"'aggw{weight}' component; the weight is what separates the arms."
        )


@pytest.mark.parametrize("config_name", sorted(CONFIG_YAMLS))
def test_shipped_templates_collapse_every_weight_onto_one_directory(config_name):
    """Check 2: the guarded failure is real - the shipped expression ignores the weight."""
    _require_hydra()
    resolved = {w: _resolve_run_dir(config_name, w, None) for w in WEIGHTS}

    distinct = set(resolved.values())
    assert len(distinct) == 1, (
        f"{config_name}: the shipped hydra.run.dir resolved to {len(distinct)} directories across "
        f"the sweep weights, so it now separates them by itself and RUN_DIR_TEMPLATES may be "
        f"redundant. Re-read this test before deleting the override. Resolved: {resolved}"
    )
    collapsed = distinct.pop()
    assert "aggw" not in collapsed, (
        f"{config_name}: the shipped template unexpectedly carries an 'aggw' component "
        f"({collapsed!r}); the two templates are no longer distinguishable."
    )


@pytest.mark.parametrize("config_name", sorted(CONFIG_YAMLS))
def test_override_preserves_everything_the_aggregator_parses(config_name):
    """Check 3a: ``aggregate_results.parse_meta`` reads the override exactly as it reads a
    ``plan.py`` run directory."""
    _require_hydra()
    import aggregate_results

    mode = PROTOCOL_MODES[config_name]
    for weight in WEIGHTS:
        run_dir = _resolve_run_dir(config_name, weight, RUN_DIR_TEMPLATES[config_name])
        meta = aggregate_results.parse_meta(f"{run_dir}/logs.json")

        assert run_dir.startswith(EXPECTED_PREFIXES[config_name] + "/"), (
            f"{config_name}: resolved {run_dir!r}, which does not sit under "
            f"{EXPECTED_PREFIXES[config_name]!r}; parse_meta recovers the planner and the setting "
            f"from that prefix alone."
        )
        assert meta["planner"] == EXPECTED_PREFIXES[config_name].replace("plan_outputs_", "")
        assert meta["setting"] == ("open-loop" if config_name == "plan_gd" else "MPC")

        # The `${replace_slash:${model_name}}_gH${goal_H}_${goal_source}` component: env and
        # curvature flavour are read out of it.
        assert f"/{MODEL_NAME}_gH25_dset/" in run_dir, (
            f"{config_name}: resolved {run_dir!r}, which lost the "
            f"'{MODEL_NAME}_gH25_dset' component; parse_meta reads env and curvature from it."
        )
        assert meta["env"] == "pusht"
        assert meta["curv"] == "curv(agg)"

        # The trailing `obj<mode>_init<sample_type>` token, which is how the mode is recovered.
        assert run_dir.endswith(f"_obj{mode}_initzero"), (
            f"{config_name}: resolved {run_dir!r}, which does not end in "
            f"'_obj{mode}_initzero'; parse_meta matches r'obj([a-z]+)_init' for the mode."
        )
        assert meta["mode"] == mode
        assert re.search(r"obj([a-z]+)_init", run_dir).group(1) == mode


def test_the_two_settings_resolve_under_different_prefixes():
    """Check 3b: the MPC leg cannot land in the open-loop tree."""
    _require_hydra()
    open_loop = {_resolve_run_dir("plan_gd", w, RUN_DIR_TEMPLATES["plan_gd"]) for w in WEIGHTS}
    mpc = {_resolve_run_dir("plan_gd_mpc", w, RUN_DIR_TEMPLATES["plan_gd_mpc"]) for w in WEIGHTS}

    assert not (open_loop & mpc), (
        f"the two settings share {sorted(open_loop & mpc)}; one string for both settings is "
        f"wrong, the MPC leg needs the plan_outputs_gd_mpc prefix."
    )
    for run_dir in mpc:
        assert not run_dir.startswith("plan_outputs_gd/"), (
            f"MPC resolved to {run_dir!r}, inside the open-loop tree; parse_meta would read it "
            f"back as an open-loop cell."
        )
    assert len(open_loop | mpc) == 2 * len(WEIGHTS)


# ---------------------------------------------------------------------------
# Check 4: single-quoting in the shell drivers (pure text, always runs)
# ---------------------------------------------------------------------------

SHELL_DRIVERS = sorted(_REPO_ROOT.glob("*.sh"))


def _single_quoted_spans(line: str):
    """Half-open spans of ``line`` that sit inside single quotes.

    Bash single quotes take no escapes, so pairing them off left to right is exact for the one
    thing this needs to decide: whether a given ``${`` is protected from expansion.
    """
    spans = []
    start = None
    for index, char in enumerate(line):
        if char != "'":
            continue
        if start is None:
            start = index + 1
        else:
            spans.append((start, index))
            start = None
    return spans


def _interpolations_are_single_quoted(line: str) -> list:
    """Positions of every ``${`` in a ``hydra.run.dir=`` token that bash would expand."""
    unprotected = []
    spans = _single_quoted_spans(line)
    for token in re.finditer(r"hydra\.run\.dir=", line):
        tail = line[token.start() :]
        for interpolation in re.finditer(r"\$\{", tail):
            position = token.start() + interpolation.start()
            if not any(begin <= position < end for begin, end in spans):
                unprotected.append(position)
    return unprotected


@pytest.mark.parametrize("driver", SHELL_DRIVERS, ids=lambda p: p.name)
def test_run_dir_overrides_are_single_quoted_in_shell_drivers(driver):
    """Check 4: an unquoted ``${...}`` is expanded by bash, to nothing, before Hydra sees it.

    That failure is silent in the worst way: the override still arrives, with its interpolations
    blanked, and the run lands in a truncated directory that ``aggregate_results.py`` happily
    parses as some other cell.
    """
    text = driver.read_text(encoding="utf-8", errors="replace")
    if "hydra.run.dir=" not in text:
        # Nothing to check in this driver. Deliberately not a skip: the assertion below is the
        # contract for whichever driver grows the override next (task 9.1), and a file-level skip
        # here would read as "unverified" rather than "no occurrences".
        return

    offenders = []
    for number, line in enumerate(text.splitlines(), start=1):
        if "hydra.run.dir=" not in line:
            continue
        for position in _interpolations_are_single_quoted(line):
            offenders.append(f"{driver.name}:{number}:{position} in {line.strip()!r}")

    assert not offenders, (
        "a hydra.run.dir override carries a ${...} interpolation that is not inside single "
        "quotes, so bash expands it (to empty) before Hydra ever sees it:\n  "
        + "\n  ".join(offenders)
    )


def test_templates_require_single_quoting_and_survive_it():
    """Check 4, the other half: the quoting rule is non-vacuous, and quoting is sufficient.

    If the templates ever stop containing ``${``, the test above becomes trivially true; if they
    ever grow a ``'``, single-quoting them in a driver would break the command line instead of
    protecting it.
    """
    for config_name, template in RUN_DIR_TEMPLATES.items():
        assert "${" in template, (
            f"{config_name}: the override template {template!r} carries no interpolation, so the "
            f"single-quoting contract the drivers rely on has become vacuous."
        )
        assert "'" not in template, (
            f"{config_name}: the override template {template!r} contains a single quote, which "
            f"cannot be passed through bash single quotes."
        )
        token = run_dir_override(config_name)
        # The value is single-quoted for HYDRA's override grammar, which is a separate layer from
        # the bash quoting the test above checks. This assertion previously read
        # ``token == f"hydra.run.dir={template}"`` -- i.e. it encoded the unquoted form, and so
        # asserted the bug that made task 11.1 fail on its first ever run with this hook:
        # Hydra's ANTLR parser rejects an unquoted '}' with
        # "mismatched input '}' expecting <EOF>" before OmegaConf ever sees the value.
        assert token == f"hydra.run.dir='{template}'"
        assert "\n" not in token and " " not in token, (
            f"{config_name}: the override token {token!r} contains whitespace and would split "
            f"into two shell words."
        )


# ---------------------------------------------------------------------------
# Check 5: the shipped-template mirror still matches the yaml (pure text, always runs)
# ---------------------------------------------------------------------------


def _yaml_hydra_run_dir(path: Path) -> str:
    """The raw ``hydra.run.dir`` string from ``path``, read as text.

    Read with a line scan rather than through OmegaConf on purpose: this check has to run in an
    environment without ``omegaconf``, and it is about the literal text of the template.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    in_hydra = False
    in_run = False
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        if indent == 0:
            in_hydra = stripped.startswith("hydra:")
            in_run = False
            continue
        if not in_hydra:
            continue
        if indent == 2:
            in_run = stripped.startswith("run:")
            continue
        if in_run and stripped.startswith("dir:"):
            return stripped[len("dir:") :].strip()
    raise AssertionError(f"{path} has no hydra.run.dir entry")


@pytest.mark.parametrize("config_name", sorted(CONFIG_YAMLS))
def test_shipped_template_constant_matches_the_yaml(config_name):
    """Check 5: ``SHIPPED_RUN_DIR_TEMPLATES`` is a mirror, and mirrors go stale."""
    from_yaml = _yaml_hydra_run_dir(CONFIG_YAMLS[config_name])
    from_constant = SHIPPED_RUN_DIR_TEMPLATES[config_name]

    assert from_constant == from_yaml, (
        f"agg_objectives.SHIPPED_RUN_DIR_TEMPLATES[{config_name!r}] no longer matches the "
        f"hydra.run.dir in {CONFIG_YAMLS[config_name].name}. The override is built by substituting "
        f"one component of this string, so a stale mirror silently produces a run directory the "
        f"aggregator parses as something else.\n  yaml:     {from_yaml}\n  constant: {from_constant}"
    )


@pytest.mark.parametrize("config_name", sorted(CONFIG_YAMLS))
def test_override_differs_from_the_shipped_template_in_one_component(config_name):
    """The override is a one-component substitution: ``${ckpt_base_path}`` -> ``aggw${agg_weight}``.

    Asserted at the text level so it holds without Hydra, and so the compose-based distinctness
    check above cannot be satisfied by an override that quietly dropped something else.
    """
    shipped = SHIPPED_RUN_DIR_TEMPLATES[config_name]
    override = RUN_DIR_TEMPLATES[config_name]

    assert "${ckpt_base_path}" in shipped
    assert override == shipped.replace("${ckpt_base_path}", AGG_WEIGHT_RUN_DIR_COMPONENT)
    assert "${agg_weight}" in override, (
        f"{config_name}: the override template carries no ${{agg_weight}}, so every sweep arm "
        f"would resolve to the same directory."
    )
    assert "${ckpt_base_path}" not in override, (
        f"{config_name}: the override still interpolates ${{ckpt_base_path}}, which is the "
        f"component the weight replaces."
    )
    # Everything else the aggregator parses is untouched text, so assert it literally.
    assert override.startswith(EXPECTED_PREFIXES[config_name] + "/")
    assert "${replace_slash:${model_name}}_gH${goal_H}_${goal_source}/" in override
    assert override.endswith("_obj${objective.mode}_init${planner.sub_planner.sample_type}")


def test_run_dir_override_parses_under_hydra_grammar():
    """The check that was missing, and the reason task 11.1 failed on its first invocation.

    Every other test in this module reasons about the override token as *text* — is it one shell
    word, are its interpolations protected from bash. None of them handed the token to Hydra, and
    Hydra parses an override's right-hand side with its own ANTLR grammar before OmegaConf sees
    anything. That grammar rejects an unquoted ``}``::

        hydra.run.dir=plan_outputs_gd/${replace_slash:${model_name}}_...
        -> OverrideParseException: mismatched input '}' expecting <EOF>

    So this asserts the real contract on the real parser: the emitted token parses, and its value
    comes back as the template **verbatim**, with the ``${...}`` intact for OmegaConf to resolve at
    composition time. Both halves matter — a quoting scheme that parsed but mangled the
    interpolations would resolve to a different directory, which ``aggregate_results.py`` would
    read as some other cell.

    The negative control is asserted too, so the test cannot pass by accident on a Hydra whose
    grammar has been relaxed: the unquoted form must still raise.
    """
    parser = pytest.importorskip(
        "hydra.core.override_parser.overrides_parser",
        reason="hydra is not installed in this environment",
    ).OverridesParser.create()

    for config_name, template in RUN_DIR_TEMPLATES.items():
        token = run_dir_override(config_name)
        parsed = parser.parse_overrides([token])
        assert len(parsed) == 1, f"{config_name}: {token!r} parsed into {len(parsed)} overrides"
        assert parsed[0].key_or_group == "hydra.run.dir"
        assert parsed[0].value() == template, (
            f"{config_name}: Hydra parsed the override but the value changed.\n"
            f"  emitted  {token!r}\n"
            f"  parsed   {parsed[0].value()!r}\n"
            f"  expected {template!r}\n"
            f"An altered value resolves to a different run directory, which "
            f"aggregate_results.py reads as a different cell."
        )

        with pytest.raises(Exception) as excinfo:
            parser.parse_overrides([f"hydra.run.dir={template}"])
        assert "mismatched input" in str(excinfo.value) or "OverrideParseException" in type(
            excinfo.value
        ).__name__, (
            f"{config_name}: the UNQUOTED override no longer fails to parse. If Hydra's grammar "
            f"has been relaxed, the quoting in run_dir_override() is no longer load-bearing and "
            f"this test's premise should be re-derived rather than deleted."
        )


def test_run_dir_override_rejects_an_unknown_config_name():
    """One source of truth: a typo'd config name must fail loudly, not silently pick a default."""
    with pytest.raises(ValueError) as excinfo:
        run_dir_override("plan_cem")
    message = str(excinfo.value)
    assert "plan_cem" in message
    assert "plan_gd" in message and "plan_gd_mpc" in message
    assert set(RUN_DIR_TEMPLATES) == set(CONFIG_YAMLS) == set(agg_objectives.SHIPPED_RUN_DIR_TEMPLATES)
