"""Task 11.3 - the long-horizon Evaluation_Protocol column, and the gate it must not weaken.

``resolve_protocol`` aborts on any deviation from the expected column (Requirement 8.7). Both
shipped columns pinned ``sub_planner.horizon 25`` and ``n_taken_actions`` 25/5, so task 11.4's
Positive_Control at ``goal_H=50`` *had* to deviate and would have aborted before loading anything.
Task 11.3 adds a second column per setting and selects on the ``(config_name, goal_H)`` pair.

Two things have to be true at once, and this module asserts both rather than one:

1. **the long column exists and is exactly task 11.4's reading (a)** -- ``goal_H 50``,
   ``sub_planner.horizon 50``, ``n_taken_actions`` 50 open-loop / 5 MPC, everything else the short
   column's value;
2. **the short columns are not weakened.** They are what protects the reported result
   (Requirement 7.2). Every value they pinned before task 11.3 is pinned to the same value after
   it, asserted against a literal copy of the pre-11.3 table rather than by reading the diff, and
   ``goal_H`` is now pinned as well -- which *strengthens* the gate, because a 50-step run used to
   satisfy the short columns on ``sub_planner.horizon`` alone.

Plus the two contracts the column selection rests on: ``goal_H`` is resolved out of the
configuration **before** the table is chosen, and a horizon no column covers aborts naming the
horizon rather than falling into the long column and being reported as three field mismatches.

Everything here runs on plain ``dict`` configs with no hydra, no omegaconf and no torch, which is
the importability contract ``plan_agg``'s docstring makes; the last test enforces that contract in
a subprocess that cannot import any of the three.

Validates: Requirements 8.1, 8.4, 8.6, 8.7
"""

from __future__ import annotations

import copy
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import plan_agg  # noqa: E402
from plan_agg import (  # noqa: E402
    FRAMESKIP,
    GOAL_H_FIELD,
    HORIZON_FIELDS,
    HORIZON_REGIMES,
    LONG_GOAL_H,
    PROTOCOL_EXPECTED,
    PROTOCOL_EXPECTED_SOURCE,
    PROTOCOL_FIELDS,
    SHORT_GOAL_H,
    ProtocolError,
    expected_table,
    horizon_regime,
    resolve_protocol,
)

SETTINGS = ("plan_gd", "plan_gd_mpc")

#: The two short-horizon columns **as they stood before task 11.3**, copied out of the previous
#: revision of ``plan_agg.py`` and pinned here as literals. This is the whole "do not weaken the
#: gate" check: it compares against text that cannot be edited by the same change that edits the
#: tables, so a value quietly relaxed while adding the long column fails here.
PRE_11_3_SHORT_COLUMNS = {
    "plan_gd": {
        "n_evals": 50,
        "objective.mode": "last",
        "objective.alpha": 1,
        "planner.max_iter": 1,
        "planner.n_taken_actions": 25,
        "planner.sub_planner.horizon": 25,
        "planner.sub_planner.lr": 0.1,
        "planner.sub_planner.sample_type": "zero",
        "planner.sub_planner.action_noise": 0,
        "planner.sub_planner.opt_steps": 100,
    },
    "plan_gd_mpc": {
        "n_evals": 50,
        "objective.mode": "staged",
        "objective.alpha": 1,
        "planner.max_iter": 20,
        "planner.n_taken_actions": 5,
        "planner.sub_planner.horizon": 25,
        "planner.sub_planner.lr": 0.1,
        "planner.sub_planner.sample_type": "zero",
        "planner.sub_planner.action_noise": 0,
        "planner.sub_planner.opt_steps": 100,
    },
}

#: Task 11.4's reading (a), written out per setting rather than derived from the module, so the
#: test states the protocol instead of restating whatever the module happens to hold.
EXPECTED_LONG_COLUMNS = {
    "plan_gd": {
        "goal_H": 50,
        "n_evals": 50,
        "objective.mode": "last",
        "objective.alpha": 1,
        "planner.max_iter": 1,
        "planner.n_taken_actions": 50,
        "planner.sub_planner.horizon": 50,
        "planner.sub_planner.lr": 0.1,
        "planner.sub_planner.sample_type": "zero",
        "planner.sub_planner.action_noise": 0,
        "planner.sub_planner.opt_steps": 100,
    },
    "plan_gd_mpc": {
        "goal_H": 50,
        "n_evals": 50,
        "objective.mode": "staged",
        "objective.alpha": 1,
        "planner.max_iter": 20,
        "planner.n_taken_actions": 5,
        "planner.sub_planner.horizon": 50,
        "planner.sub_planner.lr": 0.1,
        "planner.sub_planner.sample_type": "zero",
        "planner.sub_planner.action_noise": 0,
        "planner.sub_planner.opt_steps": 100,
    },
}


def _cfg(config_name: str, goal_H: int) -> dict:
    """A plain-``dict`` config that satisfies the expected column for ``(config_name, goal_H)``.

    Built from the table itself, by exploding the dotted field paths into nested dicts, so the
    fixture cannot drift from the column it is supposed to satisfy. It is a ``dict``, not an
    ``omegaconf`` node, which is the point: the protocol layer takes plain mappings.
    """
    cfg: dict = {}
    for path, value in PROTOCOL_EXPECTED[(config_name, goal_H)].items():
        node = cfg
        parts = path.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return cfg


# ---------------------------------------------------------------------------
# 1. The short-horizon gate is not weakened
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("config_name", SETTINGS)
def test_short_columns_still_pin_every_pre_task_value(config_name):
    """Requirement 7.2's protection: no short-horizon value moved, and none was dropped."""
    column = PROTOCOL_EXPECTED[(config_name, SHORT_GOAL_H)]
    before = PRE_11_3_SHORT_COLUMNS[config_name]

    missing = sorted(field for field in before if field not in column)
    assert not missing, (
        f"{config_name}: the short-horizon column no longer pins {missing}. The reported result is "
        f"the short-horizon confirmation run (Requirement 7.2), and task 11.3 must not be able to "
        f"weaken the gate that protects it."
    )
    changed = {
        field: (want, column[field])
        for field, want in before.items()
        if column[field] != want or type(column[field]) is not type(want)
    }
    assert not changed, (
        f"{config_name}: the short-horizon column changed a value that was pinned before task "
        f"11.3 (field: (was, now)): {changed}. The long-horizon column is additive; adding it must "
        f"not move a short-horizon number."
    )


@pytest.mark.parametrize("config_name", SETTINGS)
def test_goal_h_is_now_pinned_which_strengthens_the_short_gate(config_name):
    """``goal_H`` was unpinned before task 11.3, so a 50-step run passed the short columns.

    It satisfied ``sub_planner.horizon 25`` and nothing looked at ``goal_H`` at all. Pinning it is
    therefore strictly stronger, and this test states the strengthening as a behaviour rather than
    as a comment: a config that is short-horizon in every other field but carries ``goal_H 50``
    used to pass and must now abort.
    """
    assert GOAL_H_FIELD in PROTOCOL_FIELDS
    assert GOAL_H_FIELD not in PRE_11_3_SHORT_COLUMNS[config_name], (
        "the premise of this test is that goal_H was unpinned before task 11.3"
    )
    assert PROTOCOL_EXPECTED[(config_name, SHORT_GOAL_H)][GOAL_H_FIELD] == SHORT_GOAL_H
    assert PROTOCOL_EXPECTED[(config_name, LONG_GOAL_H)][GOAL_H_FIELD] == LONG_GOAL_H

    smuggled = copy.deepcopy(_cfg(config_name, SHORT_GOAL_H))
    smuggled[GOAL_H_FIELD] = LONG_GOAL_H
    with pytest.raises(ProtocolError) as excinfo:
        resolve_protocol(config_name, smuggled)
    message = str(excinfo.value)
    # It is now checked against the LONG column, and the horizon fields are what deviate.
    assert "planner.sub_planner.horizon" in message
    assert "long horizon" in message


def test_goal_h_leads_the_manifest_field_order():
    """``goal_H`` selects the column, so it is resolved and recorded first."""
    assert PROTOCOL_FIELDS[0] == GOAL_H_FIELD
    assert len(set(PROTOCOL_FIELDS)) == len(PROTOCOL_FIELDS)


# ---------------------------------------------------------------------------
# 2. The long-horizon column is task 11.4's reading (a)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("config_name", SETTINGS)
def test_long_column_is_reading_a(config_name):
    """``goal_H 50``, subplanner horizon 50, executed actions 50 open-loop / 5 MPC."""
    column = PROTOCOL_EXPECTED[(config_name, LONG_GOAL_H)]
    assert column == EXPECTED_LONG_COLUMNS[config_name], (
        f"{config_name}: the long-horizon column is not task 11.4's recorded reading (a).\n"
        f"  expected {EXPECTED_LONG_COLUMNS[config_name]}\n"
        f"  actual   {column}"
    )


@pytest.mark.parametrize("config_name", SETTINGS)
def test_the_two_regimes_differ_only_in_the_horizon(config_name):
    """Everything outside the horizon fields is identical, so the control isolates the horizon."""
    short = PROTOCOL_EXPECTED[(config_name, SHORT_GOAL_H)]
    long = PROTOCOL_EXPECTED[(config_name, LONG_GOAL_H)]
    differing = sorted(field for field in PROTOCOL_FIELDS if short[field] != long[field])
    unexpected = [field for field in differing if field not in HORIZON_FIELDS]
    assert not unexpected, (
        f"{config_name}: the long-horizon column differs from the short one in {unexpected}, "
        f"which are not horizon fields. The Positive_Control varies the horizon and nothing else, "
        f"so a difference here would confound it."
    )
    # n_evals 50, alpha 1, mode, lr 0.1, opt_steps 100, sample_type zero, action_noise 0 all held.
    for field in ("n_evals", "objective.alpha", "objective.mode",
                  "planner.sub_planner.lr", "planner.sub_planner.sample_type",
                  "planner.sub_planner.action_noise", "planner.sub_planner.opt_steps"):
        assert short[field] == long[field]


def test_mpc_keeps_five_executed_actions_at_both_horizons():
    """The appendix footnotes 5 executed actions for MPC independently of the horizon."""
    for goal_h in (SHORT_GOAL_H, LONG_GOAL_H):
        assert PROTOCOL_EXPECTED[("plan_gd_mpc", goal_h)]["planner.n_taken_actions"] == 5
    # Open-loop is the one that scales: executed actions track the horizon (reading (a)).
    assert PROTOCOL_EXPECTED[("plan_gd", SHORT_GOAL_H)]["planner.n_taken_actions"] == SHORT_GOAL_H
    assert PROTOCOL_EXPECTED[("plan_gd", LONG_GOAL_H)]["planner.n_taken_actions"] == LONG_GOAL_H


@pytest.mark.parametrize("key", sorted(PROTOCOL_EXPECTED))
def test_every_pinned_horizon_is_divisible_by_frameskip(key):
    """``plan.py`` integer-divides these three fields by ``frameskip``, so a non-multiple truncates.

    ``PlanEvaluator.__init__`` does ``goal_H // frameskip``, ``n_taken_actions // frameskip`` and
    ``sub_planner.horizon // frameskip`` and rejects nothing, so pinning e.g. ``horizon 26`` would
    pin a number the planner never runs. ``plan_agg._check_expected_tables()`` enforces this at
    import; this test is what makes the enforcement visible in the suite.
    """
    assert FRAMESKIP == 5
    for field in HORIZON_FIELDS:
        value = PROTOCOL_EXPECTED[key][field]
        assert isinstance(value, int) and not isinstance(value, bool)
        assert value > 0 and value % FRAMESKIP == 0, (
            f"{key}: {field} is {value!r}, not a positive multiple of frameskip {FRAMESKIP}"
        )


def test_the_table_self_check_is_not_vacuous():
    """``_check_expected_tables`` must actually reject a bad column, not just run."""
    original = dict(PROTOCOL_EXPECTED[("plan_gd", LONG_GOAL_H)])
    try:
        PROTOCOL_EXPECTED[("plan_gd", LONG_GOAL_H)] = dict(
            original, **{"planner.sub_planner.horizon": 51}
        )
        with pytest.raises(RuntimeError, match="frameskip"):
            plan_agg._check_expected_tables()
    finally:
        PROTOCOL_EXPECTED[("plan_gd", LONG_GOAL_H)] = original
    plan_agg._check_expected_tables()  # and it passes again on the real tables


# ---------------------------------------------------------------------------
# 3. Column selection: explicit, on the pair, goal_H resolved first
# ---------------------------------------------------------------------------


def test_horizon_regime_is_a_lookup_over_a_named_domain():
    assert horizon_regime(SHORT_GOAL_H) == ("short", SHORT_GOAL_H)
    assert horizon_regime(LONG_GOAL_H) == ("long", LONG_GOAL_H)
    assert horizon_regime(50.0) == ("long", LONG_GOAL_H)  # a float that is integral
    assert HORIZON_REGIMES == {SHORT_GOAL_H: "short", LONG_GOAL_H: "long"}


@pytest.mark.parametrize("bad", [30, 0, -25, 25.5, "25", None, True, False, plan_agg.MISSING])
def test_an_uncovered_horizon_aborts_naming_the_horizon(bad):
    """Not a silent fallback into the long column, and not three field mismatches."""
    with pytest.raises(ProtocolError) as excinfo:
        horizon_regime(bad)
    message = str(excinfo.value)
    assert GOAL_H_FIELD in message
    assert "25" in message and "50" in message
    assert "Nothing has been loaded" in message


@pytest.mark.parametrize("config_name", SETTINGS)
@pytest.mark.parametrize("goal_h", [SHORT_GOAL_H, LONG_GOAL_H])
def test_a_conforming_config_resolves_in_both_regimes(config_name, goal_h):
    record = resolve_protocol(config_name, _cfg(config_name, goal_h))
    assert record.ok, record.message()
    assert record.goal_H == goal_h
    assert record.horizon_regime == HORIZON_REGIMES[goal_h]
    assert record.resolved[GOAL_H_FIELD] == goal_h
    assert record.opt_steps == 100
    assert record.expected == PROTOCOL_EXPECTED[(config_name, goal_h)]
    assert record.expected_source == PROTOCOL_EXPECTED_SOURCE[(config_name, goal_h)]


@pytest.mark.parametrize("config_name", SETTINGS)
def test_the_long_config_would_have_aborted_against_the_short_column(config_name):
    """The reason task 11.3 exists: forced through the short column, the control cannot run."""
    long_cfg = _cfg(config_name, LONG_GOAL_H)
    short_column = PROTOCOL_EXPECTED[(config_name, SHORT_GOAL_H)]
    deviating = [
        field
        for field in PROTOCOL_FIELDS
        if short_column[field] != PROTOCOL_EXPECTED[(config_name, LONG_GOAL_H)][field]
    ]
    assert deviating, (
        f"{config_name}: the long-horizon config satisfies the short column, so the third column "
        f"would not have been needed and this test's premise is wrong."
    )
    # And with the pair-keyed selection it runs.
    assert resolve_protocol(config_name, long_cfg).ok


def test_goal_h_is_resolved_before_the_table_is_chosen():
    """The ordering Requirement: ``goal_H`` first, then the column.

    Proved by a configuration that is wrong in *both* keys. If the table were selected on the
    config name first, the abort would name the config name; because ``goal_H`` is resolved first,
    it names the horizon. The ordering matters because the whole point of the pair key is that a
    long-horizon run is never measured against a short-horizon column.
    """
    cfg = _cfg("plan_gd", SHORT_GOAL_H)
    cfg[GOAL_H_FIELD] = 30
    with pytest.raises(ProtocolError) as excinfo:
        resolve_protocol("plan_cem", cfg)
    message = str(excinfo.value)
    assert GOAL_H_FIELD in message
    assert "plan_cem" not in message, (
        "the abort named the config name, so the table was selected before goal_H was resolved"
    )


def test_expected_table_requires_both_halves_of_the_key():
    """No default horizon: the caller answers "which regime" out of the resolved config."""
    with pytest.raises(TypeError):
        expected_table("plan_gd")  # type: ignore[call-arg]
    assert expected_table("plan_gd.yaml", SHORT_GOAL_H) is PROTOCOL_EXPECTED[
        ("plan_gd", SHORT_GOAL_H)
    ]
    assert expected_table("conf/plan_gd_mpc.yaml", LONG_GOAL_H) is PROTOCOL_EXPECTED[
        ("plan_gd_mpc", LONG_GOAL_H)
    ]
    with pytest.raises(ProtocolError) as excinfo:
        expected_table("plan_cem", SHORT_GOAL_H)
    assert "plan_gd" in str(excinfo.value) and "plan_gd_mpc" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 4. The guess is auditable: it reaches the manifest in words
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("config_name", SETTINGS)
def test_long_horizon_source_states_that_the_paper_does_not_state_it(config_name):
    """The manifest string is the deliverable, so its content is asserted, not just its presence."""
    source = PROTOCOL_EXPECTED_SOURCE[(config_name, LONG_GOAL_H)]
    lowered = source.lower()
    for phrase in (
        "does not state",          # the planner settings are not in the paper
        "short-horizon protocol",  # what the appendix table actually is
        "judgement call",          # it is a recorded decision, not a lookup
        "reading (a)",             # which reading was taken
        "frameskip 5",             # and the constraint both readings keep
    ):
        assert phrase in lowered, (
            f"{config_name} long-horizon source text does not say {phrase!r}. This string lands in "
            f"agg_run_manifest.json for every long-horizon run and is the only record that the "
            f"long-horizon planner settings are a guess:\n{source}"
        )
    assert "judgement call, not a lookup" in lowered
    # And the short columns must not claim the same thing, since they ARE a lookup.
    short_source = PROTOCOL_EXPECTED_SOURCE[(config_name, SHORT_GOAL_H)].lower()
    assert "judgement call" not in short_source


@pytest.mark.parametrize("config_name", SETTINGS)
@pytest.mark.parametrize("goal_h", [SHORT_GOAL_H, LONG_GOAL_H])
def test_the_manifest_records_the_regime(config_name, goal_h):
    """Requirement 8.6: a manifest can now tell a 25-step run from a 50-step one."""
    record = resolve_protocol(config_name, _cfg(config_name, goal_h))
    fragment = record.to_dict()
    assert fragment["protocol_horizon_regime"] == HORIZON_REGIMES[goal_h]
    assert fragment["protocol_goal_H"] == goal_h
    assert fragment["protocol_resolved"][GOAL_H_FIELD] == goal_h
    assert fragment["protocol_expected_source"] == PROTOCOL_EXPECTED_SOURCE[(config_name, goal_h)]

    manifest = plan_agg.build_manifest(protocol=record, agg_weight=0.1, cfg=_cfg(config_name, goal_h))
    assert manifest["horizon_regime"] == HORIZON_REGIMES[goal_h]
    assert manifest["goal_H"] == goal_h
    assert manifest["protocol_ok"] is True
    if goal_h == LONG_GOAL_H:
        assert "judgement call" in manifest["protocol_expected_source"].lower()


# ---------------------------------------------------------------------------
# 5. The no-hydra importability contract
# ---------------------------------------------------------------------------


_NO_HYDRA_PROBE = textwrap.dedent(
    """
    import sys

    BLOCKED = ("hydra", "omegaconf", "custom_resolvers", "plan")


    class Blocker:
        def find_module(self, name, path=None):
            return self.find_spec(name, path)

        def find_spec(self, name, path=None, target=None):
            root = name.split(".")[0]
            if root in BLOCKED:
                raise ImportError("blocked by the no-hydra importability probe: " + name)
            return None


    sys.meta_path.insert(0, Blocker())
    sys.path.insert(0, {repo!r})

    import plan_agg

    for config_name, goal_h, regime in (
        ("plan_gd", 25, "short"),
        ("plan_gd", 50, "long"),
        ("plan_gd_mpc", 25, "short"),
        ("plan_gd_mpc", 50, "long"),
    ):
        cfg = {{}}
        for path, value in plan_agg.PROTOCOL_EXPECTED[(config_name, goal_h)].items():
            node = cfg
            parts = path.split(".")
            for part in parts[:-1]:
                node = node.setdefault(part, {{}})
            node[parts[-1]] = value
        record = plan_agg.resolve_protocol(config_name, cfg)
        assert record.ok, record.message()
        assert record.horizon_regime == regime, record.horizon_regime
        assert record.goal_H == goal_h

    try:
        plan_agg.resolve_protocol("plan_gd", dict(cfg, goal_H=30))
    except plan_agg.ProtocolError as exc:
        assert "goal_H" in str(exc)
    else:
        raise AssertionError("an uncovered horizon did not abort")

    for blocked in BLOCKED:
        assert blocked not in sys.modules, blocked

    # The entry point itself is the only thing that needs hydra, and it says so rather than
    # failing with an AttributeError.
    try:
        plan_agg.main(None)
    except ImportError as exc:
        assert "hydra" in str(exc)
    else:
        raise AssertionError("plan_agg.main did not report the missing hydra")
    print("OK")
    """
)


def test_protocol_layer_works_with_no_hydra_and_no_omegaconf():
    """The importability contract, enforced in a process that cannot import either.

    ``plan_agg``'s docstring promises the protocol layer -- the expected tables,
    :func:`resolve_protocol`, the column selection -- stays usable on a box with no ``hydra``, no
    ``omegaconf`` and no CUDA, because that is the environment task 11.3's column selection has to
    be testable in. On the pod both import fine, so the promise is unfalsifiable there unless the
    imports are actively blocked, which is what this probe does. ``custom_resolvers`` and ``plan``
    are blocked too, since both pull hydra in transitively.

    ``torch`` is deliberately *not* blocked: ``plan_agg`` imports ``agg_objectives``, which needs
    torch to compute L_agg, so the contract is a no-hydra one rather than a no-torch one. Nothing in
    the protocol layer itself touches torch, and nothing anywhere here touches CUDA.
    """
    completed = subprocess.run(
        [sys.executable, "-c", _NO_HYDRA_PROBE.format(repo=str(_REPO_ROOT))],
        cwd=str(_REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
    )
    stdout = completed.stdout.decode("utf-8", "replace")
    stderr = completed.stderr.decode("utf-8", "replace")
    assert completed.returncode == 0 and "OK" in stdout, (
        "the protocol layer is no longer usable without hydra, omegaconf and torch, so task "
        "11.3's column selection cannot be tested off the pod.\n"
        f"stdout:\n{stdout}\nstderr:\n{stderr}"
    )
