"""Plan_Wrapper: the aggregated-space planning entry point.

Feature: ``aggregated-space-planning-cost``. This is the Hydra entry that runs a planning
evaluation with ``L_plan = L_spatial + w * L_agg`` **without editing** ``plan.py``,
``planning/*.py`` or ``datasets/*.py``. It validates, publishes Agg_Head, rewrites its *own*
``cfg_dict["objective"]`` and then hands that dict to the frozen ``plan.planning_main`` unchanged.

Ordering inside one wrapper process (frozen code marked ``[frozen]``)::

    plan_agg.main(cfg)                       # @hydra.main, cwd == the Hydra run dir
      1  validate_agg_weight(cfg.agg_weight)              Req 3.1, 3.4, 3.5   <- task 8.1
      2  resolve_protocol(config_name, cfg)               Req 8.1-8.4, 8.7    <- task 8.1
      3  load Agg_Head from the checkpoint                Req 2.4, 2.6        <- task 8.2
      4  AGG_CONTEXT.publish(head, w, opt_steps, dir)     Req 2.5            <- task 8.2
      5  plan.PlanEvaluator = RecordingPlanEvaluator      Req 7.4            <- task 8.2
      6  cfg_dict["objective"]["_target_"] / ["agg_weight"]  Req 2.3         <- task 8.2
      7  write agg_run_manifest.json                      Req 2.5, 8.6       <- task 8.2
      8  plan.planning_main(cfg_dict)         [frozen]    Req 2.1            <- task 8.2
      9  finally: restore, flush, clear                   Req 5.4            <- task 8.2

Steps 1 and 2 are this file's task-8.1 half and are deliberately the *first* things that happen:
both a bad weight and a protocol deviation must cost seconds, not a dataset load, an env spawn and
a DINOv2 download. Steps 3-9 are task 8.2 and are marked by explicit seams below
(:func:`load_agg_head_from_ckpt`, :func:`write_manifest`, :func:`delegate_to_plan`).

The Evaluation_Protocol, and the two overrides it needs
------------------------------------------------------

The protocol is the eleven-field table in :data:`PROTOCOL_EXPECTED`, keyed on the
``(config_name, goal_H)`` pair -- one column per setting per horizon regime -- which combines the
shipped config defaults with the overrides ``run_ccr_pilot.sh`` already applies. It is *not* "no
override relative to the mandated config file": ``conf/plan_gd.yaml`` ships ``objective.alpha: 0``
and ``conf/plan_gd_mpc.yaml`` ships ``objective.mode: all``, while the launcher already passes
``objective.alpha=1`` and ``objective.mode=last``/``staged`` per Requirements 8.2 and 8.3. So a
launch that omits those overrides *will* abort here, naming the field -- which is the point.

Requirement 8.4's field list (``max_iter 1``, ``n_taken_actions 25``, ...) coincides with
``conf/plan_gd.yaml``'s ``planner`` block alone. ``conf/plan_gd_mpc.yaml`` ships ``max_iter: 20``
and ``n_taken_actions: 5``; read literally for MPC, 8.4 would force ``max_iter 1``, which makes the
MPC setting open-loop-with-a-staged-objective and could not reproduce the 82.00 Platform_Baseline
that Requirement 8's own user story exists to stay comparable with. The per-setting reading is
confirmed (task 13.1) and encoded below. All eleven resolved values reach the manifest either way
(Requirement 8.6), so if the literal reading were the intended one the record already shows exactly
which two fields differ and the re-run is a two-flag change.

The two horizon regimes (task 11.3)
-----------------------------------

There are two columns per setting, selected on the resolved ``goal_H``: ``short`` (``goal_H 25``)
is the Table-1 protocol and the **reported** result (Requirement 7.2); ``long`` (``goal_H 50``) is
task 11.4's Positive_Control, which reproduces the paper's own long-horizon combined-cost cell. The
short columns pin exactly what they pinned before the long ones existed, and ``goal_H`` itself is
now pinned too, which is a field the gate did not previously constrain at all. The long column's
planner settings are **not stated anywhere in the paper** -- see
:data:`PROTOCOL_EXPECTED_SOURCE`, whose text is written to be read out of a manifest.

Importability
-------------

``hydra`` and ``omegaconf`` are imported at module scope, as ``plan.py`` does, because
``@hydra.main`` is applied at import time. The import is guarded so that everything *above* the
entry point -- the expected tables, :func:`resolve_protocol` and the weight validation -- stays
importable and testable on a box with no ``hydra``, no ``omegaconf`` and no CUDA (the Windows dev
environment). Nothing in the protocol layer touches Hydra, torch or the planning stack: it reads
plain mappings, which is what makes Property 12 (task 8.3) runnable without a pod. ``plan`` itself
is imported inside the functions that need it, so the protocol layer does not drag in ``gym``,
``wandb`` or ``submitit`` either.

Requirements: 3.1, 3.5, 4.2, 8.1, 8.2, 8.3, 8.4, 8.6, 8.7
"""

from __future__ import annotations

import json
import numbers
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import agg_objectives

try:  # pragma: no cover - exercised by which environment the module is imported in
    import hydra
    from hydra.core.hydra_config import HydraConfig
    from omegaconf import open_dict

    # Registers the OmegaConf resolvers that `conf/plan_gd*.yaml` -- and the `hydra.run.dir`
    # override this module is launched with -- interpolate, `replace_slash` among them.
    # `plan.py` does the same at module level (`from custom_resolvers import replace_slash`),
    # which is why the shipped template resolves there. Without it, Hydra parses the override
    # fine and then job start-up dies in OmegaConf resolution with
    #   UnsupportedInterpolationType: Unsupported interpolation type replace_slash
    # -- one layer later than the quoting bug, and with a completely different message.
    #
    # Inside the guarded block on purpose: `custom_resolvers` imports hydra/omegaconf, so at
    # module scope it would break the importability contract this file's docstring promises,
    # namely that the protocol layer stays importable on a box with no hydra and no CUDA.
    import custom_resolvers  # noqa: F401  # imported for its registration side effect
except ImportError as _exc:  # pragma: no cover - the CPU dev box has no hydra
    hydra = None
    HydraConfig = None
    open_dict = None
    _HYDRA_IMPORT_ERROR: Optional[ImportError] = _exc
else:  # pragma: no cover
    _HYDRA_IMPORT_ERROR = None


__all__ = [
    "EXPECTED_AGG_IN_DIM",
    "EXPECTED_AGG_OUT_DIM",
    "FEATURE_NAME",
    "FRAMESKIP",
    "GOAL_H_FIELD",
    "HORIZON_FIELDS",
    "HORIZON_REGIMES",
    "LONG_GOAL_H",
    "MANIFEST_FILENAME",
    "MISSING",
    "OBJECTIVE_TARGET",
    "PROTOCOL_COMMON",
    "PROTOCOL_COMMON_LONG",
    "PROTOCOL_EXPECTED",
    "PROTOCOL_EXPECTED_LONG",
    "PROTOCOL_EXPECTED_SHORT",
    "PROTOCOL_EXPECTED_SOURCE",
    "PROTOCOL_FIELDS",
    "OPT_STEPS_FIELD",
    "ProtocolDeviation",
    "ProtocolError",
    "ProtocolRecord",
    "SETTING_NAMES",
    "SHORT_GOAL_H",
    "build_manifest",
    "delegate_to_plan",
    "expected_table",
    "horizon_regime",
    "load_agg_head_from_ckpt",
    "main",
    "normalize_config_name",
    "resolve_checkpoint",
    "resolve_protocol",
    "rewrite_objective_block",
    "write_manifest",
]


#: Written into every manifest, so a stray run directory can be attributed.
FEATURE_NAME = "aggregated-space-planning-cost"

#: The run manifest, written next to the frozen ``logs.json`` (Requirements 2.5, 8.6).
MANIFEST_FILENAME = "agg_run_manifest.json"

#: The Hydra ``_target_`` the wrapper writes into its **own** ``cfg_dict["objective"]``
#: (Requirement 2.3). Never typed on a command line: a forgotten ``_target_`` would silently run
#: the spatial-only objective and the arm would be mislabelled.
OBJECTIVE_TARGET = "agg_objectives.create_agg_objective_fn"

#: The Target_Cell's Agg_Head widths: ``196 patches * 8 projected channels = 1568 -> 128``. A
#: checkpoint whose widths differ is *recorded and warned about*, not rejected: the widths are read
#: off the encoder (never parsed out of the run-directory name), so a different-but-consistent head
#: still computes a well-defined L_agg -- it is just not the Target_Cell's.
EXPECTED_AGG_IN_DIM = 1568
EXPECTED_AGG_OUT_DIM = 128

#: Human-readable setting name per Hydra config name (Requirements 8.2, 8.3).
SETTING_NAMES: Dict[str, str] = {
    "plan_gd": "open-loop",
    "plan_gd_mpc": "mpc",
}


# ---------------------------------------------------------------------------
# The Evaluation_Protocol expected tables (Requirements 8.1-8.4)
# ---------------------------------------------------------------------------

#: Every protocol field, in manifest order. Dotted paths into the resolved config.
#:
#: ``goal_H`` heads the list because it is what **selects** the expected column: the table is keyed
#: on the ``(config_name, goal_H)`` pair (task 11.3), so the horizon regime has to be resolved out
#: of the configuration before any other field can be compared against anything at all.
#:
#: Adding ``goal_H`` **strengthens** the short-horizon gate rather than weakening it. It was
#: previously unpinned, so a 50-step run satisfied the short columns on ``sub_planner.horizon``
#: alone and a manifest could not tell a 25-step run from a 50-step one -- which matters now that
#: the output tree holds both (task 11.4). It is now pinned: 25 in the short columns, 50 in the
#: long ones. Every value the two short columns pinned before task 11.3 still holds exactly that
#: value, and ``tests/test_agg_protocol_horizon.py`` asserts that literally rather than by eye.
PROTOCOL_FIELDS: Tuple[str, ...] = (
    "goal_H",
    "n_evals",
    "objective.mode",
    "objective.alpha",
    "planner.max_iter",
    "planner.n_taken_actions",
    "planner.sub_planner.horizon",
    "planner.sub_planner.lr",
    "planner.sub_planner.sample_type",
    "planner.sub_planner.action_noise",
    "planner.sub_planner.opt_steps",
)

#: The field whose resolved value the instrumentation recorder counts against
#: (``AGG_CONTEXT.publish(opt_steps=...)``, task 8.2).
OPT_STEPS_FIELD = "planner.sub_planner.opt_steps"

#: The field the horizon regime is read from, and the two horizons this feature holds columns for:
#: 25 is the Target_Cell / Table-1 protocol and the reported result (Requirement 7.2); 50 is task
#: 11.4's long-horizon Positive_Control, which reproduces the paper's own combined-cost cell.
GOAL_H_FIELD = "goal_H"
SHORT_GOAL_H = 25
LONG_GOAL_H = 50

#: Regime name per pinned ``goal_H``. A lookup rather than a ``goal_H != 25`` test, so the
#: selection names its own domain: a horizon that is in neither column aborts naming the field and
#: the two horizons that exist, instead of quietly falling into the long column.
HORIZON_REGIMES: Dict[int, str] = {SHORT_GOAL_H: "short", LONG_GOAL_H: "long"}

#: ``plan.py`` integer-divides these three fields by the checkpoint's ``frameskip``
#: (``PlanEvaluator.__init__``: ``goal_H // frameskip``, ``n_taken_actions // frameskip``,
#: ``sub_planner.horizon // frameskip``), so a value that is not a multiple of it is silently
#: **truncated** rather than rejected -- ``horizon 26`` plans the same 5 model steps as ``25``.
#: ``frameskip`` is 5 for every Table-1 cell and is a *training* config field, so it is a constant
#: here rather than something resolved from the plan config; ``_check_expected_tables()`` asserts
#: every pinned horizon in every column is divisible by it.
FRAMESKIP = 5
HORIZON_FIELDS: Tuple[str, ...] = (
    "goal_H",
    "planner.n_taken_actions",
    "planner.sub_planner.horizon",
)

#: Fields the two settings share at the **short** horizon: 50 samples per seed (Requirement 8.1),
#: ``objective.alpha 1`` (Requirements 8.2, 8.3) and the whole ``sub_planner`` block
#: (Requirement 8.4). ``goal_H 25`` is the shipped ``conf/plan_gd*.yaml`` value and the horizon
#: every recorded Platform_Baseline number was measured at.
PROTOCOL_COMMON: Dict[str, Any] = {
    "goal_H": SHORT_GOAL_H,
    "n_evals": 50,
    "objective.alpha": 1,
    "planner.sub_planner.horizon": 25,
    "planner.sub_planner.lr": 0.1,
    "planner.sub_planner.sample_type": "zero",
    "planner.sub_planner.action_noise": 0,
    "planner.sub_planner.opt_steps": 100,
}

#: The same block at the **long** horizon: exactly two fields move, ``goal_H`` and the subplanner
#: horizon that follows it (task 11.4's reading (a)). ``n_evals``, ``objective.alpha``, the
#: subplanner ``lr``, ``sample_type``, ``action_noise`` and ``opt_steps`` are unchanged, so the two
#: regimes differ in the horizon and in nothing else.
PROTOCOL_COMMON_LONG: Dict[str, Any] = dict(
    PROTOCOL_COMMON,
    **{
        "goal_H": LONG_GOAL_H,
        "planner.sub_planner.horizon": LONG_GOAL_H,
    },
)

#: The short-horizon columns, one **per setting**, keyed by Hydra config name. The three
#: per-setting fields are the ones the two configs genuinely differ in: the objective mode
#: Requirements 8.2/8.3 mandate, and the two MPC loop parameters ``conf/plan_gd_mpc.yaml`` ships
#: (task 13.1's confirmed reading).
#:
#: **These are the columns that protect the reported result** (Requirement 7.2), and task 11.3 does
#: not touch a value in them: every field they pinned before it keeps exactly that value, and
#: ``goal_H 25`` is added to them, which pins a field that used to be free.
PROTOCOL_EXPECTED_SHORT: Dict[str, Dict[str, Any]] = {
    "plan_gd": dict(
        PROTOCOL_COMMON,
        **{
            "objective.mode": "last",
            "planner.max_iter": 1,
            "planner.n_taken_actions": 25,
        },
    ),
    "plan_gd_mpc": dict(
        PROTOCOL_COMMON,
        **{
            "objective.mode": "staged",
            "planner.max_iter": 20,
            "planner.n_taken_actions": 5,
        },
    ),
}

#: The long-horizon columns (task 11.4's Positive_Control, reading (a)). Identical to the short
#: columns except for the three horizon fields: ``goal_H 50``, ``sub_planner.horizon 50``, and
#: ``n_taken_actions`` 50 open-loop / 5 MPC -- which preserves the appendix protocol's own
#: "executed actions = horizon" relationship open-loop and its footnoted MPC value unchanged.
#: ``max_iter`` and ``objective.mode`` are per-setting and horizon-independent.
PROTOCOL_EXPECTED_LONG: Dict[str, Dict[str, Any]] = {
    "plan_gd": dict(
        PROTOCOL_COMMON_LONG,
        **{
            "objective.mode": "last",
            "planner.max_iter": 1,
            "planner.n_taken_actions": LONG_GOAL_H,
        },
    ),
    "plan_gd_mpc": dict(
        PROTOCOL_COMMON_LONG,
        **{
            "objective.mode": "staged",
            "planner.max_iter": 20,
            "planner.n_taken_actions": 5,
        },
    ),
}

#: The expected table, keyed on the ``(config_name, goal_H)`` **pair** (task 11.3). The pair is the
#: whole point: ``resolve_protocol`` reads ``goal_H`` out of the configuration *first* and then
#: selects, so the short columns keep aborting on any 50-step deviation and the long columns abort
#: on any 25-step one. Neither regime can be reached by accident, because a ``goal_H`` that is in
#: neither is an abort rather than a fallback.
PROTOCOL_EXPECTED: Dict[Tuple[str, int], Dict[str, Any]] = {
    (config_name, goal_h): table
    for goal_h, tables in (
        (SHORT_GOAL_H, PROTOCOL_EXPECTED_SHORT),
        (LONG_GOAL_H, PROTOCOL_EXPECTED_LONG),
    )
    for config_name, table in tables.items()
}

#: Where each column comes from, recorded in the manifest so the reading is auditable rather than
#: implicit (Requirement 8.6). Keyed on the same ``(config_name, goal_H)`` pair as the tables.
#:
#: The two long-horizon strings say plainly that their planner settings are a **judgement call**.
#: That text is the deliverable: it lands in ``agg_run_manifest.json`` for every long-horizon run,
#: so the guess is attached to the numbers it produced instead of being reconstructed afterwards.
_LONG_HORIZON_SOURCE_PREAMBLE = (
    "LONG-HORIZON COLUMN: A RECORDED JUDGEMENT CALL, NOT A LOOKUP. The paper does NOT state the "
    "long-horizon planner settings anywhere. Its appendix protocol table (Subplanner horizon 25, "
    "# Executed actions 25, footnoted as 5 for MPC) is the SHORT-horizon protocol, and the "
    "long-horizon paragraph and tab:long_horizon of paper_tex/sec/1_main.tex introduce "
    "L_plan = L_spatial + 0.1 * L_agg for \"a longer-horizon setting where the target is 50 steps "
    "away\" without giving a single planner value. This column encodes task 11.4's reading (a): "
    "scale the horizon with the goal distance (goal_H 50, planner.sub_planner.horizon 50, "
    "planner.n_taken_actions 50 open-loop), which preserves the appendix's own "
    "\"executed actions = horizon\" relationship, and keep the footnoted MPC value of 5 executed "
    "actions. The rejected reading (b) was to hold horizon at 25 and let open-loop cover half the "
    "distance, which would by itself explain the paper's open-loop collapse to 13.33 against MPC's "
    "24.00; (a) was chosen because it is the only reading under which open-loop is even attempting "
    "the task. Both readings keep frameskip 5 and every horizon divisible by it. This is a guess "
    "either way: it is recorded here, in the manifest, so it is auditable"
)

PROTOCOL_EXPECTED_SOURCE: Dict[Tuple[str, int], str] = {
    ("plan_gd", SHORT_GOAL_H): (
        "conf/plan_gd.yaml shipped defaults, plus the objective.alpha=1 and objective.mode=last "
        "overrides run_ccr_pilot.sh already applies (Requirements 8.2, 8.4); goal_H 25 is the "
        "shipped value and the horizon the Platform_Baseline and the Paper_Target were measured at"
    ),
    ("plan_gd_mpc", SHORT_GOAL_H): (
        "conf/plan_gd_mpc.yaml shipped defaults, plus the objective.alpha=1 and "
        "objective.mode=staged overrides run_ccr_pilot.sh already applies (Requirements 8.3, 8.4); "
        "max_iter 20 and n_taken_actions 5 are this setting's shipped values, per the confirmed "
        "per-setting reading of Requirement 8.4; goal_H 25 is the shipped value and the horizon "
        "the Platform_Baseline and the Paper_Target were measured at"
    ),
    ("plan_gd", LONG_GOAL_H): (
        _LONG_HORIZON_SOURCE_PREAMBLE + ". Everything outside the three horizon fields is the "
        "short open-loop column unchanged: conf/plan_gd.yaml shipped defaults plus the "
        "objective.alpha=1 and objective.mode=last overrides run_ccr_pilot.sh already applies "
        "(Requirements 8.2, 8.4)"
    ),
    ("plan_gd_mpc", LONG_GOAL_H): (
        _LONG_HORIZON_SOURCE_PREAMBLE + ". Everything outside the two horizon fields that move is "
        "the short MPC column unchanged: conf/plan_gd_mpc.yaml shipped defaults plus the "
        "objective.alpha=1 and objective.mode=staged overrides run_ccr_pilot.sh already applies "
        "(Requirements 8.3, 8.4), with max_iter 20 and n_taken_actions 5 this setting's shipped "
        "values. n_taken_actions stays 5 here: the appendix footnotes 5 executed actions for MPC "
        "independently of the horizon, and this is the setting the paper's long-horizon claim is "
        "actually scoped to (\"under MPC\")"
    ),
}

#: Recorded as the resolved value of a protocol field the configuration does not define at all.
#: A plain string rather than ``None``, because ``None`` is a value a config could legitimately
#: hold and the two must not read alike in the manifest.
MISSING = "<missing>"

_MISSING_SENTINEL = object()


def _check_expected_tables() -> None:
    """Self-check on the expected columns, run at import. Raises rather than asserts.

    Three statements that are cheap to hold and expensive to lose:

    1. every setting has a column in **both** regimes, so a launch cannot find one regime's table
       missing at the moment it needs it;
    2. every column pins exactly :data:`PROTOCOL_FIELDS`, so a field added to the list without a
       value in some column cannot surface as a ``KeyError`` inside :func:`resolve_protocol`;
    3. every pinned horizon is divisible by :data:`FRAMESKIP`. ``plan.py`` integer-divides
       ``goal_H``, ``planner.n_taken_actions`` and ``planner.sub_planner.horizon`` by the
       checkpoint's frameskip, so a non-multiple is silently truncated -- a protocol column that
       pinned one would be pinning a number the planner never actually uses.

    A ``raise`` rather than an ``assert``, because ``python -O`` strips the latter and this is the
    only check on the tables' shape.
    """
    settings = tuple(SETTING_NAMES)
    for goal_h in HORIZON_REGIMES:
        for config_name in settings:
            if (config_name, goal_h) not in PROTOCOL_EXPECTED:
                raise RuntimeError(
                    f"plan_agg.PROTOCOL_EXPECTED has no column for "
                    f"({config_name!r}, goal_H={goal_h}); every setting needs one per horizon "
                    f"regime, or a launch in that regime aborts on a missing table rather than on "
                    f"a protocol deviation."
                )

    fields = set(PROTOCOL_FIELDS)
    for key, table in PROTOCOL_EXPECTED.items():
        missing = sorted(fields - set(table))
        extra = sorted(set(table) - fields)
        if missing or extra:
            raise RuntimeError(
                f"plan_agg.PROTOCOL_EXPECTED[{key!r}] does not pin exactly PROTOCOL_FIELDS: "
                f"missing {missing}, unexpected {extra}."
            )
        if key not in PROTOCOL_EXPECTED_SOURCE:
            raise RuntimeError(
                f"plan_agg.PROTOCOL_EXPECTED_SOURCE has no entry for {key!r}; every column's "
                f"provenance reaches the manifest (Requirement 8.6)."
            )
        for field_path in HORIZON_FIELDS:
            value = table[field_path]
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
                or value % FRAMESKIP
            ):
                raise RuntimeError(
                    f"plan_agg.PROTOCOL_EXPECTED[{key!r}][{field_path!r}] is {value!r}, which is "
                    f"not a positive multiple of frameskip {FRAMESKIP}. plan.py integer-divides "
                    f"this field by frameskip, so the planner would silently run a truncated "
                    f"horizon while the manifest recorded the untruncated number."
                )
        if table["goal_H"] != key[1]:
            raise RuntimeError(
                f"plan_agg.PROTOCOL_EXPECTED[{key!r}] pins goal_H {table['goal_H']!r} but is keyed "
                f"at {key[1]!r}; the table is selected by goal_H, so the two must agree or the "
                f"selected column could never match the configuration that selected it."
            )


_check_expected_tables()


# ---------------------------------------------------------------------------
# Protocol resolution and comparison -- pure, plain-mapping, no Hydra, no torch
# ---------------------------------------------------------------------------


class ProtocolError(RuntimeError):
    """Raised when the Evaluation_Protocol is not the one this feature measures against.

    Always raised **before** any load: no checkpoint, no dataset, no env, no DINOv2 download
    (Requirement 8.7).
    """


@dataclass(frozen=True)
class ProtocolDeviation:
    """One protocol field whose resolved value is not the expected one."""

    field: str
    expected: Any
    resolved: Any

    def describe(self) -> str:
        return f"{self.field}: expected {self.expected!r}, resolved {self.resolved!r}"

    def to_dict(self) -> Dict[str, Any]:
        return {"field": self.field, "expected": self.expected, "resolved": self.resolved}


@dataclass(frozen=True)
class ProtocolRecord:
    """The resolved protocol, the expected table it was checked against, and any deviations.

    Produced whether or not the check passes, because Requirement 8.6 wants every resolved value
    recorded either way: if the literal reading of Requirement 8.4 turns out to be the intended one,
    the manifest already shows exactly which fields differ.
    """

    config_name: str
    setting: str
    resolved: Dict[str, Any]
    expected: Dict[str, Any]
    deviations: Tuple[ProtocolDeviation, ...]
    expected_source: str
    #: Which of :data:`HORIZON_REGIMES` selected ``expected`` -- ``"short"`` (goal_H 25, the
    #: reported result) or ``"long"`` (goal_H 50, task 11.4's Positive_Control). Recorded so the
    #: manifest states the regime in one word rather than leaving it to be inferred from three
    #: horizon numbers, and so a reader of the output tree -- which now holds both -- can tell them
    #: apart at all.
    horizon_regime: str = HORIZON_REGIMES[SHORT_GOAL_H]
    #: The resolved ``goal_H`` the regime was selected on, as an ``int``.
    goal_H: int = SHORT_GOAL_H

    @property
    def ok(self) -> bool:
        """Whether every protocol field resolved to its expected value."""
        return not self.deviations

    @property
    def opt_steps(self) -> Optional[int]:
        """The resolved ``planner.sub_planner.opt_steps``, which the recorder counts against."""
        value = self.resolved.get(OPT_STEPS_FIELD)
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            return None
        return int(value)

    def to_dict(self) -> Dict[str, Any]:
        """The manifest fragment (Requirement 8.6). Task 8.2 merges this into the manifest."""
        return {
            "protocol_resolved": dict(self.resolved),
            "protocol_expected": dict(self.expected),
            "protocol_expected_source": self.expected_source,
            "protocol_horizon_regime": self.horizon_regime,
            "protocol_goal_H": self.goal_H,
            "protocol_ok": self.ok,
            "protocol_deviations": [d.to_dict() for d in self.deviations],
        }

    def message(self) -> str:
        """The Requirement 8.7 abort message: every deviating field, expected and resolved."""
        lines = [
            f"Evaluation_Protocol deviation in the {self.setting} setting "
            f"(config name {self.config_name!r}, {self.horizon_regime} horizon, "
            f"goal_H {self.goal_H}): "
            f"{len(self.deviations)} of {len(self.expected)} fields differ from the expected "
            f"table.",
        ]
        lines.extend(f"  {deviation.describe()}" for deviation in self.deviations)
        lines.append(
            "The protocol is held fixed so these numbers stay comparable to the Platform_Baseline "
            "(75.33 open-loop / 82.00 MPC) and the Paper_Target (77.33 / 85.33); a changed field "
            "silently invalidates that comparison, so this is an abort rather than a warning."
        )
        lines.append(f"Expected table source: {self.expected_source}.")
        lines.append(
            "Nothing has been loaded: this aborts before the checkpoint, the dataset and the "
            "environment."
        )
        return "\n".join(lines)


def normalize_config_name(config_name: Any) -> str:
    """Reduce a Hydra config name to its bare stem.

    ``HydraConfig.get().job.config_name`` reports whatever ``--config-name`` was given, and the
    launcher passes ``--config-name plan_gd.yaml`` (see ``run_ccr_pilot.sh::run_eval_jobs``), so the
    raw value carries a ``.yaml`` suffix. Without this normalization every real launch would abort
    on an unknown config name rather than on a genuine protocol deviation. A directory component is
    stripped too, so ``conf/plan_gd.yaml`` resolves the same way.
    """
    if config_name is None:
        return ""
    name = str(config_name).strip().replace("\\", "/")
    name = name.rsplit("/", 1)[-1]
    for suffix in (".yaml", ".yml"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """``cfg[key]`` with a default, for an ``OmegaConf`` node or a plain mapping alike."""
    try:
        return cfg[key]
    except (KeyError, IndexError, TypeError, AttributeError):
        return default


def _lookup(cfg: Any, dotted: str) -> Any:
    """Read a dotted path out of a nested mapping. Returns the missing sentinel if absent.

    Works for a plain ``dict`` and for an ``omegaconf`` node without importing either: both raise a
    ``KeyError`` subclass for an absent key. Deliberately does *not* use attribute access, so a
    method name on a mapping subclass cannot be mistaken for a config value.
    """
    node: Any = cfg
    for part in dotted.split("."):
        try:
            node = node[part]
        except (KeyError, IndexError, TypeError, AttributeError):
            return _MISSING_SENTINEL
    return node


def _plain(value: Any) -> Any:
    """Coerce a resolved config leaf into something ``json.dump`` accepts.

    Leaf reads off an ``omegaconf`` node already come back as plain Python scalars; anything else
    (a container, an enum, a path object) is recorded by its ``repr``-free string form rather than
    dropped, since the manifest's job is to show what was actually resolved.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _values_match(expected: Any, resolved: Any) -> bool:
    """Whether a resolved protocol value is the expected one.

    Numbers compare numerically, so a config that yields ``1`` where the table says ``1`` matches
    whether Hydra typed it as ``int`` or ``float``; ``bool`` is never accepted for a numeric field,
    because ``True == 1`` would let ``max_iter: true`` pass as ``max_iter 1``. Strings compare
    exactly, after stripping surrounding whitespace only.
    """
    if isinstance(expected, str):
        return isinstance(resolved, str) and resolved.strip() == expected.strip()
    if isinstance(expected, bool):
        return isinstance(resolved, bool) and resolved == expected
    if isinstance(expected, numbers.Real):
        if isinstance(resolved, bool) or not isinstance(resolved, numbers.Real):
            return False
        return float(resolved) == float(expected)
    return resolved == expected


def _as_goal_h(value: Any) -> Optional[int]:
    """``value`` as an integral ``goal_H``, or ``None`` if it is not one.

    ``bool`` is refused for the same reason :func:`_values_match` refuses it for numeric fields:
    ``True == 1`` must not let a flag pass as a horizon. A float is accepted only when it is
    integral, since ``plan.py`` uses ``goal_H`` in integer division and slicing.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    as_int = int(value)
    return as_int if float(value) == float(as_int) else None


def horizon_regime(goal_H: Any) -> Tuple[str, int]:
    """The horizon regime ``goal_H`` selects, as ``(regime_name, goal_H_as_int)``.

    This is the explicit half of task 11.3's column selection. It is a **lookup in**
    :data:`HORIZON_REGIMES`, not a ``goal_H != 25`` test: an unrecognised horizon aborts naming the
    field and the horizons that exist, rather than silently landing in the long column, where it
    would then deviate on ``sub_planner.horizon`` and be reported as the wrong problem.

    Raises:
        ProtocolError: if ``goal_H`` is absent, non-integral, or not one of the two horizons this
            feature holds a column for. Raised before any load, exactly as a field deviation is.
    """
    as_int = _as_goal_h(goal_H)
    if as_int is not None and as_int in HORIZON_REGIMES:
        return HORIZON_REGIMES[as_int], as_int

    known = ", ".join(
        f"{value} ({name})" for value, name in sorted(HORIZON_REGIMES.items())
    )
    raise ProtocolError(
        f"{GOAL_H_FIELD} resolved to {goal_H!r}, and the aggregated-space planning cost holds an "
        f"Evaluation_Protocol column only for these horizons: {known}. The short column is the "
        f"Table-1 protocol the reported result is measured under (Requirement 7.2); the long one "
        f"is task 11.4's Positive_Control, which reproduces the paper's own long-horizon "
        f"combined-cost cell. A third horizon is not a deviation this feature can interpret -- its "
        f"planner settings would be a second undocumented guess -- so it aborts here rather than "
        f"being checked against a column that was never meant for it. Launch with "
        f"{GOAL_H_FIELD}={SHORT_GOAL_H} or {GOAL_H_FIELD}={LONG_GOAL_H}, or add a column and "
        f"record where its values come from in PROTOCOL_EXPECTED_SOURCE. Nothing has been loaded."
    )


def expected_table(config_name: str, goal_H: Any) -> Dict[str, Any]:
    """The expected protocol table for the ``(config_name, goal_H)`` pair.

    Both halves of the key are required, and neither has a default: the short columns are what
    protect the reported result, so "which horizon regime is this" must be answered by the caller
    out of the resolved configuration rather than assumed here.

    Raises:
        ProtocolError: naming the known settings if ``config_name`` names none, or -- through
            :func:`horizon_regime` -- naming the known horizons if ``goal_H`` is not one of them.
    """
    name = normalize_config_name(config_name)
    _regime, goal_h = horizon_regime(goal_H)
    try:
        return PROTOCOL_EXPECTED[(name, goal_h)]
    except KeyError:
        known = ", ".join(sorted(SETTING_NAMES))
        raise ProtocolError(
            f"the aggregated-space planning cost holds one Evaluation_Protocol table per "
            f"(setting, horizon) pair and has none for config name {name!r} at "
            f"{GOAL_H_FIELD}={goal_h} (raw config name: {config_name!r}). Requirement 8.2 "
            f"measures the open-loop setting through conf/plan_gd.yaml and Requirement 8.3 the MPC "
            f"setting through conf/plan_gd_mpc.yaml, so the known config names are: {known}. "
            f"Launch with --config-name plan_gd.yaml or --config-name plan_gd_mpc.yaml."
        ) from None


def resolve_protocol(config_name: Any, cfg: Any, strict: bool = True) -> ProtocolRecord:
    """Resolve every Evaluation_Protocol field and check it against the ``(setting, horizon)`` table.

    ``goal_H`` is resolved out of ``cfg`` **first**, and the expected column is selected on the
    ``(config_name, goal_H)`` pair (task 11.3). That order is the requirement, not an
    implementation detail: the short columns pin ``goal_H 25``, ``sub_planner.horizon 25`` and
    ``n_taken_actions`` 25/5, so a 50-step run checked against them would abort on three fields
    before it could say the one thing worth saying, which is that it is a long-horizon run.

    Args:
        config_name: what ``HydraConfig.get().job.config_name`` reports, with or without the
            ``.yaml`` suffix.
        cfg: the resolved configuration. Any nested mapping works -- an ``omegaconf`` node, or a
            plain ``dict``, which is what makes this checkable without Hydra.
        strict: raise on deviation (the default, Requirement 8.7). ``False`` returns the record with
            its deviations recorded instead, which is how the manifest keeps every resolved value
            even for a configuration this feature would refuse to run (Requirement 8.6).

    Returns:
        The :class:`ProtocolRecord`, whose ``resolved`` holds every field in
        :data:`PROTOCOL_FIELDS` order and whose ``horizon_regime`` names the column that was used.

    Raises:
        ProtocolError: if ``config_name`` names no known setting, if ``goal_H`` is not one of the
            two horizons a column exists for, or -- when ``strict`` -- if any field deviates. The
            first two raise even when ``strict`` is ``False``, because without a column there is
            nothing to record a deviation *against*; that is the same behaviour an unknown config
            name has always had. Raised before any load, naming the field, the expected value and
            the resolved value.
    """
    name = normalize_config_name(config_name)

    # The horizon regime is resolved before the table is chosen -- see the docstring.
    raw_goal_h = _lookup(cfg, GOAL_H_FIELD)
    regime, goal_h = horizon_regime(
        MISSING if raw_goal_h is _MISSING_SENTINEL else raw_goal_h
    )
    expected = expected_table(name, goal_h)

    resolved: Dict[str, Any] = {}
    deviations = []
    for field_path in PROTOCOL_FIELDS:
        raw = _lookup(cfg, field_path)
        if raw is _MISSING_SENTINEL:
            value: Any = MISSING
        else:
            value = _plain(raw)
        resolved[field_path] = value

        want = expected[field_path]
        if raw is _MISSING_SENTINEL or not _values_match(want, value):
            deviations.append(
                ProtocolDeviation(field=field_path, expected=want, resolved=value)
            )

    # `goal_H` is in `PROTOCOL_FIELDS`, so it is resolved and recorded like every other field, and
    # by construction its comparison above cannot deviate: the column was chosen by it. That is
    # deliberate -- the abort for a horizon no column covers happens earlier, in `horizon_regime`,
    # with a message about the horizon rather than about a field mismatch. What the entry buys is
    # the manifest record: `protocol_resolved["goal_H"]` and `protocol_horizon_regime` say which
    # regime produced the numbers, which nothing in the output tree said before task 11.3.
    record = ProtocolRecord(
        config_name=name,
        setting=SETTING_NAMES.get(name, name),
        resolved=resolved,
        expected=dict(expected),
        deviations=tuple(deviations),
        expected_source=PROTOCOL_EXPECTED_SOURCE[(name, goal_h)],
        horizon_regime=regime,
        goal_H=goal_h,
    )

    if strict and not record.ok:
        raise ProtocolError(record.message())
    return record


# ---------------------------------------------------------------------------
# Checkpoint side: head load, manifest, delegation
# ---------------------------------------------------------------------------
#
# Everything below is the checkpoint-side half of this file: the head load, the publication, the
# `cfg_dict` rewrite, the `plan.PlanEvaluator` rebind, the manifest write, the
# `plan.planning_main` delegation and the `finally` restore.
#
# Two of these are deliberately split into a pure part and an effectful part, so the halves that
# decide what gets recorded are exercisable on a box with no hydra, no checkpoint and no CUDA:
# `resolve_checkpoint` (path resolution, loads nothing) under `load_agg_head_from_ckpt`, and
# `build_manifest` (payload assembly, writes nothing) under `write_manifest`. Both take a plain
# mapping, exactly as the protocol layer above does.


def resolve_checkpoint(cfg: Any) -> Tuple[str, str]:
    """Resolve ``(model_path, model_ckpt)`` exactly the way ``plan.planning_main`` does.

    Pure: reads the configuration and joins paths, loads nothing. The three lines mirrored here --
    the ``ckpt_base_path.startswith("/")`` absolute-path branch, the ``{base}/{model_name}/`` join
    and ``checkpoints/model_{model_epoch}.pth`` -- are frozen code, so they are mirrored rather than
    improved: the wrapper must load the *same* file the planner will load a few seconds later, or
    Agg_Head would not be the planner's own head.

    ``cfg`` may be an ``omegaconf`` node or a plain ``dict``, which is what lets the manifest
    assembly be exercised without Hydra.
    """
    ckpt_base_path = _cfg_get(cfg, "ckpt_base_path", None)
    if ckpt_base_path is None:
        raise ProtocolError(
            "ckpt_base_path is not set, so the Target_Cell checkpoint cannot be located and "
            "Agg_Head cannot be obtained (Requirements 2.4, 8.5). Pass "
            "ckpt_base_path=<ckpt_root> model_name=<model_name>, as plan.py requires."
        )
    ckpt_base_path = str(ckpt_base_path)

    model_name = _cfg_get(cfg, "model_name", None)
    if ckpt_base_path.startswith("/"):
        model_path = ckpt_base_path
    else:
        if model_name is None:
            raise ProtocolError(
                f"model_name is not set and ckpt_base_path={ckpt_base_path!r} is not absolute, so "
                f"the checkpoint directory cannot be resolved. plan.planning_main forms "
                f"'{{ckpt_base_path}}/{{model_name}}/' in exactly this case."
            )
        model_path = f"{ckpt_base_path}/{model_name}/"
    model_path = os.path.abspath(model_path)

    model_epoch = _cfg_get(cfg, "model_epoch", "final")
    model_ckpt = Path(model_path) / "checkpoints" / f"model_{model_epoch}.pth"
    return model_path, str(model_ckpt)


def load_agg_head_from_ckpt(cfg: Any) -> Tuple[Any, int, int]:
    """Load Agg_Head from the Target_Cell checkpoint (Requirements 2.4, 2.6, 8.5).

    Resolves the checkpoint the way ``plan.planning_main`` does, calls the frozen
    ``plan.load_ckpt(model_ckpt, device="cpu")``, aborts if the payload carries no ``encoder`` key
    (Requirement 2.4 cannot be met without it), and passes the encoder through
    :func:`agg_objectives.extract_agg_head`, which raises naming the encountered ``agg_type`` if it
    is not ``mlp`` (Requirement 2.6).

    Returns ``(head, in_dim, out_dim)``. The head also carries the load's provenance as plain
    attributes -- ``checkpoint``, ``checkpoint_file``, ``checkpoint_epoch``, ``width_warnings`` --
    which :func:`write_manifest` records; ``extract_agg_head`` already sets ``in_dim``, ``out_dim``
    and ``agg_type`` the same way, so this is that module's own convention rather than a new one.

    Widths that differ from :data:`EXPECTED_AGG_IN_DIM` / :data:`EXPECTED_AGG_OUT_DIM` are warned
    about and recorded, not rejected: a consistent head of another size still defines a
    well-formed L_agg, it is simply not the Target_Cell's, and the manifest is where that has to be
    visible.

    Nothing here mutates the checkpoint (Requirement 8.5). ``plan.load_ckpt`` deserializes a *fresh*
    set of modules on the CPU, so the ``agg_mlp`` / ``agg_post_norm`` submodules Agg_Head shares are
    this load's own objects -- not the planner's encoder, which ``plan.planning_main`` loads
    separately a few seconds later. No parameter is written, cast or re-initialized anywhere in this
    path: ``extract_agg_head`` composes the existing submodules and ``agg_objectives._apply_head``
    only calls ``eval()`` and ``requires_grad_(False)``.

    This load stays **before** ``plan.planning_main`` calls ``utils.seed(cfg_dict["seed"])``, which
    reseeds ``random``, ``torch``, ``numpy`` and every CUDA generator. Every RNG state inside
    ``planning_main`` is therefore exactly what ``plan.py`` would have had, which is what keeps
    Requirement 3.3 and the Paired_Comparison exact. Do not move it later.
    """
    import plan  # local: keeps the protocol layer importable without gym/wandb/submitit

    model_path, model_ckpt = resolve_checkpoint(cfg)
    snapshot = Path(model_ckpt)
    if not snapshot.exists():
        raise FileNotFoundError(
            f"the checkpoint {model_ckpt!r} does not exist, so Agg_Head cannot be obtained from "
            f"the checkpoint's encoder (Requirement 2.4). Expected the Target_Cell checkpoint "
            f"under {model_path!r}; check ckpt_base_path, model_name and model_epoch."
        )

    print(f"[plan_agg] loading Agg_Head from {model_ckpt} (device=cpu)", flush=True)
    payload = plan.load_ckpt(snapshot, device="cpu")

    if "encoder" not in payload:
        raise RuntimeError(
            f"the checkpoint {model_ckpt!r} carries no 'encoder', so Agg_Head cannot be obtained "
            f"from it and Requirement 2.4 cannot be met. plan.load_ckpt returned the keys "
            f"{sorted(payload)!r}. Aggregated-space planning needs the checkpoint's own "
            f"aggregation head: a freshly instantiated encoder (which plan.load_model would fall "
            f"back to) has untrained agg weights and would measure L_agg in a space no training "
            f"ever shaped."
        )

    head, in_dim, out_dim = agg_objectives.extract_agg_head(payload["encoder"])

    width_warnings = []
    if in_dim != EXPECTED_AGG_IN_DIM or out_dim != EXPECTED_AGG_OUT_DIM:
        width_warnings.append(
            f"Agg_Head widths are {in_dim} -> {out_dim}; the Target_Cell's are "
            f"{EXPECTED_AGG_IN_DIM} -> {EXPECTED_AGG_OUT_DIM} (196 patches x 8 projected channels, "
            f"agg_out_dim 128). The widths are read off the checkpoint's encoder, so L_agg is "
            f"still well defined -- but this is not the Target_Cell head, and the numbers are not "
            f"comparable to the Platform_Baseline."
        )
    for message in width_warnings:
        print(f"[plan_agg][warning] {message}", flush=True)

    # Provenance for the manifest. Plain str/int/tuple values, so `nn.Module.__setattr__` stores
    # them as ordinary attributes and no parameter or submodule is touched.
    head.checkpoint = model_path
    head.checkpoint_file = str(snapshot)
    head.checkpoint_epoch = payload.get("epoch")
    head.width_warnings = tuple(width_warnings)

    print(
        f"[plan_agg] Agg_Head ready: agg_type=mlp in_dim={in_dim} out_dim={out_dim} "
        f"checkpoint_epoch={head.checkpoint_epoch!r}",
        flush=True,
    )
    return head, in_dim, out_dim


def rewrite_objective_block(cfg_dict: Dict[str, Any], agg_weight: float) -> Any:
    """Point the objective block at this feature's factory (Requirement 2.3). Returns the old target.

    Mutates the **wrapper's own** dict -- the one about to be handed to ``plan.planning_main`` --
    and nothing else: no config file is edited and no name inside ``planning/`` is rebound.

    Done here rather than on the command line so ``_target_`` cannot be forgotten: a launch that
    omitted it would run the spatial-only objective, write its numbers into an ``aggw<w>`` directory
    and mislabel the arm without ever erroring. The only override a user types is
    ``+agg_weight=<w>``.

    ``agg_weight`` is written into the block as well, so it reaches
    :func:`agg_objectives.create_agg_objective_fn` as an ordinary config kwarg and appears in the
    config Hydra records (Requirement 2.5).

    Pure apart from the mutation: a plain ``dict`` is all it needs, which is what makes it
    exercisable without Hydra.
    """
    objective_block = cfg_dict.get("objective")
    if not isinstance(objective_block, dict):
        raise ProtocolError(
            f"cfg.objective must be a mapping carrying the objective's _target_, alpha, base and "
            f"mode, because plan.py builds the objective with "
            f"hydra.utils.call(cfg_dict['objective']) and passes nothing else; resolved "
            f"{objective_block!r}."
        )
    previous_target = objective_block.get("_target_")
    objective_block["_target_"] = OBJECTIVE_TARGET
    objective_block["agg_weight"] = float(agg_weight)
    return previous_target


def _git_rev() -> Optional[str]:
    """The repository revision, or ``None`` if it cannot be read. Never raises."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).resolve().parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.decode("utf-8", "replace").strip() or None


def build_manifest(
    protocol: ProtocolRecord,
    agg_weight: float,
    cfg: Any,
    in_dim: Optional[int] = None,
    out_dim: Optional[int] = None,
    output_dir: Optional[str] = None,
    head: Any = None,
) -> Dict[str, Any]:
    """Assemble the manifest payload (Requirements 2.5, 8.6). Pure: writes nothing.

    Separated from :func:`write_manifest` so the assembly is exercisable against a plain ``dict``
    with no Hydra, no torch and no checkpoint -- which is the whole reason the protocol layer above
    takes plain mappings too.
    """
    checkpoint: Any
    checkpoint_file: Any
    try:
        checkpoint, checkpoint_file = resolve_checkpoint(cfg)
    except ProtocolError as exc:
        # A manifest that records *why* the checkpoint could not be named is more useful than no
        # manifest; the run itself has already failed by the time this can happen.
        checkpoint = f"<unresolved: {exc}>"
        checkpoint_file = MISSING

    agg_head: Dict[str, Any] = {
        "agg_type": getattr(head, "agg_type", "mlp"),
        "in_dim": in_dim,
        "out_dim": out_dim,
        "in_dim_expected": EXPECTED_AGG_IN_DIM,
        "out_dim_expected": EXPECTED_AGG_OUT_DIM,
        "widths_as_expected": (
            in_dim == EXPECTED_AGG_IN_DIM and out_dim == EXPECTED_AGG_OUT_DIM
        ),
        "width_warnings": list(getattr(head, "width_warnings", ()) or ()),
    }

    manifest: Dict[str, Any] = {
        "feature": FEATURE_NAME,
        "config_name": protocol.config_name,
        "setting": protocol.setting,
        # Top level as well as inside `protocol.to_dict()`: after task 11.4 the output tree holds
        # both regimes, and "which horizon produced this number" is a first-order fact about the
        # run rather than a detail of the protocol check.
        "horizon_regime": protocol.horizon_regime,
        "goal_H": protocol.goal_H,
        "agg_weight": float(agg_weight),
        "objective_target": OBJECTIVE_TARGET,
        "seed": _plain(_cfg_get(cfg, "seed", MISSING)),
        "model_name": _plain(_cfg_get(cfg, "model_name", MISSING)),
        "model_epoch": _plain(_cfg_get(cfg, "model_epoch", MISSING)),
        "checkpoint": checkpoint,
        "checkpoint_file": checkpoint_file,
        "checkpoint_epoch": _plain(getattr(head, "checkpoint_epoch", None)),
        "agg_head": agg_head,
        "output_dir": output_dir,
        "git_rev": _git_rev(),
    }
    # Every resolved protocol value, the expected column they were checked against, its source,
    # `protocol_ok` and any deviations -- recorded whether or not the check passed (Requirement 8.6).
    manifest.update(protocol.to_dict())
    return manifest


def write_manifest(
    protocol: ProtocolRecord,
    agg_weight: float,
    cfg: Any,
    in_dim: Optional[int] = None,
    out_dim: Optional[int] = None,
    output_dir: Optional[str] = None,
    head: Any = None,
) -> str:
    """Write ``agg_run_manifest.json`` next to the frozen ``logs.json`` (Requirements 2.5, 8.6).

    Returns the absolute path written. Called *before* ``plan.planning_main``, so a failure here
    costs no evaluation time and is allowed to propagate: the resolved Agg_Weight, the horizon
    regime and the resolved protocol fields are what make an arm attributable after the fact, and an
    unlabelled
    run directory is worse than an aborted one.
    """
    directory = os.path.abspath(output_dir if output_dir else os.getcwd())
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, MANIFEST_FILENAME)

    manifest = build_manifest(
        protocol=protocol,
        agg_weight=agg_weight,
        cfg=cfg,
        in_dim=in_dim,
        out_dim=out_dim,
        output_dir=directory,
        head=head,
    )
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    print(f"[plan_agg] wrote {path}", flush=True)
    return path


def delegate_to_plan(
    cfg: Any,
    config_name: str,
    agg_weight: float,
    protocol: ProtocolRecord,
) -> Any:
    """Publish, rewrite, delegate, restore (Requirements 2.1-2.3, 2.5, 2.7, 5.4, 7.4, 8.5).

    In order: load Agg_Head (before any reseed), build the wrapper's own ``cfg_dict``, rewrite its
    objective block, publish the head and start the recorder, rebind ``plan.PlanEvaluator``, write
    the manifest, hand the dict to the frozen ``plan.planning_main`` unchanged, and restore and
    flush in ``finally``.

    Requirement 2.7 is met by construction: ``logs.json``, ``plan_targets.pkl``, the videos and the
    PNGs are all written by frozen code into a ``plan_outputs_*`` directory, in the layout
    ``aggregate_results.py`` already parses. This wrapper only *adds* files
    (``agg_run_manifest.json``, ``agg_instrumentation.json``, ``agg_episode_outcomes.jsonl``).
    """
    import plan  # local: see load_agg_head_from_ckpt
    from utils import cfg_to_dict

    # The record must describe the setting this process was actually launched with, since the
    # per-setting expected column is chosen from it and the manifest is read as evidence of which
    # setting produced the numbers.
    if normalize_config_name(config_name) != protocol.config_name:
        raise ProtocolError(
            f"the resolved protocol describes config name {protocol.config_name!r} but this "
            f"process was launched with {config_name!r}; the per-setting expected column and the "
            f"manifest would disagree about which setting produced these numbers."
        )

    # 3. Requirements 2.4, 2.6. Before the reseed, before the dataset load, before the env spawn.
    head, in_dim, out_dim = load_agg_head_from_ckpt(cfg)

    # The wrapper's own dict. `plan.main` does exactly these two lines before `planning_main`.
    cfg_dict = cfg_to_dict(cfg)
    cfg_dict["wandb_logging"] = bool(cfg_dict.get("wandb_logging", True))

    # 6. Requirement 2.3, in the wrapper's own dict.
    frozen_target = rewrite_objective_block(cfg_dict, agg_weight)

    output_dir = os.path.abspath(str(cfg_dict.get("saved_folder") or os.getcwd()))

    # 4. Requirement 2.5, and the channel Agg_Head travels by: `hydra.utils.call` passes the config
    # block and nothing else, so the head cannot arrive as an argument.
    context = agg_objectives.AGG_CONTEXT
    context.publish(
        agg_head=head,
        agg_weight=agg_weight,
        opt_steps=protocol.opt_steps,
        output_dir=output_dir,
    )

    # Requirements 5.1-5.4. `opt_steps` is how the recorder recovers the optimizer step index by
    # counting invocations; the protocol checker has already pinned it to 100 on any run that gets
    # this far, so `None` means the record would be mislabelled and no record is better than a
    # wrong one.
    if protocol.opt_steps is None:
        print(
            "[plan_agg][warning] planner.sub_planner.opt_steps did not resolve to a number, so "
            "the Instrumentation_Record is disabled: the recorder recovers the optimizer step "
            "index by counting objective invocations and cannot label a step without it "
            "(Requirements 5.1-5.3).",
            flush=True,
        )
    else:
        context.start_instrumentation(
            objective_mode=protocol.resolved.get("objective.mode"),
        )

    # 5. Requirement 7.4. `plan.py` imports `PlanEvaluator` as a module-level name and constructs it
    # directly, so rebinding that attribute -- in this process, on the module this wrapper is
    # driving -- is what makes the per-episode success vectors observable. No file under
    # `planning/` is edited and no name inside `planning/` is rebound, so `plan.py`'s bytes and the
    # Scope_Guard assertion both still hold.
    original_evaluator = plan.PlanEvaluator
    plan.PlanEvaluator = agg_objectives.RecordingPlanEvaluator

    print(
        f"[plan_agg] objective._target_ {frozen_target!r} -> {OBJECTIVE_TARGET!r} "
        f"(agg_weight={agg_weight!r}, enabled={agg_weight > 0.0}); "
        f"plan.PlanEvaluator -> {plan.PlanEvaluator.__name__}",
        flush=True,
    )

    try:
        # 7. Requirements 2.5, 8.6.
        write_manifest(
            protocol=protocol,
            agg_weight=agg_weight,
            cfg=cfg_dict,
            in_dim=in_dim,
            out_dim=out_dim,
            output_dir=output_dir,
            head=head,
        )
        # 8. Requirement 2.1: the frozen entry, called with the wrapper's dict and nothing else.
        return plan.planning_main(cfg_dict)
    finally:
        # 9. Requirement 5.4. Restore first, so a failing flush cannot leave the rebind in place;
        # `flush_and_clear` never raises, and returns what it managed to write.
        plan.PlanEvaluator = original_evaluator
        summary = context.flush_and_clear()
        print(f"[plan_agg] records flushed: {summary}", flush=True)


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------


def _resolved_config_name() -> str:
    """The Hydra config name this process was launched with, normalized to its stem."""
    return normalize_config_name(HydraConfig.get().job.config_name)


def _main(cfg: Any) -> Any:
    """The entry body, separated from the decorator so it reads in execution order.

    Steps 1 and 2 -- the weight and the protocol -- are task 8.1 and are complete. Both run before
    anything is loaded: the weight first, because a rejected weight should cost seconds rather than
    a dataset load and a DINOv2 download.
    """
    with open_dict(cfg):
        cfg["saved_folder"] = os.getcwd()

    config_name = _resolved_config_name()

    # 1. Requirements 3.1, 3.4, 3.5. Absent `+agg_weight=<w>` this is the Baseline_Arm weight 0.
    agg_weight = agg_objectives.validate_agg_weight(_cfg_get(cfg, "agg_weight", 0.0))

    # 2. Requirements 8.1-8.4, 8.7. Raises before any load, naming field/expected/resolved.
    protocol = resolve_protocol(config_name, cfg)

    print(
        f"[plan_agg] feature={FEATURE_NAME} setting={protocol.setting} "
        f"config_name={protocol.config_name} horizon={protocol.horizon_regime} "
        f"(goal_H={protocol.goal_H}) agg_weight={agg_weight!r} "
        f"protocol_ok={protocol.ok} run_dir={cfg['saved_folder']}",
        flush=True,
    )

    # 3-9. Task 8.2.
    return delegate_to_plan(
        cfg=cfg,
        config_name=config_name,
        agg_weight=agg_weight,
        protocol=protocol,
    )


if hydra is not None:  # pragma: no cover - requires hydra, i.e. the pod

    # Written without `version_base`, exactly as `plan.main` is: `plan.planning_main` depends on the
    # cwd being the run directory, which is the Hydra-version-dependent `job.chdir` default that
    # `plan.main` already relies on. Matching the decorator keeps both entry points on the same
    # behaviour, and `conf` is the same config directory.
    @hydra.main(config_path="conf", config_name="plan_gd")
    def main(cfg: Any) -> Any:
        return _main(cfg)

else:  # pragma: no cover - the CPU dev box has no hydra

    def main(cfg: Any = None) -> Any:
        """Unavailable stand-in: this process has no ``hydra`` to build the entry point from.

        The protocol layer above -- :data:`PROTOCOL_EXPECTED`, :func:`resolve_protocol`,
        :func:`normalize_config_name` -- is fully usable without Hydra, which is the point of the
        guarded import. Only the entry point itself needs it.
        """
        raise ImportError(
            "plan_agg.main needs hydra (and omegaconf), which could not be imported here: "
            f"{_HYDRA_IMPORT_ERROR}. Planning runs on the pod, where requirements-plan.txt "
            "installs both; the protocol checker and the weight validation in this module are "
            "importable and testable without them."
        )


if __name__ == "__main__":
    main()
