"""Aggregated-space planning cost: constants, weight validation and head plumbing.

Feature: ``aggregated-space-planning-cost``. The planning objective becomes
``L_plan = L_spatial + w * L_agg``, where ``L_spatial`` is the existing patch-space squared goal
distance computed by the frozen :mod:`planning.objectives` and ``L_agg`` is the same squared goal
distance measured *after* the encoder's aggregation head (Agg_Head,
``models/dino.py::DinoV2Encoder.agg`` with ``agg_type: mlp``).

This file is reached from two directions and must stay importable from both:

* ``plan_agg.py`` imports it to publish Agg_Head and to rewrite its own
  ``cfg_dict["objective"]["_target_"]``;
* every property test in this feature imports it, on CPU, **without** a checkpoint, without CUDA,
  without a dataset and without ``omegaconf``. Nothing at module scope may load, download or
  resolve anything.

``planning.objectives`` is imported **read-only**: this module calls
:func:`planning.objectives.create_objective_fn` and rebinds no name inside that module, so the
frozen per-frame coefficients and staged dispatch are reused rather than copied
(Requirement 4.7, Property 8).

Implemented here (task 2.1): the constants, :func:`validate_agg_weight`, the :data:`AGG_CONTEXT`
holder, :func:`extract_agg_head` and :func:`_apply_head`. Task 4.1 adds
:func:`create_agg_objective_fn`, the combined objective ``L_plan = L_spatial + w * L_agg``.
Task 5.1 adds :class:`AggInstrumentation`, the Instrumentation_Record recorder. Task 6.1 adds the
per-episode outcome sink (``AGG_CONTEXT.record_episodes`` / ``flush_and_clear``) and
``RecordingPlanEvaluator``, whose base class is resolved **lazily** so this module keeps importing
on a box without the planning stack's runtime dependencies.

Task 7.1 adds :func:`select_agg_weight`, which turns the sweep's rows into the Candidate_Arm weight
(and the sweep curve, the tie record and the boundary flag the write-up needs), and
:func:`paired_counts`, the per-episode Paired_Comparison over two ``output_final`` outcome vectors.

Requirements: 1.1-1.10, 2.4, 2.6, 3.2, 3.4, 3.5, 4.1, 4.4, 4.7, 5.1-5.6, 6.1, 6.4-6.6, 7.4, 11.4
"""

from __future__ import annotations

import json
import math
import numbers
import os
import re
import weakref
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn

# Read-only import of the frozen factory. Referenced through the module object rather than
# copied into a module-level name of our own, so the single call site below is unambiguous and
# Property 8's identity check has nothing here to be confused by.
from planning import objectives as frozen_objectives

__all__ = [
    "AGG_CONTEXT",
    "AGG_WEIGHT_MAX",
    "AGG_WEIGHT_MIN",
    "AggInstrumentation",
    "AggWeightError",
    "EPISODE_OUTCOMES_FILENAME",
    "INSTRUMENTATION_FILENAME",
    "REPORTED_OUTCOME_FILENAME",
    "REPORTING_SEEDS",
    "RUN_DIR_TEMPLATES",
    "RecordingEvaluatorMixin",
    "RecordingPlanEvaluator",
    "SHIPPED_RUN_DIR_TEMPLATES",
    "STEP_100_SEMANTICS",
    "SWEEP_GRID",
    "SWEEP_GRID_MAX",
    "SWEEP_GRID_MIN",
    "SweepPoint",
    "SweepSelection",
    "SweepSelectionError",
    "TUNING_SEED",
    "UNDEFINED_RATIO",
    "create_agg_objective_fn",
    "extract_agg_head",
    "make_recording_plan_evaluator",
    "paired_counts",
    "run_dir_override",
    "select_agg_weight",
    "step_semantics",
    "validate_agg_weight",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Agg_Weight values swept at the Tuning_Seed, open-loop only (Requirement 6.1). Spans 0.01 to 3
#: and includes the paper-literal 0.1 from ``tab:long_horizon``.
SWEEP_GRID: Tuple[float, ...] = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)

#: The ends of the grid. A selection landing on either one is an **unbracketed** optimum: the sweep
#: cannot tell an interior peak from a curve still rising at the edge of the grid.
#: :func:`select_agg_weight` reports which end was hit, so the write-up does not have to recompute
#: it and the boundary branch cannot be missed.
SWEEP_GRID_MIN: float = min(SWEEP_GRID)
SWEEP_GRID_MAX: float = max(SWEEP_GRID)

#: Held-out data-sampling seed. Agg_Weight is selected here and nowhere else (Requirement 6.6).
TUNING_SEED: int = 400

#: Data-sampling seeds used only for the confirmation run and the Acceptance_Gate.
REPORTING_SEEDS: Tuple[int, ...] = (100, 200, 300)

#: Accepted Agg_Weight domain: the closed interval ``[0, 3]`` (Requirement 3.4).
AGG_WEIGHT_MIN: float = 0.0
AGG_WEIGHT_MAX: float = 3.0

#: The Instrumentation_Record file, written next to the frozen ``logs.json`` (Requirement 5.4).
INSTRUMENTATION_FILENAME: str = "agg_instrumentation.json"

#: The per-episode outcome file, written next to the frozen ``logs.json`` (Requirements 7.4, 11.4).
#: One JSON object per line, one line per ``eval_actions`` call.
EPISODE_OUTCOMES_FILENAME: str = "agg_episode_outcomes.jsonl"

#: The ``filename`` of the row that carries the *reported* per-episode outcome vector.
#:
#: ``PlanWorkspace.perform_planning`` calls ``eval_actions(..., filename="output_final")`` once, at
#: the end, and turns its ``logs`` into the ``final_eval/success_rate`` this feature's numbers are
#: read from. Under MPC the intermediate ``plan{iter}`` rows are recorded too -- they show how the
#: success set grows across outer iterations, which is useful -- but they are **not** the reported
#: result, and the Paired_Comparison must use this row alone.
REPORTED_OUTCOME_FILENAME: str = "output_final"

#: The ``ratio`` value recorded when L_spatial is exactly ``0.0`` (Requirement 5.6).
UNDEFINED_RATIO: str = "undefined"


def step_semantics(opt_steps: int) -> str:
    """The ``step_100_semantics`` string, stated plainly rather than glossed.

    ``planning/gd.py`` runs ``for i in tqdm(range(self.opt_steps))`` and evaluates the objective
    once at the top of each iteration, *before* that iteration's ``optimizer.step()``. With
    ``opt_steps: 100`` the loop indices are therefore ``0..99``: there are exactly 100 objective
    evaluations, index ``0`` is formed before any Adam update and index ``99`` is formed after 99
    updates. **No evaluation exists after the 100th update** -- producing one would need an extra
    forward pass that only frozen code could trigger.

    So Requirement 5.2's "optimizer step 100" is recorded as the 100th *evaluation*
    (``step_index 99``, ``updates_applied 99``), and every record carries both fields so the
    number can never be read as something it is not.
    """
    steps = int(opt_steps)
    last = steps - 1
    return (
        f"step_index {last} is the {steps}th objective evaluation of a plan() call, formed after "
        f"{last} Adam updates; planning/gd.py performs no evaluation after the {steps}th update. "
        f"With opt_steps: {steps} the loop indices are 0-{last}, so Requirement 5.2's "
        f'"step {steps}" is the {steps}th evaluation (step_index {last}) and not a '
        f"{steps + 1}th evaluation that does not exist."
    )


#: The semantics string for the Evaluation_Protocol's pinned ``opt_steps: 100``.
STEP_100_SEMANTICS: str = step_semantics(100)

# --- Run-directory templates ------------------------------------------------
#
# The shipped ``hydra.run.dir`` template carries neither the seed nor the weight, so all seven
# sweep arms would append their line to ONE ``logs.json`` and ``aggregate_results.py`` would
# average seven weights into a single cell without ever erroring. The override below replaces the
# ``${ckpt_base_path}`` component with ``aggw${agg_weight}`` and changes nothing else, so every
# component the aggregator parses survives: the ``plan_outputs_gd`` / ``plan_outputs_gd_mpc``
# prefix (which is how it recovers planner and setting), the
# ``${replace_slash:${model_name}}_gH..._${goal_source}`` component (env and curvature flavour) and
# the trailing ``obj${objective.mode}_init${planner.sub_planner.sample_type}`` token (mode).
#
# One source of truth: the shell driver and ``tests/test_agg_run_dir_separation.py`` (task 3.1)
# both read these strings. In the driver the value MUST be single-quoted so ``${...}`` reaches
# Hydra rather than being expanded (to empty) by bash.
#
# ``conf/plan_gd.yaml`` and ``conf/plan_gd_mpc.yaml`` remain the source of truth for the shipped
# form; the mirror below exists so the override is expressible as a one-component substitution and
# so task 3.1 can resolve both forms side by side.

_RUN_DIR_TAIL = (
    "${replace_slash:${model_name}}_gH${goal_H}_${goal_source}/"
    "${ckpt_base_path}_gd_lr${planner.sub_planner.lr}"
    "_an${planner.sub_planner.action_noise}"
    "_opt${planner.sub_planner.opt_steps}"
    "_obj${objective.mode}"
    "_init${planner.sub_planner.sample_type}"
)

#: The ``hydra.run.dir`` values as shipped in ``conf/plan_gd.yaml`` / ``conf/plan_gd_mpc.yaml``.
SHIPPED_RUN_DIR_TEMPLATES = {
    "plan_gd": "plan_outputs_gd/" + _RUN_DIR_TAIL,
    "plan_gd_mpc": "plan_outputs_gd_mpc/" + _RUN_DIR_TAIL,
}

#: The component the override substitutes in place of ``${ckpt_base_path}``.
AGG_WEIGHT_RUN_DIR_COMPONENT = "aggw${agg_weight}"

#: ``hydra.run.dir`` override values, keyed by Hydra config name (Requirements 2.7, 6.7, 7.3).
RUN_DIR_TEMPLATES = {
    name: template.replace("${ckpt_base_path}", AGG_WEIGHT_RUN_DIR_COMPONENT)
    for name, template in SHIPPED_RUN_DIR_TEMPLATES.items()
}


def run_dir_override(config_name: str) -> str:
    """Return the full ``hydra.run.dir='<template>'`` override token for ``config_name``.

    ``config_name`` is the Hydra config name (``plan_gd`` for the open-loop setting,
    ``plan_gd_mpc`` for MPC), i.e. what ``HydraConfig.get().job.config_name`` reports.

    **The value is wrapped in single quotes, and they are not decoration.** Hydra parses the
    right-hand side of a command-line override with its own ANTLR grammar, *before* OmegaConf
    ever sees it, and that grammar rejects an unquoted ``}``::

        hydra.run.dir=plan_outputs_gd/${replace_slash:${model_name}}_...
        -> OverrideParseException: mismatched input '}' expecting <EOF>

    Quoting makes the grammar read the whole value as one string literal; the ``${...}``
    interpolations survive untouched and OmegaConf resolves them later, at composition time.
    Verified against the pod's Hydra 1.2.0 with ``OverridesParser``: the unquoted form raises,
    the quoted form returns the template verbatim.

    This is a **second, independent** quoting requirement from the bash one. ``run_ccr_pilot.sh``
    already protected these interpolations from *shell* expansion — an expanded one arrives
    empty and silently truncates the directory — and
    ``tests/test_agg_run_dir_separation.py::test_run_dir_overrides_are_single_quoted_in_shell_drivers``
    guards that layer. Nothing guarded the Hydra layer, which is why task 11.1 failed on its
    first ever invocation with this hook. ``test_run_dir_override_parses_under_hydra_grammar``
    now covers it.
    """
    try:
        template = RUN_DIR_TEMPLATES[config_name]
    except KeyError:
        known = ", ".join(sorted(RUN_DIR_TEMPLATES))
        raise ValueError(
            f"no aggregated-space hydra.run.dir template for config name {config_name!r}; "
            f"known config names are: {known}"
        ) from None
    if "'" in template:
        # Unreachable with the module constants above, and checked anyway: a single quote in the
        # template would terminate the quoted value early and hand Hydra a different override.
        raise ValueError(
            f"the hydra.run.dir template for {config_name!r} contains a single quote, which "
            f"cannot be passed through Hydra's quoted-value grammar: {template!r}"
        )
    return f"hydra.run.dir='{template}'"


# ---------------------------------------------------------------------------
# Agg_Weight validation (Requirements 3.4, 3.5)
# ---------------------------------------------------------------------------


class AggWeightError(ValueError, TypeError):
    """Raised for any rejected Agg_Weight.

    Deliberately a subclass of both :class:`ValueError` and :class:`TypeError`: a weight can be
    rejected because it is out of domain (``-1``, ``nan``, ``4``) or because it is not a number at
    all (``"0.1"``, ``None``), and a caller should not have to know which of the two idiomatic
    exception types this module chose in order to catch it.
    """


def validate_agg_weight(value: Any) -> float:
    """Validate Agg_Weight and return it as a ``float``.

    Accepts any finite real number in the closed interval ``[0, 3]`` (Requirement 3.4). Rejects
    negatives, ``nan``, ``inf``, values above 3 and non-numeric input, naming the rejected value
    and the accepted interval (Requirement 3.5).

    ``bool`` is rejected: ``True`` silently meaning ``1.0`` is a footgun in a value that decides
    whether the Baseline_Arm is a valid control.
    """
    interval = f"[{AGG_WEIGHT_MIN:g}, {AGG_WEIGHT_MAX:g}]"

    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise AggWeightError(
            f"agg_weight must be a real number in the closed interval {interval}; received "
            f"{value!r} of type {type(value).__name__}, which is not a number."
        )

    weight = float(value)

    if not math.isfinite(weight):
        raise AggWeightError(
            f"agg_weight must be a finite number in the closed interval {interval}; received "
            f"{value!r}, which is not finite."
        )

    if weight < AGG_WEIGHT_MIN or weight > AGG_WEIGHT_MAX:
        raise AggWeightError(
            f"agg_weight must lie in the closed interval {interval}; received {value!r}. "
            f"The sweep grid is {SWEEP_GRID} and 0 is the Baseline_Arm."
        )

    return weight


# ---------------------------------------------------------------------------
# The late-binding Agg_Head holder
# ---------------------------------------------------------------------------

_REQUIRE_MESSAGE = (
    "agg_objectives.AGG_CONTEXT holds no Agg_Head. create_agg_objective_fn was reached without "
    "plan_agg.py publishing one first -- most likely plan.py was launched directly with "
    "objective._target_ pointing at this module. Launch plan_agg.py instead."
)

# ``planning/mpc.py`` names its per-iteration evaluations ``f"plan{self.iter}"`` and
# ``planning/gd.py`` its (unreachable, ``eval_every == -1``) intermediates
# ``f"{logging_prefix}_output_{i+1}"`` with ``logging_prefix == f"plan_{iter}"``. Both forms are
# matched here so the outer MPC iteration can be read off the filename the frozen caller chose.
# This is a read of a frozen naming convention, so it degrades rather than fails: an unrecognized
# filename inherits the last plan index seen, which is 0 in the open-loop setting where no
# ``plan{N}`` evaluation happens at all.
_PLAN_CALL_RE = re.compile(r"^plan_?(\d+)")


def _coerce_successes(successes: Any) -> List[bool]:
    """Coerce a ``successes`` vector into a JSON-serializable list of Python ``bool``.

    ``PlanEvaluator._compute_rollout_metrics`` returns ``env.eval_state(...)["success"]``, a numpy
    boolean array of length ``n_evals``. Torch tensors and plain sequences are accepted too, so the
    sink does not depend on which environment produced the vector.

    Raises:
        TypeError: if the value is not a vector of per-episode outcomes at all -- a scalar, ``None``
            or an object array. Raised rather than guessed at, because a length-1 row silently
            standing in for 50 episodes would corrupt the Paired_Comparison rather than fail it.
            :meth:`_AggContext.record_episodes` converts this into a counted failure.
    """
    if isinstance(successes, torch.Tensor):
        values = successes.detach().cpu().reshape(-1).tolist()
        return [bool(value) for value in values]

    array = np.asarray(successes)
    if array.ndim == 0:
        raise TypeError(
            f"a per-episode success vector is required, but received the scalar "
            f"{successes!r} ({type(successes).__name__}), which carries no per-episode "
            f"information."
        )
    if array.dtype == object:
        raise TypeError(
            f"a per-episode success vector is required, but received an object-dtype array of "
            f"shape {array.shape}, whose elements are not boolean outcomes."
        )
    return [bool(value) for value in array.reshape(-1).tolist()]


@dataclass
class _AggContext:
    """Process-local holder for everything the objective factory cannot receive as an argument.

    ``plan.py`` builds its objective with ``hydra.utils.call(cfg_dict["objective"])`` and passes
    nothing else, so the config block is the only argument channel and it is resolved before any
    model exists. Agg_Head therefore travels by this holder instead: ``plan_agg.py`` publishes it
    before calling ``plan.planning_main``, and the frozen call order (seed -> dataset ->
    ``load_model`` -> ``PlanWorkspace.__init__`` -> ``hydra.utils.call``) guarantees the factory is
    reached strictly afterwards, in one process on one thread.
    """

    agg_head: Optional[nn.Module] = None
    agg_weight: float = 0.0
    opt_steps: Optional[int] = None
    output_dir: Optional[str] = None
    #: Set by task 5.1's ``AggInstrumentation``; typed loosely so this file stays independent of it.
    instrumentation: Any = field(default=None)

    # --- per-episode outcome sink (task 6.1, Requirements 7.4, 11.4) -------
    #: One row per ``eval_actions`` call, in call order.
    episode_outcomes: List[Dict[str, Any]] = field(default_factory=list)
    #: Rows already appended to ``agg_episode_outcomes.jsonl``.
    outcomes_written: int = 0
    #: The most recent ``plan{N}`` index seen, so ``output_final`` can name the call it follows.
    last_plan_call: int = 0
    #: Failed writes and unusable success vectors: counted, never raised.
    record_failures: int = 0
    record_failure_details: List[str] = field(default_factory=list)

    def publish(
        self,
        agg_head: nn.Module,
        agg_weight: Any = 0.0,
        opt_steps: Optional[int] = None,
        output_dir: Optional[str] = None,
        instrumentation: Any = None,
    ) -> "_AggContext":
        """Publish Agg_Head and the run's settings. Returns ``self`` for chaining."""
        if agg_head is None:
            raise ValueError("publish() requires an Agg_Head; received None.")
        self.agg_head = agg_head
        self.agg_weight = validate_agg_weight(agg_weight)
        self.opt_steps = None if opt_steps is None else int(opt_steps)
        self.output_dir = None if output_dir is None else str(output_dir)
        self.instrumentation = instrumentation
        return self

    def require(self) -> "_AggContext":
        """Return ``self``, or raise an actionable error if no Agg_Head was published."""
        if self.agg_head is None:
            raise RuntimeError(_REQUIRE_MESSAGE)
        return self

    def clear(self) -> None:
        """Drop every published value. Called from ``plan_agg.py``'s ``finally`` block."""
        self.agg_head = None
        self.agg_weight = 0.0
        self.opt_steps = None
        self.output_dir = None
        self.instrumentation = None
        self.episode_outcomes = []
        self.outcomes_written = 0
        self.last_plan_call = 0
        self.record_failures = 0
        self.record_failure_details = []

    # --- instrumentation (task 5.1) ---------------------------------------

    def start_instrumentation(
        self,
        objective_mode: Optional[str] = None,
        opt_steps: Optional[int] = None,
        path: Optional[str] = None,
    ) -> "AggInstrumentation":
        """Attach a fresh :class:`AggInstrumentation` and return it.

        ``opt_steps`` defaults to the published value (which the protocol checker has already
        pinned to 100) and ``path`` to ``<output_dir>/agg_instrumentation.json``. The objective
        reads :attr:`instrumentation` per call rather than capturing it at factory time, so this
        may be called before or after the objective is built.
        """
        steps = self.opt_steps if opt_steps is None else opt_steps
        if steps is None:
            raise ValueError(
                "start_instrumentation needs opt_steps: the recorder recovers the optimizer step "
                "index by counting objective invocations, so it cannot know where a plan() call "
                "ends without it. plan_agg.py publishes the resolved "
                "planner.sub_planner.opt_steps."
            )
        if path is None and self.output_dir is not None:
            path = os.path.join(self.output_dir, INSTRUMENTATION_FILENAME)
        self.instrumentation = AggInstrumentation(
            opt_steps=steps,
            agg_weight=self.agg_weight,
            path=path,
            objective_mode=objective_mode,
        )
        return self.instrumentation

    def flush_instrumentation(self) -> bool:
        """Write the Instrumentation_Record, if one is attached. Never raises."""
        instrumentation = self.instrumentation
        if instrumentation is None:
            return False
        return instrumentation.write()

    # --- per-episode outcome sink (task 6.1, Requirements 7.4, 11.4) -------
    #
    # `plan.py` persists only means: `PlanEvaluator._compute_rollout_metrics` reduces `successes`
    # into `logs["success_rate"]` and the per-episode array is returned up the stack and dropped.
    # The per-episode videos that would encode the outcome are written for `n_plot_samples = 10`
    # only, and only when `decode_for_viz` is true, which the launcher sets false. Requirements 7.4
    # and 11.4 need the vectors themselves, so `RecordingPlanEvaluator` observes every
    # `eval_actions` call and lands the vector here.

    def outcomes_path(self, path: Optional[str] = None) -> Optional[str]:
        """Where the outcome rows are written, or ``None`` if nowhere is known yet.

        ``None`` is not an error: the rows stay in :attr:`episode_outcomes` and a later
        :meth:`flush_episode_outcomes` -- after ``output_dir`` is published, or with an explicit
        ``path`` -- writes every row that is still unwritten.
        """
        if path is not None:
            return str(path)
        if self.output_dir is None:
            return None
        return os.path.join(self.output_dir, EPISODE_OUTCOMES_FILENAME)

    def record_episodes(
        self,
        filename: Any,
        successes: Any,
        plan_call: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Record one ``eval_actions`` call's per-episode success vector.

        Appends ``{"filename", "plan_call", "n_evals", "successes"}`` to
        :attr:`episode_outcomes` and to ``agg_episode_outcomes.jsonl``. The row is written
        immediately rather than only at flush time, so a crash or a timeout part-way through a
        multi-iteration MPC evaluation still leaves the outcomes that were already measured on
        disk.

        ``plan_call`` defaults to the outer MPC iteration read off ``filename``: ``plan{N}`` gives
        ``N``, and any other name -- ``output_final``, ``output`` -- inherits the last ``plan{N}``
        index seen, which is ``0`` in the open-loop setting where no ``plan{N}`` evaluation occurs.

        Returns the row, or ``None`` if the success vector was unusable (counted in
        :attr:`record_failures`, never raised).

        Raises:
            OSError: if the append fails on the filesystem. Deliberately propagated: the caller
                (:class:`RecordingEvaluatorMixin`) turns it into
                :meth:`note_record_failure` so a bad write cannot lose a 15-minute evaluation, and
                keeping the raise here means a caller that *does* care can still see it. Every
                *other* write failure -- a path ``os.makedirs`` rejects outright, a value ``json``
                cannot serialize -- is counted here rather than raised, since no caller could do
                anything with it either.
        """
        try:
            vector = _coerce_successes(successes)
        except (TypeError, ValueError) as exc:
            self.note_record_failure(exc)
            return None

        name = "output" if filename is None else str(filename)
        if plan_call is None:
            plan_call = self._plan_call_from_filename(name)

        row: Dict[str, Any] = {
            "filename": name,
            "plan_call": int(plan_call),
            "n_evals": len(vector),
            "successes": vector,
        }
        self.episode_outcomes.append(row)
        try:
            self._append_outcome_rows()
        except OSError:
            raise
        except (TypeError, ValueError, RecursionError) as exc:
            self.note_record_failure(exc)
        return row

    def _plan_call_from_filename(self, name: str) -> int:
        match = _PLAN_CALL_RE.match(name)
        if match is not None:
            self.last_plan_call = int(match.group(1))
        return self.last_plan_call

    def reported_outcome(self) -> Optional[Dict[str, Any]]:
        """The row the reported success rate comes from, or ``None`` if it was never recorded.

        That is the last ``filename == "output_final"`` row: the evaluation
        ``PlanWorkspace.perform_planning`` turns into ``final_eval/success_rate``. MPC's
        intermediate ``plan{iter}`` rows are recorded but are not the reported result, so the
        Paired_Comparison reads this row and no other.
        """
        for row in reversed(self.episode_outcomes):
            if row["filename"] == REPORTED_OUTCOME_FILENAME:
                return row
        return None

    def _append_outcome_rows(self, path: Optional[str] = None) -> int:
        """Append every not-yet-written row as one JSON line each. May raise ``OSError``."""
        target = self.outcomes_path(path)
        if target is None:
            return 0
        pending = self.episode_outcomes[self.outcomes_written :]
        if not pending:
            return 0
        directory = os.path.dirname(os.path.abspath(target))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(target, "a", encoding="utf-8") as handle:
            for row in pending:
                handle.write(json.dumps(row) + "\n")
        self.outcomes_written += len(pending)
        return len(pending)

    def flush_episode_outcomes(self, path: Optional[str] = None) -> bool:
        """Write any rows still unwritten. Never raises; returns whether nothing is outstanding.

        Called from :meth:`flush_and_clear`, i.e. from ``plan_agg.py``'s ``finally`` block after a
        15-minute evaluation, so every failure mode is counted in :attr:`record_failures` and
        reported through the return value instead of raised.
        """
        try:
            self._append_outcome_rows(path)
        except (OSError, TypeError, ValueError) as exc:
            self.note_record_failure(exc)
            return False
        return self.outcomes_written == len(self.episode_outcomes)

    def note_record_failure(self, exc: BaseException) -> None:
        """Count a failed record. Never raises (Requirements 5.4, 7.4).

        Also forwarded to the attached :class:`AggInstrumentation`, so
        ``agg_instrumentation.json``'s ``record_failures`` is the single visible count of everything
        this feature failed to record during the run.
        """
        self.record_failures += 1
        self.record_failure_details.append(f"{type(exc).__name__}: {exc}")
        instrumentation = self.instrumentation
        if instrumentation is not None:
            try:
                instrumentation.note_failure(exc)
            except Exception:  # pragma: no cover - a recorder must never mask the run
                pass

    def flush_and_clear(self, clear: bool = True) -> Dict[str, Any]:
        """Write both records, then drop every published value. Never raises.

        This is what ``plan_agg.py``'s ``finally`` block calls: it wraps
        :meth:`flush_instrumentation` (Requirement 5.4) and :meth:`flush_episode_outcomes`
        (Requirements 7.4, 11.4) and then :meth:`clear`. The returned summary is read *before*
        clearing, so the caller can log what was written even though the counters are reset.
        """
        summary: Dict[str, Any] = {
            "instrumentation": self.flush_instrumentation(),
            "episode_outcomes": self.flush_episode_outcomes(),
            "outcome_rows": len(self.episode_outcomes),
            "outcome_rows_written": self.outcomes_written,
            "record_failures": self.record_failures,
            "record_failure_details": list(self.record_failure_details),
        }
        if clear:
            self.clear()
        return summary


#: The module-level singleton. One planning process, one Agg_Head.
AGG_CONTEXT = _AggContext()


# ---------------------------------------------------------------------------
# Agg_Head extraction (Requirements 2.4, 2.6)
# ---------------------------------------------------------------------------


def extract_agg_head(encoder: Any) -> Tuple[nn.Module, int, int]:
    """Extract Agg_Head from a checkpoint's encoder.

    Returns ``(head, in_dim, out_dim)`` where ``head`` composes the encoder's own ``agg_mlp`` and
    ``agg_post_norm`` modules -- the *same* module objects, so the parameters are bit-identical to
    the ones inside the planner's encoder and nothing is copied or re-initialized. Everything else
    on the encoder (the DINOv2 backbone, the projector) is dropped.

    Widths are read from ``_agg_mlp_in_dim`` / ``_agg_out_dim`` and never parsed out of the
    checkpoint directory name: the ``agg32`` token in
    ``pusht_aggmlpcos1e-1_agg32_projchannel_dim8_hw14_...`` is a literal in ``conf/train.yaml``'s
    run-dir template, not a resolved head width. For the Target_Cell the real values are
    ``in_dim = 196 * 8 = 1568`` and ``out_dim = 128``.

    Raises:
        ValueError: if ``agg_type`` is not ``"mlp"`` (naming the encountered value,
            Requirement 2.6), if the head modules are missing, or if the width attributes are
            absent or unusable.
    """
    agg_type = getattr(encoder, "agg_type", None)
    if agg_type != "mlp":
        raise ValueError(
            f"the aggregated-space planning cost requires an encoder with agg_type 'mlp', but "
            f"this checkpoint's encoder reports agg_type {agg_type!r}. Only the 'mlp' aggregation "
            f"head defines the 128-dimensional space L_agg is measured in ('mean' and 'flatten' "
            f"carry no learned head), so there is nothing to plan in here. Use the Target_Cell "
            f"checkpoint trained with agg_type: mlp."
        )

    missing = [name for name in ("agg_mlp", "agg_post_norm") if getattr(encoder, name, None) is None]
    if missing:
        raise ValueError(
            f"this checkpoint's encoder reports agg_type 'mlp' but is missing "
            f"{', '.join(missing)}, so Agg_Head cannot be extracted. The checkpoint's encoder and "
            f"its agg_type disagree; check that the checkpoint was written by a run with "
            f"agg_type: mlp."
        )

    in_dim = _read_width(encoder, "_agg_mlp_in_dim")
    out_dim = _read_width(encoder, "_agg_out_dim")

    # Same composition order as DinoV2Encoder.agg: agg_mlp then agg_post_norm. The Sequential
    # holds the encoder's own submodules, so this shares parameters rather than copying them.
    head = nn.Sequential(*encoder.agg_mlp, encoder.agg_post_norm)
    head.in_dim = in_dim
    head.out_dim = out_dim
    head.agg_type = "mlp"
    return head, in_dim, out_dim


def _read_width(encoder: Any, attr: str) -> int:
    """Read a positive integer width off the encoder, or explain why it could not be read."""
    value = getattr(encoder, attr, None)
    if value is None:
        raise ValueError(
            f"this checkpoint's encoder has no {attr}, so the Agg_Head width cannot be read from "
            f"the checkpoint. The width is deliberately not parsed out of the run-directory name "
            f"(the 'agg32' token there is a run-dir literal, not a head width), so this is fatal."
        )
    try:
        width = int(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"this checkpoint's encoder reports {attr}={value!r}, which is not an integer width."
        ) from None
    if width <= 0:
        raise ValueError(
            f"this checkpoint's encoder reports {attr}={value!r}; a head width must be positive."
        )
    return width


# ---------------------------------------------------------------------------
# Applying Agg_Head frame-wise (Requirements 1.8, 1.9)
# ---------------------------------------------------------------------------

# Heads already moved to a device/dtype and frozen, keyed weakly by the head object so a finished
# run's head is collectable. Value is ``((device, dtype), prepared_module)``; ``nn.Module.to`` is
# in-place and returns ``self``, so the cache is what keeps the move, ``eval()`` and
# ``requires_grad_(False)`` to the first call rather than every one of the 100 optimizer steps.
_PREPARED_HEADS: "weakref.WeakKeyDictionary[nn.Module, Tuple[Tuple[Any, Any], nn.Module]]" = (
    weakref.WeakKeyDictionary()
)


def _prepare_head(head: nn.Module, device: Any, dtype: Any) -> nn.Module:
    """Move ``head`` onto ``device``/``dtype`` once, put it in eval mode and freeze it.

    Freezing the parameters (Requirement 1.7) is belt-and-braces: the planner's optimizer holds
    only the action tensor, so the head could not be updated in any case. Gradients still flow to
    the head's *input*, which is what the planner needs.
    """
    cached = _PREPARED_HEADS.get(head)
    if cached is not None and cached[0] == (device, dtype):
        return cached[1]

    prepared = head.to(device=device, dtype=dtype) if dtype is not None else head.to(device=device)
    prepared.eval()
    for parameter in prepared.parameters():
        parameter.requires_grad_(False)

    try:
        _PREPARED_HEADS[head] = ((device, dtype), prepared)
    except TypeError:  # pragma: no cover - a non-weakrefable head just skips the cache
        pass
    return prepared


def _head_in_dim(head: nn.Module) -> Optional[int]:
    """The flattened per-frame width ``head`` accepts, or ``None`` if it accepts any width.

    Read from an explicit ``in_dim`` attribute when present (that is what
    :func:`extract_agg_head` records), otherwise from the first ``nn.Linear`` in the module.
    Parameter-free heads -- the identity-on-flattened-features variant Property 3 uses -- report
    ``None``, and no width check applies to them.
    """
    declared = getattr(head, "in_dim", None)
    if declared is not None:
        return int(declared)
    for module in head.modules():
        if isinstance(module, nn.Linear):
            return int(module.in_features)
    return None


def _apply_head(z_visual: torch.Tensor, head: nn.Module) -> torch.Tensor:
    """Apply Agg_Head frame-wise: ``(b, t, p, d) -> (b, t, out_dim)``.

    Mirrors ``VWorldModel.total_curvature``'s ``aggcos`` branch exactly -- ``reshape(b * t, p, d)``
    -> head -> ``reshape(b, t, -1)`` -- so the aggregation the planner sees is the same operation
    the curvature regularizer applied during training. The flattening step is
    ``x.contiguous().view(x.shape[0], -1)``, which is what ``DinoV2Encoder.agg`` itself performs
    before ``agg_mlp``.

    ``T`` is preserved, which is what lets the frozen per-frame coefficients and the staged
    dispatch (both functions of ``T`` alone) be reused unchanged for L_agg.

    Device and dtype are resolved lazily from the incoming tensor and cached (Requirement 1.8).

    Raises:
        ValueError: if the features do not flatten to the width Agg_Head requires
            (Requirement 1.9), naming the received shape, the flattened width it implies and the
            required width. Raised *before* the ``nn.Linear`` call, so a bare mat1/mat2 shape
            message can never surface in its place.
    """
    if not isinstance(z_visual, torch.Tensor):
        raise ValueError(
            f"Agg_Head expects the predicted visual features as a tensor; received "
            f"{type(z_visual).__name__}."
        )
    if z_visual.ndim < 3:
        raise ValueError(
            f"Agg_Head expects visual features shaped (B, T, patches, channels); received shape "
            f"{tuple(z_visual.shape)}, which has no per-frame feature axis to aggregate."
        )

    b, t = int(z_visual.shape[0]), int(z_visual.shape[1])
    trailing = tuple(int(s) for s in z_visual.shape[2:])
    flat_width = 1
    for size in trailing:
        flat_width *= size

    in_dim = _head_in_dim(head)
    if in_dim is not None and flat_width != in_dim:
        raise ValueError(
            f"Agg_Head cannot accept the predicted visual features: received shape "
            f"(B={b}, T={t}, {_describe_trailing(trailing)}), which flattens to {flat_width} "
            f"features per frame, but this checkpoint's aggregation head requires exactly "
            f"{in_dim} features per frame. The planner's encoder and the aggregation head "
            f"disagree; check that the checkpoint is the 14x14x8 projected-channel encoder, whose "
            f"head takes 196 patches x 8 channels = 1568 features."
        )

    dtype = z_visual.dtype if z_visual.is_floating_point() else None
    prepared = _prepare_head(head, z_visual.device, dtype)

    tokens = z_visual.reshape(b * t, *trailing)
    flat = tokens.contiguous().view(b * t, flat_width)
    aggregated = prepared(flat)
    return aggregated.reshape(b, t, -1)


def _describe_trailing(trailing: Tuple[int, ...]) -> str:
    """Name the trailing axes the way the planner's shapes read: patches and channels."""
    if len(trailing) == 2:
        return f"patches={trailing[0]}, channels={trailing[1]}"
    return f"features={trailing}"


# ---------------------------------------------------------------------------
# The frozen factory, called and not copied (Requirement 4.7)
# ---------------------------------------------------------------------------


def _create_frozen_objective_fn(alpha, base, mode: str = "last"):
    """The single call site into :mod:`planning.objectives`.

    Task 4.1 calls this twice per run: once with the configured ``alpha`` for L_spatial, and once
    with ``alpha=0`` on aggregated-space feature dicts for L_agg. Neither the per-frame
    coefficient vector nor the staged dispatch is reimplemented anywhere in this file -- both
    terms go through the one implementation in ``planning/objectives.py``, so they cannot drift
    from it. Nothing in that module is rebound here (Requirement 4.7, Property 8).
    """
    return frozen_objectives.create_objective_fn(alpha=alpha, base=base, mode=mode)


# ---------------------------------------------------------------------------
# The combined objective L_plan = L_spatial + w * L_agg
# (Requirements 1.1-1.10, 3.2)
# ---------------------------------------------------------------------------


def _agg_dicts(z_obs_pred: dict, z_obs_tgt: dict, head: nn.Module) -> Tuple[dict, dict]:
    """Map a pair of patch-space latent dicts into aggregated space, frame-wise.

    ``(B, T, p, d) -> (B, T, out_dim)`` for the prediction and ``(B, 1, p, d) ->
    (B, 1, out_dim)`` for the goal, so ``T`` is preserved. That is the whole reason the frozen
    per-frame coefficients (``[base**i for i in range(T)]``, normalized) and the frozen staged
    dispatch predicate (``step < T - 1``) apply to L_agg unchanged: both depend on ``T``, ``base``
    and the device alone, and all three are identical between the two spaces.

    The proprio entries are zero-valued. The L_agg delegate is built with ``alpha=0``, so the
    proprio term enters as ``0 * loss_proprio``; zeros make ``loss_proprio`` exactly ``0.0`` as
    well, so ``0 * 0.0 == 0.0`` and ``x + 0.0`` is bit-exact for every float a mean of squares can
    produce. L_agg is therefore exactly the aggregated-space visual term, with no proprio residue
    and no second reduction rule of our own.
    """
    a_pred = _apply_head(z_obs_pred["visual"], head)
    a_tgt = _apply_head(z_obs_tgt["visual"], head)
    p_pred = a_pred.new_zeros(a_pred.shape[0], a_pred.shape[1], 1)
    p_tgt = a_tgt.new_zeros(a_tgt.shape[0], a_tgt.shape[1], 1)
    return (
        {"visual": a_pred, "proprio": p_pred},
        {"visual": a_tgt, "proprio": p_tgt},
    )


def create_agg_objective_fn(
    alpha,
    base,
    mode: str = "last",
    agg_weight: Any = None,
    agg_head: Optional[nn.Module] = None,
    **kwargs: Any,
):
    """Build the planning objective ``L_plan = L_spatial + agg_weight * L_agg``.

    This is the Hydra ``_target_`` ``plan_agg.py`` writes into its own ``cfg_dict["objective"]``.
    The signature is a **superset** of :func:`planning.objectives.create_objective_fn`'s and ends
    in ``**kwargs``, so the unmodified objective block (which carries ``alpha``, ``base`` and
    ``mode``) resolves against either factory, and a future config key cannot turn into a
    ``TypeError`` inside frozen ``hydra.utils.call``.

    Args:
        alpha: proprio weight for L_spatial, passed through to the frozen factory untouched.
        base: per-frame coefficient base, passed through to the frozen factory untouched. Used by
            **both** terms, so the coefficients cannot differ between them.
        mode: ``last`` / ``all`` / ``staged``, passed through to the frozen factory untouched.
        agg_weight: Agg_Weight. ``None`` (the default) means "use the weight
            ``plan_agg.py`` published through :data:`AGG_CONTEXT`", which is ``0.0`` when nothing
            was published. Validated here through :func:`validate_agg_weight`.
        agg_head: Agg_Head. ``None`` (the production path) means "use the head
            :data:`AGG_CONTEXT` holds"; ``AGG_CONTEXT.require()`` then raises the actionable
            "launch ``plan_agg.py`` instead" error if none was published. Passing a head
            explicitly is the seam the property tests use, so they need no global state.

    Returns:
        A callable ``(z_obs_pred, z_obs_tgt, step=None) -> Tensor`` of shape ``(B,)``, on the
        device and in the dtype of ``z_obs_pred["visual"]`` (Requirements 1.1, 1.4, 1.8).

    Neither the per-frame coefficient vector nor the staged dispatch is reimplemented: two
    callables are built from the frozen factory, one with the configured ``alpha`` for L_spatial
    and one with ``alpha=0`` for L_agg, and both terms therefore go through the single
    implementation in ``planning/objectives.py`` (Requirements 1.2, 1.5, 1.6).

    Both terms are raw: L_agg is the raw mean-squared aggregated-space distance and L_spatial is
    the frozen callable's own value, with no rescaling or normalization of either relative to the
    other (Requirement 1.10).
    """
    # Two callables, one implementation. `alpha=0` pins the proprio channel of L_agg to nothing;
    # `base` and `mode` are shared, so the coefficients and the stage predicate are literally the
    # same code operating on the same T.
    spatial_fn = _create_frozen_objective_fn(alpha=alpha, base=base, mode=mode)
    agg_fn = _create_frozen_objective_fn(alpha=0, base=base, mode=mode)

    context = AGG_CONTEXT
    if agg_head is None:
        head = context.require().agg_head
    else:
        head = agg_head

    weight = validate_agg_weight(context.agg_weight if agg_weight is None else agg_weight)

    # Resolved ONCE, here, from the validated float -- the same shape as
    # `VWorldModel.__init__`'s `self.ccr = self.lambda_cf > 0`. No per-step `== 0.0` comparison on
    # a tensor exists anywhere below.
    enabled = weight > 0.0

    def agg_loss_fn(z_obs_pred: dict, z_obs_tgt: dict, step=None) -> torch.Tensor:
        """Raw mean-squared L_agg, shape ``(B,)``, through the frozen reduction."""
        agg_pred, agg_tgt = _agg_dicts(z_obs_pred, z_obs_tgt, head)
        return agg_fn(agg_pred, agg_tgt, step=step)

    def objective(z_obs_pred: dict, z_obs_tgt: dict, step=None) -> torch.Tensor:
        loss_spatial = spatial_fn(z_obs_pred, z_obs_tgt, step=step)

        # Read per call rather than captured at factory time, so a recorder attached after the
        # objective was built is still seen. `None` until task 5.1 publishes one.
        instrumentation = context.instrumentation
        record = instrumentation is not None and instrumentation.should_record()

        try:
            if not enabled:
                # Requirement 3.2, by identity rather than by arithmetic. `loss_spatial + 0 *
                # loss_agg` is NOT bitwise safe: `0 * inf` and `0 * nan` are both `nan` and would
                # poison the sum. So this path performs no tensor operation on `loss_spatial` at
                # all and returns the object the frozen callable returned, which makes bitwise
                # equality an identity and puts a non-finite L_agg beyond reach of the result.
                if record:
                    # Requirement 5.5: the raw L_agg magnitude is still recorded at the recorded
                    # steps, under no_grad so it never joins the autograd graph.
                    with torch.no_grad():
                        instrumentation.log(
                            step, loss_spatial, agg_loss_fn(z_obs_pred, z_obs_tgt, step=step)
                        )
                return loss_spatial

            loss_agg = agg_loss_fn(z_obs_pred, z_obs_tgt, step=step)
            if record:
                with torch.no_grad():
                    instrumentation.log(step, loss_spatial, loss_agg)
            return loss_spatial + weight * loss_agg
        finally:
            if instrumentation is not None:
                instrumentation.advance()

    # Introspection seams: the run manifest records these and the property tests read them rather
    # than reconstructing the configuration they were built from.
    objective.alpha = alpha
    objective.base = base
    objective.mode = mode
    objective.agg_weight = weight
    objective.enabled = enabled
    objective.agg_head = head
    objective.spatial_fn = spatial_fn
    objective.agg_fn = agg_fn
    objective.agg_loss_fn = agg_loss_fn
    return objective


# ---------------------------------------------------------------------------
# Instrumentation of both loss components (Requirements 5.1-5.6)
# ---------------------------------------------------------------------------


class _Unset:
    """Sentinel distinct from ``None``, because ``None`` is a *valid* ``step`` argument."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid only
        return "<unset>"


_UNSET = _Unset()

#: What a recorded ``ratio`` field can be: a number, or the string ``"undefined"``.
RatioValue = Union[float, str]


def _batch_mean(value: Any) -> float:
    """The batch-mean magnitude of a loss term, as a plain ``float``.

    Accepts the ``(B,)`` tensor the objective produces, a 0-dim tensor, or a plain number.
    Detached and cast to float32 before the mean so a half-precision run does not lose the
    magnitude to rounding, and so nothing here can join the autograd graph.
    """
    if isinstance(value, torch.Tensor):
        with torch.no_grad():
            return float(value.detach().float().mean().item())
    if isinstance(value, bool):
        raise TypeError("a loss magnitude cannot be a bool")
    if isinstance(value, numbers.Real):
        return float(value)
    raise TypeError(
        f"cannot take a batch mean of {type(value).__name__}; expected a torch.Tensor or a number."
    )


def _json_safe_step(step: Any) -> Any:
    """Coerce the frozen ``step`` argument into something ``json.dump`` accepts.

    ``step`` is ``None`` in the open-loop (`last`) setting and the *outer* MPC iteration index
    under `staged`. Anything else is stringified rather than dropped: the field exists to be read
    by a human interpreting the record, and a write that fails on an exotic type would cost the
    whole record.
    """
    if step is None:
        return None
    if isinstance(step, bool):
        return int(step)
    if isinstance(step, torch.Tensor):
        if step.numel() == 1:
            item = step.item()
            return int(item) if float(item).is_integer() else float(item)
        return str(tuple(step.shape))
    if isinstance(step, numbers.Integral):
        return int(step)
    if isinstance(step, numbers.Real):
        return float(step)
    return str(step)


class AggInstrumentation:
    """Records both loss components at the first and last optimizer step of every plan call.

    **How the optimizer step index is recovered.** The objective callable never receives it: in
    `last` mode the ``step`` argument is ``None``, and under MPC it is the *outer* MPC iteration,
    not the inner optimizer step. But ``planning/gd.py`` calls the objective exactly once per
    inner iteration, in order, at the top of the loop body, and ``eval_every`` is ``-1`` in both
    shipped planner configs so the early ``break`` is unreachable and the loop always runs
    ``opt_steps`` times. The **call index therefore is the optimizer step index**, and this
    recorder counts its own invocations: :meth:`should_record` fires at ``step_index == 0`` and
    ``step_index == opt_steps - 1``, and :meth:`advance` -- called at the end of *every* objective
    invocation -- rolls over to the next ``plan_call``.

    That reasoning is a read of frozen code, so task 5.2 pins it against the real ``GDPlanner``
    rather than leaving it asserted here.

    **What "step 100" means.** See :func:`step_semantics`: with ``opt_steps: 100`` the indices are
    ``0..99``, so Requirement 5.2's step 100 is the 100th *evaluation* (``step_index 99``, formed
    after 99 Adam updates), and every record carries ``step_index`` and ``updates_applied`` so the
    number cannot be misread.

    **Nothing here can perturb the run.** Magnitudes arrive already detached (the caller logs
    under ``torch.no_grad()``, and :func:`_batch_mean` detaches again), no tensor is created that
    any computation reads, no RNG is consumed, and every file write is failure-counted rather than
    raised -- a bad write must not lose a 15-minute evaluation.

    Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6
    """

    def __init__(
        self,
        opt_steps: int,
        agg_weight: Any = 0.0,
        path: Optional[str] = None,
        objective_mode: Optional[str] = None,
    ):
        steps = int(opt_steps)
        if steps <= 0:
            raise ValueError(
                f"opt_steps must be a positive integer; received {opt_steps!r}. The recorder "
                f"recovers the optimizer step index by counting objective invocations, so it "
                f"cannot know where a plan() call ends without it."
            )
        self.opt_steps: int = steps
        self.agg_weight: float = validate_agg_weight(agg_weight)
        self.path: Optional[str] = None if path is None else str(path)
        self.objective_mode: Optional[str] = objective_mode

        self.records: List[Dict[str, Any]] = []
        #: Failed writes, counted and never raised (Requirement 5.4's error-handling row).
        self.record_failures: int = 0
        self.failures: List[str] = []
        #: True if the frozen ``step`` argument changed inside one counted plan call.
        self.step_boundary_mismatch: bool = False
        self.step_boundary_details: List[Dict[str, Any]] = []

        self._i: int = 0
        self._plan_call: int = 0
        self._plan_call_step: Any = _UNSET

    # --- counters ---------------------------------------------------------

    @property
    def step_index(self) -> int:
        """0-based optimizer step within the current ``plan()`` call."""
        return self._i

    @property
    def plan_call(self) -> int:
        """0 for open-loop; the MPC outer iteration otherwise."""
        return self._plan_call

    @property
    def updates_applied(self) -> int:
        """Adam updates already applied when the loss at :attr:`step_index` is formed."""
        return self._i

    def should_record(self) -> bool:
        """True at the first and the last optimizer step of the current plan call.

        Requirement 5.1 (step 0) and Requirement 5.2 (the final evaluation). At ``opt_steps == 1``
        the two coincide and a single record is emitted.
        """
        return self._i == 0 or self._i == self.opt_steps - 1

    def advance(self, step: Any = _UNSET) -> None:
        """Count one objective invocation, rolling over into the next plan call.

        Called from the objective's ``finally`` block, so a raising objective cannot desynchronize
        the count from the planner's loop. ``step`` is optional: passing it extends the
        constant-``step`` self-check to invocations that were not recorded.
        """
        if step is not _UNSET:
            self._observe_step(step)
        self._i += 1
        if self._i >= self.opt_steps:
            self._i = 0
            self._plan_call += 1
            self._plan_call_step = _UNSET

    # --- recording --------------------------------------------------------

    def log(self, step: Any, l_spatial: Any, l_agg: Any) -> Dict[str, Any]:
        """Record both batch-mean magnitudes and the effective ratio at the current step.

        Requirements 5.1, 5.2 (the two recorded steps), 5.3 (the ratio) and 5.5 (the raw L_agg is
        recorded even at Agg_Weight ``0``). Returns the record, which is also appended to
        :attr:`records`.
        """
        self._observe_step(step)
        spatial = _batch_mean(l_spatial)
        aggregated = _batch_mean(l_agg)
        record: Dict[str, Any] = {
            "plan_call": self._plan_call,
            "mpc_step_arg": _json_safe_step(step),
            "step_index": self._i,
            "updates_applied": self._i,
            "l_spatial": spatial,
            "l_agg": aggregated,
            "ratio": self.ratio(spatial, aggregated),
        }
        self.records.append(record)
        return record

    def ratio(self, l_spatial: float, l_agg: float) -> RatioValue:
        """``Agg_Weight * L_agg / L_spatial``, or ``"undefined"`` when L_spatial is ``0.0``.

        Requirement 5.6 makes the string case turn on L_spatial alone, so a zero denominator is
        reported rather than divided by. At ``Agg_Weight == 0`` the ratio is ``0.0`` and *not*
        ``"undefined"``: the term contributes exactly nothing to L_plan on that path, which is a
        statement about the weight and not about a missing value. That short-circuit also keeps a
        non-finite L_agg -- which the disabled path deliberately never lets near the returned loss
        -- from turning the recorded ratio into ``nan``; the raw magnitude is in ``l_agg`` either
        way.
        """
        if l_spatial == 0.0:
            return UNDEFINED_RATIO
        if self.agg_weight == 0.0:
            return 0.0
        return self.agg_weight * l_agg / l_spatial

    def _observe_step(self, step: Any) -> None:
        """Self-check: the frozen ``step`` argument must be constant within one plan call.

        A change mid-count means ``opt_steps`` disagrees with the planner's loop, i.e. the records
        are labelled with the wrong ``step_index``. Recorded as
        ``step_boundary_mismatch: true`` rather than raised or silently mislabelled: the run is
        worth more than the record, and a flagged record can be discarded knowingly.
        """
        safe = _json_safe_step(step)
        if self._plan_call_step is _UNSET:
            self._plan_call_step = safe
            return
        if safe != self._plan_call_step:
            self.step_boundary_mismatch = True
            self.step_boundary_details.append(
                {
                    "plan_call": self._plan_call,
                    "step_index": self._i,
                    "first_step_arg": self._plan_call_step,
                    "received_step_arg": safe,
                }
            )
            self._plan_call_step = safe

    def note_failure(self, exc: BaseException) -> None:
        """Count a failed write. Never raises (Requirement 5.4's error-handling row)."""
        self.record_failures += 1
        self.failures.append(f"{type(exc).__name__}: {exc}")

    # --- serialization ----------------------------------------------------

    def headline(self) -> Dict[str, Any]:
        """The ``plan_call == 0`` records: the first evaluation and the last.

        ``step_100`` keeps the design's field name at every ``opt_steps``; its payload carries the
        true ``step_index`` and ``updates_applied``, so the name cannot mislead.
        """
        return {
            "step_0": self._find_record(0, 0),
            "step_100": self._find_record(0, self.opt_steps - 1),
        }

    def _find_record(self, plan_call: int, step_index: int) -> Optional[Dict[str, Any]]:
        for record in self.records:
            if record["plan_call"] == plan_call and record["step_index"] == step_index:
                return dict(record)
        return None

    def to_dict(self) -> Dict[str, Any]:
        """The Instrumentation_Record as written to ``agg_instrumentation.json``."""
        payload: Dict[str, Any] = {
            "agg_weight": self.agg_weight,
            "opt_steps": self.opt_steps,
            "objective_mode": self.objective_mode,
            "step_100_semantics": step_semantics(self.opt_steps),
            "headline": self.headline(),
            "records": [dict(record) for record in self.records],
            "step_boundary_mismatch": self.step_boundary_mismatch,
            "record_failures": self.record_failures,
        }
        if self.step_boundary_details:
            payload["step_boundary_details"] = [dict(d) for d in self.step_boundary_details]
        if self.failures:
            payload["failures"] = list(self.failures)
        return payload

    def write(self, path: Optional[str] = None) -> bool:
        """Write the Instrumentation_Record. Returns whether the write succeeded.

        Requirement 5.4. Every failure mode -- unwritable directory, full disk, a value
        ``json`` cannot serialize -- is counted in :attr:`record_failures` and reported by the
        return value, never raised: this is called from ``plan_agg.py``'s ``finally`` block after a
        15-minute evaluation, and losing the evaluation to a bad write would be the worse outcome.

        ``nan`` and ``inf`` are written as ``NaN`` / ``Infinity`` (``json``'s default) rather than
        rejected, so a non-finite magnitude is recorded as what it was and round-trips through
        :func:`json.load`.
        """
        target = self.path if path is None else str(path)
        if target is None:
            return False
        try:
            directory = os.path.dirname(os.path.abspath(target))
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(target, "w", encoding="utf-8") as handle:
                json.dump(self.to_dict(), handle, indent=2)
                handle.write("\n")
        except (OSError, TypeError, ValueError, RecursionError) as exc:
            self.note_failure(exc)
            return False
        return True

    #: ``plan_agg.py``'s ``finally`` block reads better as a flush.
    flush = write

    def __repr__(self) -> str:  # pragma: no cover - debugging aid only
        return (
            f"AggInstrumentation(opt_steps={self.opt_steps}, agg_weight={self.agg_weight!r}, "
            f"plan_call={self._plan_call}, step_index={self._i}, records={len(self.records)}, "
            f"record_failures={self.record_failures})"
        )

# ---------------------------------------------------------------------------
# Sweep selection (Requirements 6.4, 6.5, 6.6)
# ---------------------------------------------------------------------------
#
# The sweep produces seven open-loop rows at the Tuning_Seed: the six Sweep_Grid arms plus the
# `aggw0` Baseline_Arm reference point (Requirement 6.3). `select_agg_weight` turns those rows into
# the Candidate_Arm weight and, in the same object, into everything the write-up has to report:
# the sweep curve, the tie record and whether the selection sits on an end of the grid.
#
# Three things the selection deliberately refuses to do:
#
# * **It never considers a Reporting_Seed row.** Requirement 6.6 is the whole point of holding seed
#   400 out, so rows at seeds 100/200/300 are filtered out before any comparison and listed in
#   `excluded_rows` so the filtering is auditable rather than asserted.
# * **It never selects the Baseline_Arm.** Requirement 6.4 selects "the Sweep_Grid value with the
#   highest open-loop success rate", and `0` is not a Sweep_Grid value -- it is the same-seed
#   reference point the curve is read against. A `w = 0` row therefore lands in the curve as the
#   baseline and is ineligible for selection.
# * **It never averages two rows for the same weight.** Two conflicting rows for one weight mean
#   the run directories collided (the hazard task 3.1 guards the template against), and averaging
#   them is exactly the silent failure that check exists to prevent. Conflicts raise.


class SweepSelectionError(ValueError):
    """Raised when the sweep rows cannot yield a Candidate_Arm weight."""


#: Row keys read for each field, in order of preference. Attribute access on non-dict rows works
#: too, so a caller may pass dicts, dataclasses or simple record objects.
_SWEEP_WEIGHT_KEYS = ("agg_weight", "weight", "w")
_SWEEP_RATE_KEYS = ("success_rate", "open_loop_success_rate", "final_success_rate", "rate")
_SWEEP_SEED_KEYS = ("seed", "data_seed", "cfg_seed")
_SWEEP_SETTING_KEYS = ("setting", "config_name")
_SWEEP_INSTRUMENTATION_KEYS = ("instrumentation", "instrumentation_record", "agg_instrumentation")

#: Setting labels that mean "open-loop" (Requirement 6.2: the sweep is open-loop only).
_OPEN_LOOP_LABELS = frozenset(
    {"ol", "open_loop", "open-loop", "openloop", "open loop", "plan_gd", "gd"}
)
#: Setting labels that mean "MPC". Recognized only so such a row can be excluded by name.
_MPC_LABELS = frozenset({"mpc", "plan_gd_mpc", "gd_mpc", "plan-gd-mpc"})

#: The boundary branch task 12.8 decided in advance, carried in the returned object so it is read
#: rather than improvised.
_BOUNDARY_NOTE = (
    "boundary selection: the selected Agg_Weight sits on an end of SWEEP_GRID, so the optimum is "
    "unbracketed -- the sweep cannot distinguish an interior peak from a curve still rising at the "
    "edge of the grid. The decision recorded in advance: carry this selection into the confirmation "
    "run as-is and report it as a boundary selection. Do not extend the grid on the spot; downward "
    "is outside SWEEP_GRID and upward is refused by validate_agg_weight (the accepted interval ends "
    "at 3), so any extension is a spec change needing the Requirement 11.7 recorded approval."
)


def _sweep_field(row: Any, keys: Sequence[str]) -> Any:
    """First present value among ``keys``, from a mapping or from attributes, else :data:`_UNSET`."""
    for key in keys:
        if isinstance(row, dict):
            if key in row:
                return row[key]
        else:
            value = getattr(row, key, _UNSET)
            if value is not _UNSET:
                return value
    return _UNSET


def _grid_member(weight: float) -> Optional[float]:
    """The :data:`SWEEP_GRID` value ``weight`` is, or ``None`` if it is not a grid value.

    Compared with a tolerance rather than by ``==`` because a weight that made the round trip
    through a run-directory name and back is a parsed decimal string, not necessarily the same
    double literal this module holds.
    """
    for candidate in SWEEP_GRID:
        if math.isclose(weight, candidate, rel_tol=1e-9, abs_tol=1e-12):
            return candidate
    return None


def _sweep_rate(value: Any, weight: float) -> float:
    """Coerce a row's open-loop success rate to a finite float.

    The unit is the caller's: percent (``76.0``) and fraction (``0.76``) both work, because the
    selection only ever *compares* rates and never combines them with anything. Mixing units
    across rows is the caller's error and is not detectable here, so the curve records every rate
    as given for the write-up to read.
    """
    if value is _UNSET or value is None:
        raise SweepSelectionError(
            f"the sweep row for agg_weight={weight!r} carries no open-loop success rate; looked "
            f"for {', '.join(_SWEEP_RATE_KEYS)}. Weight selection compares success rates "
            f"(Requirement 6.4), so a row without one cannot take part."
        )
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise SweepSelectionError(
            f"the sweep row for agg_weight={weight!r} reports a success rate of {value!r} "
            f"({type(value).__name__}), which is not a number."
        )
    rate = float(value)
    if not math.isfinite(rate):
        raise SweepSelectionError(
            f"the sweep row for agg_weight={weight!r} reports a success rate of {value!r}, which "
            f"is not finite. A nan would lose every comparison silently, so it is rejected here."
        )
    if rate < 0.0:
        raise SweepSelectionError(
            f"the sweep row for agg_weight={weight!r} reports a negative success rate {value!r}."
        )
    return rate


def _sweep_seed(row: Any, weight: float) -> int:
    """The row's data-sampling seed, which Requirement 6.6 makes load-bearing.

    Absent or unreadable is fatal rather than assumed either way: defaulting to "include" could let
    a Reporting_Seed row decide the weight, which is the exact failure Requirement 6.6 exists to
    prevent, and defaulting to "exclude" would quietly drop the arm that should have won.
    """
    value = _sweep_field(row, _SWEEP_SEED_KEYS)
    if value is _UNSET or value is None:
        raise SweepSelectionError(
            f"the sweep row for agg_weight={weight!r} carries no seed; looked for "
            f"{', '.join(_SWEEP_SEED_KEYS)}. Agg_Weight is selected at the Tuning_Seed "
            f"({TUNING_SEED}) alone and Reporting_Seeds {REPORTING_SEEDS} must contribute nothing "
            f"(Requirement 6.6), so a row of unknown provenance cannot be classified and is "
            f"refused rather than guessed at."
        )
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise SweepSelectionError(
            f"the sweep row for agg_weight={weight!r} reports seed {value!r} "
            f"({type(value).__name__}), which is not an integer seed."
        )
    seed = float(value)
    if not math.isfinite(seed) or not seed.is_integer():
        raise SweepSelectionError(
            f"the sweep row for agg_weight={weight!r} reports seed {value!r}, which is not an "
            f"integer seed."
        )
    return int(seed)


def _sweep_setting(row: Any, weight: float) -> Optional[str]:
    """``"ol"``, ``"mpc"`` or ``None`` when the row does not say.

    Requirement 6.2 makes the sweep open-loop only, so an MPC row is excluded rather than compared
    against open-loop rows. An unrecognized label raises: silently dropping the arm that would have
    won because its setting was spelled in a way this function did not know is worse than stopping.
    """
    value = _sweep_field(row, _SWEEP_SETTING_KEYS)
    if value is _UNSET or value is None:
        return None
    label = str(value).strip().lower()
    if label in _OPEN_LOOP_LABELS:
        return "ol"
    if label in _MPC_LABELS:
        return "mpc"
    raise SweepSelectionError(
        f"the sweep row for agg_weight={weight!r} reports setting {value!r}, which is neither an "
        f"open-loop label ({', '.join(sorted(_OPEN_LOOP_LABELS))}) nor an MPC label "
        f"({', '.join(sorted(_MPC_LABELS))}). The sweep is open-loop only (Requirement 6.2), so "
        f"an unclassifiable row is refused rather than compared or dropped."
    )


def _ratio_pair(instrumentation: Any) -> Tuple[Any, Any]:
    """The effective ratio at the first and last recorded step, or ``(None, None)``.

    Accepts an :class:`AggInstrumentation`, its :meth:`AggInstrumentation.to_dict` payload, a
    ``headline`` block on its own, or ``None``. Task 12.8 has to report
    ``Agg_Weight * L_agg / L_spatial`` at steps 0 and 100 for every arm, and that is what this
    lifts out of ``agg_instrumentation.json`` so the caller does not re-derive it.
    """
    if instrumentation is None:
        return (None, None)

    headline: Any = None
    if isinstance(instrumentation, AggInstrumentation):
        headline = instrumentation.headline()
    elif isinstance(instrumentation, dict):
        headline = instrumentation.get("headline", instrumentation)
    else:
        getter = getattr(instrumentation, "headline", None)
        if callable(getter):
            try:
                headline = getter()
            except Exception:  # pragma: no cover - a foreign object just reports no ratios
                headline = None

    if not isinstance(headline, dict):
        return (None, None)

    def _ratio_of(key: str) -> Any:
        record = headline.get(key)
        if isinstance(record, dict):
            return record.get("ratio")
        return None

    return (_ratio_of("step_0"), _ratio_of("step_100"))


@dataclass(frozen=True)
class SweepPoint:
    """One point of the sweep curve (Requirement 6.7).

    ``role`` is ``"candidate"`` for a Sweep_Grid arm eligible for selection, ``"baseline"`` for the
    ``w = 0`` Baseline_Arm reference point, and ``"off_grid"`` for any other weight -- an
    exploratory follow-up arm, which is recorded in the curve but cannot be selected because
    Requirement 6.4 selects a Sweep_Grid value.
    """

    agg_weight: float
    success_rate: float
    seed: int
    role: str
    instrumentation: Any = None
    ratio_step_0: Any = None
    ratio_step_100: Any = None

    @property
    def eligible(self) -> bool:
        """Whether this point can be selected as the Candidate_Arm weight."""
        return self.role == "candidate"

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable form. The instrumentation payload is included when it is a mapping."""
        payload: Dict[str, Any] = {
            "agg_weight": self.agg_weight,
            "success_rate": self.success_rate,
            "seed": self.seed,
            "role": self.role,
            "eligible": self.eligible,
            "ratio_step_0": self.ratio_step_0,
            "ratio_step_100": self.ratio_step_100,
        }
        if isinstance(self.instrumentation, AggInstrumentation):
            payload["instrumentation"] = self.instrumentation.to_dict()
        elif isinstance(self.instrumentation, dict):
            payload["instrumentation"] = self.instrumentation
        return payload


@dataclass(frozen=True)
class SweepSelection:
    """The outcome of weight selection: W_STAR plus everything the write-up must report.

    Requirements 6.4 (highest open-loop success rate at the Tuning_Seed), 6.5 (smallest tied weight,
    tie recorded), 6.6 (Reporting_Seeds contribute nothing) and 6.7 (the sweep curve).

    :attr:`boundary_kind` is the piece task 12.8 needs to be *visible* rather than recomputed: a
    selection of ``0.01`` or ``3.0`` is an unbracketed optimum, and :attr:`boundary_note` states the
    branch that was decided in advance for that case.
    """

    agg_weight: float
    success_rate: float
    tie: bool
    tied_weights: Tuple[float, ...]
    boundary_kind: Optional[str]
    curve: Tuple[SweepPoint, ...]
    baseline: Optional[SweepPoint]
    tuning_seed: int = TUNING_SEED
    excluded_rows: Tuple[Dict[str, Any], ...] = ()
    off_grid_weights: Tuple[float, ...] = ()

    @property
    def boundary(self) -> bool:
        """True when the selection sits on an end of :data:`SWEEP_GRID`."""
        return self.boundary_kind is not None

    #: The optimum is unbracketed exactly when the selection is a boundary selection.
    @property
    def unbracketed(self) -> bool:
        return self.boundary

    @property
    def boundary_note(self) -> Optional[str]:
        """The pre-decided boundary branch, or ``None`` for an interior selection."""
        return _BOUNDARY_NOTE if self.boundary else None

    @property
    def baseline_success_rate(self) -> Optional[float]:
        """The same-seed Baseline_Arm rate the curve is read against (Requirement 6.3)."""
        return None if self.baseline is None else self.baseline.success_rate

    @property
    def margin_over_baseline(self) -> Optional[float]:
        """Selected rate minus the Baseline_Arm rate, in whatever unit the rows used."""
        if self.baseline is None:
            return None
        return self.success_rate - self.baseline.success_rate

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable form, shaped for the sweep record and the Negative_Result_Record."""
        return {
            "agg_weight": self.agg_weight,
            "success_rate": self.success_rate,
            "tie": self.tie,
            "tied_weights": list(self.tied_weights),
            "boundary": self.boundary,
            "boundary_kind": self.boundary_kind,
            "boundary_note": self.boundary_note,
            "unbracketed": self.unbracketed,
            "tuning_seed": self.tuning_seed,
            "sweep_grid": list(SWEEP_GRID),
            "baseline_success_rate": self.baseline_success_rate,
            "margin_over_baseline": self.margin_over_baseline,
            "curve": [point.to_dict() for point in self.curve],
            "excluded_rows": [dict(row) for row in self.excluded_rows],
            "off_grid_weights": list(self.off_grid_weights),
            "selection_basis": (
                f"highest open-loop success rate at the Tuning_Seed ({self.tuning_seed}) alone; "
                f"smallest weight on a tie. Reporting seeds {REPORTING_SEEDS} contributed nothing, "
                f"so the confirmation run is the first time they see this weight."
            ),
        }

    def __str__(self) -> str:  # pragma: no cover - reporting aid
        parts = [f"W_STAR={self.agg_weight:g} at {self.success_rate:g}"]
        if self.tie:
            parts.append(
                "tie on "
                + ", ".join(f"{w:g}" for w in self.tied_weights)
                + " broken by taking the smallest"
            )
        if self.boundary:
            parts.append(f"{self.boundary_kind} boundary selection (unbracketed)")
        return "; ".join(parts)


def select_agg_weight(rows: Iterable[Any]) -> SweepSelection:
    """Select the Candidate_Arm Agg_Weight from the sweep's rows.

    Args:
        rows: the sweep rows, as mappings or as objects with the same field names. Each row needs a
            weight (``agg_weight`` / ``weight`` / ``w``), an open-loop success rate
            (``success_rate`` / ``open_loop_success_rate`` / ``final_success_rate`` / ``rate``) and
            a ``seed``. ``setting`` (``ol`` / ``mpc``) and ``instrumentation`` are optional; the
            instrumentation payload is carried into the curve and its step-0 / step-100 ratios are
            lifted out for the record.

    Returns:
        A :class:`SweepSelection`: the selected weight and its rate, the tie record, the
        boundary/unbracketed flag, the full sweep curve including the Baseline_Arm point, and the
        list of rows that were excluded and why.

    Raises:
        SweepSelectionError: if no eligible Sweep_Grid row at the Tuning_Seed is present, if a row
            is unreadable, or if two rows give the same weight conflicting success rates.
        AggWeightError: if a row's weight is outside the accepted ``[0, 3]`` interval.

    The rules, all three of them load-bearing:

    * **Requirement 6.4** -- the highest open-loop success rate at the Tuning_Seed wins, among
      Sweep_Grid values only. ``w = 0`` is the Baseline_Arm reference point (Requirement 6.3), not a
      candidate.
    * **Requirement 6.5** -- a tie is broken by taking the *smallest* tied weight, and the tie is
      recorded in :attr:`SweepSelection.tie` / :attr:`SweepSelection.tied_weights`.
    * **Requirement 6.6** -- only rows at :data:`TUNING_SEED` are considered. Rows at
      :data:`REPORTING_SEEDS` (and at any other seed) are dropped before any comparison and listed
      in :attr:`SweepSelection.excluded_rows`, so seeds 100/200/300 cannot influence the choice no
      matter what they contain.
    """
    candidates: Dict[float, SweepPoint] = {}
    baseline: Optional[SweepPoint] = None
    off_grid: List[SweepPoint] = []
    excluded: List[Dict[str, Any]] = []
    seen_rates: Dict[float, float] = {}

    for index, row in enumerate(rows):
        raw_weight = _sweep_field(row, _SWEEP_WEIGHT_KEYS)
        if raw_weight is _UNSET or raw_weight is None:
            raise SweepSelectionError(
                f"sweep row {index} carries no Agg_Weight; looked for "
                f"{', '.join(_SWEEP_WEIGHT_KEYS)}."
            )
        weight = validate_agg_weight(raw_weight)

        seed = _sweep_seed(row, weight)
        setting = _sweep_setting(row, weight)

        if seed != TUNING_SEED:
            # Requirement 6.6, enforced before any rate is even read: a Reporting_Seed row has no
            # path to the comparison below.
            excluded.append(
                {
                    "index": index,
                    "agg_weight": weight,
                    "seed": seed,
                    "reason": (
                        f"seed {seed} is not the Tuning_Seed {TUNING_SEED}"
                        + (
                            " (Reporting_Seed: contributes nothing to selection, Requirement 6.6)"
                            if seed in REPORTING_SEEDS
                            else ""
                        )
                    ),
                }
            )
            continue

        if setting == "mpc":
            excluded.append(
                {
                    "index": index,
                    "agg_weight": weight,
                    "seed": seed,
                    "reason": "MPC row: the sweep is open-loop only (Requirement 6.2)",
                }
            )
            continue

        rate = _sweep_rate(_sweep_field(row, _SWEEP_RATE_KEYS), weight)
        instrumentation = _sweep_field(row, _SWEEP_INSTRUMENTATION_KEYS)
        if instrumentation is _UNSET:
            instrumentation = None
        ratio_0, ratio_100 = _ratio_pair(instrumentation)

        grid_weight = _grid_member(weight)
        if grid_weight is not None:
            role, key = "candidate", grid_weight
        elif weight == 0.0:
            role, key = "baseline", 0.0
        else:
            role, key = "off_grid", weight

        previous = seen_rates.get(key)
        if previous is not None and previous != rate:
            raise SweepSelectionError(
                f"two sweep rows give agg_weight={key:g} at seed {TUNING_SEED} different open-loop "
                f"success rates ({previous!r} and {rate!r}). They are not averaged: two rows for "
                f"one arm means the arms' run directories collided, which is exactly what the "
                f"weight-keyed hydra.run.dir override exists to prevent. Check that every arm wrote "
                f"its own logs.json before selecting a weight."
            )
        seen_rates[key] = rate

        point = SweepPoint(
            agg_weight=grid_weight if grid_weight is not None else weight,
            success_rate=rate,
            seed=seed,
            role=role,
            instrumentation=instrumentation,
            ratio_step_0=ratio_0,
            ratio_step_100=ratio_100,
        )
        if role == "candidate":
            candidates[key] = point
        elif role == "baseline":
            baseline = point
        else:
            off_grid.append(point)

    if not candidates:
        raise SweepSelectionError(
            f"no Sweep_Grid arm at the Tuning_Seed ({TUNING_SEED}) was supplied, so there is "
            f"nothing to select from. Requirement 6.4 selects among the Sweep_Grid values "
            f"{SWEEP_GRID}; agg_weight 0 is the Baseline_Arm reference point (Requirement 6.3) and "
            f"is never selected, and {len(excluded)} row(s) were excluded by seed or setting."
        )

    best_rate = max(point.success_rate for point in candidates.values())
    tied = tuple(
        sorted(weight for weight, point in candidates.items() if point.success_rate == best_rate)
    )
    # Requirement 6.5: smallest tied weight, and the tie itself is recorded rather than dissolved.
    selected = tied[0]

    boundary_kind: Optional[str] = None
    if math.isclose(selected, SWEEP_GRID_MIN, rel_tol=1e-9, abs_tol=1e-12):
        boundary_kind = "lower"
    elif math.isclose(selected, SWEEP_GRID_MAX, rel_tol=1e-9, abs_tol=1e-12):
        boundary_kind = "upper"

    curve_points = list(candidates.values()) + off_grid
    if baseline is not None:
        curve_points.append(baseline)
    curve_points.sort(key=lambda point: point.agg_weight)

    return SweepSelection(
        agg_weight=selected,
        success_rate=best_rate,
        tie=len(tied) > 1,
        tied_weights=tied,
        boundary_kind=boundary_kind,
        curve=tuple(curve_points),
        baseline=baseline,
        tuning_seed=TUNING_SEED,
        excluded_rows=tuple(excluded),
        off_grid_weights=tuple(sorted(point.agg_weight for point in off_grid)),
    )


# ---------------------------------------------------------------------------
# The Paired_Comparison (Requirement 11.4)
# ---------------------------------------------------------------------------


def _paired_vector(value: Any, name: str) -> List[bool]:
    """Coerce one arm's per-episode outcome vector to a list of ``bool``.

    Accepts what the pieces upstream actually hold: a list of ``bool`` (an
    ``agg_episode_outcomes.jsonl`` row's ``successes``), the row itself (so
    ``paired_counts(candidate_ctx.reported_outcome(), baseline_row)`` reads naturally), a numpy
    boolean array (what ``env.eval_state(...)["success"]`` is) or a torch tensor.

    Non-boolean elements are rejected rather than truthiness-tested: a partial success score
    silently becoming ``True`` would corrupt the Paired_Comparison instead of failing it.
    """
    if isinstance(value, dict):
        if "successes" not in value:
            raise ValueError(
                f"{name} was given a mapping with no 'successes' key (keys: "
                f"{sorted(value)}). Pass either the per-episode vector or an "
                f"agg_episode_outcomes.jsonl row, whose vector lives under 'successes'."
            )
        value = value["successes"]

    if value is None:
        raise ValueError(
            f"{name} is None; a per-episode outcome vector is required. If the run's "
            f"'{REPORTED_OUTCOME_FILENAME}' row is missing from {EPISODE_OUTCOMES_FILENAME}, the "
            f"Paired_Comparison cannot be computed for that seed."
        )

    try:
        raw = _coerce_successes_strict(value, name)
    except TypeError as exc:
        raise ValueError(str(exc)) from None
    return raw


def _coerce_successes_strict(value: Any, name: str) -> List[bool]:
    """Like :func:`_coerce_successes`, but every element must be a boolean outcome."""
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            raise TypeError(
                f"{name} is a 0-dim tensor, which carries no per-episode information; a vector of "
                f"per-episode outcomes is required."
            )
        elements: List[Any] = value.detach().cpu().reshape(-1).tolist()
    else:
        array = np.asarray(value)
        if array.ndim == 0:
            raise TypeError(
                f"{name} is the scalar {value!r} ({type(value).__name__}), which carries no "
                f"per-episode information; a vector of per-episode outcomes is required."
            )
        if array.dtype == object:
            raise TypeError(
                f"{name} is an object-dtype array of shape {array.shape}, whose elements are not "
                f"boolean outcomes."
            )
        elements = array.reshape(-1).tolist()

    outcomes: List[bool] = []
    for position, element in enumerate(elements):
        if isinstance(element, bool):
            outcomes.append(element)
            continue
        if isinstance(element, numbers.Real) and float(element) in (0.0, 1.0):
            outcomes.append(bool(element))
            continue
        raise TypeError(
            f"{name}[{position}] is {element!r}, which is not a boolean episode outcome. The "
            f"Paired_Comparison counts episodes solved and not solved, so only True/False (or "
            f"0/1) are accepted -- a partial score is not silently rounded into a success."
        )
    return outcomes


def paired_counts(candidate: Any, baseline: Any) -> Dict[str, Any]:
    """The Paired_Comparison between the Candidate_Arm and the Baseline_Arm (Requirement 11.4).

    Args:
        candidate: the Candidate_Arm's per-episode outcome vector, or the
            ``agg_episode_outcomes.jsonl`` row holding it (``AGG_CONTEXT.reported_outcome()``
            returns exactly such a row, and ``filename == "output_final"`` is the reported one).
        baseline: the Baseline_Arm's vector for the **same** seed, in the same episode order.

    Returns:
        ``{"n", "candidate_only", "baseline_only", "matching", "both_success", "both_failure",
        "candidate_total", "baseline_total"}``. ``candidate_only + baseline_only + matching == n``
        always, since the three cases partition the episodes, and
        ``matching == both_success + both_failure``.

    Raises:
        ValueError: if the two vectors differ in length, or if either is not a vector of boolean
            outcomes.

    The comparison is exact rather than statistical: ``plan.py`` seeds episode sampling from
    ``cfg.seed`` and the pod is bitwise deterministic, so both arms at one seed drew *the same*
    episodes in the same order, and index ``i`` is the same initial state and goal in both vectors
    (Requirement 9.6). That is the entire reason a per-episode count means anything here, and it is
    also why a length mismatch is fatal: two vectors of different lengths are not the same episode
    set, so no alignment of them would be the Paired_Comparison.
    """
    candidate_vector = _paired_vector(candidate, "candidate")
    baseline_vector = _paired_vector(baseline, "baseline")

    if len(candidate_vector) != len(baseline_vector):
        raise ValueError(
            f"the Paired_Comparison needs two vectors over the same episodes, but candidate has "
            f"{len(candidate_vector)} episode(s) and baseline has {len(baseline_vector)}. Both arms "
            f"are evaluated at the same cfg.seed with n_evals 50, so a length mismatch means the "
            f"two vectors are not the same episode set -- most likely one arm's "
            f"'{REPORTED_OUTCOME_FILENAME}' row came from a different seed, a different run "
            f"directory, or a truncated evaluation. Check the {EPISODE_OUTCOMES_FILENAME} rows "
            f"before pairing them."
        )

    both_success = 0
    both_failure = 0
    candidate_only = 0
    baseline_only = 0
    for won, base_won in zip(candidate_vector, baseline_vector):
        if won and base_won:
            both_success += 1
        elif won:
            candidate_only += 1
        elif base_won:
            baseline_only += 1
        else:
            both_failure += 1

    return {
        "n": len(candidate_vector),
        "candidate_only": candidate_only,
        "baseline_only": baseline_only,
        "matching": both_success + both_failure,
        "both_success": both_success,
        "both_failure": both_failure,
        "candidate_total": both_success + candidate_only,
        "baseline_total": both_success + baseline_only,
    }


# ---------------------------------------------------------------------------
# Per-episode outcome capture: the plan.PlanEvaluator rebind
# (Requirements 7.4, 11.4, 4.4)
# ---------------------------------------------------------------------------
#
# This is a DECISION the design took, not a derivation, so it is written out here rather than left
# implicit in the code:
#
# `plan.py` persists only means. `PlanEvaluator._compute_rollout_metrics` reduces the per-episode
# `successes` array into `logs["success_rate"]`; the array itself is returned up the stack and
# dropped. The per-episode videos that would encode each outcome are written for
# `n_plot_samples = 10` only, and only when `decode_for_viz` is true, which the launcher sets false.
# Requirements 7.4 and 11.4 need the vectors -- the Paired_Comparison is per-episode by definition
# -- so something has to observe them as they go past.
#
# `plan.py` does `from planning.evaluator import PlanEvaluator` and constructs that module-level
# name directly, so `plan_agg.py` rebinds *that name in the `plan` module* to the subclass below,
# and restores it in its `finally` block.
#
# Why that is inside scope: no file under `planning/` is edited and no name inside `planning/` is
# rebound. The only rebind is `plan.PlanEvaluator`, an attribute of the module the wrapper is
# driving, in the wrapper's own process. `plan.py`'s bytes are untouched, so the Scope_Guard's
# `plan.py` byte-identity assertion (Requirement 4.4, task 1.1) still holds, and Requirement 4.7 --
# module-level names in `planning.objectives` left at their original values -- is untouched too.
#
# Why it is safe: the subclass is an observer. It adds no state that any computation reads, consumes
# no RNG, performs no tensor work, calls `super()` with the arguments it received, and returns the
# object `super()` returned -- the identical tuple, not a copy. Control flow and numerics are
# therefore those of the base class exactly, which is what keeps Requirement 3.3 and the exactness
# of the Paired_Comparison intact. Property 11 (task 6.2) is the automated guard on that claim.


class RecordingEvaluatorMixin:
    """Records each ``eval_actions`` call's per-episode success vector, and changes nothing else.

    Mixed in *before* the base evaluator, so :meth:`eval_actions` delegates through ``super()``.
    Kept separate from the concrete subclass so it can be composed over any base -- the property
    test builds it over a stand-in evaluator, which is what lets the guard on this rebind run on a
    CPU box with no simulator, no checkpoint and no planning-stack dependencies.
    """

    def eval_actions(
        self,
        actions,
        action_len=None,
        filename: str = "output",
        save_video: bool = False,
    ):
        """Delegate, record ``result[1]``, return the delegate's own object.

        The return value is the identical object ``super().eval_actions`` produced: it is neither
        rebuilt nor copied, so a caller cannot tell this class apart from its base by what it gets
        back (Property 11).

        Recording failures never reach the caller. ``OSError`` from the append, and a return value
        that has no ``result[1]`` to read, are both counted through
        :meth:`_AggContext.note_record_failure`: a bad write must not lose a 15-minute evaluation.
        """
        result = super().eval_actions(actions, action_len, filename, save_video)

        try:
            successes = result[1]
        except (TypeError, IndexError, KeyError) as exc:
            AGG_CONTEXT.note_record_failure(exc)
            return result

        try:
            AGG_CONTEXT.record_episodes(filename, successes)
        except OSError as exc:
            AGG_CONTEXT.note_record_failure(exc)

        return result


_RECORDING_DOC = """Read-only observer over ``planning.evaluator.PlanEvaluator``.

Delegates every ``eval_actions`` call to the base class, records the per-episode ``successes``
vector through ``AGG_CONTEXT.record_episodes`` and returns the base class's own return value.
``plan_agg.py`` rebinds ``plan.PlanEvaluator`` to this class for the duration of one run
(Requirements 7.4, 11.4).
"""


def _resolve_plan_evaluator() -> Type[Any]:
    """Import the frozen ``PlanEvaluator``, lazily and with an explanation if it is unavailable.

    The import is deferred rather than done at module scope because ``planning/evaluator.py`` pulls
    in the planning stack's runtime dependencies (``imageio``, ``torchvision``, ``einops``) and
    this module must stay importable on a CPU box without them -- every property test in this
    feature imports it there. The pod has them; the base class is only ever needed by
    ``plan_agg.py``, which runs on the pod.
    """
    try:
        from planning.evaluator import PlanEvaluator
    except ImportError as exc:
        raise ImportError(
            f"RecordingPlanEvaluator subclasses planning.evaluator.PlanEvaluator, which could not "
            f"be imported here: {exc}. That module pulls in the planning stack's runtime "
            f"dependencies (imageio, torchvision, einops), which the pod has and a bare CPU box "
            f"may not. agg_objectives.py itself stays importable without them -- the base class is "
            f"resolved lazily, on first access to RecordingPlanEvaluator, which only plan_agg.py "
            f"performs. To build the subclass over a different base (which is how the property "
            f"test exercises it without the planning stack), call "
            f"make_recording_plan_evaluator(base=...)."
        ) from exc
    return PlanEvaluator


_DEFAULT_RECORDING_EVALUATOR: Optional[Type[Any]] = None


def make_recording_plan_evaluator(base: Optional[Type[Any]] = None) -> Type[Any]:
    """Build the recording evaluator subclass over ``base``.

    ``base`` defaults to the frozen ``planning.evaluator.PlanEvaluator``, imported on demand; the
    resulting class is cached, so ``agg_objectives.RecordingPlanEvaluator`` is one stable class
    object that ``plan_agg.py`` can rebind and restore and that ``isinstance`` can be used against.

    Passing an explicit ``base`` is the seam Property 11 uses: the guard on this rebind should not
    need a simulator, a checkpoint or the planning stack's dependencies in order to run.
    """
    global _DEFAULT_RECORDING_EVALUATOR

    if base is None:
        if _DEFAULT_RECORDING_EVALUATOR is None:
            _DEFAULT_RECORDING_EVALUATOR = make_recording_plan_evaluator(
                base=_resolve_plan_evaluator()
            )
        return _DEFAULT_RECORDING_EVALUATOR

    return type(
        "RecordingPlanEvaluator",
        (RecordingEvaluatorMixin, base),
        {"__doc__": _RECORDING_DOC, "__module__": __name__},
    )


def __getattr__(name: str) -> Any:
    """Resolve ``RecordingPlanEvaluator`` on first access (PEP 562).

    ``agg_objectives.RecordingPlanEvaluator`` and ``from agg_objectives import
    RecordingPlanEvaluator`` both work and both return the cached subclass, but nothing is imported
    from ``planning/evaluator.py`` until one of them happens. That is what keeps this module
    importable on a box with no ``imageio`` while still letting ``plan_agg.py`` write the rebind as
    plainly as the design does.
    """
    if name == "RecordingPlanEvaluator":
        return make_recording_plan_evaluator()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    return sorted(list(globals()) + ["RecordingPlanEvaluator"])
