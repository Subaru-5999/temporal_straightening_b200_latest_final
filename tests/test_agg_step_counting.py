"""Pin the step-counting scheme against the real ``planning.gd.GDPlanner`` (task 5.2).

The whole instrumentation index in ``agg_objectives.AggInstrumentation`` rests on a **read of
frozen code**, restated here so the assertions below can be checked against it:

1. ``planning/gd.py`` calls ``self.objective_fn`` exactly **once per inner iteration**, at the top
   of the loop body, *before* that iteration's ``optimizer.step()``.
2. ``eval_every`` is ``-1`` in both entry configs, so the early ``break`` is unreachable and the
   loop always runs ``opt_steps`` times.
3. Nothing calls the objective outside that loop.

Together those make the **call index the optimizer step index**, which is the only reason
``AggInstrumentation`` can recover ``step_index`` by counting its own invocations. Nothing else in
the plan pins that read, and the recorder's own ``step``-argument self-check cannot detect desync
in the open-loop setting, where ``step`` is always ``None`` and therefore trivially constant.

So this module drives the **real** ``GDPlanner`` on CPU -- real ``plan()`` body, real Adam, real
cosine scheduler, real ``init_actions`` -- against a stub world model, a stub preprocessor/
evaluator and a counting objective, and asserts the three claims directly. The final test closes
the loop by attaching a **real** ``AggInstrumentation`` to a **real** ``create_agg_objective_fn``
objective and driving the real planner with it.

Gate, deliberately not optional: if the read above is wrong, the Instrumentation_Record silently
mislabels which optimizer step it describes, and the term-magnitude interpretation Requirement 5
exists to support -- the one that decides whether the Sweep_Grid brackets anything useful -- is
read off the wrong step.

Validates: Requirements 5.1, 5.2, 5.3

NOTE ON THE ``omegaconf`` SHIM: ``planning/gd.py`` does ``from utils import move_to_device``, and
root ``utils.py`` imports ``omegaconf`` at module scope. ``omegaconf`` is absent from the Windows
dev environment, so :func:`_import_gd_planner` installs a minimal stand-in in ``sys.modules`` for
the duration of that one import and removes it again immediately. The shim is never installed when
the real package is importable (i.e. never on the pod), and ``GDPlanner`` itself is the genuine
frozen class either way -- ``move_to_device`` does not touch ``OmegaConf``. This is a workaround
rather than a skip on purpose: skipping would leave the gate unenforced exactly where it is
cheapest to enforce.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agg_objectives import AGG_CONTEXT, AggInstrumentation, create_agg_objective_fn  # noqa: E402
from planning.objectives import create_objective_fn  # noqa: E402

from .conftest import make_agg_head  # noqa: E402


# ---------------------------------------------------------------------------
# Importing the frozen planner
# ---------------------------------------------------------------------------


def _omegaconf_stub() -> types.ModuleType:
    """A deliberately inert stand-in for ``omegaconf``.

    Exists only so ``import utils`` succeeds. Every attribute raises if actually *used*, so a
    future test that needs real config resolution fails loudly rather than against a fiction.
    """

    module = types.ModuleType("omegaconf")

    class _Unavailable:
        def __getattr__(self, name):
            raise RuntimeError(
                "tests/test_agg_step_counting.py installed a minimal omegaconf stand-in so that "
                f"planning/gd.py could be imported; OmegaConf.{name} is not available in it. "
                "Install omegaconf if real config resolution is needed."
            )

    module.OmegaConf = _Unavailable()
    module.DictConfig = dict
    module.ListConfig = list
    module.__doc__ = "Minimal test-only stand-in; see tests/test_agg_step_counting.py."
    return module


def _import_gd_planner():
    """Import and return the real, frozen :class:`planning.gd.GDPlanner`."""
    installed_stub = False
    if "omegaconf" not in sys.modules:
        try:  # pragma: no cover - depends on the environment, both branches are exercised
            import omegaconf  # noqa: F401
        except ModuleNotFoundError:
            sys.modules["omegaconf"] = _omegaconf_stub()
            installed_stub = True
    try:
        from planning.gd import GDPlanner
    finally:
        if installed_stub:
            # Leave sys.modules as it was found: other test modules `importorskip("omegaconf")`
            # and must keep skipping rather than tripping over this stand-in.
            sys.modules.pop("omegaconf", None)
    return GDPlanner


GDPlanner = _import_gd_planner()


def test_the_planner_under_test_is_the_frozen_class():
    """Guard against the shim quietly swapping in something that is not the real planner."""
    assert GDPlanner.__module__ == "planning.gd"
    assert Path(sys.modules["planning.gd"].__file__).resolve() == (
        _REPO_ROOT / "planning" / "gd.py"
    ).resolve()
    # The loop this whole module is about.
    assert "for i in tqdm(range(self.opt_steps))" in (
        _REPO_ROOT / "planning" / "gd.py"
    ).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Test doubles: everything GDPlanner.plan() reaches for, and nothing more
#
# Kept in this file rather than tests/conftest.py deliberately: another task is editing
# agg_objectives.py concurrently and conftest.py is shared, so a local double avoids a write
# collision. These are planner-shaped rather than aggregation-shaped, so they are not reusable by
# the other aggregated-space property tests anyway.
# ---------------------------------------------------------------------------


class PlannerCallLog:
    """Ordered event log for one or more ``GDPlanner.plan()`` calls.

    ``events`` is the whole point: it records *what was called in what order*, so "exactly once
    per inner iteration", "nothing before the loop" and "nothing after the loop" become one
    equality against an expected list rather than three separate counts.
    """

    def __init__(self):
        self.events: list[str] = []
        self.objective_calls: list[dict] = []
        self.rollout_acts: list[torch.Tensor] = []
        self.eval_calls: list[str] = []

    def event(self, name: str) -> None:
        self.events.append(name)

    @property
    def n_objective_calls(self) -> int:
        return len(self.objective_calls)


class StubPlanningWorldModel(nn.Module):
    """Stand-in world model exposing the two methods ``GDPlanner.plan`` calls.

    ``rollout`` is differentiable in ``act`` through a ``tanh`` and a fixed random projection, so
    the real Adam step actually moves the action tensor -- which is what lets the tests observe
    *how many updates had been applied* when a given objective call was formed.

    One ``nn.Parameter`` exists because ``BasePlanner.__init__`` reads
    ``next(wm.parameters()).device``. The projections are buffers, so ``BasePlanner``'s
    ``requires_grad = False`` sweep over parameters cannot cut the path to ``act``.
    """

    def __init__(
        self,
        log: PlannerCallLog,
        *,
        patches: int = 2,
        channels: int = 3,
        proprio_dim: int = 2,
        action_dim: int = 1,
        seed: int = 0,
    ):
        super().__init__()
        self.log = log
        self.patches = int(patches)
        self.channels = int(channels)
        self.proprio_dim = int(proprio_dim)
        self.action_dim = int(action_dim)

        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        self.register_buffer(
            "visual_proj",
            torch.randn(
                self.action_dim, self.patches * self.channels, generator=generator,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "proprio_proj",
            torch.randn(
                self.action_dim, self.proprio_dim, generator=generator, dtype=torch.float32
            ),
        )
        self.device_anchor = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def encode_obs(self, obs):
        self.log.event("encode_obs")
        return {"visual": obs["visual"], "proprio": obs["proprio"]}

    def rollout(self, obs_0, act):
        self.log.event("rollout")
        self.log.rollout_acts.append(act.detach().clone())

        b, h, _ = act.shape
        gated = torch.tanh(act)
        visual = torch.cat(
            [obs_0["visual"].reshape(b, 1, -1), gated @ self.visual_proj], dim=1
        ).reshape(b, h + 1, self.patches, self.channels)
        proprio = torch.cat(
            [obs_0["proprio"].reshape(b, 1, self.proprio_dim), gated @ self.proprio_proj], dim=1
        )
        z_obses = {"visual": visual, "proprio": proprio}
        return z_obses, visual.reshape(b, h + 1, -1)


class StubPreprocessor:
    """``transform_obs`` / ``normalize_actions``, the two methods the planner calls.

    ``transform_obs`` returns a shallow copy because frozen ``utils.move_to_device`` mutates the
    dict it is handed, and the tests compare the caller's observations afterwards.
    ``normalize_actions`` is the identity, which keeps the ``sample_type="zero"`` initialization
    exactly zero and therefore bitwise checkable.
    """

    def transform_obs(self, obs):
        return dict(obs)

    def normalize_actions(self, actions):
        return actions


class StubEvaluator:
    """``frameskip`` plus ``eval_actions``, returning the frozen 4-tuple contract.

    ``all_success=True`` is the interesting setting: it is what would trip
    ``if np.all(successes): break`` if the ``eval_every != -1`` guard ever stopped holding.
    """

    def __init__(self, log: PlannerCallLog, *, frameskip: int = 1, all_success: bool = True):
        self.log = log
        self.frameskip = int(frameskip)
        self.all_success = bool(all_success)

    def eval_actions(self, actions, action_len=None, filename="output", save_video=False):
        self.log.event(f"eval_actions:{filename}")
        self.log.eval_calls.append(filename)
        successes = np.full(actions.shape[0], self.all_success, dtype=bool)
        return {"success_rate": float(successes.mean())}, successes, None, None


class StubWandbRun:
    def __init__(self):
        self.logged: list[dict] = []

    def log(self, payload):
        self.logged.append(dict(payload))


class CountingObjective:
    """Counts invocations, then delegates to the unmodified frozen objective.

    Each record carries the action tensor the *paired* rollout was given, which is the observable
    proxy for "how many Adam updates had been applied when this loss was formed".
    """

    def __init__(self, log: PlannerCallLog, inner):
        self.log = log
        self.inner = inner

    def __call__(self, z_obs_pred, z_obs_tgt, step=None):
        self.log.event("objective")
        self.log.objective_calls.append(
            {
                "ordinal": len(self.log.objective_calls),
                "step": step,
                "n_rollouts_so_far": len(self.log.rollout_acts),
                "act": self.log.rollout_acts[-1] if self.log.rollout_acts else None,
            }
        )
        return self.inner(z_obs_pred, z_obs_tgt, step=step)


PATCHES, CHANNELS, PROPRIO_DIM = 2, 3, 2
AGG_IN_DIM = PATCHES * CHANNELS


def _make_obs(batch_size: int, *, seed: int = 0):
    """``(obs_0, obs_g)`` in the shapes the stub world model consumes."""
    generator = torch.Generator(device="cpu").manual_seed(int(seed))

    def _draw(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float32)

    obs_0 = {
        "visual": _draw(batch_size, 1, PATCHES, CHANNELS),
        "proprio": _draw(batch_size, 1, PROPRIO_DIM),
    }
    obs_g = {
        "visual": _draw(batch_size, 1, PATCHES, CHANNELS),
        "proprio": _draw(batch_size, 1, PROPRIO_DIM),
    }
    return obs_0, obs_g


def build_real_planner(
    *,
    opt_steps: int,
    horizon: int,
    batch_size: int = 2,
    action_dim: int = 1,
    frameskip: int = 1,
    eval_every: int = -1,
    all_success: bool = True,
    mode: str = "last",
    alpha: float = 1.0,
    base: float = 2.0,
    inner_objective=None,
    use_cosine_scheduler: bool = True,
    lr: float = 0.1,
    seed: int = 0,
):
    """Build a real :class:`GDPlanner` over the stubs. Returns ``(planner, log, obs_0, obs_g)``.

    ``action_noise=0`` and ``sample_type="zero"`` match the Evaluation_Protocol, and the zero
    initialization is what makes "formed before any update" a bitwise-checkable claim.

    ``inner_objective`` is the callable the counting wrapper delegates to; it defaults to the
    unmodified frozen objective, and the tie-back test passes the real
    ``create_agg_objective_fn`` result instead. Either way the planner sees one counting wrapper
    logging into the same ``PlannerCallLog`` the stub world model logs into, so the
    rollout/objective interleaving is one sequence.
    """
    log = PlannerCallLog()
    wm = StubPlanningWorldModel(
        log,
        patches=PATCHES,
        channels=CHANNELS,
        proprio_dim=PROPRIO_DIM,
        action_dim=action_dim,
        seed=seed,
    )
    if inner_objective is None:
        inner_objective = create_objective_fn(alpha=alpha, base=base, mode=mode)
    objective_fn = CountingObjective(log, inner_objective)
    planner = GDPlanner(
        horizon=horizon,
        action_noise=0.0,
        sample_type="zero",
        lr=lr,
        opt_steps=opt_steps,
        eval_every=eval_every,
        wm=wm,
        action_dim=action_dim,
        objective_fn=objective_fn,
        preprocessor=StubPreprocessor(),
        evaluator=StubEvaluator(log, frameskip=frameskip, all_success=all_success),
        wandb_run=StubWandbRun(),
        log_filename=None,  # never write: tests/test_scope_guard.py fails on stray root files
        use_cosine_scheduler=use_cosine_scheduler,
    )
    obs_0, obs_g = _make_obs(batch_size, seed=seed)
    return planner, log, obs_0, obs_g


def _expected_events(opt_steps: int) -> list[str]:
    """The exact event sequence one ``plan()`` call must produce.

    ``encode_obs`` once, before the loop, then ``opt_steps`` strictly alternating
    ``rollout``/``objective`` pairs, and nothing else. No ``eval_actions`` -- ``eval_every`` is
    ``-1``.
    """
    return ["encode_obs"] + ["rollout", "objective"] * opt_steps


# Driving the real planner does a forward, a backward and an Adam step per iteration, so the
# per-example cost is real. 40 examples over the generated ranges below still covers every
# opt_steps in 1..12 many times over.
PLANNER_SETTINGS = settings(
    max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)


# ---------------------------------------------------------------------------
# Claim 1: exactly opt_steps invocations per plan() call, in order, nothing outside the loop
# ---------------------------------------------------------------------------


@PLANNER_SETTINGS
@given(
    opt_steps=st.integers(min_value=1, max_value=12),
    horizon=st.integers(min_value=1, max_value=6),
    batch_size=st.integers(min_value=1, max_value=2),
    mode=st.sampled_from(("last", "all", "staged")),
    use_cosine_scheduler=st.booleans(),
    step_arg=st.one_of(st.none(), st.integers(min_value=0, max_value=6)),
)
def test_objective_invoked_exactly_opt_steps_times_per_plan_call(
    opt_steps, horizon, batch_size, mode, use_cosine_scheduler, step_arg
):
    """The call index is the optimizer step index: one call per inner iteration, in order.

    Validates: Requirements 5.1, 5.2
    """
    planner, log, obs_0, obs_g = build_real_planner(
        opt_steps=opt_steps,
        horizon=horizon,
        batch_size=batch_size,
        mode=mode,
        use_cosine_scheduler=use_cosine_scheduler,
    )

    planner.plan(obs_0=obs_0, obs_g=obs_g, step=step_arg)

    # One equality carries all three claims: the count, the strict interleaving with the rollout
    # that feeds it, and the absence of any call before the first iteration or after the last.
    assert log.events == _expected_events(opt_steps)
    assert log.n_objective_calls == opt_steps
    assert len(log.rollout_acts) == opt_steps

    # Ordinals are dense, 0-based and in order -- which is what makes them usable as step indices.
    assert [call["ordinal"] for call in log.objective_calls] == list(range(opt_steps))
    # Call k is paired with rollout k: the objective never runs twice on one rollout, nor skips one.
    assert [call["n_rollouts_so_far"] for call in log.objective_calls] == list(
        range(1, opt_steps + 1)
    )
    # The `step` argument the recorder's self-check watches is constant within the plan call, so
    # that check is sound but -- as task 5.2 notes -- cannot itself detect a desync.
    assert {call["step"] for call in log.objective_calls} == {step_arg}
    # eval_every == -1: the evaluator is never reached, so the early `break` cannot fire.
    assert log.eval_calls == []


@PLANNER_SETTINGS
@given(
    opt_steps=st.integers(min_value=1, max_value=8),
    horizon=st.integers(min_value=1, max_value=5),
    n_plan_calls=st.integers(min_value=2, max_value=3),
)
def test_each_plan_call_contributes_exactly_opt_steps_invocations(
    opt_steps, horizon, n_plan_calls
):
    """Repeated ``plan()`` calls concatenate cleanly: ``opt_steps`` invocations each, no leakage.

    This is the premise of ``AggInstrumentation.advance``'s rollover into the next ``plan_call``.

    Validates: Requirements 5.1, 5.2
    """
    planner, log, obs_0, obs_g = build_real_planner(opt_steps=opt_steps, horizon=horizon)

    for call_index in range(n_plan_calls):
        planner.plan(obs_0=obs_0, obs_g=obs_g, step=call_index)
        assert log.n_objective_calls == opt_steps * (call_index + 1)

    assert log.events == _expected_events(opt_steps) * n_plan_calls


# ---------------------------------------------------------------------------
# Claim 2: no call before the first update, none after the last
# ---------------------------------------------------------------------------


@PLANNER_SETTINGS
@given(
    opt_steps=st.integers(min_value=1, max_value=10),
    horizon=st.integers(min_value=1, max_value=5),
    use_cosine_scheduler=st.booleans(),
)
def test_no_call_before_the_first_update_and_none_after_the_last(
    opt_steps, horizon, use_cosine_scheduler
):
    """``updates_applied == step_index``, and the final update is followed by no evaluation.

    The action tensor is the observable: ``sample_type="zero"`` with an identity
    ``normalize_actions`` makes the initialization exactly zero, every Adam step moves it, and
    ``action_noise=0`` means nothing else does.

    * invocation 0 sees the untouched zero tensor -> **0** updates applied (Requirement 5.1).
    * invocation k sees a tensor that differs from invocation k-1's -> exactly one update between
      consecutive evaluations, so invocation ``opt_steps - 1`` sees ``opt_steps - 1`` updates
      (Requirement 5.2's "step 100" == ``step_index 99``).
    * the tensor ``plan()`` returns differs from the last invocation's -> one further update
      follows the final evaluation, and **no** evaluation follows that update. That is exactly the
      ``step_100_semantics`` claim: with ``opt_steps: 100`` there are 100 evaluations, not 101.

    Validates: Requirements 5.1, 5.2
    """
    planner, log, obs_0, obs_g = build_real_planner(
        opt_steps=opt_steps, horizon=horizon, use_cosine_scheduler=use_cosine_scheduler
    )

    final_actions, _ = planner.plan(obs_0=obs_0, obs_g=obs_g)

    snapshots = [call["act"] for call in log.objective_calls]
    assert len(snapshots) == opt_steps

    # Formed before any update: bitwise zero, the value init_actions produced.
    assert torch.equal(snapshots[0], torch.zeros_like(snapshots[0]))

    # Exactly one update between consecutive evaluations.
    for k in range(1, opt_steps):
        assert not torch.equal(snapshots[k - 1], snapshots[k]), (
            f"invocation {k} saw the same actions as invocation {k - 1}: the objective was "
            f"evaluated twice without an intervening optimizer step, so the call index is not "
            f"the optimizer step index."
        )
        assert (snapshots[k] - snapshots[k - 1]).abs().max().item() > 0.0

    # An update follows the final evaluation, and nothing evaluates the result of it.
    assert not torch.equal(final_actions.detach(), snapshots[-1]), (
        "plan() returned the actions the last objective call saw, so the final optimizer step "
        "did not move them -- the 'no evaluation after the last update' reading cannot be "
        "checked on this example."
    )
    assert log.events[-1] == "objective"
    assert log.events.count("objective") == opt_steps


# ---------------------------------------------------------------------------
# Claim 3: eval_every == -1 makes the early break unreachable
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("config_name", ("plan_gd", "plan_gd_mpc"))
def test_entry_configs_pin_eval_every_to_minus_one(config_name):
    """Both entry configs ship ``eval_every: -1`` in the ``sub_planner`` block.

    Read as text rather than through Hydra on purpose: this is about the literal value the
    Evaluation_Protocol runs with, and it has to be checkable in an environment without
    ``omegaconf``.

    Validates: Requirements 5.1, 5.2
    """
    text = (_REPO_ROOT / "conf" / f"{config_name}.yaml").read_text(encoding="utf-8")
    eval_every_lines = [
        line.strip() for line in text.splitlines() if line.strip().startswith("eval_every:")
    ]
    assert eval_every_lines == ["eval_every: -1"], (
        f"conf/{config_name}.yaml no longer pins eval_every to -1 "
        f"(found {eval_every_lines}). The AggInstrumentation step index assumes the early break "
        f"in planning/gd.py is unreachable, which is true only at -1."
    )


@PLANNER_SETTINGS
@given(
    opt_steps=st.integers(min_value=2, max_value=10),
    horizon=st.integers(min_value=1, max_value=5),
)
def test_eval_every_minus_one_makes_the_early_break_unreachable(opt_steps, horizon):
    """At ``eval_every == -1`` the evaluator is never called, so ``np.all(successes)`` never runs.

    The evaluator here reports **all-success on every call**, which is precisely what would trip
    ``break`` at ``i == 0``. The loop still runs the full ``opt_steps`` times.

    Note what is doing the work: ``i % -1 == 0`` for *every* ``i`` in Python, so the modulus alone
    would evaluate at every step rather than never. It is the explicit ``self.eval_every != -1``
    conjunct that disables the branch, and that is what this test pins.

    Validates: Requirements 5.1, 5.2
    """
    planner, log, obs_0, obs_g = build_real_planner(
        opt_steps=opt_steps, horizon=horizon, eval_every=-1, all_success=True
    )

    planner.plan(obs_0=obs_0, obs_g=obs_g)

    assert log.eval_calls == []
    assert log.n_objective_calls == opt_steps
    assert log.events == _expected_events(opt_steps)


@pytest.mark.parametrize("opt_steps", (4, 7))
def test_the_break_is_reachable_only_when_eval_every_is_not_minus_one(opt_steps):
    """Contrast case: the ``break`` really can fire, so ``-1`` is load-bearing rather than moot.

    With ``eval_every = 1`` and the same all-success evaluator the loop terminates after the
    first iteration, which is what would silently truncate the Instrumentation_Record's last-step
    entry if the Evaluation_Protocol ever stopped pinning ``eval_every: -1``.

    Validates: Requirements 5.1, 5.2
    """
    planner, log, obs_0, obs_g = build_real_planner(
        opt_steps=opt_steps, horizon=3, eval_every=1, all_success=True
    )

    planner.plan(obs_0=obs_0, obs_g=obs_g)

    assert log.eval_calls == ["plan_0_output_1"]
    assert log.n_objective_calls == 1
    assert log.n_objective_calls < opt_steps


def test_constructing_the_planner_does_not_invoke_the_objective():
    """Nothing outside ``plan()``'s loop touches the objective -- not even construction."""
    planner, log, _, _ = build_real_planner(opt_steps=5, horizon=3)

    assert log.events == []
    assert log.n_objective_calls == 0
    assert planner.eval_every == -1


# ---------------------------------------------------------------------------
# Tie-back: a real AggInstrumentation attached to the real GDPlanner
# ---------------------------------------------------------------------------


@pytest.fixture
def clean_agg_context():
    """Leave :data:`agg_objectives.AGG_CONTEXT` exactly as it was found.

    The holder is a module-level singleton, so a leaked head or recorder would be visible to
    every other test in the suite.
    """
    saved = (
        AGG_CONTEXT.agg_head,
        AGG_CONTEXT.agg_weight,
        AGG_CONTEXT.opt_steps,
        AGG_CONTEXT.output_dir,
        AGG_CONTEXT.instrumentation,
    )
    AGG_CONTEXT.clear()
    try:
        yield AGG_CONTEXT
    finally:
        (
            AGG_CONTEXT.agg_head,
            AGG_CONTEXT.agg_weight,
            AGG_CONTEXT.opt_steps,
            AGG_CONTEXT.output_dir,
            AGG_CONTEXT.instrumentation,
        ) = saved


@pytest.mark.parametrize("agg_weight", (0.0, 0.5))
@pytest.mark.parametrize("opt_steps", (1, 3, 6))
@pytest.mark.parametrize("mode", ("last", "staged"))
def test_real_instrumentation_records_first_and_last_step_of_the_real_planner(
    clean_agg_context, tmp_path, agg_weight, opt_steps, mode
):
    """The whole chain: real ``GDPlanner`` -> real objective -> real ``AggInstrumentation``.

    Asserts what the Instrumentation_Record claims about itself actually holds when the counting
    is driven by the frozen loop rather than by a hand-rolled one:

    * a record at ``step_index == 0`` and one at ``step_index == opt_steps - 1``, per plan call
      (Requirements 5.1, 5.2). At ``opt_steps == 1`` the two coincide and one record is emitted.
    * exactly one ``plan_call`` per ``plan()`` call, so two calls give ``plan_call`` 0 and 1.
    * ``updates_applied`` is honest: the ``step_index == 0`` record was formed while the action
      tensor was still the untouched zero initialization.
    * ``ratio`` is populated at both recorded steps (Requirement 5.3), and the raw L_agg magnitude
      is recorded even at Agg_Weight ``0`` (Requirement 5.5).
    * ``step_boundary_mismatch`` stays false, i.e. the recorder's own self-check agrees with the
      planner.

    Validates: Requirements 5.1, 5.2, 5.3
    """
    head = make_agg_head(in_dim=AGG_IN_DIM, hidden_dim=8, out_dim=5)
    clean_agg_context.publish(
        agg_head=head,
        agg_weight=agg_weight,
        opt_steps=opt_steps,
        output_dir=str(tmp_path),
    )
    instrumentation = clean_agg_context.start_instrumentation(objective_mode=mode)
    assert isinstance(instrumentation, AggInstrumentation)

    # The counting wrapper only counts; `log()` and `advance()` still happen inside the real
    # objective, driven by the real planner loop.
    planner, log, obs_0, obs_g = build_real_planner(
        opt_steps=opt_steps,
        horizon=4,
        inner_objective=create_agg_objective_fn(
            alpha=1, base=2, mode=mode, agg_weight=agg_weight, agg_head=head
        ),
    )

    n_plan_calls = 2
    for call_index in range(n_plan_calls):
        planner.plan(obs_0=obs_0, obs_g=obs_g, step=None)

    # The planner drove the recorder exactly as many times as it invoked the objective.
    assert log.n_objective_calls == opt_steps * n_plan_calls
    assert instrumentation.plan_call == n_plan_calls, (
        "the recorder did not roll over exactly once per plan() call: "
        f"plan_call={instrumentation.plan_call} after {n_plan_calls} calls"
    )
    assert instrumentation.step_index == 0
    assert instrumentation.step_boundary_mismatch is False

    expected_indices = sorted({0, opt_steps - 1})
    for plan_call in range(n_plan_calls):
        indices = [r["step_index"] for r in instrumentation.records if r["plan_call"] == plan_call]
        assert indices == expected_indices, (
            f"plan_call {plan_call} recorded step indices {indices}, expected {expected_indices}"
        )

    for record in instrumentation.records:
        assert record["updates_applied"] == record["step_index"]
        assert record["mpc_step_arg"] is None  # open-loop: `step` is None throughout
        assert isinstance(record["l_spatial"], float)
        assert isinstance(record["l_agg"], float)  # Requirement 5.5: raw L_agg even at w == 0
        if agg_weight == 0.0:
            # The term contributes nothing to L_plan, which is a statement about the weight and
            # not a missing value -- so 0.0, never "undefined".
            assert record["ratio"] == 0.0
        else:
            assert isinstance(record["ratio"], float)
            assert record["ratio"] == pytest.approx(
                agg_weight * record["l_agg"] / record["l_spatial"]
            )

    # `step_index == 0` really does describe the pre-update state: the objective invocation that
    # produced it saw the untouched zero action tensor.
    first_calls = [log.objective_calls[0], log.objective_calls[opt_steps]]
    for call in first_calls:
        assert torch.equal(call["act"], torch.zeros_like(call["act"]))

    headline = instrumentation.headline()
    assert headline["step_0"]["step_index"] == 0
    assert headline["step_0"]["updates_applied"] == 0
    assert headline["step_100"]["step_index"] == opt_steps - 1
    assert headline["step_100"]["updates_applied"] == opt_steps - 1


def test_a_wrong_opt_steps_read_mislabels_the_record_rather_than_erroring(
    clean_agg_context, tmp_path
):
    """The failure mode this gate exists to close, exhibited rather than described.

    The recorder is given ``opt_steps = N + 1`` while the real planner runs ``N`` iterations, i.e.
    the frozen-code read is off by one. Nothing raises. Instead:

    * the last-step record of plan call 1 is never emitted (``_i`` never reaches ``opt_steps - 1``
      within that call), and
    * the *first* evaluation of plan call 2 is labelled ``plan_call 0, step_index N`` -- a step
      that describes zero applied updates, recorded as the one formed after ``N``.

    That is a silently mislabelled Instrumentation_Record: the term magnitudes that decide whether
    the Sweep_Grid brackets anything useful would be read off the wrong optimizer step. The tests
    above are what keep the read honest; this one is why they are not optional.
    """
    real_opt_steps = 4
    head = make_agg_head(in_dim=AGG_IN_DIM, hidden_dim=8, out_dim=5)
    clean_agg_context.publish(
        agg_head=head, agg_weight=0.5, opt_steps=real_opt_steps + 1, output_dir=str(tmp_path)
    )
    instrumentation = clean_agg_context.start_instrumentation(objective_mode="last")

    planner, log, obs_0, obs_g = build_real_planner(
        opt_steps=real_opt_steps,
        horizon=4,
        inner_objective=create_agg_objective_fn(
            alpha=1, base=2, mode="last", agg_weight=0.5, agg_head=head
        ),
    )
    planner.plan(obs_0=obs_0, obs_g=obs_g)
    planner.plan(obs_0=obs_0, obs_g=obs_g)

    assert log.n_objective_calls == 2 * real_opt_steps
    labels = [(r["plan_call"], r["step_index"]) for r in instrumentation.records]

    # No record for the true last step of either plan call, and the second plan call's *first*
    # evaluation is filed under plan_call 0 at step_index 4 -- 0 updates applied, labelled 4.
    assert (0, real_opt_steps - 1) not in labels
    assert (0, real_opt_steps) in labels
    assert instrumentation.plan_call == 1  # two plan() calls, one counted
    # And the self-check cannot see any of it: `step` is None throughout in the open-loop setting.
    assert instrumentation.step_boundary_mismatch is False
