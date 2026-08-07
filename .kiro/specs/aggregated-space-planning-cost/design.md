# Design Document

## Overview

The planning objective becomes `L_plan = L_spatial + w * L_agg`, evaluated against the existing Target_Cell
checkpoint with no retraining. Two new root-level files carry the whole feature:

- `agg_objectives.py` (Agg_Objective_Module) — computes L_agg and L_plan, owns the late-binding holder for
  Agg_Head, owns the instrumentation recorder and the recording evaluator subclass.
- `plan_agg.py` (Plan_Wrapper) — a Hydra entry point that mirrors `plan.main`, validates the weight, the
  `agg_type` and the Evaluation_Protocol, loads Agg_Head from the checkpoint, publishes it, rewrites its own
  in-memory `cfg_dict["objective"]["_target_"]`, and then calls `plan.planning_main` unchanged.

Nothing under `planning/` or `datasets/` and nothing in `plan.py` is edited. The one existing file that changes
is `run_ccr_pilot.sh`, which is already in the Scope_Guard allowlist (see [Launcher integration](#launcher-integration)).

The design's load-bearing idea is that **both** loss terms are produced by the *same* frozen factory. L_agg is
not a reimplementation of the frozen reduction: it is a second call to
`planning.objectives.create_objective_fn(alpha=0, base=base, mode=mode)` fed with aggregated-space feature
dicts. Stage dispatch, per-frame coefficients, broadcasting and reduction order therefore cannot drift from
`planning/objectives.py`, because they *are* `planning/objectives.py`.

## Scope

New files only. Both are root-level per Requirements 4.1 and 4.2.

| Path | Kind | Purpose | Requirements |
|---|---|---|---|
| `agg_objectives.py` | new, root | `create_agg_objective_fn` factory, `AGG_CONTEXT` holder, `AggInstrumentation` recorder, `RecordingPlanEvaluator`, `SWEEP_GRID`, `select_agg_weight`, `paired_counts` | 1.1-1.10, 3.2, 3.4, 4.1, 4.7, 5.1-5.6, 6.1, 6.4-6.6, 7.4, 11.4 |
| `plan_agg.py` | new, root | Hydra entry, weight/agg_type/protocol validation, Agg_Head extraction and publication, `cfg_dict` objective rewrite, manifest writing, delegation to `plan.planning_main` | 2.1-2.7, 3.1, 3.5, 4.2, 5.4, 8.1-8.7 |
| `tests/test_agg_objective.py` | new, covered by the `tests/` allowlist prefix | Properties 1-10 | see [Correctness Properties](#correctness-properties) |
| `tests/test_scope_guard.py` | edited, covered by the `tests/` allowlist prefix | two new allowlist entries, `plan.py` added to the frozen set, Scope_Amendment comment | 4.3-4.6 |
| `run_ccr_pilot.sh` | edited, **already** in `ALLOWED_FILES` | two env-gated additions: `PLAN_ENTRY`, `SETTINGS` | 9.1-9.3 |

Requirement 4.5 is satisfied: exactly two new paths (`agg_objectives.py`, `plan_agg.py`) are added to
`ALLOWED_FILES`. `tests/` and `.kiro/specs/` are already covered by `ALLOWED_PREFIXES`, and
`run_ccr_pilot.sh` is already an allowlist member, so editing it adds no entry.

## Architecture

Ordering inside one wrapper process, left to right. Frozen code is marked `[frozen]`.

```
plan_agg.main(cfg)                              # @hydra.main, cwd == hydra run dir
  1  validate_agg_weight(cfg.agg_weight)                       # Req 3.1, 3.4, 3.5
  2  resolve_protocol(config_name, cfg)  -> ProtocolRecord      # Req 8.1-8.4, 8.7
  3  agg_head = load_agg_head(ckpt_path)                        # Req 2.4, 2.6
  4  AGG_CONTEXT.publish(agg_head, w, opt_steps, out_dir)       # Req 2.5
  5  plan.PlanEvaluator = RecordingPlanEvaluator                # Req 7.4
  6  cfg_dict["objective"]["_target_"] = "agg_objectives.create_agg_objective_fn"
     cfg_dict["objective"]["agg_weight"] = w                    # Req 2.3
  7  write agg_run_manifest.json                                # Req 2.5, 8.6
  8  plan.planning_main(cfg_dict)              [frozen]
       seed(cfg.seed)                          [frozen]  <-- resets every RNG
       dset = ...                              [frozen]
       model = load_model(...)                 [frozen]
       PlanWorkspace.__init__                  [frozen]
         objective_fn = hydra.utils.call(cfg_dict["objective"])  --> our factory
         planner = instantiate(..., objective_fn=objective_fn)   [frozen]
       perform_planning -> planner.plan        [frozen]
         GDPlanner loop, 100 iterations        [frozen]
           loss = objective_fn(z_pred, z_g, step)  --> our callable
  9  finally: restore plan.PlanEvaluator, AGG_CONTEXT.clear(), flush records
```

### 1. Injecting the objective without editing `plan.py`

`plan.py` builds the objective with `hydra.utils.call(cfg_dict["objective"])` and passes nothing else, so the
only argument channel is the config block, and the block is resolved before any model exists in
`plan_agg`'s frame. Two things must therefore be true at once: the config must name our factory, and the
factory must be able to reach Agg_Head.

**Mechanism.** The wrapper rewrites `cfg_dict["objective"]["_target_"]` in its *own* dict, before handing that
dict to `plan.planning_main`. Agg_Head travels by a different route: a module-level holder in the **new**
module.

```python
# agg_objectives.py
@dataclass
class _AggContext:
    agg_head: torch.nn.Module | None = None
    agg_weight: float = 0.0
    opt_steps: int | None = None
    output_dir: str | None = None
    instrumentation: "AggInstrumentation | None" = None

    def publish(self, agg_head, agg_weight, opt_steps, output_dir): ...
    def require(self):
        if self.agg_head is None:
            raise RuntimeError(
                "agg_objectives.AGG_CONTEXT holds no Agg_Head. create_agg_objective_fn was "
                "reached without plan_agg.py publishing one first -- most likely plan.py was "
                "launched directly with objective._target_ pointing at this module. Launch "
                "plan_agg.py instead."
            )
        return self

AGG_CONTEXT = _AggContext()
```

The rewrite is done in the wrapper rather than on the command line so that `_target_` cannot be forgotten;
the only override a user types is `+agg_weight=<w>`. `agg_weight` is *also* written into the objective block
so it reaches the factory as a normal config kwarg and appears in the recorded config (Requirement 2.5).

**Ordering, and why the holder is always populated.** `planning_main` runs, in file order, `seed` → dataset →
`load_model` → `PlanWorkspace.__init__` → `hydra.utils.call(cfg_dict["objective"])`. The factory is therefore
reached strictly after the wrapper's step 4, in one process, on one thread, with no callbacks in between.
`require()` exists for the one reachable misuse — someone pointing `plan.py` at our `_target_` directly — and
fails with an actionable message instead of a `NoneType` traceback.

**Why Agg_Head is loaded by the wrapper and not taken from the planner's model.** The wrapper cannot see
`planning_main`'s local `model`, so it loads the checkpoint itself with the frozen helper
`plan.load_ckpt(model_ckpt, device="cpu")`, takes `payload["encoder"]`, keeps only `encoder.agg_mlp` and
`encoder.agg_post_norm`, and drops the rest. Three consequences, all in the design's favour:

- *Same weights.* One file, one `torch.load`, no conversion; the head's parameters are bit-identical to the
  ones inside the planner's encoder, which later gets `.to(cuda)` (exact for float32).
- *Fail fast.* `agg_type` (Requirement 2.6), the head's input width (Requirement 1.9) and the protocol
  (Requirement 8.7) are all checked in the first few seconds, before the dataset load and the env spawn.
- *No RNG perturbation.* The extra load happens **before** `planning_main` calls `seed(cfg_dict["seed"])`,
  and `utils.seed` reseeds `random`, `torch`, `numpy` and all CUDA generators. Every RNG state inside
  `planning_main` is therefore exactly what `plan.py` would have had, which is what keeps Requirement 3.3 and
  the Paired_Comparison exact. If the encoder key is absent from the checkpoint the wrapper aborts, since
  Requirement 2.4 cannot then be met.

Cost is one extra `torch.hub.load` plus one extra `torch.load` per job, on the order of a minute against a
5-15 minute job. Accepted in exchange for the RNG argument above.

**Rejected: monkeypatching `planning.objectives.create_objective_fn`.** Requirement 4.7 requires module-level
names in `planning.objectives` to stay at their original values, and rebinding the factory is exactly the
violation that forbids. It is also self-defeating: L_spatial is produced by *calling* that factory, so a patch
would recurse. Finally it would be invisible in the recorded config, which is the opposite of what
Requirement 8.6 is for. Rejected outright, not deferred.

**Rejected: editing `conf/plan_gd.yaml` / `conf/plan_gd_mpc.yaml`.** Not in the allowlist, and Requirements
8.2/8.3 name those files as the protocol. Everything the wrapper needs arrives as a CLI override or as a
wrapper-side `cfg_dict` rewrite.

**Rejected (kept only as a fallback): monkeypatching `plan.load_model`.** A wrapper that replaced
`plan.load_model` with `lambda *a, **k: publish(original(*a, **k))` would also give correct ordering and would
avoid the second checkpoint load. It is not chosen because it moves validation past the dataset load and gives
up the pre-`seed()` argument above. It is recorded here so a future maintainer does not have to rediscover it.

### 2. Staged and all-mode coefficient reuse

Requirement 1.6 demands that L_agg receive the *same* stage selection and the *same* per-frame coefficients as
L_spatial. `objective_fn_all` builds `coeffs = [base**i for i in range(T)]` normalized to sum 1 and
`objective_fn_staged` dispatches on `step < z_obs_pred["visual"].shape[1] - 1`. Copying either would create a
silent-drift hazard the moment `planning/objectives.py` is re-read at a different revision.

So neither is copied. The module builds **two** callables from the frozen factory:

```python
spatial_fn = create_objective_fn(alpha=alpha, base=base, mode=mode)   # L_spatial, real alpha
agg_fn     = create_objective_fn(alpha=0,     base=base, mode=mode)   # L_agg, alpha pinned to 0
```

and calls `agg_fn` on aggregated-space dicts:

```python
def _agg_dicts(z_pred, z_tgt, head):
    a_pred = _apply_head(z_pred["visual"], head)   # (B, T_pred, agg_out_dim)
    a_tgt  = _apply_head(z_tgt["visual"],  head)   # (B, 1,      agg_out_dim)
    # alpha == 0, so the proprio term is multiplied by zero; zeros make it exactly 0.0 as well.
    p_pred = a_pred.new_zeros(a_pred.shape[0], a_pred.shape[1], 1)
    p_tgt  = a_tgt.new_zeros(a_tgt.shape[0],  a_tgt.shape[1],  1)
    return ({"visual": a_pred, "proprio": p_pred},
            {"visual": a_tgt,  "proprio": p_tgt})

l_agg = agg_fn(*_agg_dicts(z_pred, z_tgt, head), step=step)           # (B,)
```

Why this reproduces the coefficients exactly rather than approximately:

- `T` is preserved. `_apply_head` maps frame-wise, so `a_pred.shape[1] == z_pred["visual"].shape[1]`. The
  staged dispatch predicate `step < shape[1] - 1` and the coefficient vector `[base**i for i in range(T)]`
  depend only on `T`, `base` and the device — all identical. There is one coefficient computation in the
  repository and both terms go through it.
- Reduction shape matches by construction. `objective_fn_all` reduces `dim=tuple(range(2, ndim))`: the spatial
  tensor has `ndim == 4` (B, T, 196, 8) and reduces over patches and channels; the agg tensor has `ndim == 3`
  (B, T, 128) and reduces over the 128 features. Both yield `(B, T)` before the coefficient product and the
  `.mean(dim=1)`.
- `T_tgt == 1` broadcasting is inherited, not re-derived. `nn.MSELoss(reduction="none")` broadcasts
  `(B, T, ...)` against `(B, 1, ...)` in both spaces; whatever the frozen code does with that, L_agg does too.
- The proprio channel contributes exactly nothing. `alpha=0` and zero-valued proprio tensors give
  `loss_proprio == 0.0` and `0 * 0.0 == 0.0`, and `x + 0.0` is bit-exact for every float except `-0.0`, which
  a mean of squares cannot produce. So `l_agg` is exactly the aggregated-space visual term.
- `last` mode needs no special case: `objective_fn_last` already slices `[:, -1:]`, which is Requirement 1.5.

At the Target_Cell shapes, `T = 6` and `base = 2`, so both terms use `coeffs = [1,2,4,8,16,32]/63`, and the
staged threshold is `step < 5`: MPC iterations 0-4 take the terminal-only branch and 5-19 take the weighted
branch, so both stages are exercised in a single confirmation run.

### 3. The bitwise-zero guarantee

`L_spatial + 0 * L_agg` is not bitwise safe: `0 * inf` is `nan`, `0 * nan` is `nan`, and either poisons the
sum. Requirement 3.2 is therefore met by a gate, not by arithmetic — the same shape as
`VWorldModel.__init__`'s `self.ccr = self.lambda_cf > 0`, which is why the CCR-disabled path costs one
attribute lookup and one comparison.

```python
def objective(z_obs_pred, z_obs_tgt, step=None):
    loss_spatial = spatial_fn(z_obs_pred, z_obs_tgt, step=step)   # frozen code, untouched
    record = instrumentation.should_record()                      # bool, see below

    if not enabled:                       # enabled = float(agg_weight) > 0.0, resolved once
        if record:
            with torch.no_grad():                                 # Req 5.5, off the autograd graph
                instrumentation.log(step, loss_spatial, _agg_loss(z_obs_pred, z_obs_tgt))
        return loss_spatial               # the delegate's own tensor object, untouched

    loss_agg = _agg_loss(z_obs_pred, z_obs_tgt, step=step)
    if record:
        instrumentation.log(step, loss_spatial, loss_agg)
    return loss_spatial + agg_weight * loss_agg
```

The `w == 0` path performs no tensor operation on `loss_spatial` at all — it returns the object the frozen
callable returned, so bitwise equality is by identity rather than by numerical luck, and a non-finite L_agg
cannot reach the result. Requirement 5.5 still gets its raw L_agg because the instrumentation call is inside
`torch.no_grad()` and fires on 2 of 100 optimizer steps, so it neither joins the autograd graph nor shows up
in the runtime. `enabled` is computed once at factory time from the validated float, so no per-step
`== 0.0` comparison on a tensor is involved.

### 4. Observing optimizer steps from outside frozen code

The objective callable never receives the optimizer step. In `last` mode the `step` argument is `None`; under
MPC it is the *outer* MPC iteration, not the inner one. But `planning/gd.py` calls the objective exactly once
per inner iteration, in order, and `eval_every` is `-1` so the early-`break` is unreachable — the loop always
runs `opt_steps` times. The call index therefore *is* the optimizer step index, and the recorder can count its
own invocations:

```python
class AggInstrumentation:
    def __init__(self, opt_steps, agg_weight, path): ...

    def should_record(self):
        return self._i == 0 or self._i == self.opt_steps - 1

    def log(self, step, l_spatial, l_agg):
        rec = {
            "plan_call": self._plan_call,      # 0 for open-loop; MPC outer iteration otherwise
            "mpc_step_arg": step,              # the frozen `step` argument, as received
            "step_index": self._i,             # 0-based optimizer step within this plan() call
            "updates_applied": self._i,        # Adam updates already applied when this loss was formed
            "l_spatial": float(l_spatial.mean()),
            "l_agg": float(l_agg.mean()),
            "ratio": self._ratio(l_spatial, l_agg),
        }
        ...

    def advance(self):                         # called at the end of every objective invocation
        self._i += 1
        if self._i >= self.opt_steps:           # a new sub_planner.plan() call starts next
            self._i = 0
            self._plan_call += 1
```

`opt_steps` comes from the resolved `planner.sub_planner.opt_steps`, which the protocol checker has already
pinned to 100, and is published through `AGG_CONTEXT`. As a self-check, the recorder asserts that the received
`step` argument is constant within a plan call; a change mid-count would mean `opt_steps` disagreed with the
planner and is recorded as `step_boundary_mismatch: true` rather than silently mislabelled.

**What "step 100" means here, stated rather than glossed.** With `opt_steps: 100` the loop indices are 0-99, so
there are exactly 100 objective evaluations: index 0 is formed before any update, index 99 is formed after 99
updates. There is no evaluation after the 100th update, and producing one would require an extra forward pass
that only frozen code can trigger. Requirement 5.2's "step 100" is therefore recorded as the 100th evaluation
(`step_index: 99`, `updates_applied: 99`), and every record carries both fields plus a
`step_100_semantics` string in the file so the number is never read as something it is not.

`ratio` is `agg_weight * l_agg / l_spatial`, or the string `"undefined"` when `l_spatial == 0.0`
(Requirement 5.6). At `agg_weight == 0` the ratio is `0.0`, not `"undefined"`, and `l_agg` is still recorded.

### 5. Shapes, and where the shape error is raised

Target_Cell shapes, with `n_evals = 50`, `goal_H = 25`, `frameskip = 5`, `num_hist = 3`, 14x14 patches and
`projector_out_dim = 8`:

| Tensor | Open-loop (`last`) | MPC (`staged`) | Notes |
|---|---|---|---|
| `z_obs_pred["visual"]` | `(50, 6, 196, 8)` | `(50, 6, 196, 8)` | `wm.rollout` returns `num_obs_init + T_act` frames plus one final prediction; 1 + 4 + 1 = 6 |
| `z_obs_tgt["visual"]` | `(50, 1, 196, 8)` | `(50, 1, 196, 8)` | `encode_obs` on a single goal frame |
| flattened for the head | `(300, 196, 8)` -> `(300, 1568)` | same | `agg` itself does `x.contiguous().view(x.shape[0], -1)` |
| `agg_mlp` -> `agg_post_norm` | `(300, 128)` | same | 1568 -> 512 -> 512 -> 128, `LayerNorm(128)` |
| aggregated pred / tgt | `(50, 6, 128)` / `(50, 1, 128)` | same | reshaped back before the frozen reduction |
| returned loss | `(50,)` | `(50,)` | Requirement 1.4 |

`_apply_head` mirrors `VWorldModel.total_curvature`'s existing `aggcos` reshape exactly — `b, t, p, d` →
`reshape(b * t, p, d)` → `head(...)` → `reshape(b, t, -1)` — so the aggregation the planner sees is the same
operation the curvature regularizer applied during training.

Device and dtype (Requirement 1.8) are resolved lazily on the first call from the incoming tensor:
`head.to(device=z.device, dtype=z.dtype)`, cached, `head.eval()`, and
`requires_grad_(False)` on its parameters. Gradients still flow to `z` (Requirement 1.7); the planner's
optimizer holds only `actions`, so the head can never be updated, and the property test asserts the parameter
bytes are unchanged after a backward.

The head's widths are **read from the checkpoint**, never hardcoded:
`in_dim = encoder._agg_mlp_in_dim`, `out_dim = encoder._agg_out_dim`. The wrapper records both and warns if
they differ from the expected 1568 and 128. (The `agg32` token in the checkpoint directory name is a literal
in `conf/train.yaml`'s run-dir template, not a resolved head width — `conf/encoder/dino_channel.yaml` sets
`agg_out_dim: 128`, `agg_mlp_hidden_dim: 512`. Reading the head rather than parsing the name avoids that trap.)

Requirement 1.9's error is raised in `_apply_head`, before the `nn.Linear` call, by comparing `p * d` against
`in_dim`:

```python
if p * d != in_dim:
    raise ValueError(
        f"Agg_Head cannot accept the predicted visual features: received shape "
        f"(B={b}, T={t}, patches={p}, channels={d}), which flattens to {p * d} features "
        f"per frame, but this checkpoint's agg_mlp requires exactly {in_dim} "
        f"(= 196 patches x {in_dim // 196} channels). The planner's encoder and the "
        f"aggregation head disagree; check that the checkpoint is the 14x14x8 "
        f"projected-channel encoder."
    )
```

Raising here, rather than letting `nn.Linear` raise a bare mat1/mat2 mismatch, is what names both shapes.

### 6. Per-episode outcomes for the Paired_Comparison

Requirements 7.4 and 11.4 need per-episode success vectors. `plan.py` persists only means:
`PlanEvaluator._compute_rollout_metrics` reduces `successes` into `logs["success_rate"]`, and the per-episode
`successes` array is returned up the stack and dropped. The per-episode videos that do encode the outcome are
written for `n_plot_samples = 10` only, and only when `decode_for_viz` is true, which the launcher sets false.

Resolution: `plan.py` imports the evaluator as a module-level name (`from planning.evaluator import
PlanEvaluator`) and constructs it directly, so the wrapper rebinds *that name in the `plan` module* to a
subclass defined in `agg_objectives.py`:

```python
class RecordingPlanEvaluator(PlanEvaluator):
    """Read-only observer. Delegates, records, returns the delegate's own result."""
    def eval_actions(self, actions, action_len=None, filename="output", save_video=False):
        result = super().eval_actions(actions, action_len, filename, save_video)
        try:
            AGG_CONTEXT.record_episodes(filename, result[1])   # result[1] is `successes`
        except OSError as exc:
            AGG_CONTEXT.note_record_failure(exc)               # never lose a 15-minute eval to a bad write
        return result
```

Why this is safe: no file under `planning/` is edited and no name inside `planning/` is rebound — only
`plan.PlanEvaluator`, an attribute of the module the wrapper is driving, in the wrapper's own process.
`plan.py`'s bytes are untouched, so the Scope_Guard's `plan.py` assertion (Requirement 4.4) still holds. The
subclass adds no state that any computation reads, consumes no RNG, performs no tensor work, and returns the
object `super()` returned, so control flow and numerics are identical. It is restored in the wrapper's
`finally` block. This is the same "strictly additive, and observational only" standard the existing
`models/vit.py` Scope_Amendment was admitted under, and it is recorded in the amendment rather than taken
quietly.

The reported outcome vector is the row with `filename == "output_final"`, which is the eval
`PlanWorkspace.perform_planning` uses for `final_eval/success_rate`. Under MPC the intermediate
`plan{iter}` rows are also recorded and are useful, but they are not the reported result.

### 7. Reading of Requirement 8.4 for the MPC setting, flagged

Requirement 8.4's list — `max_iter 1`, `n_taken_actions 25`, `sub_planner.horizon 25`, `sub_planner.lr 0.1`,
`sub_planner.sample_type zero`, `sub_planner.action_noise 0`, `sub_planner.opt_steps 100` — is exactly the
shipped default block of `conf/plan_gd.yaml`. `conf/plan_gd_mpc.yaml`, which Requirement 8.3 mandates for the
MPC setting, ships `max_iter: 20` and `n_taken_actions: 5` and is otherwise identical in the `sub_planner`
block. Taken literally for both settings, 8.4 would force MPC to run `max_iter 1`, which makes it
open-loop-with-a-staged-objective and could not reproduce the Platform_Baseline MPC number of 82.00 that
Requirement 8's own user story exists to stay comparable with.

Resolved reading, stated so it can be corrected rather than discovered later: the protocol is **no override of
any protocol field relative to the config file the setting mandates**. The checker holds one expected table
per setting —

| Field | open-loop (`plan_gd`) | MPC (`plan_gd_mpc`) |
|---|---|---|
| `n_evals` | 50 | 50 |
| `objective.mode` | `last` | `staged` |
| `objective.alpha` | 1 | 1 |
| `planner.max_iter` | 1 | 20 |
| `planner.n_taken_actions` | 25 | 5 |
| `planner.sub_planner.horizon` | 25 | 25 |
| `planner.sub_planner.lr` | 0.1 | 0.1 |
| `planner.sub_planner.sample_type` | `zero` | `zero` |
| `planner.sub_planner.action_noise` | 0 | 0 |
| `planner.sub_planner.opt_steps` | 100 | 100 |

— and Requirement 8.7 fires against the column for the resolved config name, obtained from
`HydraConfig.get().job.config_name`. All ten resolved values go into the manifest either way
(Requirement 8.6), so if the literal reading of 8.4 is the intended one, the record shows precisely which two
fields differ and a re-run is a two-flag change.

## Components and Interfaces

### `agg_objectives.py`

```python
SWEEP_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)      # Req 6.1
TUNING_SEED = 400                                   # Req 6.6
REPORTING_SEEDS = (100, 200, 300)
AGG_WEIGHT_MAX = 3.0                                # Req 3.4

AGG_CONTEXT: _AggContext

def validate_agg_weight(value) -> float: ...
    # Req 3.4, 3.5. Rejects negatives, nan, inf, non-numerics and values > 3, naming the value.

def extract_agg_head(encoder) -> tuple[torch.nn.Module, int, int]: ...
    # Req 2.4, 2.6. Returns (head, in_dim, out_dim); raises naming agg_type if it is not "mlp".

def create_agg_objective_fn(alpha, base, mode="last", agg_weight=0.0, **kwargs): ...
    # Req 1.1-1.10, 3.2. Hydra `_target_`. Signature is a superset of create_objective_fn's,
    # so the same objective block resolves against either factory.

class AggInstrumentation: ...                       # Req 5.1-5.6
class RecordingPlanEvaluator(PlanEvaluator): ...    # Req 7.4

def select_agg_weight(rows) -> SweepSelection: ...  # Req 6.4-6.6
def paired_counts(candidate, baseline) -> dict: ...  # Req 11.4
```

`create_agg_objective_fn` takes `**kwargs` so that the unmodified objective block — which carries `alpha`,
`base` and `mode` — resolves against it unchanged, and so a future config key cannot turn into a `TypeError`
inside frozen `hydra.utils.call`.

### `plan_agg.py`

```python
import hydra, plan, agg_objectives
from omegaconf import OmegaConf, open_dict
from hydra.core.hydra_config import HydraConfig
from utils import cfg_to_dict

@hydra.main(config_path="conf", config_name="plan_gd")     # no version_base, exactly as plan.main
def main(cfg: OmegaConf):
    with open_dict(cfg):
        cfg["saved_folder"] = os.getcwd()
    config_name = HydraConfig.get().job.config_name
    w = agg_objectives.validate_agg_weight(cfg.get("agg_weight", 0.0))
    protocol = resolve_protocol(config_name, cfg)                    # raises per Req 8.7
    head, in_dim, out_dim = load_agg_head_from_ckpt(cfg)
    cfg_dict = cfg_to_dict(cfg)
    cfg_dict["wandb_logging"] = bool(cfg_dict.get("wandb_logging", True))
    cfg_dict["objective"]["_target_"] = "agg_objectives.create_agg_objective_fn"
    cfg_dict["objective"]["agg_weight"] = w
    agg_objectives.AGG_CONTEXT.publish(
        agg_head=head, agg_weight=w,
        opt_steps=protocol.resolved["planner.sub_planner.opt_steps"],
        output_dir=os.path.abspath(cfg["saved_folder"]),
    )
    original_evaluator = plan.PlanEvaluator
    plan.PlanEvaluator = agg_objectives.RecordingPlanEvaluator
    try:
        write_manifest(protocol, w, in_dim, out_dim)                 # Req 2.5, 8.6
        plan.planning_main(cfg_dict)                                 # Req 2.1
    finally:
        plan.PlanEvaluator = original_evaluator
        agg_objectives.AGG_CONTEXT.flush_and_clear()                 # Req 5.4

if __name__ == "__main__":
    main()
```

`@hydra.main` is written without `version_base` deliberately: `plan.planning_main` depends on the cwd being
the run directory, which is the Hydra-version-dependent `job.chdir` default that `plan.main` already relies
on. Matching the decorator exactly keeps both entry points on the same behaviour.

Requirement 2.7 is met by construction — every result file is written by frozen code
(`logs.json`, `plan_targets.pkl`, videos, PNGs) into a `plan_outputs_*` directory, so
`aggregate_results.py` globs and parses it unmodified. The wrapper only *adds* files.

## Data models

Three files per run, written next to the frozen `logs.json`.

`agg_run_manifest.json` (Requirements 2.5, 8.6):

```json
{
  "feature": "aggregated-space-planning-cost",
  "config_name": "plan_gd",
  "setting": "open-loop",
  "agg_weight": 0.1,
  "seed": 400,
  "checkpoint": "/abs/path/checkpoints/test/pusht_aggmlpcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05",
  "model_epoch": "latest",
  "agg_head": {"agg_type": "mlp", "in_dim": 1568, "out_dim": 128, "hidden_dim": 512},
  "protocol_resolved": {
    "n_evals": 50, "objective.mode": "last", "objective.alpha": 1,
    "planner.max_iter": 1, "planner.n_taken_actions": 25,
    "planner.sub_planner.horizon": 25, "planner.sub_planner.lr": 0.1,
    "planner.sub_planner.sample_type": "zero",
    "planner.sub_planner.action_noise": 0, "planner.sub_planner.opt_steps": 100
  },
  "protocol_expected_source": "conf/plan_gd.yaml shipped defaults",
  "protocol_ok": true,
  "git_rev": "…"
}
```

`agg_instrumentation.json` (Instrumentation_Record, Requirements 5.1-5.6):

```json
{
  "agg_weight": 0.1,
  "opt_steps": 100,
  "objective_mode": "last",
  "step_100_semantics": "step_index 99 is the 100th objective evaluation of a plan() call, formed after 99 Adam updates; planning/gd.py performs no evaluation after the 100th update.",
  "headline": {
    "step_0":   {"plan_call": 0, "mpc_step_arg": null, "step_index": 0,  "updates_applied": 0,  "l_spatial": 0.04213, "l_agg": 1.87340, "ratio": 4.4468},
    "step_100": {"plan_call": 0, "mpc_step_arg": null, "step_index": 99, "updates_applied": 99, "l_spatial": 0.01072, "l_agg": 0.90211, "ratio": 8.4152}
  },
  "records": [ "… one object per (plan_call, recorded step_index) …" ],
  "step_boundary_mismatch": false,
  "record_failures": 0
}
```

`ratio` is a number, or the string `"undefined"` when `l_spatial` is exactly `0.0`. `headline` is always the
first plan call (`plan_call == 0`); under MPC every outer iteration also appears in `records`.

`agg_episode_outcomes.jsonl` (Requirements 7.4, 11.4), one line per `eval_actions` call:

```json
{"filename": "output_final", "plan_call": 19, "n_evals": 50, "successes": [true, false, true, "…"]}
```

## Error handling

| Condition | Where | Behaviour | Req |
|---|---|---|---|
| `agg_weight` negative, `nan`, `inf`, non-numeric, or > 3 | `validate_agg_weight`, before any load | abort naming the rejected value and the accepted interval | 3.4, 3.5 |
| `agg_type != "mlp"` | `extract_agg_head` | abort naming the encountered `agg_type` | 2.6 |
| checkpoint has no `encoder` key | `load_agg_head_from_ckpt` | abort: Agg_Head cannot be obtained from the checkpoint | 2.4 |
| head widths differ from 1568/128 | `load_agg_head_from_ckpt` | warn, record both in the manifest, continue | 1.9 |
| `p * d != in_dim` at call time | `_apply_head` | `ValueError` naming received shape, flattened width and required width | 1.9 |
| any protocol field deviates | `resolve_protocol`, before any load | abort naming field, expected, resolved | 8.7 |
| holder unpopulated when the factory runs | `AGG_CONTEXT.require()` | `RuntimeError` telling the user to launch `plan_agg.py` | 2.3 |
| instrumentation or outcome write fails | recorder | counted in `record_failures`, never raised | 5.4, 7.4 |
| `step` changes inside a counted plan call | recorder | `step_boundary_mismatch: true`, run continues | 5.1-5.3 |

## Launcher integration

`run_ccr_pilot.sh::run_eval_jobs` hardcodes `python plan.py --config-name plan_gd.yaml` and
`plan_gd_mpc.yaml`, and always runs both settings for every seed. Requirement 9.1 makes the Job_Launcher the
only permitted way to start a job on the pod, so the driver needs two additions. **`run_ccr_pilot.sh` is
already a member of the Scope_Guard `ALLOWED_FILES` set**, so this edit adds no allowlist entry and is
permitted as it stands; it is called out explicitly here because it is the one pre-existing file this feature
touches.

Both additions are env-gated and default to today's behaviour, so the CCR evaluation path stays
byte-behaviour-identical:

- `PLAN_ENTRY="${PLAN_ENTRY:-plan.py}"` replaces the two literal `plan.py` tokens.
- `SETTINGS="${SETTINGS:-both}"` (`ol` | `mpc` | `both`) guards the two seed loops.

Everything else — the MIG preflight refusal, the environment recipe, `PLAN_SERIAL_ENV=1`, the chaining
protocol, the one-job-at-a-time driver — is reused untouched, which is what Requirements 9.1-9.3 ask for.

### Sweep: 7 open-loop runs at the Tuning_Seed, ~40 minutes

Requirements 6.1-6.3, 6.7. `FOREGROUND=1` runs each job in this shell, so the preflight sees a free slice on
every iteration; wrap the loop in `setsid nohup bash -c '…'` if the session may drop.

```bash
export DATASET_DIR=/workspace/arun/data
CKPT="$PWD/checkpoints/test/pusht_aggmlpcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05"
RUNDIR='hydra.run.dir=plan_outputs_gd/${replace_slash:${model_name}}_gH${goal_H}_${goal_source}/aggw${agg_weight}_gd_lr${planner.sub_planner.lr}_an${planner.sub_planner.action_noise}_opt${planner.sub_planner.opt_steps}_obj${objective.mode}_init${planner.sub_planner.sample_type}'

for W in 0 0.01 0.03 0.1 0.3 1 3; do
  FOREGROUND=1 PLAN_ENTRY=plan_agg.py SETTINGS=ol SEEDS=400 \
  LOG="agg_sweep_w${W}.log" \
  bash run_ccr_pilot.sh eval "$CKPT" "+agg_weight=${W}" "$RUNDIR"
done
python aggregate_results.py
```

The `hydra.run.dir` override matters and is not cosmetic. The shipped template omits the seed on purpose —
`aggregate_results.py` relies on three seeds appending three lines to one `logs.json` to form one cell — but it
also omits the weight, so every arm would collide into the same cell. Substituting `aggw${agg_weight}` for the
`${ckpt_base_path}` component separates arms, keeps the `plan_outputs_gd` prefix, the model name and the
`obj…_init…` token the aggregator parses, and as a side effect flattens the absurd nesting an absolute
`ckpt_base_path` produced. Single quotes are required so `${…}` reaches Hydra rather than bash.

Selection is then `select_agg_weight` over the seven rows: highest open-loop success rate, smallest weight on a
tie, tie recorded (Requirements 6.4, 6.5).

### Confirmation: candidate and baseline, 3 seeds, both settings, ~1.5 hours

Requirements 7.1-7.4. Run once, for the selected `W_STAR` only.

```bash
for W in 0 "$W_STAR"; do
  FOREGROUND=1 PLAN_ENTRY=plan_agg.py SETTINGS=both SEEDS="100 200 300" \
  LOG="agg_confirm_w${W}.log" \
  bash run_ccr_pilot.sh eval "$CKPT" "+agg_weight=${W}" "$RUNDIR" \
    'hydra.run.dir=plan_outputs_gd_mpc/${replace_slash:${model_name}}_gH${goal_H}_${goal_source}/aggw${agg_weight}_gd_lr${planner.sub_planner.lr}_an${planner.sub_planner.action_noise}_opt${planner.sub_planner.opt_steps}_obj${objective.mode}_init${planner.sub_planner.sample_type}'
done
python aggregate_results.py
python ccr_acceptance_gate.py \
  --cand-ol-seeds <ol_100,ol_200,ol_300> --cand-mpc-seeds <mpc_100,mpc_200,mpc_300> \
  --base-ol 75.33 --base-mpc 82.00
```

Two notes on this block. The MPC leg needs the `plan_outputs_gd_mpc` prefix, so the driver must pass the
matching `hydra.run.dir` per setting rather than one string for both — implemented in `run_eval_jobs`
alongside the `SETTINGS` guard. And the w=0 arm is re-run rather than reusing the recorded Platform_Baseline:
the recorded numbers are means, and the Paired_Comparison needs per-episode vectors from the wrapper's own
outcome file. The 12 runs are the reason the ~1.5 h budget in Requirement 9.5 is tight; if it overruns, the
overrun is recorded, not traded against the protocol.

The gate is called with candidate means and the Platform_Baseline only — no threshold arguments — per
Requirement 10.7, and the verdict routes to Requirement 11 unchanged.

## Scope_Guard amendment

Added to `ALLOWED_FILES` in `tests/test_scope_guard.py`, following the `models/vit.py` precedent's structure
(what was touched, why it was unavoidable, why it is safe, what guards it). Also in that edit: `plan.py` joins
the byte-identity assertion (Requirement 4.4), which today covers only `FROZEN_DIRS = ("planning",
"datasets")`.

```python
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
        # Guarded by `tests/test_agg_objective.py`, which checks that at Agg_Weight 0 the returned
        # tensor is BITWISE equal to the unmodified objective's for arbitrary inputs including
        # non-finite ones, that the recording evaluator returns its delegate's result unchanged, and
        # that every attribute of `planning.objectives` keeps its original identity after use.
        "agg_objectives.py",
        "plan_agg.py",
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a
system-essentially, a formal statement about what the system should do. Properties serve as the bridge between
human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Zero weight is bitwise identity

*For any* pair of predicted and goal latent dictionaries (including entries containing `inf`, `-inf`, `nan` and
denormals), *any* `alpha`, *any* `base`, *any* mode in {`last`, `all`, `staged`}, and *any* `step`, the tensor
returned by the Agg_Objective_Module at Agg_Weight `0` has the same raw byte representation as the tensor
returned by the unmodified `planning.objectives.create_objective_fn` callable for the same inputs, while the
Instrumentation_Record for the recorded steps still carries a raw L_agg magnitude.

**Validates: Requirements 1.2, 3.1, 3.2, 5.5**

### Property 2: Additive decomposition with no hidden normalization

*For any* latent dictionaries, mode, `alpha`, `base`, `step`, and *any* two Agg_Weights `w1`, `w2` in the closed
interval 0 to 3, the returned tensor has shape `(B,)`, has the device and dtype of
`z_obs_pred["visual"]`, and satisfies `L_plan(w2) - L_spatial == (w2 / w1) * (L_plan(w1) - L_spatial)` to
floating-point tolerance, so neither term is rescaled relative to the other.

**Validates: Requirements 1.3, 1.4, 1.8, 1.10, 3.4**

### Property 3: Stage selection and coefficients are the frozen module's

*For any* frame count `T`, *any* `base`, *any* `step`, and *any* latent dictionaries, when the aggregation head
is the identity on flattened patch features and `alpha` is `0`, L_agg equals the value the unmodified
`planning.objectives.create_objective_fn` callable returns for the same mode, `base` and `step`; and in
`staged` mode L_agg equals the frozen `last`-mode value when `step < T - 1` and the frozen `all`-mode value
otherwise.

**Validates: Requirements 1.2, 1.6**

### Property 4: Last mode depends only on the final predicted frame

*For any* latent dictionaries in `last` mode, perturbing any predicted frame other than the final one leaves
L_agg unchanged, and perturbing the final predicted frame by a non-zero amount changes it.

**Validates: Requirements 1.5**

### Property 5: Agg_Head is frozen and differentiable through its input

*For any* sequence of objective calls followed by a backward pass with Agg_Weight greater than `0`, every
Agg_Head parameter tensor is byte-identical to its value before the calls, and the gradient of the returned
loss with respect to `z_obs_pred["visual"]` is populated.

**Validates: Requirements 1.7**

### Property 6: Shape mismatches are reported with both shapes

*For any* patch count and channel width whose product differs from the checkpoint head's input width, the
Agg_Objective_Module raises an error whose message contains the received shape, the flattened width it implies,
and the width Agg_Head requires.

**Validates: Requirements 1.9**

### Property 7: Instrumentation is complete, correctly indexed, and round-trips

*For any* `opt_steps` greater than `1`, *any* number of consecutive plan calls, *any* Agg_Weight, and *any*
sequence of loss magnitudes, the recorder emits exactly one record at `step_index` `0` and one at
`step_index` `opt_steps - 1` per plan call, each carrying both batch-mean magnitudes and the ratio
`Agg_Weight * L_agg / L_spatial`, with the ratio recorded as the string `"undefined"` exactly when L_spatial
is `0.0`; and parsing the written file yields the recorded values unchanged.

**Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.6**

### Property 8: planning.objectives is left untouched

*For any* sequence of Agg_Objective_Module factory constructions and objective calls across generated modes,
alphas, bases and weights, every public attribute of the `planning.objectives` module is the same object it was
before the module was imported.

**Validates: Requirements 4.7**

### Property 9: Frozen sources are byte-identical to the base revision

*For all* files in `planning/*.py`, `datasets/*.py`, and root-level `plan.py`, the newline-normalized
sha256 of the working-tree content equals the newline-normalized sha256 of the content at Base_Revision
`d73b9c6`.

**Validates: Requirements 4.3, 4.4**

### Property 10: Weight selection uses only the Tuning_Seed

*For any* sweep record table, `select_agg_weight` returns a weight attaining the maximum open-loop success rate
at the Tuning_Seed, returns the smallest such weight and flags a tie when several attain it, and returns the
same weight after arbitrary perturbation of every row whose seed is a Reporting_Seed.

**Validates: Requirements 6.4, 6.5, 6.6, 6.7**

### Property 11: The recording evaluator is transparent

*For any* return tuple produced by the base `PlanEvaluator.eval_actions`, `RecordingPlanEvaluator.eval_actions`
returns that identical object and appends one outcome row whose success vector equals the tuple's success
element.

**Validates: Requirements 7.4**

### Property 12: Protocol deviations are named

*For any* resolved configuration differing from the mandated per-setting expected table in exactly one
Evaluation_Protocol field, the Plan_Wrapper terminates with an error naming that field, its expected value and
its resolved value; and *for any* conforming configuration, the manifest contains every protocol field name
with its resolved value.

**Validates: Requirements 8.6, 8.7**

### Property 13: Weight validation rejects out-of-domain values

*For any* value that is negative, non-finite, non-numeric, or greater than `3`, the Plan_Wrapper terminates
with an error naming the rejected value; *for any* finite value in the closed interval 0 to 3, it proceeds.

**Validates: Requirements 3.4, 3.5**

### Property 14: Paired counts partition the episodes

*For any* two boolean success vectors of equal length, `paired_counts` returns candidate-only, baseline-only and
matching counts that sum to the vector length, and is invariant under any permutation applied to both vectors.

**Validates: Requirements 11.4**

## Testing strategy

- **Property tests** (`tests/test_agg_objective.py`, Hypothesis, minimum 100 iterations per property, each
  tagged `Feature: aggregated-space-planning-cost, Property N: …`). Properties 1-8 and 10-14 run on CPU
  against small synthetic tensors and a stand-in `nn.Sequential` head, so the suite needs no GPU, no dataset
  and no checkpoint. Property 1 compares raw bytes
  (`t.detach().cpu().numpy().tobytes()`), not `torch.equal`, because `torch.equal` treats `nan` as unequal to
  itself and would mask exactly the failure the property exists to catch.
- **Scope guard** (`tests/test_scope_guard.py`, Property 9). Non-optional gate, run before every launch.
- **Integration tests**, example-based, 1-3 cases each, not property tests: `aggregate_results.py` parses a
  wrapper-shaped output tree (Requirement 2.7); a tiny synthetic checkpoint yields the expected head
  (Requirement 2.4); encoder parameter hashes are unchanged across a short run (Requirement 8.5).
- **Paired zero-weight check** (Requirement 3.3), once, on the pod: `plan.py` and `plan_agg.py` with
  `+agg_weight=0` at seed 400 open-loop must produce identical per-episode success vectors. This is the
  end-to-end confirmation that the bitwise-zero design holds through a real run, and it gates the sweep.
- Requirements 9.1-9.5, 7.2, 7.5, 11.1, 11.6 and 11.7 are process rules recorded in the result document, not
  automated tests.

## Follow-on phases, sketched

Recorded so this design does not need redoing if the gate passes. Requirement 11.7 applies: nothing below
starts without recorded approval.

**Phase 2 — learned reachability / temporal-distance cost, world model frozen.** Replace the fixed Euclidean
goal distance with a small learned head predicting temporal distance (steps-to-goal) between a latent and a
goal latent, trained on the existing dataset with the world model frozen, and used as `L_plan` in place of or
alongside `L_spatial`. Related work (arXiv 2605.22164, arXiv 2607.25337) uses such costs to *rank* candidate
trajectories, which suits CEM. This cell plans by gradient descent through 100 Adam steps, so a ranking head
is not enough: the head must be differentiable in the latent argument and well conditioned, with bounded
gradients and no plateaus, or the planner will stall or diverge where the quadratic cost does not. Design work
in that phase is mostly about conditioning (output parameterization, Lipschitz control, gradient-norm
instrumentation reusing the recorder built here), not about the head's accuracy. The injection machinery,
weight gate, instrumentation and paired-comparison tooling from this phase carry over unchanged; the learned
head slots into the same `AGG_CONTEXT` position as Agg_Head.

**Phase 3 — 2x2 study: {straightening on, off} x {Euclidean cost, learned cost}.** Four cells, same
Evaluation_Protocol, same seeds, same gate. The question is whether a learned cost substitutes for
straightening or compounds with it: if the learned cost recovers most of the straightening gain on its own, the
straightening claim is a conditioning artifact of the Euclidean cost rather than a property of the
representation. Needs one additional trained checkpoint (straightening off), which is the only retraining in
the sequence; everything else is evaluation.
