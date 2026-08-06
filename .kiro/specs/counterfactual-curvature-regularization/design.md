# Design Document

## Overview

Counterfactual Curvature Regularization (CCR) adds a second curvature penalty to the training objective of
`VWorldModel`. The existing penalty straightens latent trajectories that the *dataset* visited. CCR
straightens latent trajectories that the *predictor imagines* when driven by perturbed actions, which is the
region `GDPlanner` actually traverses (it starts from a zero action sequence and takes 100 Adam steps).

CCR is counterfactual by construction, so the actions driving the imagined rollout need not be *recorded*
ones. `training.ccr_action_source` selects between two variants. `logged` perturbs the recorded normalized
actions, and is therefore bounded by the training window: `num_hist + L - 1 <= num_frames` caps the imagined
horizon at `L = 2` for the PushT target cell. `synthetic` (**the default**) keeps the recorded prefix but
synthesizes actions directly in normalized action space for imagined steps past the window edge, reaching the
full Planner_Horizon of `L = 5` with **no** change to `num_frames`, `num_hist`, `frameskip` or any other
value in Protocol_Invariants, and still at zero additional encoder forward passes. Both variants are piloted
against each other: `synthetic` buys horizon at the price of extrapolating past any real observation, and
`logged` is the built-in control that isolates exactly that price.

The design has one governing constraint that shapes almost every decision: **the model must gain no new
state**. `VWorldModel` is instantiated in `train.py` *after* `accelerator.prepare()` and is never itself
prepared, so any `nn.Module` created inside `VWorldModel.__init__` keeps CPU parameters and the run dies
about two seconds into epoch 1 with `Expected all tensors to be on the same device`. It would also never be
registered in an optimizer, so it would never learn. CCR therefore reuses existing modules only and
allocates its one new tensor (the action perturbation) with `torch.empty_like(act)`, which inherits device
and dtype from an input that `accelerate` already placed. There is no constructor in the CCR path at all, so
the failure mode is structurally unreachable rather than merely avoided.

Everything new is off by default. With `lambda_cf = 0` and `mca_weight = 0` the forward pass takes exactly
the pre-feature code path, the total loss is bitwise identical, and the Hydra run directory string is
byte-identical, including `pusht_aggmlpcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05`.

Language: Python 3.10 / PyTorch, Hydra + OmegaConf, as in the rest of the repo.

## Scope

In scope (Requirement 5.6): `models/visual_world_model.py`, `conf/train.yaml`, `train.py`,
`custom_resolvers.py` (the Run_Naming expression), and new standalone scripts
(`probe_ccr_curvature.py`, `summarize_training_log.py`, `run_ccr_pilot.sh`, `tests/`).

Out of scope: `planning/*.py`, `conf/plan_gd*.yaml`, `datasets/*`, every Table 1 cell other than PushT
`DINOv2 (patch) + proj, 14x14x8, L_curv ✓`, and Direction E (action-Gramian conditioning).

## Architecture

```
train.py (Trainer)
  ├─ _guard_run_dir()            NEW  abort if run dir holds a ckpt with a different loss signature (6.6)
  ├─ iteration budget log        NEW  steps/epoch, epochs, total, cap                          (6.1-6.3)
  ├─ init_models() ──────────────────▶ VWorldModel(..., lambda_cf, ccr_rho, ccr_rollout_len,
  │                                                 ccr_action_source, mca_weight)
  ├─ train() loop
  │    ├─ model(obs, act) ───────────▶ loss, loss_components  (now may include ccr_*/mca_*)
  │    ├─ optimizer steps
  │    ├─ global_iter += 1        NEW  checkpointed counter                                     (6.3)
  │    ├─ _write_telemetry()      NEW  training_log.jsonl: scaled values, shares, it/s          (6.7, 6.8)
  │    └─ cap reached ──▶ save_ckpt(), stop mid-epoch                                          (6.1)
  └─ save_ckpt()                       now also persists global_iter

models/visual_world_model.py (VWorldModel)
  forward(obs, act)
    z = encode(obs, act)                     ← unchanged, single encoder pass
    ... prediction / vcreg / curvature ...    ← unchanged
    if self.ccr:                    NEW      cheap bool, checked before any tensor work        (3.3)
        ccr = compute_ccr(z, act)
        loss += lambda_cf * ccr
    if self.mca:                    NEW
        loss += mca_weight * compute_mca(z)

  compute_ccr(z, act)               NEW
    act_cf  = _ccr_actions(act)                              logged | synthetic base + delta   (1.2, 1.3)
    z_ctx   = replace_actions_from_z(z[:, :num_hist].clone(), act_cf[:, :num_hist])  ← EXISTING
    z_imag  = _rollout_latents(z_ctx, act_cf[:, num_hist:])                          ← EXISTING body
    feats   = visual_only(z_imag[:, -(L+2):])
    return total_curvature(feats, mode="aggcos")                                     ← EXISTING

  _ccr_actions(act)                 NEW
    base  = act[:, :min(required, available)]                 recorded prefix
    base  = cat([base, zeros(required - available)])          only under ccr_action_source=synthetic
    return base + _sample_action_perturbation(base)           one sampler, one RNG mechanism    (1.3, 1.8)

  rollout(obs_0, act)               REFACTORED (behaviour preserved)
    z = encode(obs_0, act[:, :n]); z = _rollout_latents(z, act[:, n:]); return separate_emb(z), z

custom_resolvers.py
  ccr_tag(lambda_cf, rho, action_source, mca_weight)  NEW
                                        "" at defaults, "_cf..._rho..._src..._mca..." otherwise  (6.4, 6.5)
```

## Configuration surface

All six knobs are Hydra keys with no Python literal fallback (Requirement 3.5). Added to the `training`
block of `conf/train.yaml`:

```yaml
training:
  # --- Counterfactual Curvature Regularization (default OFF) ---
  lambda_cf: 0.0            # weight on CCR_Term; 0 disables the whole path
  ccr_rho: 0.0              # perturbation radius in normalized-action units (dimensionless)
  ccr_rollout_len: 5        # imagined predictor steps; default == Planner_Horizon (25/5)
  ccr_action_source: synthetic   # 'logged' | 'synthetic'; see below for why synthetic is the default
  # --- Metric-Consistent Aggregation (pilot only, default OFF) ---
  mca_weight: 0.0
  # --- Pilot infrastructure ---
  max_iterations: 0         # <=0 means "run the configured epochs" (current behaviour)
  telemetry_every_x_iterations: 200
  save_every_x_iterations: 1000   # unchanged; fires at i == 0 so an empty ckpt dir means a crash (6.9)
```

`train.py` forwards the five loss knobs into the Hydra `instantiate` call that builds `VWorldModel`,
alongside the existing `straighten` / `stop_grad` / `vcreg*` arguments.

**Why `synthetic` is the default.** The two candidate defaults are not symmetric. `ccr_rollout_len` defaults
to `5` because Requirement 1.4 ties it to Planner_Horizon, and under `logged` that default is *infeasible on
the target cell*: `num_hist + 5 - 1 = 7 > num_frames = 4`, so anyone enabling CCR with `lambda_cf=0.1` and
nothing else would hit the Requirement 1.10 `ValueError` at the first forward. A default pair that cannot run
is a trap, and lowering `ccr_rollout_len` to make `logged` self-consistent is not available (Requirement 1.4).
With `synthetic`, the shipped defaults are internally coherent: enabling CCR alone regularises the full
five-step planning horizon, which is what Requirement 1.4 was asking for in the first place. `logged` remains
one override away and is the arm the pilot uses as its control. The default choice is invisible to legacy
behaviour either way, because at `lambda_cf = 0` no CCR code runs and `ccr_tag` returns the empty string.

The Run_Naming expression in `conf/train.yaml` gains one appended interpolation (both `hydra.run.dir` and
`hydra.sweep.dir`), leaving the existing expression untouched:

```yaml
    dir: ${ckpt_base_path}/test/...._sg${training.stop_grad}_lr${training.encoder_lr}${ccr_tag:${training.lambda_cf},${training.ccr_rho},${training.ccr_action_source},${training.mca_weight}}
```

## Components and Interfaces

### 1. Perturbation sampler and action sources

```python
def _sample_action_perturbation(self, act: torch.Tensor) -> torch.Tensor:
    """Uniform perturbation, elementwise bounded by rho in normalized-action units."""
    rho = torch.as_tensor(self.ccr_rho, dtype=act.dtype, device=act.device)
    return torch.empty_like(act).uniform_(-1.0, 1.0).mul_(rho)
```

Three properties come out of this three-line body:

- **Bounded by construction.** `uniform_(-1, 1)` is a closed interval and multiplication by a non-negative
  `rho` is monotone, so every element lies in `[-rho, +rho]` as `rho` is represented in `act.dtype`
  (Requirement 1.3). No clamping, no rejection sampling, no distribution tail to argue about.
- **`rho = 0` is not a special case.** It produces exact zeros, so the control arm and the treatment arm run
  the identical code path and consume the identical amount of RNG. The only difference between the arms is
  the numerical value of the perturbation (Requirements 2.1, 2.2, 2.3). There is deliberately **no**
  `if rho == 0` branch.
- **No dataset constant.** `rho` is the only scalar. `normalize_action: True` makes PushT actions
  unit-variance per dimension via `ACTION_MEAN` / `ACTION_STD` inside the dataset, so `rho` is expressed in
  standard deviations and is dimensionless. Nothing environment-specific reaches this function
  (Requirement 1.8). `empty_like` also inherits device and dtype, which is what makes the device bug
  unreachable.

The perturbation is applied to the **whole** action window, including the `num_hist` context actions. That
matters: perturbing the context actions is what moves the predictor's conditioning off-log rather than only
its final steps.

**Action sources.** The sampler above supplies the *offset*; `ccr_action_source` supplies the *base* the
offset is added to. A single helper builds the full `required`-length action sequence:

```python
def _ccr_actions(self, act: torch.Tensor, required: int) -> torch.Tensor:
    """Base action sequence of length `required`, plus one bounded uniform perturbation."""
    n_logged = min(required, act.shape[1])
    base = act[:, :n_logged]                                   # recorded, normalized
    if required > n_logged:                                    # ccr_action_source == "synthetic" only
        b, _, d = base.shape
        base = torch.cat([base, base.new_zeros(b, required - n_logged, d)], dim=1)
    return base + self._sample_action_perturbation(base)
```

The helper takes no `ccr_action_source` argument, and that is deliberate: the source is enforced *upstream*,
by the guard in `compute_ccr` that rejects `required > available` under `logged`. So under `logged` the
padding branch is unreachable, and the difference between the two variants lives in exactly one `if` in one
place instead of being threaded through the rollout path. Consequences worth being explicit about:

- **The recorded prefix is perturbed under both sources.** `synthetic` does not discard the logged actions;
  it keeps and perturbs every recorded frame it can reach and only *appends* where the window runs out. The
  two variants therefore differ **only past the window edge**, which is what makes them a clean pair of
  pilot arms rather than two unrelated experiments.
- **Whenever `required <= num_frames` the two sources are bitwise identical.** No zero padding is appended,
  the sampler is called on a tensor of the same shape, and it consumes the same RNG draws. `logged` versus
  `synthetic` is a difference that only exists at horizons the window cannot supply (Property 17).
- **The synthesized action for an imagined step is `0 + U[-rho, rho]^d`.** There is no second RNG mechanism,
  no separate scale and no second distribution to reason about: the same bounded uniform sampler, scaled by
  the same `rho`, is the only stochastic element in the CCR path (Requirement 1.8).

**Why zero-centred, `rho`-bounded uniform is the right synthetic distribution.** The justification is
entirely from the protocol, with no dataset-specific constant:

1. `normalize_action: True` makes training actions unit-variance per dimension, so normalized action space
   is the space in which "a step of size `rho`" is meaningful without reference to PushT's units. `rho` is
   dimensionless in exactly the same way it is for the `logged` source.
2. `GDPlanner` initialises its action sequence from **zeros** (`sample_type: zero`, `action_noise: 0`) and
   then takes 100 Adam steps at `lr: 0.1`. Its iterates are therefore small-magnitude, zero-centred, and
   untethered from any logged action — they are not "recorded actions plus noise", they are whatever
   gradient descent produces starting from the origin. A zero-centred bounded ball in normalized action
   space is the closest cheap description of that region.
3. At `rho = 0` the synthesized suffix is exactly zero, i.e. the imagined rollout past the window edge is
   driven by the planner's *initialisation point*. The control arm is thus still a genuine code-path twin
   (Requirements 2.2, 2.3), and its interpretation under `synthetic` sharpens rather than blurs: it measures
   curvature along the trajectory the planner starts from. Requirement 2.1's literal "unperturbed recorded
   normalized actions" is satisfied for every frame the window actually records; frames past the edge have no
   recorded value to be unperturbed with respect to, and take the planner's zero initialisation instead.

### 2. Rollout body extraction

`rollout` currently interleaves encoding with the predictor loop. CCR needs the loop without the encoding,
because re-encoding would double the DINOv2 forward pass, which is the dominant cost of a step. The loop is
extracted verbatim into `_rollout_latents`, and `rollout` delegates to it:

```python
def _rollout_latents(self, z, action):
    """Predictor rollout body. Identical tensor ops, in identical order, to the previous rollout loop."""
    t, inc = 0, 1
    while t < action.shape[1]:
        z_pred = self.predict(z[:, -self.num_hist:])
        z_new = self.replace_actions_from_z(z_pred[:, -inc:, ...], action[:, t:t + inc, :])
        z = torch.cat([z, z_new], dim=1)
        t += inc
    z_pred = self.predict(z[:, -self.num_hist:])
    return torch.cat([z, z_pred[:, -1:, ...]], dim=1)

def rollout(self, obs_0, act):
    num_obs_init = obs_0["visual"].shape[1]
    z = self.encode(obs_0, act[:, :num_obs_init])
    z = self._rollout_latents(z, act[:, num_obs_init:])
    z_obses, _ = self.separate_emb(z)
    return z_obses, z
```

`rollout` keeps its signature, its return type and its numerics, so `plan.py`, `planning/*` and
`Trainer.openloop_rollout` are unaffected. The refactor is guarded by a property test against a frozen copy
of the original loop (Property 7). CCR therefore rolls forward through the same rollout body that
Rollout_Function uses, and uses no latent-geometry code of its own (Requirements 1.2, 1.7).

### 3. CCR term

```python
def compute_ccr(self, z, act):
    L = self.ccr_rollout_len
    required = self.num_hist + L - 1            # actions needed for L predictor steps
    available = act.shape[1]
    if self.ccr_action_source == "logged" and required > available:
        raise ValueError(
            f"CCR is enabled (lambda_cf={self.lambda_cf}) with ccr_rollout_len={L} and "
            f"ccr_action_source='logged', which needs an action sequence of length {required} "
            f"(num_hist={self.num_hist} + {L} - 1), but only {available} action frames are available "
            f"(num_frames={available}). Set training.ccr_rollout_len <= "
            f"{available - self.num_hist + 1}, or set training.ccr_action_source=synthetic to "
            f"synthesize the {required - available} action frames past the window edge."
        )
    act_cf = self._ccr_actions(act, required)   # logged prefix (+ synthesized tail), all perturbed
    z_ctx = self.replace_actions_from_z(z[:, :self.num_hist].clone(), act_cf[:, :self.num_hist])
    z_imag = self._rollout_latents(z_ctx, act_cf[:, self.num_hist:required])   # num_hist + L frames
    feats = self.visual_only(z_imag[:, -(L + 2):])
    return self.total_curvature(feats, mode="aggcos")
```

**Where the latents come from.** `z` is the tensor `forward` already computed with a single
`encode(obs, act)` call. CCR takes its first `num_hist` frames, clones them (because
`replace_actions_from_z` writes in place, and mutating a view of `z` would corrupt the baseline term's
autograd graph), and overwrites only the action channels with the perturbed actions. The visual and proprio
channels are reused, so **CCR costs zero additional encoder forward passes under either action source** — it
costs `L` extra predictor calls (5 for the `synthetic` default, 2 for the `logged` arm), one extra
`action_encoder` call, and one `encoder.agg` call over `b * (L + 2)` token sets. Gradients
flow into the encoder, the action encoder and the predictor, which is the point: the encoder is what CCR is
meant to pressure.

**Rollout length arithmetic.** `_rollout_latents` appends `len(action) + 1` frames, so `L` predictor steps
need `required = num_hist + L - 1` action frames. Under Protocol_Invariants the training window is
`num_frames = 4` with `num_hist = 3`, so the recorded window supplies at most `4 - 3 + 1 = 2` predictor
steps. What happens at larger `L` depends entirely on `ccr_action_source`:

| `ccr_action_source` | `L = 5` on the target cell | maximum `L` | pilot override needed |
|---|---|---|---|
| `logged` | `ValueError` at the first CCR forward (Requirement 1.10) | `num_frames - num_hist + 1 = 2` | `training.ccr_rollout_len=2` |
| `synthetic` (default) | feasible: 4 recorded action frames perturbed, 3 synthesized | unbounded | none |

Under `logged` the abort is deliberate: silently truncating `L` would make two differently-named runs
numerically identical, and lengthening the window would violate Protocol_Invariants (`num_frames = 4`). The
message names `lambda_cf`, the requested length, the required length, the available length, the maximum
permitted value, and the one-token `synthetic` escape hatch.

Under `synthetic` the same `L = 5` is feasible and must not raise, because `required - available = 3` action
frames are synthesized rather than read. `L = 5` equals Planner_Horizon, so CCR regularises the whole
planning horizon while every value in Protocol_Invariants — `num_frames = 4`, `num_hist = 3`, `frameskip = 5`
included — is untouched. The earlier conclusion that reaching `L = 5` requires `num_frames = 7` held only
under the assumption that imagined actions must be recorded actions, which CCR has no reason to assume.

**Curvature window.** `total_curvature` needs at least 3 frames. The window is the last `L + 2` frames of
`z_imag` (which has `num_hist + L` frames in total): enough that every velocity pair entering
`_cos_curvature` touches at least one imagined frame, and no more, so the purely-real triple that the
baseline term already penalises is not double-counted. A `W`-frame window yields `W - 1` velocities and
`W - 2` curvature triples, so:

- **`L = 5` (`synthetic` default).** `z_imag` has 8 frames, indices 0-2 real context and 3-7 imagined. The
  window is the last 7 frames (indices 1-7), giving 6 velocities and **5 curvature triples**. The purely-real
  triple (0,1,2) is excluded exactly as intended. Two seam triples remain — (1,2,3) and (2,3,4) each include
  at least one real frame — and three are purely imagined: (3,4,5), (4,5,6), (5,6,7). So a real/imagined seam
  term still exists, but at `L = 5` it is 2 of 5 terms rather than the whole penalty.
- **`L = 2` (`logged`).** `z_imag` has 5 frames, indices 0-2 real and 3-4 imagined. The window is the last 4
  frames (indices 1-4), giving 3 velocities and **2 curvature triples**, (1,2,3) and (2,3,4). Both touch a
  real frame, so at this horizon the penalty is entirely seam terms. That is the sharpest single reason the
  `logged` arm is a control and not the headline configuration.

**Channel selection.** `visual_only` — the same selection the baseline curvature term uses
(`feats = self.visual_only(z)` in `forward`). For the target cell this is `196 x 8`. Note a hard constraint:
`aggcos` routes tokens through `encoder.agg`, whose `agg_mlp` has a fixed input width of `196 * emb_dim =
196 * 8 = 1568`. Feeding visual **and** proprio channels (`196 x 18`) would raise a shape error inside the
aggregation MLP. Requirement 1.5's operative clause, "matching the channel selection used by the
Baseline_Objective curvature term", is therefore the one implemented; its parenthetical
"visual-and-proprio" is not reachable with `agg_type: mlp` and is recorded here as a known deviation.
Action channels are excluded either way, which is the substantive requirement.

### 4. MCA term (pilot only)

Straightness is enforced in aggregated space (`R^128` after `agg_post_norm`) while `planning/objectives.py`
scores distances in the `196 x 8` patch space. MCA penalises the aggregation map for distorting velocity
norms. It only needs `agg` to be a *similarity* (distance-preserving up to one global constant) for
straightness to transfer, so the penalty is made scale-invariant:

```python
def compute_mca(self, z, eps=1e-6):
    feats = self.visual_only(z)                                  # (b, t, 196, 8)
    b, t, p, d = feats.shape
    agg = self.encoder.agg(feats.reshape(b * t, p, d)).reshape(b, t, -1)
    v_patch = (feats[:, 1:] - feats[:, :-1]).flatten(2).norm(dim=-1)   # (b, t-1)
    v_agg = (agg[:, 1:] - agg[:, :-1]).norm(dim=-1)                    # (b, t-1)
    r = v_agg / (v_patch + eps)
    r_bar = r.mean().detach().clamp_min(eps)
    return ((r / r_bar) - 1.0).pow(2).mean()
```

`encoder.agg` is an existing module; MCA adds no module and no parameter (Requirement 4.3). MCA is excluded
from the primary-claim configuration: the Acceptance_Gate is evaluated at `mca_weight = 0`
(Requirements 4.5, 4.6), and if its own pilot gate fails it never enters the Full_Run.

### 5. Gating, no-new-state guarantee, and startup logging

`__init__` stores four floats/ints, one string and two booleans. Nothing else:

```python
CCR_ACTION_SOURCES = ("logged", "synthetic")

self.lambda_cf = float(lambda_cf)
self.ccr_rho = float(ccr_rho)
self.ccr_rollout_len = int(ccr_rollout_len)
self.ccr_action_source = str(ccr_action_source)
self.mca_weight = float(mca_weight)
for name, value in (("lambda_cf", self.lambda_cf), ("ccr_rho", self.ccr_rho),
                    ("mca_weight", self.mca_weight)):
    if value < 0:
        raise ValueError(f"training.{name} must be >= 0, got {value}.")
if self.ccr_action_source not in CCR_ACTION_SOURCES:
    raise ValueError(f"training.ccr_action_source must be one of {CCR_ACTION_SOURCES}, "
                     f"got {self.ccr_action_source!r}.")
self.ccr = self.lambda_cf > 0          # cheap boolean gate
self.mca = self.mca_weight > 0
```

`ccr_action_source` is validated eagerly even when `lambda_cf = 0`, because a typo in an unused knob that
only surfaces once the term is enabled is exactly the class of mistake a pilot cannot afford. Validation is a
string comparison, so it adds no tensor work and does not touch Requirements 3.2 or 3.3.

In `forward`, both terms sit behind the boolean, mirroring `if self.straighten and self.straighten_scale > 0:`
so that the disabled path evaluates one attribute lookup and one comparison and performs no tensor work, no
extra predictor rollout and no extra encoder pass (Requirements 3.2, 3.3):

```python
if self.ccr:
    ccr_loss = self.compute_ccr(z, act)
    loss = loss + ccr_loss * self.lambda_cf
    loss_components["ccr_loss"] = ccr_loss
    loss_components["ccr_loss_scaled"] = ccr_loss * self.lambda_cf
if self.mca:
    mca_loss = self.compute_mca(z)
    loss = loss + mca_loss * self.mca_weight
    loss_components["mca_loss"] = mca_loss
    loss_components["mca_loss_scaled"] = mca_loss * self.mca_weight
```

The startup log lines (Requirement 3.6) are emitted from `__init__`, after `accelerate` has already placed
the submodules, so the device they report is the device the terms will actually compute on:

```
CCR enabled: term=ccr, weight(lambda_cf)=0.1, rho=0.05, rollout_len=5, action_source=synthetic,
             synthesized_action_frames=3, curvature_mode=aggcos, device=cuda:0
MCA enabled: term=mca, weight(mca_weight)=0.01, rho=n/a, device=cuda:0
CCR disabled (lambda_cf=0.0); MCA disabled (mca_weight=0.0)
```

`action_source` is on the line because it is the knob that decides which of two pilot arms is running, and
`synthesized_action_frames` (`max(0, num_hist + L - 1 - num_frames)`, so `0` under `logged`) makes the
distinction visible as a number rather than a word — a `synthetic` run that synthesizes nothing is a `logged`
run and should be read as one.

The device is read with `next(self.parameters()).device`. This line is the two-minute smoke check from the
pilot checklist: term enabled, right device, before any GPU hours are spent.

### 6. Run naming: the `ccr_tag` resolver

`run_naming.variant_tag()` referenced in `SHORT_BUDGET_PILOTS.md` belongs to a different branch and does not
exist in this repo; run directories here are derived entirely inside the Hydra `hydra.run.dir` /
`hydra.sweep.dir` expression in `conf/train.yaml`. The knob-isolation contract is therefore implemented as a
new OmegaConf resolver, registered in `custom_resolvers.py` next to `replace_slash` and `replace_substring`
(both already registered at import time by `import custom_resolvers` in `train.py`, which runs before
`@hydra.main` resolves anything):

```python
# (lambda_cf, ccr_rho, ccr_action_source, mca_weight)
CCR_TAG_DEFAULTS = (0.0, 0.0, "synthetic", 0.0)

def _fmt_num(value: float) -> str:
    # '0.1' -> '0p1', '0.05' -> '0p05', '1e-05' -> '1e-05'; keeps paths free of '.'
    return f"{float(value):g}".replace(".", "p")

def ccr_tag(lambda_cf, rho, action_source, mca_weight) -> str:
    values = (float(lambda_cf), float(rho), str(action_source), float(mca_weight))
    if values == CCR_TAG_DEFAULTS:
        return ""                                    # Requirement 6.5 / 3.4
    return "_cf{}_rho{}_src{}_mca{}".format(                                # Requirement 6.4
        _fmt_num(values[0]), _fmt_num(values[1]), values[2], _fmt_num(values[3]))

OmegaConf.register_new_resolver("ccr_tag", ccr_tag)
```

Because the resolver returns the empty string at defaults and is appended to the end of the existing
expression, the resolved path at defaults is character-for-character the legacy path:

| configuration | resolved run directory (under `./checkpoints/test/`) |
|---|---|
| defaults (`0.0, 0.0, synthetic, 0.0`) | `pusht_aggmlpcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05` |
| `lambda_cf=0.1 ccr_rho=0.05` (treatment, `L=5`) | `..._sgTrue_lr1e-05_cf0p1_rho0p05_srcsynthetic_mca0` |
| `lambda_cf=0.1 ccr_rho=0.05 ccr_action_source=logged ccr_rollout_len=2` | `..._sgTrue_lr1e-05_cf0p1_rho0p05_srclogged_mca0` |
| `lambda_cf=0.1 ccr_rho=0` (control) | `..._sgTrue_lr1e-05_cf0p1_rho0_srcsynthetic_mca0` |
| `lambda_cf=0.1 mca_weight=0.01` | `..._sgTrue_lr1e-05_cf0p1_rho0p05_srcsynthetic_mca0p01` |

All four values appear as soon as any one of them is non-default, so the control arm (`rho = 0`), the
`logged` arm and the `synthetic` treatment arm can never resolve to the same directory and silently
auto-resume each other — the exact collision that cost this project a run once before. `ccr_action_source`
has to be in the tag rather than only in the loss-signature guard, because two arms can legitimately differ
in *nothing else*: at any `L` the recorded window can supply, `logged` and `synthetic` are the same run name
under the old resolver while being two distinct comparisons in the write-up.

`ccr_rollout_len` is deliberately **not** in the tag. It is in `LOSS_SIGNATURE_KEYS` instead, so two runs
differing only in `L` collide on the directory and the guard in §8 aborts loudly with the differing key
named, rather than resuming one from the other. That is the safer failure for a knob whose feasible range
depends on another knob. Negative values are rejected in `VWorldModel.__init__` and `ccr_action_source` is
restricted to two known words, so no tag ever contains a `-` or a path separator.

### 7. Iteration cap

`train.py` gains one integer of state, `self.global_iter`, initialised to `0` next to `self.epoch = 0` and
appended to `self._keys_to_save`, so `save_ckpt` persists it and `load_ckpt` restores it. A resumed pilot
honours the same total bound (Requirement 6.3). Legacy checkpoints have no `global_iter` key; the existing
`Keys not found in ckpt` warning fires and the counter starts at 0, which is documented rather than
silently patched.

At startup the budget is logged so the cap can be set from real arithmetic rather than a guess:

```
Iteration budget: steps/epoch=61929 epochs=3 total=185787 max_iterations=8000 (cap active)
```

In `train()`, after the optimizer steps for a batch:

```python
self.global_iter += 1
if 0 < self.max_iterations <= self.global_iter:
    self._stop_requested = True
    self._write_telemetry(loss_components, force=True)
    self.logs_flash_iter(iteration=i)
    self.save_ckpt()
    log.info("Iteration cap reached: global_iter=%s == training.max_iterations=%s; "
             "stopping mid-epoch (epoch %s, batch %s). Validation skipped: the epoch is incomplete.",
             self.global_iter, self.max_iterations, self.epoch, i)
    break
```

and in `run()`, the epoch loop breaks when `self._stop_requested` is set, skipping `val()` and
`logs_flash(step=...)`. Skipping `logs_flash` is required, not cosmetic: it formats `train_loss` and
`val_loss` and would raise `KeyError` with no validation pass. A partial-epoch validation number would also
invite comparison against full-epoch numbers, which is precisely the mistake `SHORT_BUDGET_PILOTS.md` §7b
warns about. Pilot judgement comes from telemetry and the offline probe.

The cap can only ever shorten a run: with `max_iterations <= 0` (the default) the loop condition is never
true and behaviour is exactly as today (Requirement 6.2), and with a positive cap the number of executed
steps is `min(cap, epochs * steps_per_epoch)`. That is why the pilot recipe sets `training.epochs=3` for a
2-epoch dataset — a generous epoch count guarantees the cap, not an epoch boundary, is what ends the run.

### 8. Run-directory loss-configuration guard

A "loss signature" is the set of configuration values that change the objective:

```python
LOSS_SIGNATURE_KEYS = ("straighten", "stop_grad", "vcreg", "vcreg_std_coeff", "vcreg_cov_coeff",
                       "lambda_cf", "ccr_rho", "ccr_rollout_len", "ccr_action_source", "mca_weight")
```

`Trainer.__init__` calls `_guard_run_dir()` immediately after `cfg["saved_folder"]` is set and **before**
`wandb.init`, before `hydra.yaml` is written and before any checkpoint is written:

1. If `checkpoints/model_latest.pth` does not exist in the resolved run directory, there is nothing to
   conflict with. Write `loss_config.json` (main process only) and return.
2. Otherwise read the recorded signature from `loss_config.json`; if absent (a run predating this feature),
   fall back to the resolved `hydra.yaml` the previous run wrote, treating missing keys as their defaults;
   if neither exists, log a warning naming the directory and proceed, so legacy runs stay resumable.
3. If the recorded signature differs from the current one, raise `RuntimeError` naming the directory and the
   differing keys with both values, and write nothing (Requirement 6.6).

Hydra itself creates the output directory and its `.hydra/` config snapshot before user code runs; the guard
covers every *training* artifact (checkpoints, `hydra.yaml`, `training_log.jsonl`, plots), which is what makes
an accidental overwrite destructive.

### 9. Telemetry

The repo currently sends loss components to wandb only, which under `WANDB_MODE=offline` means they are not
readable from the shell. A JSONL sink is added: `training_log.jsonl`, appended in the run directory (Hydra's
cwd), main process only, one object per logged iteration, flushed on write. Cadence is
`training.telemetry_every_x_iterations` (default 200, matching the reference run's telemetry so step-200 rows
are directly comparable), plus the final step of a capped run.

The record carries each term's **scaled** contribution and its share of the total (Requirement 6.7), and the
observed step rate (Requirement 6.8):

```json
{"global_iter": 4000, "epoch": 1, "iter_in_epoch": 4000, "wall_time_s": 2216.4,
 "it_per_s": 1.81, "loss": 0.2166,
 "terms": {"prediction": {"scaled": 0.0118, "share": 0.0545},
           "curvature":  {"scaled": 0.0673, "share": 0.3107},
           "ccr":        {"scaled": 0.0412, "share": 0.1902},
           "decoder":    {"scaled": 0.0963, "share": 0.4446}},
 "enabled_terms": ["prediction", "curvature", "ccr", "decoder"],
 "ccr": {"raw": 0.412, "lambda_cf": 0.1, "rho": 0.05, "rollout_len": 5,
         "action_source": "synthetic", "synthesized_action_frames": 3}}
```

Term registry (component key in `loss_components` → telemetry name): `z_loss` → `prediction`,
`curvature_loss_scaled` → `curvature`, `ccr_loss_scaled` → `ccr`, `mca_loss_scaled` → `mca`,
`z_vcreg_loss_scaled` → `vcreg`, `decoder_loss_reconstructed` → `decoder`. Shares are
`scaled / loss_components["loss"]` for the same iteration, computed from the already-gathered per-iteration
dict, so the numbers are the ones that actually drove that step rather than an epoch mean.

`curvature_loss_scaled` is added to `loss_components` (the existing unscaled
`curvature_loss_used_for_training` is kept) so the baseline term is comparable with the new ones. Adding keys
to `loss_components` does not change the loss, so Requirement 3.2 is untouched.

The `ccr` block records `action_source` next to `rho` and `rollout_len` so that a JSONL file is
self-describing: reading a pilot's telemetry six weeks later must not require reconstructing which arm it was
from the directory name alone.

`it_per_s` is `iterations_since_last_record / elapsed_seconds` from `time.perf_counter()`, which makes the
step-rate comparison against the ~2.9 it/s PushT reference (`REPRODUCTION.md`) a lookup rather than a
stopwatch exercise. A drop below ~1.93 it/s is the >50% step-time regression that Requirement 11.7 requires
be reported before the Full_Run.

`summarize_training_log.py <run_dir>` reads the JSONL and prints the term/scaled/share table, the step rate,
the step-200 row, and with `--compare <reference_run_dir>` the row-by-row delta against a reference run
(Requirements 8.3, 8.4). `--collapse-check` flags a term whose share falls below 0.1% within the first 1,000
iterations, which is the "absorbed the task without pressuring the encoder" failure of Requirement 8.6.

### 10. Offline probe

`probe_ccr_curvature.py` is standalone, read-only and CPU-only. It answers one question before any GPU hours
are spent: does perturbing actions actually bend the imagined latent trajectory more than the recorded
actions do? If the gap is ~0 there is nothing for CCR to fix.

```bash
python probe_ccr_curvature.py \
  --ckpt   checkpoints/test/pusht_aggmlpcos1e-1_.../checkpoints/model_2.pth \
  --train-cfg checkpoints/test/pusht_aggmlpcos1e-1_.../hydra.yaml \
  --rho 0.05 --rollout-len 5 --action-source synthetic --num-windows 64 --draws 4 \
  --reference pristine --max-minutes 30 --out probe_outputs/ccr_pusht.json
```

Flow:

1. **Path validation first.** If `--ckpt` or `--train-cfg` is missing, print the missing absolute path and
   `sys.exit(1)` before any model is constructed or any weight is loaded (Requirement 7.5).
2. Record `sha256`, size and mtime of the checkpoint file.
3. `torch.load(..., map_location="cpu", weights_only=False)` and rebuild the model with the same helper
   shape `plan.load_model` uses (checkpoints store whole `nn.Module` objects, so no re-download is needed);
   `model.eval()`; everything under `torch.no_grad()`. No optimizer is constructed anywhere in the file
   (Requirement 7.1).
4. Sample `--num-windows` windows from the PushT validation split through the unmodified dataset loader with
   a fixed seed, giving `obs` (4 frames), `act` and `state`.
5. For each window: encode once, then evaluate `total_curvature(visual_only(z_imag[:, -(L+2):]), "aggcos")`
   under (a) unperturbed actions and (b) `--draws` independent perturbations at `rho`. `--action-source`
   selects the same `_ccr_actions` construction training uses, so the probe measures the arm that will be
   trained; `--action-source logged` is rejected with the same message as training when
   `num_hist + L - 1 > num_frames`. Only the action
   channels are re-encoded between draws, exactly as in training, which is what keeps the probe inside its
   CPU budget: 64 windows x 4 frames = 256 DINOv2-S/14 forwards total, a few minutes, with a
   `--max-minutes` wall-clock guard that stops early and marks the report `partial: true`
   (Requirement 7.6). At `L = 5` the predictor is called five times per draw instead of twice, i.e.
   `64 x 5 draws x 5 steps = 1600` predictor calls; the predictor is a small ViT over `num_hist` frames of
   tokens, so this stays far below the 256 encoder forwards and the 30-minute budget is unaffected.
6. **Readouts, disaggregated per state dimension** (Requirement 7.2). PushT `state` dims map to
   `agent_x, agent_y, block_x, block_y, block_angle` (dims 0-4; 5-6 are velocities and are not reported):
   - `curvature_gap` = mean(perturbed) − mean(unperturbed). Per dimension, the same quantity restricted to
     the windows in the top tercile of `|state_d[last] − state_d[first]|`, i.e. the windows in which that
     dimension is the dominant motion. Aggregating over all windows is exactly the mistake §4 of
     `SHORT_BUDGET_PILOTS.md` documents, where an aggregate improved while a single dimension collapsed.
   - `state_readout_r2` = ridge-regression R² from the aggregated latent to each state dimension, held-out
     split. Per dimension by construction.
7. **Reference values** (Requirement 7.3). Every readout entry carries `reference_value` and
   `reference_source` from `{pristine, early_telemetry, control_run}`: `pristine` recomputes the same
   statistic with an untrained model (DINOv2 from the hub cache plus freshly initialised projector,
   predictor, action and proprio encoders) — free, no training; `early_telemetry` reads the reference run's
   own first-8,000-step rows out of its JSONL, which is a free matched-budget control; `control_run` is only
   used if neither is available.
8. Re-hash the checkpoint and compare against step 2; abort with a loud error if it changed. Reports are
   written to `probe_outputs/`, never inside the checkpoint directory (Requirement 7.4).

**Probe gate, written before running** (Requirement 8.1): the mechanism is present if the aggregate
`curvature_gap` at `rho = 0.05` is positive and at least 20% of the unperturbed curvature magnitude, on at
least 3 of the 5 disaggregated dimension subsets. If it is not, no pilot is launched.

## Data models

**Loss components** (`loss_components` returned by `forward`), new keys only:

| key | present when | value |
|---|---|---|
| `ccr_loss` | `lambda_cf > 0` | raw CCR curvature, scalar tensor |
| `ccr_loss_scaled` | `lambda_cf > 0` | `ccr_loss * lambda_cf` |
| `mca_loss` | `mca_weight > 0` | raw MCA distortion |
| `mca_loss_scaled` | `mca_weight > 0` | `mca_loss * mca_weight` |
| `curvature_loss_scaled` | `straighten` | `curvature_loss * straighten_scale` |

**Loss signature** (`loss_config.json` in the run directory): a flat JSON object over
`LOSS_SIGNATURE_KEYS`, values as resolved by Hydra.

**Telemetry record** (`training_log.jsonl`): one JSON object per line, schema as shown in §9. Written with
`json.dumps` and read back with `json.loads`; the round trip is a tested property because the whole pilot
verdict is read out of this file.

**Probe report** (`probe_outputs/<name>.json`):

```json
{"ckpt": "...", "ckpt_sha256": "...", "rho": 0.05, "rollout_len": 5, "action_source": "synthetic",
 "synthesized_action_frames": 3, "num_windows": 64, "draws": 4,
 "partial": false, "elapsed_s": 611.2,
 "readouts": {
   "curvature_gap": {"aggregate": 0.083,
     "per_dim": {"agent_x": 0.101, "agent_y": 0.094, "block_x": 0.072, "block_y": 0.068, "block_angle": 0.041},
     "reference_value": 0.012, "reference_source": "pristine"},
   "state_readout_r2": {"aggregate": 0.28,
     "per_dim": {"agent_x": 0.991, "agent_y": 0.988, "block_x": 0.930, "block_y": 0.921, "block_angle": 0.514},
     "reference_value": 0.943, "reference_source": "pristine"}}}
```

## Error handling

| condition | behaviour |
|---|---|
| `lambda_cf`, `ccr_rho` or `mca_weight` negative | `ValueError` in `VWorldModel.__init__` naming the key and value |
| `ccr_action_source` not in `{logged, synthetic}` | `ValueError` in `VWorldModel.__init__` naming the key, the given value and the two permitted values; raised even at `lambda_cf = 0` |
| `ccr_action_source=logged` and `ccr_rollout_len` needs more actions than the window has | `ValueError` at the first CCR forward, naming `lambda_cf`, requested length, required length, available length, the maximum permitted value, and the `synthetic` alternative (Requirement 1.10) |
| `ccr_action_source=synthetic` and `ccr_rollout_len` needs more actions than the window has | **no error**: `required - available` action frames are synthesized in normalized action space and the rollout proceeds (Requirement 1.10 does not apply) |
| `ccr_rollout_len < 1` | `ValueError` under either action source: the curvature window `L + 2` would be under `total_curvature`'s 3-frame minimum |
| `encoder` has no `agg` | the existing `total_curvature` error is raised unchanged ("curvature mode 'aggcos' requires encoder.agg()") |
| run directory holds a checkpoint with a different loss signature | `RuntimeError` before any write, naming the directory and the differing keys |
| resolved run directory has a checkpoint but no recorded signature | warning naming the directory; proceed (legacy resume) |
| probe checkpoint or train config path missing | message with the absolute path, `sys.exit(1)`, no model constructed |
| probe exceeds `--max-minutes` | stop sampling, write the report with `partial: true`, exit 0 |
| probe detects the checkpoint hash changed | `RuntimeError` naming the file; the report is still written for forensics |

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system —
essentially, a formal statement about what the system should do. Properties serve as the bridge between
human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: The disabled path is the baseline path

For any observation window, action window and baseline loss configuration, a model with `lambda_cf = 0` and
`mca_weight = 0` produces a total loss and a set of shared loss components bitwise equal to those produced
by the pre-feature reference implementation on the same seeded inputs, and its counts of `encode`,
`encode_obs`, `predict` and `total_curvature` calls are equal to the reference's counts.

**Validates: Requirements 3.2, 3.3, 1.1, 4.1**

### Property 2: Run naming is empty at defaults, complete otherwise, and injective

For any legacy configuration (environment, encoder, `straighten`, `stop_grad`, `encoder_lr`), the resolved
run directory with the new keys at their defaults is byte-identical to the one the pre-feature template
produces; for any tuple `(lambda_cf, rho, ccr_action_source, mca_weight)` that is not the default tuple, the
contributed tag is non-empty and contains all four formatted values; and distinct tuples contribute distinct
tags, in particular two tuples differing only in `ccr_action_source` never contribute the same tag.

**Validates: Requirements 3.4, 6.4, 6.5**

### Property 3: Perturbations respect the radius, for recorded and synthesized actions alike

For any action tensor, any `rho >= 0` and either action source, every element of the sampled perturbation lies
in the closed interval `[-rho, +rho]` as `rho` is represented in the action tensor's dtype, and for `rho = 0`
every element is exactly zero; consequently every synthesized action frame produced under
`ccr_action_source = synthetic` also lies elementwise in `[-rho, +rho]`, and is exactly zero at `rho = 0`.

**Validates: Requirements 1.3, 2.1, 1.8**

### Property 4: The arms differ only by the perturbation

For any observation and action window, any action source and any `rho > 0`, running CCR with the perturbation
sampler forced to return zeros yields a CCR value, a rollout length, a curvature-input shape and a sequence of
internal method calls all exactly equal to those of a run with `rho = 0` and the same action source.

**Validates: Requirements 2.1, 2.2, 2.3**

### Property 5: CCR carries no dataset-specific constant

For any two model instances that differ only in environment- or dataset-derived configuration, and for either
action source, given identical seeded inputs and identical RNG state, the computed CCR values are equal; and
the CCR code path, including the synthesized-action construction, contains no numeric literal other than the
neutral constants `0`, `1`, `2` and the shared epsilon.

**Validates: Requirements 1.8, 11.4**

### Property 6: The iteration cap only ever shortens a run

For any positive `steps_per_epoch`, any positive `epochs`, any resume point `k >= 0` and any cap `c`, the
number of optimizer steps executed after resuming at `global_iter = k` is `max(0, min(c, epochs *
steps_per_epoch) - k)` when `c > 0`, and `epochs * steps_per_epoch` when `c <= 0`; in particular it never
exceeds the uncapped count.

**Validates: Requirements 6.1, 6.2, 6.3**

### Property 7: The rollout refactor preserves rollout

For any initial observation window and action sequence, `rollout(obs_0, act)` returns latents equal to those
produced by a frozen reference copy of the original rollout loop, element for element.

**Validates: Requirements 1.2, 1.7, 5.2**

### Property 8: CCR reuses the existing geometry machinery

For any enabled CCR configuration, computing CCR calls `total_curvature` with `mode="aggcos"` exactly once,
calls `predict` exactly `ccr_rollout_len` times, calls `replace_actions_from_z` for every imagined step, adds
zero calls to `encode_obs`, and hands `total_curvature` a tensor whose shape equals
`visual_only(z_imagined[:, -(L+2):])`.

**Validates: Requirements 1.2, 1.5, 1.7**

### Property 9: Enabling a term adds no state to the model

For any pair of configurations differing only in `lambda_cf`, `ccr_rho`, `ccr_rollout_len`,
`ccr_action_source` or `mca_weight`, the two constructed models have identical sets of `named_parameters`
keys, identical total parameter counts, and identical sets of `named_modules` keys; in particular neither
action source introduces a parameter, a module or a registered buffer.

**Validates: Requirements 1.6, 4.3**

### Property 10: Loss bookkeeping and shares are consistent

For any enabled configuration, `loss_components` contains each enabled term's raw and scaled entry with
`scaled == weight * raw`; every telemetry record's share equals its scaled value divided by the same
record's total loss, lies in `[0, 1]`, and survives a JSON write/read round trip unchanged; and the reported
step rate equals iterations divided by elapsed seconds.

**Validates: Requirements 1.9, 4.4, 6.7, 6.8**

### Property 11: Rollout length beyond the available actions is an error under `logged` only

For any `ccr_rollout_len` `L >= 1` and any action window length `A`, computing CCR with
`ccr_action_source = logged` raises an error whose message contains `lambda_cf`, `L`, the required length
`num_hist + L - 1`, `A` and the maximum permitted length `A - num_hist + 1` exactly when
`num_hist + L - 1 > A`, and returns a finite scalar otherwise; and computing CCR with
`ccr_action_source = synthetic` never raises for the same `(L, A)` pair, returning a finite scalar whose
imagined trajectory spans `L` predictor steps.

**Validates: Requirements 1.10, 1.4**

### Property 12: MCA measures scale-free aggregation distortion

For any latent window and any constant `c > 0`, the MCA value computed with `agg` replaced by `c * agg` equals
the value computed with `agg`; and for any `agg` that preserves velocity norms up to a single global factor,
MCA is zero.

**Validates: Requirements 4.2**

### Property 13: The loss-configuration guard aborts exactly on conflict

For any pair of loss signatures and any checkpoint-presence state, the run-directory guard raises exactly
when a checkpoint is present and the recorded signature differs from the current one, its message contains
the directory path and the differing keys, and no file in the directory is created or modified when it
raises.

**Validates: Requirements 6.6**

### Property 14: The probe is read-only and fully disaggregated

For any checkpoint, the probe leaves every loaded file byte-identical, leaves every model parameter bitwise
unchanged with no populated gradient, and emits for every readout an aggregate value, exactly the five
per-dimension entries `agent_x, agent_y, block_x, block_y, block_angle`, a non-null reference value and a
reference source from the allowed set.

**Validates: Requirements 7.1, 7.2, 7.3, 7.4**

### Property 15: The acceptance gate is a dual, margin-aware predicate

For any candidate and Platform_Baseline success-rate pair, the gate passes exactly when the candidate exceeds
77.33 open-loop, 85.33 MPC, the baseline open-loop rate and the baseline MPC rate; when exactly one of the
two conditions holds the verdict is failure; and when the margin over the baseline is at most 6 percentage
points the verdict is inconclusive.

**Validates: Requirements 10.1, 10.2, 10.5, 10.6**

### Property 16: An enabled term announces itself with its device

For any configuration in which CCR or MCA is enabled, model construction emits a log record naming the term,
its weight, its `rho` (or `n/a` for MCA), its `ccr_action_source` (for CCR) and the device string of the
model's parameters.

**Validates: Requirements 3.6**

### Property 17: The two action sources coincide whenever the window suffices

For any observation and action window of length `A`, any `rho >= 0` and any `ccr_rollout_len` `L` with
`num_hist + L - 1 <= A`, computing CCR under `ccr_action_source = logged` and under
`ccr_action_source = synthetic` from the same seeded RNG state yields bitwise equal CCR values, equal
counterfactual action tensors, and equal counts of `predict`, `replace_actions_from_z`, `encode_obs` and
`total_curvature` calls; and whenever `num_hist + L - 1 > A`, the `synthetic` counterfactual action tensor
agrees with the `logged` one on its first `A` frames and has exactly `num_hist + L - 1 - A` further frames.

**Validates: Requirements 1.2, 1.4, 2.2, 1.10**

## Testing Strategy

**Layout.** New `tests/` directory: `tests/test_ccr_properties.py`, `tests/test_run_naming.py`,
`tests/test_iteration_cap.py`, `tests/test_telemetry.py`, `tests/test_probe.py`,
`tests/test_acceptance_gate.py`, plus `tests/conftest.py` holding the test doubles.

**Test doubles, not DINOv2.** Every property test runs on CPU in float32 against a tiny stub encoder defined
in `conftest.py`: an `nn.Module` with `name = "tiny"` (so `VWorldModel` takes the identity
`encoder_transform` branch), `emb_dim = 4`, `latent_ndim = 2`, `patch_size = 2`, a `forward` returning
`(b*t, p=4, d=4)` and an `agg` implementing the same `mean | flatten | mlp` contract as `models/dino.py`.
The predictor stub is a `Linear` over tokens; proprio and action encoders come from `models/proprio.py`;
`decoder=None`. This keeps a 100-iteration property test in the low seconds and requires no network access,
no GPU and no dataset.

**Property tests** use `hypothesis` (test-only dependency, `pip install hypothesis`; not in
`requirements-train.txt` so the training image is unchanged). Generators cover batch size 1-4, `num_frames`
3-6, `num_hist` 2-3, `rho` in `[0, 1]` including exactly 0, `lambda_cf` in `[0, 10]`, `ccr_rollout_len` in
`1-6` (deliberately spanning the feasible/infeasible boundary), `ccr_action_source` in
`{logged, synthetic}`, `agg_type` in `{mean, flatten, mlp}`, and `concat_dim` in `{0, 1}`. The
`(ccr_rollout_len, num_frames, num_hist, ccr_action_source)` product is generated jointly, so Properties 11
and 17 each see feasible and infeasible horizons under both sources rather than only the target cell's
`(5, 4, 3)`. Minimum 100 iterations per property. Each test carries the tag
**Feature: counterfactual-curvature-regularization, Property N: <property text>** in its docstring.

Where a property is stated against "the pre-feature reference implementation" (Properties 1 and 7), the
reference is a frozen copy of the original `forward` tail and the original `rollout` loop, checked into
`tests/reference_impl.py` with a comment recording the commit it was copied from. This is model-based
testing: the simple, known-correct implementation versus the refactored one.

**Determinism.** Properties 1, 4 and 5 compare exact tensor equality across two runs, so each run seeds
`torch.manual_seed` immediately before the forward and the perturbation sampler is the only RNG consumer in
the CCR path. Property 4 forces zeros by monkeypatching `_sample_action_perturbation`, which is also why that
sampler is a separate method rather than three inline lines.

**Example and edge-case unit tests** (kept few, since the properties cover input breadth):

- `conf/train.yaml` defaults: `lambda_cf == 0`, `mca_weight == 0`, `ccr_rollout_len == 5`,
  `ccr_action_source == "synthetic"`, `max_iterations == 0`, `save_every_x_iterations == 1000`
  (Requirements 3.1, 6.9, 1.4).
- Default coherence: the shipped defaults with `lambda_cf` raised to a positive value run a forward on the
  PushT target-cell shapes without raising, while the same configuration with
  `ccr_action_source=logged` raises the Requirement 1.10 `ValueError`.
- `ccr_action_source=bogus` raises at construction, at `lambda_cf = 0` as well as above it.
- The exact legacy run-directory string `pusht_aggmlpcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05`
  resolves unchanged (Requirement 3.4).
- Protocol_Invariants table equality against the resolved Full_Run config: batch 32, `num_hist` 3,
  `num_pred` 1, `num_frames` 4, `frameskip` 5, epochs 2, `encoder_lr` 1e-5, `straighten` `aggcos1e-1`,
  `stop_grad` True, `mixed_precision` bf16, `seed` 0 (Requirement 5.1).
- `conf/plan_gd.yaml` / `conf/plan_gd_mpc.yaml` hyperparameter equality: `max_iter` 1, `n_taken_actions` 25,
  `sub_planner.horizon` 25, `lr` 0.1, `sample_type` `zero`, `action_noise` 0, `opt_steps` 100
  (Requirement 5.3).
- Changed-file guard: the feature branch's changed-file set is a subset of the Requirement 5.6 allowlist,
  and `planning/*.py` plus `datasets/*.py` hashes match the base revision (Requirements 5.2, 5.4).
- Generality by configuration: resolve `conf/train.yaml` with `env=point_maze` and CCR enabled, construct the
  model and run one forward, with no code change (Requirement 11.4).
- Probe on a missing path: non-zero exit, path in the message, no model constructed (Requirement 7.5).
- Probe budget: one timed run against the target PushT checkpoint on CPU, asserted under 30 minutes
  (Requirement 7.6). This is an integration test, run once, not a property.

**Not property-tested, deliberately.** Requirement 9 (environment variables, `ps` hygiene, serial
execution) is operator procedure enforced by `run_ccr_pilot.sh`, verified once. Requirement 8 (pilot
discipline) and Requirement 11 (approval, citation, escalation, compute accounting) are process obligations;
their only automatable fragments are the telemetry shares and step rate (Property 10) and the collapse check
in `summarize_training_log.py`. Requirement 4.6 and 5.7 are scope statements.

## Runtime environment

`run_ccr_pilot.sh` (new standalone driver, no existing script modified) applies the Blackwell/MIG recipe of
Requirement 9 before every launch and refuses to start otherwise:

```bash
export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync    # 9.1
unset CUDA_VISIBLE_DEVICES                                # 9.2
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
       OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8       # 9.3
# 9.5 / 9.6: the 1g.45gb MIG slice holds exactly one job, and nvidia-smi does not
# enumerate processes on a MIG slice, so ps is the only reliable holder list.
ps -eo pid,stat,etime,cmd | grep -E "[p]ython (train|plan|probe)" && { echo "slice busy"; exit 1; }
```

Evaluation launches additionally export `PLAN_SERIAL_ENV=1` (Requirement 9.4). Jobs are chained on the
driver's PID, never on the absence of its children, and run one at a time (Requirement 9.7).

## Pilot and acceptance protocol

**Escalation ladder** (Requirement 11.3), each rung gated on the previous one's written threshold:

| rung | command | cost | gate |
|---|---|---|---|
| Offline probe | `probe_ccr_curvature.py` | ~30 min, CPU | `curvature_gap` positive and >= 20% of unperturbed curvature on >= 3 of 5 dimensions |
| Pilot x4 | `training.max_iterations=8000 training.epochs=3` sweeping `ccr_action_source`, `rho` and `lambda_cf` | ~75-85 min each; ~85-95 min for the `synthetic` `L=5` arm | all four pilot checks below |
| Triage eval | `plan.py`, 1 data seed | ~20 min | sanity only; a low number is not evidence against (a pilot predictor is ~7x worse on `z_loss`) |
| Full run | 123,858 steps, 2 epochs | ~17 h | — |
| 3-seed eval | seeds 100/200/300, open-loop + MPC | ~1.5 h | Acceptance_Gate |

**Compute reconciliation.** The recorded plan (Requirement 11.5) budgets ≈23 GPU-hours around *three*
Pilot_Runs: 0.5 h probe + 3 x (75-85 min) + 20 min triage + 17 h Full_Run + 1.5 h three-seed eval
≈ 23.1-23.6 h. Adding the `logged`-versus-`synthetic` comparison makes it four arms, and the `synthetic` arm
runs `L = 5` rather than `L = 2`. The per-step cost of that is `5` predictor calls plus their backward instead
of `2`, and **still zero additional encoder forward passes** — the encoder pass over the 4-frame window is
shared with the baseline term and is the dominant cost of a step, so the extra three predictor calls over
`num_hist` frames of tokens are a small fraction of it. The `synthetic` arm is nonetheless the one with the
least headroom against the Requirement 11.7 regression clause, so its `it_per_s` is checked first and against
the same floor: `>= 1.93 it/s` versus the ~2.9 it/s PushT reference. Revised arithmetic:

| item | recorded plan | revised |
|---|---|---|
| Pilot subtotal | 3 arms, ≈3.75-4.25 h | 4 arms, ≈5.2-5.8 h |
| Everything else (probe, triage, Full_Run, 3-seed eval) | ≈19.3 h | ≈19.3 h (unchanged) |
| **Total** | **≈23 GPU-hours** | **≈24.5-25 GPU-hours** |

The ≈1.5-2 GPU-hour overrun is reported rather than absorbed silently, since Requirement 11.5 records a
specific allocation. If the overrun is not approved, the arm to drop is the `lambda_cf` variation, not the
`logged` control: the control is what makes the `synthetic` extrapolation risk measurable. Additional
training seeds under Requirement 10.5 still cost a further ≈26 GPU-hours and require separate approval
(Requirement 11.6).

**Pilot configuration** (primary claim; note `mca_weight=0` per Requirement 4.5, and that at the default
`ccr_action_source=synthetic` no `ccr_rollout_len` override is needed — `L = 5` equals Planner_Horizon):

```bash
python train.py --config-name train.yaml env=pusht encoder=dino_channel \
  training.straighten=aggcos1e-1 training.encoder_lr=1e-5 training.stop_grad=True \
  training.lambda_cf=0.1 training.ccr_rho=0.05 \
  training.ccr_action_source=synthetic training.ccr_rollout_len=5 \
  training.mca_weight=0 training.max_iterations=8000 training.epochs=3
```

`ccr_action_source` and `ccr_rollout_len` are written out explicitly even though both are already the
defaults, so the command recorded in the progress log identifies its arm without a reader having to know the
defaults of the day.

Arms, each resolving to its own run directory via `ccr_tag`:

| arm | override delta from the block above | isolates |
|---|---|---|
| treatment | — (`synthetic`, `L=5`, `rho=0.05`) | the full-horizon off-log penalty |
| horizon control | `ccr_action_source=logged ccr_rollout_len=2` | whether the gain needs the horizon past the window edge, or only the first two off-log steps; it is also the control for the extrapolation risk `synthetic` takes on |
| perturbation control | `ccr_rho=0` | "rollout space vs encoder space" separated from "off-log vs on-log" |
| weight variation | `lambda_cf` varied | sensitivity of the result to the term's share of the objective |

**Pilot gate, written before launching** (Requirement 8.1):

1. Within two minutes: the `CCR enabled:` line names the right weight, `rho`, `action_source`,
   `synthesized_action_frames` and device, and a checkpoint exists on disk. An empty checkpoint directory a
   minute in means the run crashed. The primary confirmation that CCR is actually running is the `ccr` term
   appearing in the telemetry record's `enabled_terms` (equivalently `enabled: true` in the record's `ccr`
   block), because that is derived from the model's own gate firing rather than from config. `rho`,
   `rollout_len`, `action_source` and `synthesized_action_frames` are read only **after** CCR is confirmed
   enabled; then, as a secondary check, a `synthetic` arm reporting `synthesized_action_frames=0` is silently
   a `logged` arm and the launch is wrong.
   *Corrected after the pod smoke test: on a CCR-disabled baseline (`lambda_cf=0`) the old field still read
   `synthesized_action_frames=3`, so it never confirmed CCR was running.*
2. `it_per_s >= 1.93` (no more than a 50% step-time regression against the ~2.9 it/s reference), and the
   step-200 telemetry row matches the reference run's step-200 row for the shared terms (Requirement 8.4).
   This check is applied to the `synthetic` `L = 5` arm first, since it has the least headroom; a failure
   there triggers Requirement 11.7 reporting before the Full_Run rather than after.
3. At step 4,000 the CCR **share** is in `[0.02, 0.30]` and the prediction share is at least half its
   reference share. Shares, never raw loss values (Requirement 8.3).
4. The raw CCR term does not fall below 1e-3 within the first 1,000 iterations. If it does, the term has
   absorbed the task without pressuring the encoder and the pilot is recorded as **not** a success
   (Requirement 8.6).

Mid-run representation readouts are read as catastrophic-failure detectors only, never as trends
(Requirement 8.5). Each pilot's outcome and caveats are appended to the project progress log
(Requirement 8.8).

**Acceptance gate** (Requirement 10), a pure predicate so it can be property-tested:

```python
def acceptance_gate(cand_ol, cand_mpc, base_ol, base_mpc,
                    paper_ol=77.33, paper_mpc=85.33, se_pts=5.7, margin_pts=6.0):
    beats_paper = cand_ol > paper_ol and cand_mpc > paper_mpc
    beats_platform = cand_ol > base_ol and cand_mpc > base_mpc
    margin = min(cand_ol - base_ol, cand_mpc - base_mpc)
    if not (beats_paper and beats_platform):
        return "fail"                 # one condition alone is a failure (10.6)
    return "inconclusive" if margin <= margin_pts else "pass"   # (10.5)
```

Both candidate and Platform_Baseline (~75.3 / ~82.0) are measured under the unmodified Evaluation_Protocol:
50 test samples per data seed, seeds 100/200/300, open-loop `mode=last, alpha=1`, MPC `mode=staged, alpha=1`.
The binomial standard error at n=50 near p=0.8, ~5.7 percentage points, is reported alongside every
comparison (Requirement 10.4).

## Design decisions and rejected alternatives

**Reuse `z` instead of re-encoding.** Calling `rollout(obs_0, act)` directly would satisfy Requirement 1.2
most literally but would run a second DINOv2 forward over the observation window, roughly doubling step time
and putting the whole pilot at risk of the Requirement 11.7 regression clause. Extracting the rollout body
gives the same tensor ops through the same code path at no extra encoder cost, and `rollout` itself is
unchanged for every existing caller.

**No `nn.Module`, by construction rather than by care.** An alternative design would add a small projection
head for the counterfactual latents and register it in an optimizer with an explicit `.to(device)`. That is
the design that produced the device crash and the never-trained head before. Requirements 1.6 and 4.3 forbid
it, and this design makes it unreachable: the CCR path has no constructor call in it, so there is nothing to
place on a device or to forget to register.

**Uniform, not Gaussian, perturbations.** A Gaussian would need clamping to satisfy Requirement 1.3, and
clamping distorts the distribution near the boundary in a way that varies with `rho`. Uniform on
`[-rho, rho]` is bounded by construction and reduces to exact zeros at `rho = 0`, which is what makes the
control arm a genuine code-path twin.

**`ccr_rollout_len` default 5, and `ccr_action_source` default `synthetic` to make it reachable.**
Requirement 1.4 ties the default to Planner_Horizon and the default must generalise to cells trained with
longer windows (Requirement 11.4). Silently clamping `L` to the feasible maximum was rejected: two runs with
different names would then be numerically identical, which is worse than an explicit failure. Under `logged`
the failure message names the maximum permitted value, so the fix is a one-token override. Under the
`synthetic` default there is nothing to fix, because the horizon no longer depends on how many actions the
window happened to record.

**Both action sources, rather than picking one.** The single-source alternatives are each defensible and each
answer a different question, which is the reason to keep both. `logged`-only caps the imagined horizon at 2 on
the target cell, so CCR would regularise two seam-heavy curvature triples and a null result could not be
distinguished from "the horizon was too short to matter". `synthetic`-only reaches the planner's horizon but
buys it with extrapolation past any real observation, and a positive result could not be distinguished from
"more predictor steps helped, regardless of where the actions came from". Run as a pair they form a clean
ablation on exactly one axis — what drives the steps past the window edge — at the price of one extra pilot
arm (≈85-95 min), which is cheap against a 17-hour Full_Run committed on the basis of these pilots. A knob
also keeps the choice reversible: if the `logged` arm wins, the Full_Run is a two-token override away and no
code changes.

**Synthesized actions reuse the bounded uniform sampler.** A Gaussian was rejected for the same reason it was
rejected for the perturbation: it would need clamping to keep synthesized actions inside a stated bound, and
clamping distorts the distribution near the boundary in a `rho`-dependent way, which would break Property 3
for the synthesized frames and make the `logged`/`synthetic` pair differ in distribution *shape* rather than
only in *source*. Fitting a distribution to the logged action marginals was rejected because it reintroduces a
dataset-derived quantity into the CCR path, directly against Requirement 1.8, and because it pulls the
synthesized actions *back* toward the on-log region CCR exists to leave. A learned proposal (an action
sampler trained jointly, or a small policy head) was rejected outright: it needs an `nn.Module`, which
Requirements 1.6 and 4.3 forbid and which is the exact construction that produced the device crash and the
never-trained head before. What remains is a zero-centred ball of radius `rho` in normalized action space,
which is defensible from the protocol alone — unit-variance normalized actions, a planner that starts at zero
with `sample_type: zero` and takes 100 small Adam steps — and which reduces to the planner's own
initialisation at `rho = 0`. It also means the whole CCR path has one RNG consumer, which is what keeps
Properties 4, 5 and 17 checkable by exact tensor equality.

**Scale-invariant MCA.** Penalising `(||v_agg|| - ||v_patch||)^2` directly would fight the `LayerNorm` at the
end of `agg` and mostly measure a global scale mismatch. Only similarity up to one constant is needed for
straightness to transfer, hence the ratio-to-batch-mean form.

## Known limitations

- **Under `ccr_action_source=logged`, the imagined horizon is 2, not 5.** Protocol_Invariants
  (`num_frames = 4`, `num_hist = 3`) let the recorded window drive only two imagined predictor steps, while
  the planner's horizon is five, so this arm regularises the first two steps of an off-log rollout rather
  than the whole planning horizon. This limitation applies **only** to the `logged` arm; it is not a property
  of CCR. Lifting it within `logged` would require `num_frames = 7`, which changes the paper protocol and is
  out of scope — which is why the `synthetic` source exists instead.
- **Under `ccr_action_source=synthetic`, the imagined states past the window edge are extrapolations.** The
  last `num_hist + L - 1 - num_frames` steps (3 of 5, for the default `L = 5` on the target cell) are rolled
  under actions that no logged trajectory contains, from a latent state that no real observation corresponds
  to. The predictor is extrapolating there, and its imagined states are progressively less trustworthy the
  further out they go; CCR is straightening a trajectory the world model may simply have wrong, and there is
  no per-step ground truth in the training window to bound the drift. Two things make this acceptable rather
  than disqualifying. First, it is exactly the regime the planner operates in: `GDPlanner` rolls the same
  predictor five steps from zero-initialised actions with no observation past the first, so a predictor whose
  five-step imagination is untrustworthy is a planning problem whether or not CCR looks at it. Second, the
  `logged` arm is the built-in control for precisely this: if `synthetic` wins and `logged` does not, the gain
  came from the extrapolated horizon; if both win by a similar margin, the extrapolated steps are not carrying
  the result and the cheaper `logged` configuration is preferable.
- **The curvature window includes real context frames.** At `L = 5` two of the five curvature triples touch a
  real frame; at `L = 2` both of the two do. The purely-real triple is excluded by the `L + 2` window in
  either case, but a real/imagined seam term is unavoidable at any horizon, and at `L = 2` the penalty is
  nothing but seam terms.
- **Cosine curvature is scale-invariant and velocity-thresholded.** `_cos_curvature` masks velocity pairs
  under `1e-6`, so a predictor that shrinks imagined velocities can reduce CCR without straightening
  anything. This is the collapse hazard that pilot gate 4 and `--collapse-check` watch for.
- **Requirement 1.5's "visual-and-proprio" wording is not implementable** with `agg_type: mlp`, whose input
  width is fixed at `196 * 8`. `visual_only` is used, matching the baseline curvature term.
- **`global_iter` is absent from pre-feature checkpoints**, so resuming a legacy run starts the counter at 0
  and the cap bounds post-resume steps only.
