# Project Structure

## Entry points (repo root)
- `train.py` — training loop (`Trainer` class); Hydra `@hydra.main` over `conf/train.yaml`. Jointly trains encoder + predictor (+ decoder), applies the straightening regularizer.
- `plan.py` — planning/evaluation entry point; loads a trained checkpoint and runs a planner against goals.
- `preprocessor.py` — data preprocessing utilities.
- `utils.py` — shared helpers (seeding, trajectory-dict slicing, cfg-to-dict, tensor sampling).
- `custom_resolvers.py` — registers OmegaConf resolvers; must be imported for configs to resolve.
- `reproduce_table1.py`, `aggregate_results.py`, `collect_results.py`, `summarize_run.py` — reproduction & results tooling.
- `*.sh` — driver scripts (`setup_b200.sh`, `run_train.sh`, `run_experiments.sh`, `evaluate.sh`, `eval_pusht_3seeds.sh`, `all_results.sh`).

## Directories
- `conf/` — Hydra configs. Root configs (`train.yaml`, `plan_*.yaml`) compose config groups:
  - `conf/env/` — one file per benchmark (`point_maze`, `point_maze_medium`, `pusht`, `wall`, `rope`, `granular`, `deformable_env`).
  - `conf/encoder/` — encoder variants (`dino`, `dino_channel`, `dino_global`, `dino_cls`, `resnet`, `scratch_resnet`, `scratch_resnet_spatial`, `r3m`, `dummy`).
  - `conf/decoder/` — `vqvae`, `transposed_conv`.
  - `conf/predictor/` — `vit`.
  - `conf/planner/` — `gd`, `cem`, `mpc_gd`, `mpc_cem`.
  - `conf/action_encoder/`, `conf/proprio_encoder/` — `proprio` / `dummy`.
- `models/` — model implementations.
  - `visual_world_model.py` — `VWorldModel`, the top-level model (Hydra `_target_`).
  - `dino.py`, `vit.py`, `vqvae.py`, `proprio.py`, `dummy.py`.
  - `models/encoder/` — `resnet.py`, `vit.py`, `r3m/`.
  - `models/decoder/` — `transposed_conv.py`.
- `datasets/` — per-environment trajectory datasets (`point_maze_dset.py`, `pusht_dset.py`, `wall_dset.py`, `deformable_env_dset.py`), plus `traj_dset.py` (base) and `img_transforms.py`.
- `planning/` — planners and evaluation.
  - `base_planner.py`, `gd.py`, `cem.py`, `mpc.py` — planner implementations.
  - `objectives.py` — planning objectives (modes: `last`, `all`, `staged`; `alpha` weighting).
  - `evaluator.py` — rollout + success-rate evaluation.
- `env/` — environment/simulator wrappers.
- `metrics/` — `image_metrics.py` and `lpipsPyTorch/` (LPIPS perceptual metric).
- `distributed_fn/` — DDP / launch helpers (`distributed.py`, `launch.py`).
- `assets/` — figures (e.g. `architecture.png`).
- `checkpoints/` — default output root (`ckpt_base_path`), created at runtime.

## Conventions
- **Add features via config groups + a Hydra `_target_` class**, not by branching on strings. New encoder → add `models/encoder/*.py` and a matching `conf/encoder/*.yaml`.
- Models are instantiated by Hydra from their `_target_`; keep constructor args mirrored in the corresponding yaml.
- Datasets subclass the base in `traj_dset.py` and are selected via `conf/env/*.yaml`.
- Docs of record: `README.md` (usage) and `REPRODUCTION.md` (authoritative paper settings). `AGENT_MEMORY_2.0.md` and `POD_SETUP_LOG.md` capture prior agent/pod context.
- Log files (`*.log`), `.train_pid`, and `chain.log` are runtime artifacts, not source.
