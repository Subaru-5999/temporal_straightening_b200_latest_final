# Tech Stack & Commands

## Language & core frameworks
- **Python 3.9** (paper env) / 3.10 (B200 pod). See `environment.yaml`.
- **PyTorch** for models and training (`torch`, `torchvision`). `accelerate` for mixed precision (bf16) and distributed launch.
- **Hydra** + **OmegaConf** for all configuration (`conf/`). Custom OmegaConf resolvers live in `custom_resolvers.py` and are registered by importing the module.
- **einops** for tensor rearranging.
- **wandb** for experiment logging; **tensorboard** also available.
- Simulators/benchmarks: **MuJoCo** (`mujoco`, `mujoco-py`), `gym`, `d4rl`, `robomimic`, `robosuite`, `pymunk` (PushT), `dm-control`.
- Encoders: DINOv2 (patch/CLS/global/channel-projected), ResNet (scratch), R3M.

## Configuration model (important)
- This is a **Hydra-driven** project. Do not hardcode values; add/override config.
- Entry configs: `conf/train.yaml`, `conf/plan_gd.yaml`, `conf/plan_cem.yaml`, `conf/plan_gd_mpc.yaml`.
- Config groups: `conf/{env,encoder,decoder,predictor,planner,action_encoder,proprio_encoder}/`.
- Override on the CLI, e.g. `encoder=dino_channel training.straighten=aggcos1e-1`.
- `training.straighten`: `False` (off), `cos1e-1` (patch-wise curvature), `aggcos1e-1` (pooled-feature curvature).
- Datasets are located via the `DATASET_DIR` environment variable.

## Environment setup
```bash
# Paper environment (Linux, conda)
conda env create -f environment.yaml
conda activate ts

# NVIDIA B200 / Blackwell pod (no conda) — installs cu128 torch wheels
bash setup_b200.sh
pip install -r requirements-train.txt   # training deps
pip install -r requirements-plan.txt    # planning deps
```

## Common commands

### Train
```bash
python train.py --config-name train.yaml env=point_maze
# variant example
python train.py --config-name train.yaml env=point_maze \
  encoder=dino_channel training.straighten=aggcos1e-1
```

### Plan / evaluate
```bash
python plan.py --config-name plan_gd.yaml     ckpt_base_path=<ckpt_root> model_name=<model_name>
python plan.py --config-name plan_cem.yaml    ckpt_base_path=<ckpt_root> model_name=<model_name>
python plan.py --config-name plan_gd_mpc.yaml ckpt_base_path=<ckpt_root> model_name=<model_name>
# PushT: set objective.alpha=1 (GD-MPC also objective.mode=staged)
```

### Reproduce & aggregate results
```bash
python reproduce_table1.py        # drives Table 1 reproduction
python aggregate_results.py       # mean ± std over seeds
python collect_results.py
python summarize_run.py
bash eval_pusht_3seeds.sh         # example multi-seed eval driver
```

### Test
```bash
pytest                            # pytest is available; tests are sparse (research repo)
```

## Key conventions
- **Match the paper exactly** for reproduction: encoder lr `1e-6` (no straightening) vs `1e-5` (straightening); epochs 20 (Wall/PointMaze) / 2 (PushT); batch 32; `num_hist=3`; `frameskip=5`. See `REPRODUCTION.md` for the authoritative settings table.
- Checkpoints save under `ckpt_base_path` (default `./checkpoints`); the run's `model_name` derives from the Hydra output dir.
- Keep planning horizons divisible by `frameskip` (plan.py divides `goal_H`, `n_taken_actions`, `sub_planner.horizon` by frameskip).
- Resuming is offline: encoder (incl. DINOv2 weights) is stored in the checkpoint; auto-resumes from `model_latest.pth` unless `training.resume_from` is set.
- Mixed precision defaults to `bf16`; `stop_grad=True` by default to prevent representation collapse.
