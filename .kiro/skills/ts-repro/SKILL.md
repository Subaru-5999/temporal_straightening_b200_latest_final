---
name: ts-repro
description: >
  Reproduce "Temporal Straightening for Latent Planning" (arXiv 2603.12231v2) Table-1
  cells on the NVIDIA Blackwell B200 MIG pod. Invoke this whenever the user gives a
  GOAL such as "reproduce PushT straightening", "retrain UMaze +proj ✗ with another
  seed", "evaluate run X", or "get cell Y within the paper band". It encodes the paper
  protocol, the Blackwell/MIG environment recipe, known pitfalls, and a repeatable
  goal-execution loop. Keywords: temporal straightening, DINO-WM, planning, GD, MPC,
  PushT, PointMaze, UMaze, Table 1, reproduction, B200, MIG, blackwell.
---

# Skill: Temporal-Straightening Table-1 Reproduction (Blackwell B200 MIG)

## How to use this skill (the loop)
The user states a **GOAL** for a specific task. You run **The Goal Loop** below, then stop
and report. The user can hand you a new GOAL again and again — each time, re-run the loop
from the top. Do **not** rely on memory of the paper's numbers or protocol: re-read the
paper file every time (Rule 1).

### Rules
1. **Paper is the source of truth — re-read it every time.** The **authoritative**
   source is the paper's **LaTeX** in `arXiv-2603.12231v2.tar.gz` (extract with
   `tar -xzf`): `sec/1_main.tex` (main body incl. **Table 1** at ~line 408–409) and
   `sec/2_appendix.tex` (App: PushT 2 epochs; **Table 3** training hyperparams;
   **Table 4** planning hyperparams; App straightening λ). Prefer the `.tex` over the
   OCR'd `_paper.txt` (which has spacing/encoding artifacts) and `2603.12231v2.pdf`.
   For any protocol detail (epochs, lr, objective, seeds, horizons, straightening λ),
   quote the `.tex`, don't guess. Re-open the relevant section on every goal.
   **Authoritative Table-1 PushT targets (from `sec/1_main.tex`):**
   DINOv2 patch+proj 14×14×8 ✗ → OL 70.00±1.63 / MPC 78.67±0.94;
   ✓ → OL 77.33±6.18 / MPC 85.33±4.99.
   **Note:** the paper specifies NO training seed (code default `training.seed=0`);
   its "three data sampling seeds" are EVAL seeds (100/200/300) on one trained model.
   Also **cross-reference `REPRODUCTION.md`** — it is the authoritative *old-reproduction*
   record: the exact settings table (which matches the paper), the exact prior CLI
   commands for train/eval, and the earlier B200-pod per-cell numbers ("Ours"). Use it to
   confirm flags and to sanity-check new results against both the paper AND the prior run.
2. **Use the internet if necessary** (web search/fetch) for tooling/library/GPU issues
   (e.g., torch+MIG allocator, mujoco-py device selection, CUDA/Blackwell) — but never
   for the paper's numbers (those come from `_paper.txt`).
3. **Keep the Blackwell/MIG configs in mind** — always apply the env recipe in §"Env
   recipe" and the GPU hygiene in §"Pre-flight" before any run. This slice breaks in
   MIG-specific ways; the recipe is non-negotiable.

## The Goal Loop
1. **Restate the goal** in one line and classify it:
   - (a) *evaluate* an existing trained run, (b) *(re)train* a model, (c) *reproduce a
     Table-1 cell end-to-end* (train → eval), (d) *analyze/verify*.
2. **Read the paper** (`_paper.txt`) for the exact protocol of that cell/task
   (§5.3 planning, Table 3 training, Table 4 planning hyperparams, App. A epochs,
   App. B.6 straightening λ). Confirm encoder, lr, epochs, objective mode, alpha, seeds.
   **Then open `REPRODUCTION.md`** for the authoritative settings table, the exact prior
   train/eval CLI, and the earlier "Ours" numbers to reuse and compare against.
3. **Apply the Blackwell/MIG env recipe** (§"Env recipe").
4. **Pre-flight the GPU** (§"Pre-flight"): the 45 GB slice holds ONE job and fills from a
   single stray process. Kill strays first.
5. **Execute**:
   - eval → `python reproduce_table1.py [<run> ...]` (or a single `plan.py` for one cell).
   - train → `train.py` with the paper's flags (§"Training recipe").
6. **Aggregate & compare**: `python summarize_run.py --all` (re-scans `logs.json`,
   counts only `final_eval/success_rate`), then compare each cell to the paper target
   in §"The 5 cells" **and to the earlier reproduction's "Ours" numbers in
   `REPRODUCTION.md` (§4 Results / §7 validated checklist)**. A cell is "reproduced" if
   the mean is within the paper's ±std band (allow the documented small upward bias on ✗
   rows — see §"Pitfalls") and is consistent with the prior run.
7. **Decide & iterate**: if the goal isn't met, diagnose with §"Pitfalls" and the paper
   (e.g., ✗-row too high → needs more training seeds; wrong number → check lr/epochs/λ
   against the paper; crash → match the pitfall). Iterate, then report.
8. **Record**: append the outcome and any new issue/fix to `AGENT_MEMORY_2.0.md`.

## The 5 cells & paper targets (Table 1, GD planner, 50 samples, 3 seeds 100/200/300)
| Run dir (`checkpoints/test/<name>`) | Env / config | Paper OL | Paper MPC | eval alpha / MPC mode |
|---|---|---|---|---|
| `umaze_False_agg32_projnone_dim384_hw14_sgTrue_lr1e-05` | UMaze DINOv2 patch 14×14×384 ✗ | 35.33±4.11 | 80.67±6.18 | 0 / all |
| `umaze_False_agg32_projchannel_dim8_hw14_sgTrue_lr1e-06` | UMaze +proj 14×14×8 ✗ | 44.00±7.12 | 81.33±6.80 | 0 / all |
| `umaze_aggmlpcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05` | UMaze +proj 14×14×8 ✓ | 94.00±1.63 | 100.00±0.00 | 0 / all |
| `pusht_False_agg32_projchannel_dim8_hw14_sgTrue_lr1e-06` | PushT +proj 14×14×8 ✗ | 70.00±1.63 | 78.67±0.94 | 1 / staged |
| `pusht_aggmlpcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05` | PushT +proj 14×14×8 ✓ | 77.33±6.18 | 85.33±4.99 | 1 / staged |

Objective mapping (verified vs §5.3): open-loop always `mode=last` (terminal MSE);
MPC maze/wall `mode=all` (weighted), PushT MPC `mode=staged` (terminal within H, weighted
beyond H). Proprio: UMaze `alpha=0` (images only), PushT `alpha=1` (images+proprio).

## Paper protocol cheat-sheet (always re-verify in _paper.txt)
- Training (Table 3 / App. A): batch 32, num_hist 3, frameskip 5; predictor lr 5e-4,
  action/proprio lr 5e-4; encoder lr **1e-6 for ✗**, **1e-5 for ✓** (Table 3 footnote);
  straightening on the agg head **λ=0.1** → `training.straighten=aggcos1e-1` (App. B.6);
  epochs: Wall/PointMaze **20**, PushT **2** (App. A).
- Planning (Table 4): horizon 25 (÷frameskip 5 → H=5), Adam, zero init, lr 0.1, 100 steps;
  executed actions 25 (open-loop) / 5 (MPC); 50 test samples; 3 data seeds 100/200/300.

## Env recipe (Blackwell B200 MIG — apply for BOTH train and eval)
```bash
cd /workspace/arun/temporal_straightening_old
unset CUDA_VISIBLE_DEVICES                       # MIG UUID crashes mujoco-py int-parse; leave unset
export DATASET_DIR=/workspace/arun/data
export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export WANDB_MODE=disabled WANDB_SILENT=true
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia
export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync   # avoids MIG NVML allocator assert
export PLAN_SERIAL_ENV=1                                  # MIG fork-safety for eval envs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
```
`reproduce_table1.py` sets all of these as defaults itself (and auto-unsets a non-integer
`CUDA_VISIBLE_DEVICES`). torch must stay `2.7.0+cu128` (native sm_100). One-time installs
already done: requirements-plan.txt, MuJoCo 210 + apt libs, d4rl, h5py,
requirements-train.txt, `typing_extensions>=4.12`, `numpy<2`.

## Pre-flight (run before EVERY GPU job)
```bash
ps -eo pid,etime,rss,cmd | grep -i python | grep -v grep   # kill -9 any stray/stopped (Tl) python
nvidia-smi | sed -n '/MIG dev/,/Processes/p'                # want ~2-16 MiB used, not ~41 GB
```
`nvidia-smi` cannot list MIG processes — trust `ps`. Never Ctrl-Z a GPU python job (it
leaks the whole slice). Only one job fits in 45 GB.

## Training recipe (for (re)train goals; from REPRODUCTION.md + paper)
```bash
# example: PushT +proj, straightening ON (✓), 2 epochs, lr 1e-5
setsid nohup python train.py --config-name train.yaml env=pusht encoder=dino_channel \
  training.straighten=aggcos1e-1 training.encoder_lr=1e-5 training.epochs=2 env.num_workers=4 \
  ckpt_base_path=$PWD/checkpoints/repro \
  > train_pusht_channel_on.log 2>&1 < /dev/null &
tail -f train_pusht_channel_on.log
```
lr rule: ✗ → `training.straighten=False training.encoder_lr=1e-6`; ✓ → `aggcos1e-1` +
`1e-5`. Epochs: UMaze/Wall 20, PushT 2. Trained ckpt lands under
`checkpoints/repro/test/<name>/checkpoints/model_latest.pth`; move/point eval `--base` at it.

## Eval recipe
```bash
# all 5 cells (detached):
nohup python reproduce_table1.py > eval_all.log 2>&1 &
# a subset:
python reproduce_table1.py <run_name> [<run_name> ...]
# re-derive the table from existing logs.json (no re-run):
python summarize_run.py --all
```
Results: `results/table1_reproduction.md` / `.csv`, `results/<run>.json`. Compare to §"The 5 cells".

## Pitfalls (symptom → cause → fix) — consult before diagnosing anew
- `No module 'gym'/'hydra'` → missing deps → `pip install -r requirements-plan.txt` / `-train.txt`.
- `wandb TypeIs` → old typing_extensions → `pip install -U "typing_extensions>=4.12"`; `WANDB_MODE=disabled`.
- gym NumPy 2.0 → `pip install "numpy<2"`.
- ~250s "hang" at setup_model → CPU thread oversubscription in DINOv2 init → thread caps=8.
- `plan_targets.pkl` FileNotFoundError → Hydra cwd → already patched in plan.py (makedirs+chdir).
- `NVML_SUCCESS==r` assert → MIG allocator → `PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync`.
- `int('MIG-...')` → mujoco-py parses CUDA_VISIBLE_DEVICES → keep it UNSET.
- `transformer_engine` ImportError (`undefined symbol: _ZNSt...`) at TRAIN time only →
  `accelerate` auto-imports NGC's TE, ABI-mismatched vs torch 2.7 → `pip uninstall -y
  transformer-engine transformer_engine`. Eval (plan.py) doesn't use accelerate, so it's
  unaffected — this only bites the first training on a fresh NGC pod. (See POD_SETUP_LOG.md.)
- `OutOfMemoryError` → almost always a leaked/stopped python holding 41.5 GB → `ps`+`kill -9`, not a workload issue.
- Aggregation `n=6/63`, MPC too low → counting per-iteration logs → summarizer already fixed to count only `final_eval/success_rate`.
- ✗ (no-straighten) rows run a few points HIGH here (single training seed, platform variance) — expected, per REPRODUCTION.md; to tighten, train multiple seeds. ✓ rows should match the paper.

## Tools in the repo
- `reproduce_table1.py` — driver (5 cells, OL×3 + MPC×3, paper objectives, env baked in, run-scoped).
- `summarize_run.py` — per-run + master aggregator (ours vs paper); `--all` rescans logs.json.
- `check_dataset_sync.py` — verify data + trained-run configs match the loaders.
- `AGENT_MEMORY_2.0.md` — full issue/fix log; `REPRODUCTION.md`, `POD_SETUP_LOG.md` — original protocol.

## Example goal (first use): re-do PushT +proj straightening (✓) end-to-end
```bash
cd /workspace/arun/temporal_straightening_old
NAME=pusht_aggmlpcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05
# 1) delete the current ✓ artifacts (checkpoint + eval outputs + result json)
rm -rf checkpoints/test/$NAME
rm -rf plan_outputs_gd/${NAME}_*/ plan_outputs_gd_mpc/${NAME}_*/
rm -f results/$NAME.json
# 2) (re)train per the paper (PushT ✓ = 2 epochs, lr 1e-5, aggcos1e-1) -- see Training recipe
#    then place/point --base at the new checkpoints/repro/test/$NAME
# 3) pre-flight GPU, then evaluate + aggregate
python reproduce_table1.py $NAME
python summarize_run.py --all
# 4) compare PushT ✓ to paper OL 77.33±6.18 / MPC 85.33±4.99; iterate if outside band
```
