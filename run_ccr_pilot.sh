#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_ccr_pilot.sh — standalone launcher for the CCR pilot / full run / eval.
#
# This script is NEW and modifies nothing else: run_train.sh, evaluate.sh,
# eval_pusht_3seeds.sh, redo_pusht_paperexact.sh and reproduce_table1.py are
# untouched.  Everything Requirement 9 asks for lives here.
#
# Usage
#   DATASET_DIR=/workspace/arun/data bash run_ccr_pilot.sh pilot [hydra overrides...]
#   DATASET_DIR=/workspace/arun/data bash run_ccr_pilot.sh full  [hydra overrides...]
#   DATASET_DIR=/workspace/arun/data bash run_ccr_pilot.sh eval  <run_dir> [plan overrides...]
#
#   # horizon control arm (task 15.3)
#   bash run_ccr_pilot.sh pilot training.ccr_action_source=logged training.ccr_rollout_len=2
#   # perturbation control arm (task 15.4)
#   bash run_ccr_pilot.sh pilot training.ccr_rho=0
#   # weight variation arm (task 15.5), kept out of the shared checkpoint tree
#   CKPT_BASE=$PWD/checkpoints_cf03 bash run_ccr_pilot.sh pilot training.lambda_cf=0.3
#   # single-seed triage eval (task 16.1)
#   SEEDS=100 bash run_ccr_pilot.sh eval checkpoints/test/<run_name>
#   # aggregated-space sweep arm, open-loop only, held-out tuning seed
#   PLAN_ENTRY=plan_agg.py SETTINGS=ol SEEDS=400 HYDRA_RUN_DIR=agg \
#     bash run_ccr_pilot.sh eval checkpoints/test/<run_name> '+agg_weight=0.1'
#   # the paired zero-weight plan.py leg, steered away from the recorded baseline cell
#   SETTINGS=ol SEEDS=400 \
#     HYDRA_RUN_DIR='plan_outputs_gd_scratch/${replace_slash:${model_name}}_seed${seed}' \
#     bash run_ccr_pilot.sh eval checkpoints/test/<run_name>
#
# Environment overrides
#   DATASET_DIR     REQUIRED. Taken from the environment; unset is a hard error.
#   MAX_ITERS       training.max_iterations (default 8000 for pilot, 0 for full).
#   CKPT_BASE       ckpt_base_path          (default $PWD/checkpoints).
#   SEEDS           eval data seeds, space separated (default "100 200 300").
#   MODEL_EPOCH     eval checkpoint epoch   (default latest).
#   PLAN_ENTRY      eval entry script       (default plan.py). plan_agg.py runs the
#                   aggregated-space objective L_plan = L_spatial + w * L_agg; pass its
#                   weight as a normal override, e.g. '+agg_weight=0.1'.
#   SETTINGS        which eval loops run: ol | mpc | both (default both).
#   HYDRA_RUN_DIR   run-directory control for eval jobs. Unset (default) passes no
#                   override at all, so the shipped conf/plan_gd*.yaml expression decides
#                   and the CCR eval path is unchanged. "agg" resolves the per-setting
#                   template from agg_objectives.RUN_DIR_TEMPLATES, which is what keeps
#                   each sweep weight in its own logs.json. Any other value is used
#                   verbatim, for steering a leg into a scratch tree. Quote it in SINGLE
#                   quotes: a Hydra interpolation left unquoted is expanded by bash, to
#                   nothing, and the run lands in a truncated directory that
#                   aggregate_results.py parses as some other cell.
#   LOG / PIDFILE   log and pid file paths  (default ccr_<mode>_<timestamp>.log/.pid).
#   CHAIN_ON_PID    wait for this DRIVER pid to exit before launching (see below).
#   FOREGROUND=1    do not detach; run in this shell (debugging).
#   DRY_RUN=1       print each resolved command instead of running it.
#   MUJOCO_BIN      MuJoCo 210 bin dir      (default $HOME/.mujoco/mujoco210/bin).
#   RUNDIR_WAIT     seconds to wait for the run dir to appear in the log (default 240;
#                   0 skips the wait). This wait is a sleep loop in the launching
#                   shell only — Ctrl-C is safe and leaves the detached job running.
#
# Serial execution (Requirement 9.7).  The 1g.45gb MIG slice holds exactly one
# job.  To queue an arm behind a running one, chain on the running DRIVER's PID:
#
#   CHAIN_ON_PID="$(cat ccr_pilot_<earlier>.pid)" bash run_ccr_pilot.sh pilot training.ccr_rho=0
#
# Never chain on the absence of a driver's children: a pgrep poll has gaps
# between a driver's sequential jobs and two jobs then land on one slice
# (SHORT_BUDGET_PILOTS.md section 9).
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename -- "${BASH_SOURCE[0]}")"

# A live train.py / plan.py / probe_*.py python process.  The leading [p] keeps a
# stale `grep` line in the snapshot from matching itself.  The trailing
# [A-Za-z0-9_]* is what makes the guard cover plan_agg.py as well, so a PLAN_ENTRY
# job holds the slice against the next launch exactly as a plan.py job does.
JOB_PATTERN='[p]ython[0-9.]*[[:space:]]+(-[^[:space:]]+[[:space:]]+)*(train|plan|probe)[A-Za-z0-9_]*\.py'
# A STOPPED python (stat T / Tl): it still owns its CUDA context and its GPU
# memory.  This is how 41.5 GB of the slice leaked once (AGENT_MEMORY_2.0 §2.9).
STOPPED_PATTERN='^[[:space:]]*[0-9]+[[:space:]]+T[^[:space:]]*[[:space:]].*[p]ython'

usage() {
  # The contiguous comment block after the shebang is the usage text.
  awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$SCRIPT_PATH"
}

die() { echo "ERROR: $*" >&2; exit 2; }

# ---------------------------------------------------------------------------
# Eval entry point and setting selection.
#
# Both hooks default to exactly what this launcher already did — plan.py, both settings,
# no run-directory override — so the CCR evaluation path is unchanged.  plan_agg.py is
# the aggregated-space wrapper (L_plan = L_spatial + w * L_agg); it takes the same Hydra
# config names and overrides plan.py takes, plus '+agg_weight=<w>'.
#
# Read from the environment here, at the top of the script, so the detached driver — which
# re-execs this same file — resolves them from the environment it inherited rather than
# from a value the foreground shell had to pass along.
# ---------------------------------------------------------------------------
PLAN_ENTRY="${PLAN_ENTRY:-plan.py}"
SETTINGS="${SETTINGS:-both}"

# True when SETTINGS selects the named eval loop, "ol" or "mpc".
setting_selected() {
  [[ "$SETTINGS" == "both" || "$SETTINGS" == "$1" ]]
}

validate_eval_hooks() {
  case "$SETTINGS" in
    ol|mpc|both) ;;
    *) die "SETTINGS='${SETTINGS}' is not one of: ol | mpc | both." ;;
  esac
  [[ -f "$PLAN_ENTRY" ]] || die "PLAN_ENTRY='${PLAN_ENTRY}' is not a file in ${PWD}.
       The entry script is resolved relative to the directory the jobs run in, which is
       this launcher's cwd."
}

# One eval job's run directory, appended as a Hydra override only when HYDRA_RUN_DIR asks
# for it — unset means no override at all, which is what keeps the CCR path byte-identical.
#
# "agg" resolves the PER-SETTING template through agg_objectives.run_dir_override(), so the
# open-loop leg lands under plan_outputs_gd and the MPC leg under plan_outputs_gd_mpc; one
# string for both settings would put MPC results in the open-loop tree.  The template text
# is never retyped here: agg_objectives.RUN_DIR_TEMPLATES is the single source of truth, and
# a drifted copy would resolve to a directory aggregate_results.py parses as some other cell.
#
# The resolved value is captured into a variable and passed quoted, so bash performs no
# expansion on the Hydra interpolations it contains (an expanded one arrives empty, silently
# truncating the directory).
add_run_dir_default() {
  local config_name="$1" value="${HYDRA_RUN_DIR:-}" token
  if [[ -z "$value" ]]; then
    return 0
  fi
  if [[ "$value" == "agg" ]]; then
    token="$(python -c 'import sys, agg_objectives; print(agg_objectives.run_dir_override(sys.argv[1]))' \
      "$config_name")" || die "HYDRA_RUN_DIR=agg: could not resolve the run-directory template for
       config name '${config_name}'. agg_objectives.py must be importable from ${PWD}."
    [[ -n "$token" ]] || \
      die "HYDRA_RUN_DIR=agg: agg_objectives.run_dir_override('${config_name}') printed nothing."
    add_default "$token"
  else
    add_default "hydra.run.dir=$value"
  fi
}

# ---------------------------------------------------------------------------
# Blackwell / MIG environment recipe (Requirement 9.1-9.4).
# Applied before EVERY launch, in the foreground shell and again inside the
# detached driver, so a job never inherits a half-configured environment.
# ---------------------------------------------------------------------------
apply_env() {
  local mode="$1"

  # 9.2: mujoco-py does int(os.environ["CUDA_VISIBLE_DEVICES"]) to pick its
  # render device, and a MIG UUID is not an integer.  Leave it unset; torch
  # still sees the MIG device through the container.
  unset CUDA_VISIBLE_DEVICES

  # 9.1: torch 2.7's caching allocator makes an NVML query that assert-fails on
  # a MIG slice during the first backward.  The async allocator bypasses it.
  export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync

  # 9.3: uncapped threads turn DINOv2 init into a ~250 s "hang" on this node.
  export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8

  # Respected from the environment, never guessed: a wrong dataset root is a
  # silently different experiment.
  if [[ -z "${DATASET_DIR:-}" ]]; then
    die "DATASET_DIR is unset. conf/env/*.yaml resolves the dataset root from
       \${oc.env:DATASET_DIR}, so this run would fail at config resolution.
       Set it explicitly, e.g.:
         export DATASET_DIR=/workspace/arun/data"
  fi
  [[ -d "$DATASET_DIR" ]] || die "DATASET_DIR=$DATASET_DIR is not a directory."
  export DATASET_DIR

  export D4RL_SUPPRESS_IMPORT_ERROR=1
  export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl        # headless offscreen GL
  export WANDB_MODE=disabled WANDB_SILENT=true      # losses come from the JSONL telemetry

  # mujoco-py needs libmujoco210 and the nvidia GL libs on the loader path.
  # Appended idempotently: the detached driver re-applies this recipe and would
  # otherwise inherit and then duplicate the entries.
  local mujoco_bin="${MUJOCO_BIN:-$HOME/.mujoco/mujoco210/bin}" p
  for p in "$mujoco_bin" /usr/lib/nvidia; do
    case ":${LD_LIBRARY_PATH:-}:" in
      *":${p}:"*) ;;
      *) export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${p}" ;;
    esac
  done

  # 9.4: evaluation only.  Forking an env worker after CUDA/NVML init on a MIG
  # slice trips the allocator assert, so plan.py runs its envs serially.
  if [[ "$mode" == "eval" ]]; then
    export PLAN_SERIAL_ENV=1
  else
    unset PLAN_SERIAL_ENV || true
  fi
}

# ---------------------------------------------------------------------------
# ps pre-flight (Requirement 9.5 / 9.6).  REFUSES to start; it does not warn.
#
# nvidia-smi's process table is empty on a MIG slice even when a job owns all
# 45 GB of it, so `ps` is the only reliable list of GPU-memory holders.
#
# Written as `if ... grep -q ...; then`, deliberately.  The tempting form
#     ps -eo pid,stat,etime,cmd | grep python | head -3 || echo "slice free"
# is broken: `||` binds to `head`, which succeeds on empty input, so the
# "slice free" branch prints unconditionally and the guard never fires
# (SHORT_BUDGET_PILOTS.md section 9).
#
# The snapshot is taken into a variable and matched from a here-string rather
# than piped straight into `grep -q`: `grep -q` exits on the first match, and
# under `set -o pipefail` ps's resulting SIGPIPE status would become the
# status of the test — i.e. a match could read as "slice free".
# ---------------------------------------------------------------------------
preflight_or_die() {
  local snapshot
  snapshot="$(ps -eo pid,stat,etime,cmd)"

  if grep -qE "$JOB_PATTERN" <<<"$snapshot"; then
    {
      echo "REFUSING TO START: a train/plan/probe python process is already alive."
      echo "The 1g.45gb MIG slice holds exactly one job (Requirement 9.7); a second"
      echo "job OOMs both.  Offending processes:"
      grep -E "$JOB_PATTERN" <<<"$snapshot"
      echo
      echo "Wait for it to finish (chain on its DRIVER pid via CHAIN_ON_PID), or"
      echo "'kill -9 <pid>' if it is a stray.  Do not consult nvidia-smi's process"
      echo "table: it does not enumerate processes on a MIG slice."
    } >&2
    return 1
  fi

  if grep -qE "$STOPPED_PATTERN" <<<"$snapshot"; then
    {
      echo "REFUSING TO START: a STOPPED python process is present (stat T/Tl)."
      echo "A suspended python keeps its CUDA context and its GPU memory; this is"
      echo "how 41.5 GB of the slice leaked once.  Offending processes:"
      grep -E "$STOPPED_PATTERN" <<<"$snapshot"
      echo
      echo "'kill -9 <pid>' it, then relaunch."
    } >&2
    return 1
  fi

  echo "pre-flight OK: no live or stopped train/plan/probe python process."
}

# ---------------------------------------------------------------------------
# Chain on the DRIVER's pid, never on the absence of its children.
# ---------------------------------------------------------------------------
wait_for_driver_pid() {
  local pid="$1" state waited=0
  if ! [[ "$pid" =~ ^[0-9]+$ ]]; then
    die "CHAIN_ON_PID='$pid' is not a pid. Pass the contents of an earlier run's .pid file."
  fi
  if [[ "$pid" == "$$" ]]; then
    die "CHAIN_ON_PID=$pid is this driver's own pid; it would wait for itself forever."
  fi
  echo "chaining: waiting for driver pid ${pid} to exit before launching..."
  # `kill -0` alone is NOT enough. setsid detaches the driver, so when it exits its
  # parent is PID 1 -- and in a container PID 1 is often not a reaping init. The
  # driver then lingers as a ZOMBIE (stat Z, "<defunct>"), the pid stays in the
  # process table, `kill -0` keeps succeeding, and this loop waits forever on a job
  # that finished hours ago. That has already cost this project ~6 h of idle GPU.
  # A zombie holds no CUDA context and no GPU memory, so it must read as "gone".
  while kill -0 "$pid" 2>/dev/null; do
    state="$(ps -p "$pid" -o stat= 2>/dev/null | tr -d '[:space:]')"
    if [[ -z "$state" || "$state" == Z* ]]; then
      echo "chaining: driver pid ${pid} is a zombie (stat='${state:-<none>}'); it has" \
           "exited and holds no GPU memory. Treating it as gone."
      break
    fi
    sleep 30
    waited=$((waited + 30))
    # Heartbeat every 30 min: a silent multi-hour wait is indistinguishable from a
    # hang, which is exactly how the failure above went unnoticed.
    if (( waited % 1800 == 0 )); then
      echo "chaining: still waiting on pid ${pid} (stat=${state}) after $((waited / 60)) min."
    fi
  done
  echo "chaining: driver pid ${pid} is gone; continuing."
}

# ---------------------------------------------------------------------------
# Override assembly.  Hydra rejects the same key twice on one command line, so
# a shipped default is emitted only when the caller did not pass that key.
# ---------------------------------------------------------------------------
USER_ARGS=()
CMD=()

_user_overrides_key() {
  local key="$1" arg bare
  for arg in ${USER_ARGS[@]+"${USER_ARGS[@]}"}; do
    bare="${arg#[+~]}"; bare="${bare#+}"
    [[ "${bare%%=*}" == "$key" ]] && return 0
  done
  return 1
}

add_default() {
  local kv="$1"
  _user_overrides_key "${kv%%=*}" || CMD+=("$kv")
}

# The training.lambda_cf this launch will actually resolve to: the caller's
# override if they passed one, otherwise the shipped pilot default. Used only to
# print the right verification guidance -- a baseline launch (lambda_cf=0) has no
# CCR arm to verify, so telling the reader to check arm fields is misleading.
CCR_DEFAULT_LAMBDA_CF=0.1

resolved_lambda_cf() {
  local arg bare
  for arg in ${USER_ARGS[@]+"${USER_ARGS[@]}"}; do
    bare="${arg#[+~]}"; bare="${bare#+}"
    if [[ "${bare%%=*}" == "training.lambda_cf" ]]; then
      printf '%s\n' "${bare#*=}"
      return 0
    fi
  done
  printf '%s\n' "$CCR_DEFAULT_LAMBDA_CF"
}

# True when the resolved lambda_cf is non-zero. awk does the numeric compare so
# 0, 0.0 and 0e0 all read as off; a non-numeric value reads as off too, which is
# the safe side for printed guidance.
ccr_launch_enabled() {
  awk -v v="$1" 'BEGIN { exit !(v + 0 != 0) }'
}

build_train_cmd() {
  local mode="$1"
  CMD=(python train.py --config-name train.yaml)

  # The pilot block of design.md "Pilot and acceptance protocol", verbatim:
  # ccr_action_source and ccr_rollout_len are written out even though they are
  # already the defaults, so the recorded command identifies its arm.
  add_default env=pusht
  add_default encoder=dino_channel
  add_default training.straighten=aggcos1e-1
  add_default training.encoder_lr=1e-5
  add_default training.stop_grad=True
  add_default training.lambda_cf=0.1
  add_default training.ccr_rho=0.05
  add_default training.ccr_action_source=synthetic
  add_default training.ccr_rollout_len=5
  add_default training.mca_weight=0                      # Requirement 4.5

  if [[ "$mode" == "pilot" ]]; then
    # epochs deliberately generous: the iteration cap can only ever SHORTEN a
    # run, so the cap must be what ends it rather than an epoch boundary.
    add_default "training.max_iterations=${MAX_ITERS:-8000}"
    add_default "training.epochs=${PILOT_EPOCHS:-3}"
  else
    add_default "training.max_iterations=${MAX_ITERS:-0}"   # 0 == no cap, paper budget
    add_default "training.epochs=${FULL_EPOCHS:-2}"         # PushT is 2 epochs (App. A.3)
  fi

  add_default "ckpt_base_path=${CKPT_BASE:-${PWD}/checkpoints}"
  CMD+=(${USER_ARGS[@]+"${USER_ARGS[@]}"})
}

FAILURES=0

run_job() {
  local label="$1"; shift
  echo
  echo "=================== ${label} ==================="
  echo "+ $*"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY_RUN=1: not executed."
    return 0
  fi
  echo "started $(date -Is)"
  if "$@"; then
    echo "OK: ${label}  ($(date -Is))"
  else
    echo "!!! FAILED: ${label}  ($(date -Is))"
    FAILURES=$((FAILURES + 1))
  fi
}

# One job at a time, in this one process, in this one session (Requirement 9.7).
run_eval_jobs() {
  local run_dir="$1" name seeds s
  validate_eval_hooks
  run_dir="$(readlink -f "$run_dir")"
  [[ -f "${run_dir}/hydra.yaml" ]] || \
    die "${run_dir}/hydra.yaml not found. Pass the run dir that holds hydra.yaml + checkpoints/."
  name="$(basename "$run_dir")"
  read -r -a seeds <<<"${SEEDS:-100 200 300}"

  echo "eval entry     = ${PLAN_ENTRY}"
  echo "settings       = ${SETTINGS}"
  echo "hydra run dir  = ${HYDRA_RUN_DIR:-<shipped conf template, no override>}"

  # Evaluation_Protocol, unmodified: 50 samples per data seed, seeds 100/200/300,
  # open-loop mode=last alpha=1, MPC mode=staged alpha=1.  decode_for_viz=false
  # does not touch success_rate (computed from env state) and keeps MPC's growing
  # rollout from pressuring the MIG allocator.
  if setting_selected ol; then
    for s in "${seeds[@]}"; do
      CMD=(python "$PLAN_ENTRY" --config-name plan_gd.yaml)
      add_default "ckpt_base_path=${run_dir}"
      add_default "model_name=${name}"
      add_default "model_epoch=${MODEL_EPOCH:-latest}"
      add_default decode_for_viz=false
      add_default objective.alpha=1
      add_default objective.mode=last
      add_default "seed=${s}"
      add_run_dir_default plan_gd
      CMD+=(${USER_ARGS[@]+"${USER_ARGS[@]}"})
      run_job "OPEN-LOOP seed=${s}  ${name}" "${CMD[@]}"
    done
  fi

  if setting_selected mpc; then
    for s in "${seeds[@]}"; do
      CMD=(python "$PLAN_ENTRY" --config-name plan_gd_mpc.yaml)
      add_default "ckpt_base_path=${run_dir}"
      add_default "model_name=${name}"
      add_default "model_epoch=${MODEL_EPOCH:-latest}"
      add_default decode_for_viz=false
      add_default objective.alpha=1
      add_default objective.mode=staged
      add_default "seed=${s}"
      add_run_dir_default plan_gd_mpc
      CMD+=(${USER_ARGS[@]+"${USER_ARGS[@]}"})
      run_job "MPC seed=${s}  ${name}" "${CMD[@]}"
    done
  fi

  echo
  echo "Binomial SE at n=50 near p=0.8 is ~5.7 percentage points (Requirement 10.4);"
  echo "report it alongside every success rate. Aggregate with:"
  echo "  python aggregate_results.py"
}

# ---------------------------------------------------------------------------
# Detached driver body.  Re-entry point, not for direct use.
# Its own $$ is the pid to chain the next arm on: it stays alive for the whole
# sequence of jobs, whereas any individual python child comes and goes.
# ---------------------------------------------------------------------------
worker_main() {
  local mode="$1"; shift
  USER_ARGS=(${@+"$@"})

  echo "$$" > "$CCR_PIDFILE"
  # shellcheck disable=SC2064
  trap "rm -f '$CCR_PIDFILE'" EXIT

  echo "driver pid   = $$   (chain the next arm on THIS pid: CHAIN_ON_PID=$$)"
  echo "mode         = ${mode}"
  echo "log          = ${CCR_LOG}"
  echo "pid file     = ${CCR_PIDFILE}"
  echo "cwd          = ${PWD}"
  echo "host         = $(hostname)"

  if [[ -n "${CCR_CHAIN_ON_PID:-}" ]]; then
    wait_for_driver_pid "$CCR_CHAIN_ON_PID"
  fi

  apply_env "$mode"
  echo
  echo "--- environment (Requirement 9.1-9.4) ---"
  echo "PYTORCH_CUDA_ALLOC_CONF = ${PYTORCH_CUDA_ALLOC_CONF}"
  echo "CUDA_VISIBLE_DEVICES    = ${CUDA_VISIBLE_DEVICES-<unset>}"
  echo "OMP/MKL/OPENBLAS/NUMEXPR= ${OMP_NUM_THREADS}/${MKL_NUM_THREADS}/${OPENBLAS_NUM_THREADS}/${NUMEXPR_NUM_THREADS}"
  echo "PLAN_SERIAL_ENV         = ${PLAN_SERIAL_ENV-<unset>}"
  echo "DATASET_DIR             = ${DATASET_DIR}"
  echo "LD_LIBRARY_PATH         = ${LD_LIBRARY_PATH}"
  echo

  # Re-run the guard here as well: with CHAIN_ON_PID the slice was legitimately
  # busy at launch time, so this is the only check that sees it free.
  preflight_or_die || exit 1

  if [[ "$mode" == "eval" ]]; then
    local run_dir="${CCR_RUN_DIR:?internal: CCR_RUN_DIR unset for eval}"
    run_eval_jobs "$run_dir"
  else
    build_train_cmd "$mode"
    run_job "TRAIN (${mode})" "${CMD[@]}"
  fi

  echo
  if (( FAILURES > 0 )); then
    echo "=== ${mode} FINISHED WITH ${FAILURES} FAILED JOB(S) ==="
    exit 1
  fi
  echo "=== ${mode} DONE ($(date -Is)) ==="
}

# ---------------------------------------------------------------------------
# Resolve the run directory by GREPPING THE LOG, never from a hardcoded path.
# The Hydra run dir is derived from the objective (and now from ccr_tag), so a
# launcher's printed path goes stale the moment the naming expression changes —
# this has already bitten this project once (SHORT_BUDGET_PILOTS.md section 3).
# ---------------------------------------------------------------------------
resolve_run_dir_from_log() {
  local waited=0 step=5 limit="${RUNDIR_WAIT:-240}"
  while (( waited < limit )); do
    if [[ -s "$CCR_LOG" ]] && grep -qaE 'Model saved dir:' "$CCR_LOG"; then
      grep -m1 -aE 'Model saved dir:' "$CCR_LOG" | sed -E 's/.*Model saved dir:[[:space:]]*//'
      return 0
    fi
    if [[ -s "$CCR_LOG" ]] && grep -qaE '!!! FAILED|Traceback|REFUSING TO START' "$CCR_LOG"; then
      return 2
    fi
    sleep "$step"
    waited=$((waited + step))
  done
  return 1
}

report_launch() {
  local run_dir rc lam
  lam="$(resolved_lambda_cf)"

  # A chained launch has not started train.py yet: the driver is in
  # wait_for_driver_pid, polling every 30s. Waiting RUNDIR_WAIT seconds for a
  # "Model saved dir:" line that cannot exist yet, and then reporting its absence
  # as a problem, is a false alarm -- so say what is actually happening instead.
  if [[ -n "${CCR_CHAIN_ON_PID:-}" ]]; then
    echo
    echo "Chained launch: this driver is waiting for pid ${CCR_CHAIN_ON_PID} to exit,"
    echo "so train.py has NOT started and there is no run directory yet. That is"
    echo "expected -- do not read it as a failure."
    echo
    echo "When the chain releases, resolve the run dir and run the smoke check:"
    echo "  grep -m1 -aE 'Model saved dir:' '${CCR_LOG}'"
    echo "  grep -aE 'CCR enabled|CCR disabled|MCA enabled|Iteration budget' '${CCR_LOG}'"
    echo "  python summarize_training_log.py \"\$(grep -m1 -aE 'Model saved dir:' '${CCR_LOG}' \\"
    echo "    | sed -E 's/.*Model saved dir:[[:space:]]*//')\""
    return 0
  fi

  echo
  echo "Resolving the run directory from the log (not from a hardcoded path)..."
  set +e
  run_dir="$(resolve_run_dir_from_log)"
  rc=$?
  set -e
  case "$rc" in
    0)
      echo "run dir = ${run_dir}"
      echo
      echo "Two-minute smoke check (pilot gate check 1):"
      echo "  grep -aE 'CCR enabled|CCR disabled|MCA enabled|Iteration budget|Model saved dir' '${CCR_LOG}'"
      echo "  ls -l '${run_dir}/checkpoints/'      # model_latest.pth exists within seconds"
      if ccr_launch_enabled "$lam"; then
        echo "This launch enables CCR (training.lambda_cf=${lam}), so confirm the term"
        echo "really ran: a 'ccr' entry in the telemetry record's enabled_terms (the ccr"
        echo "block reads enabled: true). ONLY THEN does the arm field matter -- a"
        echo "'synthetic' arm reporting synthesized_action_frames=0 is silently a"
        echo "'logged' arm and the launch is wrong."
      else
        echo "This launch does NOT enable CCR (training.lambda_cf=${lam}), so the check is"
        echo "the opposite: expect 'CCR disabled (lambda_cf=0.0)' in the log and NO 'ccr'"
        echo "term in the telemetry (the ccr block reads enabled: false). The arm fields"
        echo "(action_source / synthesized_action_frames) describe nothing on a CCR-off"
        echo "run and are not reported."
      fi
      echo
      echo "Telemetry:"
      echo "  python summarize_training_log.py '${run_dir}'"
      ;;
    2)
      echo "The log already reports a failure. Read it before relaunching:" >&2
      echo "  tail -40 '${CCR_LOG}'" >&2
      ;;
    *)
      echo "No 'Model saved dir:' line after ${RUNDIR_WAIT:-240}s. Find it yourself:" >&2
      echo "  grep -m1 -aE 'Model saved dir:' '${CCR_LOG}'" >&2
      echo "  ls -d checkpoints*/test/*/ -t | head -3" >&2
      ;;
  esac
}

main() {
  local mode="${1:-}"
  [[ $# -gt 0 ]] && shift || true

  case "$mode" in
    -h|--help|help|"")
      usage
      [[ -z "$mode" ]] && exit 2 || exit 0
      ;;
    pilot|full|eval) ;;
    *) echo "ERROR: unknown mode '${mode}'. Expected pilot | full | eval." >&2; echo >&2; usage >&2; exit 2 ;;
  esac

  # In eval mode the first positional that is not a key=value override is the
  # run dir; everything else is passed through to plan.py.
  local run_dir="${RUN_DIR:-}"
  USER_ARGS=()
  local arg
  for arg in ${@+"$@"}; do
    if [[ "$mode" == "eval" && -z "$run_dir" && "$arg" != *=* ]]; then
      run_dir="$arg"
    else
      USER_ARGS+=("$arg")
    fi
  done
  if [[ "$mode" == "eval" && -z "$run_dir" ]]; then
    die "eval needs a run dir: bash run_ccr_pilot.sh eval <run_dir> [overrides...]"
  fi
  # Fail on a typo'd PLAN_ENTRY / SETTINGS here, in the foreground, rather than in a line
  # buried in a detached log; run_eval_jobs re-checks inside the driver.
  if [[ "$mode" == "eval" ]]; then
    validate_eval_hooks
  fi

  export CCR_LOG="${LOG:-ccr_${mode}_$(date +%Y%m%d_%H%M%S).log}"
  export CCR_PIDFILE="${PIDFILE:-${CCR_LOG%.log}.pid}"
  export CCR_CHAIN_ON_PID="${CHAIN_ON_PID:-}"
  if [[ -n "$run_dir" ]]; then
    export CCR_RUN_DIR="$run_dir"
  fi

  # Validate the environment now, in the foreground, so a missing DATASET_DIR is
  # an immediate non-zero exit instead of a line buried in a detached log.
  apply_env "$mode"

  if [[ -n "$CCR_CHAIN_ON_PID" ]]; then
    echo "CHAIN_ON_PID=${CCR_CHAIN_ON_PID} set: the slice is expected to be busy now,"
    echo "so the ps pre-flight runs inside the detached driver once that pid exits."
  else
    preflight_or_die || exit 1
  fi

  if [[ "${FOREGROUND:-0}" == "1" ]]; then
    worker_main "$mode" ${USER_ARGS[@]+"${USER_ARGS[@]}"}
    return
  fi

  echo "launching detached; log = ${CCR_LOG}"
  # setsid + nohup: survives a dropped browser/laptop session.  NOTE: $! here is
  # setsid's pid, and setsid forks and exits immediately (whenever it is already
  # a process-group leader), so $! is NOT this driver's pid and must never be
  # used for chaining. The detached driver writes its own $$ to CCR_PIDFILE;
  # that pid spans all of its sequential jobs, which is what a chain needs.
  setsid nohup bash "$SCRIPT_PATH" __worker "$mode" ${USER_ARGS[@]+"${USER_ARGS[@]}"} \
    > "$CCR_LOG" 2>&1 < /dev/null &

  local waited=0
  while (( waited < 20 )) && [[ ! -s "$CCR_PIDFILE" ]]; do
    sleep 1
    waited=$((waited + 1))
  done

  if [[ -s "$CCR_PIDFILE" ]]; then
    local driver_pid
    driver_pid="$(cat "$CCR_PIDFILE")"
    echo "driver pid = ${driver_pid}  (written to ${CCR_PIDFILE})"
    echo
    echo "  watch:        tail -f ${CCR_LOG}"
    echo "  alive?:       ps -p ${driver_pid} -o pid,stat,etime,cmd"
    # `kill ${driver_pid}` is WRONG and this script used to print it: setsid puts the
    # driver, train.py and its ~16 dataloader workers in one process group, and killing
    # only the driver ORPHANS the python child, which keeps running and keeps holding the
    # whole MIG slice. The negative-pid form signals the entire group.
    echo "  stop:         kill -- -${driver_pid}       # whole process group, not just the driver"
    echo "  verify stop:  ps -eo pid,stat,etime,cmd | grep '[p]ython train' || echo clear"
    echo "  queue next:   CHAIN_ON_PID=${driver_pid} bash run_ccr_pilot.sh ${mode} <overrides>"
  else
    echo "WARNING: ${CCR_PIDFILE} was not written. Check ${CCR_LOG}." >&2
  fi

  if [[ "$mode" == "eval" ]]; then
    echo
    echo "  results:      grep -ah success_rate ${CCR_LOG} | tail -n 2"
    echo "                python aggregate_results.py"
  else
    report_launch
  fi
}

if [[ "${1:-}" == "__worker" ]]; then
  shift
  worker_main ${@+"$@"}
else
  main ${@+"$@"}
fi
