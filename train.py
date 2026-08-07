import os
import time
import json
import hydra
import torch
import wandb
import logging
import warnings
import threading
import itertools
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf, open_dict
from einops import rearrange
from accelerate import Accelerator
from torchvision import utils
import torch.distributed as dist
from pathlib import Path
from collections import OrderedDict
from hydra.types import RunMode
from hydra.core.hydra_config import HydraConfig
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor
from metrics.image_metrics import eval_images
from utils import slice_trajdict_with_t, cfg_to_dict, seed, sample_tensors
import custom_resolvers  # noqa: F401  # Registers OmegaConf resolvers at import time.

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)


# --- Run-directory loss-configuration guard (Requirement 6.6) -----------------
# The Hydra run directory is derived from the objective, so two arms that differ
# only in a loss knob can resolve to the same directory and silently auto-resume
# each other. That has already cost this project a run once
# (SHORT_BUDGET_PILOTS.md section 3). A "loss signature" is every configuration
# value that changes the objective.
LOSS_SIGNATURE_KEYS = (
    "straighten",
    "stop_grad",
    "vcreg",
    "vcreg_std_coeff",
    "vcreg_cov_coeff",
    "lambda_cf",
    "ccr_rho",
    "ccr_rollout_len",
    "ccr_action_source",
    "mca_weight",
)

# Used only when falling back to a run that predates `loss_config.json`: a key the
# previous run never wrote is read as the value it would have had.
LOSS_SIGNATURE_DEFAULTS = {
    "straighten": False,
    "stop_grad": True,
    "vcreg": False,
    "vcreg_std_coeff": 0,
    "vcreg_cov_coeff": 0,
    "lambda_cf": 0.0,
    "ccr_rho": 0.0,
    "ccr_rollout_len": 5,
    "ccr_action_source": "synthetic",
    "mca_weight": 0.0,
}

LOSS_CONFIG_BASENAME = "loss_config.json"

# --- Telemetry (Requirements 6.7, 6.8) ----------------------------------------
# Loss components currently go to wandb only, which under WANDB_MODE=offline means
# they are not readable from the shell. This sink makes each term's scaled value and
# its share of the objective greppable, which is what decides a pilot
# (SHORT_BUDGET_PILOTS.md section 6). Consumer: summarize_training_log.py.
TELEMETRY_BASENAME = "training_log.jsonl"

# (key in `loss_components`, telemetry name). Ordered, so `enabled_terms` and the
# `terms` block come out in a stable order run to run. A key that is absent from
# `loss_components` (its term is disabled) is simply omitted from the record.
TELEMETRY_TERMS = (
    ("z_loss", "prediction"),
    ("curvature_loss_scaled", "curvature"),
    ("ccr_loss_scaled", "ccr"),
    ("mca_loss_scaled", "mca"),
    ("z_vcreg_loss_scaled", "vcreg"),
    ("decoder_loss_reconstructed", "decoder"),
)

# The `loss_components` key that decides whether the CCR term contributed to this
# iteration's objective. Single source of truth for both the `ccr` entry in `terms`
# and the `enabled` flag of the `ccr` block, so the two can never disagree.
TELEMETRY_CCR_KEY = "ccr_loss_scaled"


def _cfg_value(cfg_node, key, default):
    """
    Read one key from a config node, treating an absent key and an explicit `null`
    the same way: fall back to `default`. Hydra keys are the single source of truth
    (Requirement 3.5); `default` here only covers a config node that predates a key.
    """
    if cfg_node is None:
        return default
    try:
        value = cfg_node.get(key, default)
    except AttributeError:
        value = getattr(cfg_node, key, default)
    return default if value is None else value


def _cfg_int(cfg_node, key, default):
    try:
        return int(_cfg_value(cfg_node, key, default))
    except (TypeError, ValueError):
        log.warning(
            "training.%s is not an integer; falling back to %s", key, default
        )
        return int(default)


def _cfg_float(cfg_node, key, default):
    try:
        return float(_cfg_value(cfg_node, key, default))
    except (TypeError, ValueError):
        log.warning("training.%s is not a number; falling back to %s", key, default)
        return float(default)


def _as_plain_float(value):
    """
    Plain Python float for JSON, or None if the value cannot be one.

    `.item()` is called only on tensors: by the time telemetry sees them the loss
    components have already been gathered and reduced to floats, and calling
    `.item()` on a float would raise.
    """
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach()
        if value.numel() != 1:
            value = value.mean()
        value = value.float().cpu().item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _rounded(value, places=6):
    if value is None:
        return None
    return round(float(value), places)


def _normalize_signature_value(value):
    """
    Put a signature value into a form that survives a JSON round trip and compares
    stably: booleans stay booleans, numbers become floats (so `0` recorded by one
    run and `0.0` by another do not read as a conflict), everything else becomes a
    string (`straighten` is either `False` or a tag like `aggcos1e-1`).
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    return str(value)


def loss_signature_from_training_cfg(training_cfg):
    """Flat, JSON-serializable loss signature over LOSS_SIGNATURE_KEYS."""
    return {
        key: _normalize_signature_value(
            _cfg_value(training_cfg, key, LOSS_SIGNATURE_DEFAULTS[key])
        )
        for key in LOSS_SIGNATURE_KEYS
    }


def _is_rank_zero():
    """
    Which process may write, decided from the environment.

    The guard runs before the Accelerator is constructed, on purpose: it must abort
    before *any* training artifact is written, and that includes wandb's. So the
    rank comes from the launcher's environment rather than from `self.accelerator`.
    """
    for var in ("RANK", "SLURM_PROCID", "LOCAL_RANK"):
        value = os.environ.get(var)
        if value is None or not value.strip():
            continue
        try:
            return int(value) == 0
        except ValueError:
            return True
    return True


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        with open_dict(cfg):
            cfg["saved_folder"] = os.getcwd()
            log.info(f"Model saved dir: {cfg['saved_folder']}")
        # Immediately after `saved_folder` is known and before wandb.init, before
        # hydra.yaml is written and before any checkpoint write (Requirement 6.6).
        self._guard_run_dir()
        cfg_dict = cfg_to_dict(cfg)
        model_name = cfg_dict["saved_folder"].split("checkpoints/")[-1]
        model_name += f"_f{self.cfg.frameskip}_h{self.cfg.num_hist}_p{self.cfg.num_pred}"

        if HydraConfig.get().mode == RunMode.MULTIRUN:
            log.info(" Multirun setup begin...")
            log.info(f"SLURM_JOB_NODELIST={os.environ['SLURM_JOB_NODELIST']}")
            log.info(f"DEBUGVAR={os.environ['DEBUGVAR']}")
            # ==== init ddp process group ====
            os.environ["RANK"] = os.environ["SLURM_PROCID"]
            os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
            os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
            try:
                dist.init_process_group(
                    backend="nccl",
                    init_method="env://",
                    timeout=timedelta(minutes=5),  # Set a 5-minute timeout
                )
                log.info("Multirun setup completed.")
            except Exception as e:
                log.error(f"DDP setup failed: {e}")
                raise
            torch.distributed.barrier()
            # # ==== /init ddp process group ====

        mixed_precision = self.cfg.training.get("mixed_precision", "no")
        self.accelerator = Accelerator(
            log_with="wandb",
            mixed_precision=mixed_precision,
        )
        log.info(f"Accelerate mixed precision: {mixed_precision}")
        log.info(
            f"rank: {self.accelerator.local_process_index}  model_name: {model_name}"
        )
        self.device = self.accelerator.device
        log.info(f"device: {self.device}   model_name: {model_name}")
        self.base_path = os.path.dirname(os.path.abspath(__file__))

        self.num_reconstruct_samples = self.cfg.training.num_reconstruct_samples
        self.total_epochs = self.cfg.training.epochs
        self.epoch = 0
        # Optimizer steps completed across the whole run, checkpointed alongside
        # `epoch` so a resumed pilot honours the same total bound (Requirement 6.3).
        self.global_iter = 0
        # <= 0 means "run the configured epochs", i.e. exactly today's behaviour
        # (Requirement 6.2). The cap can only ever shorten a run, never lengthen it.
        self.max_iterations = _cfg_int(self.cfg.training, "max_iterations", 0)
        self._stop_requested = False

        # JSONL telemetry sink, appended in the Hydra run directory (which is cwd).
        # Default cadence 200 matches the reference run, so step-200 rows are
        # directly comparable between arms.
        self.telemetry_every = _cfg_int(
            self.cfg.training, "telemetry_every_x_iterations", 200
        )
        self._telemetry_path = Path(cfg["saved_folder"]) / TELEMETRY_BASENAME
        self._telemetry_start = time.perf_counter()
        self._telemetry_last_time = self._telemetry_start
        # Steps taken by *this process*, used for it_per_s. `global_iter` cannot be
        # used for the rate because a resumed run starts it from the checkpoint.
        self._iters_this_process = 0
        self._telemetry_last_iters = 0
        self._telemetry_last_written = None
        self._telemetry_failed = False
        self.decoder_start_epoch = int(self.cfg.training.get("decoder_start_epoch", 1))
        if self.decoder_start_epoch < 1:
            log.warning(
                f"decoder_start_epoch={self.decoder_start_epoch} is invalid; clamping to 1"
            )
            self.decoder_start_epoch = 1
        log.info(f"Decoder training will start at epoch {self.decoder_start_epoch}")

        assert cfg.training.batch_size % self.accelerator.num_processes == 0, (
            "Batch size must be divisible by the number of processes. "
            f"Batch_size: {cfg.training.batch_size} num_processes: {self.accelerator.num_processes}."
        )

        OmegaConf.set_struct(cfg, False)
        cfg.effective_batch_size = cfg.training.batch_size
        cfg.gpu_batch_size = cfg.training.batch_size // self.accelerator.num_processes
        OmegaConf.set_struct(cfg, True)

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            wandb_run_id = None
            if os.path.exists("hydra.yaml"):
                existing_cfg = OmegaConf.load("hydra.yaml")
                wandb_run_id = existing_cfg["wandb_run_id"]
                log.info(f"Resuming Wandb run {wandb_run_id}")

            wandb_dict = OmegaConf.to_container(cfg, resolve=True)
            if self.cfg.debug:
                log.info("WARNING: Running in debug mode...")
                self.wandb_run = wandb.init(
                    project=f"temporal_straightening_{self.cfg.env.name}",
                    config=wandb_dict,
                    id=wandb_run_id,
                    resume="allow",
                )
            else:
                self.wandb_run = wandb.init(
                    project=f"temporal_straightening_{self.cfg.env.name}",
                    config=wandb_dict,
                    id=wandb_run_id,
                    resume="allow",
                )
            OmegaConf.set_struct(cfg, False)
            cfg.wandb_run_id = self.wandb_run.id
            OmegaConf.set_struct(cfg, True)
            wandb.run.name = "{}".format(model_name)
            with open(os.path.join(os.getcwd(), "hydra.yaml"), "w") as f:
                f.write(OmegaConf.to_yaml(cfg, resolve=True))

        seed(cfg.training.seed)
        log.info(f"Loading dataset from {self.cfg.env.dataset.data_path} ...")
        self.datasets, traj_dsets = hydra.utils.call(
            self.cfg.env.dataset,
            num_hist=self.cfg.num_hist,
            num_pred=self.cfg.num_pred,
            frameskip=self.cfg.frameskip,
        )

        self.train_traj_dset = traj_dsets["train"]
        self.val_traj_dset = traj_dsets["valid"]

        self.dataloaders = {
            x: torch.utils.data.DataLoader(
                self.datasets[x],
                batch_size=self.cfg.gpu_batch_size,
                shuffle=False, # already shuffled in TrajSlicerDataset
                num_workers=self.cfg.env.num_workers,
                collate_fn=None,
                pin_memory=True,
                persistent_workers=True,
            )
            for x in ["train", "valid"]
        }

        log.info(f"dataloader batch size: {self.cfg.gpu_batch_size}")

        self.dataloaders["train"], self.dataloaders["valid"] = self.accelerator.prepare(
            self.dataloaders["train"], self.dataloaders["valid"]
        )

        self.encoder = None
        self.action_encoder = None
        self.proprio_encoder = None
        self.predictor = None
        self.decoder = None
        self.train_encoder = self.cfg.model.train_encoder
        self.train_predictor = self.cfg.model.train_predictor
        self.train_decoder = self.cfg.model.train_decoder
        log.info(f"Train encoder, predictor, decoder:\
            {self.cfg.model.train_encoder}\
            {self.cfg.model.train_predictor}\
            {self.cfg.model.train_decoder}")

        self._keys_to_save = [
            "epoch",
            # Persisted so the cap is counted against the whole run, not this
            # process (Requirement 6.3). Checkpoints written before this feature
            # have no such key: `load_ckpt` reports it through the existing
            # "Keys not found in ckpt" warning and the counter simply starts at 0.
            "global_iter",
        ]
        self._keys_to_save += (
            ["encoder", "encoder_optimizer"] if self.train_encoder else []
        )
        self._keys_to_save += (
            ["predictor", "predictor_optimizer"]
            if self.train_predictor and self.cfg.has_predictor
            else []
        )
        self._keys_to_save += (
            ["decoder", "decoder_optimizer"] if self.train_decoder else []
        )
        self._keys_to_save += ["action_encoder", "proprio_encoder"]

        self.init_models()
        self.init_optimizers()

        # After init_models, so the line reflects the resumed `global_iter` rather
        # than a pre-resume 0.
        self._log_iteration_budget()

        self.epoch_log = OrderedDict()

    def _guard_run_dir(self, run_dir=None):
        """
        Refuse to continue a run directory whose checkpoint was trained under a
        different objective (Requirement 6.6).

        The Hydra run directory is derived from the objective, so two arms differing
        only in a loss knob that is not in the run name resolve to one directory and
        silently auto-resume each other. Hydra creates the output directory and its
        `.hydra/` snapshot before user code runs; this guard covers every *training*
        artifact (checkpoints, hydra.yaml, training_log.jsonl, plots), which is what
        makes an accidental overwrite destructive.

            1. No `checkpoints/model_latest.pth`: nothing to conflict with. Record
               the signature and return.
            2. Otherwise compare against `loss_config.json`, falling back to the
               previous run's resolved `hydra.yaml` with missing keys read as their
               defaults. If neither is readable, warn and proceed so legacy runs
               stay resumable.
            3. On mismatch, raise before writing anything.
        """
        run_dir = Path(self.cfg["saved_folder"] if run_dir is None else run_dir)
        current = loss_signature_from_training_cfg(self.cfg.training)
        ckpt_path = run_dir / "checkpoints" / "model_latest.pth"
        record_path = run_dir / LOSS_CONFIG_BASENAME

        if not ckpt_path.exists():
            if _is_rank_zero():
                self._write_loss_config(record_path, current)
            return

        recorded, source = self._read_recorded_loss_signature(run_dir)
        if recorded is None:
            log.warning(
                "Run directory %s already contains %s but no readable loss "
                "configuration (%s or hydra.yaml), so the loss configuration of the "
                "existing checkpoint cannot be verified. Proceeding, which keeps "
                "runs predating this check resumable: confirm by hand that this "
                "launch shares the objective of that checkpoint.",
                run_dir,
                ckpt_path.name,
                LOSS_CONFIG_BASENAME,
            )
            return

        differing = {
            key: (recorded.get(key), current[key])
            for key in LOSS_SIGNATURE_KEYS
            if recorded.get(key) != current[key]
        }
        if differing:
            detail = "; ".join(
                f"{key}: recorded={rec!r} current={cur!r}"
                for key, (rec, cur) in differing.items()
            )
            raise RuntimeError(
                f"Loss-configuration conflict in run directory {run_dir}: it already "
                f"contains {ckpt_path.name} trained under a different objective "
                f"(recorded signature read from {source}). Differing keys -> "
                f"{detail}. Nothing was written. Auto-resuming here would continue "
                f"and overwrite that run. Give this arm its own directory (put the "
                f"knob in the run name, or override ckpt_base_path) before relaunching."
            )

        log.info(
            "Loss configuration matches the checkpoint already in %s (verified "
            "against %s); resuming is safe.",
            run_dir,
            source,
        )

    def _write_loss_config(self, record_path, signature):
        try:
            record_path.parent.mkdir(parents=True, exist_ok=True)
            with open(record_path, "w", encoding="utf-8") as fh:
                json.dump(signature, fh, indent=2, sort_keys=False)
                fh.write("\n")
        except OSError as exc:
            # Not fatal: failing to record the signature must not stop a run, but a
            # later launch then cannot be guarded, so say so loudly.
            log.warning(
                "Could not write %s (%s); a later launch into this directory cannot "
                "be checked for a loss-configuration conflict.",
                record_path,
                exc,
            )
        else:
            log.info("Recorded loss configuration in %s", record_path)

    def _read_recorded_loss_signature(self, run_dir):
        """
        (signature, source) for the run already in `run_dir`, or (None, "").

        Keys absent from the recorded source are read as their defaults, which is
        what makes a run predating a knob comparable against one that has it.
        """
        record_path = Path(run_dir) / LOSS_CONFIG_BASENAME
        if record_path.is_file():
            try:
                with open(record_path, "r", encoding="utf-8") as fh:
                    raw = json.load(fh)
            except (OSError, ValueError) as exc:
                log.warning("Could not read %s (%s)", record_path, exc)
            else:
                if isinstance(raw, dict):
                    return (
                        {
                            key: _normalize_signature_value(
                                raw[key]
                                if raw.get(key) is not None
                                else LOSS_SIGNATURE_DEFAULTS[key]
                            )
                            for key in LOSS_SIGNATURE_KEYS
                        },
                        str(record_path),
                    )
                log.warning("%s is not a JSON object; ignoring it", record_path)

        # Fall back to the resolved config the previous run wrote.
        hydra_yaml = Path(run_dir) / "hydra.yaml"
        if hydra_yaml.is_file():
            try:
                previous = OmegaConf.load(hydra_yaml)
            except Exception as exc:  # OmegaConf raises a family of errors here
                log.warning("Could not read %s (%s)", hydra_yaml, exc)
            else:
                training_cfg = None
                if previous is not None:
                    try:
                        training_cfg = previous.get("training", None)
                    except AttributeError:
                        training_cfg = None
                if training_cfg is not None:
                    return (
                        loss_signature_from_training_cfg(training_cfg),
                        str(hydra_yaml),
                    )
                log.warning("%s has no `training` section; ignoring it", hydra_yaml)

        return None, ""

    def _write_telemetry(self, components, batch_index, force=False):
        """
        Append one telemetry record for the current iteration (Requirements 6.7, 6.8).

        Cadence is `training.telemetry_every_x_iterations`, plus `force=True` for the
        final step of a capped run. Main process only, one JSON object per line,
        flushed on write so a killed job still leaves every completed row on disk.
        """
        if self._telemetry_failed or not self.accelerator.is_main_process:
            return
        if not force:
            if self.telemetry_every <= 0:
                return
            if self.global_iter % self.telemetry_every != 0:
                return
        if self._telemetry_last_written == self.global_iter:
            # The cap's forced row would otherwise duplicate a cadence row when the
            # cap happens to land on the cadence.
            return

        record = self._telemetry_record(components, batch_index)
        try:
            with open(self._telemetry_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record))
                fh.write("\n")
                fh.flush()
        except OSError as exc:
            # A log write must not kill a 17-hour run, but losing telemetry loses the
            # pilot verdict, so report it once and stop trying.
            self._telemetry_failed = True
            log.warning(
                "Telemetry disabled: could not append to %s (%s). The pilot verdict "
                "is read out of this file, so fix the path before relaunching.",
                self._telemetry_path,
                exc,
            )
            return
        self._telemetry_last_written = self.global_iter

    def _telemetry_record(self, components, batch_index):
        now = time.perf_counter()
        elapsed = now - self._telemetry_last_time
        steps = self._iters_this_process - self._telemetry_last_iters
        it_per_s = steps / elapsed if elapsed > 0 and steps > 0 else None
        self._telemetry_last_time = now
        self._telemetry_last_iters = self._iters_this_process

        total = _as_plain_float(components.get("loss"))
        terms = OrderedDict()
        for key, name in TELEMETRY_TERMS:
            if key not in components:
                continue  # the term is disabled this run
            scaled = _as_plain_float(components[key])
            if scaled is None:
                continue
            # Share against the same iteration's total, from the already-gathered
            # per-iteration dict, so it is the number that drove this step rather
            # than an epoch mean.
            share = scaled / total if total not in (None, 0.0) else None
            terms[name] = {"scaled": _rounded(scaled), "share": _rounded(share)}

        return {
            "global_iter": int(self.global_iter),
            "epoch": int(self.epoch),
            # Batches completed in this epoch, so it equals `global_iter` for a run
            # that started from scratch in its first epoch.
            "iter_in_epoch": int(batch_index) + 1,
            "wall_time_s": _rounded(now - self._telemetry_start, 3),
            "it_per_s": _rounded(it_per_s, 4),
            "loss": _rounded(total),
            "terms": terms,
            "enabled_terms": list(terms),
            "ccr": self._ccr_telemetry_block(components),
        }

    def _ccr_telemetry_block(self, components):
        """
        Self-describing CCR block: reading a pilot's telemetry six weeks later must
        not require reconstructing which arm it was from the directory name.

        `enabled` comes first and is derived exactly like the `terms` block: from the
        presence of the CCR key in `loss_components`, i.e. the model's own gate
        actually fired this iteration. It is NOT re-read from config, because config
        says what was asked for and `loss_components` says what ran. When the term
        did not run, the arm fields (`rho`, `rollout_len`, `action_source`,
        `synthesized_action_frames`) describe nothing, so they are omitted; only
        `lambda_cf` is kept, because it says *why* the arm is off.
        """
        training_cfg = self.cfg.training
        lambda_cf = _cfg_float(training_cfg, "lambda_cf", 0.0)
        # Ground truth: the same key `terms`/`enabled_terms` are built from.
        enabled = TELEMETRY_CCR_KEY in components
        if enabled != (lambda_cf > 0.0):
            # Config and the model's gate disagree. Report what ran; flag the rest.
            log.warning(
                "CCR telemetry: training.lambda_cf=%s but %s is %s in loss_components; "
                "reporting enabled=%s (what actually ran, not what config asked for).",
                lambda_cf,
                TELEMETRY_CCR_KEY,
                "present" if enabled else "absent",
                enabled,
            )
        if not enabled:
            block = OrderedDict()
            block["enabled"] = False
            block["lambda_cf"] = lambda_cf
            return block

        rollout_len = _cfg_int(training_cfg, "ccr_rollout_len", 5)
        action_source = str(_cfg_value(training_cfg, "ccr_action_source", "synthetic"))
        num_hist = int(self.cfg.num_hist)
        num_frames = num_hist + int(self.cfg.num_pred)
        # Frames past the recorded window that have to be synthesized to reach the
        # imagined horizon. `logged` never synthesizes: an infeasible horizon is
        # rejected upstream instead. A `synthetic` arm reporting 0 here is silently
        # a `logged` arm, which summarize_training_log.py flags.
        synthesized = (
            max(0, num_hist + rollout_len - 1 - num_frames)
            if action_source == "synthetic"
            else 0
        )
        block = OrderedDict()
        block["enabled"] = True
        raw = _as_plain_float(components.get("ccr_loss"))
        if raw is not None:
            block["raw"] = _rounded(raw)
        block["lambda_cf"] = lambda_cf
        block["rho"] = _cfg_float(training_cfg, "ccr_rho", 0.0)
        block["rollout_len"] = rollout_len
        block["action_source"] = action_source
        block["synthesized_action_frames"] = synthesized
        return block

    def _log_iteration_budget(self):
        """
        Log the run's iteration arithmetic at startup so a pilot cap can be set from
        real numbers instead of a guess (SHORT_BUDGET_PILOTS.md section 2):

            Iteration budget: steps/epoch=61929 epochs=3 total=185787 max_iterations=8000 (cap active)
        """
        try:
            steps_per_epoch = len(self.dataloaders["train"])
        except TypeError:  # an iterable-style dataloader has no length
            steps_per_epoch = None
        total = (
            steps_per_epoch * self.total_epochs if steps_per_epoch is not None else None
        )
        log.info(
            "Iteration budget: steps/epoch=%s epochs=%s total=%s max_iterations=%s %s",
            steps_per_epoch if steps_per_epoch is not None else "unknown",
            self.total_epochs,
            total if total is not None else "unknown",
            self.max_iterations,
            "(cap active)" if self.max_iterations > 0 else "(no cap)",
        )
        if self.max_iterations > 0 and total is not None and self.max_iterations >= total:
            # The cap can only shorten a run, so a cap at or above the budget is inert.
            log.warning(
                "training.max_iterations=%s is >= the %s-step budget, so the epoch "
                "boundary will end this run, not the cap. SHORT_BUDGET_PILOTS.md "
                "section 2: raise training.epochs so the cap is what stops it.",
                self.max_iterations,
                total,
            )
        if self.global_iter:
            log.info(
                "Resumed with global_iter=%s already completed; %s step(s) remain "
                "under the cap.",
                self.global_iter,
                max(0, self.max_iterations - self.global_iter)
                if self.max_iterations > 0
                else "unbounded",
            )

    def _configure_encoder_trainability(self):
        base_model = getattr(self.encoder, "base_model", None)
        if base_model is not None:
            for param in base_model.parameters():
                param.requires_grad = False
            log.info("Encoder base_model is frozen.")

        if not self.train_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            log.info("Encoder is fully frozen (train_encoder=False).")
            return

        # train_encoder=True: keep non-backbone encoder modules trainable.
        for name, param in self.encoder.named_parameters():
            if not name.startswith("base_model."):
                param.requires_grad = True
        log.info("Encoder base_model frozen; non-backbone encoder modules are trainable.")

    def _log_trainable_params(self, module, module_name):
        if not self.accelerator.is_main_process:
            return
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        log.info(f"[{module_name}] trainable params: {trainable} / {total}")
        for name, param in module.named_parameters():
            if param.requires_grad:
                log.info(f"[{module_name}] trainable: {name} shape={tuple(param.shape)}")

    def decoder_training_active(self):
        return (
            self.cfg.has_decoder
            and self.train_decoder
            and self.epoch >= self.decoder_start_epoch
        )

    def save_ckpt(self):
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            if not os.path.exists("checkpoints"):
                os.makedirs("checkpoints")
            ckpt = {}
            for k in self._keys_to_save:
                v = self.__dict__.get(k, None)
                if k.endswith("_optimizer") and v is not None:
                    ckpt[k] = v.state_dict()
                elif hasattr(v, "module"):
                    ckpt[k] = self.accelerator.unwrap_model(v)
                else:
                    ckpt[k] = v
            torch.save(ckpt, "checkpoints/model_latest.pth")
            torch.save(ckpt, f"checkpoints/model_{self.epoch}.pth")
            log.info("Saved model to {}".format(os.getcwd()))
            ckpt_path = os.path.join(os.getcwd(), f"checkpoints/model_{self.epoch}.pth")
        else:
            ckpt_path = None
        model_name = self.cfg["saved_folder"].split("/")[-1]
        model_epoch = self.epoch
        return ckpt_path, model_name, model_epoch

    def load_ckpt(self, filename="model_latest.pth"):
        # weights_only=False: checkpoints store full nn.Module objects, not just
        # state-dicts. Required on torch>=2.6 (where weights_only defaults to True).
        ckpt = torch.load(filename, weights_only=False)
        self._loaded_optim_state = {}
        for k, v in ckpt.items():
            if k.endswith("_optimizer") and isinstance(v, dict):
                self._loaded_optim_state[k] = v
            else:
                self.__dict__[k] = v
        not_in_ckpt = set(self._keys_to_save) - set(ckpt.keys())
        if len(not_in_ckpt):
            log.warning("Keys not found in ckpt: %s", not_in_ckpt)

    def init_models(self):
        # Resume priority:
        #   1. An explicit checkpoint via training.resume_from=/abs/path/model_X.pth
        #      (useful to continue offline from a specific saved epoch).
        #   2. Otherwise auto-resume from model_latest.pth in this run's folder.
        # Because the encoder (incl. the DINOv2 backbone weights) is stored in the
        # checkpoint, resuming does NOT re-download anything and works with no internet.
        resume_from = self.cfg.training.get("resume_from", None)
        if resume_from:
            model_ckpt = Path(resume_from).expanduser()
            if not model_ckpt.exists():
                raise FileNotFoundError(
                    f"training.resume_from='{model_ckpt}' does not exist. "
                    "Point it at a valid model_<epoch>.pth checkpoint."
                )
        else:
            model_ckpt = Path(self.cfg.saved_folder) / "checkpoints" / "model_latest.pth"

        if model_ckpt.exists():
            self.load_ckpt(model_ckpt)
            log.info(f"Resuming from epoch {self.epoch}: {model_ckpt}")
        else:
            log.info("No checkpoint found; starting training from scratch.")

        # initialize encoder
        if self.encoder is None:
            encoder_kwargs = {}
            if (
                hasattr(self.cfg.encoder, "projector_config")
                and self.cfg.encoder.projector_config is not None
            ):
                encoder_kwargs["projector_config"] = hydra.utils.instantiate(
                    self.cfg.encoder.projector_config
                )
            self.encoder = hydra.utils.instantiate(
                self.cfg.encoder,
                **encoder_kwargs,
            )
        self._configure_encoder_trainability()

        self.proprio_encoder = hydra.utils.instantiate(
            self.cfg.proprio_encoder,
            in_chans=self.datasets["train"].proprio_dim,
            emb_dim=self.cfg.proprio_emb_dim,
        )
        proprio_emb_dim = self.proprio_encoder.emb_dim
        print(f"Proprio encoder type: {type(self.proprio_encoder)}")
        self.proprio_encoder = self.accelerator.prepare(self.proprio_encoder)

        self.action_encoder = hydra.utils.instantiate(
            self.cfg.action_encoder,
            in_chans=self.datasets["train"].action_dim,
            emb_dim=self.cfg.action_emb_dim,
        )
        action_emb_dim = self.action_encoder.emb_dim
        print(f"Action encoder type: {type(self.action_encoder)}")

        self.action_encoder = self.accelerator.prepare(self.action_encoder)

        if self.accelerator.is_main_process:
            self.wandb_run.watch(self.action_encoder)
            self.wandb_run.watch(self.proprio_encoder)

        # initialize predictor
        if self.encoder.latent_ndim == 1:  # if feature is 1D
            num_patches = 1
        else:
            decoder_scale = 16  # from vqvae
            num_side_patches = self.cfg.img_size // decoder_scale
            num_patches = num_side_patches**2

        if self.cfg.concat_dim == 0:
            num_patches += 2

        if self.cfg.has_predictor:
            if self.predictor is None:
                self.predictor = hydra.utils.instantiate(
                    self.cfg.predictor,
                    num_patches=num_patches,
                    num_frames=self.cfg.num_hist,
                    dim=self.encoder.emb_dim
                    + (
                        proprio_emb_dim * self.cfg.num_proprio_repeat
                        + action_emb_dim * self.cfg.num_action_repeat
                    )
                    * (self.cfg.concat_dim),
                )
            if not self.train_predictor:
                for param in self.predictor.parameters():
                    param.requires_grad = False

        # initialize decoder
        if self.cfg.has_decoder:
            if self.decoder is None:
                if self.cfg.env.decoder_path is not None:
                    decoder_path = os.path.join(
                        self.base_path, self.cfg.env.decoder_path
                    )
                    ckpt = torch.load(decoder_path, weights_only=False)
                    if isinstance(ckpt, dict):
                        self.decoder = ckpt["decoder"]
                    else:
                        self.decoder = torch.load(decoder_path, weights_only=False)
                    log.info(f"Loaded decoder from {decoder_path}")
                else:
                    decoder_kwargs = {
                        "emb_dim": self.encoder.emb_dim,  
                    }
                    if (
                        hasattr(self.cfg.encoder, "projector_config")
                        and self.cfg.encoder.projector_config is not None
                        and "conv_layers" in self.cfg.encoder.projector_config
                    ):
                        decoder_kwargs["projector_cfg"] = self.cfg.encoder.projector_config
                        log.info(f"Passing projector_cfg to decoder")
                    decoder_kwargs["_recursive_"] = False
                    self.decoder = hydra.utils.instantiate(self.cfg.decoder, **decoder_kwargs)
            if not self.train_decoder:
                for param in self.decoder.parameters():
                    param.requires_grad = False
        self.encoder, self.predictor, self.decoder = self.accelerator.prepare(
            self.encoder, self.predictor, self.decoder
        )
        self.model = hydra.utils.instantiate(
            self.cfg.model,
            encoder=self.encoder,
            proprio_encoder=self.proprio_encoder,
            action_encoder=self.action_encoder,
            predictor=self.predictor,
            decoder=self.decoder,
            proprio_dim=proprio_emb_dim,
            action_dim=action_emb_dim,
            concat_dim=self.cfg.concat_dim,
            num_action_repeat=self.cfg.num_action_repeat,
            num_proprio_repeat=self.cfg.num_proprio_repeat,
            straighten=self.cfg.training.get("straighten", False),
            stop_grad=self.cfg.training.get("stop_grad", True),
            vcreg=self.cfg.training.get("vcreg", False),
            vcreg_std_coeff=self.cfg.training.get("vcreg_std_coeff", 0),
            vcreg_cov_coeff=self.cfg.training.get("vcreg_cov_coeff", 0),
            vcreg_apply_to=self.cfg.training.get("vcreg_apply_to", "enc"),
            # CCR / MCA knobs come from conf/train.yaml only (no Python literal
            # fallback that could diverge from the yaml).
            lambda_cf=self.cfg.training.get("lambda_cf"),
            ccr_rho=self.cfg.training.get("ccr_rho"),
            ccr_rollout_len=self.cfg.training.get("ccr_rollout_len"),
            ccr_grad_checkpoint=self.cfg.training.get("ccr_grad_checkpoint"),
            ccr_fast_attention=self.cfg.training.get("ccr_fast_attention"),
            ccr_action_source=self.cfg.training.get("ccr_action_source"),
            mca_weight=self.cfg.training.get("mca_weight"),
        )
        self._log_trainable_params(self.model, "model")

    def init_optimizers(self):
        self.encoder_optimizer = torch.optim.Adam(
            self.encoder.parameters(),
            lr=self.cfg.training.encoder_lr,
        )
        self.encoder_optimizer = self.accelerator.prepare(self.encoder_optimizer)
        if getattr(self, "_loaded_optim_state", None) and "encoder_optimizer" in self._loaded_optim_state:
            try:
                self.encoder_optimizer.load_state_dict(self._loaded_optim_state["encoder_optimizer"])
                log.info(f"Loaded encoder optimizer state from checkpoint.")
            except Exception as e:
                log.warning(f"Failed to load encoder optimizer state: {e}")
        if self.cfg.has_predictor:
            self.predictor_optimizer = torch.optim.AdamW(
                self.predictor.parameters(),
                lr=self.cfg.training.predictor_lr,
            )
            self.predictor_optimizer = self.accelerator.prepare(
                self.predictor_optimizer
            )
            if getattr(self, "_loaded_optim_state", None) and "predictor_optimizer" in self._loaded_optim_state:
                try:
                    self.predictor_optimizer.load_state_dict(self._loaded_optim_state["predictor_optimizer"])
                    log.info(f"Loaded predictor optimizer state from checkpoint.")
                except Exception as e:
                    log.warning(f"Failed to load predictor optimizer state: {e}")

            self.action_encoder_optimizer = torch.optim.AdamW(
                itertools.chain(
                    self.action_encoder.parameters(), self.proprio_encoder.parameters()
                ),
                lr=self.cfg.training.action_encoder_lr,
            )
            self.action_encoder_optimizer = self.accelerator.prepare(
                self.action_encoder_optimizer
            )
            if getattr(self, "_loaded_optim_state", None) and "action_encoder_optimizer" in self._loaded_optim_state:
                try:
                    self.action_encoder_optimizer.load_state_dict(self._loaded_optim_state["action_encoder_optimizer"])
                    log.info(f"Loaded action/proprio optimizer state from checkpoint.")
                except Exception as e:
                    log.warning(f"Failed to load action/proprio optimizer state: {e}")

        if self.cfg.has_decoder:
            self.decoder_optimizer = torch.optim.Adam(
                self.decoder.parameters(), lr=self.cfg.training.decoder_lr
            )
            self.decoder_optimizer = self.accelerator.prepare(self.decoder_optimizer)
            if getattr(self, "_loaded_optim_state", None) and "decoder_optimizer" in self._loaded_optim_state:
                try:
                    self.decoder_optimizer.load_state_dict(self._loaded_optim_state["decoder_optimizer"])
                    log.info(f"Loaded decoder optimizer state from checkpoint.")
                except Exception as e:
                    log.warning(f"Failed to load decoder optimizer state: {e}")

    def monitor_jobs(self, lock):
        """
        check planning eval jobs' status and update logs
        """
        while True:
            with lock:
                finished_jobs = [
                    job_tuple for job_tuple in self.job_set if job_tuple[2].done()
                ]
                for epoch, job_name, job in finished_jobs:
                    result = job.result()
                    print(f"Logging result for {job_name} at epoch {epoch}: {result}")
                    log_data = {
                        f"{job_name}/{key}": value for key, value in result.items()
                    }
                    log_data["epoch"] = epoch
                    self.wandb_run.log(log_data)
                    self.job_set.remove((epoch, job_name, job))
            time.sleep(1)

    def run(self):
        if self.accelerator.is_main_process:
            executor = ThreadPoolExecutor(max_workers=4)
            self.job_set = set()
            lock = threading.Lock()

            self.monitor_thread = threading.Thread(
                target=self.monitor_jobs, args=(lock,), daemon=True
            )
            self.monitor_thread.start()

        # Baseline the telemetry clock here so `wall_time_s` and `it_per_s` measure
        # training, not the dataset and DINOv2 load that precede it.
        self._telemetry_start = time.perf_counter()
        self._telemetry_last_time = self._telemetry_start

        init_epoch = self.epoch + 1  # epoch starts from 1
        for epoch in range(init_epoch, init_epoch + self.total_epochs):
            self.epoch = epoch
            if self.accelerator.is_main_process:
                decoder_active = self.decoder_training_active()
                log.info(
                    "Epoch %s decoder_active=%s (train_decoder=%s, decoder_start_epoch=%s)",
                    self.epoch,
                    decoder_active,
                    self.train_decoder,
                    self.decoder_start_epoch,
                )
            self.accelerator.wait_for_everyone()
            self.train()
            self.accelerator.wait_for_everyone()
            if self._stop_requested:
                # The iteration cap stopped this epoch mid-way, so `val()` and
                # `logs_flash` are deliberately skipped. `logs_flash` formats
                # `train_loss` and `val_loss` and would raise KeyError with no
                # validation pass, and a partial-epoch validation number invites
                # comparison against full-epoch numbers, which
                # SHORT_BUDGET_PILOTS.md section 7b warns against. The cap already
                # wrote the final checkpoint and telemetry row; pilot judgement
                # comes from that telemetry and the offline probe.
                log.info(
                    "Stopping after epoch %s at global_iter=%s: iteration cap reached. "
                    "Validation and the epoch log flush are skipped because the epoch "
                    "is incomplete.",
                    self.epoch,
                    self.global_iter,
                )
                break
            self.val()
            self.logs_flash(step=self.epoch)
            if self.epoch % self.cfg.training.save_every_x_epoch == 0:
                ckpt_path, model_name, model_epoch = self.save_ckpt()
                # main thread only: launch planning jobs on the saved ckpt
                if (
                    self.cfg.plan_settings.plan_cfg_path is not None
                    and ckpt_path is not None
                ):  # ckpt_path is only not None for main process
                    from plan import build_plan_cfg_dicts, launch_plan_jobs

                    cfg_dicts = build_plan_cfg_dicts(
                        plan_cfg_path=os.path.join(
                            self.base_path, self.cfg.plan_settings.plan_cfg_path
                        ),
                        ckpt_base_path=self.cfg.ckpt_base_path,
                        model_name=model_name,
                        model_epoch=model_epoch,
                        planner=self.cfg.plan_settings.planner,
                        goal_source=self.cfg.plan_settings.goal_source,
                        goal_H=self.cfg.plan_settings.goal_H,
                        alpha=self.cfg.plan_settings.alpha,
                    )
                    jobs = launch_plan_jobs(
                        epoch=self.epoch,
                        cfg_dicts=cfg_dicts,
                        plan_output_dir=os.path.join(
                            os.getcwd(), "submitit-evals", f"epoch_{self.epoch}"
                        ),
                    )
                    with lock:
                        self.job_set.update(jobs)

    def err_eval_single(self, z_pred, z_tgt):
        logs = {}
        for k in z_pred.keys():
            loss = self.model.emb_criterion(z_pred[k], z_tgt[k])
            logs[k] = loss
        return logs

    def err_eval(self, z_out, z_tgt, state_tgt=None):
        """
        z_pred: (b, n_hist, n_patches, emb_dim), doesn't include action dims
        z_tgt: (b, n_hist, n_patches, emb_dim), doesn't include action dims
        state:  (b, n_hist, dim)
        """
        logs = {}
        slices = {
            "full": (None, None),
            "pred": (-self.model.num_pred, None),
            "next1": (-self.model.num_pred, -self.model.num_pred + 1),
        }
        for name, (start_idx, end_idx) in slices.items():
            z_out_slice = slice_trajdict_with_t(
                z_out, start_idx=start_idx, end_idx=end_idx
            )
            z_tgt_slice = slice_trajdict_with_t(
                z_tgt, start_idx=start_idx, end_idx=end_idx
            )
            z_err = self.err_eval_single(z_out_slice, z_tgt_slice)

            logs.update({f"z_{k}_err_{name}": v for k, v in z_err.items()})

        return logs

    def train(self):
        for i, data in enumerate(
            tqdm(self.dataloaders["train"], desc=f"Epoch {self.epoch} Train")
        ):
            obs, act, state = data
            plot = i == 0  # only plot from the first batch
            decoder_active = self.decoder_training_active()
            self.model.train_decoder = decoder_active
            self.model.train()
            if self.cfg.has_decoder:
                self.decoder.train(decoder_active)
            z_out, visual_out, visual_reconstructed, loss, loss_components = self.model(
                obs, act
            )

            self.encoder_optimizer.zero_grad()
            if decoder_active:
                self.decoder_optimizer.zero_grad()
            if self.cfg.has_predictor:
                self.predictor_optimizer.zero_grad()
                self.action_encoder_optimizer.zero_grad()

            self.accelerator.backward(loss)

            if self.model.train_encoder:
                self.encoder_optimizer.step()
            if decoder_active:
                self.decoder_optimizer.step()
            if self.cfg.has_predictor and self.model.train_predictor:
                self.predictor_optimizer.step()
                self.action_encoder_optimizer.step()

            loss = self.accelerator.gather_for_metrics(loss).mean()

            loss_components = self.accelerator.gather_for_metrics(loss_components)
            loss_components = {
                key: value.mean().item() for key, value in loss_components.items()
            }
            # Snapshot of this iteration's already-gathered scalars, taken before the
            # `train_` prefixing below. Telemetry shares are computed from these, so
            # they are the numbers that actually drove this step rather than an
            # epoch mean.
            iter_components = dict(loss_components)
            if decoder_active and plot:
                # only eval images when plotting due to speed
                if self.cfg.has_predictor:
                    z_obs_out, z_act_out = self.model.separate_emb(z_out)
                    z_gt = self.model.encode_obs(obs)
                    z_tgt = slice_trajdict_with_t(z_gt, start_idx=self.model.num_pred)

                    state_tgt = state[:, -self.model.num_hist :]  # (b, num_hist, dim)
                    err_logs = self.err_eval(z_obs_out, z_tgt)

                    err_logs = self.accelerator.gather_for_metrics(err_logs)
                    err_logs = {
                        key: value.mean().item() for key, value in err_logs.items()
                    }
                    err_logs = {f"train_{k}": [v] for k, v in err_logs.items()}

                    self.logs_update(err_logs)

                if visual_out is not None:
                    for t in range(
                        self.cfg.num_hist, self.cfg.num_hist + self.cfg.num_pred
                    ):
                        img_pred_scores = eval_images(
                            visual_out[:, t - self.cfg.num_pred], obs["visual"][:, t]
                        )
                        img_pred_scores = self.accelerator.gather_for_metrics(
                            img_pred_scores
                        )
                        img_pred_scores = {
                            f"train_img_{k}_pred": [v.mean().item()]
                            for k, v in img_pred_scores.items()
                        }
                        self.logs_update(img_pred_scores)

                if visual_reconstructed is not None:
                    for t in range(obs["visual"].shape[1]):
                        img_reconstruction_scores = eval_images(
                            visual_reconstructed[:, t], obs["visual"][:, t]
                        )
                        img_reconstruction_scores = self.accelerator.gather_for_metrics(
                            img_reconstruction_scores
                        )
                        img_reconstruction_scores = {
                            f"train_img_{k}_reconstructed": [v.mean().item()]
                            for k, v in img_reconstruction_scores.items()
                        }
                        self.logs_update(img_reconstruction_scores)

                self.plot_samples(
                    obs["visual"],
                    visual_out,
                    visual_reconstructed,
                    self.epoch,
                    batch=i,
                    num_samples=self.num_reconstruct_samples,
                    phase="train",
                )

            loss_components = {f"train_{k}": [v] for k, v in loss_components.items()}
            self.logs_update(loss_components)

            # The optimizer steps for this batch are done, so the step counts.
            self.global_iter += 1
            self._iters_this_process += 1
            self._write_telemetry(iter_components, batch_index=i)

            if 0 < self.max_iterations <= self.global_iter:
                self._stop_requested = True
                self._write_telemetry(iter_components, batch_index=i, force=True)
                self.logs_flash_iter(iteration=i)
                self.save_ckpt()
                log.info(
                    "Iteration cap reached: global_iter=%s == training.max_iterations=%s; "
                    "stopping mid-epoch (epoch %s, batch %s). Validation skipped: the "
                    "epoch is incomplete.",
                    self.global_iter,
                    self.max_iterations,
                    self.epoch,
                    i,
                )
                break

            # Unchanged, deliberately: `i % N == 0` also fires at i == 0, so a
            # checkpoint exists within seconds and an empty checkpoint directory a
            # minute into a run is diagnosable as a crash (Requirement 6.9).
            if (
                self.cfg.training.save_every_x_iterations > 0
                and i % self.cfg.training.save_every_x_iterations == 0
            ):
                self.logs_flash_iter(iteration=i)
                self.save_ckpt()

    @torch.no_grad()
    def val(self):
        decoder_active = self.decoder_training_active()
        self.model.train_decoder = decoder_active
        self.model.eval()
        if len(self.train_traj_dset) > 0 and self.cfg.has_predictor:
            train_rollout_logs = self.openloop_rollout(
                self.train_traj_dset, mode="train"
            )
            train_rollout_logs = {
                f"train_{k}": [v] for k, v in train_rollout_logs.items()
            }
            self.logs_update(train_rollout_logs)
            val_rollout_logs = self.openloop_rollout(self.val_traj_dset, mode="val")
            val_rollout_logs = {
                f"val_{k}": [v] for k, v in val_rollout_logs.items()
            }
            self.logs_update(val_rollout_logs)

        self.accelerator.wait_for_everyone()
        for i, data in enumerate(
            tqdm(self.dataloaders["valid"], desc=f"Epoch {self.epoch} Valid")
        ):
            obs, act, state = data
            plot = i == 0
            self.model.eval()
            z_out, visual_out, visual_reconstructed, loss, loss_components = self.model(
                obs, act
            )

            loss = self.accelerator.gather_for_metrics(loss).mean()

            loss_components = self.accelerator.gather_for_metrics(loss_components)
            loss_components = {
                key: value.mean().item() for key, value in loss_components.items()
            }

            if decoder_active and plot:
                # only eval images when plotting due to speed
                if self.cfg.has_predictor:
                    z_obs_out, z_act_out = self.model.separate_emb(z_out)
                    z_gt = self.model.encode_obs(obs)
                    z_tgt = slice_trajdict_with_t(z_gt, start_idx=self.model.num_pred)

                    state_tgt = state[:, -self.model.num_hist :]  # (b, num_hist, dim)
                    err_logs = self.err_eval(z_obs_out, z_tgt)

                    err_logs = self.accelerator.gather_for_metrics(err_logs)
                    err_logs = {
                        key: value.mean().item() for key, value in err_logs.items()
                    }
                    err_logs = {f"val_{k}": [v] for k, v in err_logs.items()}

                    self.logs_update(err_logs)

                if visual_out is not None:
                    for t in range(
                        self.cfg.num_hist, self.cfg.num_hist + self.cfg.num_pred
                    ):
                        img_pred_scores = eval_images(
                            visual_out[:, t - self.cfg.num_pred], obs["visual"][:, t]
                        )
                        img_pred_scores = self.accelerator.gather_for_metrics(
                            img_pred_scores
                        )
                        img_pred_scores = {
                            f"val_img_{k}_pred": [v.mean().item()]
                            for k, v in img_pred_scores.items()
                        }
                        self.logs_update(img_pred_scores)

                if visual_reconstructed is not None:
                    for t in range(obs["visual"].shape[1]):
                        img_reconstruction_scores = eval_images(
                            visual_reconstructed[:, t], obs["visual"][:, t]
                        )
                        img_reconstruction_scores = self.accelerator.gather_for_metrics(
                            img_reconstruction_scores
                        )
                        img_reconstruction_scores = {
                            f"val_img_{k}_reconstructed": [v.mean().item()]
                            for k, v in img_reconstruction_scores.items()
                        }
                        self.logs_update(img_reconstruction_scores)

                self.plot_samples(
                    obs["visual"],
                    visual_out,
                    visual_reconstructed,
                    self.epoch,
                    batch=i,
                    num_samples=self.num_reconstruct_samples,
                    phase="valid",
                )
            loss_components = {f"val_{k}": [v] for k, v in loss_components.items()}
            self.logs_update(loss_components)

    def openloop_rollout(
        self, dset, num_rollout=10, rand_start_end=True, min_horizon=2, mode="train"
    ):
        np.random.seed(self.cfg.training.seed)
        min_horizon = min_horizon + self.cfg.num_hist
        plotting_dir = f"rollout_plots/e{self.epoch}_rollout"
        if self.accelerator.is_main_process:
            os.makedirs(plotting_dir, exist_ok=True)
        self.accelerator.wait_for_everyone()
        logs = {}

        # rollout with both num_hist and 1 frame as context
        num_past = [(self.cfg.num_hist, ""), (1, "_1framestart")]

        # sample traj
        for idx in range(num_rollout):
            valid_traj = False
            while not valid_traj:
                traj_idx = np.random.randint(0, len(dset))
                obs, act, state, _ = dset[traj_idx]
                act = act.to(self.device)
                if rand_start_end:
                    if obs["visual"].shape[0] > min_horizon * self.cfg.frameskip + 1:
                        start = np.random.randint(
                            0,
                            obs["visual"].shape[0] - min_horizon * self.cfg.frameskip - 1,
                        )
                    else:
                        start = 0
                    max_horizon = (obs["visual"].shape[0] - start - 1) // self.cfg.frameskip
                    if max_horizon > min_horizon:
                        valid_traj = True
                        horizon = np.random.randint(min_horizon, max_horizon + 1)
                else:
                    valid_traj = True
                    start = 0
                    horizon = (obs["visual"].shape[0] - 1) // self.cfg.frameskip

            for k in obs.keys():
                obs[k] = obs[k][
                    start : 
                    start + horizon * self.cfg.frameskip + 1 : 
                    self.cfg.frameskip
                ]
            act = act[start : start + horizon * self.cfg.frameskip]
            act = rearrange(act, "(h f) d -> h (f d)", f=self.cfg.frameskip)

            obs_g = {}
            for k in obs.keys():
                obs_g[k] = obs[k][-1].unsqueeze(0).unsqueeze(0).to(self.device)
            z_g = self.model.encode_obs(obs_g)
            actions = act.unsqueeze(0)

            for past in num_past:
                n_past, postfix = past

                obs_0 = {}
                for k in obs.keys():
                    obs_0[k] = (
                        obs[k][:n_past].unsqueeze(0).to(self.device)
                    )  # unsqueeze for batch, (b, t, c, h, w)

                z_obses, z = self.model.rollout(obs_0, actions)
                z_obs_last = slice_trajdict_with_t(z_obses, start_idx=-1, end_idx=None)
                div_loss = self.err_eval_single(z_obs_last, z_g)

                for k in div_loss.keys():
                    log_key = f"z_{k}_err_rollout{postfix}"
                    if log_key in logs:
                        logs[f"z_{k}_err_rollout{postfix}"].append(
                            div_loss[k]
                        )
                    else:
                        logs[f"z_{k}_err_rollout{postfix}"] = [
                            div_loss[k]
                        ]

                if self.cfg.has_decoder:
                    visuals = self.model.decode_obs(z_obses)[0]["visual"]
                    imgs = torch.cat([obs["visual"], visuals[0].cpu()], dim=0)
                    self.plot_imgs(
                        imgs,
                        obs["visual"].shape[0],
                        f"{plotting_dir}/e{self.epoch}_{mode}_{idx}{postfix}.png",
                    )
        logs = {
            key: sum(values) / len(values) for key, values in logs.items() if values
        }
        return logs

    def logs_update(self, logs):
        for key, value in logs.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().item()
            length = len(value)
            count, total = self.epoch_log.get(key, (0, 0.0))
            self.epoch_log[key] = (
                count + length,
                total + sum(value),
            )

    def logs_flash(self, step):
        epoch_log = OrderedDict()
        for key, value in self.epoch_log.items():
            count, sum = value
            to_log = sum / count
            epoch_log[key] = to_log
        epoch_log["epoch"] = step
        log.info(f"Epoch {self.epoch}  Training loss: {epoch_log['train_loss']:.4f}  \
                Validation loss: {epoch_log['val_loss']:.4f}")

        if self.accelerator.is_main_process:
            self.wandb_run.log(epoch_log)
        self.epoch_log = OrderedDict()

    def logs_flash_iter(self, iteration):
        iter_log = OrderedDict()
        for key, value in self.epoch_log.items():
            count, sum = value
            to_log = sum / count
            iter_log[key] = to_log
        iter_log["iter"] = iteration
        iter_log["epoch"] = self.epoch

        if self.accelerator.is_main_process:
            self.wandb_run.log(iter_log)

    def plot_samples(
        self,
        gt_imgs,
        pred_imgs,
        reconstructed_gt_imgs,
        epoch,
        batch,
        num_samples=2,
        phase="train",
    ):
        """
        input:  gt_imgs, reconstructed_gt_imgs: (b, num_hist + num_pred, 3, img_size, img_size)
                pred_imgs: (b, num_hist, 3, img_size, img_size)
        output:   imgs: (b, num_frames, 3, img_size, img_size)
        """
        num_frames = gt_imgs.shape[1]
        # sample num_samples images
        gt_imgs, pred_imgs, reconstructed_gt_imgs = sample_tensors(
            [gt_imgs, pred_imgs, reconstructed_gt_imgs],
            num_samples,
            indices=list(range(num_samples))[: gt_imgs.shape[0]],
        )

        num_samples = min(num_samples, gt_imgs.shape[0])

        # fill in blank images for frameskips
        if pred_imgs is not None:
            pred_imgs = torch.cat(
                (
                    torch.full(
                        (num_samples, self.model.num_pred, *pred_imgs.shape[2:]),
                        -1,
                        device=self.device,
                    ),
                    pred_imgs,
                ),
                dim=1,
            )
        else:
            pred_imgs = torch.full(gt_imgs.shape, -1, device=self.device)

        pred_imgs = rearrange(pred_imgs, "b t c h w -> (b t) c h w")
        gt_imgs = rearrange(gt_imgs, "b t c h w -> (b t) c h w")
        reconstructed_gt_imgs = rearrange(
            reconstructed_gt_imgs, "b t c h w -> (b t) c h w"
        )
        imgs = torch.cat([gt_imgs, pred_imgs, reconstructed_gt_imgs], dim=0)

        if self.accelerator.is_main_process:
            os.makedirs(phase, exist_ok=True)
        self.accelerator.wait_for_everyone()

        self.plot_imgs(
            imgs,
            num_columns=num_samples * num_frames,
            img_name=f"{phase}/{phase}_e{str(epoch).zfill(5)}_b{batch}.png",
        )

    def plot_imgs(self, imgs, num_columns, img_name):
        utils.save_image(
            imgs,
            img_name,
            nrow=num_columns,
            normalize=True,
            value_range=(-1, 1),
        )


@hydra.main(config_path="conf", config_name="train")
def main(cfg: OmegaConf):
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
