import math

import torch
import torch.nn as nn
import torch.nn.functional as F
# Explicit: `import torch` does not reliably bind the `torch.utils.checkpoint` submodule,
# and the CCR rollout depends on it (see _predict_maybe_checkpointed).
import torch.utils.checkpoint
import logging
from torchvision import transforms
from einops import rearrange, repeat

from models.vit import sdpa_attention

log = logging.getLogger(__name__)

# Permitted values for `training.ccr_action_source`.
#   'logged'    -- perturb recorded normalized actions only; the training window caps L
#   'synthetic' -- keep and perturb the recorded prefix, synthesize actions past the edge
CCR_ACTION_SOURCES = ("logged", "synthetic")

# Permitted values for `training.acs_action_reduce` -- how the `f` env actions of one
# latent step are reduced to the single action vector the gate compares.
#   'sum'   -- net commanded displacement over the substeps (the pre-registered default)
#   'raw'   -- the concatenated block itself, no reduction
#   'first' -- the first substep only
ACS_ACTION_REDUCTIONS = ("sum", "raw", "first")

# Permitted values for `training.acs_gate` -- how consecutive-action similarity becomes a
# per-triple weight. Closed enum, not a continuous sharpness constant, so there is nothing
# to calibrate ('permuted' is the null-control arm).
ACS_GATES = ("relu_cos", "affine_cos", "hard", "permuted")

# Accepted forms of `training.straighten`, quoted in the parser's error message.
STRAIGHTEN_FORMS = ("False", "cos<scale>", "aggcos<scale>", "acsaggcos<scale>")


class VWorldModel(nn.Module):
    def __init__(
        self,
        image_size,  # 224
        num_hist,
        num_pred,
        encoder,
        proprio_encoder,
        action_encoder,
        decoder,
        predictor,
        proprio_dim=0,
        action_dim=0,
        concat_dim=0,
        num_action_repeat=7,
        num_proprio_repeat=7,
        train_encoder=True,
        train_predictor=False,
        train_decoder=True,
        straighten=False,
        stop_grad=True,
        vcreg=False,
        vcreg_std_coeff=0,
        vcreg_cov_coeff=0,
        vcreg_apply_to="enc",
        lambda_cf=0.0,
        ccr_rho=0.0,
        ccr_rollout_len=5,
        ccr_grad_checkpoint=False,
        ccr_fast_attention=True,
        ccr_action_source="synthetic",
        mca_weight=0.0,
        acs_action_reduce="sum",
        acs_gate="relu_cos",
        **kwargs,
    ):
        super().__init__()
        self.num_hist = num_hist
        self.num_pred = num_pred
        self.encoder = encoder
        self.proprio_encoder = proprio_encoder
        self.action_encoder = action_encoder
        self.decoder = decoder  # decoder could be None
        self.predictor = predictor  # predictor could be None
        self.train_encoder = train_encoder
        self.train_predictor = train_predictor
        self.train_decoder = train_decoder
        self.num_action_repeat = num_action_repeat
        self.num_proprio_repeat = num_proprio_repeat
        self.proprio_dim = proprio_dim * num_proprio_repeat 
        self.action_dim = action_dim * num_action_repeat 
        self.emb_dim = self.encoder.emb_dim + (self.action_dim + self.proprio_dim) * (concat_dim) # Not used
        self.straighten = False
        self.straighten_scale = 0.0
        self.curvature_mode = None
        self.stop_grad = bool(stop_grad)
        self.vcreg = bool(vcreg)
        self.std_coeff = float(vcreg_std_coeff)
        self.cov_coeff = float(vcreg_cov_coeff)
        if vcreg_apply_to != "enc":
            raise ValueError(
                f"Only encoder VCReg is supported, got vcreg_apply_to='{vcreg_apply_to}'."
            )

        # `training.straighten` parser. `False`, `None` and the empty string mean "off",
        # exactly as before; every other non-empty string either selects a known mode or
        # raises here, before the first training step.
        #
        # Prefix order is specificity order: 'acsaggcos' is tested before 'aggcos' and
        # 'aggcos' before 'cos'. `"acsaggcos1e-1".startswith("aggcos")` happens to be
        # False, so the branches cannot actually shadow one another, but the order is the
        # one a reader has to be able to trust when a mode string is added later.
        if isinstance(straighten, str) and straighten != "":
            if straighten.startswith("acsaggcos"):
                suffix = straighten.replace("acsaggcos", "", 1)
                mode = "acsaggcos"
            elif straighten.startswith("aggcos"):
                suffix = straighten.replace("aggcos", "", 1)
                mode = "aggcos"
            elif straighten.startswith("cos"):
                suffix = straighten.replace("cos", "", 1)
                mode = "cos"
            else:
                # Before this branch existed an unrecognized string fell straight through
                # to `curvature_mode = None`, so a typo like 'acsagcos1e-1' trained a full
                # run with no curvature term at all while logging "Straightening disabled"
                # in a wall of startup lines. Raising is a bug fix on a path that was
                # already broken: no shipped config uses an unrecognized string.
                raise ValueError(
                    f"training.straighten={straighten!r} matches no known curvature mode; "
                    f"expected one of {', '.join(STRAIGHTEN_FORMS)} "
                    f"(e.g. False, 'cos1e-1', 'aggcos1e-1', 'acsaggcos1e-1')."
                )
            try:
                scale = float(suffix) if suffix else 1.0
            except ValueError:
                raise ValueError(
                    f"training.straighten={straighten!r} has a non-numeric scale suffix "
                    f"{suffix!r}; expected one of {', '.join(STRAIGHTEN_FORMS)} "
                    f"(e.g. '{mode}1e-1')."
                ) from None
            # `float()` accepts 'nan', 'inf' and '-inf', so this check has to come *before*
            # the sign check: `nan <= 0` is False, so a non-finite scale slips past it and
            # then `self.straighten = ... and self.straighten_scale > 0` evaluates
            # `nan > 0` as False. That reproduces F4 exactly -- `curvature_mode ==
            # "cos"` while the run logs "Straightening disabled" and trains with no
            # curvature term -- which is the hole the `else: raise` above was added to
            # close. `cosinf` is the other half: it parses, enables the term, and makes
            # the loss infinite on the first step.
            if not math.isfinite(scale):
                raise ValueError(
                    f"training.straighten={straighten!r} parses to a non-finite curvature "
                    f"scale of {scale} (float() accepts 'nan', 'inf' and '-inf'); a "
                    "non-finite scale either silently disables the term while naming a "
                    "curvature mode or makes the loss non-finite. Use "
                    "training.straighten=False to disable straightening, or one of "
                    f"{', '.join(STRAIGHTEN_FORMS)} with a finite positive scale "
                    f"(e.g. '{mode}1e-1')."
                )
            if scale <= 0:
                raise ValueError(
                    f"training.straighten={straighten!r} parses to a curvature scale of "
                    f"{scale}, which disables the term while naming it; use "
                    "training.straighten=False to disable straightening, or a positive "
                    f"scale (e.g. '{mode}1e-1')."
                )
            self.straighten_scale = scale
            self.curvature_mode = mode

        self.straighten = self.curvature_mode is not None and self.straighten_scale > 0

        # --- Counterfactual Curvature Regularization / Metric-Consistent Aggregation ---
        # These are plain Python scalars and booleans. Nothing in the CCR/MCA path may
        # construct an nn.Module, parameter or buffer: VWorldModel is built *after*
        # accelerator.prepare() in train.py and is never itself prepared, so any module
        # created here would keep CPU parameters, never be registered in an optimizer, and
        # kill the run about two seconds into epoch 1 with a device mismatch
        # (SHORT_BUDGET_PILOTS.md section 9).
        #
        # train.py forwards these with `self.cfg.training.get(key)` and no fallback, so a
        # yaml key that is absent arrives here as None. None means "use the default".
        self.lambda_cf = float(0.0 if lambda_cf is None else lambda_cf)
        self.ccr_rho = float(0.0 if ccr_rho is None else ccr_rho)
        self.ccr_rollout_len = int(5 if ccr_rollout_len is None else ccr_rollout_len)
        # Two independent ways to make the CCR rollout affordable, both confined to the CCR
        # path and both numerically neutral to bf16 rounding, so they are deliberately in
        # neither ccr_tag nor LOSS_SIGNATURE_KEYS -- toggling either must not rename a run
        # or block a resume.
        #
        #   ccr_fast_attention   scaled_dot_product_attention instead of a materialised
        #                        (b, heads, 588, 588) score matrix. FASTER AND LIGHTER, so
        #                        it is the default and the preferred lever.
        #   ccr_grad_checkpoint  recompute the rollout in backward instead of storing it.
        #                        Trades ~33% more compute for most of the memory. Only
        #                        needed when fast attention is off; default False.
        self.ccr_grad_checkpoint = bool(
            False if ccr_grad_checkpoint is None else ccr_grad_checkpoint
        )
        self.ccr_fast_attention = bool(
            True if ccr_fast_attention is None else ccr_fast_attention
        )
        self.ccr_action_source = str(
            "synthetic" if ccr_action_source is None else ccr_action_source
        )
        self.mca_weight = float(0.0 if mca_weight is None else mca_weight)

        # --- Action-Conditioned Straightening (selected by straighten=acsaggcos<scale>) ---
        # Plain Python strings, for the same reason as the CCR knobs above: no module, no
        # parameter, no buffer may be created here. Both are closed enums with
        # pre-registered defaults rather than continuous constants, so there is nothing to
        # calibrate. train.py forwards them with `self.cfg.training.get(key)`, so an absent
        # yaml key arrives as None and means "use the default".
        self.acs_action_reduce = str(
            "sum" if acs_action_reduce is None else acs_action_reduce
        )
        self.acs_gate = str("relu_cos" if acs_gate is None else acs_gate)

        for _name, _value in (
            ("lambda_cf", self.lambda_cf),
            ("ccr_rho", self.ccr_rho),
            ("mca_weight", self.mca_weight),
        ):
            if _value < 0:
                raise ValueError(f"training.{_name} must be >= 0, got {_value}.")
        # Validated eagerly, even when lambda_cf == 0: a typo in an unused knob that only
        # surfaces once the term is enabled is exactly the class of mistake a pilot cannot
        # afford. This is a string comparison, so it adds no tensor work to the off path.
        if self.ccr_action_source not in CCR_ACTION_SOURCES:
            raise ValueError(
                f"training.ccr_action_source must be one of {CCR_ACTION_SOURCES}, "
                f"got {self.ccr_action_source!r}."
            )
        # Same precedent, same reason, and deliberately unconditional: these are validated
        # even on a plain `aggcos1e-1` baseline run, so a typo in a knob this run does not
        # read cannot survive until the run that enables it. String comparisons only, so
        # the off path gains no tensor work.
        if self.acs_action_reduce not in ACS_ACTION_REDUCTIONS:
            raise ValueError(
                f"training.acs_action_reduce must be one of {ACS_ACTION_REDUCTIONS}, "
                f"got {self.acs_action_reduce!r}."
            )
        if self.acs_gate not in ACS_GATES:
            raise ValueError(
                f"training.acs_gate must be one of {ACS_GATES}, "
                f"got {self.acs_gate!r}."
            )
        # L + 2 is the curvature window, and total_curvature needs at least 3 frames.
        if self.ccr_rollout_len < 1:
            raise ValueError(
                f"training.ccr_rollout_len must be >= 1, got {self.ccr_rollout_len} "
                "(the CCR curvature window is ccr_rollout_len + 2 frames and "
                "total_curvature requires at least 3)."
            )
        # Cheap boolean gates: the disabled path is one attribute lookup and one comparison.
        self.ccr = self.lambda_cf > 0
        self.mca = self.mca_weight > 0
        # Plain bool, not a buffer: `synthesized_action_frames` needs num_frames, which is
        # not available in __init__ (it is a property of the batch, not of the model), so
        # the startup line below reports the formula and the first CCR forward reports the
        # resolved count.
        self._ccr_forward_logged = False

        log.info("num_action_repeat: %s", self.num_action_repeat)
        log.info("num_proprio_repeat: %s", self.num_proprio_repeat)
        log.info("proprio encoder: %s", proprio_encoder)
        log.info("action encoder: %s", action_encoder)
        log.info("proprio_dim: %s, after repeat: %s", proprio_dim, self.proprio_dim)
        log.info("action_dim: %s, after repeat: %s", action_dim, self.action_dim)
        log.info("emb_dim: %s", self.emb_dim)
        if self.straighten:
            log.info(
                "Straightening enabled: mode=%s, scale=%s",
                self.curvature_mode,
                self.straighten_scale,
            )
        else:
            log.info("Straightening disabled")
        if self.curvature_mode == "acsaggcos":
            log.info(
                "ACS gate config: action_reduce=%s, gate=%s",
                self.acs_action_reduce,
                self.acs_gate,
            )
        log.info("Stop-grad enabled: %s", self.stop_grad)
        log.info(
            "VCReg enabled: %s, apply_to=enc, std_coeff=%s, cov_coeff=%s",
            self.vcreg,
            self.std_coeff,
            self.cov_coeff,
        )
        # Emitted here, after accelerate has already placed the submodules, so the device
        # reported is the device the terms will actually compute on. This is the two-minute
        # smoke check from the pilot checklist.
        _ccr_device = self._param_device()
        if self.ccr:
            log.info(
                "CCR enabled: term=ccr, weight(lambda_cf)=%s, rho=%s, rollout_len=%s, "
                "action_source=%s, synthesized_action_frames=max(0, %s + %s - 1 - "
                "num_frames) [resolved on the first forward], curvature_mode=aggcos, "
                "fast_attention=%s, grad_checkpoint=%s, device=%s",
                self.lambda_cf,
                self.ccr_rho,
                self.ccr_rollout_len,
                self.ccr_action_source,
                self.num_hist,
                self.ccr_rollout_len,
                self.ccr_fast_attention,
                self.ccr_grad_checkpoint,
                _ccr_device,
            )
            if not self.ccr_fast_attention and not self.ccr_grad_checkpoint:
                log.warning(
                    "CCR has fast_attention=False AND grad_checkpoint=False. At the PushT "
                    "target-cell shapes each of the %s predictor calls stores ~6-8 GB of "
                    "attention activations, so this configuration needs tens of GB more "
                    "than a 45 GB MIG slice has and will almost certainly raise "
                    "torch.OutOfMemoryError on the first backward. Enable one of them.",
                    self.ccr_rollout_len,
                )
        if self.mca:
            log.info(
                "MCA enabled: term=mca, weight(mca_weight)=%s, rho=n/a, device=%s",
                self.mca_weight,
                _ccr_device,
            )
        _disabled = []
        if not self.ccr:
            _disabled.append(f"CCR disabled (lambda_cf={self.lambda_cf})")
        if not self.mca:
            _disabled.append(f"MCA disabled (mca_weight={self.mca_weight})")
        if _disabled:
            log.info("; ".join(_disabled))

        self.concat_dim = concat_dim # 0 or 1
        assert concat_dim == 0 or concat_dim == 1, f"concat_dim {concat_dim} not supported."
        log.info("Model emb_dim: %s", self.emb_dim)

        if "dino" in self.encoder.name:
            decoder_scale = 16  # from vqvae
            num_side_patches = image_size // decoder_scale
            self.encoder_image_size = num_side_patches * encoder.patch_size
            self.encoder_transform = transforms.Compose(
                [transforms.Resize(self.encoder_image_size)]
            )
        else:
            # set self.encoder_transform to identity transform
            self.encoder_transform = lambda x: x

        self.decoder_criterion = nn.MSELoss()
        self.decoder_latent_loss_weight = 0.25
        self.emb_criterion = nn.MSELoss()

    def _param_device(self):
        """Device the model's tensors live on, i.e. `next(self.parameters()).device`.

        Falls back to buffers and then CPU so that a parameterless test double cannot
        turn a log line into a StopIteration.
        """
        param = next(self.parameters(), None)
        if param is not None:
            return param.device
        buf = next(self.buffers(), None)
        if buf is not None:
            return buf.device
        return torch.device("cpu")

    def train(self, mode=True):
        super().train(mode)
        if self.train_encoder:
            self.encoder.train(mode)
        if self.predictor is not None and self.train_predictor:
            self.predictor.train(mode)
        self.proprio_encoder.train(mode)
        self.action_encoder.train(mode)
        if self.decoder is not None and self.train_decoder:
            self.decoder.train(mode)

    def eval(self):
        super().eval()
        self.encoder.eval()
        if self.predictor is not None:
            self.predictor.eval()
        self.proprio_encoder.eval()
        self.action_encoder.eval()
        if self.decoder is not None:
            self.decoder.eval()

    def encode(self, obs, act): 
        """
        input :  obs (dict): "visual", "proprio", (b, num_frames, 3, img_size, img_size) 
        output:    z (tensor): (b, num_frames, num_patches, emb_dim)
        """
        z_dct = self.encode_obs(obs)
        act_emb = self.encode_act(act)
        if self.concat_dim == 0:
            z = torch.cat(
                    [z_dct['visual'], z_dct['proprio'].unsqueeze(2), act_emb.unsqueeze(2)], dim=2 # add as an extra token
                )  # (b, num_frames, num_patches + 2, dim)
        if self.concat_dim == 1:
            proprio_tiled = repeat(z_dct['proprio'].unsqueeze(2), "b t 1 a -> b t f a", f=z_dct['visual'].shape[2])
            proprio_repeated = proprio_tiled.repeat(1, 1, 1, self.num_proprio_repeat)
            act_tiled = repeat(act_emb.unsqueeze(2), "b t 1 a -> b t f a", f=z_dct['visual'].shape[2])
            act_repeated = act_tiled.repeat(1, 1, 1, self.num_action_repeat)
            z = torch.cat(
                [z_dct['visual'], proprio_repeated, act_repeated], dim=3
            )  # (b, num_frames, num_patches, dim + action_dim)
        return z
    
    def encode_act(self, act):
        act = self.action_encoder(act) # (b, num_frames, action_emb_dim)
        return act
    
    def encode_proprio(self, proprio):
        proprio = self.proprio_encoder(proprio)
        return proprio

    def encode_obs(self, obs):
        """
        input : obs (dict): "visual", "proprio" (b, t, 3, img_size, img_size)
        output:   z (dict): "visual", "proprio" (b, t, num_patches, encoder_emb_dim)
        """
        visual = obs['visual']
        b = visual.shape[0]
        visual = rearrange(visual, "b t ... -> (b t) ...")
        visual = self.encoder_transform(visual)
        visual_embs = self.encoder.forward(visual)
        visual_embs = rearrange(visual_embs, "(b t) p d -> b t p d", b=b)

        proprio = obs['proprio']
        proprio_emb = self.encode_proprio(proprio)
        return {"visual": visual_embs, "proprio": proprio_emb}

    def predict(self, z):  # in embedding space
        """
        input : z: (b, num_hist, num_patches, emb_dim)
        output: z: (b, num_hist, num_patches, emb_dim)
        """
        T = z.shape[1]
        # reshape to a batch of windows of inputs
        z = rearrange(z, "b t p d -> b (t p) d")
        # (b, num_hist * num_patches per img, emb_dim)
        z = self.predictor(z)
        z = rearrange(z, "b (t p) d -> b t p d", t=T)
        return z

    def decode(self, z):
        """
        input :   z: (b, num_frames, num_patches, emb_dim)
        output: obs: (b, num_frames, 3, img_size, img_size)
        """
        z_obs, z_act = self.separate_emb(z)
        obs, diff = self.decode_obs(z_obs)
        return obs, diff

    def decode_obs(self, z_obs):
        """
        input :   z: (b, num_frames, num_patches, emb_dim)
        output: obs: (b, num_frames, 3, img_size, img_size)
        """
        b, num_frames, num_patches, emb_dim = z_obs["visual"].shape
        visual, diff = self.decoder(z_obs["visual"])  # (b*num_frames, 3, 224, 224)
        visual = rearrange(visual, "(b t) c h w -> b t c h w", t=num_frames)
        obs = {
            "visual": visual,
            "proprio": z_obs["proprio"], # Note: no decoder for proprio for now!
        }
        return obs, diff
    
    def separate_emb(self, z):
        """
        input: z (tensor)
        output: z_obs (dict), z_act (tensor)
        """
        if self.concat_dim == 0:
            z_visual, z_proprio, z_act = z[:, :, :-2, :], z[:, :, -2, :], z[:, :, -1, :]
        elif self.concat_dim == 1:
            z_visual, z_proprio, z_act = z[..., :-(self.proprio_dim + self.action_dim)], \
                                         z[..., -(self.proprio_dim + self.action_dim) :-self.action_dim],  \
                                         z[..., -self.action_dim:]
            # remove tiled dimensions
            z_proprio = z_proprio[:, :, 0, : self.proprio_dim // self.num_proprio_repeat]
            z_act = z_act[:, :, 0, : self.action_dim // self.num_action_repeat]
        z_obs = {"visual": z_visual, "proprio": z_proprio}
        return z_obs, z_act

    def visual_only(self, z):
        if self.concat_dim == 0:
            return z[:, :, :-2, :]
        drop = self.proprio_dim + self.action_dim
        return z[..., :-drop] if drop > 0 else z

    def visual_prop(self, z):
        if self.concat_dim == 0:
            return z[:, :, :-1, :]
        return z[..., :-self.action_dim]

    def vcreg_std_loss(self, z: torch.Tensor) -> torch.Tensor:
        x = z.reshape(-1, z.shape[-1])
        std_x = torch.sqrt(x.var(dim=0) + 1e-4)
        return torch.mean(F.relu(1 - std_x))

    def vcreg_cov_loss(self, z: torch.Tensor) -> torch.Tensor:
        x = z.reshape(-1, z.shape[-1])
        _, d = x.shape
        x = x - x.mean(dim=0)
        cov_x = (x.T @ x) / (x.shape[0] - 1)
        cov_loss = self.off_diagonal(cov_x).pow_(2).sum() / d
        return cov_loss

    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def _agg_velocities(self, features):
        """Consecutive latent velocities in the paper's aggregated space.

        `features` is `(b, t, p, d)` -- what `visual_only(z)` returns. The patch
        tokens of every frame are pooled by `encoder.agg` into the aggregated
        space, and the two consecutive velocity fields are differenced out of it.
        Returns `(v1, v2)`, each `(b, t - 2, agg_dim)`.

        This is the *single* implementation of the aggregated velocities: both
        `total_curvature(mode="aggcos")` and the action-conditioned term read it,
        so the two cannot drift apart. The operations, their order and their
        dtypes are the pre-refactor `aggcos` branch's, verbatim, which is what
        keeps the baseline path bitwise identical.
        """
        if not hasattr(self.encoder, "agg"):
            raise ValueError("curvature mode 'aggcos' requires encoder.agg().")
        b, t, p, d = features.shape
        tokens = features.reshape(b * t, p, d)
        z = self.encoder.agg(tokens).reshape(b, t, -1)
        v1 = z[:, 1:-1] - z[:, :-2]
        v2 = z[:, 2:] - z[:, 1:-1]
        return v1, v2

    def _cos_curvature_terms(self, v1, v2, eps=1e-6, step_thresh=1e-6):
        """Per-triple curvature `1 - cos(v1, v2)` and the static-velocity mask.

        Returns `(loss, mask)`, both `(b, t - 2)`. `loss[mask].mean()` is exactly
        the pre-refactor `_cos_curvature` return value, which is why the split is
        bitwise neutral; a gated term forms a weighted mean over the same two
        tensors instead of averaging them uniformly.

        `step_thresh <= 0` disables the mask, and the all-true mask returned in
        that case reproduces the old unmasked `loss.mean()`.
        """
        cos = F.cosine_similarity(v1, v2, dim=-1, eps=eps)
        loss = 1.0 - cos
        if step_thresh > 0:
            step1 = v1.norm(dim=-1)
            step2 = v2.norm(dim=-1)
            mask = (step1 > step_thresh) & (step2 > step_thresh)
        else:
            mask = torch.ones_like(loss, dtype=torch.bool)
        return loss, mask

    def _cos_curvature(self, v1, v2, eps=1e-6, step_thresh=1e-6):
        loss, mask = self._cos_curvature_terms(v1, v2, eps=eps, step_thresh=step_thresh)
        return loss[mask].mean()

    def total_curvature(self, features, mode="cos"):
        if features.shape[1] < 3:
            raise ValueError(f"Features must have at least 3 frames for curvature calculation, got {features.shape[1]}")

        if mode == "aggcos":
            v1, v2 = self._agg_velocities(features)
        elif mode == "cos":
            v1 = features[:, 1:-1] - features[:, :-2]
            v2 = features[:, 2:] - features[:, 1:-1]
        else:
            raise ValueError(f"Unknown curvature mode '{mode}'. Use 'cos' or 'aggcos'.")

        return self._cos_curvature(v1, v2)

    def _resolve_env_action_dim(self, act, env_action_dim=None):
        """The environment action dimension `d` for *this* batch, and the substep count.

        `act` is `(b, t, f * d)`: `datasets/traj_dset.py` packs the `f = frameskip`
        env actions of one latent step with `rearrange(act, "(n f) d -> n (f d)")`,
        so `act.shape[-1]` alone cannot tell `f * d` apart from `d * f`. `d` is a
        protocol value of the batch (`dset.dataset.action_dim`, 2 on PushT), so it
        is supplied by the caller -- explicitly, or through the optional
        `acs_env_action_dim` attribute a caller may set once from the dataset --
        and **never** read from a config constant. Guessing it would corrupt every
        gate value while still returning a plausibly-shaped tensor.

        Returns `(f, d)`.
        """
        if env_action_dim is None:
            env_action_dim = getattr(self, "acs_env_action_dim", None)
        if env_action_dim is None:
            raise ValueError(
                "reduce_action needs the environment action dimension to split "
                f"act.shape[-1]={act.shape[-1]} into substeps for "
                f"acs_action_reduce={self.acs_action_reduce!r}; pass "
                "env_action_dim=<dset.dataset.action_dim> (2 on PushT). It is a "
                "protocol value of the batch and must not be inferred from a config "
                "constant."
            )
        d = int(env_action_dim)
        total = int(act.shape[-1])
        if d <= 0:
            raise ValueError(
                f"env_action_dim must be positive, got {d} (act.shape[-1]={total})."
            )
        if total % d != 0:
            # E5. Reshaping silently here would mix dimension j of one substep with
            # dimension j+1 of the next, so every cos(a_t, a_{t+1}) -- and therefore
            # every gate weight -- would be wrong while nothing looked wrong.
            raise ValueError(
                f"act.shape[-1]={total} is not divisible by the environment action "
                f"dim {d}, so the {total} channels do not split into whole substeps "
                f"for acs_action_reduce={self.acs_action_reduce!r}; this is a "
                "frameskip / action_dim mismatch between the batch and the caller."
            )
        return total // d, d

    def reduce_action(self, act, env_action_dim=None):
        """Reduce each latent step's `f` env actions to the one vector the gate compares.

        `act` is `(b, t, f * d)` with channel `s * d + j` holding dimension `j` of
        substep `s` (`traj_dset.py`'s `rearrange("(n f) d -> n (f d)")`, F1), so
        `act[:, t]` is exactly the control that produces `v_t = z_{t+1} - z_t`.

        Per `self.acs_action_reduce`:

        - ``'sum'``   -> `(b, t, d)`, `out[..., j] = sum_s act[..., s * d + j]`: the
          net commanded displacement over the latent step, which is the action
          variable whose direction change the gate is a hypothesis about. A new
          tensor.
        - ``'first'`` -> `act[..., :d]`, the first substep only. A **view** of `act`,
          as `visual_only` also returns; the gate only reads it.
        - ``'raw'``   -> `act` **itself**, an identity. Documented, so no caller
          assumes a copy or a fresh tensor to write into.

        Neither ``'sum'`` nor ``'first'`` mutates `act`.

        `mean` is deliberately not offered: `mean = sum / f` is a single positive
        scalar applied to both vectors of the cosine and `cos(alpha u, alpha v) ==
        cos(u, v)` for `alpha > 0`, so it is the *same gate* as ``'sum'`` (P5).

        `env_action_dim` resolves the substep count together with `act.shape[-1]`;
        a non-divisible pair raises rather than reshaping silently (E5).
        """
        if self.acs_action_reduce == "raw":
            # Identity on purpose: the 10-d profile cosine is a different (measured
            # at Stage 0, not selected) question, and returning `act` keeps that
            # arm free of a copy.
            return act
        f, d = self._resolve_env_action_dim(act, env_action_dim)
        if self.acs_action_reduce == "first":
            return act[..., :d]
        if self.acs_action_reduce == "sum":
            # unflatten gives a view; sum allocates the (b, t, d) result, so `act`
            # is untouched. f == 1 degenerates to a copy of `act`, which is correct.
            return act.unflatten(-1, (f, d)).sum(dim=-2)
        # Unreachable: __init__ validates the enum eagerly against
        # ACS_ACTION_REDUCTIONS. Kept so a future enum member cannot fall through
        # to a silently wrong reduction -- the F4 failure mode, one level down.
        raise ValueError(
            f"acs_action_reduce must be one of {ACS_ACTION_REDUCTIONS}, "
            f"got {self.acs_action_reduce!r}."
        )

    def _permute_gate(self, w, mask=None):
        """Permute `w` across the batch's unmasked triples ('permuted' null control).

        The attribution arm (Requirement 13.4) needs a gate that keeps *everything*
        about the weight population and destroys *only* which triple each weight
        lands on. So the unmasked entries are gathered, shuffled among themselves
        and scattered back; masked entries keep their own values, because they are
        dropped by the same mask before the weighted mean and their positions must
        not absorb weight that belongs to a live triple.

        Consequences, which are what make the arm interpretable and are themselves
        a check on it (Requirement 13.5): the multiset of unmasked weights is
        preserved *exactly*, hence so are `mean(w)`, the quantiles and `gate_tv` --
        up to the float summation order of the scalars derived from them, which
        reordering the same addends can move by an ulp. Compare sorted values, not
        reduction outputs, when the assertion has to be bit-exact.

        `mask is None` permutes across the whole `(b, t - 2)` tensor, which is the
        same thing when nothing is masked.
        """
        if mask is not None and tuple(mask.shape) != tuple(w.shape):
            raise ValueError(
                f"action_gate mask shape {tuple(mask.shape)} must match the gate "
                f"shape {tuple(w.shape)} elementwise; the permuted arm shuffles "
                "weights across exactly the unmasked triples the weighted mean "
                "sums over."
            )
        flat = w.reshape(-1)
        if mask is None:
            index = torch.arange(flat.numel(), device=flat.device)
        else:
            index = mask.reshape(-1).nonzero(as_tuple=True)[0]
        out = flat.clone()
        if index.numel() > 1:
            perm = torch.randperm(int(index.numel()), device=flat.device)
            out[index] = flat[index][perm]
        return out.reshape(w.shape)

    def action_gate(self, act, mask=None, env_action_dim=None):
        """Per-triple action-similarity weights `w`, shape `(b, t - 2)`, in `[0, 1]`.

        `w[b, k]` gates the curvature triple `(z_k, z_{k+1}, z_{k+2})` by how
        similar the two controls that produced its velocities are:

            a     = reduce_action(act)                       # (b, t, d)
            cos_a = cosine_similarity(a[:, :-2], a[:, 1:-1]) # (b, t - 2)
            w     = gate_fn(cos_a)

        Dispatch on `self.acs_gate`, a closed four-member enum validated eagerly in
        `__init__`:

        - ``'relu_cos'``   -> `relu(cos)`. The pre-registered default: the whole
          action-reversing half-space gets exactly zero pressure, and the surviving
          mass stays graded.
        - ``'affine_cos'`` -> `(1 + cos) / 2`. The softer fallback; note it gives
          `0.5` at orthogonality and reaches `0` only at exact antiparallelism.
        - ``'hard'``       -> `1[cos > 0]`. Same support as `relu_cos`, grading
          thrown away.
        - ``'permuted'``   -> `relu(cos)` shuffled across the batch's unmasked
          triples (Requirement 13.4). The null control: same weight multiset, same
          `mean(w)`, same `gate_tv`, no correspondence to the triple it gates.

        `mask` is the static-velocity mask from `_cos_curvature_terms`, `(b, t - 2)`
        bool. It is read *only* by ``'permuted'``, to shuffle over the same set the
        weighted mean reduces over; every other gate is elementwise and ignores it.

        Two contracts the term's attributability rests on (design 4.2):

        1. `w` is computed from the **raw `act` tensor of the batch**, never from
           `self.action_encoder(act)`. `act` is data, so nothing the encoder can
           learn moves `w`; an encoded-action gate would let the trained
           `action_encoder` drive `w -> 0` on hard triples and lower total
           straightening pressure without improving any geometry -- the
           λ-reduction confound back again, adaptive and invisible.
        2. `w` is `.detach()`ed anyway, as an executable contract:
           `w.requires_grad is False` and `w.grad_fn is None` (P4). The only
           descent direction `L_acs` offers is the trajectory geometry.

        `cos` is clamped to `[-1, 1]` before the gate, so `0 <= w <= 1` holds
        elementwise (Requirement 5.4) rather than up to `cosine_similarity`'s
        float32 slop. No threshold, exponent or sharpness constant is introduced
        (Requirement 5.17).

        A zero-norm reduced action block -- a latent step commanding no net motion
        -- falls out as `w = 0` for `relu_cos` and `hard` through
        `cosine_similarity`'s own `eps`, which floors the norms and returns `0`
        instead of dividing by zero. No raise (E10); it is the same semantics as
        `step_thresh` masking near-static *latent* steps. (`affine_cos` maps that
        same `cos = 0` to `0.5` by its own definition; special-casing it would be
        the threshold constant Requirement 5.17 forbids.)

        `env_action_dim` is threaded to `reduce_action`, which needs it to split
        `act.shape[-1] = f * d` into substeps and refuses to guess.
        """
        t = int(act.shape[1])
        if t < 3:
            # E6, and the same requirement total_curvature states for `z`: a
            # curvature triple spans 3 frames, so it needs 3 action blocks.
            raise ValueError(
                f"action_gate needs at least 3 frames to form a curvature triple, "
                f"got act.shape[1]={t}."
            )
        a = self.reduce_action(act, env_action_dim=env_action_dim)
        # eps left at cosine_similarity's default on purpose: it is what turns a
        # zero-norm action block into cos = 0 rather than a division by zero (E10).
        cos_a = F.cosine_similarity(a[:, :-2], a[:, 1:-1], dim=-1)
        cos_a = cos_a.clamp(-1.0, 1.0)
        gate = self.acs_gate
        if gate in ("relu_cos", "permuted"):
            w = F.relu(cos_a)
        elif gate == "affine_cos":
            w = (1.0 + cos_a) * 0.5
        elif gate == "hard":
            w = (cos_a > 0).to(cos_a.dtype)
        else:
            # Unreachable: __init__ validates against ACS_GATES eagerly. Kept so a
            # future enum member cannot fall through to a silently wrong gate.
            raise ValueError(
                f"acs_gate must be one of {ACS_GATES}, got {self.acs_gate!r}."
            )
        w = w.detach()
        if gate == "permuted":
            w = self._permute_gate(w, mask)
        return w

    def forward(self, obs, act):
        """
        input:  obs (dict):  "visual", "proprio" (b, num_frames, 3, img_size, img_size)
                act: (b, num_frames, action_dim)
        output: z_pred: (b, num_hist, num_patches, emb_dim)
                visual_pred: (b, num_hist, 3, img_size, img_size)
                visual_reconstructed: (b, num_frames, 3, img_size, img_size)
        """
        loss = 0
        loss_components = {}
        decoder_enabled = self.decoder is not None and self.train_decoder
        z = self.encode(obs, act)
        z_src = z[:, : self.num_hist, :, :]  # (b, num_hist, num_patches, dim)
        z_tgt = z[:, self.num_pred :, :, :]  # (b, num_hist, num_patches, dim)
        visual_src = obs['visual'][:, : self.num_hist, ...]  # (b, num_hist, 3, img_size, img_size)
        visual_tgt = obs['visual'][:, self.num_pred :, ...]  # (b, num_hist, 3, img_size, img_size)

        if self.predictor is not None:
            z_pred = self.predict(z_src)
            if decoder_enabled:
                obs_pred, diff_pred = self.decode(
                    z_pred.detach()
                )  # recon loss should only affect decoder
                visual_pred = obs_pred['visual']
                recon_loss_pred = self.decoder_criterion(visual_pred, visual_tgt)
                decoder_loss_pred = (
                    recon_loss_pred + self.decoder_latent_loss_weight * diff_pred
                )
                loss_components["decoder_recon_loss_pred"] = recon_loss_pred
                loss_components["decoder_vq_loss_pred"] = diff_pred
                loss_components["decoder_loss_pred"] = decoder_loss_pred
            else:
                visual_pred = None

            # Compute loss for visual, proprio dims (i.e. exclude action dims)
            z_tgt_for_loss = z_tgt.detach() if self.stop_grad else z_tgt
            if self.concat_dim == 0:
                z_visual_loss = self.emb_criterion(z_pred[:, :, :-2, :], z_tgt_for_loss[:, :, :-2, :])
                z_proprio_loss = self.emb_criterion(z_pred[:, :, -2, :], z_tgt_for_loss[:, :, -2, :])
                z_loss = self.emb_criterion(z_pred[:, :, :-1, :], z_tgt_for_loss[:, :, :-1, :])
            elif self.concat_dim == 1:
                z_visual_loss = self.emb_criterion(
                    z_pred[:, :, :, :-(self.proprio_dim + self.action_dim)], \
                    z_tgt_for_loss[:, :, :, :-(self.proprio_dim + self.action_dim)]
                )
                z_proprio_loss = self.emb_criterion(
                    z_pred[:, :, :, -(self.proprio_dim + self.action_dim): -self.action_dim], 
                    z_tgt_for_loss[:, :, :, -(self.proprio_dim + self.action_dim): -self.action_dim]
                )
                z_loss = self.emb_criterion(
                    z_pred[:, :, :, :-self.action_dim], 
                    z_tgt_for_loss[:, :, :, :-self.action_dim]
                )

            loss = loss + z_loss
            loss_components["z_loss"] = z_loss
            loss_components["z_visual_loss"] = z_visual_loss
            loss_components["z_proprio_loss"] = z_proprio_loss

            if self.vcreg:
                z_vic_in = self.visual_prop(z)
                z_std_loss = self.vcreg_std_loss(z_vic_in)
                z_cov_loss = self.vcreg_cov_loss(z_vic_in)
                z_reg_loss = z_std_loss * self.std_coeff + z_cov_loss * self.cov_coeff
                loss_components["z_vicreg_std_loss"] = z_std_loss
                loss_components["z_vicreg_cov_loss"] = z_cov_loss
                loss_components["z_vcreg_loss_scaled"] = z_reg_loss
                loss = loss + z_reg_loss

            if self.straighten and self.straighten_scale > 0:
                feats = self.visual_only(z)
                curvature_loss = self.total_curvature(feats, mode=self.curvature_mode)
                loss = loss + curvature_loss * self.straighten_scale
                loss_components["curvature_loss_used_for_training"] = curvature_loss
                # Telemetry only: the scaled value makes the baseline curvature term
                # comparable against ccr_loss_scaled in loss shares. Adding a key does not
                # change the loss.
                loss_components["curvature_loss_scaled"] = (
                    curvature_loss * self.straighten_scale
                )

            # Both new terms sit behind a boolean, so the disabled path is one attribute
            # lookup and one comparison: no tensor work, no extra predictor rollout, no
            # extra encoder forward pass.
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
        else:
            visual_pred = None
            z_pred = None

        if decoder_enabled:
            obs_reconstructed, diff_reconstructed = self.decode(
                z.detach()
            )  # recon loss should only affect decoder
            visual_reconstructed = obs_reconstructed["visual"]
            recon_loss_reconstructed = self.decoder_criterion(visual_reconstructed, obs['visual'])
            decoder_loss_reconstructed = (
                recon_loss_reconstructed
                + self.decoder_latent_loss_weight * diff_reconstructed
            )

            loss_components["decoder_recon_loss_reconstructed"] = (
                recon_loss_reconstructed
            )
            loss_components["decoder_vq_loss_reconstructed"] = diff_reconstructed
            loss_components["decoder_loss_reconstructed"] = (
                decoder_loss_reconstructed
            )
            loss = loss + decoder_loss_reconstructed
        else:
            visual_reconstructed = None
        loss_components["loss"] = loss
        return z_pred, visual_pred, visual_reconstructed, loss, loss_components

    def replace_actions_from_z(self, z, act):
        act_emb = self.encode_act(act)
        if self.concat_dim == 0:
            z[:, :, -1, :] = act_emb
        elif self.concat_dim == 1:
            act_tiled = repeat(act_emb.unsqueeze(2), "b t 1 a -> b t f a", f=z.shape[2])
            act_repeated = act_tiled.repeat(1, 1, 1, self.num_action_repeat)
            z[..., -self.action_dim:] = act_repeated
        return z


    def _predict_maybe_checkpointed(self, z, checkpoint, fast_attention=False):
        """``predict``, optionally recomputed in backward instead of stored.

        A single ``predict`` call at the PushT target-cell shapes stores about **8 GB**
        of activations for backward, and essentially all of it is the attention matrix:
        ``models/vit.py`` materialises ``dots`` of shape
        ``(b, heads, T, T) = (32, 16, 588, 588)`` -- 177 M elements, 354 MB in bf16 --
        and then keeps the softmax output and the dropout output at the same size, for
        each of ``depth=6`` layers.  ``T = num_hist * num_patches = 3 * 196 = 588``.

        The baseline objective calls ``predict`` once.  CCR at ``L = 5`` calls it five
        more times, so it asks for roughly **40 GB of extra activation memory** and OOMs
        a 45 GB MIG slice.  The design's claim that the extra predictor calls are "a
        small fraction" of the encoder pass was about *compute time* and is simply wrong
        about *memory*.

        ``use_reentrant=False`` with the default ``preserve_rng_state=True`` saves and
        restores the RNG around the recomputation, which matters here because the
        predictor runs ``dropout=0.1``: without it the recomputed forward would draw a
        different mask and the gradient would be wrong.  With it, the result is
        numerically what the un-checkpointed path produces.

        Checkpointing is skipped when grad is disabled -- under ``torch.no_grad`` there
        is no backward to save memory for, and ``checkpoint`` would only warn.

        ``fast_attention`` routes this call through
        ``F.scaled_dot_product_attention``, which never forms the T x T score matrix and
        is both faster and lighter.  The context manager is entered *inside* the
        checkpointed callable, not around it: a checkpointed segment is recomputed during
        **backward**, long after an enclosing ``with`` block would have exited, and a
        recomputation on the other branch would silently produce activations that do not
        match the ones the forward recorded.
        """
        def _run(z_in):
            with sdpa_attention(fast_attention):
                return self.predict(z_in)

        if not checkpoint or not torch.is_grad_enabled():
            return _run(z)
        return torch.utils.checkpoint.checkpoint(_run, z, use_reentrant=False)

    def _rollout_latents(self, z, action, checkpoint=False, fast_attention=False):
        """Predictor rollout body.

        Identical tensor ops, in identical order, to the previous ``rollout`` loop.

        input:  z: (b, n, num_patches, emb_dim) latent context (already encoded)
                action: (b, t, action_dim) actions past the context window
                checkpoint: recompute each ``predict`` in backward rather than storing
                    its activations.  Default False, so ``rollout`` -- and therefore
                    ``plan.py``, ``planning/*`` and ``Trainer.openloop_rollout`` -- take
                    the original path unchanged (Property 7).  Only ``compute_ccr``
                    passes True.
                fast_attention: run these predictor calls through
                    ``scaled_dot_product_attention``.  Also default False for the same
                    reason, and also only ever True from ``compute_ccr``.
        output: z: (b, n+t+1, num_patches, emb_dim)
        """
        t = 0
        inc = 1
        while t < action.shape[1]:
            z_pred = self._predict_maybe_checkpointed(
                z[:, -self.num_hist :], checkpoint, fast_attention
            )
            z_new = z_pred[:, -inc:, ...]
            z_new = self.replace_actions_from_z(z_new, action[:, t : t + inc, :])
            z = torch.cat([z, z_new], dim=1)
            t += inc

        z_pred = self._predict_maybe_checkpointed(
            z[:, -self.num_hist :], checkpoint, fast_attention
        )
        z_new = z_pred[:, -1 :, ...] # take only the next pred
        z = torch.cat([z, z_new], dim=1)
        return z

    def rollout(self, obs_0, act):
        """
        input:  obs_0 (dict): (b, n, 3, img_size, img_size)
                  act: (b, t+n, action_dim)
        output: embeddings of rollout obs
                visuals: (b, t+n+1, 3, img_size, img_size)
                z: (b, t+n+1, num_patches, emb_dim)
        """
        num_obs_init = obs_0['visual'].shape[1]
        act_0 = act[:, :num_obs_init]
        action = act[:, num_obs_init:] 
        z = self.encode(obs_0, act_0)
        z = self._rollout_latents(z, action)
        z_obses, z_acts = self.separate_emb(z)
        return z_obses, z

    def _sample_action_perturbation(self, act):
        """Uniform perturbation, elementwise bounded by rho in normalized-action units.

        `uniform_(-1, 1)` is a closed interval and multiplication by a non-negative rho is
        monotone, so every element lies in [-rho, +rho] as rho is represented in
        `act.dtype`. No clamping and no rejection sampling.

        There is deliberately **no** `if rho == 0` branch: rho = 0 must produce exact zeros
        through the identical code path and consume the identical RNG draws, so the control
        arm is a genuine code-path twin of the treatment arm.

        `empty_like` inherits device and dtype from `act`, which is what makes the
        "new tensor on the wrong device" failure structurally unreachable. Nothing here is
        moved with `.to(device)`, and rho is the only scalar involved, so no
        environment-specific or dataset-specific constant reaches this function.
        """
        rho = torch.as_tensor(self.ccr_rho, dtype=act.dtype, device=act.device)
        return torch.empty_like(act).uniform_(-1.0, 1.0).mul_(rho)

    def _ccr_actions(self, act, required):
        """Base action sequence of length `required`, plus one bounded uniform perturbation.

        input:  act: (b, available, action_dim) recorded normalized actions
                required: total number of action frames the imagined rollout needs
        output: (b, required, action_dim)

        The recorded prefix is perturbed under both action sources; frames past the window
        edge are synthesized as zeros (the planner's initialisation point) and then
        perturbed by the same sampler. There is no `ccr_action_source` argument on purpose:
        the source is enforced upstream by the guard in `compute_ccr`, which rejects
        `required > available` under 'logged', so the padding branch below is reachable only
        under 'synthetic'.
        """
        n_logged = min(required, act.shape[1])
        base = act[:, :n_logged]                                # recorded, normalized
        if required > n_logged:                                 # 'synthetic' only
            b, _, d = base.shape
            base = torch.cat([base, base.new_zeros(b, required - n_logged, d)], dim=1)
        return base + self._sample_action_perturbation(base)

    def compute_ccr(self, z, act):
        """Counterfactual curvature of an imagined, off-log rollout.

        input:  z: (b, num_frames, num_patches, emb_dim) latents `forward` already encoded
                act: (b, num_frames, action_dim) recorded normalized actions
        output: scalar curvature loss

        Zero additional encoder forward passes: `z` is reused, only its action channels are
        overwritten with the perturbed actions. The cost is L extra predictor calls, one
        action_encoder call and one encoder.agg call.
        """
        L = self.ccr_rollout_len
        required = self.num_hist + L - 1            # actions needed for L predictor steps
        available = act.shape[1]
        if self.ccr_action_source == "logged" and required > available:
            raise ValueError(
                f"CCR is enabled (lambda_cf={self.lambda_cf}) with ccr_rollout_len={L} and "
                f"ccr_action_source='logged', which needs an action sequence of length "
                f"{required} (num_hist={self.num_hist} + {L} - 1), but only {available} "
                f"action frames are available (num_frames={available}). Set "
                f"training.ccr_rollout_len <= {available - self.num_hist + 1}, or set "
                f"training.ccr_action_source=synthetic to synthesize the "
                f"{required - available} action frames past the window edge."
            )
        # One-shot completion of the __init__ startup line: `synthesized_action_frames`
        # needs num_frames, which is a property of the batch and is unknown in __init__.
        # This is the only place where both `required` and `available` are in scope, so the
        # resolved count is logged once here rather than being recomputed elsewhere.
        # A 'synthetic' arm reporting synthesized_action_frames=0 is silently a 'logged'
        # arm, which is exactly what the pilot smoke check reads this number for.
        if not self._ccr_forward_logged:
            self._ccr_forward_logged = True
            log.info(
                "CCR first forward: term=ccr, weight(lambda_cf)=%s, rho=%s, rollout_len=%s, "
                "action_source=%s, synthesized_action_frames=%s, curvature_mode=aggcos, "
                "device=%s",
                self.lambda_cf,
                self.ccr_rho,
                L,
                self.ccr_action_source,
                max(0, required - available),
                z.device,
            )
        act_cf = self._ccr_actions(act, required)   # logged prefix (+ synthesized tail)
        # `.clone()` is essential: replace_actions_from_z writes in place, and mutating a
        # view of `z` would corrupt the baseline prediction term's autograd graph.
        z_ctx = self.replace_actions_from_z(
            z[:, : self.num_hist].clone(), act_cf[:, : self.num_hist]
        )
        z_imag = self._rollout_latents(
            z_ctx,
            act_cf[:, self.num_hist : required],
            checkpoint=self.ccr_grad_checkpoint,
            fast_attention=self.ccr_fast_attention,
        )
        # Last L + 2 frames: every velocity pair entering _cos_curvature touches at least
        # one imagined frame, so the purely-real triple the baseline term already penalises
        # is not double-counted. `visual_only` matches the baseline curvature term's channel
        # selection (visual+proprio is unreachable: agg_mlp's input width is fixed at
        # num_patches * emb_dim). `aggcos` is hardcoded: CCR is defined on the aggregated
        # geometry.
        feats = self.visual_only(z_imag[:, -(L + 2) :])
        return self.total_curvature(feats, mode="aggcos")

    def compute_mca(self, z, eps=1e-6):
        """Metric-Consistent Aggregation (pilot only).

        Penalises `encoder.agg` for distorting velocity norms: straightness is enforced in
        aggregated space while planning/objectives.py scores distances in patch space.
        `agg` only needs to be a *similarity* (distance-preserving up to one global
        constant) for straightness to transfer, so the penalty compares each velocity's
        norm ratio against the batch-mean ratio and is therefore scale-invariant.

        `encoder.agg` is an existing module; this adds no module and no parameter.
        """
        feats = self.visual_only(z)                                  # (b, t, p, d)
        b, t, p, d = feats.shape
        agg = self.encoder.agg(feats.reshape(b * t, p, d)).reshape(b, t, -1)
        v_patch = (feats[:, 1:] - feats[:, :-1]).flatten(2).norm(dim=-1)   # (b, t-1)
        v_agg = (agg[:, 1:] - agg[:, :-1]).norm(dim=-1)                    # (b, t-1)
        r = v_agg / (v_patch + eps)
        r_bar = r.mean().detach().clamp_min(eps)
        return ((r / r_bar) - 1.0).pow(2).mean()