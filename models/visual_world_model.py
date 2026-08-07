import torch
import torch.nn as nn
import torch.nn.functional as F
# Explicit: `import torch` does not reliably bind the `torch.utils.checkpoint` submodule,
# and the CCR rollout depends on it (see _predict_maybe_checkpointed).
import torch.utils.checkpoint
import logging
from torchvision import transforms
from einops import rearrange, repeat

log = logging.getLogger(__name__)

# Permitted values for `training.ccr_action_source`.
#   'logged'    -- perturb recorded normalized actions only; the training window caps L
#   'synthetic' -- keep and perturb the recorded prefix, synthesize actions past the edge
CCR_ACTION_SOURCES = ("logged", "synthetic")


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
        ccr_grad_checkpoint=True,
        ccr_action_source="synthetic",
        mca_weight=0.0,
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

        if isinstance(straighten, str):
            if straighten.startswith("aggcos"):
                suffix = straighten.replace("aggcos", "")
                self.straighten_scale = float(suffix) if suffix else 1.0
                self.curvature_mode = "aggcos"
            elif straighten.startswith("cos"):
                suffix = straighten.replace("cos", "")
                self.straighten_scale = float(suffix) if suffix else 1.0
                self.curvature_mode = "cos"

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
        # Memory/compute tradeoff on the CCR rollout only; numerically neutral, so it is
        # deliberately in neither ccr_tag nor LOSS_SIGNATURE_KEYS. Default True because
        # without it the PushT target cell at L=5 needs ~40 GB of extra activation
        # memory and OOMs the 45 GB MIG slice.
        self.ccr_grad_checkpoint = bool(
            True if ccr_grad_checkpoint is None else ccr_grad_checkpoint
        )
        self.ccr_action_source = str(
            "synthetic" if ccr_action_source is None else ccr_action_source
        )
        self.mca_weight = float(0.0 if mca_weight is None else mca_weight)

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
                "grad_checkpoint=%s, device=%s",
                self.lambda_cf,
                self.ccr_rho,
                self.ccr_rollout_len,
                self.ccr_action_source,
                self.num_hist,
                self.ccr_rollout_len,
                self.ccr_grad_checkpoint,
                _ccr_device,
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

    def _cos_curvature(self, v1, v2, eps=1e-6, step_thresh=1e-6):
        cos = F.cosine_similarity(v1, v2, dim=-1, eps=eps)
        loss = 1.0 - cos
        if step_thresh > 0:
            step1 = v1.norm(dim=-1)
            step2 = v2.norm(dim=-1)
            mask = (step1 > step_thresh) & (step2 > step_thresh)
            loss = loss[mask]
        return loss.mean()

    def total_curvature(self, features, mode="cos"):
        if features.shape[1] < 3:
            raise ValueError(f"Features must have at least 3 frames for curvature calculation, got {features.shape[1]}")

        if mode == "aggcos":
            if not hasattr(self.encoder, "agg"):
                raise ValueError("curvature mode 'aggcos' requires encoder.agg().")
            b, t, p, d = features.shape
            tokens = features.reshape(b * t, p, d)
            z = self.encoder.agg(tokens).reshape(b, t, -1)
            v1 = z[:, 1:-1] - z[:, :-2]
            v2 = z[:, 2:] - z[:, 1:-1]
        elif mode == "cos":
            v1 = features[:, 1:-1] - features[:, :-2]
            v2 = features[:, 2:] - features[:, 1:-1]
        else:
            raise ValueError(f"Unknown curvature mode '{mode}'. Use 'cos' or 'aggcos'.")

        return self._cos_curvature(v1, v2)

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


    def _predict_maybe_checkpointed(self, z, checkpoint):
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
        """
        if not checkpoint or not torch.is_grad_enabled():
            return self.predict(z)
        return torch.utils.checkpoint.checkpoint(
            self.predict, z, use_reentrant=False
        )

    def _rollout_latents(self, z, action, checkpoint=False):
        """Predictor rollout body.

        Identical tensor ops, in identical order, to the previous ``rollout`` loop.

        input:  z: (b, n, num_patches, emb_dim) latent context (already encoded)
                action: (b, t, action_dim) actions past the context window
                checkpoint: recompute each ``predict`` in backward rather than storing
                    its activations.  Default False, so ``rollout`` -- and therefore
                    ``plan.py``, ``planning/*`` and ``Trainer.openloop_rollout`` -- take
                    the original path unchanged (Property 7).  Only ``compute_ccr``
                    passes True.
        output: z: (b, n+t+1, num_patches, emb_dim)
        """
        t = 0
        inc = 1
        while t < action.shape[1]:
            z_pred = self._predict_maybe_checkpointed(z[:, -self.num_hist :], checkpoint)
            z_new = z_pred[:, -inc:, ...]
            z_new = self.replace_actions_from_z(z_new, action[:, t : t + inc, :])
            z = torch.cat([z, z_new], dim=1)
            t += inc

        z_pred = self._predict_maybe_checkpointed(z[:, -self.num_hist :], checkpoint)
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