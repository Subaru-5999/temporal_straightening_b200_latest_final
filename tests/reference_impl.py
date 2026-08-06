"""Frozen pre-feature reference implementations of ``VWorldModel`` internals.

Base commit: ``d73b9c6`` ("Initial commit",
full SHA ``d73b9c6ba94cc404dd5ac66011824dfcda693d93``).

The two functions below are **verbatim copies** of the bodies of
``VWorldModel.forward`` and ``VWorldModel.rollout`` in
``models/visual_world_model.py`` as they exist at that commit, rewritten only so
that they are standalone functions taking the model as their first argument
(every ``self.`` became ``model.``). Nothing else was changed: the tensor
operations, their order, the ``concat_dim == 0`` / ``concat_dim == 1`` branches,
the vcreg branch, the straighten branch and the decoder branches are all as in
the original.

This is the model side of the model-based tests:

- **Property 1** (the disabled path is the baseline path) compares
  ``VWorldModel.forward`` with ``lambda_cf = 0`` and ``mca_weight = 0`` against
  :func:`reference_forward_loss`.
- **Property 7** (the rollout refactor preserves rollout) compares
  ``VWorldModel.rollout`` after the ``_rollout_latents`` extraction against
  :func:`reference_rollout`.

Do **not** "improve", simplify, reformat or extend this file. Its only value is
being a faithful frozen snapshot of the pre-feature behaviour. If the frozen
behaviour ever needs to change, the base commit comment above must change with
it.
"""

import torch


def reference_forward_loss(model, obs, act):
    """Frozen copy of ``VWorldModel.forward`` at commit ``d73b9c6``.

    input:  obs (dict):  "visual", "proprio" (b, num_frames, 3, img_size, img_size)
            act: (b, num_frames, action_dim)
    output: z_pred: (b, num_hist, num_patches, emb_dim)
            visual_pred: (b, num_hist, 3, img_size, img_size)
            visual_reconstructed: (b, num_frames, 3, img_size, img_size)
    """
    loss = 0
    loss_components = {}
    decoder_enabled = model.decoder is not None and model.train_decoder
    z = model.encode(obs, act)
    z_src = z[:, : model.num_hist, :, :]  # (b, num_hist, num_patches, dim)
    z_tgt = z[:, model.num_pred :, :, :]  # (b, num_hist, num_patches, dim)
    visual_src = obs['visual'][:, : model.num_hist, ...]  # (b, num_hist, 3, img_size, img_size)
    visual_tgt = obs['visual'][:, model.num_pred :, ...]  # (b, num_hist, 3, img_size, img_size)

    if model.predictor is not None:
        z_pred = model.predict(z_src)
        if decoder_enabled:
            obs_pred, diff_pred = model.decode(
                z_pred.detach()
            )  # recon loss should only affect decoder
            visual_pred = obs_pred['visual']
            recon_loss_pred = model.decoder_criterion(visual_pred, visual_tgt)
            decoder_loss_pred = (
                recon_loss_pred + model.decoder_latent_loss_weight * diff_pred
            )
            loss_components["decoder_recon_loss_pred"] = recon_loss_pred
            loss_components["decoder_vq_loss_pred"] = diff_pred
            loss_components["decoder_loss_pred"] = decoder_loss_pred
        else:
            visual_pred = None

        # Compute loss for visual, proprio dims (i.e. exclude action dims)
        z_tgt_for_loss = z_tgt.detach() if model.stop_grad else z_tgt
        if model.concat_dim == 0:
            z_visual_loss = model.emb_criterion(z_pred[:, :, :-2, :], z_tgt_for_loss[:, :, :-2, :])
            z_proprio_loss = model.emb_criterion(z_pred[:, :, -2, :], z_tgt_for_loss[:, :, -2, :])
            z_loss = model.emb_criterion(z_pred[:, :, :-1, :], z_tgt_for_loss[:, :, :-1, :])
        elif model.concat_dim == 1:
            z_visual_loss = model.emb_criterion(
                z_pred[:, :, :, :-(model.proprio_dim + model.action_dim)], \
                z_tgt_for_loss[:, :, :, :-(model.proprio_dim + model.action_dim)]
            )
            z_proprio_loss = model.emb_criterion(
                z_pred[:, :, :, -(model.proprio_dim + model.action_dim): -model.action_dim],
                z_tgt_for_loss[:, :, :, -(model.proprio_dim + model.action_dim): -model.action_dim]
            )
            z_loss = model.emb_criterion(
                z_pred[:, :, :, :-model.action_dim],
                z_tgt_for_loss[:, :, :, :-model.action_dim]
            )

        loss = loss + z_loss
        loss_components["z_loss"] = z_loss
        loss_components["z_visual_loss"] = z_visual_loss
        loss_components["z_proprio_loss"] = z_proprio_loss

        if model.vcreg:
            z_vic_in = model.visual_prop(z)
            z_std_loss = model.vcreg_std_loss(z_vic_in)
            z_cov_loss = model.vcreg_cov_loss(z_vic_in)
            z_reg_loss = z_std_loss * model.std_coeff + z_cov_loss * model.cov_coeff
            loss_components["z_vicreg_std_loss"] = z_std_loss
            loss_components["z_vicreg_cov_loss"] = z_cov_loss
            loss_components["z_vcreg_loss_scaled"] = z_reg_loss
            loss = loss + z_reg_loss

        if model.straighten and model.straighten_scale > 0:
            feats = model.visual_only(z)
            curvature_loss = model.total_curvature(feats, mode=model.curvature_mode)
            loss = loss + curvature_loss * model.straighten_scale
            loss_components["curvature_loss_used_for_training"] = curvature_loss
    else:
        visual_pred = None
        z_pred = None

    if decoder_enabled:
        obs_reconstructed, diff_reconstructed = model.decode(
            z.detach()
        )  # recon loss should only affect decoder
        visual_reconstructed = obs_reconstructed["visual"]
        recon_loss_reconstructed = model.decoder_criterion(visual_reconstructed, obs['visual'])
        decoder_loss_reconstructed = (
            recon_loss_reconstructed
            + model.decoder_latent_loss_weight * diff_reconstructed
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


def reference_rollout(model, obs_0, act):
    """Frozen copy of ``VWorldModel.rollout`` at commit ``d73b9c6``.

    input:  obs_0 (dict): (b, n, 3, img_size, img_size)
              act: (b, t+n, action_dim)
    output: embeddings of rollout obs
            visuals: (b, t+n+1, 3, img_size, img_size)
            z: (b, t+n+1, num_patches, emb_dim)
    """
    num_obs_init = obs_0['visual'].shape[1]
    act_0 = act[:, :num_obs_init]
    action = act[:, num_obs_init:]
    z = model.encode(obs_0, act_0)
    t = 0
    inc = 1
    while t < action.shape[1]:
        z_pred = model.predict(z[:, -model.num_hist :])
        z_new = z_pred[:, -inc:, ...]
        z_new = model.replace_actions_from_z(z_new, action[:, t : t + inc, :])
        z = torch.cat([z, z_new], dim=1)
        t += inc

    z_pred = model.predict(z[:, -model.num_hist :])
    z_new = z_pred[:, -1 :, ...] # take only the next pred
    z = torch.cat([z, z_new], dim=1)
    z_obses, z_acts = model.separate_emb(z)
    return z_obses, z
