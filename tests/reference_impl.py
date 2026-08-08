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

Do **not** "improve", simplify, reformat or extend the two functions frozen at
``d73b9c6``. Their only value is being a faithful frozen snapshot of the
pre-feature behaviour. If the frozen behaviour ever needs to change, the base
commit comment above must change with it.

Second snapshot: the pre-ACS curvature path
-------------------------------------------

The functions in the *"Curvature path frozen at ``d3c3ce5``"* section below are a
**second, independent** frozen snapshot, taken at a later commit and covering a
different code path. They exist because the ACS feature refactors
``_cos_curvature`` into ``_cos_curvature_terms`` / ``_agg_velocities``, and
Property 1 (default-off bitwise) and Property 12 (unweighted diagnostic bitwise)
need something to compare the refactored code against. The two ``d73b9c6``
functions above keep their names and their bodies: ``tests/test_rollout_refactor.py``
and ``tests/test_agg_zero_bitwise.py`` read them, so this addition is purely
additive. The same rule applies to the second snapshot: it is a copy, not a
maintained implementation.
"""

import torch
import torch.nn.functional as F


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

# ---------------------------------------------------------------------------
# Curvature path frozen at ``d3c3ce5``
# ---------------------------------------------------------------------------
#
# Base commit: ``d3c3ce5`` ("Implement aggregated-space planning cost
# (L_plan = L_spatial + w*L_agg)", full SHA
# ``d3c3ce55120912b650970f4fb52c75f47f08b0ea``).
#
# The three functions below are **verbatim copies** of
# ``VWorldModel._cos_curvature``, ``VWorldModel.total_curvature`` (both the
# ``cos`` and the ``aggcos`` branch) and the straightening tail of
# ``VWorldModel.forward`` in ``models/visual_world_model.py`` as they exist at
# that commit, rewritten only so that they are standalone functions taking the
# model as their first argument (every ``self.`` became ``model.``). Nothing
# else was changed: the same operations, the same order, the same dtypes, the
# same ``eps`` and ``step_thresh`` defaults, the same ``mask`` construction, the
# same ``loss_components`` keys.
#
# They are frozen *before* the ACS refactor splits ``_cos_curvature`` into
# ``_cos_curvature_terms`` / ``_agg_velocities``, so:
#
# - **Property 1** (the disabled path is bitwise the baseline) compares the
#   refactored ``VWorldModel._cos_curvature`` / ``VWorldModel.total_curvature``
#   and the ``aggcos`` forward tail against :func:`reference_cos_curvature`,
#   :func:`reference_total_curvature` and
#   :func:`reference_curvature_forward_tail`.
# - **Property 12** (the unweighted diagnostic is the baseline's number) compares
#   ``curvature_loss_unweighted`` against
#   ``reference_total_curvature(model, model.visual_only(z), mode="aggcos")``.
#
# Note the one difference against the ``d73b9c6`` snapshot above:
# :func:`reference_forward_loss` writes only
# ``curvature_loss_used_for_training``, because ``curvature_loss_scaled`` did not
# exist at ``d73b9c6``. :func:`reference_curvature_forward_tail` is the current
# tail and writes **both** keys. Use the tail below, not the one embedded in
# :func:`reference_forward_loss`, when the comparison is against present-day
# ``loss_components``.
#
# Do **not** "improve", simplify or reformat these three functions either.


def reference_cos_curvature(model, v1, v2, eps=1e-6, step_thresh=1e-6):
    """Frozen copy of ``VWorldModel._cos_curvature`` at commit ``d3c3ce5``."""
    cos = F.cosine_similarity(v1, v2, dim=-1, eps=eps)
    loss = 1.0 - cos
    if step_thresh > 0:
        step1 = v1.norm(dim=-1)
        step2 = v2.norm(dim=-1)
        mask = (step1 > step_thresh) & (step2 > step_thresh)
        loss = loss[mask]
    return loss.mean()


def reference_total_curvature(model, features, mode="cos"):
    """Frozen copy of ``VWorldModel.total_curvature`` at commit ``d3c3ce5``.

    Both the ``aggcos`` branch (velocities in the aggregated space produced by
    ``encoder.agg``) and the ``cos`` branch (patch-wise velocities) are copied,
    together with the two ``ValueError`` guards.
    """
    if features.shape[1] < 3:
        raise ValueError(f"Features must have at least 3 frames for curvature calculation, got {features.shape[1]}")

    if mode == "aggcos":
        if not hasattr(model.encoder, "agg"):
            raise ValueError("curvature mode 'aggcos' requires encoder.agg().")
        b, t, p, d = features.shape
        tokens = features.reshape(b * t, p, d)
        z = model.encoder.agg(tokens).reshape(b, t, -1)
        v1 = z[:, 1:-1] - z[:, :-2]
        v2 = z[:, 2:] - z[:, 1:-1]
    elif mode == "cos":
        v1 = features[:, 1:-1] - features[:, :-2]
        v2 = features[:, 2:] - features[:, 1:-1]
    else:
        raise ValueError(f"Unknown curvature mode '{mode}'. Use 'cos' or 'aggcos'.")

    return reference_cos_curvature(model, v1, v2)


def reference_curvature_forward_tail(model, z, loss, loss_components):
    """Frozen copy of ``VWorldModel.forward``'s straightening tail at ``d3c3ce5``.

    The block is ``if self.straighten and self.straighten_scale > 0:`` inside
    ``forward``. It is reproduced verbatim, including its comment, with the
    curvature call routed through :func:`reference_total_curvature` so that the
    frozen tail stays independent of the ACS refactor of ``_cos_curvature``.

    ``loss_components`` is mutated in place, exactly as ``forward`` mutates its
    own dict; the updated ``loss`` is returned because ``loss`` is a value, not a
    container.
    """
    if model.straighten and model.straighten_scale > 0:
        feats = model.visual_only(z)
        curvature_loss = reference_total_curvature(model, feats, mode=model.curvature_mode)
        loss = loss + curvature_loss * model.straighten_scale
        loss_components["curvature_loss_used_for_training"] = curvature_loss
        # Telemetry only: the scaled value makes the baseline curvature term
        # comparable against ccr_loss_scaled in loss shares. Adding a key does not
        # change the loss.
        loss_components["curvature_loss_scaled"] = (
            curvature_loss * model.straighten_scale
        )
    return loss

# ---------------------------------------------------------------------------
# MCA path frozen at ``6a5741c``
# ---------------------------------------------------------------------------
#
# Base commit: ``6a5741c`` ("Pre-register the MCA rung-1 gate before running the
# offline probe", full SHA ``6a5741c55a96b7aa1a460f704cd54ea41b58314c``).
#
# :func:`reference_compute_mca` is a **verbatim copy** of the body of
# ``VWorldModel.compute_mca`` in ``models/visual_world_model.py`` as it exists at
# that commit, rewritten only so that it is a standalone function taking the model
# as its first argument (every ``self.`` became ``model.``). Nothing else was
# changed: the same operations, the same order, the same dtypes, the same ``eps``
# default, the same ``flatten(2)`` / ``norm(dim=-1)`` reductions, the same
# ``r_bar.detach().clamp_min(eps)``.
#
# It is frozen *before* the MCA rung-1 refactor splits ``compute_mca`` into
# ``_mca_terms`` plus a reduction (`PROGRESS_MCA.md` §4.1, which requires that
# split to be bitwise neutral and the probe to call ``_mca_terms`` rather than
# re-derive ``r``). ``tests/test_mca_terms.py`` compares the refactored
# ``VWorldModel.compute_mca`` against this copy **bitwise**, over every
# ``agg_type`` in ``tests.conftest.AGG_TYPES``. That comparison is the structural
# fix for the CCR calibration error: the rung-1 headline statistic and the
# training penalty are the same number because they are the same code.
#
# This is a third, independent snapshot. It does not touch the ``d73b9c6`` or
# ``d3c3ce5`` sections above; they keep their names and their bodies, because
# ``tests/test_rollout_refactor.py``, ``tests/test_agg_zero_bitwise.py`` and
# ``tests/test_acs_off_bitwise.py`` read them.
#
# Do **not** "improve", simplify, reformat or extend the function below either.
# Its only value is being a faithful frozen snapshot of the pre-refactor
# behaviour. If the frozen behaviour ever needs to change, the base commit
# comment above must change with it.


def reference_compute_mca(model, z, eps=1e-6):
    """Frozen copy of ``VWorldModel.compute_mca`` at commit ``6a5741c``.

    Metric-Consistent Aggregation (pilot only).

    Penalises `encoder.agg` for distorting velocity norms: straightness is enforced in
    aggregated space while planning/objectives.py scores distances in patch space.
    `agg` only needs to be a *similarity* (distance-preserving up to one global
    constant) for straightness to transfer, so the penalty compares each velocity's
    norm ratio against the batch-mean ratio and is therefore scale-invariant.

    `encoder.agg` is an existing module; this adds no module and no parameter.
    """
    feats = model.visual_only(z)                                  # (b, t, p, d)
    b, t, p, d = feats.shape
    agg = model.encoder.agg(feats.reshape(b * t, p, d)).reshape(b, t, -1)
    v_patch = (feats[:, 1:] - feats[:, :-1]).flatten(2).norm(dim=-1)   # (b, t-1)
    v_agg = (agg[:, 1:] - agg[:, :-1]).norm(dim=-1)                    # (b, t-1)
    r = v_agg / (v_patch + eps)
    r_bar = r.mean().detach().clamp_min(eps)
    return ((r / r_bar) - 1.0).pow(2).mean()
