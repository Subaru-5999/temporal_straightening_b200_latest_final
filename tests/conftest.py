"""CPU test doubles, shared fixtures and hypothesis strategies for the CCR feature.

Everything here runs on CPU in float32 against a tiny stub encoder: no DINOv2 download, no
GPU, no dataset. The stub deliberately mirrors the parts of ``models/dino.py`` that
``VWorldModel`` and the curvature machinery actually touch (``name``, ``emb_dim``,
``latent_ndim``, ``patch_size``, ``forward`` and ``agg``), so a property that passes here is a
statement about the real code path and not about a convenient fiction.

Requirements: 5.6
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from hypothesis import assume, event, settings
from hypothesis import strategies as st

# The repo root must be importable so `models.*` resolves when pytest is invoked from
# anywhere. tests/ is a package, so pytest already prepends the root under the default
# import mode; this keeps direct imports of this module working too.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.proprio import ProprioceptiveEmbedding  # noqa: E402
from models.visual_world_model import VWorldModel  # noqa: E402

# ---------------------------------------------------------------------------
# Hypothesis profiles (minimum 100 examples per property, no per-example deadline:
# a stub forward + rollout is fast but not microsecond-fast).
# ---------------------------------------------------------------------------

settings.register_profile("ccr", max_examples=100, deadline=None, print_blob=True)
settings.register_profile("ccr-fast", max_examples=25, deadline=None)
settings.register_profile("ccr-thorough", max_examples=500, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "ccr"))


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class StubEncoder(nn.Module):
    """Tiny stand-in for :class:`models.dino.DinoV2Encoder`.

    ``name = "tiny"`` contains no ``"dino"``, so ``VWorldModel`` takes the identity
    ``encoder_transform`` branch instead of the DINOv2 resize branch. ``forward`` returns
    ``(b * t, num_patches, emb_dim)`` and is differentiable in its input, so gradients flow
    through the encoder exactly as they do in training.
    """

    def __init__(
        self,
        emb_dim: int = 4,
        num_patches: int = 4,
        in_chans: int = 3,
        patch_size: int = 2,
        latent_ndim: int = 2,
        agg_type: str = "mlp",
        agg_out_dim: int | None = None,
        agg_mlp_hidden_dim: int | None = None,
    ):
        super().__init__()
        grid_hw = int(round(num_patches**0.5))
        if grid_hw * grid_hw != num_patches:
            raise ValueError(f"num_patches must be a perfect square, got {num_patches}")

        self.name = "tiny"
        self.emb_dim = int(emb_dim)
        self.num_patches = int(num_patches)
        self.grid_hw = grid_hw
        self.in_chans = int(in_chans)
        self.patch_size = int(patch_size)
        self.latent_ndim = int(latent_ndim)
        self.feature_key = "x_norm_patchtokens"
        self.agg_type = agg_type
        self.agg_out_dim = agg_out_dim
        self.agg_mlp_hidden_dim = agg_mlp_hidden_dim

        self.proj = nn.Linear(self.in_chans, self.emb_dim)

        # Mirrors models/dino.py: the aggregation MLP exists only for agg_type == "mlp",
        # and its input width is num_patches * emb_dim (196 * emb_dim on the real encoder).
        if self.agg_type == "mlp":
            self._agg_mlp_in_dim = self.num_patches * self.emb_dim
            self._agg_out_dim = (
                int(self.agg_out_dim) if self.agg_out_dim is not None else int(self.emb_dim)
            )
            self._agg_mlp_hidden_dim = (
                int(self.agg_mlp_hidden_dim)
                if self.agg_mlp_hidden_dim is not None
                else 4 * self._agg_out_dim
            )
            self.agg_mlp = nn.Sequential(
                nn.Linear(self._agg_mlp_in_dim, self._agg_mlp_hidden_dim),
                nn.ReLU(),
                nn.Linear(self._agg_mlp_hidden_dim, self._agg_mlp_hidden_dim),
                nn.ReLU(),
                nn.Linear(self._agg_mlp_hidden_dim, self._agg_out_dim),
            )
            self.agg_post_norm = nn.LayerNorm(self._agg_out_dim)

    def agg(self, x):
        """Same ``mean | flatten | mlp`` contract as :meth:`models.dino.DinoV2Encoder.agg`."""
        if self.agg_type == "mean":
            return x.mean(dim=1)
        x = x.contiguous().view(x.shape[0], -1)
        if self.agg_type == "flatten":
            return x
        if self.agg_type == "mlp":
            x = self.agg_mlp(x)
            return self.agg_post_norm(x)
        return x

    def forward(self, x, return_agg=False):
        # x: (b * t, c, h, w) -> pooled to a grid_hw x grid_hw token grid.
        if x.dim() != 4:
            raise ValueError(f"StubEncoder expects (n, c, h, w), got {tuple(x.shape)}")
        x = F.adaptive_avg_pool2d(x, (self.grid_hw, self.grid_hw))
        x = x.flatten(2).transpose(1, 2)  # (n, num_patches, c)
        emb = self.proj(x)  # (n, num_patches, emb_dim)
        if return_agg and emb.dim() == 3:
            emb = self.agg(emb)
        if self.latent_ndim == 1:
            emb = emb.unsqueeze(1)  # dummy patch dim
        return emb


class StubPredictor(nn.Module):
    """Linear over tokens, matching the ``(b, t * p, d) -> (b, t * p, d)`` predictor contract."""

    def __init__(self, dim: int, num_patches: int | None = None, num_frames: int | None = None):
        super().__init__()
        self.dim = int(dim)
        self.num_patches = num_patches
        self.num_frames = num_frames
        self.net = nn.Linear(self.dim, self.dim)

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

DEFAULT_EMB_DIM = 4
DEFAULT_NUM_PATCHES = 4
DEFAULT_PROPRIO_RAW_DIM = 3
DEFAULT_ACTION_RAW_DIM = 2
DEFAULT_IMAGE_SIZE = 8


def seed_all(seed: int = 0) -> None:
    """Seed the single RNG the CPU test doubles draw from."""
    torch.manual_seed(int(seed))


def build_stub_world_model(
    *,
    num_hist: int = 3,
    num_pred: int = 1,
    concat_dim: int = 1,
    agg_type: str = "mlp",
    straighten: str | bool = "aggcos1e-1",
    stop_grad: bool = True,
    lambda_cf: float = 0.0,
    ccr_rho: float = 0.0,
    ccr_rollout_len: int = 5,
    ccr_action_source: str = "synthetic",
    mca_weight: float = 0.0,
    emb_dim: int = DEFAULT_EMB_DIM,
    num_patches: int = DEFAULT_NUM_PATCHES,
    proprio_raw_dim: int = DEFAULT_PROPRIO_RAW_DIM,
    action_raw_dim: int = DEFAULT_ACTION_RAW_DIM,
    proprio_emb_dim: int | None = None,
    action_emb_dim: int | None = None,
    num_proprio_repeat: int = 1,
    num_action_repeat: int = 1,
    agg_out_dim: int | None = None,
    agg_mlp_hidden_dim: int | None = None,
    image_size: int = 224,
    train_encoder: bool = True,
    train_predictor: bool = True,
    vcreg: bool = False,
    vcreg_std_coeff: float = 0.0,
    vcreg_cov_coeff: float = 0.0,
    seed: int = 0,
    **extra_model_kwargs,
) -> VWorldModel:
    """Build a float32 CPU :class:`VWorldModel` from the stubs.

    The five CCR/MCA knobs (``lambda_cf``, ``ccr_rho``, ``ccr_rollout_len``,
    ``ccr_action_source``, ``mca_weight``) are passed straight through. They do not exist on
    ``VWorldModel`` until task 4.1 adds them; until then its ``**kwargs`` absorbs them, so
    this helper is correct both before and after that task lands.

    Weights are deterministic in ``seed``: two calls with the same arguments produce
    bitwise-identical models, which is what Properties 1, 4 and 5 compare across.
    """
    if concat_dim not in (0, 1):
        raise ValueError(f"concat_dim must be 0 or 1, got {concat_dim}")

    if proprio_emb_dim is None:
        proprio_emb_dim = emb_dim if concat_dim == 0 else 2
    if action_emb_dim is None:
        action_emb_dim = emb_dim if concat_dim == 0 else 2

    if concat_dim == 0 and (proprio_emb_dim != emb_dim or action_emb_dim != emb_dim):
        # concat_dim == 0 appends proprio/action as extra *tokens*, so their width has to
        # match the visual embedding width or torch.cat fails.
        raise ValueError(
            "concat_dim=0 requires proprio_emb_dim == action_emb_dim == emb_dim "
            f"(got {proprio_emb_dim}, {action_emb_dim}, {emb_dim})"
        )

    seed_all(seed)

    encoder = StubEncoder(
        emb_dim=emb_dim,
        num_patches=num_patches,
        patch_size=2,
        latent_ndim=2,
        agg_type=agg_type,
        agg_out_dim=agg_out_dim,
        agg_mlp_hidden_dim=agg_mlp_hidden_dim,
    )
    proprio_encoder = ProprioceptiveEmbedding(
        num_frames=1,
        tubelet_size=1,
        in_chans=proprio_raw_dim,
        emb_dim=proprio_emb_dim,
        use_3d_pos=False,
        use_layernorm=True,
    )
    action_encoder = ProprioceptiveEmbedding(
        num_frames=1,
        tubelet_size=1,
        in_chans=action_raw_dim,
        emb_dim=action_emb_dim,
        use_3d_pos=False,
        use_layernorm=True,
    )

    # Same width arithmetic train.py uses when it builds the real predictor.
    token_dim = emb_dim + (
        proprio_emb_dim * num_proprio_repeat + action_emb_dim * num_action_repeat
    ) * concat_dim
    predictor_patches = num_patches + (2 if concat_dim == 0 else 0)
    predictor = StubPredictor(
        dim=token_dim, num_patches=predictor_patches, num_frames=num_hist
    )

    model = VWorldModel(
        image_size=image_size,
        num_hist=num_hist,
        num_pred=num_pred,
        encoder=encoder,
        proprio_encoder=proprio_encoder,
        action_encoder=action_encoder,
        decoder=None,
        predictor=predictor,
        proprio_dim=proprio_emb_dim,
        action_dim=action_emb_dim,
        concat_dim=concat_dim,
        num_action_repeat=num_action_repeat,
        num_proprio_repeat=num_proprio_repeat,
        train_encoder=train_encoder,
        train_predictor=train_predictor,
        train_decoder=False,
        straighten=straighten,
        stop_grad=stop_grad,
        vcreg=vcreg,
        vcreg_std_coeff=vcreg_std_coeff,
        vcreg_cov_coeff=vcreg_cov_coeff,
        vcreg_apply_to="enc",
        lambda_cf=lambda_cf,
        ccr_rho=ccr_rho,
        ccr_rollout_len=ccr_rollout_len,
        ccr_action_source=ccr_action_source,
        mca_weight=mca_weight,
        **extra_model_kwargs,
    )
    return model.to(device="cpu", dtype=torch.float32)


def make_stub_batch(
    model: VWorldModel | None = None,
    *,
    batch_size: int = 2,
    num_frames: int = 4,
    image_size: int = DEFAULT_IMAGE_SIZE,
    proprio_raw_dim: int | None = None,
    action_raw_dim: int | None = None,
    num_action_frames: int | None = None,
    seed: int = 0,
):
    """Build ``(obs, act)`` on CPU in float32.

    ``obs["visual"]`` is ``(b, t, 3, image_size, image_size)``, ``obs["proprio"]`` is
    ``(b, t, proprio_raw_dim)`` and ``act`` is ``(b, num_action_frames or t, action_raw_dim)``
    in ``[-1, 1]`` (the dataset normalizes actions). Raw dims are read off the model's
    proprio/action encoders when a model is given.
    """
    if model is not None:
        if proprio_raw_dim is None:
            proprio_raw_dim = int(model.proprio_encoder.in_chans)
        if action_raw_dim is None:
            action_raw_dim = int(model.action_encoder.in_chans)
    proprio_raw_dim = DEFAULT_PROPRIO_RAW_DIM if proprio_raw_dim is None else proprio_raw_dim
    action_raw_dim = DEFAULT_ACTION_RAW_DIM if action_raw_dim is None else action_raw_dim
    act_frames = num_frames if num_action_frames is None else num_action_frames

    gen = torch.Generator(device="cpu").manual_seed(int(seed))

    def _uniform(*shape, low=-1.0, high=1.0):
        return torch.rand(*shape, generator=gen, dtype=torch.float32) * (high - low) + low

    obs = {
        "visual": _uniform(batch_size, num_frames, 3, image_size, image_size, low=0.0, high=1.0),
        "proprio": _uniform(batch_size, num_frames, proprio_raw_dim),
    }
    act = _uniform(batch_size, act_frames, action_raw_dim)
    return obs, act


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

MIN_NUM_FRAMES, MAX_NUM_FRAMES = 3, 6
MIN_NUM_HIST, MAX_NUM_HIST = 2, 3
MIN_ROLLOUT_LEN, MAX_ROLLOUT_LEN = 1, 6
ACTION_SOURCES = ("logged", "synthetic")
AGG_TYPES = ("mean", "flatten", "mlp")

batch_size_strategy = st.integers(min_value=1, max_value=4)
num_frames_strategy = st.integers(min_value=MIN_NUM_FRAMES, max_value=MAX_NUM_FRAMES)
num_hist_strategy = st.integers(min_value=MIN_NUM_HIST, max_value=MAX_NUM_HIST)
ccr_rollout_len_strategy = st.integers(min_value=MIN_ROLLOUT_LEN, max_value=MAX_ROLLOUT_LEN)
ccr_action_source_strategy = st.sampled_from(ACTION_SOURCES)
agg_type_strategy = st.sampled_from(AGG_TYPES)
concat_dim_strategy = st.sampled_from((0, 1))

# rho and lambda_cf both include exactly 0: the disabled path and the zero-perturbation
# control arm are the two cases most worth hitting, and plain float ranges hit them rarely.
rho_strategy = st.one_of(
    st.just(0.0),
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
lambda_cf_strategy = st.one_of(
    st.just(0.0),
    st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
)
positive_lambda_cf_strategy = st.floats(
    min_value=1e-3, max_value=10.0, allow_nan=False, allow_infinity=False
)


@dataclass(frozen=True)
class CCRWindow:
    """A jointly generated ``(ccr_rollout_len, num_frames, num_hist, ccr_action_source)`` tuple.

    Generated as one composite so the feasible/infeasible horizon boundary is exercised under
    both action sources, rather than only at the PushT target cell ``(5, 4, 3)``.
    """

    num_frames: int
    num_hist: int
    num_pred: int
    ccr_rollout_len: int
    ccr_action_source: str

    @property
    def available(self) -> int:
        """Logged action frames on hand."""
        return self.num_frames

    @property
    def required(self) -> int:
        """``num_hist + L - 1`` action frames the counterfactual rollout consumes."""
        return self.num_hist + self.ccr_rollout_len - 1

    @property
    def feasible(self) -> bool:
        return self.required <= self.available

    @property
    def max_permitted_rollout_len(self) -> int:
        return self.available - self.num_hist + 1

    @property
    def synthesized_action_frames(self) -> int:
        return max(0, self.required - self.available)

    @property
    def matches_forward_shapes(self) -> bool:
        """True when ``forward``'s source and target windows line up (``num_hist + num_pred``)."""
        return self.num_frames == self.num_hist + self.num_pred


@st.composite
def ccr_windows(
    draw,
    *,
    match_forward_shapes: bool = False,
    action_sources: tuple[str, ...] = ACTION_SOURCES,
    feasible: bool | None = None,
) -> CCRWindow:
    """Joint strategy over rollout length, window shape and action source.

    ``match_forward_shapes=True`` pins ``num_frames = num_hist + num_pred`` for tests that call
    ``VWorldModel.forward`` (whose prediction target is ``z[:, num_pred:]``). ``feasible`` pins
    the horizon to one side of the boundary; left as ``None``, both sides are drawn.
    """
    num_pred = 1
    num_hist = draw(num_hist_strategy)
    if match_forward_shapes:
        num_frames = num_hist + num_pred
    else:
        num_frames = draw(
            st.integers(
                min_value=max(MIN_NUM_FRAMES, num_hist + num_pred), max_value=MAX_NUM_FRAMES
            )
        )
    source = draw(st.sampled_from(tuple(action_sources)))

    max_feasible_len = num_frames - num_hist + 1  # num_hist + L - 1 <= num_frames
    feasible_lens = st.integers(
        min_value=MIN_ROLLOUT_LEN, max_value=min(MAX_ROLLOUT_LEN, max_feasible_len)
    )
    infeasible_lens = (
        st.integers(min_value=max_feasible_len + 1, max_value=MAX_ROLLOUT_LEN)
        if max_feasible_len < MAX_ROLLOUT_LEN
        else None
    )

    if feasible is True:
        rollout_len = draw(feasible_lens)
    elif feasible is False:
        # No infeasible L within the generator's range for this window shape.
        assume(infeasible_lens is not None)
        rollout_len = draw(infeasible_lens)
    else:
        branches = [feasible_lens] if infeasible_lens is None else [feasible_lens, infeasible_lens]
        rollout_len = draw(st.one_of(*branches))

    window = CCRWindow(
        num_frames=num_frames,
        num_hist=num_hist,
        num_pred=num_pred,
        ccr_rollout_len=rollout_len,
        ccr_action_source=source,
    )
    event(f"action_source={source}, feasible={window.feasible}")
    return window


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def cpu_float32_defaults():
    """Everything in this suite is CPU float32 and single-threaded for determinism."""
    prev_dtype = torch.get_default_dtype()
    prev_threads = torch.get_num_threads()
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(1)
    yield
    torch.set_default_dtype(prev_dtype)
    torch.set_num_threads(prev_threads)


@pytest.fixture
def make_world_model():
    """Factory fixture wrapping :func:`build_stub_world_model`."""
    return build_stub_world_model


@pytest.fixture
def make_batch():
    """Factory fixture wrapping :func:`make_stub_batch`."""
    return make_stub_batch


@pytest.fixture
def stub_encoder():
    seed_all(0)
    return StubEncoder()


@pytest.fixture
def target_cell_model():
    """PushT target-cell shapes: ``num_hist=3``, ``num_pred=1``, ``concat_dim=1``, ``agg_type=mlp``."""
    return build_stub_world_model(
        num_hist=3,
        num_pred=1,
        concat_dim=1,
        agg_type="mlp",
        straighten="aggcos1e-1",
        stop_grad=True,
    )


@pytest.fixture
def target_cell_batch(target_cell_model):
    """The batch that goes with :func:`target_cell_model`: ``num_frames = num_hist + num_pred``."""
    return make_stub_batch(target_cell_model, batch_size=2, num_frames=4)

# ===========================================================================
# Aggregated-space planning cost: test doubles, strategies and fixtures
#
# Feature: aggregated-space-planning-cost (task 1.2). Strictly ADDITIVE — every
# name above this banner keeps its identity and its value, because the CCR
# property tests depend on them.
#
# Everything here is CPU float32 and tiny. The stand-in Agg_Head mirrors
# ``models/dino.py::DinoV2Encoder.agg``'s ``mlp`` branch exactly
# (``x.contiguous().view(x.shape[0], -1)`` -> ``agg_mlp`` -> ``agg_post_norm``)
# but with parameterised widths, so a property that passes here is a statement
# about the real head's shape contract rather than about a 1568-wide fiction.
#
# Requirements: 1.1, 1.3, 1.8, 2.4
# ===========================================================================

# Target_Cell widths, recorded so tests can assert against them without
# building a ~1M-parameter head. `DinoV2Encoder` hardcodes 196 patches in
# `_agg_mlp_in_dim = 196 * emb_dim`; conf/encoder/dino_channel.yaml sets
# emb_dim 8 via the channel projector, agg_out_dim 128, agg_mlp_hidden_dim 512.
AGG_TARGET_CELL_PATCHES = 196
AGG_TARGET_CELL_CHANNELS = 8
AGG_TARGET_CELL_IN_DIM = AGG_TARGET_CELL_PATCHES * AGG_TARGET_CELL_CHANNELS  # 1568
AGG_TARGET_CELL_HIDDEN_DIM = 512
AGG_TARGET_CELL_OUT_DIM = 128

# Defaults for the stand-in head: small on purpose, so 100 hypothesis examples
# stay fast. `in_dim` is a product of a patch count and a channel width, which
# is the invariant `_apply_head` checks (Requirement 1.9).
AGG_DEFAULT_PATCHES = 4
AGG_DEFAULT_CHANNELS = 3
AGG_DEFAULT_IN_DIM = AGG_DEFAULT_PATCHES * AGG_DEFAULT_CHANNELS  # 12
AGG_DEFAULT_HIDDEN_DIM = 8
AGG_DEFAULT_OUT_DIM = 5


class AggMlpHead(nn.Module):
    """Stand-in Agg_Head: the ``mlp`` branch of :meth:`models.dino.DinoV2Encoder.agg`.

    Accepts ``(n, patches, channels)`` — the shape ``_apply_head`` hands over after
    reshaping ``(b, t, p, d)`` to ``(b * t, p, d)`` — and flattens it with the real
    encoder's own ``x.contiguous().view(x.shape[0], -1)``. Already-flat
    ``(n, in_dim)`` input is accepted too, since that view is idempotent.

    Three views of the same parameters are exposed on purpose:

    - ``net``: the flat 6-element ``nn.Sequential(Linear, ReLU, Linear, ReLU,
      Linear, LayerNorm)`` — the composed form ``extract_agg_head`` returns.
    - ``agg_mlp`` / ``agg_post_norm``: the checkpoint-shaped pair, named exactly as
      the real encoder names them.

    ``net`` holds the *same* module objects as ``agg_mlp`` and ``agg_post_norm``, so
    ``parameters()`` (which dedupes by identity) reports each tensor once.
    """

    def __init__(
        self,
        in_dim: int = AGG_DEFAULT_IN_DIM,
        hidden_dim: int = AGG_DEFAULT_HIDDEN_DIM,
        out_dim: int = AGG_DEFAULT_OUT_DIM,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.agg_type = "mlp"

        self.agg_mlp = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.out_dim),
        )
        self.agg_post_norm = nn.LayerNorm(self.out_dim)
        self.net = nn.Sequential(*self.agg_mlp, self.agg_post_norm)

    def forward(self, x):
        x = x.contiguous().view(x.shape[0], -1)
        return self.net(x)


class IdentityAggHead(nn.Module):
    """Identity-on-flattened-features Agg_Head variant, which Property 3 needs.

    ``(n, patches, channels) -> (n, patches * channels)`` and nothing else: the
    same ``x.contiguous().view(x.shape[0], -1)`` the real ``agg`` performs, with the
    MLP and the LayerNorm removed. With this head and ``alpha = 0`` the
    aggregated-space loss must equal the frozen ``planning.objectives`` value for the
    same mode, ``base`` and ``step``, because a mean over ``p * d`` flattened features
    is the same reduction as a mean over the ``(p, d)`` axes.

    Parameter-free, so ``requires_grad_(False)``, ``eval()`` and ``to(...)`` are all
    no-ops that still work, and gradients pass straight through to the input.
    """

    def __init__(self, in_dim: int | None = None):
        super().__init__()
        self.in_dim = None if in_dim is None else int(in_dim)
        self.out_dim = self.in_dim
        self.agg_type = "flatten"

    def forward(self, x):
        return x.contiguous().view(x.shape[0], -1)


class StubAggEncoder(nn.Module):
    """Encoder stand-in for :func:`agg_objectives.extract_agg_head`, no checkpoint on disk.

    Exposes exactly the five attributes the extractor reads — ``agg_type``,
    ``agg_mlp``, ``agg_post_norm``, ``_agg_mlp_in_dim``, ``_agg_out_dim`` — and, like
    the real :class:`models.dino.DinoV2Encoder`, builds ``agg_mlp`` / ``agg_post_norm``
    **only** when ``agg_type == "mlp"``. Any other ``agg_type`` therefore reproduces
    the checkpoint the wrapper has to abort on (Requirement 2.6), including the
    missing-attribute shape of it.
    """

    def __init__(
        self,
        agg_type: str = "mlp",
        num_patches: int = AGG_DEFAULT_PATCHES,
        emb_dim: int = AGG_DEFAULT_CHANNELS,
        agg_out_dim: int | None = AGG_DEFAULT_OUT_DIM,
        agg_mlp_hidden_dim: int | None = AGG_DEFAULT_HIDDEN_DIM,
    ):
        super().__init__()
        self.name = "tiny"
        self.agg_type = agg_type
        self.num_patches = int(num_patches)
        self.emb_dim = int(emb_dim)
        self.agg_out_dim = agg_out_dim
        self.agg_mlp_hidden_dim = agg_mlp_hidden_dim
        self.latent_ndim = 2
        self.feature_key = "x_norm_patchtokens"

        if self.agg_type == "mlp":
            self._agg_mlp_in_dim = self.num_patches * self.emb_dim
            self._agg_out_dim = (
                int(self.agg_out_dim) if self.agg_out_dim is not None else int(self.emb_dim)
            )
            self._agg_mlp_hidden_dim = (
                int(self.agg_mlp_hidden_dim)
                if self.agg_mlp_hidden_dim is not None
                else 4 * self._agg_out_dim
            )
            head = AggMlpHead(
                in_dim=self._agg_mlp_in_dim,
                hidden_dim=self._agg_mlp_hidden_dim,
                out_dim=self._agg_out_dim,
            )
            self.agg_mlp = head.agg_mlp
            self.agg_post_norm = head.agg_post_norm

    def agg(self, x):
        """Same ``mean | flatten | mlp`` contract as :meth:`models.dino.DinoV2Encoder.agg`."""
        if self.agg_type == "mean":
            return x.mean(dim=1)
        x = x.contiguous().view(x.shape[0], -1)
        if self.agg_type == "flatten":
            return x
        if self.agg_type == "mlp":
            return self.agg_post_norm(self.agg_mlp(x))
        return x


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def make_agg_head(
    *,
    in_dim: int = AGG_DEFAULT_IN_DIM,
    hidden_dim: int = AGG_DEFAULT_HIDDEN_DIM,
    out_dim: int = AGG_DEFAULT_OUT_DIM,
    seed: int = 0,
) -> AggMlpHead:
    """Deterministic stand-in Agg_Head on CPU in float32.

    Two calls with the same arguments give bitwise-identical parameters, which is
    what Property 5's before/after byte comparison rests on.
    """
    seed_all(seed)
    head = AggMlpHead(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim)
    return head.to(device="cpu", dtype=torch.float32)


def make_identity_agg_head(*, in_dim: int | None = None) -> IdentityAggHead:
    """The identity-on-flattened-features head variant (Property 3)."""
    return IdentityAggHead(in_dim=in_dim)


def make_stub_agg_encoder(
    *,
    agg_type: str = "mlp",
    num_patches: int = AGG_DEFAULT_PATCHES,
    emb_dim: int = AGG_DEFAULT_CHANNELS,
    agg_out_dim: int | None = AGG_DEFAULT_OUT_DIM,
    agg_mlp_hidden_dim: int | None = AGG_DEFAULT_HIDDEN_DIM,
    seed: int = 0,
) -> StubAggEncoder:
    """Deterministic :class:`StubAggEncoder` for ``extract_agg_head`` tests."""
    seed_all(seed)
    encoder = StubAggEncoder(
        agg_type=agg_type,
        num_patches=num_patches,
        emb_dim=emb_dim,
        agg_out_dim=agg_out_dim,
        agg_mlp_hidden_dim=agg_mlp_hidden_dim,
    )
    return encoder.to(device="cpu", dtype=torch.float32)


# ---------------------------------------------------------------------------
# Hypothesis strategies (aggregated-space)
# ---------------------------------------------------------------------------

AGG_MODES = ("last", "all", "staged")

# Non-finite and denormal float32 values. Property 1 asserts that Agg_Weight 0 is a
# *bitwise* identity even when L_agg is inf or nan, so the strategy has to actually
# generate those rather than merely permit them. The denormals sit below float32's
# smallest normal (1.1754944e-38) and survive the cast into a float32 tensor.
AGG_NONFINITE_VALUES = (float("inf"), float("-inf"), float("nan"))
AGG_DENORMAL_VALUES = (1.4e-45, -1.4e-45, 5.9e-44, -5.9e-44, 5.9e-39, -5.9e-39)
AGG_SPECIAL_VALUES = AGG_NONFINITE_VALUES + AGG_DENORMAL_VALUES

AGG_MIN_BATCH, AGG_MAX_BATCH = 1, 4
AGG_MIN_FRAMES, AGG_MAX_FRAMES = 2, 6
AGG_MIN_PATCHES, AGG_MAX_PATCHES = 1, 6
AGG_MIN_CHANNELS, AGG_MAX_CHANNELS = 1, 6

agg_batch_size_strategy = st.integers(min_value=AGG_MIN_BATCH, max_value=AGG_MAX_BATCH)
agg_num_frames_strategy = st.integers(min_value=AGG_MIN_FRAMES, max_value=AGG_MAX_FRAMES)
agg_patch_count_strategy = st.integers(min_value=AGG_MIN_PATCHES, max_value=AGG_MAX_PATCHES)
agg_channel_width_strategy = st.integers(min_value=AGG_MIN_CHANNELS, max_value=AGG_MAX_CHANNELS)
agg_proprio_dim_strategy = st.integers(min_value=1, max_value=3)
agg_hidden_dim_strategy = st.integers(min_value=2, max_value=8)
agg_out_dim_strategy = st.integers(min_value=1, max_value=6)

agg_mode_strategy = st.sampled_from(AGG_MODES)

# alpha and agg_weight both include exactly 0: alpha=0 is what pins the proprio term
# of L_agg to nothing (design section 2), and agg_weight=0 is the Baseline_Arm, so
# both are the values most worth hitting and plain float ranges hit them rarely.
agg_alpha_strategy = st.one_of(
    st.just(0.0),
    st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False),
)
agg_weight_strategy = st.one_of(
    st.just(0.0),
    st.floats(min_value=0.0, max_value=3.0, allow_nan=False, allow_infinity=False),
)
positive_agg_weight_strategy = st.floats(
    min_value=1e-3, max_value=3.0, allow_nan=False, allow_infinity=False
)

# `base` feeds `coeffs = [base ** i for i in range(T)]` in the frozen objective, so
# both the integer values the configs ship and arbitrary floats in range are drawn.
agg_base_strategy = st.one_of(
    st.sampled_from((1, 2, 3, 4)),
    st.floats(min_value=1.0, max_value=4.0, allow_nan=False, allow_infinity=False),
)

agg_nonfinite_value_strategy = st.sampled_from(AGG_SPECIAL_VALUES)


def agg_step_strategy(num_frames: int | None = None):
    """``step`` as the frozen objective receives it: ``None``, or ``0 .. T``.

    ``None`` is what open-loop (`last` mode) passes and what ``objective_fn_staged``
    treats as "no stage information". The upper bound is ``T`` rather than ``T - 1``
    so the staged dispatch predicate ``step < T - 1`` is exercised on both sides.
    """
    upper = AGG_MAX_FRAMES if num_frames is None else int(num_frames)
    return st.one_of(st.none(), st.integers(min_value=0, max_value=upper))


@st.composite
def agg_tensors(draw, shape, *, nonfinite: bool = False, max_injections: int = 4):
    """A CPU float32 tensor of ``shape``, optionally seeded with non-finite values.

    Values come from a torch generator seeded by a drawn integer rather than from a
    drawn list of floats: the shapes here reach a few hundred elements, and drawing
    them elementwise would dominate the runtime of every property. Specials are then
    written into drawn positions, so ``inf``, ``-inf``, ``nan`` and denormals appear
    inside otherwise ordinary tensors, which is the case that matters.
    """
    gen = torch.Generator(device="cpu").manual_seed(draw(st.integers(0, 2**31 - 1)))
    tensor = torch.randn(tuple(int(s) for s in shape), generator=gen, dtype=torch.float32)
    if nonfinite:
        flat = tensor.view(-1)
        n = flat.numel()
        count = draw(st.integers(min_value=1, max_value=min(max_injections, n)))
        for _ in range(count):
            flat[draw(st.integers(min_value=0, max_value=n - 1))] = draw(
                agg_nonfinite_value_strategy
            )
    return tensor


@dataclass
class AggLatents:
    """A generated ``(z_obs_pred, z_obs_tgt)`` pair plus the shapes that produced it.

    ``visual`` is ``(b, T, p, d)`` for the prediction and ``(b, 1, p, d)`` for the goal,
    which is what ``wm.rollout`` and ``encode_obs`` hand the objective; ``in_dim`` is
    the ``p * d`` width the head must accept (Requirement 1.9).
    """

    z_pred: dict
    z_tgt: dict
    batch_size: int
    num_frames: int
    patches: int
    channels: int
    proprio_dim: int

    @property
    def in_dim(self) -> int:
        return self.patches * self.channels


@st.composite
def agg_latent_dicts(
    draw,
    *,
    batch_size: int | None = None,
    num_frames: int | None = None,
    patches: int | None = None,
    channels: int | None = None,
    proprio_dim: int | None = None,
    nonfinite: bool = False,
) -> AggLatents:
    """Latent dictionaries in the shapes the planning objective is called with.

    Any dimension can be pinned by the caller; the rest are drawn. ``nonfinite=True``
    routes both visual tensors through the special-value strategy, which is what
    Property 1 needs.
    """
    b = draw(agg_batch_size_strategy) if batch_size is None else int(batch_size)
    t = draw(agg_num_frames_strategy) if num_frames is None else int(num_frames)
    p = draw(agg_patch_count_strategy) if patches is None else int(patches)
    d = draw(agg_channel_width_strategy) if channels is None else int(channels)
    pdim = draw(agg_proprio_dim_strategy) if proprio_dim is None else int(proprio_dim)

    z_pred = {
        "visual": draw(agg_tensors((b, t, p, d), nonfinite=nonfinite)),
        "proprio": draw(agg_tensors((b, t, pdim), nonfinite=nonfinite)),
    }
    z_tgt = {
        "visual": draw(agg_tensors((b, 1, p, d), nonfinite=nonfinite)),
        "proprio": draw(agg_tensors((b, 1, pdim), nonfinite=nonfinite)),
    }
    event(f"agg latents: T={t}, p={p}, d={d}, nonfinite={nonfinite}")
    return AggLatents(
        z_pred=z_pred,
        z_tgt=z_tgt,
        batch_size=b,
        num_frames=t,
        patches=p,
        channels=d,
        proprio_dim=pdim,
    )


@dataclass
class AggShapeMismatch:
    """A deliberately mismatched ``(patches, channels)`` against a head width (Property 6)."""

    patches: int
    channels: int
    in_dim: int

    @property
    def flattened(self) -> int:
        return self.patches * self.channels


@st.composite
def agg_mismatched_shapes(draw) -> AggShapeMismatch:
    """``p * d != in_dim``, so ``_apply_head`` must raise before ``nn.Linear`` does."""
    p = draw(agg_patch_count_strategy)
    d = draw(agg_channel_width_strategy)
    in_dim = draw(st.integers(min_value=1, max_value=AGG_MAX_PATCHES * AGG_MAX_CHANNELS))
    assume(p * d != in_dim)
    return AggShapeMismatch(patches=p, channels=d, in_dim=in_dim)


@st.composite
def agg_head_shapes(draw):
    """A jointly generated ``(patches, channels, hidden_dim, out_dim)`` for a matching head.

    Drawn together so ``in_dim == patches * channels`` holds by construction: that is
    the one relation between the latent shape and the head that has to be right for
    ``_apply_head`` to accept the input at all.
    """
    p = draw(agg_patch_count_strategy)
    d = draw(agg_channel_width_strategy)
    return {
        "patches": p,
        "channels": d,
        "in_dim": p * d,
        "hidden_dim": draw(agg_hidden_dim_strategy),
        "out_dim": draw(agg_out_dim_strategy),
    }


# ---------------------------------------------------------------------------
# Fixtures (aggregated-space)
# ---------------------------------------------------------------------------


@pytest.fixture
def make_head():
    """Factory fixture wrapping :func:`make_agg_head`."""
    return make_agg_head


@pytest.fixture
def make_identity_head():
    """Factory fixture wrapping :func:`make_identity_agg_head`."""
    return make_identity_agg_head


@pytest.fixture
def make_agg_encoder():
    """Factory fixture wrapping :func:`make_stub_agg_encoder`."""
    return make_stub_agg_encoder


@pytest.fixture
def agg_head():
    """The default small stand-in Agg_Head: ``12 -> 8 -> 8 -> 5`` plus ``LayerNorm(5)``."""
    return make_agg_head()


@pytest.fixture
def identity_agg_head():
    """The identity-on-flattened-features head variant."""
    return make_identity_agg_head(in_dim=AGG_DEFAULT_IN_DIM)


@pytest.fixture
def stub_agg_encoder():
    """A stub encoder carrying an ``agg_type == "mlp"`` head, for ``extract_agg_head``."""
    return make_stub_agg_encoder()
