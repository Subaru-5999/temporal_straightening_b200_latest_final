"""The SDPA attention branch computes the same function as the materialised one.

Feature: counterfactual-curvature-regularization

`Attention.use_sdpa` swaps a materialised `(b, heads, T, T)` score matrix for
`F.scaled_dot_product_attention`. That is the *same* mathematical function, but "same
function" is a claim, and CCR's gradient flows through the fast branch while the baseline
prediction loss flows through the slow one. If the two disagree by more than floating-point
reduction order, the two terms are pulling on the shared predictor weights through
inconsistent linearisations, and the whole arm is invalid.

Two things are checked:

1. **Forward agreement** in float64 with dropout off, where the only permissible difference
   is summation order. Tolerances are tight on purpose.
2. **Gradient agreement**, because it is the gradient that CCR actually uses. A forward
   that matches while the backward does not would be worse than an obvious failure.

Dropout must be off for the comparison: the two branches consume randomness differently
(`nn.Dropout` on the score matrix versus SDPA's `dropout_p`), so with dropout on they are
equal only in distribution, which is not something a unit test should pretend to assert.
That is also exactly why the production wiring enters `sdpa_attention` *inside* the
checkpointed callable -- see `VWorldModel._predict_maybe_checkpointed`.
"""
import pytest

torch = pytest.importorskip("torch")

from models.vit import Attention, sdpa_attention  # noqa: E402


DIM = 32
HEADS = 4
DIM_HEAD = 8
BATCH = 2
NUM_PATCHES = 4
NUM_FRAMES = 3
T = NUM_PATCHES * NUM_FRAMES


def _build_attention():
    """An `Attention` with the block-causal mask the predictor uses, on the CPU.

    `Attention.__init__` reads the module-level NUM_PATCHES / NUM_FRAMES globals that
    `ViTPredictor.__init__` sets, and pins the mask to 'cuda'. Both are worked around here
    rather than changed, because the training and planning paths depend on that behaviour.
    """
    import models.vit as vit

    previous = (vit.NUM_PATCHES, vit.NUM_FRAMES)
    vit.NUM_PATCHES, vit.NUM_FRAMES = NUM_PATCHES, NUM_FRAMES
    try:
        attn = Attention(DIM, heads=HEADS, dim_head=DIM_HEAD, dropout=0.0)
    finally:
        vit.NUM_PATCHES, vit.NUM_FRAMES = previous
    attn.bias = attn.bias.to("cpu")          # __init__ hardcodes .to('cuda')
    return attn.double().eval()              # float64: isolate reduction order from noise


def _forward(attn, x, use_sdpa):
    with sdpa_attention(use_sdpa):
        assert Attention.use_sdpa is use_sdpa
        return attn(x)


def test_sdpa_forward_matches_materialised_attention():
    torch.manual_seed(0)
    attn = _build_attention()
    x = torch.randn(BATCH, T, DIM, dtype=torch.float64)

    slow = _forward(attn, x, False)
    fast = _forward(attn, x, True)

    assert slow.shape == fast.shape
    torch.testing.assert_close(fast, slow, rtol=1e-9, atol=1e-9)


def test_sdpa_gradients_match_materialised_attention():
    torch.manual_seed(0)
    attn = _build_attention()
    base = torch.randn(BATCH, T, DIM, dtype=torch.float64)

    grads = {}
    for use_sdpa in (False, True):
        attn.zero_grad(set_to_none=True)
        x = base.clone().requires_grad_(True)
        _forward(attn, x, use_sdpa).pow(2).sum().backward()
        grads[use_sdpa] = (
            x.grad.clone(),
            {name: p.grad.clone() for name, p in attn.named_parameters()},
        )

    slow_x, slow_params = grads[False]
    fast_x, fast_params = grads[True]
    torch.testing.assert_close(fast_x, slow_x, rtol=1e-8, atol=1e-8)
    assert set(fast_params) == set(slow_params)
    for name in slow_params:
        torch.testing.assert_close(
            fast_params[name], slow_params[name], rtol=1e-8, atol=1e-8,
            msg=lambda s, n=name: f"parameter grad mismatch for {n}: {s}",
        )


def test_masked_positions_are_excluded_under_both_branches():
    """The block-causal mask must still be enforced on the fast branch.

    SDPA's `attn_mask` uses the opposite convention to `masked_fill` -- True means "may
    attend" where the mask value 0 means "may not" -- so an inverted mask is the obvious way
    to get this wrong, and it would fail silently as a mild quality regression rather than an
    error. Perturbing a future position must leave an earlier query untouched.
    """
    torch.manual_seed(0)
    attn = _build_attention()
    x = torch.randn(BATCH, T, DIM, dtype=torch.float64)

    # Frame 0 occupies rows [0, NUM_PATCHES); it may attend to frame 0 only. Perturb the
    # last frame, which frame 0 must not see.
    x_perturbed = x.clone()
    x_perturbed[:, -NUM_PATCHES:] += 10.0

    for use_sdpa in (False, True):
        before = _forward(attn, x, use_sdpa)[:, :NUM_PATCHES]
        after = _forward(attn, x_perturbed, use_sdpa)[:, :NUM_PATCHES]
        torch.testing.assert_close(
            after, before, rtol=1e-9, atol=1e-9,
            msg=lambda s, u=use_sdpa: (
                f"use_sdpa={u}: an earlier frame's output changed when a later frame was "
                f"perturbed, so the block-causal mask is not being applied: {s}"
            ),
        )


def test_context_manager_restores_the_previous_value_on_exception():
    """A raised exception must not leave the fast path enabled for the baseline term."""
    assert Attention.use_sdpa is False
    with pytest.raises(RuntimeError):
        with sdpa_attention(True):
            assert Attention.use_sdpa is True
            raise RuntimeError("boom")
    assert Attention.use_sdpa is False
