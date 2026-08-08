"""Task 4.2 - one implementation of the gate, shared by the Stage-0 probe and training.

Feature: action-conditioned-straightening, Property 19: One implementation of the gate,
shared by probe and training. ∀ windows: the Stage-0 readout's ``a_t`` and ``w_t`` are
produced by calling ``VWorldModel.reduce_action`` and ``VWorldModel.action_gate``; no second
implementation of either exists in the repository. Additionally, on 32 randomly selected
windows the action-only loader's tensor is **bitwise** equal to ``dset[idx][1]``, so skipping
the video decode cannot silently change what is measured.

This module is a **gate**, deliberately not optional. It is the structural fix for the CCR
calibration error applied to the gate: CCR shipped two implementations of one quantity, they
drifted, and the number the probe predicted was not the number training measured. Here the
cosine of consecutive actions is computed in exactly one place -
``VWorldModel.action_gate`` - and ``probe_ccr_curvature.py`` reaches it by constructing a real
``VWorldModel`` and calling the bound method. Three independent checks pin that down:

1. **Structure** (``cosine_similarity`` is absent from the probe; ``cos_and_gate`` contains
   nothing but two ``action_gate`` calls and the exact affine inversion; the only
   ``def reduce_action`` / ``def action_gate`` in the repository are the shipped ones). A
   source scan is what Requirement 15.3 asks for, and it is the only check that fails when a
   *future* edit adds a convenient local cosine.
2. **Behaviour** (``cos_and_gate`` is bitwise equal to the shipped gate on generated actions,
   and monkeypatching the two methods on the class proves the probe really routes through
   them). Structure without behaviour would pass on a probe that imported the method and
   then ignored it.
3. **Data** (the 32-window bitwise check against ``dset[idx][1]``). The fast Stage-0 loader
   reimplements ``TrajSlicerDataset.__getitem__``'s *indexing* on purpose - going through
   ``__getitem__`` opens a ``VideoReader`` and decodes 20 frames per window, turning minutes
   into hours - so the one thing that must be verified against the real dataset is that the
   tensor it produces is the same tensor, bit for bit.

Check 3 needs the datasets on disk, so it skips cleanly when ``DATASET_DIR`` is unset and the
CPU suite stays green; it is run as part of task 5.1, on the box that has the data.

Two things the source scan is careful about:

- **Comments and docstrings are stripped before scanning.** The probe's docstrings *discuss*
  the cosine at length ("no second cosine-of-actions implementation exists in this file"), so
  a raw-text grep for ``cosine_similarity`` would fail on prose. The flip side is recorded
  honestly: a cosine assembled at runtime out of string literals would evade this scan. No
  automated check covers that, and nothing in the repo does it.
- **The forbidden list is a list of primitives, not of the substring ``cos``.** The probe's
  ridge readout legitimately calls ``np.linalg.solve``, and ``models/proprio.py`` legitimately
  defines ``get_1d_sincos_pos_embed``; a substring scan would flag both. What is forbidden is
  a *dot product or a norm of actions*: ``cosine_similarity``, ``einsum``, ``tensordot``,
  ``dot``, ``norm``, ``normalize``, ``acos``/``arccos``, and the ``@`` operator inside the
  gate function.

Validates: Requirements 1.16, 15.1, 15.2, 15.3
"""

from __future__ import annotations

import ast
import io
import os
import random
import re
import sys
import tokenize
from pathlib import Path

import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import probe_ccr_curvature as probe  # noqa: E402
from models.visual_world_model import VWorldModel  # noqa: E402
from tests.conftest import (  # noqa: E402
    acs_antiparallel_action_cases,
    acs_cases,
    acs_parallel_action_cases,
    acs_zero_action_cases,
    build_stub_world_model,
)

PROBE_PATH = _REPO_ROOT / "probe_ccr_curvature.py"
MODEL_PATH = _REPO_ROOT / "models" / "visual_world_model.py"

# Requirement 1.16: "32 randomly selected windows". Seeded from the probe's own
# `PROBE_SEED` so two runs of the check inspect the same windows and a failure is
# reproducible by index.
BITWISE_WINDOWS = 32
BITWISE_SEED = probe.PROBE_SEED

# The gates Stage 0 measures. `permuted` is excluded on purpose: it is the training-time
# null control and shuffles `w` through `torch.randperm`, so it is not a function of `act`
# and cannot be compared bitwise across two calls. The probe's own enum excludes it, and
# `test_probe_gate_enum_excludes_the_nondeterministic_control` pins that.
PROBE_GATES = probe.ACS_GATES
PROBE_REDUCTIONS = probe.ACS_ACTION_REDUCTIONS

# Primitives that would constitute a second cosine-of-actions implementation. Each entry is
# (regex, what it would mean). Matched against the probe's *code*, with comments and string
# literals removed.
FORBIDDEN_PRIMITIVES = (
    (r"cosine_similarity", "a direct second call to the cosine primitive"),
    (r"\beinsum\b", "a hand-rolled inner product"),
    (r"\btensordot\b", "a hand-rolled inner product"),
    (r"\bmatmul\b", "a hand-rolled inner product"),
    (r"\.dot\s*\(", "a hand-rolled inner product"),
    (r"\bdot\s*\(", "a hand-rolled inner product"),
    (r"\.norm\s*\(", "a hand-rolled vector norm, i.e. the cosine's denominator"),
    (r"linalg\.norm", "a hand-rolled vector norm, i.e. the cosine's denominator"),
    (r"\bnormalize\s*\(", "a hand-rolled unit-vector normalization"),
    (r"\barccos\b", "an angle recovered from a locally computed cosine"),
    (r"\bacos\s*\(", "an angle recovered from a locally computed cosine"),
)

# `cos_and_gate`'s whole body, as an allowlist. Anything else is either a second
# implementation or a transformation of the gate's output that the reported statistics were
# not defined against.
COS_AND_GATE_ALLOWED_ATTR_CALLS = frozenset({"no_grad", "action_gate", "reshape", "float"})
COS_AND_GATE_ALLOWED_NAME_CALLS = frozenset({"acs_gate_model"})
# `2.0` and `1.0` are the affine inversion `2w - 1`; `1` is the `-1` of `reshape(-1)`.
COS_AND_GATE_ALLOWED_NUMBERS = frozenset({1, 1.0, 2.0})

# Directories with no first-party source in them. `tests/` is excluded from the
# repository-wide `def` scan because a test double may legitimately define a stand-in.
REPO_SCAN_SKIP_DIRS = frozenset(
    {
        ".git",
        ".hypothesis",
        ".kiro",
        ".venv",
        "__pycache__",
        "checkpoints",
        "checkpoints_ctrl8k",
        "node_modules",
        "outputs",
        "probe_outputs",
        "tests",
        "venv",
        "wandb",
    }
)

GATE_METHOD_NAMES = ("reduce_action", "action_gate")
_GATE_DEF_RE = re.compile(
    r"^[ \t]*def\s+(" + "|".join(GATE_METHOD_NAMES) + r")\s*\(", re.MULTILINE
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _code_only(source: str) -> str:
    """`source` with every comment and every string literal blanked out, layout preserved.

    The probe documents the single-implementation promise in prose, so the scan has to look
    at code. Tokenizing (rather than regex-stripping) is what makes that reliable for
    triple-quoted docstrings, f-strings and adjacent string concatenation alike.

    Characters are replaced with spaces rather than deleted, so a pattern that spans tokens -
    ``linalg.norm``, ``.dot(`` - still matches the surviving code exactly as it is written.
    """
    lines = [list(line) for line in source.splitlines(keepends=True)]

    def blank(token):
        (srow, scol), (erow, ecol) = token.start, token.end
        for row in range(srow, erow + 1):
            line = lines[row - 1]
            lo = scol if row == srow else 0
            hi = ecol if row == erow else len(line)
            for i in range(lo, min(hi, len(line))):
                if line[i] != "\n":
                    line[i] = " "

    removed = {tokenize.COMMENT, tokenize.STRING}
    # Python >= 3.12 tokenizes f-strings in pieces; the literal text arrives as
    # FSTRING_MIDDLE, while the `{...}` expressions arrive as ordinary code tokens.
    for name in ("FSTRING_START", "FSTRING_MIDDLE", "FSTRING_END"):
        if hasattr(tokenize, name):  # pragma: no cover - version dependent
            removed.add(getattr(tokenize, name))
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type in removed:
            blank(token)
    return "".join("".join(line) for line in lines)


def _function_def(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"no function or method named {name!r} in the parsed module")


def _calls(node: ast.AST) -> tuple[list[str], list[str]]:
    """(attribute-call names, plain-name-call names) inside `node`."""
    attrs, names = [], []
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        if isinstance(sub.func, ast.Attribute):
            attrs.append(sub.func.attr)
        elif isinstance(sub.func, ast.Name):
            names.append(sub.func.id)
    return attrs, names


def _bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Byte-for-byte equality, so `-0.0` vs `0.0` and NaN payloads are not glossed over."""
    if a.dtype != b.dtype or tuple(a.shape) != tuple(b.shape):
        return False
    return a.detach().contiguous().numpy().tobytes() == b.detach().contiguous().numpy().tobytes()


_MODEL_CACHE: dict[tuple[str, str], VWorldModel] = {}


def _shipped_model(action_reduce: str, gate: str) -> VWorldModel:
    """A stub-backed `VWorldModel` for one (reduction, gate) pair, cached.

    Cached because `action_gate` reads no weights at all - it is a function of `act` and the
    two enum strings - so one instance per pair is enough for 100 hypothesis examples, and
    building a fresh encoder per example would dominate the runtime.
    """
    key = (str(action_reduce), str(gate))
    model = _MODEL_CACHE.get(key)
    if model is None:
        model = build_stub_world_model(
            straighten=False, acs_action_reduce=action_reduce, acs_gate=gate
        )
        _MODEL_CACHE[key] = model
    return model


def _repo_py_files():
    for root, dirs, files in os.walk(_REPO_ROOT):
        dirs[:] = sorted(d for d in dirs if d not in REPO_SCAN_SKIP_DIRS)
        for name in sorted(files):
            if name.endswith(".py"):
                yield Path(root) / name


# ---------------------------------------------------------------------------
# Property 19, part 1: the source scan (Requirement 15.3)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pattern,meaning", FORBIDDEN_PRIMITIVES)
def test_probe_contains_no_independent_action_cosine(pattern, meaning):
    """No cosine, inner product or norm primitive appears in the probe's code.

    Requirement 15.3 in its literal form. Comments and docstrings are stripped first, so the
    probe is free to *explain* the promise it keeps.
    """
    code = _code_only(_read(PROBE_PATH))
    hits = [m.group(0) for m in re.finditer(pattern, code)]

    assert not hits, (
        f"probe_ccr_curvature.py contains {hits!r}, which is {meaning}. The Stage-0 gate must "
        f"come from VWorldModel.action_gate and nowhere else (Requirement 15.3): a second "
        f"implementation is exactly the CCR calibration error, where the number the probe "
        f"predicted stopped being the number training measured."
    )


def test_the_cosine_primitive_lives_in_action_gate():
    """The positive control: the cosine the probe must not have does exist, once, upstream.

    Without this the scan above would also pass on a repository that had lost the gate
    entirely.
    """
    tree = ast.parse(_read(MODEL_PATH))
    attrs, _names = _calls(_function_def(tree, "action_gate"))

    assert "cosine_similarity" in attrs, (
        "VWorldModel.action_gate no longer calls cosine_similarity, so the single "
        "implementation the probe delegates to has moved or gone."
    )


def test_action_gate_reduces_the_action_through_reduce_action():
    """`a_t` reaches the probe through `reduce_action` (Requirement 15.1, transitively).

    The probe calls `action_gate`, and `action_gate` is what calls `reduce_action`; pinning
    that edge is what makes "the probe's `a_t` is the shipped reduction" a structural fact
    rather than a coincidence of the current call graph.
    """
    tree = ast.parse(_read(MODEL_PATH))
    attrs, _names = _calls(_function_def(tree, "action_gate"))

    assert "reduce_action" in attrs, (
        "VWorldModel.action_gate no longer calls reduce_action, so the Stage-0 readout's a_t "
        "is no longer the shipped reduction (Requirement 15.1)."
    )


def test_cos_and_gate_is_nothing_but_two_shipped_gate_calls():
    """`cos_and_gate`'s body is the allowlist and nothing else (Requirements 15.1, 15.2).

    The two calls are the gate under test and the `affine_cos` gate the cosine is *recovered*
    from as `2w - 1`. Any third kind of call, any other arithmetic, or an `@` would be a local
    computation of a quantity the reported statistics were not defined against.
    """
    node = _function_def(ast.parse(_read(PROBE_PATH)), "cos_and_gate")
    attrs, names = _calls(node)

    assert sorted(attrs).count("action_gate") == 2, (
        f"cos_and_gate calls action_gate {attrs.count('action_gate')} time(s); expected "
        f"exactly two (the requested gate, and affine_cos to recover cos)."
    )
    assert set(attrs) <= COS_AND_GATE_ALLOWED_ATTR_CALLS, (
        f"cos_and_gate calls {sorted(set(attrs) - COS_AND_GATE_ALLOWED_ATTR_CALLS)}, which is "
        f"outside the allowlist {sorted(COS_AND_GATE_ALLOWED_ATTR_CALLS)}."
    )
    assert set(names) <= COS_AND_GATE_ALLOWED_NAME_CALLS, (
        f"cos_and_gate calls {sorted(set(names) - COS_AND_GATE_ALLOWED_NAME_CALLS)}; the only "
        f"function it may call is acs_gate_model, which builds the real VWorldModel."
    )

    operators = {type(sub.op) for sub in ast.walk(node) if isinstance(sub, ast.BinOp)}
    assert operators <= {ast.Mult, ast.Sub}, (
        f"cos_and_gate uses {sorted(op.__name__ for op in operators)}; the affine inversion "
        f"2w - 1 needs only Mult and Sub, and a MatMult or Div here would be a locally "
        f"computed cosine."
    )

    numbers = {
        sub.value
        for sub in ast.walk(node)
        if isinstance(sub, ast.Constant) and isinstance(sub.value, (int, float))
        and not isinstance(sub.value, bool)
    }
    assert numbers <= COS_AND_GATE_ALLOWED_NUMBERS, (
        f"cos_and_gate contains the numeric constant(s) "
        f"{sorted(numbers - COS_AND_GATE_ALLOWED_NUMBERS)}. The inversion of affine_cos is "
        f"exactly 2w - 1; a third constant is a threshold, a sharpness or a rescaling that "
        f"Stage 0 never declared."
    )


def test_probe_builds_the_real_world_model_for_the_gate():
    """`acs_gate_model` constructs `VWorldModel` and hands it both enum values.

    This is how the probe gets a *bound* `action_gate`: not a copied function, not a shim, but
    the shipped constructor - which is also what validates the two enums, so a bad
    `--acs-gate` raises at Stage 0 exactly as it would in training.
    """
    node = _function_def(ast.parse(_read(PROBE_PATH)), "acs_gate_model")
    source = ast.dump(node)

    assert "VWorldModel" in source, (
        "probe_ccr_curvature.acs_gate_model no longer constructs a VWorldModel; the Stage-0 "
        "gate would then not be the shipped gate (Requirements 15.1, 15.2)."
    )
    imported = {
        alias.name
        for sub in ast.walk(node)
        if isinstance(sub, ast.ImportFrom) and sub.module == "models.visual_world_model"
        for alias in sub.names
    }
    assert "VWorldModel" in imported, (
        "acs_gate_model must import VWorldModel from models.visual_world_model, so the class "
        "it instantiates is the one training uses."
    )
    kwargs = {
        kw.arg
        for sub in ast.walk(node)
        if isinstance(sub, ast.Call)
        for kw in sub.keywords
        if kw.arg is not None
    }
    for required in ("acs_action_reduce", "acs_gate"):
        assert required in kwargs, (
            f"acs_gate_model does not pass {required} to VWorldModel, so the Stage-0 model is "
            f"not configured for the arm being measured."
        )


def test_only_one_reduce_action_and_one_action_gate_in_the_repository():
    """`def reduce_action` and `def action_gate` each exist exactly once, in the model.

    The design's form of Property 19 is repository-wide, not probe-local: a second definition
    anywhere is the drift this feature is built to prevent, whoever calls it.
    """
    found: dict[str, list[str]] = {name: [] for name in GATE_METHOD_NAMES}
    for path in _repo_py_files():
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:  # pragma: no cover - unreadable file, not a definition
            continue
        for match in _GATE_DEF_RE.finditer(text):
            found[match.group(1)].append(str(path.relative_to(_REPO_ROOT)).replace("\\", "/"))

    expected = ["models/visual_world_model.py"]
    for name in GATE_METHOD_NAMES:
        assert found[name] == expected, (
            f"`def {name}` is defined in {found[name]}; expected exactly one definition, in "
            f"{expected[0]}. Two implementations of the gate is the CCR calibration error."
        )


def test_probe_gate_enum_excludes_the_nondeterministic_control():
    """Stage 0 measures the real gate, so `permuted` is not one of the probe's choices.

    `permuted` shuffles `w` across triples through `torch.randperm`; it is a training-time
    attribution arm and measures nothing about a dataset, so the probe's enum is the model's
    minus that member.
    """
    from models.visual_world_model import ACS_ACTION_REDUCTIONS, ACS_GATES

    assert probe.ACS_ACTION_REDUCTIONS == ACS_ACTION_REDUCTIONS
    assert set(PROBE_GATES) == set(ACS_GATES) - {"permuted"}


# ---------------------------------------------------------------------------
# Property 19, part 2: the probe's gate *is* the shipped gate (Requirements 15.1, 15.2)
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    case=acs_cases(),
    action_reduce=st.sampled_from(PROBE_REDUCTIONS),
    gate=st.sampled_from(PROBE_GATES),
)
def test_probe_gate_is_bitwise_the_shipped_gate(case, action_reduce, gate):
    """`cos_and_gate` returns the shipped `action_gate`'s values, bitwise, on every case.

    Structure is not enough on its own: a probe could import the method and then post-process
    its output. This is the behavioural half - same tensor in, same bits out, across the
    ordinary case and every degenerate one (static, one-moving, parallel, antiparallel,
    zero-norm actions).
    """
    cos, w = probe.cos_and_gate(case.act, action_reduce, gate, case.env_action_dim)

    gated = _shipped_model(action_reduce, gate)
    affine = _shipped_model(action_reduce, "affine_cos")
    with torch.no_grad():
        expected_w = gated.action_gate(case.act, env_action_dim=case.env_action_dim)
        expected_cos = affine.action_gate(case.act, env_action_dim=case.env_action_dim) * 2.0 - 1.0
    expected_w = expected_w.reshape(-1).float()
    expected_cos = expected_cos.reshape(-1).float()

    assert cos.shape == (case.num_triples,), (cos.shape, case.num_triples)
    assert w.shape == (case.num_triples,), (w.shape, case.num_triples)
    assert _bitwise_equal(w, expected_w), (
        f"the probe's w differs from VWorldModel.action_gate on kind={case.kind}, "
        f"reduce={action_reduce}, gate={gate}: probe {w.tolist()} vs shipped "
        f"{expected_w.tolist()}."
    )
    assert _bitwise_equal(cos, expected_cos), (
        f"the probe's cos differs from 2 * affine_cos - 1 on kind={case.kind}, "
        f"reduce={action_reduce}: probe {cos.tolist()} vs shipped {expected_cos.tolist()}."
    )
    # The gate is detached in the model; the probe must not hand back something that carries
    # a graph into the statistics.
    assert w.requires_grad is False and w.grad_fn is None
    assert cos.requires_grad is False and cos.grad_fn is None


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    case=st.one_of(
        acs_parallel_action_cases(),
        acs_antiparallel_action_cases(),
        acs_zero_action_cases(),
    ),
    action_reduce=st.sampled_from(PROBE_REDUCTIONS),
)
def test_recovered_cosine_matches_the_known_degenerate_values(case, action_reduce):
    """The `2w - 1` inversion recovers the cosine the construction guarantees.

    `parallel` gives `cos = +1`, `antiparallel` `cos = -1`, `zero_action` `cos = 0` through
    `cosine_similarity`'s own `eps` (E10). Asserting the *values* here, not just agreement
    with the model, is what would catch an inversion that agreed with a broken gate.
    """
    cos, _w = probe.cos_and_gate(case.act, action_reduce, "relu_cos", case.env_action_dim)

    expected = {"parallel": 1.0, "antiparallel": -1.0, "zero_action": 0.0}[case.kind]
    assert torch.allclose(cos, torch.full_like(cos, expected), atol=1e-6), (
        f"kind={case.kind}, reduce={action_reduce}: recovered cos {cos.tolist()} is not "
        f"{expected}."
    )


def test_cos_and_gate_routes_through_the_shipped_methods(monkeypatch):
    """Requirements 15.1 and 15.2, proven by observation rather than by reading.

    Both methods are replaced on the class - which is where the probe's cached instances look
    them up - and the call counts are asserted: two `action_gate` calls (the requested gate
    and `affine_cos`), and one `reduce_action` inside each of them.
    """
    calls = {name: 0 for name in GATE_METHOD_NAMES}
    real_reduce = VWorldModel.reduce_action
    real_gate = VWorldModel.action_gate

    def spy_reduce(self, *args, **kwargs):
        calls["reduce_action"] += 1
        return real_reduce(self, *args, **kwargs)

    def spy_gate(self, *args, **kwargs):
        calls["action_gate"] += 1
        return real_gate(self, *args, **kwargs)

    monkeypatch.setattr(VWorldModel, "reduce_action", spy_reduce)
    monkeypatch.setattr(VWorldModel, "action_gate", spy_gate)

    act = torch.randn(3, 4, 6, generator=torch.Generator().manual_seed(0))
    probe.cos_and_gate(act, "sum", "relu_cos", 2)

    assert calls["action_gate"] == 2, calls
    assert calls["reduce_action"] == 2, calls


# ---------------------------------------------------------------------------
# Property 19, part 3: the fast loader is bitwise the decoded loader (Requirement 1.16)
# ---------------------------------------------------------------------------

_DATASET_DIR = os.environ.get("DATASET_DIR")
requires_dataset = pytest.mark.skipif(
    not _DATASET_DIR,
    reason=(
        "DATASET_DIR is unset, so the 32-window bitwise check has no dataset to read. "
        "It runs as part of task 5.1 on the box that has the four datasets; the rest of "
        "this module is CPU-only and always runs."
    ),
)


@requires_dataset
@pytest.mark.parametrize("env_name", probe.ACS_ENVS)
def test_action_only_loader_is_bitwise_equal_to_getitem(env_name):
    """On 32 random windows, `act[idx]` is byte-for-byte `dset[idx][1]` (Requirement 1.16).

    The fast loader exists because `dset[idx]` routes PushT through `PushTDataset.get_frames`,
    opens a `VideoReader` and decodes 20 frames per window - hours instead of minutes for a
    statistic that reads one tensor. That optimization is only legitimate if it changes
    nothing, which is a claim about bits: dtype, shape and every byte, so a silent float64
    round-trip or a `-0.0` would fail here rather than shift a histogram bin at Stage 0.
    """
    train_cfg = probe.compose_env_cfg(env_name)
    data_path = Path(str(train_cfg.env.dataset.data_path))
    if not data_path.exists():
        pytest.skip(f"env={env_name}: no dataset at {data_path}")

    dset, act, meta = probe.load_action_windows(train_cfg, "train")

    n_windows = len(dset)
    assert int(act.shape[0]) == n_windows, (act.shape, n_windows)
    assert int(act.shape[1]) == meta["num_frames"]
    assert int(act.shape[2]) == meta["block_dim"]

    indices = random.Random(BITWISE_SEED).sample(
        range(n_windows), min(BITWISE_WINDOWS, n_windows)
    )
    for idx in indices:
        reference = dset[idx][1]
        if not isinstance(reference, torch.Tensor):  # pragma: no cover - every shipped dataset
            reference = torch.as_tensor(reference)  # returns a tensor here
        fast = act[idx]
        assert fast.dtype == reference.dtype, (idx, fast.dtype, reference.dtype)
        assert tuple(fast.shape) == tuple(reference.shape), (idx, fast.shape, reference.shape)
        assert _bitwise_equal(fast, reference), (
            f"env={env_name}, window {idx}: the action-only loader's tensor is not bitwise "
            f"equal to dset[{idx}][1]. Skipping the video decode has changed what Stage 0 "
            f"measures (Requirement 1.16)."
        )
