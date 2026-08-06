"""Property 2 - run naming is empty at defaults, complete otherwise, and injective.

The run directory is derived entirely inside the Hydra ``hydra.run.dir`` /
``hydra.sweep.dir`` expressions in ``conf/train.yaml``, so this file tests the real
templates read out of that file rather than a copy of them:

* the "pre-feature template" is the shipped template with the appended ``ccr_tag``
  interpolation removed, which is only a well-defined operation because the feature is
  required to append at the very end (asserted below);
* the legacy-equivalence property resolves both templates side by side through OmegaConf
  over generated legacy configurations;
* one plain unit test resolves the whole thing through Hydra's compose API, so the
  byte-identical PushT target-cell path is checked against the real ``hydra.run.dir``
  rather than against the resolver in isolation.

Validates: Requirements 3.4, 6.4, 6.5

NOTE ON EXECUTION: hydra/omegaconf/pytest/hypothesis are not installed in the Windows dev
environment this file was written in, so the suite is verified on the pod
(``pytest tests/test_run_naming.py``). Local verification was limited to byte-compilation.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from omegaconf import OmegaConf

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Import first: this is what registers `replace_substring` / `ccr_tag` with OmegaConf, and
# both templates below are unresolvable without it.
import custom_resolvers  # noqa: E402
from custom_resolvers import CCR_TAG_DEFAULTS, _fmt_num, ccr_tag  # noqa: E402

CONF_DIR = _REPO_ROOT / "conf"
TRAIN_YAML = CONF_DIR / "train.yaml"

# The one interpolation task 3.2 appends to the end of the Run_Naming expression.
CCR_TAG_INTERPOLATION = (
    "${ccr_tag:${training.lambda_cf},${training.ccr_rho},"
    "${training.ccr_action_source},${training.mca_weight}}"
)

# Requirement 3.4: the PushT target cell, byte for byte.
LEGACY_PUSHT_RUN_NAME = "pusht_aggmlpcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05"

TAG_FORBIDDEN_CHARS = (".", "/", "\\")


# ---------------------------------------------------------------------------
# Template access
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _raw_templates() -> dict:
    """The unresolved ``hydra.run.dir`` / ``hydra.sweep.dir`` strings from conf/train.yaml."""
    raw = OmegaConf.to_container(OmegaConf.load(TRAIN_YAML), resolve=False)
    hydra_node = raw["hydra"]
    return {"run": hydra_node["run"]["dir"], "sweep": hydra_node["sweep"]["dir"]}


def _pre_feature_template(template: str) -> str:
    """Strip the appended ``ccr_tag`` interpolation, recovering the pre-feature template."""
    assert template.endswith(CCR_TAG_INTERPOLATION), (
        "the ccr_tag interpolation must be appended to the very END of the Run_Naming "
        f"expression, leaving the legacy expression untouched; got: {template!r}"
    )
    return template[: -len(CCR_TAG_INTERPOLATION)]


_UNDER_TEST = "run_dir_under_test"


def _resolve(template: str, cfg_values: dict) -> str:
    """Resolve one Run_Naming template against a config skeleton."""
    cfg = OmegaConf.create({**cfg_values, _UNDER_TEST: template})
    return str(cfg[_UNDER_TEST])


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

ENV_NAMES = (
    "point_maze",
    "point_maze_medium",
    "pusht",
    "wall",
    "rope",
    "granular",
    "deformable_env",
)
STRAIGHTEN_VALUES = (False, "cos1e-1", "cos1e-2", "aggcos1e-1", "aggcos1e-2")
ENCODER_LRS = (1e-6, 1e-5, 3e-4)
AGG_TYPES = ("mean", "flatten", "mlp")
PROJECTORS = ("channel", "none")

# Values that are pairwise distinct under `_fmt_num`, so "distinct tuple" and "distinct
# formatted tuple" coincide for the worked cases the pilot actually launches.
DISTINCT_WEIGHTS = (0.0, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 10.0)
ACTION_SOURCES = ("synthetic", "logged")

#: ``DISTINCT_WEIGHTS`` without 0.0, i.e. the regime where CCR is actually switched on
#: (``VWorldModel`` sets ``self.ccr = self.lambda_cf > 0``). Defined locally from
#: ``DISTINCT_WEIGHTS`` rather than imported from tests/conftest.py so this module keeps
#: owning its own strategies. Used only for ``lambda_cf``; see
#: ``test_action_source_alone_separates_two_arms`` for why.
POSITIVE_DISTINCT_WEIGHTS = tuple(w for w in DISTINCT_WEIGHTS if w != 0.0)

weight_strategy = st.one_of(
    st.sampled_from(DISTINCT_WEIGHTS),
    st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
)
distinct_weight_strategy = st.sampled_from(DISTINCT_WEIGHTS)
positive_distinct_weight_strategy = st.sampled_from(POSITIVE_DISTINCT_WEIGHTS)
action_source_strategy = st.sampled_from(ACTION_SOURCES)

ccr_tuple_strategy = st.tuples(
    weight_strategy, weight_strategy, action_source_strategy, weight_strategy
)
distinct_ccr_tuple_strategy = st.tuples(
    distinct_weight_strategy,
    distinct_weight_strategy,
    action_source_strategy,
    distinct_weight_strategy,
)


@st.composite
def legacy_configs(draw) -> dict:
    """A pre-feature configuration: the five knobs the legacy run name is built from."""
    encoder: dict = {"agg_type": draw(st.sampled_from(AGG_TYPES))}
    if draw(st.booleans()):
        encoder["projector"] = draw(st.sampled_from(PROJECTORS))
    if draw(st.booleans()):
        encoder["projector_out_dim"] = draw(st.sampled_from((8, 32, 384)))
    if draw(st.booleans()):
        encoder["projector_target_hw"] = draw(st.sampled_from((7, 14)))

    env: dict = {"name": draw(st.sampled_from(ENV_NAMES))}
    if draw(st.booleans()):
        env["save_name"] = draw(st.sampled_from(ENV_NAMES))

    return {
        "ckpt_base_path": "./checkpoints",
        "env": env,
        "encoder": encoder,
        "training": {
            "straighten": draw(st.sampled_from(STRAIGHTEN_VALUES)),
            "stop_grad": draw(st.booleans()),
            "encoder_lr": draw(st.sampled_from(ENCODER_LRS)),
            # CCR keys are filled in per-test.
            "lambda_cf": CCR_TAG_DEFAULTS[0],
            "ccr_rho": CCR_TAG_DEFAULTS[1],
            "ccr_action_source": CCR_TAG_DEFAULTS[2],
            "mca_weight": CCR_TAG_DEFAULTS[3],
        },
    }


def _with_ccr(cfg_values: dict, ccr: tuple) -> dict:
    lambda_cf, rho, source, mca = ccr
    training = dict(cfg_values["training"])
    training.update(
        lambda_cf=lambda_cf, ccr_rho=rho, ccr_action_source=source, mca_weight=mca
    )
    return dict(cfg_values, training=training)


def _formatted_key(ccr: tuple) -> tuple:
    lambda_cf, rho, source, mca = ccr
    return (_fmt_num(lambda_cf), _fmt_num(rho), str(source), _fmt_num(mca))


def _is_default(ccr: tuple) -> bool:
    lambda_cf, rho, source, mca = ccr
    return (float(lambda_cf), float(rho), str(source), float(mca)) == CCR_TAG_DEFAULTS


# ---------------------------------------------------------------------------
# Property 2
# ---------------------------------------------------------------------------


@given(cfg_values=legacy_configs())
def test_defaults_resolve_byte_identical_to_the_pre_feature_template(cfg_values):
    """Feature: counterfactual-curvature-regularization, Property 2: Run naming is empty at defaults, complete otherwise, and injective

    With the new keys at their defaults, both Run_Naming templates resolve to exactly the
    string the pre-feature template produces, for any legacy configuration.

    Validates: Requirements 3.4, 6.5
    """
    for template in _raw_templates().values():
        legacy = _resolve(_pre_feature_template(template), cfg_values)
        current = _resolve(template, cfg_values)
        assert current == legacy


@given(cfg_values=legacy_configs(), ccr=ccr_tuple_strategy)
def test_non_default_tuples_append_a_complete_tag(cfg_values, ccr):
    """Feature: counterfactual-curvature-regularization, Property 2: Run naming is empty at defaults, complete otherwise, and injective

    Any non-default tuple contributes a non-empty suffix carrying all four formatted
    values; the legacy part of the string is left untouched either way.

    Validates: Requirements 6.4, 6.5
    """
    lambda_cf, rho, source, mca = ccr
    template = _raw_templates()["run"]
    resolved_cfg = _with_ccr(cfg_values, ccr)

    legacy = _resolve(_pre_feature_template(template), resolved_cfg)
    current = _resolve(template, resolved_cfg)

    assert current.startswith(legacy)
    tag = current[len(legacy) :]
    assert tag == ccr_tag(lambda_cf, rho, source, mca)

    if _is_default(ccr):
        assert tag == ""
        return

    assert tag != ""
    for value in (_fmt_num(lambda_cf), _fmt_num(rho), str(source), _fmt_num(mca)):
        assert value in tag
    assert tag == "_cf{}_rho{}_src{}_mca{}".format(
        _fmt_num(lambda_cf), _fmt_num(rho), str(source), _fmt_num(mca)
    )


@given(ccr=ccr_tuple_strategy)
def test_tag_never_contains_a_path_separator_or_dot(ccr):
    """Feature: counterfactual-curvature-regularization, Property 2: Run naming is empty at defaults, complete otherwise, and injective

    A '.', '/' or '\\' in the tag would break run-directory derivation downstream.

    Validates: Requirements 6.4
    """
    tag = ccr_tag(*ccr)
    for char in TAG_FORBIDDEN_CHARS:
        assert char not in tag


@given(left=ccr_tuple_strategy, right=ccr_tuple_strategy)
def test_tags_are_injective_in_the_formatted_tuple(left, right):
    """Feature: counterfactual-curvature-regularization, Property 2: Run naming is empty at defaults, complete otherwise, and injective

    Distinct tuples contribute distinct tags: two tuples share a tag exactly when they
    share their formatted values. In particular two tuples differing only in
    ``ccr_action_source`` never collide, so the `logged` control and the `synthetic`
    treatment arm can never auto-resume each other.

    Validates: Requirements 6.4, 6.5
    """
    same_formatted = _formatted_key(left) == _formatted_key(right)
    assert (ccr_tag(*left) == ccr_tag(*right)) == same_formatted


@given(
    lambda_cf=positive_distinct_weight_strategy,
    rho=distinct_weight_strategy,
    mca=distinct_weight_strategy,
)
def test_action_source_alone_separates_two_arms(lambda_cf, rho, mca):
    """Feature: counterfactual-curvature-regularization, Property 2: Run naming is empty at defaults, complete otherwise, and injective

    The `logged` and `synthetic` arms differ in nothing else, so the action source has to
    be in the tag rather than only in the loss-signature guard.

    WHY ``lambda_cf > 0`` (and only ``lambda_cf``): the tag is required to be *empty* at
    the default tuple ``(0.0, 0.0, "synthetic", 0.0) == CCR_TAG_DEFAULTS``, because that is
    the byte-identical-legacy-run-directory contract of Requirement 6.5 / 3.4 --- the same
    contract ``test_legacy_pusht_run_dir_is_byte_identical_through_hydra_compose`` checks
    through Hydra. So ``"srcsynthetic" in ""`` cannot hold there, and demanding it would be
    demanding a regression, not catching one. Excluding that point is also not an exception
    carved out to make a test pass: at ``lambda_cf == 0`` the CCR path is switched off
    entirely (``VWorldModel`` sets ``self.ccr = self.lambda_cf > 0``), so there are no two
    arms to separate in the first place. The excluded point is still covered explicitly, and
    still shown to be collision-free, by
    ``test_default_tuple_yields_empty_synthetic_tag_but_still_separates`` below.

    ``rho`` and ``mca`` stay free to take 0.0: ``(0.1, 0.0, "synthetic", 0.0)`` is the real
    rho=0 perturbation control arm and it must still be separated from its ``logged``
    counterpart.

    Validates: Requirements 6.4, 6.5
    """
    assert lambda_cf > 0.0  # the regime where CCR runs; see the docstring.
    synthetic = ccr_tag(lambda_cf, rho, "synthetic", mca)
    logged = ccr_tag(lambda_cf, rho, "logged", mca)
    assert synthetic != logged
    assert "srcsynthetic" in synthetic
    assert "srclogged" in logged


def test_default_tuple_yields_empty_synthetic_tag_but_still_separates():
    """The default tuple, the one point excluded from the property above (Requirement 6.5).

    At ``CCR_TAG_DEFAULTS`` the synthetic tag is empty --- that *is* the byte-identical
    legacy run directory --- while the ``logged`` counterpart is non-empty. The two still
    differ, so the run-directory collision the property above guards against does not exist
    at the default tuple either; the arm-separation guarantee holds there for a different
    reason (empty vs non-empty) than in the ``lambda_cf > 0`` regime.

    Validates: Requirements 6.4, 6.5
    """
    lambda_cf, rho, source, mca = CCR_TAG_DEFAULTS
    assert (lambda_cf, source) == (0.0, "synthetic")

    synthetic = ccr_tag(lambda_cf, rho, "synthetic", mca)
    logged = ccr_tag(lambda_cf, rho, "logged", mca)

    assert synthetic == ""
    assert logged != ""
    assert "srclogged" in logged
    assert synthetic != logged


@given(tuples=st.lists(distinct_ccr_tuple_strategy, min_size=2, max_size=12))
def test_injectivity_over_a_generated_set(tuples):
    """Feature: counterfactual-curvature-regularization, Property 2: Run naming is empty at defaults, complete otherwise, and injective

    Over a generated set of tuples, the number of distinct tags equals the number of
    distinct tuples: no two pilot arms can collide on a run directory.

    Validates: Requirements 6.4, 6.5
    """
    distinct = {_formatted_key(t) for t in tuples}
    tags = {ccr_tag(*t) for t in tuples}
    assert len(tags) == len(distinct)


# ---------------------------------------------------------------------------
# CRITICAL non-property assertion: the real Hydra run.dir at the PushT target cell
# ---------------------------------------------------------------------------


def test_legacy_pusht_run_dir_is_byte_identical_through_hydra_compose():
    """Requirement 3.4: the PushT target-cell run directory is unchanged, byte for byte.

    Resolved through Hydra's compose API so this checks the real ``hydra.run.dir``
    expression (launcher override, config groups and all), not the resolver in isolation.

    Verified on the pod: hydra is not installed in the Windows dev environment.
    """
    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra

    overrides = [
        "env=pusht",
        "encoder=dino_channel",
        "training.straighten=aggcos1e-1",
        "training.encoder_lr=1e-5",
        "training.stop_grad=True",
    ]

    GlobalHydra.instance().clear()
    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(
            config_name="train.yaml", overrides=overrides, return_hydra_config=True
        )
        run_dir = str(cfg.hydra.run.dir)
        sweep_dir = str(cfg.hydra.sweep.dir)
        ckpt_base_path = str(cfg.ckpt_base_path)

        # All four CCR keys sit at their defaults, so nothing may be appended.
        assert (float(cfg.training.lambda_cf), float(cfg.training.ccr_rho),
                str(cfg.training.ccr_action_source),
                float(cfg.training.mca_weight)) == CCR_TAG_DEFAULTS

    assert run_dir == f"{ckpt_base_path}/test/{LEGACY_PUSHT_RUN_NAME}"
    assert run_dir.endswith(LEGACY_PUSHT_RUN_NAME)
    assert sweep_dir == run_dir


def test_worked_example_table_from_design_section_6():
    """The design section 6 worked-example table, row for row (Requirements 6.4, 6.5)."""
    assert ccr_tag(*CCR_TAG_DEFAULTS) == ""
    assert ccr_tag(0.1, 0.05, "synthetic", 0.0) == "_cf0p1_rho0p05_srcsynthetic_mca0"
    assert ccr_tag(0.1, 0.05, "logged", 0.0) == "_cf0p1_rho0p05_srclogged_mca0"
    assert ccr_tag(0.1, 0.0, "synthetic", 0.0) == "_cf0p1_rho0_srcsynthetic_mca0"
    assert ccr_tag(0.1, 0.05, "synthetic", 0.01) == "_cf0p1_rho0p05_srcsynthetic_mca0p01"


def test_ccr_tag_interpolation_is_appended_to_both_run_naming_expressions():
    """Both ``hydra.run.dir`` and ``hydra.sweep.dir`` end with the tag interpolation."""
    templates = _raw_templates()
    for name, template in templates.items():
        assert template.endswith(CCR_TAG_INTERPOLATION), name
    assert _pre_feature_template(templates["run"]) == _pre_feature_template(
        templates["sweep"]
    )


def test_resolver_module_registers_ccr_tag():
    """`import custom_resolvers` is what makes ``${ccr_tag:...}`` resolvable."""
    assert custom_resolvers.ccr_tag is ccr_tag
    cfg = OmegaConf.create({"tag": "${ccr_tag:0.1,0.05,synthetic,0.0}"})
    assert str(cfg.tag) == "_cf0p1_rho0p05_srcsynthetic_mca0"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__]))
