"""Invariants on the FEATURES table itself.

These tests don't touch probes or pip — they just check that the
metadata is internally consistent.
"""

from __future__ import annotations

import pytest

from src.shared.python.feature_registry import (
    FEATURES,
    Feature,
    all_features,
    features_for_stage,
    get_feature,
)


pytestmark = pytest.mark.unit


def test_features_have_unique_names() -> None:
    names = [f.name for f in FEATURES]
    assert len(names) == len(set(names)), "Feature names must be unique"


def test_feature_names_are_normalized() -> None:
    """Names must already be lowercase — :func:`get_feature` lowercases input."""
    for feature in FEATURES:
        assert feature.name == feature.name.lower()
        assert feature.name.strip() == feature.name


def test_get_feature_is_case_insensitive() -> None:
    feature = get_feature("MUJOCO")
    assert feature.name == "mujoco"


def test_get_feature_unknown_raises() -> None:
    with pytest.raises(KeyError, match="Unknown feature"):
        get_feature("not-a-feature")


def test_get_feature_rejects_empty() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        get_feature("   ")


def test_get_feature_rejects_non_string() -> None:
    with pytest.raises(TypeError):
        get_feature(123)  # type: ignore[arg-type]


def test_all_features_is_immutable() -> None:
    assert isinstance(all_features(), tuple)
    # ``Feature`` is frozen — mutation must raise.
    feature = all_features()[0]
    with pytest.raises((AttributeError, TypeError)):
        feature.name = "mutated"  # type: ignore[misc]


def test_features_for_stage_filters_correctly() -> None:
    mujoco_stage = features_for_stage("mujoco")
    assert any(f.name == "mujoco" for f in mujoco_stage)
    assert all(f.docker_stage == "mujoco" for f in mujoco_stage)


def test_dependencies_reference_known_features() -> None:
    known = {f.name for f in FEATURES}
    for feature in FEATURES:
        for dep in feature.depends_on:
            assert dep in known, (
                f"Feature {feature.name!r} depends on {dep!r} which is not registered"
            )


def test_pip_extra_features_are_listed_in_pyproject() -> None:
    """Smoke check: every ``pip_extra`` should match an entry in pyproject."""
    pyproject = (
        __import__("pathlib").Path(__file__).resolve().parents[3] / "pyproject.toml"
    )
    text = pyproject.read_text(encoding="utf-8")
    for feature in FEATURES:
        if feature.pip_extra is None:
            continue
        # We only check the substring — the actual TOML parser path is
        # heavier than needed and would require tomllib.
        assert f"{feature.pip_extra} = [" in text, (
            f"Feature {feature.name!r} declares pip_extra "
            f"{feature.pip_extra!r} but it is not in pyproject.toml"
        )


def test_tier_values_are_valid() -> None:
    allowed = {"core", "extended", "experimental", "tooling"}
    for feature in FEATURES:
        assert feature.tier in allowed


def test_size_estimates_are_non_negative() -> None:
    for feature in FEATURES:
        assert feature.approx_size_mb >= 0


def test_feature_is_frozen() -> None:
    feature = FEATURES[0]
    assert isinstance(feature, Feature)
    with pytest.raises((AttributeError, TypeError)):
        feature.name = "mutated"  # type: ignore[misc]
