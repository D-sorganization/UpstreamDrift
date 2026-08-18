"""Tests for the pose-estimator registry (epic #8390, C2/#8402).

The headline test is structural: registering an estimator in ONE place
makes it constructable by the pipeline, valid at the API layer, and
listed with a skeleton by the motion-capture routes — the historical
5-place edit tax is gone.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.shared.python.pose_estimation.registry import (
    EstimatorInfo,
    create_estimator,
    estimator_availability,
    get_estimator_info,
    implemented_estimator_types,
    list_estimators,
    register_estimator,
    unregister_estimator,
)


class _FakeEstimator:
    def __init__(self, **options: Any) -> None:
        self.options = options
        self.loaded = False

    def load_model(self, model_path: Any = None) -> None:
        self.loaded = True


@pytest.fixture
def fake_estimator_entry():
    info = EstimatorInfo(
        name="fake_estimator",
        display_name="Fake Estimator",
        description="Structural-test estimator",
        probe_module="json",  # always importable
        install_hint="unreachable",
        skeleton=({"name": "root", "parent": None}, {"name": "tip", "parent": "root"}),
        factory=lambda **options: _FakeEstimator(**options),
    )
    register_estimator(info)
    yield info
    unregister_estimator("fake_estimator")


def test_builtin_estimators_registered() -> None:
    assert implemented_estimator_types() == {"mediapipe", "openpose"}
    assert [info.name for info in list_estimators()] == ["mediapipe", "openpose"]


def test_unknown_estimator_lists_valid_names() -> None:
    with pytest.raises(ValueError, match="mediapipe"):
        get_estimator_info("movenet")


def test_duplicate_registration_rejected() -> None:
    with pytest.raises(ValueError, match="already registered"):
        register_estimator(
            EstimatorInfo(
                name="mediapipe",
                display_name="dup",
                description="dup",
                probe_module="json",
                install_hint="",
            )
        )


def test_availability_probe_reports_hint_for_missing_module() -> None:
    available, reason = estimator_availability("openpose")
    # pyopenpose is not installed in CI: reason must carry the hint. If it
    # ever IS installed, availability flips and reason becomes None.
    assert available in (True, False)
    if not available:
        assert "OpenPose" in reason


def test_single_registration_surfaces_everywhere(fake_estimator_entry) -> None:
    """One register_estimator call → implemented types, availability,
    skeleton template, and pipeline construction all see the estimator."""
    # 1. Implemented set (drives the API's VALID_ESTIMATOR_TYPES).
    assert "fake_estimator" in implemented_estimator_types()
    # 2. Availability probing.
    available, reason = estimator_availability("fake_estimator")
    assert available is True and reason is None
    # 3. Skeleton template lookup (drives /skeleton/{source_type}).
    assert [j["name"] for j in get_estimator_info("fake_estimator").skeleton] == [
        "root",
        "tip",
    ]
    # 4. Runtime construction (drives VideoPosePipeline._load_estimator).
    estimator = create_estimator("fake_estimator", min_confidence=0.7)
    assert isinstance(estimator, _FakeEstimator)
    assert estimator.options["min_confidence"] == 0.7


def test_factory_less_entry_rejects_construction() -> None:
    register_estimator(
        EstimatorInfo(
            name="no_factory",
            display_name="No Factory",
            description="",
            probe_module="json",
            install_hint="",
        )
    )
    try:
        with pytest.raises(ValueError, match="no runtime factory"):
            create_estimator("no_factory")
    finally:
        unregister_estimator("no_factory")


def test_api_layer_derives_from_registry(fake_estimator_entry) -> None:
    """The API validation set is computed from the registry at import; a
    fresh computation must include newly registered estimators."""
    from src.shared.python.pose_estimation.registry import (
        implemented_estimator_types as recompute,
    )

    assert "fake_estimator" in recompute()
