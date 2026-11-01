"""Tests for src.engines.physics_engines.pinocchio.python.pinocchio_golf.manipulability (HCM-4, #14005)."""

from unittest.mock import MagicMock
import pytest

from src.engines.physics_engines.pinocchio.python.pinocchio_golf.manipulability import (
    EllipsoidParams,
    ManipulabilityResult,
    PinocchioManipulabilityAnalyzer,
)

pytestmark = pytest.mark.unit


def test_analyzer_is_clearly_labeled_kinematic_only() -> None:
    """PinocchioManipulabilityAnalyzer is explicitly labeled as kinematic-only (HCM-4, #14005)."""
    assert PinocchioManipulabilityAnalyzer.is_kinematic_only is True
    model = MagicMock()
    data = MagicMock()
    analyzer = PinocchioManipulabilityAnalyzer(model, data)
    assert analyzer.is_dynamic is False


def test_analyzer_docstrings_reference_operational_space() -> None:
    """Docstrings distinguish kinematic manipulability from operational-space mobility."""
    from src.engines.physics_engines.pinocchio.python.pinocchio_golf import (
        manipulability,
    )

    assert "kinematic" in manipulability.__doc__.lower()
    assert "operational-space" in manipulability.__doc__.lower()
    assert "pinocchiocapabilitysource" in manipulability.__doc__.lower()
    assert "kinematic" in PinocchioManipulabilityAnalyzer.__doc__.lower()


def test_analyzer_requires_model() -> None:
    """PinocchioManipulabilityAnalyzer raises ValueError if model is None."""
    with pytest.raises(ValueError, match="model must be provided"):
        PinocchioManipulabilityAnalyzer(None, MagicMock())


def test_find_potential_bodies() -> None:
    """find_potential_bodies filters frames matching candidates."""
    model = MagicMock()
    f1 = MagicMock()
    f1.name = "lead_wrist"
    f2 = MagicMock()
    f2.name = "trail_hand"
    f3 = MagicMock()
    f3.name = "pelvis"
    model.frames = [f1, f2, f3]
    analyzer = PinocchioManipulabilityAnalyzer(model, MagicMock())
    candidates = analyzer.find_potential_bodies()
    assert "lead_wrist" in candidates
    assert "trail_hand" in candidates
    assert "pelvis" not in candidates
