"""Tests for the OpenSim ScaleTool wrapper using mocks to avoid skipping."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from src.shared.python.motion_pipeline.contracts import (
    JointDef,
    Marker,
    MarkerFrame,
    SkeletonRig,
)
from src.shared.python.motion_pipeline.scaling.opensim_scale import (
    OpenSimScaleBackend,
)


def _rig() -> SkeletonRig:
    return SkeletonRig(
        id="generic",
        joints={
            "pelvis": JointDef(
                name="pelvis",
                parent=None,
                children=["femur"],
                tpose_offset=[0.0, 0.0, 0.0],
                axes=["X"],
            ),
            "femur": JointDef(
                name="femur",
                parent="pelvis",
                children=[],
                tpose_offset=[0.0, -0.4, 0.0],
                axes=["X"],
            ),
        },
        root_joint="pelvis",
    )


def _markers() -> MarkerFrame:
    return MarkerFrame(
        timestamp=0.0,
        markers={
            "femur_prox": Marker(name="femur_prox", x=0.0, y=0.0, z=0.0),
            "femur_dist": Marker(name="femur_dist", x=0.0, y=-0.45, z=0.0),
            "pelvis_l": Marker(name="pelvis_l", x=-0.1, y=0.0, z=0.0),
            "pelvis_r": Marker(name="pelvis_r", x=0.1, y=0.0, z=0.0),
        },
    )


def test_opensim_scale_backend_initialization_validation():
    # Test valid initializations
    backend = OpenSimScaleBackend(
        mass_kg=70.0, height_m=1.75, generic_model_path="model.osim"
    )
    assert backend.mass_kg == 70.0
    assert backend.height_m == 1.75
    assert backend.generic_model_path == Path("model.osim")

    # Test invalid mass
    with pytest.raises(ValueError, match="mass_kg must be a positive finite number"):
        OpenSimScaleBackend(mass_kg=0)
    with pytest.raises(ValueError, match="mass_kg must be a positive finite number"):
        OpenSimScaleBackend(mass_kg=-10.0)
    with pytest.raises(ValueError, match="mass_kg must be a positive finite number"):
        OpenSimScaleBackend(mass_kg=float("inf"))

    # Test invalid height
    with pytest.raises(ValueError, match="height_m must be a positive finite number"):
        OpenSimScaleBackend(height_m=0)
    with pytest.raises(ValueError, match="height_m must be a positive finite number"):
        OpenSimScaleBackend(height_m=-1.80)
    with pytest.raises(ValueError, match="height_m must be a positive finite number"):
        OpenSimScaleBackend(height_m=float("nan"))


def test_scale_without_opensim_installed():
    """Verify that scale() raises RuntimeError if opensim is not installed/importable."""
    rig = _rig()
    markers = _markers()
    backend = OpenSimScaleBackend(mass_kg=70.0, height_m=1.75)

    with (
        patch.dict(sys.modules, {"opensim": None}),
        pytest.raises(RuntimeError, match="opensim not installed"),
    ):
        # In python, setting a module to None in sys.modules triggers an ImportError when importing it
        backend.scale(rig, markers, {"femur_prox": "femur"})


def test_scale_success_with_mocked_opensim():
    """Test successful scaling using a mocked opensim module."""
    rig = _rig()
    markers = _markers()
    marker_to_segment = {
        "femur_prox": "femur",
        "femur_dist": "femur",
        "pelvis_l": "pelvis",
        "pelvis_r": "pelvis",
    }

    mock_opensim = MagicMock()
    mock_scale_tool = MagicMock()
    mock_opensim.ScaleTool.return_value = mock_scale_tool

    with patch.dict(sys.modules, {"opensim": mock_opensim}):
        backend = OpenSimScaleBackend(
            mass_kg=70.0, height_m=1.75, generic_model_path="generic.osim"
        )
        scaled = backend.scale(rig, markers, marker_to_segment)

        # Verify calls to ScaleTool
        mock_opensim.ScaleTool.assert_called_once()
        mock_scale_tool.setName.assert_called_once_with("generic-scale")
        mock_scale_tool.setSubjectMass.assert_called_once_with(70.0)
        mock_scale_tool.setSubjectHeight.assert_called_once_with(1750.0)

        # Verify scaled rig properties
        assert scaled.id == "generic-scaled"
        assert scaled.metadata["scaled_by"] == "opensim"

        # pelvis is root so it doesn't change from 0.0 offset. femur should have a scaled offset based on markers.
        # femur_prox = [0,0,0], femur_dist = [0,-0.45,0], so distance is 0.45
        femur_offset = scaled.joints["femur"].tpose_offset
        assert np.linalg.norm(femur_offset) == pytest.approx(0.45)


def test_scale_validation_parameters():
    """Test input parameter validation in scale()."""
    backend = OpenSimScaleBackend(mass_kg=70.0, height_m=1.75)
    rig = _rig()
    markers = _markers()
    marker_to_segment = {"femur_prox": "femur"}

    with pytest.raises(ValueError, match="rig must be provided"):
        backend.scale(None, markers, marker_to_segment)

    with pytest.raises(ValueError, match="calibration_markers must contain markers"):
        backend.scale(rig, MarkerFrame(timestamp=0.0, markers={}), marker_to_segment)

    with pytest.raises(ValueError, match="marker_to_segment must be provided"):
        backend.scale(rig, markers, None)


def test_scale_zero_or_nan_length_fallback():
    """Test fallback when current offset is zero or length is non-finite/negative."""
    # A rig where the femur tpose_offset is zero.
    rig = SkeletonRig(
        id="zero_offset",
        joints={
            "pelvis": JointDef(
                name="pelvis",
                parent=None,
                children=["femur"],
                tpose_offset=[0.0, 0.0, 0.0],
                axes=["X"],
            ),
            "femur": JointDef(
                name="femur",
                parent="pelvis",
                children=[],
                tpose_offset=[0.0, 0.0, 0.0],
                axes=["X"],
            ),
        },
        root_joint="pelvis",
    )
    markers = _markers()
    # Marker to segment has no valid pairings for femur, so segment length defaults to 1.0 (or current if valid)
    marker_to_segment = {}

    mock_opensim = MagicMock()
    with patch.dict(sys.modules, {"opensim": mock_opensim}):
        backend = OpenSimScaleBackend(mass_kg=70.0, height_m=1.75)
        scaled = backend.scale(rig, markers, marker_to_segment)

        # For femur, current offset norm is 0.0, target is default 1.0.
        # Since scale is 0.0 (current <= 1e-9), it falls back to new_offset = [target, 0.0, 0.0] = [1.0, 0.0, 0.0]
        assert scaled.joints["femur"].tpose_offset == [1.0, 0.0, 0.0]


def test_scale_nan_coordinates_fallback():
    """Test that nan/infinite marker coordinates fallback gracefully to default/previous lengths."""
    rig = _rig()
    markers = MarkerFrame(
        timestamp=0.0,
        markers={
            "femur_prox": Marker.model_construct(
                name="femur_prox", x=float("nan"), y=0.0, z=0.0
            ),
            "femur_dist": Marker(name="femur_dist", x=0.0, y=-0.45, z=0.0),
        },
    )
    marker_to_segment = {"femur_prox": "femur", "femur_dist": "femur"}

    mock_opensim = MagicMock()
    with patch.dict(sys.modules, {"opensim": mock_opensim}):
        backend = OpenSimScaleBackend(mass_kg=70.0, height_m=1.75)
        scaled = backend.scale(rig, markers, marker_to_segment)
        # target becomes NaN, fallback uses current offset (0.4)
        assert np.linalg.norm(scaled.joints["femur"].tpose_offset) == pytest.approx(0.4)


def test_scale_non_positive_postcondition_raises():
    """Verify that scale() raises RuntimeError if a non-root joint has non-positive length in postcondition check."""
    rig = _rig()
    markers = _markers()
    mock_opensim = MagicMock()
    bad_rig = SkeletonRig(
        id="bad",
        joints={
            "pelvis": JointDef(
                name="pelvis",
                parent=None,
                children=["femur"],
                tpose_offset=[0.0, 0.0, 0.0],
                axes=["X"],
            ),
            "femur": JointDef(
                name="femur",
                parent="pelvis",
                children=[],
                tpose_offset=[0.0, 0.0, 0.0],
                axes=["X"],
            ),
        },
        root_joint="pelvis",
    )
    with patch.dict(sys.modules, {"opensim": mock_opensim}):
        backend = OpenSimScaleBackend(mass_kg=70.0, height_m=1.75)
        # Patch _apply_lengths_to_rig to return a rig with zero offset for femur
        with (
            patch.object(OpenSimScaleBackend, "_apply_lengths_to_rig") as mock_apply,
            pytest.raises(
                RuntimeError,
                match="Scaled rig has non-positive segment length for femur",
            ),
        ):
            mock_apply.return_value = bad_rig
            backend.scale(rig, markers, {})


def test_scale_additional_branches():
    """Test remaining branch coverage in opensim_scale.py."""
    rig = _rig()
    markers = _markers()
    marker_to_segment = {
        "femur_prox": "single_marker_seg",  # segment with 1 marker (line 118->117)
        "femur_dist": "femur",
        "pelvis_l": "pelvis",
        "pelvis_r": "pelvis",
        "missing_marker": "femur",  # missing marker key (line 113->112)
    }

    mock_opensim = MagicMock()
    mock_scale_tool = MagicMock()
    mock_opensim.ScaleTool.return_value = mock_scale_tool

    # Mock Path.exists to return True for out_osim
    with (
        patch.dict(sys.modules, {"opensim": mock_opensim}),
        patch(
            "src.shared.python.motion_pipeline.scaling.opensim_scale.Path.exists",
            return_value=True,
        ),
    ):
        backend = OpenSimScaleBackend(
            mass_kg=70.0, height_m=1.75, generic_model_path="generic.osim"
        )
        scaled = backend.scale(rig, markers, marker_to_segment)
        assert scaled.joints["femur"].tpose_offset is not None
