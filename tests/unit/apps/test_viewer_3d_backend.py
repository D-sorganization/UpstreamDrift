"""Tests for C3D 3D-viewer renderer backend selection."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MODULE_PATH = (
    _REPO_ROOT
    / "src"
    / "engines"
    / "Simscape_Multibody_Models"
    / "3D_Golf_Model"
    / "python"
    / "src"
    / "apps"
    / "ui"
    / "tabs"
    / "viewer_3d_backend.py"
)

_SPEC = importlib.util.spec_from_file_location("viewer_3d_backend", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

FPS_TARGET = _MODULE.FPS_TARGET
PARITY_FEATURES = _MODULE.PARITY_FEATURES
RendererBackend = _MODULE.RendererBackend
Viewer3DBackendDecision = _MODULE.Viewer3DBackendDecision
select_viewer_3d_backend = _MODULE.select_viewer_3d_backend


def test_selects_pyqtgl_when_gpu_preferred_and_available() -> None:
    decision = select_viewer_3d_backend(
        prefer_gpu=True,
        pyqtgl_available=True,
        marker_count=45,
        frame_count=1000,
    )

    assert decision.backend is RendererBackend.PYQTGL
    assert decision.target_fps == FPS_TARGET
    assert decision.reason == "pyqtgraph.opengl available"


def test_falls_back_to_matplotlib_when_optional_dependency_missing() -> None:
    decision = select_viewer_3d_backend(
        prefer_gpu=True,
        pyqtgl_available=False,
        marker_count=45,
        frame_count=1000,
    )

    assert decision.backend is RendererBackend.MATPLOTLIB
    assert "pyqtgraph.opengl unavailable" in decision.reason


def test_respects_explicit_matplotlib_preference() -> None:
    decision = select_viewer_3d_backend(
        prefer_gpu=False,
        pyqtgl_available=True,
        marker_count=45,
        frame_count=1000,
    )

    assert decision.backend is RendererBackend.MATPLOTLIB
    assert decision.reason == "matplotlib fallback explicitly requested"


def test_decision_rejects_invalid_dataset_shape() -> None:
    with pytest.raises(ValueError, match="marker_count"):
        select_viewer_3d_backend(
            prefer_gpu=True,
            pyqtgl_available=True,
            marker_count=0,
            frame_count=1000,
        )

    with pytest.raises(ValueError, match="frame_count"):
        select_viewer_3d_backend(
            prefer_gpu=True,
            pyqtgl_available=True,
            marker_count=45,
            frame_count=0,
        )


def test_backend_decision_contract_is_complete() -> None:
    decision = Viewer3DBackendDecision(
        backend=RendererBackend.PYQTGL,
        reason="available",
        target_fps=60,
        marker_count=45,
        frame_count=1000,
        parity_features=PARITY_FEATURES,
    )

    assert set(decision.parity_features) == {
        "scrubbing",
        "speed_control",
        "loop",
        "marker_groups",
        "view_presets",
        "skeleton_overlay",
    }
