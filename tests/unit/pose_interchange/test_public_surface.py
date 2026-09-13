"""Public surface tests for :mod:`pose_interchange`."""

from __future__ import annotations

import pytest

import src.shared.python.pose_interchange as pose_interchange

pytestmark = pytest.mark.unit


def test_public_surface_documents_canonical_v2_contract() -> None:
    """The module docs point consumers to the additive dynamic-state contract."""

    module_docs = pose_interchange.__doc__ or ""

    assert "docs/adr/0012-canonical-pose-interchange.md" in module_docs
    assert "docs/conventions/canonical-v2.md" in module_docs


def test_public_surface_keeps_canonical_v1_exports_and_adds_v2() -> None:
    """Documenting canonical-v2 must preserve v1 and expose v2 additions."""

    exported_names = set(pose_interchange.__all__)

    assert "CanonicalPose" in exported_names
    assert "PoseConventionAdapter" in exported_names
    assert "canonical_zero_pose" in exported_names
    assert "CONVENTION_TAG_V2" in exported_names
    assert "CanonicalState" in exported_names
    assert "canonical_state_zero" in exported_names
    assert pose_interchange.SCHEMA_VERSION == "1.0.0"
    assert pose_interchange.__version__ == "2.1.0"


def test_public_surface_exposes_native_joint_and_frame_transport() -> None:
    assert {
        "SerialRotationChart",
        "SingularChartError",
        "FixedFrameTransport",
        "NativeJointStateAdapter",
        "NativeManifoldState",
    } <= set(pose_interchange.__all__)


def test_native_numeric_import_does_not_require_reference_pose_application() -> None:
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    script = """
import importlib.abc
import sys
class BlockApplication(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'src.shared.python.motion_matching.diagnostics.reference_pose':
            raise ImportError('Application reference pose unavailable in numeric runtime')
sys.meta_path.insert(0, BlockApplication())
from src.shared.python.pose_interchange import SerialRotationChart, NativeJointStateAdapter
assert SerialRotationChart('XYZ').quaternion([0, 0, 0])[0] == 1
assert NativeJointStateAdapter.__name__ == 'NativeJointStateAdapter'
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_every_public_export_remains_resolvable_and_discoverable() -> None:
    for name in pose_interchange.__all__:
        assert name in dir(pose_interchange)
        assert getattr(pose_interchange, name) is not None
    with pytest.raises(AttributeError):
        _ = pose_interchange.nonexistent_public_symbol


def test_public_surface_exposes_native_motion_sequence() -> None:
    from src.shared.python.pose_interchange.native_motion_sequence import (
        NativeMotionSequence,
        export_native_motion,
        restore_native_motion,
    )

    for name, expected in (
        ("NativeMotionSequence", NativeMotionSequence),
        ("export_native_motion", export_native_motion),
        ("restore_native_motion", restore_native_motion),
    ):
        assert name in pose_interchange.__all__
        assert getattr(pose_interchange, name) is expected
