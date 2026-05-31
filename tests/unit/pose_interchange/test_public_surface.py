"""Public surface tests for :mod:`pose_interchange`."""

from __future__ import annotations

import src.shared.python.pose_interchange as pose_interchange


def test_public_surface_documents_canonical_v2_contract() -> None:
    """The module docs point consumers to the additive dynamic-state contract."""

    module_docs = pose_interchange.__doc__ or ""

    assert "docs/adr/0012-canonical-pose-interchange.md" in module_docs
    assert "docs/conventions/canonical-v2.md" in module_docs


def test_public_surface_keeps_canonical_v1_exports() -> None:
    """Documenting canonical-v2 must not remove canonical-v1 API exports."""

    exported_names = set(pose_interchange.__all__)

    assert "CanonicalPose" in exported_names
    assert "PoseConventionAdapter" in exported_names
    assert "canonical_zero_pose" in exported_names
    assert pose_interchange.SCHEMA_VERSION == "1.0.0"
    assert pose_interchange.__version__ == "1.0.0"
