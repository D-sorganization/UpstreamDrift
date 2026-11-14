"""Regression and contract tests for UD #15014c (mocap conventions & guards)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ANTHRO_GUIDE = _REPO_ROOT / "docs" / "user_guide" / "anthropometrics.md"
_CONV_GUIDE = (
    _REPO_ROOT / "docs" / "user_guide" / "pose_studio" / "cross_engine_conventions.md"
)


def test_de_leva_verify_local_module_rejects_external_checkout(
    tmp_path: Path,
) -> None:
    """C11: DeLevaEstimator must reject humanoid_character_builder outside UD."""
    from anthropometrics.estimators import from_de_leva
    from humanoid_character_builder.core import anthropometry

    assert hasattr(from_de_leva, "_verify_local_anthropometry_module")

    verified = from_de_leva._verify_local_anthropometry_module(anthropometry)
    assert verified is anthropometry
    assert sum(
        anthropometry.DE_LEVA_DATA.male[class_id].mass_ratio
        for class_id in anthropometry._SEGMENT_NAME_MAP.values()
    ) == pytest.approx(1.0, abs=1e-3)

    foreign_file = (
        tmp_path
        / "Tools"
        / "src"
        / "shared"
        / "python"
        / "humanoid_character_builder"
        / "core"
        / "anthropometry.py"
    )
    foreign_mod = SimpleNamespace(__file__=str(foreign_file))
    with pytest.raises(ImportError, match="src/shared/python"):
        from_de_leva._verify_local_anthropometry_module(foreign_mod)

    no_file_mod = SimpleNamespace(__file__=None)
    with pytest.raises(ImportError, match="src/shared/python"):
        from_de_leva._verify_local_anthropometry_module(no_file_mod)


def test_anthropometrics_guide_accuracy_and_c5_disclosure() -> None:
    """C3/C5: Fix Zatsiorsky source text and disclose model-geometry lengths."""
    text = _ANTHRO_GUIDE.read_text(encoding="utf-8")
    lower = text.lower()

    assert "cadaver-derived" not in lower
    assert "gamma" in lower
    assert "living" in lower
    assert "drillis" in lower or "winter" in lower
    assert "humanoid_character_builder" in text
    assert "endpoint" in lower
    assert "model-geometry" in lower or "model geometry" in lower


def test_drake_adapter_and_conventions_clarify_free_base_euler_vs_rpy() -> None:
    """C8: Distinguish intrinsic-XYZ revolute chain from Drake RollPitchYaw and DrakeNativeState."""
    from src.shared.python.pose_interchange.adapters import drake

    adapter_doc = drake.__doc__ or ""
    assert "R = Rx * Ry * Rz" in adapter_doc
    assert "RollPitchYaw" in adapter_doc
    assert "R = Rz * Ry * Rx" in adapter_doc
    assert "DrakeNativeState" in adapter_doc
    assert "[qw, qx, qy, qz, x, y, z]" in adapter_doc

    guide = _CONV_GUIDE.read_text(encoding="utf-8")
    assert "R = Rx * Ry * Rz" in guide
    assert "RollPitchYaw" in guide
    assert "R = Rz * Ry * Rx" in guide
    assert "DrakeNativeState" in guide
    assert "[qw, qx, qy, qz, x, y, z]" in guide


def test_filter_default_cutoff_documents_gait_origin_and_golf_guidance() -> None:
    """C10: apply_filter documents that the 6 Hz default attenuates downswing content."""
    from src.shared.python.motion_pipeline.preprocessing.filter import apply_filter

    doc = (apply_filter.__doc__ or "").lower()
    assert "gait" in doc
    assert "downswing" in doc
    assert "12" in doc and "20" in doc
