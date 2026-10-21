"""Unit checks for the shared gravity-only URDF/MJCF fixture."""

from __future__ import annotations

from pathlib import Path

import pytest

from gravity_equivalence_fixture import write_free_fall_mjcf, write_free_fall_urdf

_CANONICAL_GOLFER_URDF = (
    Path(__file__).resolve().parents[3]
    / "src/engines/physics_engines/drake/models/generated/golfer.urdf"
)


@pytest.mark.unit
def test_write_free_fall_mjcf_compiles_canonical_golfer_urdf(tmp_path: Path) -> None:
    """Generated golfer.urdf has no explicit pelvis_floating joint at rest."""
    pytest.importorskip("mujoco")
    assert _CANONICAL_GOLFER_URDF.is_file(), "canonical golfer URDF must exist on disk"
    urdf = write_free_fall_urdf(tmp_path)
    xml_path, nu = write_free_fall_mjcf(urdf)
    assert xml_path.is_file()
    assert nu > 0
