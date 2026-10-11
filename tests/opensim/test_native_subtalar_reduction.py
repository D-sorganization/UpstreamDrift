"""Native OpenSim proof of the exact bilateral zero-subtalar derivation."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re

import pytest

pytestmark = pytest.mark.integration
opensim = pytest.importorskip("opensim")

from src.engines.physics_engines.opensim.python.native_subtalar_reduction import (  # noqa: E402
    ZeroSubtalarReductionRequest,
    derive_zero_subtalar_model,
)


_SOURCE_SHA = "8d4d349747060efbac31c707c4e01f452746a8fba32be77a425b5ac5be1d34a3"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source() -> Path:
    text = os.environ.get("UD_BUET_HAMNER_ASSEMBLED_SOURCE")
    if text is None:
        pytest.skip("set UD_BUET_HAMNER_ASSEMBLED_SOURCE to the retained v4 artifact")
    path = Path(text)
    assert _sha(path) == _SOURCE_SHA
    return path


def _request(source: Path, derived: Path) -> ZeroSubtalarReductionRequest:
    return ZeroSubtalarReductionRequest(
        source,
        _sha(source),
        derived,
        (("subtalar_angle_r", 0.0), ("subtalar_angle_l", 0.0)),
    )


def test_exact_557_muscle_source_preserves_sampled_mechanics(tmp_path: Path) -> None:
    source = _source()
    derived = tmp_path / "zero-subtalar.osim"

    receipt = derive_zero_subtalar_model(_request(source, derived))

    assert _sha(derived) == receipt.derived_sha256
    assert receipt.source_sha256 == _SOURCE_SHA
    assert receipt.removed_coordinate_names == ("subtalar_angle_r", "subtalar_angle_l")
    assert receipt.observed_pose_count >= 3
    assert (
        receipt.mobility_lift_rank == opensim.Model(str(derived)).initSystem().getNU()
    )
    for error in (
        receipt.max_body_transform_error,
        receipt.max_body_velocity_error,
        receipt.max_muscle_length_error,
        receipt.max_muscle_speed_error,
        receipt.max_projected_mass_error,
        receipt.max_projected_total_force_error,
        receipt.max_full_lift_mass_error,
        receipt.max_full_lift_force_error,
        receipt.max_constraint_residual,
    ):
        assert error < 1e-10
    assert opensim.Model(str(derived)).getMuscles().getSize() == 557


def test_nonzero_source_default_cannot_be_welded(tmp_path: Path) -> None:
    source = _source()
    edited = tmp_path / "changed-default.osim"
    body = source.read_text(encoding="utf-8")
    changed, count = re.subn(
        r'(<Coordinate name="subtalar_angle_l">[\s\S]*?<default_value>)0(</default_value>)',
        r"\g<1>0.2\g<2>",
        body,
        count=1,
    )
    assert count == 1
    edited.write_text(changed, encoding="utf-8")

    with pytest.raises(ValueError, match="zero|target"):
        derive_zero_subtalar_model(_request(edited, tmp_path / "invalid.osim"))


def test_changed_source_bytes_rejected_before_native_load(tmp_path: Path) -> None:
    source = _source()
    copied = tmp_path / "source.osim"
    copied.write_bytes(source.read_bytes())
    request = _request(copied, tmp_path / "derived.osim")
    copied.write_bytes(copied.read_bytes() + b"\n<!-- changed provenance -->\n")

    with pytest.raises(ValueError, match="source|hash"):
        derive_zero_subtalar_model(request)


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("<locked>false</locked>", "locked"),
        ("<coefficients> 1.01 0</coefficients>", "law"),
    ],
)
def test_rehashed_invalid_native_joint_is_rejected(
    tmp_path: Path, mutation: str, reason: str
) -> None:
    source = _source()
    edited = tmp_path / "invalid-joint.osim"
    body = source.read_text(encoding="utf-8")
    if "locked" in mutation:
        pattern = r'(<Coordinate name="subtalar_angle_l">[\s\S]*?)<locked>true</locked>'
        changed, count = re.subn(pattern, r"\g<1>" + mutation, body, count=1)
    else:
        pattern = r'(<CustomJoint name="subtalar_l">[\s\S]*?)<coefficients> 1 0</coefficients>'
        changed, count = re.subn(pattern, r"\g<1>" + mutation, body, count=1)
    assert count == 1
    edited.write_text(changed, encoding="utf-8")

    with pytest.raises(ValueError, match=reason):
        derive_zero_subtalar_model(_request(edited, tmp_path / "derived.osim"))
