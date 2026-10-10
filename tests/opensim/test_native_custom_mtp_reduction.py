"""Native evidence for the exact bilateral CustomJoint MTP profile."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re

import pytest

pytestmark = pytest.mark.integration
opensim = pytest.importorskip("opensim")

from src.engines.physics_engines.opensim.python.native_custom_mtp_reduction import (  # noqa: E402
    ZeroCustomMtpReductionRequest,
    derive_zero_custom_mtp_model,
)

_SOURCE_SHA = "8d4d349747060efbac31c707c4e01f452746a8fba32be77a425b5ac5be1d34a3"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source() -> Path:
    value = os.environ.get("UD_BUET_HAMNER_ASSEMBLED_SOURCE")
    if value is None:
        pytest.skip("set UD_BUET_HAMNER_ASSEMBLED_SOURCE to retained v4 source")
    source = Path(value)
    assert _sha(source) == _SOURCE_SHA
    return source


def _request(source: Path, output: Path) -> ZeroCustomMtpReductionRequest:
    return ZeroCustomMtpReductionRequest(
        source,
        _sha(source),
        output,
        (("mtp_angle_r", 0.0), ("mtp_angle_l", 0.0)),
    )


def test_exact_custom_mtp_preserves_native_mechanics(tmp_path: Path) -> None:
    source = _source()
    output = tmp_path / "custom-mtp.osim"
    receipt = derive_zero_custom_mtp_model(_request(source, output))
    reduced = opensim.Model(str(output))

    assert _sha(output) == receipt.derived_sha256
    assert receipt.removed_coordinate_names == ("mtp_angle_r", "mtp_angle_l")
    assert reduced.getMuscles().getSize() == 557
    assert receipt.mobility_lift_rank == reduced.initSystem().getNU()
    assert receipt.observed_pose_count >= 3
    assert (
        max(
            receipt.max_body_transform_error,
            receipt.max_body_velocity_error,
            receipt.max_muscle_length_error,
            receipt.max_muscle_speed_error,
            receipt.max_projected_mass_error,
            receipt.max_projected_total_force_error,
            receipt.max_full_lift_mass_error,
            receipt.max_full_lift_force_error,
            receipt.max_constraint_residual,
        )
        < 1e-10
    )


def test_actual_abd_subtalar_parent_admits_final_mtp_step(tmp_path: Path) -> None:
    value = os.environ.get("UD_BUET_HAMNER_COMPOSED_SUBTALAR_SOURCE")
    if value is None:
        pytest.skip("set UD_BUET_HAMNER_COMPOSED_SUBTALAR_SOURCE to retained parent")
    source = Path(value)
    assert _sha(source) == (
        "8d8c3cf1dbe5e149df07b0309d508d78d2524e3c7beeaee327e24c4c43c8b81b"
    )
    receipt = derive_zero_custom_mtp_model(
        _request(source, tmp_path / "composed-mtp.osim")
    )
    assert receipt.mobility_lift_rank == 56
    assert receipt.observed_pose_count == 4
    assert receipt.max_full_lift_mass_error == 0.0
    assert receipt.max_full_lift_force_error == 0.0


@pytest.mark.parametrize(
    ("pattern", "replacement", "reason"),
    [
        (
            r'(<Coordinate name="mtp_angle_l">[\s\S]*?<default_value>)0(</default_value>)',
            r"\g<1>0.2\g<2>",
            "zero|target",
        ),
        (
            r'(<Coordinate name="mtp_angle_l">[\s\S]*?)<locked>true</locked>',
            r"\g<1><locked>false</locked>",
            "locked",
        ),
        (
            r'(<CustomJoint name="mtp_l">[\s\S]*?)<coefficients> 1 0</coefficients>',
            r"\g<1><coefficients> 1.01 0</coefficients>",
            "law",
        ),
    ],
)
def test_rehashed_invalid_custom_mtp_rejected(
    tmp_path: Path,
    pattern: str,
    replacement: str,
    reason: str,
) -> None:
    source = _source()
    edited = tmp_path / "changed.osim"
    body, count = re.subn(
        pattern, replacement, source.read_text(encoding="utf-8"), count=1
    )
    assert count == 1
    edited.write_text(body, encoding="utf-8")
    with pytest.raises(ValueError, match=reason):
        derive_zero_custom_mtp_model(_request(edited, tmp_path / "out.osim"))


def test_changed_source_bytes_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source.osim"
    source.write_bytes(_source().read_bytes())
    request = _request(source, tmp_path / "out.osim")
    source.write_bytes(source.read_bytes() + b"\n<!-- altered -->\n")
    with pytest.raises(ValueError, match="source|hash"):
        derive_zero_custom_mtp_model(request)
