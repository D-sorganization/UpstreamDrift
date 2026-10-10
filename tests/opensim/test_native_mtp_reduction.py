"""Native, source-bound evidence for zero-target bilateral MTP reduction."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

opensim = pytest.importorskip("opensim")

from src.engines.physics_engines.opensim.python import native_mtp_reduction as reduction  # noqa: E402
from src.engines.physics_engines.opensim.python.native_mtp_reduction import (  # noqa: E402
    ZeroMtpReductionRequest,
    _native_total_generalized_force,
    derive_zero_mtp_model,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(
    path: Path,
    *,
    left_target: float = 0.0,
    left_speed: float = 0.0,
    left_locked: bool = True,
    dependent: bool = False,
) -> None:
    model = opensim.Model()
    parent = model.getGround()
    for side, angle in (("r", 0.0), ("l", left_target)):
        body = opensim.Body(f"toe_{side}", 1.0, opensim.Vec3(0), opensim.Inertia(0.1))
        model.addBody(body)
        joint = opensim.PinJoint(
            f"mtp_{side}",
            parent,
            opensim.Vec3(0.1, 0, 0),
            opensim.Vec3(0),
            body,
            opensim.Vec3(0, 0.1, 0),
            opensim.Vec3(0),
        )
        model.addJoint(joint)
        coord = joint.updCoordinate()
        coord.setName(f"mtp_angle_{side}")
        coord.setDefaultValue(angle)
        coord.setDefaultSpeedValue(left_speed if side == "l" else 0.0)
        coord.setDefaultLocked(left_locked if side == "l" else True)
        parent = body
    if dependent:
        actuator = opensim.CoordinateActuator("mtp_angle_r")
        actuator.setName("forbidden_mtp_motor")
        model.addForce(actuator)
    model.initSystem()
    model.printToXML(str(path))


def _request(source: Path, output: Path) -> ZeroMtpReductionRequest:
    return ZeroMtpReductionRequest(
        source_model_path=source,
        source_sha256=_sha(source),
        derived_model_path=output,
        declared_target_rad=(("mtp_angle_r", 0.0), ("mtp_angle_l", 0.0)),
    )


def test_fresh_native_zero_target_reduction_preserves_geometry_and_mass(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.osim"
    output = tmp_path / "derived.osim"
    _fixture(source)

    receipt = derive_zero_mtp_model(_request(source, output))

    assert output.is_file()
    assert receipt.source_sha256 == _sha(source)
    assert receipt.derived_sha256 == _sha(output)
    assert receipt.removed_coordinate_names == ("mtp_angle_r", "mtp_angle_l")
    assert receipt.max_body_transform_error < 1e-12
    assert receipt.max_projected_mass_error < 1e-12
    assert receipt.max_projected_total_force_error < 1e-12
    assert receipt.max_body_velocity_error < 1e-12
    assert receipt.max_full_lift_mass_error < 1e-12
    assert receipt.max_full_lift_force_error < 1e-12
    assert receipt.native_reload_verified
    assert (
        opensim.Model(str(output)).initSystem().getNY()
        < opensim.Model(str(source)).initSystem().getNY()
    )


def test_native_total_force_includes_body_applied_gravity() -> None:
    model = opensim.Model()
    body = opensim.Body("offset_mass", 2.0, opensim.Vec3(1, 0, 0), opensim.Inertia(0.1))
    model.addBody(body)
    model.addJoint(opensim.PinJoint("pin", model.getGround(), body))
    state = model.initSystem()
    model.realizeDynamics(state)

    assert model.getMobilityForces(state).get(0) == 0.0
    assert abs(_native_total_generalized_force(model, state)[0]) > 10.0


def test_nonfinite_native_observation_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.osim"
    _fixture(source)
    monkeypatch.setattr(reduction, "_transform", lambda *_: np.full(12, np.nan))

    with pytest.raises(ValueError, match="nonfinite"):
        derive_zero_mtp_model(_request(source, tmp_path / "derived.osim"))


def test_nonzero_native_lock_target_is_not_silently_welded(tmp_path: Path) -> None:
    source = tmp_path / "nonzero.osim"
    _fixture(source, left_target=0.3)

    with pytest.raises(ValueError, match="zero|target"):
        derive_zero_mtp_model(_request(source, tmp_path / "derived.osim"))


@pytest.mark.parametrize("left_speed,left_locked", [(0.1, True), (0.0, False)])
def test_mtp_must_be_fixed_at_zero_speed(
    tmp_path: Path, left_speed: float, left_locked: bool
) -> None:
    source = tmp_path / "unfixed.osim"
    _fixture(source, left_speed=left_speed, left_locked=left_locked)

    with pytest.raises(ValueError, match="locked|speed"):
        derive_zero_mtp_model(_request(source, tmp_path / "derived.osim"))


def test_removed_coordinate_dependency_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "dependent.osim"
    _fixture(source, dependent=True)

    with pytest.raises(ValueError, match="reference|dependency|actuator"):
        derive_zero_mtp_model(_request(source, tmp_path / "derived.osim"))


def test_changed_source_bytes_are_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source.osim"
    _fixture(source)
    request = _request(source, tmp_path / "derived.osim")
    source.write_bytes(source.read_bytes() + b"\n<!-- provenance mutation -->\n")

    with pytest.raises(ValueError, match="hash|source"):
        derive_zero_mtp_model(request)


def test_declared_target_must_cover_exact_bilateral_pair(tmp_path: Path) -> None:
    source = tmp_path / "source.osim"
    _fixture(source)
    request = _request(source, tmp_path / "derived.osim")

    with pytest.raises(ValueError, match="bilateral|target"):
        derive_zero_mtp_model(
            ZeroMtpReductionRequest(
                source,
                request.source_sha256,
                request.derived_model_path,
                (("mtp_angle_r", 0.0),),
            )
        )


@pytest.mark.integration
def test_exact_rajagopal_native_source_reduces_without_muscle_loss(
    tmp_path: Path,
) -> None:
    source_text = os.environ.get("UD_RAJAGOPAL_SOURCE")
    if source_text is None:
        pytest.skip("set UD_RAJAGOPAL_SOURCE to the retained native donor model")
    source = Path(source_text)
    expected = "8708ed0d6a212080a72b9c277071ddd12939c8306efceb3ee4a9592735f43f80"
    assert _sha(source) == expected

    receipt = derive_zero_mtp_model(
        _request(source, tmp_path / "rajagopal_derived.osim")
    )

    assert receipt.max_body_transform_error < 1e-10
    assert receipt.max_muscle_length_error < 1e-10
    assert receipt.max_projected_mass_error < 1e-10
    assert receipt.max_projected_total_force_error < 1e-10
    assert receipt.max_body_velocity_error < 1e-10
    assert receipt.max_full_lift_mass_error < 1e-10
    assert receipt.max_full_lift_force_error < 1e-10
    assert receipt.mobility_lift_rank > 0
    assert receipt.observed_pose_count == 3
    original = opensim.Model(str(source))
    derived = opensim.Model(str(tmp_path / "rajagopal_derived.osim"))
    assert original.getMuscles().getSize() == derived.getMuscles().getSize() == 80


@pytest.mark.integration
def test_scaled_club_factory_is_a_distinct_derived_comparator(tmp_path: Path) -> None:
    source_text = os.environ.get("UD_RAJAGOPAL_SOURCE")
    if source_text is None:
        pytest.skip("set UD_RAJAGOPAL_SOURCE to the retained native donor model")
    from src.engines.physics_engines.opensim.python.musculoskeletal_swing import (
        build_musculoskeletal_model,
    )

    golf_model = (
        Path(__file__).parents[2]
        / "src/engines/physics_engines/opensim/models/golf_humanoid_scaled.osim"
    )
    model, _info = build_musculoskeletal_model(
        golf_model, base_model_path=Path(source_text)
    )
    source = tmp_path / "scaled_club_source.osim"
    model.printToXML(str(source))

    receipt = derive_zero_mtp_model(
        _request(source, tmp_path / "scaled_club_mtp_derived.osim")
    )

    assert receipt.max_body_transform_error < 1e-10
    assert receipt.max_muscle_length_error < 1e-10
    assert receipt.max_projected_mass_error < 1e-10
    assert receipt.max_projected_total_force_error < 1e-10
    assert receipt.max_body_velocity_error < 1e-10
    assert receipt.max_full_lift_mass_error < 1e-10
    assert receipt.max_full_lift_force_error < 1e-10
    assert receipt.mobility_lift_rank > 0
    assert receipt.observed_pose_count == 3
