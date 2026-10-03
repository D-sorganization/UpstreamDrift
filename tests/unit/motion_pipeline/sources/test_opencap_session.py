"""Tests for whole-session OpenCap import (#11403)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from src.shared.python.motion_pipeline.contracts import MotionTrajectory
from src.shared.python.motion_pipeline.sources.opencap_session import (
    load_opencap_session,
)
from src.shared.python.motion_pipeline.sources.osim_coordinates import (
    CoordinateKind,
    read_osim_coordinates,
)
from tests.unit.motion_pipeline.sources.opencap_fixtures import (
    KINEMATICS_COLUMNS,
    write_kinematics,
    write_opencap_session,
    write_scaled_model,
)

pytestmark = pytest.mark.unit


def _full_session(tmp_path: Path, trials: tuple[str, ...]) -> Path:
    session = write_opencap_session(tmp_path, trials)
    write_scaled_model(session)
    for trial in trials:
        write_kinematics(session, trial)
    return session


def _column(kinematics: MotionTrajectory, name: str, frame: int) -> float:
    index = KINEMATICS_COLUMNS.index(name)
    return kinematics.trajectory.frames[frame].q[index]


def test_explicit_trial_loads_model_kinematics_and_subject(tmp_path: Path) -> None:
    session_dir = _full_session(tmp_path, ("neutral", "swing1", "swing2"))

    session = load_opencap_session(session_dir, trial="swing2")

    assert session.trial == "swing2"
    assert session.observations.source_provenance["trial"] == "swing2"
    assert session.model_file is not None
    assert session.model_file.name == "LaiUhlrich2022_scaled.osim"
    assert session.subject.mass_kg == pytest.approx(79.5)
    assert session.subject.height_m == pytest.approx(1.82)
    assert session.subject.opensim_model == "LaiUhlrich2022"
    assert session.trials == ["neutral", "swing1", "swing2"]


def test_translations_stay_in_metres_and_rotations_become_radians(
    tmp_path: Path,
) -> None:
    session_dir = _full_session(tmp_path, ("swing1",))

    kinematics = load_opencap_session(session_dir).kinematics

    assert kinematics is not None
    assert _column(kinematics, "pelvis_tx", 1) == pytest.approx(0.6)
    assert _column(kinematics, "pelvis_ty", 0) == pytest.approx(0.9)
    assert _column(kinematics, "pelvis_tilt", 0) == pytest.approx(math.pi / 2)
    assert _column(kinematics, "hip_flexion_r", 0) == pytest.approx(math.pi / 6)
    # The knee drives translation axes too, but is an angle.
    assert _column(kinematics, "knee_angle_r", 0) == pytest.approx(math.pi / 4)


def test_kinematics_record_their_model_and_units(tmp_path: Path) -> None:
    session_dir = _full_session(tmp_path, ("swing1",))

    session = load_opencap_session(session_dir)

    assert session.kinematics is not None
    metadata = session.kinematics.trajectory.metadata
    assert metadata["opensim_model"] == str(session.model_file)
    assert metadata["translational_coordinates"] == ["pelvis_tx", "pelvis_ty"]


def test_single_motion_trial_is_chosen_over_neutral(tmp_path: Path) -> None:
    session_dir = _full_session(tmp_path, ("neutral", "swing1"))

    assert load_opencap_session(session_dir).trial == "swing1"


def test_several_motion_trials_require_an_explicit_choice(tmp_path: Path) -> None:
    session_dir = _full_session(tmp_path, ("swing1", "swing2"))

    with pytest.raises(ValueError, match=r"swing1, swing2.*trial="):
        load_opencap_session(session_dir)


def test_unknown_trial_is_rejected(tmp_path: Path) -> None:
    session_dir = _full_session(tmp_path, ("swing1",))

    with pytest.raises(KeyError, match="swing9"):
        load_opencap_session(session_dir, trial="swing9")


def test_kinematics_columns_must_exist_in_the_model(tmp_path: Path) -> None:
    session_dir = write_opencap_session(tmp_path, ("swing1",))
    write_scaled_model(session_dir)
    write_kinematics(
        session_dir,
        "swing1",
        columns=("pelvis_tilt", "elbow_flex_r"),
        rows=((0.0, 1.0, 2.0), (0.1, 1.0, 2.0)),
    )

    with pytest.raises(ValueError, match="elbow_flex_r"):
        load_opencap_session(session_dir)


def test_kinematics_without_a_model_are_not_unit_converted(tmp_path: Path) -> None:
    session_dir = write_opencap_session(tmp_path, ("swing1",))
    write_kinematics(session_dir, "swing1")

    session = load_opencap_session(session_dir)

    assert session.model_file is None
    assert session.kinematics is None
    assert "no scaled model" in session.notes[0]


def test_session_without_kinematics_still_loads_markers(tmp_path: Path) -> None:
    session_dir = write_opencap_session(tmp_path, ("swing1",))
    write_scaled_model(session_dir)

    session = load_opencap_session(session_dir)

    assert session.kinematics is None
    assert session.observations.num_frames == 3


def test_invalid_subject_mass_is_rejected(tmp_path: Path) -> None:
    session_dir = _full_session(tmp_path, ("swing1",))
    metadata = session_dir / "sessionMetadata.yaml"
    metadata.write_text(
        metadata.read_text(encoding="utf-8").replace("mass_kg: 79.5", "mass_kg: -1"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="mass_kg"):
        load_opencap_session(session_dir)


def test_osim_coordinate_kinds(tmp_path: Path) -> None:
    model = write_scaled_model(tmp_path)

    kinds = read_osim_coordinates(model)

    assert list(kinds) == [
        "pelvis_tilt",
        "pelvis_tx",
        "pelvis_ty",
        "hip_flexion_r",
        "knee_angle_r",
        "mtp_angle_r",
        "sled_x",
    ]
    assert kinds["pelvis_tx"] is CoordinateKind.TRANSLATIONAL
    assert kinds["knee_angle_r"] is CoordinateKind.ROTATIONAL
    assert kinds["mtp_angle_r"] is CoordinateKind.ROTATIONAL
    assert kinds["sled_x"] is CoordinateKind.TRANSLATIONAL


def test_osim_without_coordinates_is_rejected(tmp_path: Path) -> None:
    model = tmp_path / "empty.osim"
    model.write_text(
        '<OpenSimDocument Version="40000"><Model name="m" /></OpenSimDocument>',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="no coordinates"):
        read_osim_coordinates(model)
