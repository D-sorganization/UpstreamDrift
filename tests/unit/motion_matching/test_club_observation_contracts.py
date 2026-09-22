"""CO-01 club observation contracts, calibration and legacy adapters (#10605)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.shared.python.math_utils.quaternion import slerp
from src.shared.python.motion_matching._geodesic import quaternion_geodesic_angles
from src.shared.python.motion_matching.club_calibration import (
    ClubGeometryCalibration,
    calibrate_tool_to_model,
    cross_check_grip_face_consistency,
    mid_hands_to_butt_end,
    qualify_derived_orientation_axes,
    transform_points,
)
from src.shared.python.motion_matching.club_models import DRIVER, IRON_7
from src.shared.python.motion_matching.club_only import (
    CANONICAL_TRIAL_SHEETS,
    NATIVE_SAMPLE_RATE_HZ,
)
from src.shared.python.motion_matching.club_only.adapters import (
    club_target_to_observation,
    observation_to_club_target,
)
from src.shared.python.motion_matching.club_only.observation import (
    OBSERVATION_SCHEMA,
    ClubObservation,
    ClubObservationKinematics,
    ClubObservationProvenance,
    ComponentMask,
    ComponentStatus,
    DerivationMetadata,
    ObservationEvent,
    UncertaintyMetadata,
    build_calibrated_observation_fixture,
    interpolate_observation,
    load_observation_fixture_pack,
    orientation_residual_so3,
    scored_component_subset,
)
from src.shared.python.motion_matching.club_target import (
    AlignOptions,
    ClubTarget,
    SourceProvenance,
)
from src.shared.python.motion_matching.loaders.excel import load_club_target_excel

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
CLUB_DATA = REPO_ROOT / "data" / "Club_Data.xlsx"
EVIDENCE = (
    REPO_ROOT
    / "docs"
    / "plans"
    / "club_only_matching"
    / "evidence"
    / "club_observation_contracts.json"
)


def _identity_quat(n: int) -> np.ndarray:
    return np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))


def _yaw_quat(angle_rad: float) -> np.ndarray:
    half = 0.5 * angle_rad
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)


def _make_source() -> SourceProvenance:
    return SourceProvenance(
        filename="synthetic.npz",
        format="synthetic",
        subject_id="SYN",
        trial_id="TW_wiffle",
        sha256="0" * 64,
    )


def _full_mask(
    *,
    mid_orientation: ComponentStatus = ComponentStatus.MEASURED,
    face_orientation: ComponentStatus = ComponentStatus.MEASURED,
    twist: ComponentStatus = ComponentStatus.MEASURED,
) -> ComponentMask:
    return ComponentMask(
        mid_hands_position=ComponentStatus.MEASURED,
        face_position=ComponentStatus.MEASURED,
        mid_hands_orientation=mid_orientation,
        face_orientation=face_orientation,
        twist=twist,
    )


def _make_observation(
    *,
    n: int = 5,
    mask: ComponentMask | None = None,
    mid_quat: np.ndarray | None = None,
    face_quat: np.ndarray | None = None,
    club_type: str = DRIVER.name,
    length_m: float = DRIVER.length_m,
) -> ClubObservation:
    time = np.arange(n, dtype=np.float64) / NATIVE_SAMPLE_RATE_HZ
    mid = np.zeros((n, 3), dtype=np.float64)
    face = np.zeros((n, 3), dtype=np.float64)
    face[:, 1] = length_m
    if mid_quat is None:
        mid_quat = _identity_quat(n)
    if face_quat is None:
        face_quat = _identity_quat(n)
    return ClubObservation(
        native_time_s=time,
        mid_hands_xyz=mid,
        face_xyz=face,
        mid_hands_quat=mid_quat,
        face_quat=face_quat,
        mask=mask or _full_mask(),
        derivation=DerivationMetadata(
            orientation_axis_status="measured",
            degenerate_axes=False,
            notes=(),
        ),
        uncertainty=UncertaintyMetadata(
            position_sigma_m=0.001,
            orientation_sigma_rad=0.01,
        ),
        events=(
            ObservationEvent(
                label="I", sample_index=n // 2, time_s=float(time[n // 2])
            ),
        ),
        sample_rate_hz=NATIVE_SAMPLE_RATE_HZ,
        club_type=club_type,
        catalog_length_m=length_m,
        source=_make_source(),
        trial_id="TW_wiffle",
    )


def test_se3_tool_to_model_round_trip() -> None:
    rng = np.random.default_rng(7)
    model_pts = rng.normal(size=(8, 3))
    angle = 0.35
    c, s = np.cos(angle), np.sin(angle)
    r = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    t = np.array([0.1, -0.2, 0.05])
    tool_pts = (model_pts @ r.T) + t
    calib = calibrate_tool_to_model(tool_pts, model_pts)
    recovered = transform_points(calib.inverse(), transform_points(calib, tool_pts))
    assert np.allclose(recovered, tool_pts, atol=1e-9)
    round_trip = transform_points(calib, tool_pts)
    assert np.allclose(round_trip, model_pts, atol=1e-9)


def test_quaternion_sign_equivalence_so3_residual() -> None:
    q = _yaw_quat(0.4)
    q_neg = -q
    residual = orientation_residual_so3(q[None, :], q_neg[None, :])
    assert residual.shape == (1,)
    assert float(residual[0]) == pytest.approx(0.0, abs=1e-12)
    geodesic = quaternion_geodesic_angles(q[None, :], q_neg[None, :])
    assert float(geodesic[0]) == pytest.approx(0.0, abs=1e-12)


def test_degenerate_axes_require_flag() -> None:
    x = np.array([1.0, 0.0, 0.0])
    y = np.array([1.0, 1e-12, 0.0])
    with pytest.raises(ValueError, match="degener"):
        qualify_derived_orientation_axes(x, y, degenerate=False)
    flagged = qualify_derived_orientation_axes(x, y, degenerate=True)
    assert flagged.degenerate_axes is True
    assert flagged.orientation_axis_status == "derived_not_measured"


def test_invalid_rotation_determinant_rejected() -> None:
    bad = np.eye(3)
    bad[0, 0] = -1.0  # det = -1
    kinematics = ClubObservationKinematics(
        native_time_s=np.array([0.0, 1.0 / 240.0]),
        mid_hands_xyz=np.zeros((2, 3)),
        face_xyz=np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]),
        mid_hands_rotmats=np.stack([np.eye(3), np.eye(3)]),
        face_rotmats=np.stack([bad, np.eye(3)]),
    )
    provenance = ClubObservationProvenance(
        mask=_full_mask(),
        derivation=DerivationMetadata(
            orientation_axis_status="measured",
            degenerate_axes=False,
            notes=(),
        ),
        uncertainty=UncertaintyMetadata(0.001, 0.01),
        events=(),
        sample_rate_hz=NATIVE_SAMPLE_RATE_HZ,
        club_type=DRIVER.name,
        catalog_length_m=DRIVER.length_m,
        source=_make_source(),
        trial_id="TW_wiffle",
    )
    with pytest.raises(ValueError, match="determinant|proper rotation|SO\\(3\\)"):
        ClubObservation.from_frames(kinematics, provenance)


def test_wrong_length_and_club_type_fail_closed() -> None:
    obs = _make_observation(club_type=DRIVER.name, length_m=DRIVER.length_m)
    # Face far too short for a driver catalog length.
    short = ClubObservation(
        native_time_s=obs.native_time_s,
        mid_hands_xyz=obs.mid_hands_xyz,
        face_xyz=obs.mid_hands_xyz + np.array([0.0, 0.2, 0.0]),
        mid_hands_quat=obs.mid_hands_quat,
        face_quat=obs.face_quat,
        mask=obs.mask,
        derivation=obs.derivation,
        uncertainty=obs.uncertainty,
        events=obs.events,
        sample_rate_hz=obs.sample_rate_hz,
        club_type=DRIVER.name,
        catalog_length_m=DRIVER.length_m,
        source=obs.source,
        trial_id=obs.trial_id,
    )
    with pytest.raises(ValueError, match="length|catalog"):
        ClubGeometryCalibration.from_observation(short, club_type=DRIVER.name)
    with pytest.raises(ValueError, match="club type|unknown|mismatch"):
        ClubGeometryCalibration.from_observation(obs, club_type="putter")


def test_native_240hz_clock_and_event_preservation() -> None:
    pack = load_observation_fixture_pack(EVIDENCE)
    assert pack["schema"] == OBSERVATION_SCHEMA
    assert pack["native_sample_rate_hz"] == NATIVE_SAMPLE_RATE_HZ
    for trial_id in CANONICAL_TRIAL_SHEETS:
        fixture = pack["trials"][trial_id]
        assert fixture["sample_rate_hz"] == NATIVE_SAMPLE_RATE_HZ
        assert "I" in fixture["events"]
        obs = build_calibrated_observation_fixture(trial_id)
        assert obs.sample_rate_hz == NATIVE_SAMPLE_RATE_HZ
        assert abs(float(np.median(np.diff(obs.native_time_s))) - (1.0 / 240.0)) < 1e-9
        impact = next(e for e in obs.events if e.label == "I")
        assert impact.time_s == pytest.approx(
            float(obs.native_time_s[impact.sample_index]), abs=1e-12
        )


def test_position_only_target_masks_twist_not_identity() -> None:
    n = 4
    nan_quat = np.full((n, 4), np.nan)
    obs = _make_observation(
        n=n,
        mask=_full_mask(
            mid_orientation=ComponentStatus.UNOBSERVED,
            face_orientation=ComponentStatus.UNOBSERVED,
            twist=ComponentStatus.UNOBSERVED,
        ),
        mid_quat=nan_quat,
        face_quat=nan_quat,
    )
    assert obs.mask.twist is ComponentStatus.UNOBSERVED
    assert not np.any(np.isfinite(obs.face_quat))
    # Must not silently become identity for scoring.
    scored = scored_component_subset(obs)
    assert "face_orientation" not in scored
    assert "twist" not in scored
    assert "mid_hands_position" in scored
    assert "face_position" in scored
    with pytest.raises(ValueError, match="unobserved|orientation"):
        observation_to_club_target(obs, AlignOptions())


def test_legacy_excel_load_behavior_unchanged() -> None:
    opts = AlignOptions(
        sample_rate_hz=200.0,
        simulation_time_s=0.1,
        time_alignment="impact",
        impact_target_t_s=0.05,
    )
    target = load_club_target_excel(CLUB_DATA, "TW_ProV1", opts)
    assert isinstance(target, ClubTarget)
    assert target.time[0] == pytest.approx(0.0)
    assert target.butt.shape[1] == 3
    assert target.clubhead.shape[1] == 3
    assert target.club_quat.shape[1] == 4
    # Legacy adapter must not rename mid-hands as butt-end without offset.
    obs = club_target_to_observation(target, trial_id="TW_ProV1")
    assert np.allclose(obs.mid_hands_xyz, target.butt)
    assert obs.mask.mid_hands_position is ComponentStatus.MEASURED
    with pytest.raises(ValueError, match="offset|butt-end|mid-hands"):
        mid_hands_to_butt_end(obs.mid_hands_xyz, obs.face_xyz, offset_m=None)


def test_both_grip_and_head_orientations_retained() -> None:
    n = 3
    mid_q = np.stack([_yaw_quat(0.1), _yaw_quat(0.2), _yaw_quat(0.3)])
    face_q = np.stack([_yaw_quat(-0.1), _yaw_quat(-0.2), _yaw_quat(-0.3)])
    obs = _make_observation(n=n, mid_quat=mid_q, face_quat=face_q)
    assert not np.allclose(obs.mid_hands_quat, obs.face_quat)
    legacy = observation_to_club_target(obs, AlignOptions(time_alignment="none"))
    # Face orientation feeds legacy club_quat; grip orientation remains on obs.
    assert np.allclose(legacy.club_quat, obs.face_quat)
    assert np.allclose(obs.mid_hands_quat, mid_q)


def test_so3_interpolation_uses_slerp_not_lerp() -> None:
    q0 = _yaw_quat(0.0)
    q1 = _yaw_quat(np.pi / 2)
    mid = slerp(q0, q1, 0.5)
    expected_angle = quaternion_geodesic_angles(q0[None, :], mid[None, :])[0]
    assert float(expected_angle) == pytest.approx(np.pi / 4, rel=1e-6)
    n = 2
    obs = _make_observation(
        n=n,
        mid_quat=np.stack([q0, q1]),
        face_quat=np.stack([q0, q1]),
    )
    denser = interpolate_observation(obs, sample_rate_hz=480.0)
    assert denser.native_time_s.shape[0] > n
    # Mid sample should stay on the geodesic (unit quat, not chord lerp).
    mid_row = denser.face_quat[len(denser.face_quat) // 2]
    assert float(np.linalg.norm(mid_row)) == pytest.approx(1.0, abs=1e-9)


def test_grip_face_rigidity_cross_check() -> None:
    obs = _make_observation()
    report = cross_check_grip_face_consistency(obs)
    assert report.median_length_m == pytest.approx(DRIVER.length_m, rel=1e-6)
    assert report.passed
    # Non-rigid stretch should fail.
    stretched = ClubObservation(
        native_time_s=obs.native_time_s,
        mid_hands_xyz=obs.mid_hands_xyz,
        face_xyz=obs.face_xyz * 1.2,
        mid_hands_quat=obs.mid_hands_quat,
        face_quat=obs.face_quat,
        mask=obs.mask,
        derivation=obs.derivation,
        uncertainty=obs.uncertainty,
        events=obs.events,
        sample_rate_hz=obs.sample_rate_hz,
        club_type=obs.club_type,
        catalog_length_m=obs.catalog_length_m,
        source=obs.source,
        trial_id=obs.trial_id,
    )
    bad = cross_check_grip_face_consistency(stretched)
    assert not bad.passed


def test_iron_catalog_accepted() -> None:
    obs = _make_observation(club_type=IRON_7.name, length_m=IRON_7.length_m)
    calib = ClubGeometryCalibration.from_observation(obs, club_type=IRON_7.name)
    assert calib.club_spec.name == IRON_7.name
    assert calib.length_m == pytest.approx(IRON_7.length_m)


def test_four_trial_fixtures_declare_component_coverage() -> None:
    pack = load_observation_fixture_pack(EVIDENCE)
    for trial_id in CANONICAL_TRIAL_SHEETS:
        coverage = pack["trials"][trial_id]["component_coverage"]
        assert coverage["mid_hands_position"] == "measured"
        assert coverage["face_position"] == "measured"
        assert coverage["face_orientation"] in {"measured", "derived_not_measured"}
        assert "twist" in coverage


def test_public_input_validation_survives_python_dash_o() -> None:
    """Public ValueError checks must not rely on assert/__debug__ alone."""
    import subprocess
    import sys

    script = (
        "import numpy as np\n"
        "from src.shared.python.motion_matching.club_calibration "
        "import mid_hands_to_butt_end\n"
        "try:\n"
        "    mid_hands_to_butt_end(np.zeros((2, 3)), np.ones((2, 3)), None)\n"
        "except ValueError as exc:\n"
        "    assert 'offset' in str(exc).lower() or 'butt' in str(exc).lower()\n"
        "else:\n"
        "    raise SystemExit('expected ValueError under python -O')\n"
    )
    completed = subprocess.run(
        [sys.executable, "-O", "-c", script],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert completed.returncode == 0, completed.stderr
