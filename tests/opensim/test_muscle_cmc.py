"""Tests for the post-MVP Rajagopal2015 muscle CMC scaffold (issue #4296).

These tests are intentionally **dependency-gated**. They report missing
fixtures and missing optional bindings with explicit, typed reasons so CI
logs surface *which* asset is absent, rather than silently passing an
untested code path.

Test markers:
  - ``requires_opensim`` — needs ``import opensim`` to succeed.
  - ``requires_mocap_fixtures`` — needs the body-marker mocap fixture
    bundle described in
    ``src/engines/physics_engines/opensim/python/POST_MVP_MUSCLES.md``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from src.engines.physics_engines.opensim.python.muscle_analysis import (
    DEFAULT_RAJAGOPAL2015_OSIM_RELPATH,
    RAJAGOPAL2015_MUSCLE_COUNT,
    RAJAGOPAL2015_REQUIRED_MARKERS,
    CMCResult,
    MuscleFixturesUnavailableError,
    TrajectorySchemaError,
    _resolve_mocap_fixtures_root,
    build_rajagopal2015_muscle_model,
    run_cmc_smoke,
    validate_marker_trajectory,
)
from src.engines.physics_engines.opensim.python.tour_matching.muscle_qualification import (
    AnatomicalRegion,
    AnatomyCoverageScope,
    IncompleteAnatomyCoverageError,
    InvalidMuscleParameterError,
    InvalidMusclePathError,
    MomentArmDerivativeMismatchError,
    MuscleEquilibriumState,
    MuscleParameterProvenance,
    MusclePathGeometry,
    MuscleQualificationReceipt,
    NativeShortReplayReceipt,
    UninitializedTendonStateError,
    UnsupportedAnatomyClaimError,
    audit_activation_dynamics,
    audit_anatomy_coverage,
    audit_initial_muscle_equilibrium,
    compute_path_length_finite_difference_moment_arm,
    qualify_muscle_extensions,
    validate_moment_arm_consistency,
    validate_muscle_parameters,
    validate_muscle_path_and_wrapping,
)

pytestmark = [pytest.mark.unit]

_OPENSIM_AVAILABLE: bool = importlib.util.find_spec("opensim") is not None


def _fixture_root() -> Path:
    return _resolve_mocap_fixtures_root()


def _osim_path() -> Path:
    return _fixture_root() / DEFAULT_RAJAGOPAL2015_OSIM_RELPATH


def _have_mocap_fixtures() -> bool:
    return _osim_path().is_file()


def _skip_unless_opensim() -> None:
    if not _OPENSIM_AVAILABLE:
        pytest.skip("OpenSim binding missing: `import opensim` failed")


def _skip_unless_fixtures() -> None:
    if not _have_mocap_fixtures():
        pytest.skip(f"mocap fixtures not at {_osim_path()}")


# --------------------------------------------------------------------------- #
# Fixture-presence sentinels (acceptance criterion 1 — must fail loudly)
# --------------------------------------------------------------------------- #


@pytest.mark.requires_mocap_fixtures
def test_rajagopal_fixture_present_for_active_runs() -> None:
    """Loud fail when mocap fixtures are missing.

    Per #4296 acceptance criterion 1, the muscle restore path must be
    *explicitly* reported as missing until body-marker mocap fixtures
    ship. This test is selected only when the user opts in to the
    ``requires_mocap_fixtures`` marker; CI with
    ``-m "not requires_mocap_fixtures"`` excludes it cleanly.
    """
    osim_path = _osim_path()
    if not osim_path.is_file():
        pytest.fail(
            "Rajagopal2015 muscle CMC fixture is unavailable. Expected "
            f"asset at {osim_path}. See POST_MVP_MUSCLES.md for sourcing "
            "instructions."
        )


# --------------------------------------------------------------------------- #
# Trajectory schema validation (no external deps required)
# --------------------------------------------------------------------------- #


def _good_trajectory(n: int = 8) -> dict[str, object]:
    time = np.linspace(0.0, 1.0, n)
    markers = {
        name: np.zeros((n, 3), dtype=float) for name in RAJAGOPAL2015_REQUIRED_MARKERS
    }
    return {
        "time": time,
        "markers": markers,
        "units": "m",
        "frame": "y_up",
    }


def test_validate_marker_trajectory_accepts_valid_input() -> None:
    validate_marker_trajectory(_good_trajectory())


def test_validate_marker_trajectory_rejects_none() -> None:
    with pytest.raises(TrajectorySchemaError):
        validate_marker_trajectory(None)  # type: ignore[arg-type]


def test_validate_marker_trajectory_rejects_wrong_units() -> None:
    traj = _good_trajectory()
    traj["units"] = "mm"
    with pytest.raises(TrajectorySchemaError, match="units"):
        validate_marker_trajectory(traj)


def test_validate_marker_trajectory_rejects_unknown_frame() -> None:
    traj = _good_trajectory()
    traj["frame"] = "x_up"
    with pytest.raises(TrajectorySchemaError, match="frame"):
        validate_marker_trajectory(traj)


def test_validate_marker_trajectory_rejects_non_monotonic_time() -> None:
    traj = _good_trajectory()
    t: np.ndarray = traj["time"]  # type: ignore[assignment]
    t[3] = t[2]
    with pytest.raises(TrajectorySchemaError, match="monotonic"):
        validate_marker_trajectory(traj)


def test_validate_marker_trajectory_rejects_short_time() -> None:
    traj = _good_trajectory(n=1)
    with pytest.raises(TrajectorySchemaError, match="at least 2"):
        validate_marker_trajectory(traj)


def test_validate_marker_trajectory_rejects_missing_markers() -> None:
    traj = _good_trajectory()
    traj["markers"].pop("R.Heel")  # type: ignore[union-attr]
    with pytest.raises(TrajectorySchemaError, match="missing required markers"):
        validate_marker_trajectory(traj)


def test_validate_marker_trajectory_rejects_wrong_marker_shape() -> None:
    traj = _good_trajectory()
    bad = np.zeros((traj["time"].size, 2), dtype=float)  # type: ignore[union-attr]
    traj["markers"]["R.ASIS"] = bad  # type: ignore[index]
    with pytest.raises(TrajectorySchemaError, match="shape"):
        validate_marker_trajectory(traj)


def test_validate_marker_trajectory_rejects_nonfinite() -> None:
    traj = _good_trajectory()
    traj["markers"]["L.Knee"][0, 0] = np.nan  # type: ignore[index]
    with pytest.raises(TrajectorySchemaError, match="non-finite"):
        validate_marker_trajectory(traj)


def test_validate_marker_trajectory_accepts_z_up() -> None:
    traj = _good_trajectory()
    traj["frame"] = "z_up"
    validate_marker_trajectory(traj)


# --------------------------------------------------------------------------- #
# Loader / runner DbC checks (no fixture required)
# --------------------------------------------------------------------------- #


def test_build_rajagopal2015_rejects_bad_model_path_type() -> None:
    with pytest.raises(TypeError, match="model_path"):
        build_rajagopal2015_muscle_model(model_path=123)  # type: ignore[arg-type]


def test_build_rajagopal2015_rejects_bad_output_path_type() -> None:
    with pytest.raises(TypeError, match="output_path"):
        build_rajagopal2015_muscle_model(output_path=object())  # type: ignore[arg-type]


def test_build_rajagopal2015_raises_typed_error_for_missing_path(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "does_not_exist.osim"
    with pytest.raises(MuscleFixturesUnavailableError) as exc_info:
        build_rajagopal2015_muscle_model(missing)
    assert exc_info.value.missing_path == missing.resolve()


def test_run_cmc_smoke_rejects_none_path() -> None:
    with pytest.raises(TypeError, match="trajectory_path"):
        run_cmc_smoke(None, model=object())  # type: ignore[arg-type]


def test_run_cmc_smoke_rejects_none_model(tmp_path: Path) -> None:
    fake_traj = tmp_path / "traj.mot"
    fake_traj.write_text("# placeholder")
    with pytest.raises(ValueError, match="model"):
        run_cmc_smoke(fake_traj, model=None)


def test_run_cmc_smoke_rejects_invalid_duration(tmp_path: Path) -> None:
    fake_traj = tmp_path / "traj.mot"
    fake_traj.write_text("# placeholder")

    class _Stub:
        def getMuscles(self) -> object:
            class _MS:
                def getSize(self) -> int:
                    return RAJAGOPAL2015_MUSCLE_COUNT

            return _MS()

    with pytest.raises(ValueError, match="duration_s"):
        run_cmc_smoke(fake_traj, model=_Stub(), duration_s=-1.0)


def test_run_cmc_smoke_raises_typed_error_for_missing_trajectory(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.mot"

    class _Stub:
        def getMuscles(self) -> object:
            class _MS:
                def getSize(self) -> int:
                    return RAJAGOPAL2015_MUSCLE_COUNT

            return _MS()

    with pytest.raises(MuscleFixturesUnavailableError) as exc_info:
        run_cmc_smoke(missing, model=_Stub())
    assert exc_info.value.missing_path == missing.resolve()


# --------------------------------------------------------------------------- #
# Conditional integration tests (require both deps)
# --------------------------------------------------------------------------- #


@pytest.mark.requires_opensim
@pytest.mark.requires_mocap_fixtures
def test_rajagopal2015_model_has_eighty_muscles() -> None:
    """Acceptance criterion 3: 80-muscle Rajagopal2015 loads cleanly."""
    _skip_unless_opensim()
    _skip_unless_fixtures()

    model = build_rajagopal2015_muscle_model()
    assert model is not None
    assert int(model.getMuscles().getSize()) == RAJAGOPAL2015_MUSCLE_COUNT


@pytest.mark.requires_opensim
@pytest.mark.requires_mocap_fixtures
def test_cmc_smoke_returns_finite_consistent_arrays() -> None:
    """Acceptance criterion 4: CMC smoke yields finite, consistent outputs."""
    _skip_unless_opensim()
    _skip_unless_fixtures()

    trajectory_path = _fixture_root() / "kinematics" / "smoke.mot"
    if not trajectory_path.is_file():
        pytest.fail(
            f"mocap kinematics fixture missing at {trajectory_path} — required "
            "for CMC smoke test"
        )

    model = build_rajagopal2015_muscle_model()
    result = run_cmc_smoke(trajectory_path, model)

    assert isinstance(result, CMCResult)
    n_time = int(result.time.size)
    assert result.excitations.shape == (n_time, RAJAGOPAL2015_MUSCLE_COUNT)
    assert result.activations.shape == (n_time, RAJAGOPAL2015_MUSCLE_COUNT)
    assert result.forces.shape == (n_time, RAJAGOPAL2015_MUSCLE_COUNT)
    assert np.all(np.isfinite(result.time))
    assert np.all(np.isfinite(result.excitations))
    assert np.all(np.isfinite(result.activations))
    assert np.all(np.isfinite(result.forces))
    assert len(result.muscle_names) == RAJAGOPAL2015_MUSCLE_COUNT


# --------------------------------------------------------------------------- #
# OG-08: Muscle and Tendon Extension Qualification Contracts
# --------------------------------------------------------------------------- #


def test_anatomy_coverage_audit_rejects_lower_limb_only_claiming_full_golf() -> None:
    """Acceptance criterion: lower-limb model cannot claim full golf muscle capability."""
    lower_limb_muscles = [
        "gluteus_maximus_r",
        "gluteus_medius_r",
        "rectus_femoris_r",
        "vastus_lateralis_r",
        "vastus_medialis_r",
        "biceps_femoris_r",
        "semitendinosus_r",
        "tibialis_anterior_r",
        "gastrocnemius_medialis_r",
        "soleus_r",
    ]
    with pytest.raises(UnsupportedAnatomyClaimError, match="lower-limb"):
        audit_anatomy_coverage(lower_limb_muscles, claimed_capability="full_body_golf")


def test_anatomy_coverage_audit_accepts_comprehensive_golf_anatomy() -> None:
    """Acceptance criterion: comprehensive model with declared omissions is accepted."""
    comprehensive_muscles = [
        # Lower extremity
        "gluteus_maximus_r",
        "rectus_femoris_r",
        "gastrocnemius_r",
        # Torso / spine
        "erector_spinae_r",
        "external_oblique_r",
        "rectus_abdominis",
        # Shoulder / scapula
        "deltoid_anterior_r",
        "latissimus_dorsi_r",
        "pectoralis_major_r",
        "trapezius_r",
        # Arm / forearm
        "biceps_brachii_r",
        "triceps_brachii_r",
        "pronator_teres_r",
        # Wrist / hand
        "flexor_carpi_radialis_r",
        "extensor_carpi_radialis_r",
        "flexor_digitorum_superficialis_r",
    ]
    scope = audit_anatomy_coverage(
        comprehensive_muscles,
        claimed_capability="full_body_golf",
        declared_omissions=(AnatomicalRegion.HEAD_NECK,),
        omission_notes={"head_neck": "Rigidly affixed torso-neck segment per #10394"},
    )
    assert scope.is_full_golf_model
    assert AnatomicalRegion.HEAD_NECK in scope.declared_omissions
    assert AnatomicalRegion.SHOULDER_SCAPULA in scope.covered_regions


def test_validate_muscle_parameters_rejects_negative_force() -> None:
    """DbC check: F_max must be positive."""
    with pytest.raises(InvalidMuscleParameterError, match="F_max"):
        MuscleParameterProvenance(
            muscle_name="deltoid",
            F_max=-500.0,
            l_opt=0.098,
            l_slack=0.093,
            pennation_angle=0.22,
            source_citation="Holzbaur 2005",
            license_terms="SimTK OpenSim Models",
        )


def test_validate_muscle_parameters_rejects_invalid_pennation() -> None:
    """DbC check: pennation angle must be in [0, pi/2)."""
    with pytest.raises(InvalidMuscleParameterError, match="pennation"):
        MuscleParameterProvenance(
            muscle_name="soleus",
            F_max=2800.0,
            l_opt=0.05,
            l_slack=0.25,
            pennation_angle=1.65,  # > pi/2 (1.5708)
            source_citation="Rajagopal 2016",
            license_terms="SimTK OpenSim Models",
        )


def test_validate_muscle_parameters_accepts_valid_provenance() -> None:
    """Verify parameter table provenance and deterministic hashing."""
    good_param = MuscleParameterProvenance(
        muscle_name="latissimus_dorsi_r",
        F_max=742.0,
        l_opt=0.255,
        l_slack=0.180,
        pennation_angle=0.10,
        source_citation="Holzbaur et al. 2005",
        license_terms="Creative Commons Attribution 4.0 / SimTK",
    )
    report = validate_muscle_parameters([good_param])
    assert report["valid"] is True
    assert len(good_param.parameter_hash) == 64  # SHA-256


def test_validate_muscle_path_rejects_single_point() -> None:
    """DbC check: muscle path must have at least origin and insertion."""
    bad_path = MusclePathGeometry(
        muscle_name="biceps",
        path_points=(("scapula", (0.02, 0.05, 0.01)),),
        wrapping_surfaces=(),
    )
    with pytest.raises(InvalidMusclePathError, match="at least 2 points"):
        validate_muscle_path_and_wrapping(bad_path)


def test_validate_muscle_path_rejects_same_body() -> None:
    """DbC check: muscle path points cannot all reside on the same body."""
    bad_path = MusclePathGeometry(
        muscle_name="biceps",
        path_points=(
            ("humerus_r", (0.0, 0.2, 0.0)),
            ("humerus_r", (0.0, 0.0, 0.0)),
        ),
        wrapping_surfaces=(),
    )
    with pytest.raises(InvalidMusclePathError, match="distinct parent bodies"):
        validate_muscle_path_and_wrapping(bad_path)


def test_moment_arm_finite_difference_derivative_agreement() -> None:
    """Qualify moment arm against negative partial derivative of path length."""
    # Virtual joint angle q (elbow flexion)
    # MTU length model: l_MT(q) = L0 - r * sin(q)
    # True moment arm: -d(l_MT)/dq = r * cos(q)
    r = 0.04  # 4 cm moment arm
    l0 = 0.30

    def path_length_fn(q: float) -> float:
        return float(l0 - r * np.sin(q))

    q_eval = 0.7854  # 45 deg
    expected_moment_arm = r * np.cos(q_eval)

    # Compute finite-difference moment arm
    fd_moment_arm = compute_path_length_finite_difference_moment_arm(
        path_length_fn, q_eval, dq=1e-5
    )
    assert np.isclose(fd_moment_arm, expected_moment_arm, atol=1e-5)

    # Validate agreement via qualification function
    validate_moment_arm_consistency(
        expected_moment_arm, path_length_fn, q_eval, tol=1e-4
    )


def test_moment_arm_finite_difference_mismatch_raises() -> None:
    """Discrepancy between moment arm and path length derivative raises error."""
    r = 0.04
    l0 = 0.30

    def path_length_fn(q: float) -> float:
        return float(l0 - r * np.sin(q))

    q_eval = 0.5
    corrupted_moment_arm = 0.12  # Incorrect moment arm value

    with pytest.raises(MomentArmDerivativeMismatchError, match="Moment arm"):
        validate_moment_arm_consistency(
            corrupted_moment_arm, path_length_fn, q_eval, tol=1e-3
        )


def test_audit_initial_muscle_equilibrium_detects_uninitialized_state() -> None:
    """Uninitialized or non-equilibrated fiber length raises typed error."""
    param = MuscleParameterProvenance(
        muscle_name="rectus_femoris",
        F_max=1169.0,
        l_opt=0.084,
        l_slack=0.346,
        pennation_angle=0.087,
        source_citation="Rajagopal 2016",
        license_terms="SimTK",
    )
    # Passing uninitialized fiber length far from equilibrium
    with pytest.raises(UninitializedTendonStateError, match="equilibrium"):
        audit_initial_muscle_equilibrium(
            param,
            l_MT=0.45,
            activation=0.05,
            initial_l_CE=0.01,  # Severely non-equilibrated
            tol_n=0.1,
            enforce_equilibrium=True,
        )


def test_audit_initial_muscle_equilibrium_succeeds_at_equilibrium() -> None:
    """Solved equilibrium returns valid state with residual near zero."""
    param = MuscleParameterProvenance(
        muscle_name="rectus_femoris",
        F_max=1169.0,
        l_opt=0.084,
        l_slack=0.346,
        pennation_angle=0.087,
        source_citation="Rajagopal 2016",
        license_terms="SimTK",
    )
    # Solve for equilibrium
    state = audit_initial_muscle_equilibrium(
        param,
        l_MT=0.43,
        activation=0.05,
        enforce_equilibrium=False,
    )
    assert isinstance(state, MuscleEquilibriumState)
    assert state.is_equilibrated
    assert abs(state.residual_force_n) < 1.0  # < 1 N residual


def test_audit_activation_dynamics_bounds() -> None:
    """Activation bounds [a_min, 1.0] are enforced."""
    good_acts = [0.02, 0.5, 0.95]
    audit_activation_dynamics(good_acts, min_activation=0.01)

    bad_acts_low = [-0.05, 0.5]
    with pytest.raises(InvalidMuscleParameterError, match="[Aa]ctivation"):
        audit_activation_dynamics(bad_acts_low)

    bad_acts_high = [0.5, 1.2]
    with pytest.raises(InvalidMuscleParameterError, match="[Aa]ctivation"):
        audit_activation_dynamics(bad_acts_high)


def test_qualify_muscle_extensions_generates_receipt() -> None:
    """Receipt compiles all audits, keeps baseline torque unreplaced, and reports residuals."""
    receipt = qualify_muscle_extensions(
        model_variant_id="golf_humanoid_muscle_variant",
        base_model_sha256="abc123def456",
        muscles=[
            MuscleParameterProvenance(
                muscle_name="pectoralis_major_r",
                F_max=824.0,
                l_opt=0.144,
                l_slack=0.039,
                pennation_angle=0.26,
                source_citation="Holzbaur 2005",
                license_terms="SimTK",
            )
        ],
        claimed_capability="upper_body_pilot",
        declared_omissions=(AnatomicalRegion.HEAD_NECK,),
        short_replay_duration_s=0.05,
    )
    assert isinstance(receipt, MuscleQualificationReceipt)
    assert receipt.muscle_complete_status == "IN_PROGRESS_QUALIFICATION"
    assert receipt.independent_validation_status == "PENDING_10375"
    assert receipt.short_replay_receipt is not None
    assert isinstance(receipt.short_replay_receipt, NativeShortReplayReceipt)
    assert len(receipt.receipt_sha256) == 64
