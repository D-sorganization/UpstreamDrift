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


@pytest.mark.requires_mocap_fixtures
def test_typed_error_when_fixtures_absent() -> None:
    """When the fixture is absent, the loader raises the typed error.

    This exercises the *negative* contract: when the user opts into the
    mocap-gated tests but the asset is still missing, we want the typed
    ``MuscleFixturesUnavailableError`` to fire with the absolute path
    embedded so the gap is visible in CI logs.
    """
    if _have_mocap_fixtures():
        pytest.skip(
            "fixtures are present; this test exercises the missing-fixture branch only"
        )
    osim_path = _osim_path()
    with pytest.raises(MuscleFixturesUnavailableError) as exc_info:
        build_rajagopal2015_muscle_model(osim_path)
    assert exc_info.value.missing_path == osim_path
    assert str(osim_path) in str(exc_info.value)


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
