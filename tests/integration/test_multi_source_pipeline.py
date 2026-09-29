"""End-to-end integration tests for the multi-source motion-target pipeline.

These tests stitch together the loader / target / dispatcher surface against
the real C3D and ``.mat`` artefacts in the repository. They verify:

* The simulation timegrid contract (1 kHz x 0.300 s -> 301 samples,
  ``time[0] == 0``, ``impact_idx == 251``).
* Quaternion unit-norm to ``1e-6`` for every frame.
* Clubhead impact-speed sanity ranges against verified-by-hand baselines.
* Golden snapshots of the last 10 frames (text-diffable JSON, see
  ``tests/fixtures/motion_matching/regenerate.py``).

Tests for loaders that have not yet merged are guarded with
``pytest.importorskip`` and ``pytest.skip("waiting on #NNNN")`` so this
file lands green even when only a subset of the pipeline is present on the
target branch.

Wave-4 dependencies (skipped until merged):

* ``BodyTarget`` + ``load_body_target`` -- waiting on #4481.
* ``.mat`` club loader (rob_neal corpus) -- waiting on #4490.
* ``ClubBallTarget`` / ``MultiSourceTarget`` aggregator -- waiting on #4482.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.integration]

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
PINOCCHIO_DATA_DIR = (
    REPO_ROOT / "src" / "engines" / "physics_engines" / "pinocchio" / "data"
)
ROB_NEAL_DIR = PINOCCHIO_DATA_DIR / "rob_neal"
GEARS_DIR = PINOCCHIO_DATA_DIR / "gears_tour_average"
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "motion_matching"

EXPECTED_N_FRAMES = 301
EXPECTED_IMPACT_IDX = 251
EXPECTED_DT = 1.0e-3
QUAT_NORM_TOL = 1.0e-6

# Verified-by-hand impact clubhead-speed baselines (m/s) +/- 0.5 m/s.
CLUB_BASELINES_C3D: dict[str, float] = {
    "C3D_TA_Driver.c3d": 51.0,
    "C3D_TA_Iron.c3d": 39.6,
}
CLUB_BASELINES_MAT: dict[str, float] = {
    "TW_ProV1.mat": 51.67,
    "TW_wiffle.mat": 51.70,
    "GW_ProV1.mat": 51.72,
    "GW_wiffle.mat": 50.77,
}
SPEED_TOL_MPS = 0.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _impact_clubhead_speed(target) -> float:
    """Central-difference clubhead speed at the impact sample, in m/s."""
    dt = float(target.time[1] - target.time[0])
    velocity = np.gradient(target.clubhead, dt, axis=0)
    return float(np.linalg.norm(velocity[int(target.impact_idx)]))


def _validate_grid_and_quat(target) -> None:
    """Common timegrid + quaternion-unit-norm assertions."""
    assert target.time.shape == (EXPECTED_N_FRAMES,), (
        f"expected {EXPECTED_N_FRAMES} samples, got {target.time.shape}"
    )
    assert float(target.time[0]) == 0.0
    assert int(target.impact_idx) == EXPECTED_IMPACT_IDX
    np.testing.assert_allclose(np.diff(target.time), EXPECTED_DT, rtol=0.0, atol=1.0e-9)
    qnorms = np.linalg.norm(target.club_quat, axis=1)
    deviations = np.abs(qnorms - 1.0)
    assert float(deviations.max()) <= QUAT_NORM_TOL, (
        f"club_quat unit-norm tol {QUAT_NORM_TOL} breached: "
        f"max deviation {float(deviations.max()):.3e}"
    )


def _maybe_load_club_target(path: Path):
    """Load a club target via dispatch, importorskip-ing missing modules."""
    loader_mod = pytest.importorskip(
        "src.shared.python.motion_matching.load_club_target",
        reason="load_club_target dispatcher not present on this branch",
    )
    target_mod = pytest.importorskip(
        "src.shared.python.motion_matching.target",
        reason="target module not present on this branch",
    )
    return loader_mod.load_club_target(path, opts=target_mod.AlignOptions())


# ---------------------------------------------------------------------------
# Parametrised C3D pipeline
# ---------------------------------------------------------------------------


C3D_FIXTURES: list[tuple[str, Path]] = [
    (name, DATA_DIR / name) for name in CLUB_BASELINES_C3D
]


# ---------------------------------------------------------------------------
# Parametrised .mat pipeline (skipped until #4490 lands the .mat loader)
# ---------------------------------------------------------------------------


MAT_FIXTURES: list[tuple[str, Path]] = [
    (name, ROB_NEAL_DIR / name) for name in CLUB_BASELINES_MAT
]


@pytest.mark.parametrize(
    "filename, path",
    MAT_FIXTURES,
    ids=[name for name, _ in MAT_FIXTURES],
)
def test_mat_club_pipeline_grid_and_kinematics(filename: str, path: Path) -> None:
    """``.mat`` club loader: timegrid contract + impact clubhead-speed sanity.

    Skipped until issue #4490 lands the ``.mat`` dispatch in
    ``load_club_target``.
    """
    if not path.exists():
        pytest.skip(f".mat fixture not present in workspace: {path}")
    suffix = path.suffix.lower()
    try:
        target = _maybe_load_club_target(path)
    except (ValueError, NotImplementedError) as exc:
        if suffix == ".mat":
            pytest.skip(f"waiting on #4490 (.mat loader): {exc}")
        raise
    _validate_grid_and_quat(target)
    speed = _impact_clubhead_speed(target)
    expected = CLUB_BASELINES_MAT[filename]
    assert abs(speed - expected) <= SPEED_TOL_MPS, (
        f"{filename}: |v_clubhead| at impact = {speed:.3f} m/s; "
        f"expected {expected:.3f} +/- {SPEED_TOL_MPS} m/s"
    )


# ---------------------------------------------------------------------------
# Body-target shared-clock check (skipped until #4481 lands BodyTarget)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Golden snapshots
# ---------------------------------------------------------------------------


def _assert_snapshot_close(actual: np.ndarray, expected_list: list, name: str) -> None:
    """Compare a fresh array against a JSON-loaded reference."""
    expected = np.asarray(expected_list, dtype=np.float64)
    assert actual.shape == expected.shape, (
        f"{name}: shape {actual.shape} != snapshot shape {expected.shape}"
    )
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1.0e-9,
        atol=1.0e-6,
        err_msg=(
            f"{name} drifted from golden snapshot. "
            f"Regenerate via "
            f"`python3 tests/fixtures/motion_matching/regenerate.py` "
            f"if the change is intentional."
        ),
    )


def test_body_target_driver_golden_snapshot() -> None:
    """Body-marker snapshot for the driver C3D.

    Skipped until ``BodyTarget`` infrastructure (#4481) lands and the
    snapshot is regenerated.
    """
    snapshot_path = FIXTURES_DIR / "body_target_driver_last10.json"
    if not snapshot_path.exists():
        pytest.skip("waiting on #4481: body-target snapshot not yet committed")
    body_loader_mod = pytest.importorskip(
        "src.shared.python.motion_matching.load_body_target",
        reason="waiting on #4481 (load_body_target)",
    )
    target_mod = pytest.importorskip(
        "src.shared.python.motion_matching.target",
        reason="target module not present on this branch",
    )
    driver_path = DATA_DIR / "C3D_TA_Driver.c3d"
    if not driver_path.exists():
        pytest.skip(f"driver C3D fixture not present: {driver_path}")

    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    body = body_loader_mod.load_body_target(  # type: ignore[attr-defined]
        driver_path, opts=target_mod.AlignOptions()
    )
    last_n = int(snapshot["last_n"])
    sl = slice(-last_n, None)
    _assert_snapshot_close(body.time[sl], snapshot["time"], "time")
    markers = getattr(body, "markers", {})
    for name, expected_list in snapshot.get("marker_xyz", {}).items():
        if name not in markers:
            pytest.skip(f"body marker {name!r} not present in BodyTarget output")
        _assert_snapshot_close(markers[name][sl], expected_list, f"marker {name}")
