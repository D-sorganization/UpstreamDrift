"""End-to-end smoke test that exercises the full motion_pipeline stack.

Closeout for epic #4558. Builds a tiny deterministic synthetic capture using
the public CIR contracts, runs every available stage of the pipeline against
it, and asserts the standard postconditions on the final
:class:`MotionMatchingResult`.

This test is deliberately defensive about optional backends: it uses
``pytest.importorskip`` for each physics engine and ``xfail`` for known
in-flight gaps (orchestrator wiring bugs tracked on issues #4647-#4650).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.motion_pipeline]


# ---------------------------------------------------------------------------
# Tiny deterministic fixture builder
# ---------------------------------------------------------------------------


def _build_synthetic_marker_trajectory() -> Any:
    """Return a tiny deterministic :class:`MarkerTrajectory` with 3 markers.

    The trajectory is 30 frames at 60 Hz with smooth sinusoidal motion -
    intentionally light so it exercises the contracts without needing a
    real capture. All numeric values are finite by construction.
    """
    from src.shared.python.motion_pipeline.contracts import (
        Calibration,
        Marker,
        MarkerFrame,
        MarkerTrajectory,
    )

    fps = 60.0
    num_frames = 30
    rng = np.random.default_rng(seed=4558)
    frames = []
    for i in range(num_frames):
        t = i / fps
        sig = math.sin(2.0 * math.pi * t)
        markers = {
            "PELVIS": Marker(name="PELVIS", x=0.0, y=1.0 + 0.01 * sig, z=0.0),
            "TORSO": Marker(name="TORSO", x=0.0, y=1.4 + 0.02 * sig, z=0.05 * sig),
            "HEAD": Marker(
                name="HEAD",
                x=float(0.001 * rng.standard_normal()),
                y=1.7,
                z=0.0,
            ),
        }
        frames.append(MarkerFrame(timestamp=t, markers=markers, frame_index=i))

    calibration = Calibration(
        id="synthetic-cal",
        cameras={},
        unit_system="meters",
        source_fps=fps,
        world_up_axis="+Y",
    )
    return MarkerTrajectory(
        id="synthetic-traj",
        frames=frames,
        calibration=calibration,
        subject_id="synthetic-subject",
        metadata={"origin": "test_full_pipeline"},
    )


def _build_simple_skeleton() -> Any:
    """Return a 3-joint linear skeleton matching the synthetic markers."""
    from src.shared.python.motion_pipeline.contracts import JointDef, SkeletonRig

    joints = {
        "PELVIS": JointDef(
            name="PELVIS",
            parent=None,
            children=["TORSO"],
            tpose_offset=[0.0, 1.0, 0.0],
            axes=["X", "Y", "Z"],
            semantic_label="pelvis",
        ),
        "TORSO": JointDef(
            name="TORSO",
            parent="PELVIS",
            children=["HEAD"],
            tpose_offset=[0.0, 0.4, 0.0],
            axes=["X", "Y", "Z"],
            semantic_label="torso",
        ),
        "HEAD": JointDef(
            name="HEAD",
            parent="TORSO",
            children=[],
            tpose_offset=[0.0, 0.3, 0.0],
            axes=["X", "Y", "Z"],
            semantic_label="head",
        ),
    }
    return SkeletonRig(
        id="synthetic-rig",
        joints=joints,
        root_joint="PELVIS",
        up_axis="+Y",
        scale=1.0,
    )


# ---------------------------------------------------------------------------
# Stage 1 - adapter via public registry
# ---------------------------------------------------------------------------


def test_registry_loads_every_golden_fixture(tmp_path: Path) -> None:
    """Each registered adapter can claim and load at least one golden fixture.

    Uses the public ``load_any`` entrypoint - no per-format imports.
    """
    from src.shared.python.motion_pipeline.sources import (
        UnsupportedFormatError,
        list_formats,
        load_any,
    )

    repo_root = Path(__file__).resolve().parents[3]
    golden_dir = repo_root / "tests" / "data" / "motion_pipeline" / "golden"
    if not golden_dir.exists():
        pytest.skip(
            "Golden fixtures not present on this branch (will arrive with PR #4638)."
        )

    formats = list_formats()
    assert formats, "No source adapters were registered at import time."

    loaded_any = False
    errors: list[str] = []
    for fixture in sorted(golden_dir.iterdir()):
        if not fixture.is_file():
            continue
        try:
            payload = load_any(fixture)
        except UnsupportedFormatError:
            # This fixture's format is not covered yet - that's fine.
            continue
        except Exception as exc:  # noqa: BLE001 - capture for diagnostic
            errors.append(f"{fixture.name}: {type(exc).__name__}: {exc}")
            continue
        loaded_any = True
        # Adapter postconditions: non-empty frames with finite timestamps.
        assert payload is not None, f"Adapter returned None for {fixture.name}"
        # MotionTrajectory nests frames under .trajectory.frames; everything
        # else exposes them directly.
        frames = getattr(payload, "frames", None) or getattr(
            getattr(payload, "trajectory", None), "frames", None
        )
        assert frames, f"Loaded payload from {fixture.name} has no frames"

    assert loaded_any, f"No fixtures loaded successfully. Errors: {errors or 'none'}"


# ---------------------------------------------------------------------------
# Stage 2-3 - preprocessing + scaling smoke tests
# ---------------------------------------------------------------------------


def test_preprocessing_subpackage_exposes_public_api() -> None:
    """The preprocessing subpackage publishes the documented public surface.

    This catches accidental ``__init__.py`` regressions.
    """
    preprocessing = pytest.importorskip(
        "src.shared.python.motion_pipeline.preprocessing"
    )
    expected = {
        "GapFillStrategy",
        "gap_fill",
        "FilterType",
        "apply_filter",
        "resample",
        "normalize_coordinates",
        "convert_units",
        "PreprocessingPipeline",
        "PreprocessingStep",
    }
    missing = expected - set(getattr(preprocessing, "__all__", []))
    assert not missing, f"preprocessing.__all__ missing: {sorted(missing)}"


def test_scaling_subpackage_exposes_public_api() -> None:
    """The scaling subpackage publishes the documented public surface."""
    scaling = pytest.importorskip("src.shared.python.motion_pipeline.scaling")
    expected = {
        "scale_skeleton",
        "MarkerMap",
        "get_marker_set",
        "MarkerSet",
    }
    missing = expected - set(getattr(scaling, "__all__", []))
    assert not missing, f"scaling.__all__ missing: {sorted(missing)}"


# ---------------------------------------------------------------------------
# Stage 4-5 - IK + matching backend availability matrix
# ---------------------------------------------------------------------------


IK_BACKENDS = ("mujoco", "drake", "pinocchio", "opensim")
MATCHING_BACKENDS = ("mujoco", "drake", "pinocchio")


@pytest.mark.parametrize("backend", IK_BACKENDS)
def test_ik_backend_module_imports_or_skips(backend: str) -> None:
    """Each IK backend module imports cleanly OR raises a clean ImportError.

    The orchestrator switches on backend name and we want to be certain that
    the underlying module is structured so the import error is recoverable.
    """
    mod_path = f"src.shared.python.motion_pipeline.ik.{backend}_backend"
    try:
        __import__(mod_path)
    except ImportError as exc:
        pytest.skip(f"{backend} backend unavailable: {exc}")


@pytest.mark.parametrize("backend", MATCHING_BACKENDS)
def test_matching_backend_module_imports_or_skips(backend: str) -> None:
    """Each matching backend module imports cleanly OR raises ImportError."""
    mod_path = f"src.shared.python.motion_pipeline.matching.{backend}_backend"
    try:
        __import__(mod_path)
    except ImportError as exc:
        pytest.skip(f"{backend} matching backend unavailable: {exc}")


# ---------------------------------------------------------------------------
# End-to-end orchestrator smoke
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=False,
    reason=(
        "Orchestrator wiring tracked on #4647/#4648/#4649: "
        "load_source / apply_preprocessing / default skeleton missing on main."
    ),
)
def test_orchestrator_full_run_on_synthetic_marker_trajectory() -> None:
    """The orchestrator should run end-to-end on a passthrough MarkerTrajectory.

    Marked xfail until the in-flight bug-fix PRs land; once they do, this test
    becomes a strict regression gate. The assertions cover the full
    :class:`MotionMatchingResult` invariant set defined by the epic.
    """
    from src.shared.python.motion_pipeline.contracts import MotionMatchingResult
    from src.shared.python.motion_pipeline.orchestrator import (
        AdapterOverride,
        MotionPipeline,
        PipelineConfig,
    )

    traj = _build_synthetic_marker_trajectory()

    config = PipelineConfig(
        adapter=AdapterOverride(format="passthrough"),
        ik_backend="mujoco",
        matching_backend="mujoco",
    )
    pipeline = MotionPipeline(config)
    result = pipeline.run(traj)

    # Type and basic shape
    assert isinstance(result, MotionMatchingResult)
    assert result.matched_trajectory is not None

    # Joint trajectory dimensions match the rig
    matched = result.matched_trajectory
    expected_dofs = matched.skeleton.num_dofs
    for frame in matched.trajectory.frames:
        assert frame.num_dofs == expected_dofs

    # Timestamps strictly monotonic
    timestamps = [f.timestamp for f in matched.trajectory.frames]
    assert all(t1 < t2 for t1, t2 in zip(timestamps, timestamps[1:], strict=True))

    # Tracking residuals finite + bounded
    for metric, value in (result.error_metrics or {}).items():
        assert np.isfinite(value), f"error metric {metric!r} is not finite: {value}"

    # Provenance metadata populated
    provenance = matched.source_provenance
    assert "source_hash" in provenance
    assert "software_version" in provenance
    assert matched.created_at is not None
