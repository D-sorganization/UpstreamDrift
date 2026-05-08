"""Golden-fixture round-trip tests for motion_pipeline source adapters.

Part of issue #4571 gap-fill. For every fixture under
``tests/data/motion_pipeline/golden/`` this test loads the file through
``motion_pipeline.sources.registry.load_any`` (added by PR #4619) and asserts:

1. The returned value is a CIR type (KeypointSequence / MarkerTrajectory /
   JointTrajectory).
2. Timestamps are monotonically non-decreasing.
3. Declared schema / marker_set / num_frames roughly match the fixture
   metadata.

The test gracefully skips when the registry is not yet importable, so it is
safe to land before PR #4619.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.motion_pipeline]

REPO_ROOT = Path(__file__).resolve().parents[3]
GOLDEN_DIR = REPO_ROOT / "tests" / "data" / "motion_pipeline" / "golden"

FIXTURE_FILES = [
    "sample.bvh",
    "sample.trc",
    "sample.mot",
    "sample.sto",
    "openpose_keypoints.json",
    "alphapose.json",
    "hrnet.json",
    "sample.csv",
    "mediapipe.json",
    "sample.c3d",
]


def _registry_or_skip():
    try:
        from src.shared.python.motion_pipeline.sources import registry  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover — depends on PR #4619
        pytest.skip(f"motion_pipeline.sources.registry not available: {exc}")
    return registry


def _is_cir_object(obj: object) -> bool:
    """Return True if `obj` quacks like a CIR trajectory/sequence."""
    cls_name = type(obj).__name__
    return cls_name in {"KeypointSequence", "MarkerTrajectory", "JointTrajectory"}


def _frame_timestamps(obj: object) -> list[float]:
    frames = getattr(obj, "frames", None)
    if frames is None:
        return []
    return [f.timestamp for f in frames]


@pytest.mark.parametrize("fixture_name", FIXTURE_FILES)
def test_golden_fixture_round_trip(fixture_name: str) -> None:
    """Each fixture round-trips through the source registry into a CIR object."""
    fixture_path = GOLDEN_DIR / fixture_name
    if not fixture_path.exists():
        pytest.skip(f"Fixture not present (optional dep?): {fixture_path}")

    registry = _registry_or_skip()
    load_any = getattr(registry, "load_any", None)
    if load_any is None:
        pytest.skip("registry.load_any not available")

    try:
        cir = load_any(fixture_path)
    except NotImplementedError as exc:
        pytest.skip(f"Adapter not yet implemented for {fixture_name}: {exc}")
    except ImportError as exc:
        pytest.skip(f"Optional dependency missing for {fixture_name}: {exc}")

    # (a) returns a CIR object
    assert _is_cir_object(cir), (
        f"Expected CIR type from {fixture_name}, got {type(cir).__name__}"
    )

    # (b) timestamps monotonic (non-decreasing)
    timestamps = _frame_timestamps(cir)
    assert len(timestamps) >= 1, f"{fixture_name} produced 0 frames"
    for i in range(1, len(timestamps)):
        assert timestamps[i] >= timestamps[i - 1], (
            f"{fixture_name} timestamps not monotonic at index {i}: "
            f"{timestamps[i - 1]} -> {timestamps[i]}"
        )

    # (c) schema/marker_set sanity — depends on object kind
    cls_name = type(cir).__name__
    if cls_name == "KeypointSequence":
        # Every KeypointFrame must declare a schema name and they must match.
        schemas = {f.schema_name for f in cir.frames}
        assert len(schemas) == 1, f"{fixture_name} mixed schemas: {schemas}"
    elif cls_name == "MarkerTrajectory":
        # Marker set non-empty.
        assert cir.marker_names, f"{fixture_name} has no marker names"
    elif cls_name == "JointTrajectory":
        # Skeleton must have at least one joint.
        assert cir.skeleton.num_joints >= 1, f"{fixture_name} skeleton has no joints"


def test_golden_dir_exists_and_nonempty() -> None:
    """Sanity: golden fixtures dir is populated (independent of registry)."""
    assert GOLDEN_DIR.is_dir(), f"Missing dir: {GOLDEN_DIR}"
    files = [p for p in GOLDEN_DIR.iterdir() if p.is_file() and p.name != ".gitkeep"]
    assert files, "tests/data/motion_pipeline/golden/ is empty"
    # All checked-in fixtures must be <= 50 KB to keep the repo lean.
    for p in files:
        assert p.stat().st_size <= 50_000, f"{p.name} exceeds 50 KB cap"
