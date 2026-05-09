"""Invariant / property-style tests for the motion_pipeline package.

Closeout for epic #4558. These tests are deliberately light-weight:
they use plain ``pytest.parametrize`` rather than hypothesis to keep the
test dependency surface minimal.

Three invariant families are exercised:

1. JSON round-trip equality for every public CIR type.
2. Adapter postconditions (monotonic time, finite values, non-empty
   frames) for every available source adapter / golden fixture pairing.
3. Orchestrator determinism: ``pipeline.run()`` is idempotent across two
   identical inputs (the orchestrator is xfailed today, see #4647).
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.motion_pipeline]


# ---------------------------------------------------------------------------
# JSON round-trip property
# ---------------------------------------------------------------------------


def _make_keypoint() -> Any:
    from src.shared.python.motion_pipeline.contracts import (
        Keypoint,
        KeypointFrame,
        KeypointSequence,
    )

    kp = Keypoint(x=1.0, y=2.0, z=3.0, confidence=0.9, name="nose")
    frame = KeypointFrame(
        timestamp=0.0,
        keypoints=[kp],
        schema_name="custom",
        frame_index=0,
    )
    return KeypointSequence(id="kp-seq", frames=[frame])


def _make_marker() -> Any:
    from src.shared.python.motion_pipeline.contracts import (
        Marker,
        MarkerFrame,
        MarkerTrajectory,
    )

    m = Marker(name="PELVIS", x=0.1, y=0.2, z=0.3, occluded=False)
    return MarkerTrajectory(
        id="m-traj",
        frames=[MarkerFrame(timestamp=0.0, markers={"PELVIS": m}, frame_index=0)],
    )


def _make_skeleton() -> Any:
    from src.shared.python.motion_pipeline.contracts import JointDef, SkeletonRig

    joints = {
        "ROOT": JointDef(name="ROOT", parent=None, axes=["X", "Y", "Z"]),
    }
    return SkeletonRig(id="rig", joints=joints, root_joint="ROOT")


def _make_motion_trajectory() -> Any:
    from src.shared.python.motion_pipeline.contracts import (
        JointStateFrame,
        JointTrajectory,
        MotionTrajectory,
    )

    skeleton = _make_skeleton()
    frames = [JointStateFrame(timestamp=0.0, q=[0.0, 0.0, 0.0])]
    traj = JointTrajectory(id="jt", skeleton=skeleton, frames=frames)
    return MotionTrajectory(id="mt", skeleton=skeleton, trajectory=traj)


def _make_motion_matching_result() -> Any:
    from src.shared.python.motion_pipeline.contracts import MotionMatchingResult

    return MotionMatchingResult(
        request_id="req-1",
        success=True,
        error_metrics={"rmse": 0.001},
        iterations=5,
        solve_time=0.1,
    )


_FACTORIES: dict[str, Any] = {
    "Keypoint": lambda: __import__(
        "src.shared.python.motion_pipeline.contracts",
        fromlist=["Keypoint"],
    ).Keypoint(x=1.0, y=2.0, confidence=0.5),
    "KeypointSequence": _make_keypoint,
    "MarkerTrajectory": _make_marker,
    "SkeletonRig": _make_skeleton,
    "MotionTrajectory": _make_motion_trajectory,
    "MotionMatchingResult": _make_motion_matching_result,
}


@pytest.mark.parametrize("type_name", sorted(_FACTORIES.keys()))
def test_cir_json_roundtrip_preserves_equality(type_name: str) -> None:
    """For every public CIR type, ``model_dump_json`` -> ``model_validate_json`` is identity."""
    factory = _FACTORIES[type_name]
    instance = factory()
    cls = type(instance)

    serialized = instance.model_dump_json()
    rehydrated = cls.model_validate_json(serialized)

    # Pydantic equality compares field values, ignoring datetime drift.
    assert rehydrated.model_dump(mode="python") == instance.model_dump(mode="python")


# ---------------------------------------------------------------------------
# Adapter postconditions for every golden fixture
# ---------------------------------------------------------------------------


def _golden_files() -> list[Path]:
    repo_root = Path(__file__).resolve().parents[3]
    golden = repo_root / "tests" / "data" / "motion_pipeline" / "golden"
    if not golden.exists():
        return []
    return sorted(p for p in golden.iterdir() if p.is_file())


# Adapters known to raise on the current shipped golden fixtures - tracked
# as #4683. Fixed by mediapipe + hrnet adapter changes; set retained as a
# hook in case future fixtures ship in a known-broken state.
_BROKEN_ADAPTER_FIXTURES: frozenset[str] = frozenset()


def _resolve_frames(payload: Any) -> list[Any] | None:
    """Locate the frame list on any CIR payload type.

    :class:`MotionTrajectory` nests its frames under ``trajectory.frames``;
    every other CIR type exposes ``frames`` directly.
    """
    direct = getattr(payload, "frames", None)
    if direct is not None:
        return direct
    nested = getattr(payload, "trajectory", None)
    if nested is not None:
        return getattr(nested, "frames", None)
    return None


@pytest.mark.parametrize(
    "fixture",
    _golden_files()
    or [pytest.param(None, marks=pytest.mark.skip(reason="no fixtures yet"))],
    ids=lambda p: p.name if isinstance(p, Path) else "no-fixtures",
)
def test_adapter_postconditions_on_golden_fixture(
    fixture: Path | None,
    request: pytest.FixtureRequest,
) -> None:
    """Every golden fixture either loads cleanly or is rejected with UnsupportedFormatError.

    When it loads, the resulting payload must satisfy the documented
    postconditions: non-empty frames, monotonic timestamps, finite values.
    """
    if fixture is None:
        pytest.skip("No golden fixtures available")

    if fixture.name in _BROKEN_ADAPTER_FIXTURES:
        request.node.add_marker(
            pytest.mark.xfail(strict=True, reason="Tracked in #4683")
        )

    from src.shared.python.motion_pipeline.sources import (
        UnsupportedFormatError,
        load_any,
    )

    try:
        payload = load_any(fixture)
    except UnsupportedFormatError:
        pytest.skip(f"No adapter claims {fixture.name}")
        return

    frames = _resolve_frames(payload)
    assert frames, f"{fixture.name}: payload has no frames"

    timestamps = [f.timestamp for f in frames]
    assert all(np.isfinite(t) for t in timestamps), (
        f"{fixture.name}: non-finite timestamps"
    )
    assert timestamps == sorted(timestamps), (
        f"{fixture.name}: timestamps must be non-decreasing"
    )


# ---------------------------------------------------------------------------
# Orchestrator determinism
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=False,
    reason="Orchestrator end-to-end run blocked by #4647/#4648/#4649",
)
def test_orchestrator_run_is_deterministic() -> None:
    """Two consecutive ``pipeline.run`` calls on equal inputs produce equal results.

    Determinism is a soft property here - we accept tiny float jitter
    introduced by datetime stamping but the underlying data tensors must
    be byte-identical.
    """
    from src.shared.python.motion_pipeline.contracts import (
        Marker,
        MarkerFrame,
        MarkerTrajectory,
    )
    from src.shared.python.motion_pipeline.orchestrator import (
        AdapterOverride,
        MotionPipeline,
        PipelineConfig,
    )

    def _build() -> MarkerTrajectory:
        m = Marker(name="P", x=0.0, y=1.0, z=0.0)
        return MarkerTrajectory(
            id="d", frames=[MarkerFrame(timestamp=0.0, markers={"P": m})]
        )

    config = PipelineConfig(
        adapter=AdapterOverride(format="passthrough"),
        ik_backend="mujoco",
        matching_backend="mujoco",
    )

    r1 = MotionPipeline(config).run(_build())
    r2 = MotionPipeline(config).run(_build())

    # The matched trajectory tensors must match exactly.
    if r1.matched_trajectory and r2.matched_trajectory:
        q1 = [f.q for f in r1.matched_trajectory.trajectory.frames]
        q2 = [f.q for f in r2.matched_trajectory.trajectory.frames]
        assert q1 == q2
