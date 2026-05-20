"""Contract-stability snapshots for the CIR public surface.

Closeout for epic #4558. Downstream consumers of ``motion_pipeline``
(Tools, Gasification_Model integration paths, GAAI agents) read and
write Pydantic-serialized CIR documents. Once a field is required, we
must not silently rename or drop it; once an enum value is published,
old serialized data must continue to deserialize.

Tests in this module are deliberately *snapshot-style*: they assert that
specific symbols and field names are present, rather than that the
shape is exactly some dictionary. Adding new optional fields is fine -
removing or renaming required fields will fail these checks.
"""

from __future__ import annotations

import json
from typing import Any, get_args

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.motion_pipeline]


# Required-field snapshots. Maps CIR class name to the *minimum* set of
# field names that must exist on ``model_fields``. Adding a field is OK;
# removing one trips a snapshot mismatch.
REQUIRED_FIELDS: dict[str, set[str]] = {
    "Calibration": {"id", "cameras", "unit_system", "source_fps", "world_up_axis"},
    "CameraIntrinsics": {"fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"},
    "CameraExtrinsics": {"rotation", "translation"},
    "Keypoint": {"x", "y", "z", "confidence", "name"},
    "KeypointFrame": {"timestamp", "keypoints", "schema_name", "frame_index"},
    "KeypointSequence": {"id", "frames", "calibration", "metadata"},
    "Marker": {"name", "x", "y", "z", "residual", "occluded"},
    "MarkerFrame": {"timestamp", "markers", "frame_index"},
    "MarkerTrajectory": {
        "id",
        "frames",
        "calibration",
        "subject_id",
        "metadata",
    },
    "JointDef": {
        "name",
        "parent",
        "children",
        "tpose_offset",
        "axes",
        "limits",
        "semantic_label",
    },
    "JointLimit": {"lower", "upper", "soft_lower", "soft_upper"},
    "SkeletonRig": {"id", "joints", "root_joint", "up_axis", "scale", "metadata"},
    "JointStateFrame": {"timestamp", "q", "qdot", "qddot", "frame_index"},
    "JointTrajectory": {"id", "skeleton", "frames", "metadata"},
    "MotionTrajectory": {
        "id",
        "skeleton",
        "trajectory",
        "marker_reference",
        "subject",
        "sport",
        "club",
        "source_provenance",
        "created_at",
        "metadata",
    },
    "MotionMatchingRequest": {
        "id",
        "target_trajectory",
        "target_markers",
        "target_keypoints",
        "skeleton",
        "constraints",
        "solver_config",
    },
    "MotionMatchingResult": {
        "request_id",
        "success",
        "matched_trajectory",
        "error_metrics",
        "iterations",
        "solve_time",
        "message",
        "metadata",
    },
}


@pytest.mark.parametrize("cls_name", sorted(REQUIRED_FIELDS.keys()))
def test_required_field_snapshot(cls_name: str) -> None:
    """Every CIR class still exposes its documented field set (at minimum)."""
    contracts = pytest.importorskip("src.shared.python.motion_pipeline.contracts")
    cls = getattr(contracts, cls_name, None)
    assert cls is not None, f"CIR class {cls_name} missing from public contracts."

    actual = set(cls.model_fields.keys())
    expected = REQUIRED_FIELDS[cls_name]
    missing = expected - actual
    assert not missing, (
        f"{cls_name} is missing fields {sorted(missing)} - "
        f"contract regression. Actual fields: {sorted(actual)}."
    )


# ---------------------------------------------------------------------------
# Enum / Literal snapshots
# ---------------------------------------------------------------------------


def test_schema_name_literal_values_stable() -> None:
    """Schema names are part of the public contract."""
    contracts = pytest.importorskip("src.shared.python.motion_pipeline.contracts")
    expected = {"BODY_25", "MediaPipe_33", "COCO_17", "OpenPose_25", "custom"}
    actual = set(get_args(contracts.SchemaName))
    missing = expected - actual
    assert (
        not missing
    ), f"SchemaName Literal lost values {sorted(missing)}; actual: {sorted(actual)}"


def test_up_axis_literal_values_stable() -> None:
    """World-up axis values are part of the public contract."""
    contracts = pytest.importorskip("src.shared.python.motion_pipeline.contracts")
    expected = {"+Y", "+Z", "+X", "-Y", "-Z", "-X"}
    actual = set(get_args(contracts.UpAxis))
    assert (
        expected == actual
    ), f"UpAxis Literal changed: expected {sorted(expected)}, got {sorted(actual)}"


# ---------------------------------------------------------------------------
# Old-document forward compatibility
# ---------------------------------------------------------------------------


# A *minimal* serialized CIR document representing the kind of data a
# downstream consumer might have stored at the ratification of these
# contracts. New optional fields must not break this load.
LEGACY_MARKER_TRAJECTORY_JSON = json.dumps(
    {
        "id": "legacy-traj",
        "frames": [
            {
                "timestamp": 0.0,
                "markers": {
                    "PELVIS": {
                        "name": "PELVIS",
                        "x": 0.0,
                        "y": 1.0,
                        "z": 0.0,
                    }
                },
                "frame_index": 0,
            }
        ],
        "metadata": {},
    }
)


LEGACY_MOTION_MATCHING_RESULT_JSON = json.dumps(
    {
        "request_id": "legacy-req",
        "success": True,
        "error_metrics": {"rmse": 0.001},
        "iterations": 5,
        "solve_time": 0.1,
    }
)


def test_legacy_marker_trajectory_still_loads() -> None:
    """A pre-existing :class:`MarkerTrajectory` JSON document still validates."""
    contracts = pytest.importorskip("src.shared.python.motion_pipeline.contracts")
    obj = contracts.MarkerTrajectory.model_validate_json(LEGACY_MARKER_TRAJECTORY_JSON)
    assert obj.id == "legacy-traj"
    assert len(obj.frames) == 1
    assert "PELVIS" in obj.frames[0].markers


def test_legacy_motion_matching_result_still_loads() -> None:
    """A pre-existing :class:`MotionMatchingResult` JSON document still validates.

    Regression for #4842: legacy successful results predate the
    payload-on-success invariant (no ``matched_trajectory``/``torques``/
    ``activations``) and must still load via the v1->v2 migration.
    """
    contracts = pytest.importorskip("src.shared.python.motion_pipeline.contracts")
    obj = contracts.MotionMatchingResult.model_validate_json(
        LEGACY_MOTION_MATCHING_RESULT_JSON
    )
    assert obj.success is True
    assert obj.error_metrics["rmse"] == pytest.approx(0.001)
    assert obj.matched_trajectory is None  # field added later, optional
    # The migration tags legacy documents (no ``schema_version``) as v1
    # so the post-validator knows to relax the payload invariant.
    assert obj.schema_version == 1


def test_v2_motion_matching_result_still_enforces_payload_invariant() -> None:
    """New (v2) successful results without a payload must still fail.

    Companion to #4842: legacy compatibility must not regress the
    invariant for newly-created results.
    """
    contracts = pytest.importorskip("src.shared.python.motion_pipeline.contracts")
    v2_no_payload = json.dumps(
        {
            "request_id": "v2-req",
            "success": True,
            "error_metrics": {"rmse": 0.001},
            "iterations": 5,
            "solve_time": 0.1,
            "schema_version": contracts.MOTION_MATCHING_RESULT_SCHEMA_VERSION,
        }
    )
    with pytest.raises(Exception, match="must include at least one"):
        contracts.MotionMatchingResult.model_validate_json(v2_no_payload)
