"""Unit tests for FullBodyPinkTasks (Packet P1, #10276).

Follows strict TDD, DbC, LoD, and DRY principles.
Tests the task translation, request validation, coordinate mapping,
dropout masks, zero observed targets, locked coordinates, bounds conversion,
and error residual auditing against mock Pinocchio and Pink interfaces.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.engines.physics_engines.pinocchio.python.pink_tasks import (
    ConfigurationState,
    FrameResiduals,
    FrameTaskBundle,
    FrameTaskOptions,
    FrameTaskRequest,
    FullBodyPinkTasks,
    StanceClosurePolicy,
)

pytestmark = pytest.mark.unit


def _mock_spec() -> dict[str, Any]:
    """Minimal valid full-body specification fixture for unit tests."""
    return {
        "schema_version": "full-body-v1",
        "coordinate_order": [f"coord_{i}" for i in range(41)],
        "coordinate_ranges_deg": {f"coord_{i}": (-90.0, 90.0) for i in range(41)},
        "bodies": [
            {"name": "body_a", "solids": []},
            {"name": "body_b", "solids": []},
            {"name": "body_c", "solids": []},
        ],
        "joints": [],
        "frames": [
            {
                "name": "HeadTop",
                "body": "body_a",
                "placement": np.eye(4).tolist(),
            },
            {
                "name": "WaistLeft",
                "body": "body_b",
                "placement": np.eye(4).tolist(),
            },
            {
                "name": "ClubTip",
                "body": "body_c",
                "placement": np.eye(4).tolist(),
            },
        ],
        "closure": {
            "name": "grip_weld",
            "body_a": "body_a",
            "body_b": "body_b",
            "placement_a": np.eye(4).tolist(),
            "placement_b": np.eye(4).tolist(),
        },
        "marker_attachments": {
            "HeadTop": {"body": "HeadTop", "offset_m": [0.0, 0.0, 0.1]},
            "WaistLeft": {"body": "WaistLeft", "offset_m": None},
            "ClubTip": {"body": "ClubTip", "offset_m": [0.0, 0.0, -0.5]},
        },
        "contact": {
            "law": "hunt_crossley_coulomb",
            "parameters": {},
            "ground": {"normal_policy": "up", "height_m": 0.0},
            "spheres": [
                {
                    "name": "heel_r",
                    "body": "body_a",
                    "position_m": [0.0, 0.0, 0.0],
                    "radius_m": 0.05,
                }
            ],
            "provenance": "test",
        },
        "gravity_m_s2": [0.0, 0.0, -9.81],
        "lower_limb_provenance": "test",
        "provenance": "test",
        "qualification": "test",
        "upper_body_counts": {"bodies": 1, "coordinates": 1, "joints": 1},
        "upper_body_qualification": "test",
        "upper_body_schema_version": 1,
        "upper_body_sha256": "0" * 64,
    }


class FakePinModel:
    """Mock Pinocchio model with configurable dimensions and frame queries."""

    def __init__(self, nq: int = 41, nv: int = 41) -> None:
        self.nq = nq
        self.nv = nv
        self.lowerPositionLimit = np.full(nq, -math.pi)
        self.upperPositionLimit = np.full(nq, math.pi)
        self.velocityLimit = np.full(nv, 10.0)
        self._frames: dict[str, int] = {
            "HeadTop": 1,
            "WaistLeft": 2,
            "ClubTip": 3,
            "heel_r": 4,
            "closure_frame_a": 5,
            "closure_frame_b": 6,
        }

    def getFrameId(self, name: str) -> int:
        if name in self._frames:
            return self._frames[name]
        raise KeyError(f"Frame {name} not found")

    def hasFrame(self, name: str) -> bool:
        return name in self._frames

    def addFrame(self, frame: Any) -> int:
        fid = len(self._frames) + 1
        self._frames[frame.name] = fid
        return fid


class FakePinData:
    def __init__(self) -> None:
        self.oMf = [
            SimpleNamespace(
                translation=np.zeros(3), rotation=np.eye(3), homogeneous=np.eye(4)
            )
            for _ in range(20)
        ]


# ---------------------------------------------------------------------------
# TDD Test Suite for FullBodyPinkTasks
# ---------------------------------------------------------------------------


def test_request_validation_unknown_or_duplicate_labels() -> None:
    """Duplicate or unknown marker names must be rejected before native calls."""
    spec = _mock_spec()
    model = FakePinModel(41, 41)
    facade = FullBodyPinkTasks(spec, model=model)

    # Unknown label
    with pytest.raises(ValueError, match="Unknown marker label"):
        facade.build(
            FrameTaskRequest(
                marker_targets={"UnknownMarker": np.zeros(3)},
                validity_mask={"UnknownMarker": True},
            )
        )

    # Inconsistent keys between targets and mask
    with pytest.raises(ValueError, match="Inventory mismatch"):
        FrameTaskRequest(
            marker_targets={"HeadTop": np.zeros(3)},
            validity_mask={"HeadTop": True, "WaistLeft": False},
        )


def test_zero_observed_targets_raises_insufficient_data() -> None:
    """Zero observed targets must produce an explicit insufficient-data error, not posture fit."""
    spec = _mock_spec()
    model = FakePinModel(41, 41)
    facade = FullBodyPinkTasks(spec, model=model)

    # All masked out
    req_all_false = FrameTaskRequest(
        marker_targets={"HeadTop": np.zeros(3)},
        validity_mask={"HeadTop": False},
    )
    with pytest.raises(
        ValueError, match="Insufficient data: zero observed marker targets"
    ):
        facade.build(req_all_false)

    # Empty dictionary
    req_empty = FrameTaskRequest(
        marker_targets={},
        validity_mask={},
    )
    with pytest.raises(
        ValueError, match="Insufficient data: zero observed marker targets"
    ):
        facade.build(req_empty)


def test_dropout_masks_prevent_masked_targets_from_entering_residuals() -> None:
    """Masked targets (validity=False) must never enter QP residuals / soft tasks."""
    spec = _mock_spec()
    model = FakePinModel(41, 41)
    facade = FullBodyPinkTasks(spec, model=model)

    # HeadTop is valid, ClubTip is invalid (dropout)
    request = FrameTaskRequest(
        marker_targets={
            "HeadTop": np.array([0.1, 0.2, 1.8]),
            "ClubTip": np.array(
                [np.nan, np.nan, np.nan]
            ),  # NaN is allowed when masked!
        },
        validity_mask={
            "HeadTop": True,
            "ClubTip": False,
        },
    )
    bundle = facade.build(request)

    # Only HeadTop task should exist in soft tasks
    task_frames = [t.frame for t in bundle.tasks]
    assert "HeadTop" in task_frames
    assert "ClubTip" not in task_frames


def test_invalid_unmasked_target_fails_validation() -> None:
    """Invalid (NaN/Inf) unmasked target must fail fast before native calls."""
    spec = _mock_spec()
    model = FakePinModel(41, 41)
    facade = FullBodyPinkTasks(spec, model=model)

    request = FrameTaskRequest(
        marker_targets={
            "HeadTop": np.array([0.1, np.nan, 1.8]),
        },
        validity_mask={
            "HeadTop": True,
        },
    )
    with pytest.raises(ValueError, match="Target coordinates must be finite"):
        facade.build(request)


def test_canonical_vs_permuted_coordinate_order() -> None:
    """Coordinate order permutations must be mapped faithfully to the model indices."""
    spec = _mock_spec()
    model = FakePinModel(41, 41)
    facade = FullBodyPinkTasks(spec, model=model)

    # Posture target supplied with permuted keys
    keys = list(spec["coordinate_order"])
    permuted_keys = list(reversed(keys))
    permuted_vals = np.linspace(-0.5, 0.5, len(keys))
    posture_dict = dict(zip(permuted_keys, permuted_vals, strict=True))

    request = FrameTaskRequest(
        marker_targets={"HeadTop": np.array([0.1, 0.2, 1.8])},
        validity_mask={"HeadTop": True},
        posture_target=posture_dict,
    )
    bundle = facade.build(request)
    assert bundle.posture_task is not None
    expected_target = np.array([posture_dict[name] for name in facade.coordinate_order])
    np.testing.assert_allclose(bundle.posture_task.target_q, expected_target)


def test_bound_unit_conversion_and_locked_coordinates() -> None:
    """Degree-to-radian conversion from document bounds and locked coordinate constraints."""
    spec = _mock_spec()
    spec["coordinate_ranges_deg"] = {"coord_0": (-30.0, 45.0)}
    model = FakePinModel(41, 41)
    facade = FullBodyPinkTasks(spec, model=model)

    policy = StanceClosurePolicy(
        locked_coordinates={"coord_1": 0.25},
    )
    request = FrameTaskRequest(
        marker_targets={"HeadTop": np.array([0.1, 0.2, 1.8])},
        validity_mask={"HeadTop": True},
        policy=policy,
    )
    bundle = facade.build(request)

    # Locked coordinate must be translated into equality constraint
    locked_task = [
        c
        for c in bundle.constraints
        if getattr(c, "coordinate_name", None) == "coord_1"
    ]
    assert len(locked_task) == 1
    assert locked_task[0].target_value == 0.25

    # Bounds must be converted to radians and present in limits
    config_limits = [lim for lim in bundle.limits if hasattr(lim, "lower_limit")]
    assert len(config_limits) >= 1
    np.testing.assert_allclose(config_limits[0].lower_limit[0], math.radians(-30.0))
    np.testing.assert_allclose(config_limits[0].upper_limit[0], math.radians(45.0))


def test_displaced_weld_and_near_pi_rejection() -> None:
    """Displaced 6D weld equality task and near-pi rotation branch rejection."""
    spec = _mock_spec()
    model = FakePinModel(41, 41)
    facade = FullBodyPinkTasks(spec, model=model)

    policy = StanceClosurePolicy(enforce_weld=True)
    request = FrameTaskRequest(
        marker_targets={"HeadTop": np.array([0.1, 0.2, 1.8])},
        validity_mask={"HeadTop": True},
        policy=policy,
    )
    bundle = facade.build(request)

    weld_tasks = [c for c in bundle.constraints if getattr(c, "is_weld_closure", False)]
    assert len(weld_tasks) == 1

    # Check near-pi rejection on audit
    q = np.zeros(41)
    bad_rot_residual = np.array([0.0, 0.0, 0.0, 0.0, 0.0, math.pi - 1e-8])
    conf_state = ConfigurationState(
        q=q,
        weld_pose_error=bad_rot_residual,
    )
    with pytest.raises(ValueError, match="near the rotation-pi branch"):
        facade.audit(conf_state)


def test_free_flyer_support_nq_not_equal_nv() -> None:
    """Support floating base models with nq != nv (e.g. nq=48, nv=47 with quaternion)."""
    spec = _mock_spec()
    model = FakePinModel(nq=48, nv=47)
    facade = FullBodyPinkTasks(spec, model=model)

    assert facade.nq == 48
    assert facade.nv == 47


def test_audit_reports_all_residual_categories() -> None:
    """audit reports marker errors, SE(3) translation/rotation closure, bound violations, stance errors."""
    spec = _mock_spec()
    model = FakePinModel(41, 41)
    facade = FullBodyPinkTasks(spec, model=model)

    q = np.zeros(41)
    q[0] = math.radians(100.0)  # Violates 90 deg bound
    conf_state = ConfigurationState(
        q=q,
        marker_positions={"HeadTop": np.array([0.1, 0.2, 1.85])},
        weld_pose_error=np.array([0.01, -0.02, 0.005, 0.001, -0.002, 0.003]),
        stance_contacts={"heel_r": 0.004},
    )
    request = FrameTaskRequest(
        marker_targets={"HeadTop": np.array([0.1, 0.2, 1.8])},
        validity_mask={"HeadTop": True},
    )
    residuals = facade.audit(conf_state, request)
    assert isinstance(residuals, FrameResiduals)
    assert "HeadTop" in residuals.marker_errors_m
    np.testing.assert_allclose(residuals.marker_errors_m["HeadTop"], 0.05, atol=1e-6)
    assert residuals.weld_translation_error_m > 0.0
    assert residuals.weld_rotation_error_rad > 0.0
    assert "coord_0" in residuals.bound_violations
    assert "heel_r" in residuals.stance_errors_m


def test_facade_public_argument_budget() -> None:
    """Public methods must adhere to <= 3 public arguments."""
    import inspect

    build_params = inspect.signature(FullBodyPinkTasks.build).parameters
    assert len(build_params) <= 3

    audit_params = inspect.signature(FullBodyPinkTasks.audit).parameters
    assert len(audit_params) <= 3
