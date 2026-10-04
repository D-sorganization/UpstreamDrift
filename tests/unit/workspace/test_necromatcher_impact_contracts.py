"""Explicit immutable geometry/flight assumptions; no historical calibration."""

from dataclasses import replace

import numpy as np
import pytest
from src.shared.python.workspace.necromatcher_impact import (
    ReplayImpactGeometry,
    ReplayImpactSelection,
)

pytestmark = pytest.mark.unit


def geometry() -> ReplayImpactGeometry:
    return ReplayImpactGeometry(
        "club",
        (0, 0.1, 0),
        (1, 0, 0),
        (0, 0, 1),
        0.2,
        0.005,
        "Authored point/face/effective mass and inertia",
    )


def selection() -> ReplayImpactSelection:
    return ReplayImpactSelection(
        1, np.eye(3), (0, 0, 0), "Authored recorded impact sample"
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("body", []),
        ("local_head_point_m", (False, 0, 0)),
        ("local_head_point_m", (float("nan"), 0, 0)),
        ("local_face_normal", (2, 0, 0)),
        ("local_face_up", (1, 0, 0)),
        ("mass_kg", True),
        ("moi_kg_m2", 0),
        ("assumption_description", ""),
    ],
)
def test_geometry_rejects_malformed_assumptions(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        replace(geometry(), **{field: value})


@pytest.mark.parametrize(
    "field,value",
    [
        ("recorded_sample_index", True),
        ("recorded_sample_index", -1),
        ("world_to_flight_rotation", np.diag([1, 1, -1])),
        ("world_to_flight_rotation", np.eye(3) * 2),
        ("world_to_flight_translation_m", (0, float("inf"), 0)),
        ("selection_description", ""),
    ],
)
def test_selection_rejects_bad_transform_or_sample(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        replace(selection(), **{field: value})


def test_contracts_copy_arrays_and_expose_detached_records() -> None:
    point = np.array([0.0, 0.1, 0.0])
    rotation = np.eye(3)
    g = replace(geometry(), local_head_point_m=point)
    s = replace(selection(), world_to_flight_rotation=rotation)
    point[:] = 9
    rotation[:] = 9
    assert g.local_head_point_m == (0, 0.1, 0)
    assert s.world_to_flight_rotation[0] == (1, 0, 0)
    record = g.to_record()
    record["local_head_point_m"][0] = 123
    assert g.local_head_point_m[0] == 0
    assert ReplayImpactGeometry.from_record(g.to_record()) == g
    assert ReplayImpactSelection.from_record(s.to_record()) == s


def test_direct_record_rejects_extra_fields() -> None:
    with pytest.raises(ValueError):
        ReplayImpactGeometry.from_record({**geometry().to_record(), "qualified": True})
