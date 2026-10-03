"""Resolve an authored shaft line from exact solids, without endpoint aliases."""

from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from src.shared.python.motion_matching.club_models import DRIVER, club_solids
from src.shared.python.motion_matching.historical_fit.shaft_geometry import (
    AuthoredShaftAxis,
    resolve_authored_shaft_axis,
)

pytestmark = pytest.mark.unit


def definition():
    body = "prefix/Clubface Vector"
    return {"bodies": [{"name": body, "solids": club_solids(body, DRIVER)}]}


def resolve(document):
    raw = json.dumps(document, allow_nan=False).encode()
    return resolve_authored_shaft_axis(raw, hashlib.sha256(raw).hexdigest())


def test_exact_authored_centers_not_clubhead_frame_or_physical_tips():
    doc = definition()
    doc["frames"] = [{"name": "Clubhead", "placement": "deliberately irrelevant"}]
    axis = resolve(doc)
    assert axis.point_a_m == (0.0, -0.44549999999999995, 0.064)
    assert axis.point_b_m == (0.0, -1.0234999999999999, 0.064)
    assert axis.body == "prefix/Clubface Vector"
    assert axis.resolution_method == "authored_shaft_grip_solid_axis_v1"
    assert axis.physical_geometry_qualified is False
    assert axis.to_record()["semantic"] == "infinite_authored_shaft_axis"


def test_rotated_translated_solids_resolve_independent_local_axis():
    doc = definition()
    rotation = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
    translation = np.array([0.3, -0.1, 0.2])
    expected = []
    for solid in doc["bodies"][0]["solids"]:
        pose = np.asarray(solid["placement"])
        pose[:3, 3] = rotation @ pose[:3, 3] + translation
        pose[:3, :3] = rotation
        solid["placement"] = pose.tolist()
        if solid["name"].endswith(("/Rigid Shaft", "/Grip")):
            expected.append(pose[:3, 3])
    axis = resolve(doc)
    np.testing.assert_allclose([axis.point_a_m, axis.point_b_m], expected)


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "duplicate_body",
        "duplicate_solid",
        "collapse",
        "misaligned",
        "bad_transform",
        "nonzero_com",
    ],
)
def test_unsupported_or_ambiguous_geometry_rejected(fault):
    doc = deepcopy(definition())
    body = doc["bodies"][0]
    if fault == "missing":
        body["solids"] = []
    elif fault == "duplicate_body":
        doc["bodies"].append(deepcopy(body))
    elif fault == "duplicate_solid":
        body["solids"].append(deepcopy(body["solids"][1]))
    elif fault == "collapse":
        body["solids"][2]["placement"] = body["solids"][1]["placement"]
    elif fault == "misaligned":
        body["solids"][2]["placement"][0][3] = 1
    elif fault == "bad_transform":
        body["solids"][1]["placement"][0][0] = 2
    else:
        body["solids"][1]["com_m"] = [1, 0, 0]
    with pytest.raises(ValueError):
        resolve(doc)


def test_exact_definition_hash_required():
    with pytest.raises(ValueError, match="hash"):
        resolve_authored_shaft_axis(json.dumps(definition()).encode(), "f" * 64)


@pytest.mark.parametrize(
    "fault", ["body_name_type", "solid_name_type", "bool_transform"]
)
def test_malformed_declared_types_rejected_explicitly(fault):
    doc = definition()
    if fault == "body_name_type":
        doc["bodies"][0]["name"] = True
    elif fault == "solid_name_type":
        doc["bodies"][0]["solids"][1]["name"] = True
    else:
        doc["bodies"][0]["solids"][1]["placement"][0][0] = True
    with pytest.raises(ValueError):
        resolve(doc)


@pytest.mark.parametrize("point", [(False, 0, 0), ("0", 0, 0)])
def test_public_axis_dto_rejects_boolean_and_string_coordinates(point):
    with pytest.raises(ValueError):
        AuthoredShaftAxis("body", point, (1, 0, 0), "a" * 64, "a" * 64)
