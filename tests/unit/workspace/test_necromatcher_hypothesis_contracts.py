"""Versioned, immutable hypothesis recipes remain explicitly unqualified."""

from copy import deepcopy
import hashlib
import json

import pytest

pytestmark = pytest.mark.unit


def test_direct_model_requires_supported_definition_byte_serialization() -> None:
    from src.shared.python.workspace.necromatcher_hypothesis_contracts import (
        HypothesisModel,
    )

    record = request_record()["model"]
    canonical = json.dumps(record["definition"], allow_nan=False).encode("utf-8")
    direct = HypothesisModel(
        record["model_id"], record["model_hash"], canonical, record["attachments"]
    )
    assert HypothesisModel.from_record(direct.to_record()).definition_bytes == canonical
    pretty = json.dumps(record["definition"], indent=2, allow_nan=False).encode("utf-8")
    with pytest.raises(ValueError, match="canonical|serialization"):
        HypothesisModel(
            record["model_id"], record["model_hash"], pretty, record["attachments"]
        )


def request_record() -> dict:
    definition = {"coordinate_order": ["joint"], "bodies": [{"name": "arm"}]}
    return {
        "schema_version": "necromatcher/native-hypothesis-request/1",
        "parents": {
            "source_fit_id": "parent",
            "source_fit_hash": "sha256:" + "a" * 64,
            "capture_id": "capture",
            "capture_hash": "sha256:" + "b" * 64,
            "source_sha256": "sha256:" + "c" * 64,
            "source_clock_sha256": "sha256:" + "d" * 64,
        },
        "model": {
            "model_id": "candidate",
            "model_hash": "sha256:" + "e" * 64,
            "definition": definition,
            "attachments": {"wrist": ["arm", [0, 0, 0]]},
        },
        "camera": {
            "intrinsics": [[100, 0, 50], [0, 100, 40], [0, 0, 1]],
            "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "translation": [0, 0, 2],
        },
        "mapping": {
            "coordinate_order": ["joint"],
            "coordinate_units": ["rad"],
            "free_coordinates": ["joint"],
            "reference_pose": [0],
        },
        "gauge": {
            "stature_m": 1.71,
            "world_origin": "authored_ground",
            "world_orientation": "model_world",
            "stature_source": "Generic conditional hypothesis; no subject measurement",
        },
    }


def test_request_roundtrip_is_detached_and_definition_digest_exact() -> None:
    from src.shared.python.workspace.necromatcher_hypothesis_contracts import (
        NativeHypothesisRequest,
    )

    record = request_record()
    request = NativeHypothesisRequest.from_record(record)
    before = deepcopy(record)
    record["model"]["definition"]["bodies"][0]["name"] = "mutated"
    detached = request.to_record()
    detached["model"]["attachments"]["wrist"][1][0] = 99
    assert request.to_record() == before
    assert (
        request.model.definition_sha256
        == "sha256:"
        + hashlib.sha256(
            json.dumps(before["model"]["definition"], allow_nan=False).encode()
        ).hexdigest()
    )
    assert request.camera.translation.flags.writeable is False


@pytest.mark.parametrize(
    "fault",
    [
        "hash",
        "bool_stature",
        "camera",
        "unit",
        "marker",
        "unknown",
        "strictrestart",
        "bool_camera",
        "missing_units",
    ],
)
def test_request_rejects_malformed_or_promoted_recipe(fault: str) -> None:
    from src.shared.python.workspace.necromatcher_hypothesis_contracts import (
        NativeHypothesisRequest,
    )

    record = request_record()
    if fault == "hash":
        record["parents"]["capture_hash"] = "bad"
    elif fault == "bool_stature":
        record["gauge"]["stature_m"] = True
    elif fault == "camera":
        record["camera"]["intrinsics"][0][0] = -1
    elif fault == "unit":
        record["mapping"]["coordinate_units"] = ["deg"]
    elif fault == "marker":
        record["model"]["attachments"]["wrist"][1] = [0, float("nan"), 0]
    elif fault == "unknown":
        record["physical_time_qualified"] = True
    elif fault == "bool_camera":
        record["camera"]["translation"][0] = True
    elif fault == "missing_units":
        record["mapping"]["coordinate_units"] = None
    else:
        record["initialization_source"] = "preserved_spline"
    with pytest.raises(ValueError):
        NativeHypothesisRequest.from_record(record)
