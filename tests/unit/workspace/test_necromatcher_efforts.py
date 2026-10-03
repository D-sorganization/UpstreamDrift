"""Mixed-coordinate authored controls retain units and immutable bindings."""

import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.unit


@pytest.fixture
def effort_case(fit_case, tmp_path):
    library, source, fit = fit_case
    model = library.add_model(
        "mixed-model",
        "practice",
        tmp_path / "model.xml",
        engine="mujoco",
        dofs=("root_x", "hip"),
    )
    fit.update(
        model_id=model.dataset_id,
        model_hash=model.metadata["hash"],
        coordinate_order=["root_x", "hip"],
        coordinate_units=["m", "rad"],
        q=[[0.1, 0.2], [0.2, 0.3]],
    )
    source.write_text(json.dumps(fit))
    saved = library.add_fit("mixed-fit", "practice", source)
    profile = {
        "schema_version": "necromatcher/effort-profile/2",
        "model_id": model.dataset_id,
        "model_hash": model.metadata["hash"],
        "fit_id": saved.dataset_id,
        "fit_hash": saved.metadata["hash"],
        "dofs": fit["coordinate_order"],
        "coordinate_units": ["m", "rad"],
        "effort_units": ["N", "N*m"],
        "timebase": "physical_seconds",
        "provenance": {
            "kind": "authored",
            "description": "Operator controls; no measured forces",
        },
        "segments": [
            {
                "start_s": 0.0,
                "end_s": 1.0,
                "coefficients": [[1.0, 3.0], [2.0, 4.0]],
                "is_bernstein": True,
            }
        ],
    }
    path = tmp_path / "effort.json"
    path.write_text(json.dumps(profile))
    return library, path, profile


def test_mixed_profile_recall_and_export(effort_case, tmp_path):
    library, path, _ = effort_case
    library.add_profile("effort", "practice", path)
    controls = library.load_effort_profile("effort", "mixed-model")
    assert controls.coordinate_units == ("m", "rad")
    assert controls.effort_units == ("N", "N*m")
    assert controls.fit_id == "mixed-fit"
    assert controls.dofs == ("root_x", "hip")
    assert controls.model_hash == library.load_asset("mixed-model").metadata["hash"]
    assert controls.fit_hash == library.load_asset("mixed-fit").metadata["hash"]
    np.testing.assert_allclose(controls.evaluate(0.5), [2.0, 3.0])
    library.export_swing("practice", tmp_path / "export.zip")
    for time in (-0.1, 1.1, float("nan"), True):
        with pytest.raises(ValueError):
            controls.evaluate(time)
    with pytest.raises(ValueError, match="units"):
        library.load_torque("effort", "mixed-model")


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_hash", "0" * 64),
        ("fit_hash", "0" * 64),
        ("effort_units", ["N*m", "N*m"]),
        ("coordinate_units", ["rad", "m"]),
        ("dofs", ["hip", "root_x"]),
        ("timebase", "presentation_seconds"),
        ("unexpected", True),
    ],
)
def test_invalid_binding_is_not_published(effort_case, field, value):
    library, path, payload = effort_case
    payload[field] = value
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        library.add_profile("invalid", "practice", path)
    assert all(asset.dataset_id != "invalid" for asset in library.assets("practice"))


@pytest.mark.parametrize(
    "change",
    [
        {"coefficients": [[True, False], [1, 2]]},
        {"coefficients": [["1", "2"], ["3", "4"]]},
        {"is_bernstein": "yes"},
        {"start_s": False},
    ],
)
def test_strict_segment_numbers(effort_case, change):
    library, path, payload = effort_case
    payload["segments"][0].update(change)
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        library.add_profile("invalid", "practice", path)


def test_legacy_torque_cannot_hide_known_translation(effort_case):
    library, path, payload = effort_case
    payload = {
        key: payload[key]
        for key in ("model_id", "dofs", "timebase", "provenance", "segments")
    }
    payload.update(schema_version="necromatcher/torque-profile/1", units="N*m")
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="translation"):
        library.add_profile("legacy", "practice", path)


def test_export_revalidates_legacy_profile_against_new_fit(effort_case, tmp_path):
    library, path, payload = effort_case
    legacy = {
        key: payload[key]
        for key in ("model_id", "dofs", "timebase", "provenance", "segments")
    }
    legacy.update(schema_version="necromatcher/torque-profile/1", units="N*m")
    path.write_text(json.dumps(legacy))
    # Represent an old saved asset admitted before translation units were known.
    library._save_asset(
        "legacy",
        "practice",
        path,
        "torque_profile",
        {"schema": legacy["schema_version"], "model_id": legacy["model_id"]},
    )
    with pytest.raises(ValueError, match="translation"):
        library.export_swing("practice", tmp_path / "invalid.zip")
    assert not (tmp_path / "invalid.zip").exists()


def test_cross_swing_and_wrong_fit_binding_rejected(effort_case):
    library, path, payload = effort_case
    library.add_swing("other", "hogan", "Other")
    with pytest.raises(ValueError, match="same swing"):
        library.add_profile("cross", "other", path)
    payload["fit_id"] = "mixed-model"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="same swing"):
        library.add_profile("wrong-kind", "practice", path)


def test_profile_coefficients_cannot_be_mutated(effort_case):
    library, path, _ = effort_case
    library.add_profile("effort", "practice", path)
    controls = library.load_effort_profile("effort", "mixed-model")
    values = controls._curve.segments[0].coefficients
    with pytest.raises(ValueError):
        values[0, 0] = 99
    with pytest.raises(ValueError):
        values.setflags(write=True)
    np.testing.assert_allclose(controls.evaluate(0.5), [2.0, 3.0])


def test_evaluation_overflow_is_rejected(effort_case):
    library, path, payload = effort_case
    payload["segments"][0].update(
        is_bernstein=False, coefficients=[[1e308, 1e308], [0, 0]]
    )
    path.write_text(json.dumps(payload))
    library.add_profile("large", "practice", path)
    with pytest.raises(ValueError, match="finite"):
        library.load_effort_profile("large", "mixed-model").evaluate(1.0)


def test_local_api_import_preserves_mixed_profile(effort_case):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from src.api.routes.necromatcher import get_library, router

    library, path, payload = effort_case
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_library] = lambda: library
    with TestClient(app) as client:
        response = client.post(
            "/necromatcher/swings/practice/profiles",
            json={"id": "api-effort", "source_path": str(path)},
        )
        assert response.status_code == 201
        assert "path" not in response.json()
        assert library.load_effort_profile(
            "api-effort", "mixed-model"
        ).effort_units == ("N", "N*m")
        payload["effort_units"] = ["N*m", "N*m"]
        path.write_text(json.dumps(payload))
        assert (
            client.post(
                "/necromatcher/swings/practice/profiles",
                json={"id": "invalid-api", "source_path": str(path)},
            ).status_code
            == 422
        )


def test_effort_profile_crosses_canonical_handoff_without_qualification(effort_case):
    from src.shared.python.workspace import (
        ArtifactKind,
        ArtifactReference,
        SessionProjectStore,
        WorkspaceHandoff,
    )

    library, path, _ = effort_case
    asset = library.add_profile("effort", "practice", path)
    controls = library.load_effort_profile("effort", "mixed-model")
    reference = ArtifactReference(
        asset.dataset_id,
        asset.path,
        asset.metadata["hash"],
        asset.metadata["schema"],
        ArtifactKind.DRIVING_PROFILE,
    )
    handoff = WorkspaceHandoff(
        handoff_id="research-controls",
        project_id="necromatcher",
        session_id="practice",
        subject_id="hogan",
        engine="mujoco",
        model_id=controls.model_id,
        club={},
        frame="model",
        units=dict(zip(controls.dofs, controls.effort_units, strict=True)),
        timebase={"kind": "authored_physical_seconds", "source_clock_qualified": False},
        parameters={"fit_id": controls.fit_id, "fit_hash": controls.fit_hash},
        inputs=(reference,),
        qualification={"passed": False, "kind": "authored_controls"},
    )
    store = SessionProjectStore(library.root)
    saved = store.register_run(handoff)
    recalled = store.load_run(saved.run_id)
    assert recalled.units == {"root_x": "N", "hip": "N*m"}
    assert recalled.qualification["passed"] is False
    assert recalled.status == "draft"
    recalled.inputs[0].verify_on_disk(library.root)
