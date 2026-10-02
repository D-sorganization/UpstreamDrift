"""Compiled coordinate units and exact historical native bindings are checked."""

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching.pipeline import plant

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"


def test_units_follow_compiled_joint_types_and_declared_order():
    mj = pytest.importorskip("mujoco")
    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )

    definition = json.loads(SPEC.read_text())
    definition["coordinate_order"] = list(reversed(definition["coordinate_order"]))
    source = json.dumps(definition).encode()
    native = plant.get_plant("mujoco", source)
    xml, _ = export_full_body_mjcf(source)
    compiled = mj.MjModel.from_xml_string(xml)
    units = {int(mj.mjtJoint.mjJNT_SLIDE): "m", int(mj.mjtJoint.mjJNT_HINGE): "rad"}
    expected = tuple(
        units[int(compiled.joint(name).type[0])] for name in native.coordinate_order
    )
    assert native.coordinate_units == expected
    assert expected.count("m") == 3
    assert expected.count("rad") == 41


def test_binding_compiles_once_for_projection_and_downstream_use(
    native_fit_case, monkeypatch
):
    from src.shared.python.workspace import necromatcher_native as boundary

    library, source, _ = native_fit_case
    library.add_fit("native-fit", "practice", source)
    calls = []
    original = boundary.get_plant

    def tracked(engine, definition):
        calls.append(engine)
        return original(engine, definition)

    monkeypatch.setattr(boundary, "get_plant", tracked)
    bound = boundary.load_native_fit_binding(library, "native-fit")
    assert bound.coordinate_units == ("m",) * 3 + ("rad",) * 41
    assert bound.project(0)["physical_time_qualified"] is False
    assert bound.project(2)["qualification"] == "monocular_research_hypothesis"
    assert len(calls) == 1
    assert len(bound.plant.coordinate_order) == 44


@pytest.mark.parametrize("invalid", ["units", "definition"])
def test_mismatched_declared_native_units_or_definition_rejected(
    native_fit_case, invalid
):
    from src.shared.python.workspace.necromatcher_native import load_native_fit_binding

    library, source, payload = native_fit_case
    if invalid == "units":
        payload["coordinate_units"][0] = "rad"
    else:
        payload["provenance"]["native_definition"]["bodies"][1]["solids"][0][
            "mass_kg"
        ] += 1
    source.write_text(json.dumps(payload))
    library.add_fit("bad-native", "practice", source)
    with pytest.raises(ValueError, match="native model|units"):
        load_native_fit_binding(library, "bad-native")


def test_missing_compiled_unit_capability_never_assumes_radians(
    native_fit_case, monkeypatch
):
    from types import SimpleNamespace
    from src.shared.python.workspace import necromatcher_native as boundary

    library, source, payload = native_fit_case
    library.add_fit("native-fit", "practice", source)
    monkeypatch.setattr(
        boundary,
        "get_plant",
        lambda *_: SimpleNamespace(coordinate_order=tuple(payload["coordinate_order"])),
    )
    with pytest.raises(ValueError, match="compiled coordinate units"):
        boundary.load_native_fit_binding(library, "native-fit")


@pytest.mark.parametrize("camera", [None, {}, {"unexpected": 1}])
def test_malformed_camera_mapping_raises_boundary_value_error(native_fit_case, camera):
    from src.shared.python.workspace.necromatcher_native import load_native_fit_binding

    library, source, payload = native_fit_case
    payload["evidence"]["original_fit"]["camera"] = camera
    source.write_text(json.dumps(payload))
    library.add_fit("bad-camera", "practice", source)
    with pytest.raises(ValueError, match="Camera"):
        load_native_fit_binding(library, "bad-camera").review_inputs()


def test_semantic_definition_key_order_does_not_change_native_binding(native_fit_case):
    from src.shared.python.workspace.necromatcher_native import load_native_fit_binding

    library, source, payload = native_fit_case
    original = payload["provenance"]["native_definition"]
    payload["provenance"]["native_definition"] = dict(reversed(list(original.items())))
    source.write_text(json.dumps(payload))
    library.add_fit("reordered", "practice", source)
    assert len(load_native_fit_binding(library, "reordered").coordinate_units) == 44


@pytest.fixture
def bound_controls(native_fit_case):
    from src.shared.python.workspace.necromatcher_native import load_native_fit_binding

    library, source, fit = native_fit_case
    saved = library.add_fit("native-fit", "practice", source)
    payload = {
        "schema_version": "necromatcher/effort-profile/2",
        "model_id": fit["model_id"],
        "model_hash": fit["model_hash"],
        "fit_id": saved.dataset_id,
        "fit_hash": saved.metadata["hash"],
        "dofs": fit["coordinate_order"],
        "coordinate_units": fit["coordinate_units"],
        "effort_units": ["N"] * 3 + ["N*m"] * 41,
        "timebase": "physical_seconds",
        "provenance": {
            "kind": "authored",
            "description": "Synthetic operator controls; not measured efforts",
        },
        "segments": [
            {
                "start_s": 0.0,
                "end_s": 1.0,
                "is_bernstein": True,
                "coefficients": [[float(index)] for index in range(44)],
            }
        ],
    }
    path = source.parent / "controls.json"
    path.write_text(json.dumps(payload))
    library.add_profile("controls", "practice", path)
    return load_native_fit_binding(library, "native-fit"), library.load_effort_profile(
        "controls", fit["model_id"]
    )


def test_unit_checked_controls_map_to_native_effort_channels(bound_controls):
    bound, controls = bound_controls
    efforts = bound.efforts(controls, 0.5)
    assert tuple(efforts) == bound.plant.coordinate_order
    assert list(efforts.values()) == list(range(44))
    with pytest.raises(ValueError):
        bound.efforts(controls, 1.5)


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_id", "other-model"),
        ("model_hash", "wrong-model-hash"),
        ("fit_id", "other-fit"),
        ("fit_hash", "wrong-fit-hash"),
        ("dofs", ("wrong-coordinate",)),
        ("coordinate_units", ("rad",) * 44),
        ("effort_units", ("N*m",) * 44),
    ],
)
def test_controls_cannot_drive_different_native_bindings(bound_controls, field, value):
    from dataclasses import replace

    bound, controls = bound_controls
    with pytest.raises(ValueError, match="Controls"):
        bound.efforts(replace(controls, **{field: value}), 0.5)


def test_mutating_detached_fit_cannot_change_compiled_control_identity(bound_controls):
    from dataclasses import replace

    bound, controls = bound_controls
    bound.fit["model_hash"] = "forged-native-model"
    with pytest.raises(ValueError, match="Controls"):
        bound.efforts(replace(controls, model_hash="forged-native-model"), 0.5)


def test_native_frame_queries_preserve_requested_model_aliases(native_fit_case):
    library, path, payload = native_fit_case
    library.add_fit("alias-fit", "practice", path)
    from src.shared.python.workspace import load_native_fit_binding

    binding = load_native_fit_binding(library, "alias-fit")
    closure = payload["provenance"]["native_definition"]["closure"]
    mapping = {
        name: (closure[key], [0, 0, 0])
        for name, key in (("hand", "body_a"), ("club", "body_b"))
    }
    q = np.array(payload["q"][0])
    poses = binding.plant.frame_poses(mapping, q)
    assert set(poses) == {record[0] for record in mapping.values()}
    points = binding.plant.marker_positions(q, mapping)
    for index, (body, _) in enumerate(mapping.values()):
        rotation, translation = poses[body]
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(translation, points[index], atol=1e-12)


@pytest.mark.parametrize("invalid", [np.zeros(43), np.full(44, np.nan)])
def test_native_frame_query_rejects_invalid_coordinates(native_fit_case, invalid):
    library, path, payload = native_fit_case
    library.add_fit("alias-fit", "practice", path)
    from src.shared.python.workspace import load_native_fit_binding

    binding = load_native_fit_binding(library, "alias-fit")
    body = payload["provenance"]["native_definition"]["closure"]["body_a"]
    with pytest.raises(ValueError, match="finite vector"):
        binding.plant.frame_poses({"hand": (body, [0, 0, 0])}, invalid)


def test_native_frame_query_rejects_unknown_frame(native_fit_case):
    library, path, payload = native_fit_case
    library.add_fit("alias-fit", "practice", path)
    from src.shared.python.workspace import load_native_fit_binding

    binding = load_native_fit_binding(library, "alias-fit")
    with pytest.raises(ValueError, match="unknown body"):
        binding.plant.frame_poses(
            {"hand": ("missing-frame", [0, 0, 0])}, payload["q"][0]
        )
