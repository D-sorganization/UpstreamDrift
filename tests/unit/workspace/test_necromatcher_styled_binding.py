"""Exact native identity for an explicitly styled full-body definition."""

import copy
import json

import numpy as np
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.timeout(240)]


@pytest.fixture(scope="module")
def styled():
    """Create current canonical physics and an independently styled MJCF."""
    pytest.importorskip("mujoco")
    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )
    from src.shared.python.model_appearance import document_from_dict
    from src.shared.python.motion_matching.execution.golfer_reference_specs import (
        reference_spec,
    )

    spec = reference_spec("driver")
    appearance = {
        "schema_version": "appearance-v1",
        "skin_tone": "skin_5",
        "head": {"headwear": "none"},
    }
    definition = {**spec, "appearance": appearance}
    xml, _ = export_full_body_mjcf(
        json.dumps(spec).encode(), appearance=document_from_dict(appearance)
    )
    return spec, definition, xml


def test_bind_explicit_appearance_and_authored_ranges(fit_case, tmp_path, styled):
    """Import actual styled XML and preserve its hash through both consumers."""
    from src.shared.python.workspace.necromatcher_native import (
        load_native_model_binding,
    )
    from src.shared.python.workspace.necromatcher_ranges import extract_authored_bounds
    from src.shared.python.motion_matching.pipeline.plant import get_plant

    library, _, _ = fit_case
    spec, definition, xml = styled
    path = tmp_path / "styled.xml"
    path.write_text(xml, encoding="utf-8")
    asset = library.add_model(
        "styled",
        "practice",
        path,
        engine="mujoco",
        dofs=tuple(spec["coordinate_order"]),
    )
    raw = json.dumps(definition).encode()
    plain = get_plant("mujoco", json.dumps(spec).encode())
    bound = load_native_model_binding(
        library, asset.dataset_id, raw, plain.coordinate_units
    )
    assert bound.model_hash == asset.metadata["hash"]
    bounds = extract_authored_bounds(
        raw, bound.model_hash, tuple(spec["coordinate_order"]), plain.coordinate_units
    )
    assert bounds.xml_sha256 == bound.model_hash
    q = np.zeros(len(spec["coordinate_order"]))
    np.testing.assert_array_equal(
        bound.plant.closure_residuals(q), plain.closure_residuals(q)
    )
    # The bound physical-only plant remains identical at a nonzero state.
    import mujoco

    states = []
    for plant in (bound.plant, plain):
        model = plant.adapter.model
        data = mujoco.MjData(model)
        data.qpos[:] = np.linspace(-0.02, 0.02, model.nq)
        data.qvel[:] = np.linspace(0.01, -0.01, model.nv)
        mujoco.mj_forward(model, data)
        states.append(data)
    for field in ("qM", "qfrc_bias", "qacc"):
        np.testing.assert_array_equal(
            getattr(states[0], field), getattr(states[1], field)
        )
    for changed in ("appearance", "physics"):
        altered = copy.deepcopy(definition)
        if changed == "appearance":
            altered["appearance"]["skin_tone"] = "skin_1"
        else:
            altered["bodies"][1]["solids"][0]["mass_kg"] += 1
        with pytest.raises(ValueError, match="hash"):
            load_native_model_binding(
                library,
                asset.dataset_id,
                json.dumps(altered).encode(),
                plain.coordinate_units,
            )
        with pytest.raises(ValueError, match="XML"):
            extract_authored_bounds(
                json.dumps(altered).encode(),
                bound.model_hash,
                tuple(spec["coordinate_order"]),
                plain.coordinate_units,
            )
    invalid = copy.deepcopy(definition)
    invalid["appearance"] = None
    with pytest.raises((ValueError, TypeError)):
        load_native_model_binding(
            library,
            asset.dataset_id,
            json.dumps(invalid).encode(),
            plain.coordinate_units,
        )
