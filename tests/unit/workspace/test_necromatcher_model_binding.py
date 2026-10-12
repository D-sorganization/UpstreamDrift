"""Public native candidates are verified before a hypothesis fit exists."""

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def test_candidate_model_binding_requires_no_stored_fit(native_fit_case) -> None:
    from src.shared.python.workspace.necromatcher_native import (
        load_native_model_binding,
    )

    library, _, payload = native_fit_case
    bound = load_native_model_binding(
        library,
        payload["model_id"],
        json.dumps(payload["provenance"]["native_definition"]).encode(),
        tuple(payload["coordinate_units"]),
    )
    assert bound.model_hash == payload["model_hash"]
    assert bound.coordinate_units == tuple(payload["coordinate_units"])
    assert tuple(bound.plant.coordinate_order) == tuple(payload["coordinate_order"])
    assert all(asset.kind != "kinematic_fit" for asset in library.assets("practice"))


@pytest.mark.parametrize("mutation", ["definition", "units", "order", "bytes"])
def test_candidate_binding_rejects_identity_mismatch(
    native_fit_case, mutation: str
) -> None:
    from src.shared.python.workspace.necromatcher_native import (
        load_native_model_binding,
    )

    library, _, payload = native_fit_case
    definition = payload["provenance"]["native_definition"]
    units = list(payload["coordinate_units"])
    if mutation == "definition":
        definition["bodies"][1]["solids"][0]["mass_kg"] += 1
    elif mutation == "units":
        units[0] = "rad"
    elif mutation == "order":
        definition["coordinate_order"].reverse()
    else:
        (library.root / library.load_asset(payload["model_id"]).path).write_text(
            "changed", encoding="utf-8"
        )
    with pytest.raises(ValueError, match="hash|units|order|model"):
        load_native_model_binding(
            library, payload["model_id"], json.dumps(definition).encode(), tuple(units)
        )
