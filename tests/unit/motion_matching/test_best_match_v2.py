"""SHOW-A8 (#14732): the best-match driver model carries the v2 lumbar+thoracic trunk.

The v1 driver spec stays untouched (many engines pin it); the best-match path
consumes the derived ``full_body_spec_anthro_driver_v2.json`` with q migrated by
``TrunkMigrator``.  The migration is a replay of the v1 fit, not a refit.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.model_appearance.body_meshes import body_meshes
from src.shared.python.model_appearance.defaults import default_appearance
from src.shared.python.motion_matching.best_match_v2 import (
    migrate_q_to_v2,
    trunk_segments,
)
from src.shared.python.motion_matching.full_body_v2 import canonical_sha256

pytestmark = pytest.mark.unit
MODELS = Path(__file__).resolve().parents[3] / "docs/development/full_body_models"
V1 = json.loads((MODELS / "full_body_spec_anthro_driver.json").read_text("utf-8"))
V2 = json.loads((MODELS / "full_body_spec_anthro_driver_v2.json").read_text("utf-8"))
TRUNK_COORDS = (
    "lumbar_flexion",
    "lumbar_lateral_bending",
    "lumbar_axial_rotation",
    "thoracic_flexion",
    "thoracic_lateral_bending",
    "thoracic_axial_rotation",
)


def test_v2_driver_spec_has_the_lumbar_and_thoracic_segments() -> None:
    assert set(TRUNK_COORDS) <= set(V2["coordinate_order"])
    assert any(s.endswith("trunk_lumbar") for s in trunk_segments(V2))


def test_v1_driver_spec_is_left_unmodified_without_a_lumbar_segment() -> None:
    assert not set(TRUNK_COORDS) & set(V1["coordinate_order"])
    assert not any(s.endswith("trunk_lumbar") for s in trunk_segments(V1))


def test_v2_spec_is_derived_from_this_exact_v1_spec() -> None:
    assert V2["source_spec_sha256"] == canonical_sha256(V1)


def test_every_v2_body_is_styled_with_skin_and_garment_not_a_grey_capsule() -> None:
    styled: dict[str, set[str]] = {}
    for mesh in body_meshes(V2, default_appearance()):
        styled.setdefault(mesh.body, set()).add(mesh.label.rstrip("0123456789"))
    unstyled = [
        b["name"]
        for b in V2["bodies"]
        if b["name"] != "world"
        and not b["name"].endswith("Head")
        and {"skin", "garment"} - styled.get(b["name"], set())
    ]
    assert unstyled == []


def test_migration_keeps_non_trunk_coordinates_and_fills_the_trunk() -> None:
    order = V1["coordinate_order"]
    rng = np.random.default_rng(3)
    q = 0.05 * rng.standard_normal((4, len(order)))
    out = migrate_q_to_v2(V1, V2, q, order)
    assert out.shape == (4, len(V2["coordinate_order"]))
    for name in order:
        if name in V2["coordinate_order"]:
            np.testing.assert_allclose(
                out[:, V2["coordinate_order"].index(name)], q[:, order.index(name)]
            )
    trunk = [V2["coordinate_order"].index(n) for n in TRUNK_COORDS]
    assert np.isfinite(out[:, trunk]).all()


def test_migration_refuses_a_q_that_is_not_in_v1_coordinate_order() -> None:
    with pytest.raises(ValueError, match="coordinate_order"):
        migrate_q_to_v2(V1, V2, np.zeros((2, 3)), ["a", "b", "c"])
