"""Every engine emitter draws the hands of the spec's grip and hand size (#13995)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.model_appearance.club_assembly import assembly_from_spec
from src.shared.python.model_appearance.club_chain_meshes import club_meshes_by_body
from src.shared.python.model_appearance.defaults import default_document_dict
from src.shared.python.model_appearance.grip_pose import GripStyle
from src.shared.python.model_appearance.hand_spec import (
    ResolvedHandSpec,
    resolved_hand_spec,
)
from src.shared.python.model_appearance.schema import document_from_dict
from src.shared.python.motion_matching.execution.grip_rotation import (
    apply_grip_config,
)
from src.shared.python.motion_matching.grip_contract import GripSpec

pytestmark = pytest.mark.unit

MODELS = Path(__file__).resolve().parents[3] / "docs/development/full_body_models"
BASE = json.loads((MODELS / "full_body_spec_anthro_driver.json").read_text("utf-8"))
DOC = document_from_dict(default_document_dict())


def _custom() -> dict:
    spec = apply_grip_config(BASE, GripSpec(style=GripStyle.INTERLOCK))
    spec = copy.deepcopy(spec)
    spec["subject"]["hand_length_m"] = 0.21
    return spec


def _hand_vertices(emitter: str, spec: dict, tmp_path: Path) -> dict[str, np.ndarray]:
    club = assembly_from_spec(spec)
    if emitter == "mujoco":
        by_body = club_meshes_by_body(spec, club, "satin_steel", DOC)
        items = [(n, m) for parts in by_body.values() for n, m, _ in parts]
    elif emitter == "native":
        from src.tools.native_viewer_export.backends._club import club_parts

        items = [(p.name.removeprefix("club_"), p.mesh) for p in club_parts(spec)[1]]
    else:
        from src.engines.physics_engines.opensim.python import club_visuals

        club_visuals.write_club_assets(club, tmp_path, resolved_hand_spec(spec))
        pytest.importorskip("trimesh")
        import trimesh

        items = [
            (n, trimesh.load(p))
            for n, p in club_visuals.hand_asset_paths(club.head_alias, tmp_path).items()
        ]
    return {n: np.asarray(m.vertices) for n, m in items if n.endswith("_hand")}


@pytest.mark.parametrize("emitter", ["mujoco", "native", "opensim"])
def test_emitted_hand_mesh_uses_spec_grip_config(emitter: str, tmp_path: Path) -> None:
    default = _hand_vertices(emitter, BASE, tmp_path / "a")
    custom = _hand_vertices(emitter, _custom(), tmp_path / "b")
    assert set(custom) == {"lead_hand", "trail_hand"}
    for label in custom:
        # The vertex extent scales with the 0.21 m hand and the style changes it.
        ext_default = np.ptp(default[label], axis=0)
        ext_custom = np.ptp(custom[label], axis=0)
        assert not np.allclose(ext_default, ext_custom, rtol=1e-3)
    assert not np.allclose(
        default["trail_hand"].mean(axis=0), custom["trail_hand"].mean(axis=0)
    )


def test_resolved_spec_reads_style_and_hand_length() -> None:
    hands = resolved_hand_spec(_custom())
    assert hands.pose.style is GripStyle.INTERLOCK
    assert hands.hand_length_m == 0.21
    assert hands.grip_radius_m(0.0508) > 0.0


def test_hand_length_outside_range_raises() -> None:
    spec = copy.deepcopy(BASE)
    spec["subject"]["hand_length_m"] = 0.5
    with pytest.raises(ValueError, match="hand_length_m"):
        resolved_hand_spec(spec)
    with pytest.raises(ValueError, match="hand_length_m"):
        ResolvedHandSpec(hand_length_m=0.01)


def test_unknown_style_raises() -> None:
    spec = copy.deepcopy(_custom())
    spec["subject"]["grip_config"]["style"] = "claw"
    with pytest.raises(ValueError):
        resolved_hand_spec(spec)
