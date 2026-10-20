"""R6 (#12686) criterion 6: the chain shaft renders with visible deflection.

With ``club.shaft.model = "chain"`` the MuJoCo visual layers (plain
``visual=True`` and the appearance layer) draw the shaft as one piece per
chain body: the grip, both grip hands and the root shaft piece stay on the
club (grip) body, each link carries its own shaft piece and the head mesh sits
on the tip link, so bending the chain bends the drawn shaft and moves the
drawn head. The rigid default draws the whole club on the club body.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import defusedxml.ElementTree as DET
import numpy as np
import pytest

from src.shared.python.model_appearance import document_from_dict
from src.shared.python.model_appearance.club_assembly import (
    assembly_from_spec,
    assembly_meshes,
)
from src.shared.python.model_appearance.club_chain_meshes import (
    chain_layout,
    shaft_piece_meshes,
)
from src.shared.python.motion_matching.club_models import DRIVER
from src.shared.python.motion_matching.club_shaft_chain import (
    DEFAULT_LINKS,
    apply_shaft_chain,
    build_shaft_chain,
)

mujoco = pytest.importorskip("mujoco")

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
SPEC_PATH = (
    ROOT
    / "docs/development/full_body_models/evidence/ground_support/anthro_driver_seeds"
    / "full_body_spec_hipcal_scaled.json"
)
APPEARANCE = {"schema_version": "appearance-v1"}
CLUB_PART = re.compile(r"vmesh_\w*?_(shaft(_link\d+)?|grip|head|lead_hand|trail_hand)")


@pytest.fixture(scope="module")
def rigid_spec() -> dict[str, Any]:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def chain_spec(rigid_spec: dict[str, Any]) -> dict[str, Any]:
    return apply_shaft_chain(rigid_spec, build_shaft_chain(DRIVER))


def _export(spec: dict[str, Any], *, appearance: bool) -> str:
    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )

    raw = json.dumps(spec).encode()
    if appearance:
        return export_full_body_mjcf(raw, appearance=document_from_dict(APPEARANCE))[0]
    return export_full_body_mjcf(raw, visual=True)[0]


def _club_meshes_by_body(xml: str) -> dict[str, list[str]]:
    """Club-part mesh geoms (``vmesh_<..>_<club part>``) per body."""
    out: dict[str, list[str]] = {}
    for body in DET.fromstring(xml).iter("body"):
        for geom in body.findall("geom"):
            mesh = geom.get("mesh") or ""
            if CLUB_PART.fullmatch(mesh):
                out.setdefault(body.get("name", ""), []).append(mesh)
    return out


def test_layout_is_none_for_rigid_and_lists_links_for_chain(
    rigid_spec: dict[str, Any], chain_spec: dict[str, Any]
) -> None:
    assert chain_layout(rigid_spec) is None
    layout = chain_layout(chain_spec)
    assert layout is not None
    shaft = chain_spec["club"]["shaft"]
    assert layout.club_body == shaft["club_body"]
    assert layout.tip_link == shaft["tip_link"] == layout.links[-1].body
    assert len(layout.links) == DEFAULT_LINKS
    # Links tile the exposed span from the root joint to the tip, in order.
    ys = [(link.start_y_m, link.end_y_m) for link in layout.links]
    for (_, end), (start, _) in zip(ys, ys[1:], strict=False):
        assert end == pytest.approx(start, abs=1e-12)
    assert ys[-1][1] == pytest.approx(0.0, abs=1e-12)
    assert all(start < end for start, end in ys)


def test_shaft_pieces_tile_the_rigid_shaft(chain_spec: dict[str, Any]) -> None:
    club = assembly_from_spec(chain_spec)
    layout = chain_layout(chain_spec)
    assert club is not None and layout is not None
    rigid = assembly_meshes(club)["shaft"].vertices
    pieces = shaft_piece_meshes(club, layout)
    assert set(pieces) == {layout.club_body, *(link.body for link in layout.links)}
    spans = sorted(
        (float(m.vertices[:, 1].min()), float(m.vertices[:, 1].max()))
        for m in pieces.values()
    )
    assert spans[0][0] == pytest.approx(rigid[:, 1].min(), abs=1e-12)
    assert spans[-1][1] == pytest.approx(rigid[:, 1].max(), abs=1e-12)
    for (_, top), (bottom, _) in zip(spans, spans[1:], strict=False):
        assert top == pytest.approx(bottom, abs=1e-12)
    # Same tube radius as the rigid shaft at the cut (linear taper).
    radial = np.hypot(rigid[:, 0], rigid[:, 2] - club.axis_offset_m)
    for mesh in pieces.values():
        r = np.hypot(mesh.vertices[:, 0], mesh.vertices[:, 2] - club.axis_offset_m)
        assert r.max() <= radial.max() + 1e-12


@pytest.mark.parametrize("appearance", [False, True], ids=["plain", "appearance"])
def test_chain_club_meshes_sit_on_their_chain_bodies(
    chain_spec: dict[str, Any], appearance: bool
) -> None:
    layout = chain_layout(chain_spec)
    assert layout is not None
    by_body = _club_meshes_by_body(_export(chain_spec, appearance=appearance))
    club = by_body[layout.club_body]
    assert any(m.endswith("_grip") for m in club)
    assert any(m.endswith("_lead_hand") for m in club)
    assert any(m.endswith("_trail_hand") for m in club)
    assert sum("shaft" in m for m in club) == 1
    assert not any(m.endswith("_head") for m in club)
    for link in layout.links:
        meshes = by_body[link.body]
        assert sum("shaft" in m for m in meshes) == 1, link.body
        assert any(m.endswith("_head") for m in meshes) == (
            link.body == layout.tip_link
        )
    heads = [m for ms in by_body.values() for m in ms if m.endswith("_head")]
    assert len(heads) == 1


@pytest.mark.parametrize("appearance", [False, True], ids=["plain", "appearance"])
def test_rigid_club_meshes_stay_on_the_club_body(
    rigid_spec: dict[str, Any], appearance: bool
) -> None:
    by_body = _club_meshes_by_body(_export(rigid_spec, appearance=appearance))
    assert len(by_body) == 1
    (meshes,) = by_body.values()
    assert sum(m.endswith("_shaft") for m in meshes) == 1
    assert sum(m.endswith("_head") for m in meshes) == 1


def test_bent_chain_moves_the_drawn_head_not_the_grip(
    chain_spec: dict[str, Any],
) -> None:
    model = mujoco.MjModel.from_xml_string(_export(chain_spec, appearance=False))
    data = mujoco.MjData(model)

    def geom_with(suffix: str) -> int:
        return next(
            g
            for g in range(model.ngeom)
            if (model.geom(g).name or "").endswith(f"vmesh_club_{suffix}")
        )

    head, grip = geom_with("head"), geom_with("grip")
    mujoco.mj_forward(model, data)
    head0, grip0 = data.geom_xpos[head].copy(), data.geom_xpos[grip].copy()
    for name in chain_spec["club"]["shaft"]["coordinates"]:
        if name.startswith("shaft_bend_x_"):
            data.qpos[model.joint(name).qposadr[0]] = 0.05
    mujoco.mj_forward(model, data)
    assert np.linalg.norm(data.geom_xpos[head] - head0) > 0.02
    np.testing.assert_allclose(data.geom_xpos[grip], grip0, atol=1e-12)
