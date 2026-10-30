"""Per-hand constrained mobility on the real driver spec (HCM-3, #14004)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

from src.engines.physics_engines.mujoco.python.hand_mobility import (  # noqa: E402
    HandMobilitySource,
)
from src.shared.python.motion_matching.full_body_spec import (  # noqa: E402
    load_full_body_spec,
)
from src.shared.python.spatial_algebra.operational_space import (  # noqa: E402
    transport_spatial_mobility,
)

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[4]
UPPER = (
    ROOT
    / "docs/development/simscape_tour_matching/native_evidence/native_geometry_spec_9967.json"
)
FULL = ROOT / "docs/development/full_body_models/full_body_spec_v1.json"


@pytest.fixture(scope="module")
def source() -> HandMobilitySource:
    spec = load_full_body_spec(FULL, json.loads(UPPER.read_text(encoding="utf-8")))
    return HandMobilitySource(json.dumps(spec).encode("utf-8"))


@pytest.fixture(scope="module")
def states(source) -> list[np.ndarray]:
    rng = np.random.default_rng(14004)
    n = len(source.coordinate_order)
    return [rng.normal(0.0, 0.15, n) for _ in range(4)]


def test_constrained_mobility_is_symmetric_psd(source, states) -> None:
    for q in states:
        for hm in source.hand_mobility(q).values():
            w = hm.mobility
            assert np.allclose(w, w.T, atol=1e-12)
            assert np.linalg.eigvalsh(w).min() >= -1e-9
            assert (hm.effective_mass_kg > 0).all() and hm.condition >= 1.0
            assert hm.provenance["frame"] == "world"


def test_centres_are_the_grip_points(source, states) -> None:
    from src.shared.python.model_appearance.grip_hands import grip_hand_frame
    from src.shared.python.model_appearance.grip_pose import DEFAULT_GRIP_POSE, Hand
    from src.shared.python.model_appearance.club_assembly import assembly_from_spec

    club = assembly_from_spec(json.loads(json.dumps(source._spec_for_partition)))
    q = states[0]
    coords = dict(zip(source.coordinate_order, q, strict=True))
    rot, origin = source.club_frame_world(coords)
    for hand, hm in source.hand_mobility(q).items():
        local = grip_hand_frame(club, Hand(hand), DEFAULT_GRIP_POSE)[:3, 3]
        assert np.abs(np.array(hm.centre) - (origin + rot @ local)).max() < 1e-3


def test_constrained_is_loewner_below_unconstrained(source, states) -> None:
    for q in states:
        con = source.hand_mobility(q, mode="constrained")
        free = source.hand_mobility(q, mode="unconstrained")
        for hand in con:
            gap = free[hand].mobility - con[hand].mobility
            assert np.linalg.eigvalsh(0.5 * (gap + gap.T)).min() >= -1e-9


def test_rigid_club_body_transport_equality(source, states) -> None:
    """Frame/point bug detector: both grip points ride one rigid body."""
    for q in states:
        lead = source.spatial_mobility(q, "lead")
        trail = source.spatial_mobility(q, "trail")
        coords = dict(zip(source.coordinate_order, q, strict=True))
        r = source.grip_point_world("trail", coords) - source.grip_point_world(
            "lead", coords
        )
        moved = transport_spatial_mobility(lead, r)
        scale = np.abs(trail).max()
        assert np.abs(moved - trail).max() <= 1e-6 * scale


def test_bad_inputs_raise(source) -> None:
    with pytest.raises(ValueError, match="mode"):
        source.hand_mobility(np.zeros(len(source.coordinate_order)), mode="x")
    with pytest.raises(ValueError, match="finite"):
        source.hand_mobility(np.zeros(3))


def test_swing_overlay_adds_two_ellipsoid_geoms_per_frame(source, states) -> None:
    """A swing with the hand mobility overlay adds exactly two ellipsoid geoms per frame (HCM-3, #14004)."""
    import mujoco
    from src.shared.python.motion_matching.same_input import InputBundle
    from src.tools.native_viewer_export.core import SwingInput
    from src.tools.native_viewer_export.overlay import build_overlay_feed
    from src.tools.native_viewer_export.backends._scene_overlays import (
        apply_scene_overlays,
    )

    spec_dict = load_full_body_spec(FULL, json.loads(UPPER.read_text(encoding="utf-8")))
    spec = json.dumps(spec_dict).encode("utf-8")
    coords = tuple(spec_dict["coordinate_order"])
    q = np.array(states)
    bundle = InputBundle(
        spec_bytes=spec,
        coordinate_order=coords,
        dt_s=0.01,
        q0=q[0],
        v0=np.zeros_like(q[0]),
        efforts=np.zeros((len(q) - 1, len(coords))),
        reference_q=q,
        reference_v=np.zeros_like(q),
        reference_engine="mujoco",
    )
    swing = SwingInput(bundle, q, "s", "driver", "mujoco")
    feed, _ = build_overlay_feed(
        swing, "mujoco", hand_mobility=True, show_force=False, show_torque=False
    )

    dummy_model = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
    dummy_data = mujoco.MjData(dummy_model)

    for k in range(len(states)):
        scene = mujoco.MjvScene(dummy_model, maxgeom=100)
        mujoco.mjv_updateScene(
            dummy_model,
            dummy_data,
            mujoco.MjvOption(),
            None,
            mujoco.MjvCamera(),
            mujoco.mjtCatBit.mjCAT_ALL,
            scene,
        )
        before = scene.ngeom
        glyphs = feed.glyphs_at(k)
        assert len(glyphs.ellipsoids) == 2
        apply_scene_overlays(dummy_model, scene, 0, None, None, [glyphs])
        assert scene.ngeom == before + 2
        for g in (scene.geoms[before], scene.geoms[before + 1]):
            assert g.type == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID)

    hud_lines = feed.hud_lines_at(0)
    assert len(hud_lines) >= 1
    assert "Lead mobility:" in hud_lines[0]
    assert "Trail mobility:" in hud_lines[0]
    assert "cond=" in hud_lines[0]
