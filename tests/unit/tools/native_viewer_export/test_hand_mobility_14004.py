"""Exporter wiring of the hand mobility ellipsoids (HCM-3, #14004)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")
import mujoco  # noqa: E402

from src.engines.physics_engines.mujoco.python.hand_mobility import (  # noqa: E402
    HandMobilitySource,
)
from src.shared.python.force_overlay.contracts import ForceTorqueFrame  # noqa: E402
from src.shared.python.motion_matching.full_body_spec import (  # noqa: E402
    load_full_body_spec,
)
from src.tools.native_viewer_export import cli  # noqa: E402
from src.tools.native_viewer_export.backends._scene_overlays import (  # noqa: E402
    apply_scene_overlays,
)
from src.tools.native_viewer_export.core import OverlayFeed  # noqa: E402
from src.tools.native_viewer_export.hand_mobility_glyphs import (  # noqa: E402
    MAX_RADIUS_M,
    MIN_RADIUS_M,
    hand_mobility_glyphs,
)

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[4]
UPPER = (
    ROOT
    / "docs/development/simscape_tour_matching/native_evidence/native_geometry_spec_9967.json"
)
FULL = ROOT / "docs/development/full_body_models/full_body_spec_v1.json"
BASE = ["--bundle", "b.npz", "--out", "o", "--swing", "s"]


@pytest.fixture(scope="module")
def mobilities():
    spec = load_full_body_spec(FULL, json.loads(UPPER.read_text(encoding="utf-8")))
    source = HandMobilitySource(json.dumps(spec).encode("utf-8"))
    q = np.random.default_rng(14004).normal(0.0, 0.15, len(source.coordinate_order))
    return source.hand_mobility(q)


def _feed(mobilities, kind: str) -> OverlayFeed:
    return OverlayFeed(
        lambda i: ForceTorqueFrame(0.0, "mujoco", ()),
        ellipsoids_at=lambda i: hand_mobility_glyphs(mobilities, kind),
    )


def _scene_geoms(feed: OverlayFeed) -> int:
    model = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    scene = mujoco.MjvScene(model, maxgeom=50)
    before = scene.ngeom
    apply_scene_overlays(model, scene, 0, None, None, [feed.glyphs_at(0)])
    return scene.ngeom - before


@pytest.mark.parametrize(
    ("kind", "count"), [("mobility", 2), ("force", 2), ("both", 4)]
)
def test_option_adds_exactly_the_ellipsoid_geoms(mobilities, kind, count) -> None:
    feed = _feed(mobilities, kind)
    assert len(feed.glyphs_at(0).ellipsoids) == count
    assert _scene_geoms(feed) == count


def test_off_adds_none() -> None:
    feed = OverlayFeed(lambda i: ForceTorqueFrame(0.0, "mujoco", ()))
    assert feed.glyphs_at(0).ellipsoids == ()


def test_fixed_scale_clamp_flag_and_wording(mobilities) -> None:
    for g in hand_mobility_glyphs(mobilities, "both"):
        assert min(g.radii) >= MIN_RADIUS_M - 1e-12
        assert max(g.radii) <= MAX_RADIUS_M + 1e-12
        assert "effective mass (kg)" in g.label and "force capacity" not in g.label
    force = hand_mobility_glyphs(mobilities, "force")[0]
    top = max(mobilities["lead"].effective_mass_kg)
    assert ("[clamped]" in force.label) == (0.10 * np.sqrt(top) > MAX_RADIUS_M)
    assert "effective mass" in OverlayFeed.legend(
        _feed(mobilities, "both").glyphs_at(0)
    )


def test_cli_flags() -> None:
    parse = cli.build_parser().parse_args
    assert cli.build_settings(parse(BASE)).hand_mobility is None
    s = cli.build_settings(
        parse(
            [*BASE, "--hand-mobility", "both", "--hand-mobility-mode", "unconstrained"]
        )
    )
    assert (s.hand_mobility, s.hand_mobility_mode) == ("both", "unconstrained")
    with pytest.raises(SystemExit):
        parse([*BASE, "--hand-mobility", "bogus"])


def test_bad_kind_raises(mobilities) -> None:
    with pytest.raises(ValueError, match="kind"):
        hand_mobility_glyphs(mobilities, "x")
