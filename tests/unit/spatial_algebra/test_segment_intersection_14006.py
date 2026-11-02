"""Unit tests for segment intersection mobility points (HCM-5, #14006)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.engines.physics_engines.mujoco.python.hand_mobility import HandMobility
from src.shared.python.spatial_algebra.segment_intersection import (
    IntersectionPoint,
    extract_intersection_points,
    resolve_intersection_points,
)
from src.tools.native_viewer_export import cli
from src.tools.native_viewer_export.overlay import (
    OverlayOptions,
    _build_hand_ellipsoid_feed,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]
SPEC_PATH = (
    Path(__file__).resolve().parents[3]
    / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
)


@pytest.fixture(scope="module")
def anthro_spec() -> dict:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


def test_extract_and_round_trip(anthro_spec: dict) -> None:
    pts = extract_intersection_points(anthro_spec)
    expected = ("lead_hand", "trail_hand", "lead_wrist", "trail_wrist") + (
        "lead_elbow",
        "trail_elbow",
        "lead_shoulder",
        "trail_shoulder",
        "pelvis",
    )
    for name in expected:
        pt = pts[name]
        assert pt.name == name and pt.group and pt.body
        assert len(pt.anchor_local) == 3 and len(pt.color) == 4
        assert np.all(np.isfinite(pt.anchor_local)) and all(
            0 <= c <= 1 for c in pt.color
        )
        assert IntersectionPoint.from_dict(pt.to_dict()) == pt


def test_resolve_points_and_clutter_control(anthro_spec: dict) -> None:
    def res(sel="hands", max_p=None):
        return [
            p.name
            for p in resolve_intersection_points(
                anthro_spec, selector=sel, max_points=max_p
            )
        ]

    for sel, exp in [
        ("hands", ["lead_hand", "trail_hand"]),
        ("wrists", ["lead_wrist", "trail_wrist"]),
        ("elbows", ["lead_elbow", "trail_elbow"]),
        ("shoulders", ["lead_shoulder", "trail_shoulder"]),
        ("hands,wrists", ["lead_hand", "trail_hand", "lead_wrist", "trail_wrist"]),
    ]:
        assert res(sel) == exp
    assert res("hands,wrists,elbows", 3) == ["lead_hand", "trail_hand", "lead_wrist"]
    with pytest.raises(ValueError, match="unknown mobility point"):
        resolve_intersection_points(anthro_spec, selector="knees")


def test_overlay_feed_and_cli_options() -> None:
    spec = b'{"frames": {"LW": {"anchor": [0.1, 0.2, 0.3]}, "RW": {"anchor": [-0.1, 0.2, 0.3]}}}'
    eye, v1, z3 = np.eye(3), np.ones(3), np.zeros(3, bool)

    def _hm(n: str) -> HandMobility:
        c = 5.0 if "lead" in n else 50.0
        return HandMobility(n, (0.1, 0.2, 0.3), eye, eye, v1, v1, c, z3, {})

    def seg_mob(idx: int, points: Any = None, **kw: Any):
        return {p.name: _hm(p.name) for p in points or []}

    swing = SimpleNamespace(
        bundle=SimpleNamespace(spec_bytes=spec, coordinate_order=("j1",)),
        n_states=10,
        q=np.zeros((10, 1)),
        segment_mobility_at=seg_mob,
    )
    g_fn, h_fn = _build_hand_ellipsoid_feed(
        swing,  # type: ignore[arg-type]
        active_ellipsoids=("mobility",),
        hand_mobility_mode="constrained",
        ellipsoid_scale="auto",
        mobility_points="wrists",
    )
    assert len(g_fn(0)) == 2 and "Lead_wrist" in h_fn(0)[0]
    g_cut, h_cut = _build_hand_ellipsoid_feed(
        swing,  # type: ignore[arg-type]
        active_ellipsoids=("mobility",),
        hand_mobility_mode="constrained",
        ellipsoid_scale="auto",
        mobility_points="wrists",
        mobility_condition_cutoff=10.0,
    )
    assert len(g_cut(0)) == 1 and "Trail_wrist" not in h_cut(0)[0]

    args = [
        "--bundle",
        "b",
        "--out",
        "o",
        "--swing",
        "s",
        "--mobility-points",
        "wrists",
    ]
    args += ["--max-mobility-points", "4", "--mobility-condition-cutoff", "150.0"]
    s = cli.build_settings(cli.build_parser().parse_args(args))
    assert s.mobility_points == "wrists" and s.max_mobility_points == 4
    assert s.mobility_condition_cutoff == 150.0
    opts = OverlayOptions(
        mobility_points="elbows", max_mobility_points=6, mobility_condition_cutoff=200.0
    )
    assert opts.mobility_points == "elbows" and opts.max_mobility_points == 6
    assert opts.mobility_condition_cutoff == 200.0
