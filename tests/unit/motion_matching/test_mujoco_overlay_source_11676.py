"""MuJoCo overlay source drives BundleOverlayProvider on the real spec (NV-3, #11676)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

from src.engines.physics_engines.mujoco.python.overlay_source import (  # noqa: E402
    MujocoOverlaySource,
)
from src.shared.python.force_overlay import WrenchKind  # noqa: E402
from src.shared.python.force_overlay.bundle_provider import (  # noqa: E402
    BundleOverlayProvider,
    ContactSource,
    KinematicsSource,
)
from src.shared.python.motion_matching.full_body_spec import (  # noqa: E402
    load_full_body_spec,
)
from src.shared.python.motion_matching.same_input import InputBundle  # noqa: E402

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
UPPER = (
    ROOT
    / "docs/development/simscape_tour_matching/native_evidence/native_geometry_spec_9967.json"
)
FULL = ROOT / "docs/development/full_body_models/full_body_spec_v1.json"


@pytest.fixture(scope="module")
def spec_bytes() -> bytes:
    spec = load_full_body_spec(FULL, json.loads(UPPER.read_text(encoding="utf-8")))
    return json.dumps(spec).encode("utf-8")


@pytest.fixture(scope="module")
def source(spec_bytes: bytes) -> MujocoOverlaySource:
    return MujocoOverlaySource(spec_bytes)


def test_source_implements_both_protocols(source: MujocoOverlaySource) -> None:
    assert isinstance(source, ContactSource)
    assert isinstance(source, KinematicsSource)
    assert 40.0 < source.total_mass_kg < 150.0


def test_joint_frames_are_unit_axes_and_cover_hinges_only(
    source: MujocoOverlaySource,
) -> None:
    q = dict.fromkeys(source.coordinate_order, 0.0)
    frames = source.joint_frames(q)
    assert frames and set(frames) <= set(source.coordinate_order)
    assert not any(n.startswith("Translation") for n in frames)
    for f in frames.values():
        assert np.linalg.norm(f.axis_world) == pytest.approx(1.0, abs=1e-9)


def test_provider_over_real_spec_gives_weight_and_torques(
    spec_bytes: bytes, source: MujocoOverlaySource
) -> None:
    names = tuple(source.coordinate_order)
    nv, steps = len(names), 3
    q = np.zeros((steps + 1, nv))
    bundle = InputBundle(
        spec_bytes=spec_bytes,
        coordinate_order=names,
        dt_s=0.001,
        q0=q[0],
        v0=q[0],
        efforts=np.full((steps, nv), 3.0),
        reference_q=q,
        reference_v=q.copy(),
        reference_engine="mujoco",
    )
    frame = BundleOverlayProvider(bundle, source, source, engine="mujoco").frame_at(1)
    (weight,) = frame.by_kind(WrenchKind.GRAVITY)
    assert weight.force_n is not None
    assert weight.force_n[2] == pytest.approx(-source.total_mass_kg * 9.80665)
    assert len(frame.by_kind(WrenchKind.JOINT_ACTUATOR)) >= 5
