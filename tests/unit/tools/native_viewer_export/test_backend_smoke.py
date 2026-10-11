"""Real-viewer smoke renders (NV-4/5/6): skipped when a backend's runtime is missing.

Each engine renders one tiny face-on frame of the full-body model with and
without force/torque overlays; the overlay must change the pixels, proving the
glyphs reach the engine's native viewer (3D in MeshCat/MuJoCo, 2D in OpenSim).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching.full_body_spec import load_full_body_spec
from src.shared.python.motion_matching.same_input import InputBundle
from src.tools.native_viewer_export.backends.registry import make_backend
from src.tools.native_viewer_export.core import (
    ENGINES,
    ExportSettings,
    SwingInput,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.slow,
    pytest.mark.integration,
    pytest.mark.timeout(900),
]

ROOT = Path(__file__).resolve().parents[4]
FULL = ROOT / "docs/development/full_body_models/full_body_spec_v1.json"
UPPER = (
    ROOT
    / "docs/development/simscape_tour_matching/native_evidence/native_geometry_spec_9967.json"
)


@pytest.fixture(scope="module")
def swing() -> SwingInput:
    pytest.importorskip("mujoco")
    spec = load_full_body_spec(FULL, json.loads(UPPER.read_text(encoding="utf-8")))
    names = tuple(spec["coordinate_order"])
    q = np.zeros((3, len(names)))
    bundle = InputBundle(
        spec_bytes=json.dumps(spec).encode("utf-8"),
        coordinate_order=names,
        dt_s=0.001,
        q0=q[0],
        v0=q[0],
        efforts=np.zeros((2, len(names))),
        reference_q=q,
        reference_v=q.copy(),
        reference_engine="mujoco",
    )
    return SwingInput(bundle, q, "smoke", "Driver", "mujoco")


@pytest.mark.parametrize("engine", ENGINES)
def test_overlay_changes_native_pixels(engine: str, swing: SwingInput) -> None:
    from src.tools.native_viewer_export.overlay import build_overlay_feed

    backend = make_backend(engine)
    reason = backend.unavailable_reason()
    if reason is not None:
        pytest.skip(reason)
    feed, lookat = build_overlay_feed(swing, engine)
    settings = ExportSettings(
        views=("face_on",),
        width=320,
        height=272,
        lookat_m=lookat,
        multiview=False,
        overlays=True,
    )
    with_glyphs = next(iter(backend.render(swing, settings, [0], feed)))["face_on"]
    plain = next(iter(backend.render(swing, settings, [0], None)))["face_on"]
    assert with_glyphs.shape == plain.shape == (272, 320, 3)
    assert np.abs(with_glyphs.astype(int) - plain.astype(int)).sum() > 0
