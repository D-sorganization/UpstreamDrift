"""MuJoCo native export backend (GCV-14, #11720): registration and scene.

The frame-count and 0 ms impact HUD contract for every engine, MuJoCo
included, lives in ``test_speed_variants_engines.py`` (it iterates ENGINES).
"""

from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np
import pytest

from src.shared.python.motion_matching.same_input import InputBundle
from src.tools.native_viewer_export.backends import mujoco_native
from src.tools.native_viewer_export.backends.mujoco_native import (
    MuJoCoRendererBackend,
)
from src.tools.native_viewer_export.backends.registry import (
    BACKEND_FACTORIES,
    make_backend,
)
from src.tools.native_viewer_export.core import (
    ENGINES,
    VIEWER_NAMES,
    ExportSettings,
    SwingInput,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

ROOT = Path(__file__).resolve().parents[4]
SPEC = ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"


def test_mujoco_is_a_registered_engine() -> None:
    assert "mujoco" in ENGINES
    assert "mujoco" in BACKEND_FACTORIES
    assert "mujoco" in VIEWER_NAMES
    backend = make_backend("mujoco")
    assert isinstance(backend, MuJoCoRendererBackend)
    assert backend.engine == "mujoco"


def test_unavailable_reason_without_mujoco(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mujoco_native, "find_spec", lambda name: None)
    assert MuJoCoRendererBackend().unavailable_reason() == "mujoco is not installed"


def test_available_when_mujoco_importable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mujoco_native, "find_spec", lambda name: object())
    assert MuJoCoRendererBackend().unavailable_reason() is None


def test_command_runs_the_worker_module() -> None:
    cmd = MuJoCoRendererBackend().command()
    assert cmd[1:] == ["-m", mujoco_native.WORKER_MODULE]


@pytest.fixture(scope="module")
def spec_bytes() -> bytes:
    return SPEC.read_bytes()


def test_scene_has_appearance_body_and_club(spec_bytes: bytes) -> None:
    from src.tools.native_viewer_export.backends.mujoco_worker import (
        build_scene_xml,
    )

    xml, meta = build_scene_xml(spec_bytes)
    layer = meta["visual_layer"]
    assert layer["meshes"] > 0 and layer["floor"] is True
    assert "visual_capsule" not in xml  # appearance replaces the stick figure
    assert layer["club_head"] == "driver"  # shaft, grip and head meshes attached
    for part in ("shaft", "grip", "head"):
        assert re.search(rf'<mesh name="vmesh_\d+_{part}"', xml), part


@pytest.mark.requires_gl
def test_renders_nonblank_frame_at_requested_size(spec_bytes: bytes) -> None:
    pytest.importorskip("mujoco")
    names = tuple(json.loads(spec_bytes)["coordinate_order"])
    q = np.zeros((3, len(names)))
    bundle = InputBundle(
        spec_bytes=spec_bytes,
        coordinate_order=names,
        dt_s=0.001,
        q0=q[0],
        v0=q[0],
        efforts=np.zeros((2, len(names))),
        reference_q=q,
        reference_v=q.copy(),
        reference_engine="mujoco",
    )
    swing = SwingInput(bundle, q, "t", "Driver", "mujoco")
    settings = ExportSettings(
        views=("face_on",), width=320, height=240, multiview=False
    )
    backend = make_backend("mujoco")
    try:
        frame = next(iter(backend.render(swing, settings, [0], None)))["face_on"]
    except RuntimeError as exc:
        pytest.skip(f"no usable GL context: {exc}")
    assert frame.shape == (240, 320, 3)
    assert len(np.unique(frame.reshape(-1, 3), axis=0)) > 20
