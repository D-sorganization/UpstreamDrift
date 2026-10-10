"""Drake MeshCat shapes must land on the glyph segment (OSV-3b, #11729).

The shared renderer sends transforms for a centred shape along local +y (the
three.js cylinder convention). Drake's ``Cylinder`` runs along local +z and
``MeshcatCone`` has its apex at the origin pointing along +z, so without a
correction every shaft lies across its segment instead of along it.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.unit


def _real_pydrake() -> bool:
    """True only for a genuine pydrake install, not a ``sys.modules`` mock."""
    import importlib

    try:
        module = importlib.import_module("pydrake")
    except ImportError:
        return False
    return hasattr(module, "__path__") and type(module).__module__ != "unittest.mock"


if not _real_pydrake():
    pytest.skip("pydrake is not installed (or is mocked)", allow_module_level=True)


class _RecordingMeshcat:
    def __init__(self) -> None:
        self.objects: dict[str, object] = {}
        self.poses: dict[str, np.ndarray] = {}

    def SetObject(self, path, shape, rgba) -> None:  # noqa: N802 - Drake API name
        self.objects[path] = shape

    def SetTransform(self, path, pose) -> None:  # noqa: N802 - Drake API name
        self.poses[path] = pose.GetAsMatrix4()

    def Delete(self, path) -> None:  # noqa: N802 - Drake API name
        self.objects.pop(path, None)


def _render(tail, base, tip) -> _RecordingMeshcat:
    from src.engines.physics_engines.drake.python.src.drake_meshcat_sink import (
        DrakeMeshcatSink,
    )
    from src.shared.python.force_overlay.contracts import WrenchKind
    from src.shared.python.force_overlay.glyphs import ArrowGlyph, GlyphSet, LegendSpec
    from src.shared.python.force_overlay.renderers.meshcat_glyphs import (
        MeshcatGlyphRenderer,
    )

    arrow = ArrowGlyph(
        label="a",
        kind=WrenchKind.EXTERNAL,
        tail_m=tail,
        tip_m=tip,
        head_base_m=base,
        shaft_radius_m=0.01,
        head_radius_m=0.03,
        rgba=(1.0, 0.0, 0.0, 1.0),
        magnitude=1.0,
        units="m",
        clamped=False,
    )
    meshcat = _RecordingMeshcat()
    MeshcatGlyphRenderer(DrakeMeshcatSink(meshcat), root="/o").update(
        GlyphSet(time_s=0.0, arrows=(arrow,), torque_arcs=(), legend=LegendSpec())
    )
    return meshcat


def _world(pose: np.ndarray, local) -> np.ndarray:
    return (pose @ np.append(np.asarray(local, dtype=float), 1.0))[:3]


@pytest.mark.parametrize(
    "tail, base, tip",
    [
        ((0.0, 0.0, 1.6), (0.0, 0.0, 0.3), (0.0, 0.0, 0.2)),  # straight down
        ((0.1, 0.2, 1.5), (0.6, -0.1, 0.4), (0.65, -0.12, 0.33)),  # oblique
    ],
)
def test_shaft_runs_from_tail_to_head_base(tail, base, tip) -> None:
    meshcat = _render(tail, base, tip)
    shape = meshcat.objects["o/a/shaft"]
    pose = meshcat.poses["o/a/shaft"]
    half = shape.length() / 2.0
    ends = {tuple(np.round(_world(pose, (0, 0, s * half)), 9)) for s in (-1, 1)}
    assert ends == {tuple(np.round(tail, 9)), tuple(np.round(base, 9))}


@pytest.mark.parametrize(
    "tail, base, tip",
    [
        ((0.0, 0.0, 1.6), (0.0, 0.0, 0.3), (0.0, 0.0, 0.2)),
        ((0.1, 0.2, 1.5), (0.6, -0.1, 0.4), (0.65, -0.12, 0.33)),
    ],
)
def test_cone_apex_is_the_tip_and_base_sits_on_the_shaft(tail, base, tip) -> None:
    meshcat = _render(tail, base, tip)
    cone = meshcat.objects["o/a/head"]
    pose = meshcat.poses["o/a/head"]
    np.testing.assert_allclose(_world(pose, (0, 0, 0)), tip, atol=1e-9)
    np.testing.assert_allclose(_world(pose, (0, 0, cone.height())), base, atol=1e-9)
