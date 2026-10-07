"""The Drake MeshCat sink must pass cone height first (NV-4, #11677)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

pytest.importorskip("pydrake")


class _RecordingMeshcat:
    def __init__(self) -> None:
        self.objects: dict[str, object] = {}

    def SetObject(self, path, shape, rgba) -> None:  # noqa: N802 - Drake API name
        self.objects[path] = shape


def test_cone_height_is_the_arrow_length_not_its_radius() -> None:
    from src.engines.physics_engines.drake.python.src.drake_meshcat_sink import (
        DrakeMeshcatSink,
    )

    meshcat = _RecordingMeshcat()
    DrakeMeshcatSink(meshcat).set_cylinder("a/head", 0.12, 0.0, 0.03, (1, 0, 0, 1))
    cone = meshcat.objects["a/head"]
    assert cone.height() == pytest.approx(0.12)
    assert cone.a() == pytest.approx(0.03) and cone.b() == pytest.approx(0.03)
