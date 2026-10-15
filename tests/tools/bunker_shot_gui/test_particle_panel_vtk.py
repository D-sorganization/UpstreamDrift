"""Camera presets and clipping in the particle panel (issue #9545, slice 9).

Gated on the optional ``viz3d`` extra; ``test_particle_panel`` proves the
disabled-with-a-reason path these controls take without it. Every artifact
is the SYNTHETIC TEST FIXTURE: these check that a control changes the
picture, not physics.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PyQt6", reason="the workbench shell needs a Qt binding")
pytest.importorskip("pytestqt", reason="the panel tests drive Qt through qtbot")
pytest.importorskip("h5py", reason="the trajectory artifact is HDF5")
pytest.importorskip("pyvista", reason="the VTK/PyVista viz3d extra is optional")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from src.tools.bunker_shot_gui.particle_panel import ParticlePlaybackPanel  # noqa: E402
from src.tools.bunker_shot_gui.shot3d import CameraPreset  # noqa: E402
from tests.unit.tools.bunker_shot_gui._synthetic_particle_artifact import (  # noqa: E402
    write_synthetic_artifact,
)

pytestmark = [pytest.mark.unit, pytest.mark.requires_pyvista]


@pytest.fixture()
def loaded(qtbot, tmp_path: Path):  # type: ignore[no-untyped-def]
    panel = ParticlePlaybackPanel()
    qtbot.addWidget(panel)
    assert panel.load_trajectory(write_synthetic_artifact(tmp_path / "s.h5"))
    if not panel.viewport_available:
        pytest.skip(f"no offscreen render window here: {panel.reason_text}")
    yield panel
    panel.unload()


def _pixels(panel) -> np.ndarray:  # type: ignore[no-untyped-def]
    image = panel.rendered_image()
    assert image is not None
    return image


def test_with_vtk_every_control_is_enabled(loaded) -> None:  # type: ignore[no-untyped-def]
    assert all(reason is None for reason in loaded.disabled_reasons().values())
    assert loaded.reason_text == "All particle controls are active."


def test_a_camera_preset_changes_the_picture(loaded) -> None:  # type: ignore[no-untyped-def]
    before = _pixels(loaded)
    loaded.set_camera(CameraPreset.FACE_ON)
    assert loaded.camera is CameraPreset.FACE_ON
    assert not np.array_equal(before, _pixels(loaded))


def test_the_clipping_plane_hides_part_of_the_picture(loaded) -> None:  # type: ignore[no-untyped-def]
    loaded.set_camera(CameraPreset.FACE_ON)
    before = _pixels(loaded)
    loaded.set_clipping(True)
    assert loaded.clipping
    clipped = _pixels(loaded)
    assert not np.array_equal(before, clipped)
    loaded.set_clipping(False)
    np.testing.assert_array_equal(before, _pixels(loaded))


def test_stepping_changes_the_rendered_frame(loaded) -> None:  # type: ignore[no-untyped-def]
    before = _pixels(loaded)
    loaded.step_forward()
    loaded.step_forward()
    assert not np.array_equal(before, _pixels(loaded))
