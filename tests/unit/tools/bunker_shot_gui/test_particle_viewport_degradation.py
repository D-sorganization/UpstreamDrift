"""The particle viewport degrades cleanly without VTK (issue #9545 / #9693).

No ``pytest.importorskip("pyvista")`` here, on purpose: this file proves
the fallback on a runner without the ``viz3d`` extra, by faking absence
rather than requiring it. The artifact is the SYNTHETIC TEST FIXTURE.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("h5py", reason="the trajectory artifact is HDF5")

from bunkershot3d.fields.trajectory_reader import ParticleTrajectoryReader  # noqa: E402
from src.tools.bunker_shot_gui import particle_render_vtk, render3d_vtk  # noqa: E402
from src.tools.bunker_shot_gui.particle_playback import ParticlePlayback  # noqa: E402
from tests.unit.tools.bunker_shot_gui._synthetic_particle_artifact import (  # noqa: E402
    write_synthetic_artifact,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


@pytest.fixture()
def playback(tmp_path: Path):  # type: ignore[no-untyped-def]
    path = write_synthetic_artifact(tmp_path / "synthetic.h5")
    with ParticleTrajectoryReader(path) as reader:
        yield ParticlePlayback(reader)


def test_the_module_imports_without_pyvista() -> None:
    for name in particle_render_vtk.__all__:
        assert hasattr(particle_render_vtk, name), name


def test_no_pyvista_yields_a_fallback_that_states_the_reason(
    playback, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(render3d_vtk, "find_spec", lambda name: None)
    result = particle_render_vtk.open_particle_viewport(playback)
    assert isinstance(result, particle_render_vtk.ParticleViewportUnavailable)
    assert "viz3d" in result.reason
    assert "3-D particle viewport unavailable" in result.describe()
    assert "playback" in result.describe()


def test_a_render_window_that_cannot_open_yields_a_fallback(
    playback, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    def _no_gl(**_: object) -> None:
        raise RuntimeError("no OpenGL context")

    monkeypatch.setattr(particle_render_vtk, "pyvista_available", lambda: True)
    monkeypatch.setattr(
        particle_render_vtk,
        "require_pyvista",
        lambda: SimpleNamespace(Plotter=_no_gl),
    )
    result = particle_render_vtk.open_particle_viewport(playback)
    assert isinstance(result, particle_render_vtk.ParticleViewportUnavailable)
    assert "no OpenGL context" in result.reason
