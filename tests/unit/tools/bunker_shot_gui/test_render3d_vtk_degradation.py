"""render3d_vtk.py stays import-safe and degrades cleanly (issue #8706).

PyVista is an optional extra (``viz3d``); a stock CI runner does not have it
installed. This file has no ``pytest.importorskip`` -- it is the one place
proving the *fallback* path is covered on a runner where pyvista is absent,
which is exactly the case ``pytest.importorskip`` would otherwise skip past
untested. It works whether or not pyvista actually happens to be installed
on the machine running the suite, because the degradation path is exercised
by faking absence rather than by requiring it.
"""

from __future__ import annotations

import pytest

from src.tools.bunker_shot_gui import render3d_vtk

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


class TestTheModuleImportsWithoutPyVista:
    """Importing this module must never require the optional extra."""

    def test_the_module_is_already_imported_by_this_test(self) -> None:
        """If this file loaded at all, the import above already succeeded."""
        assert render3d_vtk.RENDERER == "pyvista"

    def test_the_public_surface_is_importable_without_pyvista_installed(self) -> None:
        """Every name in ``__all__`` resolves; none of it needs the import."""
        for name in render3d_vtk.__all__:
            assert hasattr(render3d_vtk, name), name


class TestDegradationWhenPyVistaIsAbsent:
    """Mirrors the ADR-0027 viewport layer's own degradation reasons."""

    def test_pyvista_available_reports_false_when_the_module_cannot_be_found(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(render3d_vtk, "find_spec", lambda name: None)
        assert render3d_vtk.pyvista_available() is False

    def test_pyvista_available_reports_true_when_the_module_can_be_found(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(render3d_vtk, "find_spec", lambda name: object())
        assert render3d_vtk.pyvista_available() is True

    def test_a_lookup_error_from_find_spec_reads_as_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A malformed/partial install can make ``find_spec`` raise, not just
        return ``None``; either way this must read as "not available" rather
        than propagate an exception from a plain availability probe."""

        def _raise(name: str) -> None:
            raise ValueError("corrupt finder cache")

        monkeypatch.setattr(render3d_vtk, "find_spec", _raise)
        assert render3d_vtk.pyvista_available() is False

    def test_require_pyvista_raises_a_clear_install_hint_when_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(render3d_vtk, "find_spec", lambda name: None)
        with pytest.raises(render3d_vtk.PyVistaNotAvailableError, match="viz3d"):
            render3d_vtk.require_pyvista()

    def test_the_install_hint_names_the_pip_extra(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(render3d_vtk, "find_spec", lambda name: None)
        try:
            render3d_vtk.require_pyvista()
        except render3d_vtk.PyVistaNotAvailableError as error:
            assert "pip install" in str(error)
            assert "viz3d" in str(error)
        else:
            pytest.fail("require_pyvista should have raised")

    def test_the_not_available_error_is_a_runtime_error(self) -> None:
        """Callers that catch broad runtime failures still catch this one."""
        assert issubclass(render3d_vtk.PyVistaNotAvailableError, RuntimeError)


class TestPureHelpersNeedNoPyVista:
    """The small numpy-only helpers behind the renderer, tested in isolation."""

    def test_hex_to_rgb01_converts_known_colours(self) -> None:
        assert render3d_vtk._hex_to_rgb01("#ffffff") == (1.0, 1.0, 1.0)
        assert render3d_vtk._hex_to_rgb01("#000000") == (0.0, 0.0, 0.0)
        assert render3d_vtk._hex_to_rgb01("#ff0000") == (1.0, 0.0, 0.0)

    def test_hex_to_rgb01_refuses_a_malformed_colour(self) -> None:
        with pytest.raises(ValueError, match="rrggbb"):
            render3d_vtk._hex_to_rgb01("#fff")

    def test_pv_faces_pads_triangles_with_their_vertex_count(self) -> None:
        import numpy as np

        faces = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
        padded = render3d_vtk._pv_faces(faces)
        assert list(padded) == [3, 0, 1, 2, 3, 1, 2, 3]

    def test_polyline_connectivity_spans_every_point_in_order(self) -> None:
        connectivity = render3d_vtk._polyline_connectivity(4)
        assert list(connectivity) == [4, 0, 1, 2, 3]

    def test_posed_mm_matches_the_scene_pose_transform(self) -> None:
        import numpy as np

        body = np.array([[1.0, 0.0, 0.0]])
        rotation = np.eye(3)
        position = np.array([2.0, 0.0, 0.0])
        posed = render3d_vtk._posed_mm(body, rotation, position)
        assert np.allclose(posed, [[3000.0, 0.0, 0.0]])
