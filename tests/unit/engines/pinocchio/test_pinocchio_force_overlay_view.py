"""Tests for PinocchioForceOverlayView and visualization mixin consolidation (ADR-0052, FTO-14, #11299)."""

from __future__ import annotations

import re
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.shared.python.body_part_viz import AxialLoadFrame
from src.shared.python.force_overlay import (
    ForceTorqueFrame,
    OverlayWrench,
    WrenchKind,
)
from src.shared.python.force_overlay.renderers.meshcat_glyphs import MeshcatSink

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


class FakeMeshcatSink:
    """Recording sink implementing MeshcatSink protocol."""

    def __init__(self) -> None:
        self.cylinder_calls: list[
            tuple[str, float, float, float, tuple[float, float, float, float]]
        ] = []
        self.transform_calls: list[tuple[str, np.ndarray]] = []
        self.delete_calls: list[str] = []

    def set_cylinder(
        self,
        path: str,
        length_m: float,
        radius_top_m: float,
        radius_bottom_m: float,
        rgba: tuple[float, float, float, float],
    ) -> None:
        self.cylinder_calls.append(
            (path, length_m, radius_top_m, radius_bottom_m, rgba)
        )

    def set_transform(self, path: str, matrix4x4: np.ndarray) -> None:
        self.transform_calls.append((path, matrix4x4))

    def delete(self, path: str) -> None:
        self.delete_calls.append(path)


class RecordingColorSession:
    """Mock recording force color session."""

    def __init__(self) -> None:
        self.frames: list[AxialLoadFrame | None] = []
        self.bound_adapter: Any = None
        self.bound_model: Any = None

    def set_frame(self, frame: AxialLoadFrame | None) -> None:
        self.frames.append(frame)

    def bind(self, adapter: Any, model: Any) -> None:
        self.bound_adapter = adapter
        self.bound_model = model


class FakeProvider:
    """Mock provider returning deterministic ForceTorqueFrame and ZTCF frame."""

    def __init__(
        self, frame: ForceTorqueFrame, ztcf_frame: ForceTorqueFrame | None = None
    ) -> None:
        self.frame = frame
        self.ztcf_frame = ztcf_frame

    def get_force_torque_frame(self) -> ForceTorqueFrame | None:
        return self.frame

    def get_ztcf_frame(self) -> ForceTorqueFrame | None:
        return self.ztcf_frame


def _make_sample_frame() -> ForceTorqueFrame:
    reaction = OverlayWrench(
        kind=WrenchKind.JOINT_REACTION,
        label="reaction:joint_1",
        body="synthetic_rod",
        point_m=(0.0, 0.0, 1.0),
        force_n=(100.0, 0.0, 0.0),
        torque_nm=(0.0, 0.0, 10.0),
        source="test",
    )
    contact = OverlayWrench(
        kind=WrenchKind.CONTACT,
        label="contact:ground",
        body="synthetic_base",
        point_m=(0.0, 0.0, 0.0),
        force_n=(0.0, 0.0, 50.0),
        torque_nm=None,
        source="test",
    )
    axial = AxialLoadFrame(
        time_s=0.2,
        values_n={"synthetic_rod": 100.0, "synthetic_base": -50.0},
        source="test",
    )
    return ForceTorqueFrame(
        time_s=0.2,
        engine="pinocchio",
        wrenches=(reaction, contact),
        axial_loads=axial,
    )


def _make_ztcf_frame() -> ForceTorqueFrame:
    cf_wrench = OverlayWrench(
        kind=WrenchKind.JOINT_REACTION,
        label="reaction:joint_1",
        body="synthetic_rod",
        point_m=(0.0, 0.0, 1.0),
        force_n=(0.0, 0.0, 20.0),
        torque_nm=(0.0, 0.0, 0.0),
        source="pinocchio:ztcf",
    )
    return ForceTorqueFrame(
        time_s=0.2,
        engine="pinocchio",
        wrenches=(cf_wrench,),
    )


def test_update_with_forces_on_yields_shaft_and_head_calls() -> None:
    from src.engines.physics_engines.pinocchio.python.pinocchio_golf.force_overlay_view import (
        PinocchioForceOverlayView,
    )

    sink = FakeMeshcatSink()
    session = RecordingColorSession()
    provider = FakeProvider(_make_sample_frame())

    view = PinocchioForceOverlayView(provider, sink, session, root="/force_overlay")
    view.update({"forces": True, "torques": True})

    # Must yield shaft and head cylinder calls for reaction and contact glyphs
    paths = [call[0] for call in sink.cylinder_calls]
    assert any("reaction:joint_1/shaft" in p for p in paths), (
        f"reaction shaft missing from {paths}"
    )
    assert any("reaction:joint_1/head" in p for p in paths), (
        f"reaction head missing from {paths}"
    )
    assert any("contact:ground/shaft" in p for p in paths), (
        f"contact shaft missing from {paths}"
    )
    assert any("contact:ground/head" in p for p in paths), (
        f"contact head missing from {paths}"
    )


def test_shading_toggle_calls_set_frame_with_provider_axial_loads() -> None:
    from src.engines.physics_engines.pinocchio.python.pinocchio_golf.force_overlay_view import (
        PinocchioForceOverlayView,
    )

    sink = FakeMeshcatSink()
    session = RecordingColorSession()
    frame = _make_sample_frame()
    provider = FakeProvider(frame)

    view = PinocchioForceOverlayView(provider, sink, session)

    # Shading enabled
    view.update({"shading": True})
    assert len(session.frames) == 1
    assert session.frames[-1] is frame.axial_loads

    # Shading disabled
    view.update({"shading": False})
    assert len(session.frames) == 2
    assert session.frames[-1] is None


def test_meshcat_paths_for_links_maps_synthetic_visual_model() -> None:
    from src.engines.physics_engines.pinocchio.python.pinocchio_golf.force_overlay_view import (
        meshcat_paths_for_links,
    )

    class FakeGeom:
        def __init__(self, name: str, parent_link: str | None = None) -> None:
            self.name = name
            self.parent_link = parent_link

    class FakeVisualModel:
        def __init__(self, geoms: list[FakeGeom]) -> None:
            self.geometryObjects = geoms

    vm = FakeVisualModel(
        [
            FakeGeom("synthetic_base_0", parent_link="synthetic_base"),
            FakeGeom("synthetic_rod_0", parent_link="synthetic_rod"),
        ]
    )

    paths = meshcat_paths_for_links(vm, root="pinocchio")
    assert "synthetic_base" in paths
    assert "synthetic_rod" in paths
    assert paths["synthetic_base"] == ["pinocchio/visuals/synthetic_base_0"]
    assert paths["synthetic_rod"] == ["pinocchio/visuals/synthetic_rod_0"]

    # Also test with custom root containing leading/trailing slashes
    paths_custom = meshcat_paths_for_links(vm, root="/custom_view/")
    assert paths_custom["synthetic_rod"] == ["custom_view/visuals/synthetic_rod_0"]


def test_both_mixins_delegate_to_helper_and_draw_arrow_is_deleted() -> None:
    import pathlib

    # Acceptance requirement: rg -n "def _draw_arrow" src/engines/physics_engines/pinocchio returns nothing
    repo_pinocchio_dir = (
        pathlib.Path(__file__).parents[4] / "src/engines/physics_engines/pinocchio"
    )
    for py_file in repo_pinocchio_dir.rglob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        assert "def _draw_arrow" not in content, f"def _draw_arrow found in {py_file}"

    # Verify PinocchioVisualizationMixin delegates
    from src.engines.physics_engines.pinocchio.python.pinocchio_golf.pinocchio_visualization_mixin import (
        PinocchioVisualizationMixin,
    )

    class FakePinGUI(PinocchioVisualizationMixin):
        def __init__(self) -> None:
            self.force_overlay_view = MagicMock()
            self.chk_forces = MagicMock(isChecked=MagicMock(return_value=True))
            self.chk_torques = MagicMock(isChecked=MagicMock(return_value=False))
            self.chk_cf = MagicMock(isChecked=MagicMock(return_value=False))
            self.spin_force_scale = MagicMock(value=MagicMock(return_value=0.001))
            self.spin_torque_scale = MagicMock(value=MagicMock(return_value=0.005))
            self.segment_force_colors = MagicMock()
            self.segment_force_colors._scale.enabled = True

    gui1 = FakePinGUI()
    gui1._draw_vectors()
    gui1.force_overlay_view.update.assert_called_once()

    # Verify VisualizationMixin delegates
    from src.engines.physics_engines.pinocchio.python.pinocchio_golf.ui.visualization_mixin import (
        VisualizationMixin,
    )

    class FakeUIGUI(VisualizationMixin):
        def __init__(self) -> None:
            self.force_overlay_view = MagicMock()
            self.chk_forces = MagicMock(isChecked=MagicMock(return_value=True))
            self.chk_torques = MagicMock(isChecked=MagicMock(return_value=False))
            self.chk_cf = MagicMock(isChecked=MagicMock(return_value=False))
            self.spn_f_scale = MagicMock(value=MagicMock(return_value=0.001))
            self.spn_t_scale = MagicMock(value=MagicMock(return_value=0.005))
            self.segment_force_colors = MagicMock()
            self.segment_force_colors._scale.enabled = False

    gui2 = FakeUIGUI()
    gui2._draw_vectors()
    gui2.force_overlay_view.update.assert_called_once()


def test_ztcf_toggle_produces_cf_labelled_glyphs_only_when_enabled() -> None:
    from src.engines.physics_engines.pinocchio.python.pinocchio_golf.force_overlay_view import (
        PinocchioForceOverlayView,
    )

    sink = FakeMeshcatSink()
    session = RecordingColorSession()
    frame = _make_sample_frame()
    ztcf = _make_ztcf_frame()
    provider = FakeProvider(frame, ztcf_frame=ztcf)

    view = PinocchioForceOverlayView(provider, sink, session, root="/force_overlay")

    # When ztcf is False (default)
    view.update({"forces": True, "ztcf": False})
    paths_no_ztcf = [call[0] for call in sink.cylinder_calls]
    assert not any("cf:" in p for p in paths_no_ztcf), (
        f"Unexpected cf: glyph in {paths_no_ztcf}"
    )

    # When ztcf is True
    sink.cylinder_calls.clear()
    view.update({"forces": True, "ztcf": True})
    paths_with_ztcf = [call[0] for call in sink.cylinder_calls]
    assert any("cf:" in p for p in paths_with_ztcf), (
        f"Expected cf: glyph missing from {paths_with_ztcf}"
    )
