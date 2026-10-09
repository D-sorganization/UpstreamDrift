"""Visualization tab GRF scale-mode and group controls (GCV-4, #11710)."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("cv2")
try:
    from PyQt6 import QtWidgets
except (ImportError, OSError) as exc:  # pragma: no cover - optional GUI dependency
    pytest.skip(f"PyQt6 not loadable: {exc}", allow_module_level=True)

from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.gui.tabs.visualization_tab import (  # noqa: E402
    VisualizationTab,
)
from src.shared.python.force_overlay.glyphs import ALL_GROUPS, DEFAULT_GROUPS  # noqa: E402

pytestmark = pytest.mark.unit


class _SimWidget:
    def __init__(self) -> None:
        self.options: dict[str, Any] | None = None

    def set_force_style_options(self, options: dict[str, Any] | None) -> None:
        self.options = options

    def __getattr__(self, name: str) -> Any:
        return lambda *args, **kwargs: None


@pytest.fixture
def tab() -> Any:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    sim = _SimWidget()
    widget = VisualizationTab(sim)  # type: ignore[arg-type]
    yield widget, sim
    widget.deleteLater()
    app.processEvents()


def test_group_checkboxes_cover_all_groups_with_defaults(tab: Any) -> None:
    widget, _ = tab
    assert set(widget.force_group_checkboxes) == set(ALL_GROUPS)
    checked = {g for g, b in widget.force_group_checkboxes.items() if b.isChecked()}
    assert checked == set(DEFAULT_GROUPS)


def test_body_weight_mode_pushes_reference_force(tab: Any) -> None:
    widget, sim = tab
    widget.force_scale_mode_combo.setCurrentIndex(
        widget.force_scale_mode_combo.findData("body_weight")
    )
    widget.body_mass_spin.setValue(80.0)
    assert sim.options is not None
    assert sim.options["scale_mode"] == "body_weight"
    assert sim.options["reference_force_n"] == pytest.approx(80.0 * 9.80665)
    assert sim.options["reference_length_m"] == pytest.approx(0.5)


def test_group_toggle_updates_groups_and_fixed_mode_has_no_reference(tab: Any) -> None:
    widget, sim = tab
    widget.force_group_checkboxes["contact_points"].setChecked(True)
    assert sim.options is not None
    assert "contact_points" in sim.options["groups"]
    assert "reference_force_n" not in sim.options
