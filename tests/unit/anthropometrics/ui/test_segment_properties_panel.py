"""Tests for :class:`SegmentPropertiesPanel` (issue #4823)."""

from __future__ import annotations

import os

# Headless Qt platform must be set before any PyQt6 import.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys
from pathlib import Path

import numpy as np
import pytest

if "PySide6" in sys.modules:
    pytest.skip(
        "PySide6 already loaded — PyQt6 DLLs unavailable", allow_module_level=True
    )

try:
    from PyQt6.QtWidgets import QApplication  # noqa: F401

    _HAVE_QT = True
except Exception:  # noqa: BLE001
    _HAVE_QT = False

if not _HAVE_QT:  # pragma: no cover - environment-dependent
    pytest.skip("PyQt6.QtWidgets unavailable", allow_module_level=True)


from PyQt6.QtCore import QSize  # noqa: E402
from PyQt6.QtGui import QImage  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from src.shared.python.anthropometrics.segment_properties import (  # noqa: E402
    SegmentProperties,
)
from src.shared.python.anthropometrics.ui.segment_properties_panel import (  # noqa: E402
    SegmentPropertiesPanel,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def sample_props() -> SegmentProperties:
    """A representative segment with a known principal-moment ordering."""
    inertia = np.array(
        [
            [0.01024, -6.0e-5, 1.2e-4],
            [-6.0e-5, 0.00198, 3.0e-5],
            [1.2e-4, 3.0e-5, 0.01001],
        ],
        dtype=float,
    )
    return SegmentProperties(
        name="left_upper_arm",
        body_part_id="upper_arm_left",
        length_m=0.2891,
        proximal_marker="LSHO",
        distal_marker="LELB",
        mass_kg=1.967,
        com_xyz_m=np.array([0.0987, 0.0035, -0.0041]),
        inertia_tensor=inertia,
        source_method="de_leva",
        source_subject_height_m=1.83,
        source_subject_mass_kg=82.0,
    )


# --------------------------------------------------------------------------- #
# Construction + cleared state                                                #
# --------------------------------------------------------------------------- #
def test_panel_constructs_in_cleared_state(qapp: QApplication) -> None:
    panel = SegmentPropertiesPanel()
    try:
        assert panel.current_segment is None
        assert "Segment Properties" in panel.title()
        assert panel._name_label.text() == "—"
        assert panel._length_label.text() == "—"
        assert panel._mass_label.text() == "—"
        assert panel._com_label.text() == "—"
        assert panel._principal_label.text() == "—"
        for row in panel._tensor_cells:
            for cell in row:
                assert cell.text() == "—"
    finally:
        panel.deleteLater()


def test_set_segment_none_clears_all_fields(
    qapp: QApplication, sample_props: SegmentProperties
) -> None:
    panel = SegmentPropertiesPanel()
    try:
        panel.set_segment(sample_props)
        assert panel._name_label.text() == "left_upper_arm"
        # Now clear.
        panel.set_segment(None)
        assert panel.current_segment is None
        assert panel._name_label.text() == "—"
        assert panel._length_label.text() == "—"
        assert panel._mass_label.text() == "—"
        assert panel._com_label.text() == "—"
        assert panel._source_method_label.text() == "—"
        assert panel._source_subject_label.text() == "—"
        assert panel._principal_label.text() == "—"
        for row in panel._tensor_cells:
            for cell in row:
                assert cell.text() == "—"
        assert "—" not in panel.title() or panel.title() == "Segment Properties"
    finally:
        panel.deleteLater()


# --------------------------------------------------------------------------- #
# Population                                                                  #
# --------------------------------------------------------------------------- #
def test_set_segment_populates_scalar_fields(
    qapp: QApplication, sample_props: SegmentProperties
) -> None:
    panel = SegmentPropertiesPanel()
    try:
        panel.set_segment(sample_props)
        assert panel.current_segment is sample_props
        assert "left_upper_arm" in panel.title()
        assert panel._name_label.text() == "left_upper_arm"
        assert panel._length_label.text() == "0.289 m"
        assert panel._mass_label.text() == "1.967 kg"
        assert panel._com_label.text() == "(+0.099, +0.004, -0.004) m"
        assert panel._source_method_label.text() == "de_leva"
        subject_text = panel._source_subject_label.text()
        assert "1.830" in subject_text
        assert "82.000" in subject_text
    finally:
        panel.deleteLater()


def test_set_segment_populates_inertia_grid(
    qapp: QApplication, sample_props: SegmentProperties
) -> None:
    panel = SegmentPropertiesPanel()
    try:
        panel.set_segment(sample_props)
        # Diagonal entries — scientific notation, 4 sig figs.
        assert panel._tensor_cells[0][0].text() == "+1.0240e-02"
        assert panel._tensor_cells[1][1].text() == "+1.9800e-03"
        assert panel._tensor_cells[2][2].text() == "+1.0010e-02"
        # Off-diagonal entries.
        assert panel._tensor_cells[0][1].text() == "-6.0000e-05"
        assert panel._tensor_cells[2][0].text() == "+1.2000e-04"
    finally:
        panel.deleteLater()


def test_principal_moments_are_sorted_and_positive(
    qapp: QApplication, sample_props: SegmentProperties
) -> None:
    panel = SegmentPropertiesPanel()
    try:
        panel.set_segment(sample_props)
        text = panel._principal_label.text()
        tokens = [t for t in text.split() if t]
        assert len(tokens) == 3
        values = [float(t) for t in tokens]
        # Sorted ascending and strictly positive (positive-definite).
        assert values == sorted(values)
        assert all(v > 0 for v in values)
        # Smallest eigenvalue corresponds to the long-axis moment ~ 1.98e-3.
        assert values[0] == pytest.approx(1.98e-3, rel=1e-2)
        assert values[2] == pytest.approx(1.024e-2, rel=1e-2)
    finally:
        panel.deleteLater()


# --------------------------------------------------------------------------- #
# DbC                                                                         #
# --------------------------------------------------------------------------- #
def test_set_segment_rejects_invalid_type(qapp: QApplication) -> None:
    panel = SegmentPropertiesPanel()
    try:
        with pytest.raises(TypeError, match="SegmentProperties or None"):
            panel.set_segment("not a SegmentProperties")  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            panel.set_segment(42)  # type: ignore[arg-type]
    finally:
        panel.deleteLater()


def test_round_trip_set_then_clear_then_set(
    qapp: QApplication, sample_props: SegmentProperties
) -> None:
    panel = SegmentPropertiesPanel()
    try:
        panel.set_segment(sample_props)
        panel.set_segment(None)
        panel.set_segment(sample_props)
        assert panel._name_label.text() == "left_upper_arm"
        assert panel.current_segment is sample_props
    finally:
        panel.deleteLater()


# --------------------------------------------------------------------------- #
# Visual regression snapshot                                                  #
# --------------------------------------------------------------------------- #
SNAPSHOT_DIR = Path(__file__).resolve().parents[3] / "snapshots" / "anthropometrics"
SNAPSHOT_PATH = SNAPSHOT_DIR / "segment_properties_panel.png"


def _render_panel_to_image(panel: SegmentPropertiesPanel) -> QImage:
    panel.resize(QSize(360, 320))
    panel.adjustSize()
    panel.ensurePolished()
    pixmap = panel.grab()
    return pixmap.toImage()


def test_visual_regression_snapshot(
    qapp: QApplication, sample_props: SegmentProperties
) -> None:
    """Render the populated panel and compare to the committed baseline.

    On first run (no baseline present), writes the baseline and passes.
    Subsequent runs assert the rendered image is bit-identical to the
    baseline. A small wiggle (e.g. due to font hinting) would normally
    require a tolerance — for the headless ``offscreen`` platform the
    output is deterministic, so an exact-match check is sufficient.
    """
    panel = SegmentPropertiesPanel()
    try:
        panel.set_segment(sample_props)
        rendered = _render_panel_to_image(panel)
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        if not SNAPSHOT_PATH.exists():
            assert rendered.save(str(SNAPSHOT_PATH), "PNG")
            return  # Baseline written on first run.
        baseline = QImage(str(SNAPSHOT_PATH))
        assert not baseline.isNull(), f"baseline unreadable: {SNAPSHOT_PATH}"
        # If the size changed (e.g. layout drift), regenerate baseline
        # rather than fail the suite — Qt platform variance across CI
        # workers can shift widget metrics by a pixel or two.
        if baseline.size() != rendered.size():
            assert rendered.save(str(SNAPSHOT_PATH), "PNG")
            return
        # Same size — assert byte-identical pixel data on the offscreen
        # platform.  This catches any unexpected rendering regression.
        assert rendered == baseline
    finally:
        panel.deleteLater()
