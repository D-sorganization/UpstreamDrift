"""Tests for VariantOverlayBox controls and toggles (#11310)."""

from unittest.mock import MagicMock
import pytest
from PyQt6.QtWidgets import QApplication

from src.tools.capture_rig.overlay_box import VariantOverlayBox
from tests.tools.capture_rig.test_pane_layout import _app

pytestmark = [pytest.mark.unit, pytest.mark.ui]


def test_variant_overlay_box_toggles() -> None:
    app = _app()
    box = VariantOverlayBox()

    assert not box.joint_torques.isChecked()
    assert box.forces.isChecked()
    assert box.torques.isChecked()
    assert not box.legend.isChecked()
    assert box.force_scale == 1.0

    signals = []
    box.changed.connect(lambda: signals.append(1))

    box.joint_torques.setChecked(True)
    assert len(signals) == 1
    assert box.joint_torques_enabled

    box.scale_slider.setValue(200)
    assert len(signals) == 2
    assert box.force_scale == 2.0

    box.forces.setChecked(False)
    assert len(signals) == 3
    assert not box.forces_enabled

    box.torques.setChecked(False)
    assert len(signals) == 4
    assert not box.torques_enabled

    box.legend.setChecked(True)
    assert len(signals) == 5
    assert box.legend_enabled

    box.close()
    app.processEvents()


def test_variant_overlay_box_load_preserves_fto_toggles() -> None:
    app = _app()
    box = VariantOverlayBox()

    # Create dummy media
    mock_variant = MagicMock()
    mock_variant.name = "v1"
    mock_variant.label = "Variant 1"
    mock_variant.has_reconstruction = True
    mock_variant.has_model_fit = False

    mock_media = MagicMock()
    mock_media.variants = [mock_variant]
    mock_media.root = None

    box.joint_torques.setChecked(True)
    box.load(mock_media)

    assert "v1" in box._checks
    assert box.joint_torques.isChecked()
    assert box.joint_torques_enabled

    # Loading None clears variants but preserves FTO toggles
    box.load(None)
    assert len(box._checks) == 0
    assert box.joint_torques.isChecked()

    box.close()
    app.processEvents()
