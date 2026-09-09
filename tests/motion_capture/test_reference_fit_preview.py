"""Standalone graphics preserve source observations alongside fitted geometry."""

from pathlib import Path

import pytest

from src.motion_capture.reference.fit_preview import render_fit_preview
from src.motion_capture.reference.fit_pipeline import fit_reference, save_reference_fit
from tests.motion_capture.test_reference_fit_pipeline import pendulum_input

pytestmark = pytest.mark.unit


def test_preview_renders_saved_geometry_without_source_file(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    draft, profile = pendulum_input()
    result = fit_reference(draft, profile, "double_pendulum")
    bundle = save_reference_fit(result, tmp_path / "bundle")
    output = render_fit_preview(bundle, tmp_path / "preview.png")
    assert output.read_bytes().startswith(b"\x89PNG")
    assert output.stat().st_size > 10000
