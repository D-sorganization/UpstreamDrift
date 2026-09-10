"""Standalone graphics preserve source observations alongside fitted geometry."""

from pathlib import Path
from dataclasses import replace

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


def test_preview_retains_missing_club_samples(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    draft, profile = pendulum_input()
    result = fit_reference(draft, profile, "double_pendulum")
    asset = result.asset.changed(
        joint_names=(*result.asset.joint_names, "club"),
        source_names=(*result.asset.source_names, "club"),
        edges=(*result.asset.edges, (1, 2)),
        club_edges=((1, 2),),
        points_m=tuple((*row, None) for row in result.asset.points_m),
    )
    bundle = save_reference_fit(replace(result, asset=asset), tmp_path / "bundle")
    assert render_fit_preview(bundle, tmp_path / "preview.png").is_file()
