"""Tests for launch_simulation_viewer CLI script (MV-05 #10481)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from scripts.launch_simulation_viewer import build_parser, main

pytestmark = pytest.mark.unit


def test_build_parser_options():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--candidate",
            "dummy.npz",
            "--viewer",
            "meshcat",
            "--model-bundle",
            "bundle.zip",
            "--urdf",
            "model.urdf",
            "--view-mode",
            "native",
            "--speed",
            "0.5",
            "--output-html",
            "out.html",
        ]
    )
    assert args.candidate == Path("dummy.npz")
    assert args.viewer == "meshcat"
    assert args.model_bundle == Path("bundle.zip")
    assert args.urdf == Path("model.urdf")
    assert args.view_mode == "native"
    assert args.speed == 0.5
    assert args.output_html == Path("out.html")


def test_main_check_only_available(capsys):
    with patch(
        "scripts.launch_simulation_viewer.is_backend_available",
        return_value=True,
    ):
        code = main(["--candidate", "dummy.npz", "--viewer", "meshcat", "--check-only"])
        assert code == 0
        captured = capsys.readouterr()
        assert "Viewer 'meshcat' available: True" in captured.out


def test_main_check_only_unavailable(capsys):
    with patch(
        "scripts.launch_simulation_viewer.is_backend_available",
        return_value=False,
    ):
        code = main(["--candidate", "dummy.npz", "--viewer", "opensim", "--check-only"])
        assert code == 1
        captured = capsys.readouterr()
        assert "Viewer 'opensim' available: False" in captured.out


def test_main_missing_candidate(capsys, tmp_path):
    missing = tmp_path / "nonexistent.npz"
    code = main(["--candidate", str(missing), "--viewer", "pyvista"])
    assert code == 1
    captured = capsys.readouterr()
    assert "not found" in captured.err
