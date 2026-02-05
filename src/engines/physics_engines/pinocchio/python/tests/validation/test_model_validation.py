"""Validation tests for model correctness."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "python"))

from dtack.utils.urdf_exporter import URDFExporter  # isort: skip # noqa: E402


class TestModelValidation:
    """Test model correctness and physics accuracy."""

    def test_canonical_yaml_exists(self) -> None:
        """Test that canonical YAML specification exists."""
        yaml_path = REPO_ROOT / "models/spec/golfer_canonical.yaml"
        assert (
            yaml_path.exists()
        ), f"Canonical YAML specification not found at {yaml_path}"

    def test_yaml_structure(self) -> None:
        """Test that YAML has required structure."""

        yaml_path = REPO_ROOT / "models/spec/golfer_canonical.yaml"
        if yaml_path.exists():
            with yaml_path.open() as f:
                spec = yaml.safe_load(f)
            assert "root" in spec, "Missing root segment"
            assert "segments" in spec, "Missing segments"
            assert "constraints" in spec, "Missing constraints"

    def test_urdf_export_contains_links(self, tmp_path: Path) -> None:
        """Ensure URDF exporter produces all major links and joints."""

        yaml_path = REPO_ROOT / "models/spec/golfer_canonical.yaml"
        exporter = URDFExporter(yaml_path)

        output_path = tmp_path / "golfer.urdf"
        exporter.export(output_path)

        contents = output_path.read_text(encoding="utf-8")
        expected_links = [
            "left_thigh",
            "right_thigh",
            "thorax3",
            "club_head",
            "fingers_left",
        ]

        for link in expected_links:
            assert f'link name="{link}"' in contents, f"Missing link {link} in URDF"

        assert contents.count("<joint") >= 10, "Expected multiple joints in URDF export"
