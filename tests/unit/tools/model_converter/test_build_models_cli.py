"""Unit tests for build_models CLI orchestrator (Issue #9965)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.model_converter.build_models import build_models, main

CANONICAL_YAML = (
    REPO_ROOT
    / "src"
    / "engines"
    / "physics_engines"
    / "pinocchio"
    / "models"
    / "spec"
    / "golfer_canonical.yaml"
)


class TestBuildModelsCLI:
    """Test the single-command build_models workflow."""

    def test_build_models_custom_output_dir(self, tmp_path: Path) -> None:
        urdf_out = tmp_path / "test.urdf"
        mjcf_out = tmp_path / "test.xml"
        matlab_out = tmp_path / "InitParams.m"

        rc = build_models(
            canonical_yaml=CANONICAL_YAML,
            urdf_out=urdf_out,
            mjcf_out=mjcf_out,
            matlab_out=matlab_out,
            check=False,
            validate_only=False,
        )
        assert rc == 0
        assert urdf_out.exists()
        assert mjcf_out.exists()
        assert matlab_out.exists()

    def test_validate_only_does_not_write(self, tmp_path: Path) -> None:
        urdf_out = tmp_path / "unwritten.urdf"
        mjcf_out = tmp_path / "unwritten.xml"
        matlab_out = tmp_path / "unwritten.m"

        rc = build_models(
            canonical_yaml=CANONICAL_YAML,
            urdf_out=urdf_out,
            mjcf_out=mjcf_out,
            matlab_out=matlab_out,
            validate_only=True,
        )
        assert rc == 0
        assert not urdf_out.exists()
        assert not mjcf_out.exists()

    def test_check_mode_passes_on_matching_files(self, tmp_path: Path) -> None:
        urdf_out = tmp_path / "golfer.urdf"
        mjcf_out = tmp_path / "golfer.xml"
        matlab_out = tmp_path / "InitModelParameters.m"

        # Generate initial files
        rc_gen = build_models(
            canonical_yaml=CANONICAL_YAML,
            urdf_out=urdf_out,
            mjcf_out=mjcf_out,
            matlab_out=matlab_out,
        )
        assert rc_gen == 0

        # Check mode should return 0
        rc_check = build_models(
            canonical_yaml=CANONICAL_YAML,
            urdf_out=urdf_out,
            mjcf_out=mjcf_out,
            matlab_out=matlab_out,
            check=True,
        )
        assert rc_check == 0

    def test_check_mode_detects_drift(self, tmp_path: Path) -> None:
        urdf_out = tmp_path / "golfer.urdf"
        mjcf_out = tmp_path / "golfer.xml"
        matlab_out = tmp_path / "InitModelParameters.m"

        # Generate initial files
        build_models(
            canonical_yaml=CANONICAL_YAML,
            urdf_out=urdf_out,
            mjcf_out=mjcf_out,
            matlab_out=matlab_out,
        )

        # Mutate one file to introduce drift
        urdf_out.write_text("CORRUPTED DRIFT", encoding="utf-8")

        rc_check = build_models(
            canonical_yaml=CANONICAL_YAML,
            urdf_out=urdf_out,
            mjcf_out=mjcf_out,
            matlab_out=matlab_out,
            check=True,
        )
        assert rc_check == 1

    def test_main_cli_validate_only(self) -> None:
        rc = main(["--validate-only"])
        assert rc == 0
