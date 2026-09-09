"""Smoke tests for the BunkerShot3D Phase 1 MVP notebook runner (#8842).

The script used to import a nonexistent top-level ``bunkershot3d`` package
and point at a nonexistent repo-root ``configs/`` tree, so it could not run
at all. These tests pin the fixed contract: the imports resolve, the
packaged ``canonical.yaml`` is found, phases 1-2 produce their artifacts,
and the optional Chrono phase fails with the typed backend error rather
than an import error.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from bunkershot3d.exceptions import BackendNotImplementedError

_NOTEBOOK = (
    Path(__file__).resolve().parents[2] / "notebooks" / "bunkershot3d" / "phase1_mvp.py"
)


def _load_phase1_module():
    spec = importlib.util.spec_from_file_location("phase1_mvp_notebook", _NOTEBOOK)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
class TestPhase1MvpNotebook:
    def test_module_imports_resolve(self) -> None:
        """The script imports its first-party modules through canonical ``bunkershot3d``."""
        module = _load_phase1_module()
        assert callable(module.run_phase1)

    def test_phases_1_and_2_generate_artifacts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Trajectory CSV and clubhead STL are produced; Chrono is optional."""
        module = _load_phase1_module()

        class _FakeChronoUnavailable:
            def __init__(self, config_path: Path) -> None:
                assert Path(config_path).is_file(), (
                    "the packaged canonical.yaml must resolve"
                )

            def setup(self) -> None:
                raise module.BackendNotImplementedError(
                    "Backend 'chrono' requires pychrono"
                )

        monkeypatch.setattr(module, "ChronoDriver", _FakeChronoUnavailable)

        with pytest.raises(module.BackendNotImplementedError):
            module.run_phase1(artifact_dir=tmp_path)

        csv_path = tmp_path / "reference_swing.csv"
        stl_path = tmp_path / "wedge.stl"
        assert csv_path.is_file()
        assert stl_path.is_file()

    def test_default_config_path_exists(self) -> None:
        """The packaged default ``canonical.yaml`` resolves on disk."""
        module = _load_phase1_module()
        default_config = (
            module.REPO_ROOT
            / "src"
            / "bunkershot3d"
            / "calibration"
            / "configs"
            / "canonical.yaml"
        )
        assert default_config.is_file()
