"""Tests for the cross-engine leaderboard runner script."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from scripts import run_cross_engine_leaderboard as runner
from src.shared.python.motion_matching import leaderboard, provider


@pytest.mark.unit
def test_skip_fits_generates_report_without_optional_loader_imports(
    tmp_path: Path,
) -> None:
    results_dir = tmp_path / "results"
    leaderboard_path = results_dir / "CROSS_ENGINE_LEADERBOARD.md"

    status = runner.main(
        [
            "--skip-fits",
            "--results-dir",
            str(results_dir),
            "--leaderboard-path",
            str(leaderboard_path),
        ]
    )

    assert status == 0
    assert leaderboard_path.exists()


@pytest.mark.unit
def test_fit_driver_modules_import() -> None:
    """Every module in the map imports or is listed in KNOWN_UNAVAILABLE with a reason."""
    assert hasattr(runner, "KNOWN_UNAVAILABLE")
    assert "simscape" in runner.KNOWN_UNAVAILABLE
    assert runner.KNOWN_UNAVAILABLE["simscape"] == "no python provider"

    for engine, (mod_path, attr) in runner._FIT_DRIVER_MODULES.items():
        if engine in runner.KNOWN_UNAVAILABLE:
            continue
        try:
            mod = importlib.import_module(mod_path)
        except ImportError as exc:
            pytest.fail(
                f"Module {mod_path} for engine {engine} failed to import: {exc}"
            )
        assert hasattr(mod, attr), f"Module {mod_path} missing attribute {attr}"


@pytest.mark.unit
def test_valid_engines_equals_registry() -> None:
    """_VALID_ENGINES in leaderboard matches _FIT_DRIVER_MODULES in runner."""
    assert set(leaderboard.valid_engines()) == set(runner._FIT_DRIVER_MODULES.keys())


@pytest.mark.unit
def test_simscape_row_is_explicit_unavailable(tmp_path: Path) -> None:
    """simscape produces an explicit row with solver='unavailable: no python provider'."""
    results_dir = tmp_path / "results"
    leaderboard_path = results_dir / "CROSS_ENGINE_LEADERBOARD.md"

    status = runner.main(
        [
            "--trial",
            "TW_ProV1",
            "--engine",
            "simscape",
            "--results-dir",
            str(results_dir),
            "--leaderboard-path",
            str(leaderboard_path),
        ]
    )
    assert status == 0
    simscape_json = results_dir / "TW_ProV1" / "simscape.json"
    assert simscape_json.exists()

    payload = json.loads(simscape_json.read_text(encoding="utf-8"))
    assert payload["engine"] == "simscape"
    assert "unavailable: no python provider" in payload["solver"]

    # Verify that leaderboard rendered simscape row
    assert leaderboard_path.exists()
    content = leaderboard_path.read_text(encoding="utf-8")
    assert "simscape" in content
    assert "unavailable: no python provider" in content
