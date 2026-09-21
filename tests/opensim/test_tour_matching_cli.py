"""Tests for OpenSim tour matching CLI, checkpoint determinism, and visualization (OS-6).

Exercises pure-Python CLI argument parsing, 7-subcommand routing,
deterministic configuration hashing, checkpoint manifests, corruption rejection,
and headless plot generation.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import numpy as np
import pytest

from src.engines.physics_engines.opensim.python.tour_matching.cli import (
    CheckpointManifest,
    RunConfig,
    build_parser,
    compute_run_hash,
)
from src.engines.physics_engines.opensim.python.tour_matching.visualization import (
    plot_effort_and_rates,
    plot_marker_error_timecourse,
)


@pytest.mark.unit
def test_cli_parser_subcommands() -> None:
    """Verify all 7 proposed operations exist in the CLI parser."""
    parser = build_parser()
    subparsers_actions = [
        action
        for action in parser._actions
        if action.__class__.__name__ == "_SubParsersAction"
    ]
    assert len(subparsers_actions) == 1
    subparser_choices = set(subparsers_actions[0].choices.keys())
    expected = {
        "prepare",
        "qualify",
        "calibrate",
        "fit",
        "replay",
        "compare",
        "resume",
    }
    assert expected.issubset(subparser_choices)


@pytest.mark.unit
def test_run_config_hash_determinism() -> None:
    """Identical configurations produce identical run hashes."""
    cfg1 = RunConfig(
        model_path="models/golf_humanoid.osim",
        trc_path="data/tour.trc",
        duration_s=0.10,
        polynomial_degree=6,
        seed=42,
    )
    cfg2 = RunConfig(
        model_path="models/golf_humanoid.osim",
        trc_path="data/tour.trc",
        duration_s=0.10,
        polynomial_degree=6,
        seed=42,
    )
    hash1 = compute_run_hash(cfg1)
    hash2 = compute_run_hash(cfg2)
    assert hash1 == hash2
    assert len(hash1) == 16

    # Changed parameter produces a different hash
    cfg3 = RunConfig(
        model_path="models/golf_humanoid.osim",
        trc_path="data/tour.trc",
        duration_s=0.10,
        polynomial_degree=6,
        seed=43,
    )
    assert compute_run_hash(cfg3) != hash1


@pytest.mark.unit
def test_checkpoint_manifest_validation() -> None:
    """CheckpointManifest rejects corrupted JSON or hash mismatch."""
    manifest = CheckpointManifest(
        run_id="run_test_01",
        stage="fit",
        model_sha256="abc1234",
        status="completed",
        artifacts={"coeffs": "polynomial_coefficients.json"},
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "manifest.json"
        manifest.save(path)
        loaded = CheckpointManifest.load(path)
        assert loaded.run_id == "run_test_01"
        assert loaded.stage == "fit"

        # Corrupt manifest
        with open(path, "w", encoding="utf-8") as f:
            f.write("{corrupted json...")

        with pytest.raises(ValueError, match="Failed to load manifest"):
            CheckpointManifest.load(path)


@pytest.mark.unit
def test_visualization_plot_generation() -> None:
    """Visualization generates valid PNG plot files without GUI display."""
    times = np.linspace(0.0, 0.10, 21)
    errors = {
        "Pelvis": np.linspace(0.01, 0.02, 21),
        "Club": np.linspace(0.02, 0.05, 21),
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        plot_path = Path(tmpdir) / "error_plot.png"
        plot_marker_error_timecourse(times, errors, plot_path)
        assert plot_path.is_file()
        assert plot_path.stat().st_size > 1000

        effort_path = Path(tmpdir) / "effort_plot.png"
        efforts = {"tau_1": np.sin(times)}
        rates = {"tau_1": np.cos(times)}
        plot_effort_and_rates(times, efforts, rates, effort_path)
        assert effort_path.is_file()
        assert effort_path.stat().st_size > 1000
