from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CANONICAL_FIT_DRIVERS = (
    REPO_ROOT
    / "src"
    / "engines"
    / "physics_engines"
    / "mujoco"
    / "python"
    / "motion_matching"
    / "fit_swing.py",
    REPO_ROOT
    / "src"
    / "engines"
    / "physics_engines"
    / "drake"
    / "python"
    / "motion_matching"
    / "fit_swing.py",
    REPO_ROOT
    / "src"
    / "engines"
    / "physics_engines"
    / "pinocchio"
    / "python"
    / "motion_matching"
    / "fit_swing.py",
    REPO_ROOT
    / "src"
    / "engines"
    / "physics_engines"
    / "opensim"
    / "python"
    / "motion_matching"
    / "fit_swing.py",
)
CANONICAL_ENGINE_TESTS = (
    REPO_ROOT / "tests" / "test_opensim_fit_swing.py",
    REPO_ROOT / "tests" / "heavy_integration" / "test_pinocchio_fit_swing.py",
)


def test_motion_matching_fit_result_exports_are_canonical() -> None:
    for path in CANONICAL_FIT_DRIVERS:
        source = path.read_text(encoding="utf-8")

        assert "CanonicalFitResult as FitResult" in source


def test_canonical_engine_tests_use_theta_optimal_field() -> None:
    for path in CANONICAL_ENGINE_TESTS:
        source = path.read_text(encoding="utf-8")

        assert re.search(r"\.theta(?!_)", source) is None
