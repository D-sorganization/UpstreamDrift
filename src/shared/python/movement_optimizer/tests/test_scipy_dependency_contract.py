from __future__ import annotations

from pathlib import Path

from scipy.interpolate import CubicSpline

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_scipy_dependency_has_no_legacy_1_16_ceiling() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    scipy_specs = [dep for dep in pyproject["project"]["dependencies"] if dep.startswith("scipy")]

    assert scipy_specs == ["scipy>=1.10"]


def test_cubic_spline_imports_with_supported_scipy() -> None:
    assert CubicSpline.__name__ == "CubicSpline"
