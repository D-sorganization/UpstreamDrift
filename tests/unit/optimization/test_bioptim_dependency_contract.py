"""Separate the qualified Bioptim runtime from the general CasADi backend."""

from pathlib import Path
import tomllib

from packaging.requirements import Requirement
import pytest

pytestmark = pytest.mark.unit


def _accepts(version: str, groups: tuple[str, ...]) -> bool:
    root = Path(__file__).resolve().parents[3]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    extras = project["project"]["optional-dependencies"]
    requirements = [Requirement(value) for group in groups for value in extras[group]]
    constraints = [item.specifier for item in requirements if item.name == "casadi"]
    assert constraints, "the tested stack must declare its CasADi dependency"
    return all(version in constraint for constraint in constraints)


def test_bioptim_admits_the_numerically_checked_runtime() -> None:
    assert _accepts("3.6.7", ("optimal-control", "bioptim"))


@pytest.mark.parametrize("version", ["3.7.2", "3.8.0"])
def test_bioptim_does_not_select_an_unqualified_runtime(version: str) -> None:
    assert not _accepts(version, ("optimal-control", "bioptim"))


def test_general_casadi_backend_retains_its_independent_version_range() -> None:
    assert _accepts("3.8.0", ("optimal-control",))
