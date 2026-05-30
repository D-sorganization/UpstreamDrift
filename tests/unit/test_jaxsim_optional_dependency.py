"""Contracts for the optional JaxSim dependency gate."""

from __future__ import annotations

from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib


def _optional_dependencies() -> dict[str, list[str]]:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    return pyproject["project"]["optional-dependencies"]


def test_jaxsim_extra_is_pinned_and_isolated_from_engine_rollup() -> None:
    """JaxSim is opt-in while the native-engine coexistence gate is open."""
    optional = _optional_dependencies()

    assert optional["jaxsim"] == ["jaxsim==0.9.0"]
    assert optional["all-engines"] == ["upstream-drift[drake,pinocchio]"]
    assert "jaxsim" not in optional["all-engines"][0]


@pytest.mark.requires_jaxsim
def test_jaxsim_optional_stack_imports_and_steps_sdf_fixture() -> None:
    """The optional JaxSim stack can load and step a minimal SDF model."""
    jax = pytest.importorskip("jax")
    jaxlib = pytest.importorskip("jaxlib")
    jaxsim = pytest.importorskip("jaxsim")
    js = pytest.importorskip("jaxsim.api")

    fixture = Path("tests/fixtures/jaxsim/single_link.sdf")
    model = js.model.JaxSimModel.build_from_model_description(
        fixture,
        is_urdf=False,
        time_step=0.001,
    )
    data = js.data.JaxSimModelData.build(model)

    for _ in range(3):
        data = js.model.step(model, data)

    assert jax.__version__ >= "0.4.34"
    assert jaxlib.__version__ >= "0.4.34"
    assert jaxsim.__version__ == "0.9.0"
    assert model.dofs() == 0
    assert data is not None
