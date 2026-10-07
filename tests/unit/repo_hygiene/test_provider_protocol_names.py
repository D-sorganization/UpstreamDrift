"""Two unrelated protocols once shared the name ``DynamicsProvider`` (#11553).

* ``DimeDynamicsProvider`` -- the stateful DIME provider (snapshot/step/...).
* ``EquationsOfMotionProvider`` -- pure ``mass_matrix`` / ``bias_forces``.

The old name stays importable for one release as a deprecated alias that warns
and resolves to the new class. No ``src/`` module may use the old name.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

from src.shared.python.estimation import dime_contracts
from src.shared.python.estimation.dime_contracts import DimeDynamicsProvider
from src.shared.python.estimation.dime_providers import (
    AnalyticPendulumProvider,
    DeterministicFakeProvider,
    UnderactuatedAnalyticProvider,
)
from src.shared.python.estimation.synthetic_fixtures import (
    make_fixed_base_pendulum_fixture,
)
from src.shared.python.simulation_backends import (
    GolfModelParams,
    has_mujoco,
    make_backend,
)
from src.shared.python.simulation_backends.protocol import (
    EquationsOfMotionProvider,
)

OLD_NAME = "DynamicsProvider"
REPO_ROOT = Path(__file__).resolve().parents[3]
# First-party code that must not import the deprecated name.
SCANNED_ROOTS = ("src", "scripts", "tests")

# (module path, new class) pairs whose old name must warn and alias the new class.
DEPRECATED_ALIASES = [
    ("src.shared.python.estimation.dime_contracts", DimeDynamicsProvider),
    ("src.shared.python.estimation", DimeDynamicsProvider),
    ("src.shared.python.simulation_backends.protocol", EquationsOfMotionProvider),
    ("src.shared.python.simulation_backends", EquationsOfMotionProvider),
]


@pytest.mark.unit
def test_new_names_exist_and_are_distinct_types() -> None:
    assert DimeDynamicsProvider is not EquationsOfMotionProvider
    assert DimeDynamicsProvider.__name__ == "DimeDynamicsProvider"
    assert EquationsOfMotionProvider.__name__ == "EquationsOfMotionProvider"
    assert "DimeDynamicsProvider" in dime_contracts.__all__


@pytest.mark.unit
def test_dime_providers_satisfy_dime_protocol_only() -> None:
    fixture = make_fixed_base_pendulum_fixture(n_frames=5)
    analytic = AnalyticPendulumProvider(fixture=fixture)
    fake = DeterministicFakeProvider(n_q=1, n_v=1, model_hash=analytic.model_hash)
    assert isinstance(analytic, DimeDynamicsProvider)
    assert isinstance(fake, DimeDynamicsProvider)
    assert isinstance(UnderactuatedAnalyticProvider(), DimeDynamicsProvider)
    # The DIME providers expose no mass_matrix/bias_forces, so the protocols differ.
    assert not isinstance(analytic, EquationsOfMotionProvider)


@pytest.mark.unit
def test_ode_backend_satisfies_eom_protocol_only() -> None:
    backend = make_backend("ode", GolfModelParams.default())
    assert isinstance(backend, EquationsOfMotionProvider)
    assert not isinstance(backend, DimeDynamicsProvider)


@pytest.mark.unit
@pytest.mark.requires_mujoco
@pytest.mark.skipif(not has_mujoco(), reason="mujoco not installed")
def test_mujoco_backend_satisfies_eom_protocol() -> None:
    backend = make_backend("mujoco", GolfModelParams.default())
    assert isinstance(backend, EquationsOfMotionProvider)


@pytest.mark.unit
@pytest.mark.parametrize(("module_path", "new_cls"), DEPRECATED_ALIASES)
def test_old_name_warns_and_returns_new_class(module_path: str, new_cls: type) -> None:
    module = importlib.import_module(module_path)
    with pytest.warns(DeprecationWarning, match=OLD_NAME):
        old = getattr(module, OLD_NAME)
    assert old is new_cls


@pytest.mark.unit
@pytest.mark.parametrize(("module_path", "new_cls"), DEPRECATED_ALIASES)
def test_old_name_is_still_exported_in_all(module_path: str, new_cls: type) -> None:
    if not hasattr(importlib.import_module(module_path), "__all__"):
        pytest.skip("module defines no __all__")
    module = importlib.import_module(module_path)
    assert new_cls.__name__ in module.__all__
    assert OLD_NAME in module.__all__


@pytest.mark.unit
def test_unknown_attribute_still_raises() -> None:
    with pytest.raises(AttributeError):
        _ = dime_contracts.NoSuchProvider  # type: ignore[attr-defined]


def _uses_old_name(path: Path) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            lines += [node.lineno for a in node.names if a.name == OLD_NAME]
        elif (isinstance(node, ast.Name) and node.id == OLD_NAME) or (
            isinstance(node, ast.Attribute) and node.attr == OLD_NAME
        ):
            lines.append(node.lineno)
    return lines


@pytest.mark.unit
def test_no_first_party_module_imports_deprecated_names() -> None:
    offenders = {
        str(p.relative_to(REPO_ROOT)): lines
        for root in SCANNED_ROOTS
        for p in (REPO_ROOT / root).rglob("*.py")
        if (lines := _uses_old_name(p))
    }
    assert not offenders, f"deprecated {OLD_NAME} still used: {offenders}"
