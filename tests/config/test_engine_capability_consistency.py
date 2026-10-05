"""Consistency tests between engine get_capabilities() and engine capability matrix (Issue #11571).

Validates that for every engine whose wrapper module can be imported, every
capability explicitly declared by get_capabilities() matches the capability
support level specified in src/config/engine_capability_matrix.json.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import json
from pathlib import Path
import textwrap
from typing import Any

import pytest

from src.shared.python.engine_core.capabilities import EngineCapabilities

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = REPO_ROOT / "src" / "config" / "engine_capability_matrix.json"

ENGINE_SOURCES: dict[str, tuple[str, str]] = {
    "mujoco": (
        "src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine",
        "MuJoCoPhysicsEngine",
    ),
    "drake": (
        "src.engines.physics_engines.drake.python.drake_physics_engine",
        "DrakePhysicsEngine",
    ),
    "pinocchio": (
        "src.engines.physics_engines.pinocchio.python.pinocchio_physics_engine",
        "PinocchioPhysicsEngine",
    ),
    "opensim": (
        "src.engines.physics_engines.opensim.python.opensim_physics_engine",
        "OpenSimPhysicsEngine",
    ),
    "myosuite": (
        "src.engines.physics_engines.myosuite.python.myosuite_physics_engine",
        "MyoSuitePhysicsEngine",
    ),
    "jaxsim": (
        "src.engines.physics_engines.jaxsim.jaxsim_backend",
        "JaxSimBackend",
    ),
}


def _probe_engine_class(module_name: str, class_name: str) -> tuple[type | None, str]:
    """Attempt to import engine class, returning (class, '') or (None, error_msg)."""
    try:
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        return cls, ""
    except (ImportError, AttributeError, RuntimeError) as exc:
        return None, str(exc)


def _extract_declared_capabilities(cls: type) -> set[str]:
    """Extract capability arguments explicitly passed to EngineCapabilities in get_capabilities."""
    src = inspect.getsource(cls.get_capabilities)
    tree = ast.parse(textwrap.dedent(src))
    declared: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg and kw.arg not in ("engine_name", "extra"):
                    declared.add(kw.arg)
    return declared


@pytest.fixture(scope="module")
def capability_matrix() -> dict[str, Any]:
    """Load the committed engine capability matrix JSON."""
    assert MATRIX_PATH.is_file(), f"Missing {MATRIX_PATH}"
    return json.loads(MATRIX_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize("engine_name", list(ENGINE_SOURCES.keys()))
def test_engine_capabilities_match_matrix(
    engine_name: str,
    capability_matrix: dict[str, Any],
) -> None:
    """Declared capabilities in engine get_capabilities() must match matrix JSON."""
    module_name, class_name = ENGINE_SOURCES[engine_name]
    engine_cls, error_msg = _probe_engine_class(module_name, class_name)

    if engine_cls is None:
        pytest.skip(f"Engine {engine_name} could not be imported: {error_msg}")

    # Design by Contract: verify get_capabilities is present and callable
    assert hasattr(engine_cls, "get_capabilities"), (
        f"{class_name} does not implement get_capabilities"
    )

    declared_caps = _extract_declared_capabilities(engine_cls)
    assert declared_caps, (
        f"{class_name}.get_capabilities does not declare any capabilities"
    )

    engine_instance = engine_cls.__new__(engine_cls)
    caps: EngineCapabilities = engine_instance.get_capabilities()
    assert isinstance(caps, EngineCapabilities), (
        f"{class_name}.get_capabilities must return EngineCapabilities"
    )

    profiles = capability_matrix.get("profiles", {})
    assert engine_name in profiles, f"Engine {engine_name} not found in matrix profiles"
    matrix_caps = profiles[engine_name].get("capabilities", {})

    discrepancies: list[str] = []
    for cap_name in sorted(declared_caps):
        code_level = caps.level_for(cap_name).name.lower()
        matrix_level = matrix_caps.get(cap_name)
        if code_level != matrix_level:
            discrepancies.append(
                f"  - {cap_name}: code={code_level!r} != matrix={matrix_level!r}"
            )

    assert not discrepancies, (
        f"Capability discrepancies found for {engine_name}:\n"
        + "\n".join(discrepancies)
    )
