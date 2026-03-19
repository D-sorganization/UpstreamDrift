"""Tests for issue #1772: Silent exception handlers in MuJoCo/Drake GUI code.

Verifies that bare ``pass`` exception handlers have been replaced with
``logger.debug()`` calls so that debugging information is surfaced.
"""

from __future__ import annotations

import ast
from pathlib import Path

# Files that were identified in issue #1772 as containing silent pass handlers
_FIXED_FILES: list[str] = [
    "src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/sim_widget.py",
    "src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/sim_rendering_mixin.py",
    "src/engines/physics_engines/drake/python/src/drake_gui_viz.py",
    "src/engines/physics_engines/mujoco/docker/src/humanoid_golf/sim.py",
]

_REPO_ROOT = Path(__file__).parent.parent.parent


def _find_bare_pass_handlers(source: str) -> list[int]:
    """Return line numbers of except-clauses whose bodies are bare ``pass``."""
    tree = ast.parse(source)
    bare_lines: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        body = node.body
        if len(body) == 1 and isinstance(body[0], ast.Pass):
            bare_lines.append(node.lineno)
    return bare_lines


def test_no_bare_pass_in_fixed_files() -> None:
    """All except-handlers in the fixed GUI files must have a body."""
    violations: dict[str, list[int]] = {}
    for rel_path in _FIXED_FILES:
        filepath = _REPO_ROOT / rel_path
        if not filepath.exists():
            continue
        source = filepath.read_text(encoding="utf-8")
        lines = _find_bare_pass_handlers(source)
        if lines:
            violations[rel_path] = lines

    assert (
        not violations
    ), "Bare ``pass`` exception handlers still present (issue #1772):\n" + "\n".join(
        f"  {path}: lines {lns}" for path, lns in violations.items()
    )


def test_sim_widget_handler_uses_logger_debug() -> None:
    """sim_widget.py plane-rendering exception must log at DEBUG level."""
    filepath = _REPO_ROOT / _FIXED_FILES[0]
    if not filepath.exists():
        return
    source = filepath.read_text(encoding="utf-8")
    assert (
        "logger.debug" in source
    ), "sim_widget.py must use logger.debug in exception handlers (issue #1772)"


def test_sim_rendering_mixin_handler_uses_logger_debug() -> None:
    """sim_rendering_mixin.py must log rendering failures at DEBUG level."""
    filepath = _REPO_ROOT / _FIXED_FILES[1]
    if not filepath.exists():
        return
    source = filepath.read_text(encoding="utf-8")
    assert (
        source.count("logger.debug") >= 2
    ), "sim_rendering_mixin.py must have at least 2 logger.debug calls (issue #1772)"


def test_drake_gui_viz_handler_uses_logger() -> None:
    """drake_gui_viz.py must log actuator-index parse errors."""
    filepath = _REPO_ROOT / _FIXED_FILES[2]
    if not filepath.exists():
        return
    source = filepath.read_text(encoding="utf-8")
    assert (
        "LOGGER.debug" in source or "logger.debug" in source
    ), "drake_gui_viz.py must use a logger.debug call in the ValueError handler (issue #1772)"


def test_docker_sim_handlers_use_logger_debug() -> None:
    """mujoco/docker/sim.py exception handlers must use logger.debug."""
    filepath = _REPO_ROOT / _FIXED_FILES[3]
    if not filepath.exists():
        return
    source = filepath.read_text(encoding="utf-8")
    assert (
        source.count("logger.debug") >= 4
    ), "mujoco docker sim.py must have at least 4 logger.debug calls (issue #1772)"
