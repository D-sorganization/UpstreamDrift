"""Architecture guard: motion-matching quaternion/geodesic/rotmat helpers must live in _geodesic.py (UD-24, #13868).

All motion-matching quaternion<->rotmat, geodesic, and multiply operations must route
through ``src.shared.python.motion_matching._geodesic`` so non-unit simulation
quaternions produce consistent behavior across all engines.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

SCAN_ROOTS = (
    "src/shared/python/motion_matching",
    "src/engines/physics_engines/drake/python/motion_matching",
    "src/engines/physics_engines/mujoco/python/motion_matching",
    "src/engines/physics_engines/opensim/python/motion_matching",
    "src/engines/physics_engines/pinocchio/python/motion_matching",
)

EXCLUDED_FILES = {
    "src/shared/python/motion_matching/_geodesic.py",
}

HELPER_PATTERN = re.compile(r"^_(?:.*quat.*|.*geodesic.*|.*rotmat.*)$", re.IGNORECASE)

pytestmark = [pytest.mark.architecture, pytest.mark.gate]


def collect_private_quaternion_helpers() -> list[str]:
    """Scan configured motion-matching roots for private quaternion/rotmat/geodesic helpers."""
    helpers: list[str] = []
    for root_rel in SCAN_ROOTS:
        full_dir = ROOT / root_rel
        if not full_dir.exists():
            continue
        for py_path in sorted(full_dir.rglob("*.py")):
            rel_str = py_path.relative_to(ROOT).as_posix()
            if rel_str in EXCLUDED_FILES:
                continue
            try:
                tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=rel_str)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if HELPER_PATTERN.match(node.name):
                        helpers.append(f"{rel_str}:{node.lineno}: {node.name}")
    return helpers


def test_single_quaternion_helper() -> None:
    """Architecture guard counts private quaternion helpers = 0 outside _geodesic.py (UD-24)."""
    helpers = collect_private_quaternion_helpers()
    assert helpers == [], (
        f"Found {len(helpers)} private quaternion/geodesic/rotmat helpers outside _geodesic.py:\n  "
        + "\n  ".join(helpers)
    )
