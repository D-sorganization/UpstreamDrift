"""Architecture guard: motion-matching quaternion/geodesic/rotmat helpers must live in _geodesic.py (UD-24, #13868).

All motion-matching quaternion<->rotmat, geodesic, and multiply operations must route
through ``src.shared.python.motion_matching._geodesic`` so non-unit simulation
quaternions produce consistent behavior across all engines.

This guard enforces:
1. No unapproved private quaternion/geodesic/rotmat helpers exist outside ``_geodesic.py``.
2. The count of private helpers ratchets strictly downward (target count 0).
3. Sibling functions found during repo survey are tracked and documented.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
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

# Baseline allowable private helpers on main as of UD-24 (#13868).
# Target count is 0. Each engine migration PR shrinks this allowlist.
BASELINE_ALLOWLIST: frozenset[str] = frozenset(
    {
        "src/engines/physics_engines/drake/python/motion_matching/simulate.py: _quat_to_rotmat_series",
        "src/engines/physics_engines/drake/python/motion_matching/viz/render_error_timecourse.py: _to_unit_quat",
        "src/engines/physics_engines/mujoco/python/motion_matching/synthesize.py: _normalise_quaternions",
        "src/engines/physics_engines/mujoco/python/motion_matching/viz/render_swing.py: _quat_geodesic_deg",
        "src/engines/physics_engines/opensim/python/motion_matching/synthesize.py: _normalise_quat_rows",
        "src/engines/physics_engines/pinocchio/python/motion_matching/club_target_adapter.py: _dircos_to_rotmat",
        "src/engines/physics_engines/pinocchio/python/motion_matching/club_target_adapter.py: _rotmat_stack_to_quat",
        "src/engines/physics_engines/pinocchio/python/motion_matching/club_target_adapter.py: _rotmat_to_quat_local",
        "src/engines/physics_engines/pinocchio/python/motion_matching/fit_swing.py: _interp_quat_to",
        "src/engines/physics_engines/pinocchio/python/motion_matching/simulate.py: _quat_to_rotmat_series",
        "src/engines/physics_engines/pinocchio/python/motion_matching/viz/error_timecourse.py: _quat_geodesic_deg",
        "src/shared/python/motion_matching/api_contracts.py: _validate_quaternion",
        "src/shared/python/motion_matching/club_only/observation.py: _validate_quats",
        "src/shared/python/motion_matching/diagnostics/clubhead_trace.py: _quat_to_shaft_dir",
        "src/shared/python/motion_matching/loaders/_machinelearning_compat.py: _quat_from_velocity_acceleration",
        "src/shared/python/motion_matching/loaders/c3d.py: _shaft_quaternions",
        "src/shared/python/motion_matching/loaders/excel.py: _frame_to_quat",
        "src/shared/python/motion_matching/surrogate/model.py: _canonicalize_quaternion_sign",
        "src/shared/python/motion_matching/surrogate/perstep/optimize.py: _desired_quaternions",
        "src/shared/python/motion_matching/synthesize_target_from_coefficients.py: _normalise_quat_rows",
    }
)

# Sibling functions identified across the repo that perform orientation error/geodesic calculations
SIBLING_SWEEP_TRACKED: frozenset[str] = frozenset(
    {
        "src/engines/physics_engines/drake/python/motion_matching/viz/render_fit_quality_card.py: _orientation_error_deg",
        "src/engines/physics_engines/opensim/python/motion_matching/viz/_adapters.py: quat_geodesic_deg",
    }
)

pytestmark = pytest.mark.gate


@dataclass(frozen=True)
class HelperLocation:
    path: str
    lineno: int
    name: str

    @property
    def key(self) -> str:
        return f"{self.path}: {self.name}"


def collect_private_quaternion_helpers() -> list[HelperLocation]:
    """Scan configured motion-matching roots for private quaternion/rotmat/geodesic helpers."""
    helpers: list[HelperLocation] = []
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
                        helpers.append(
                            HelperLocation(
                                path=rel_str, lineno=node.lineno, name=node.name
                            )
                        )
    return helpers


def test_single_quaternion_helper() -> None:
    """Architecture guard counts private quaternion helpers outside _geodesic.py (UD-24).

    Enforces that:
    1. No new private quaternion helpers may be introduced outside the allowlist.
    2. Total count never exceeds the current baseline limit.
    3. The allowlist ratchets down to 0 as engines migrate to _geodesic.py.
    """
    helpers = collect_private_quaternion_helpers()
    helper_keys = {h.key for h in helpers}

    unapproved = helper_keys - BASELINE_ALLOWLIST
    assert not unapproved, (
        f"Found {len(unapproved)} unapproved private quaternion/geodesic helper(s) outside _geodesic.py:\n  "
        + "\n  ".join(sorted(unapproved))
        + "\nRoute all motion-matching quaternion operations through src.shared.python.motion_matching._geodesic."
    )

    # Invariant: guard count never increases without explicit allow-list
    assert len(helper_keys) <= len(BASELINE_ALLOWLIST), (
        f"Quaternion helper count ratcheted up: {len(helper_keys)} > {len(BASELINE_ALLOWLIST)}"
    )


def test_baseline_allowlist_has_no_stale_entries() -> None:
    """Ensure baseline entries are removed when helpers are migrated (prevent ghost entries)."""
    helpers = collect_private_quaternion_helpers()
    helper_keys = {h.key for h in helpers}

    stale = BASELINE_ALLOWLIST - helper_keys
    assert not stale, (
        f"Found {len(stale)} stale allowlist entry/entries that no longer exist in code:\n  "
        + "\n  ".join(sorted(stale))
        + "\nTighten the ratchet by removing these from BASELINE_ALLOWLIST!"
    )


def test_sibling_sweep_coverage() -> None:
    """Verify that sibling orientation/geodesic helpers identified in the repo survey exist and are tracked."""
    for entry in SIBLING_SWEEP_TRACKED:
        path_str, func_name = entry.split(": ")
        full_path = ROOT / path_str
        assert full_path.exists(), f"Tracked sibling file does not exist: {path_str}"
        tree = ast.parse(full_path.read_text(encoding="utf-8"), filename=path_str)
        found_names = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        assert func_name in found_names, (
            f"Sibling function {func_name} not found in {path_str}; remove or update sibling record"
        )
