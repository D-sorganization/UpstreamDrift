"""NSX-24 (#12567) acceptance: no SLSQP remains on convex allocation paths.

Convex allocation (weighted least squares over linear rows, boxes and Coulomb
cones) is solved only by the conic core (``loop_nullspace.solvers`` through
``loop_nullspace.qp_core``).  This test is the grep-style guard:

1. The allocator modules contain no scipy ``minimize`` and no ``"SLSQP"``.
2. Every other ``"SLSQP"`` literal under ``src/`` is listed in
   ``ALLOWED_SLSQP`` with the reason it is not a convex allocation path, so a
   new use cannot appear silently.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"
SHARED = SRC / "shared" / "python"

ALLOCATION_PATHS = (
    SHARED / "motion_matching" / "contact_force_allocator.py",
    SHARED / "motion_matching" / "multi_engine_torque_allocator.py",
    *sorted((SHARED / "motion_matching" / "force_nullspace").glob("*.py")),
    *sorted((SHARED / "loop_nullspace").glob("*.py")),
)

_NONCONVEX = "non-convex (nonlinear simulation/objective); not an allocation QP"
_FOLLOW_UP = (
    "convex but a different subsystem from the NSX-24 allocators; "
    "not migrated here, tracked as a follow-up"
)

ALLOWED_SLSQP: dict[str, str] = {
    "src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/motion_matching/"
    "option4_python_bridge/fit_swing_python.py": _NONCONVEX,
    "src/engines/physics_engines/drake/python/motion_matching/fit_swing.py": _NONCONVEX,
    "src/engines/physics_engines/drake/python/motion_optimization.py": _NONCONVEX,
    "src/engines/physics_engines/mujoco/python/motion_matching/fit_swing.py": _NONCONVEX,
    "src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/"
    "motion_optimization.py": _NONCONVEX,
    "src/engines/physics_engines/mujoco/python/native_manifold_box_fddp.py": _NONCONVEX,
    "src/engines/physics_engines/opensim/python/motion_matching/fit_swing.py": _NONCONVEX,
    "src/shared/python/motion_matching/bounded_candidate_search.py": _NONCONVEX,
    "src/shared/python/motion_matching/equality_least_squares.py": (
        "non-convex residual least squares on simulated physical outputs"
    ),
    "src/shared/python/optimization/_swing_models.py": _NONCONVEX,
    "src/shared/python/optimization/swing_optimizer.py": _NONCONVEX,
    "src/shared/python/physics/swing_optimizer.py": _NONCONVEX,
    "src/shared/python/physics/rigid_contact.py": (
        "impulse-level contact resolution, not torque/contact-force allocation; "
        + _FOLLOW_UP
    ),
    "src/shared/python/estimation/mosaic/contact_wrenches.py": (
        "MOSAIC inner wrench solve owned by the MOSAIC epic; " + _FOLLOW_UP
    ),
    "src/robotics/contact/grasp_analysis.py": (
        "robotics grasp force optimiser (box-relaxed cone); " + _FOLLOW_UP
    ),
    "src/robotics/control/whole_body/qp_solver.py": (
        "generic robotics QP backend selected by name; " + _FOLLOW_UP
    ),
}


def _tree(path: pathlib.Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _slsqp_literals(tree: ast.AST) -> int:
    return sum(
        isinstance(n, ast.Constant) and n.value == "SLSQP" for n in ast.walk(tree)
    )


def _scipy_minimize_uses(tree: ast.AST) -> list[str]:
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "scipy.optimize"
        ):
            hits += [a.name for a in node.names if a.name == "minimize"]
        elif isinstance(node, ast.Attribute) and node.attr == "minimize":
            hits.append("minimize")
        elif isinstance(node, ast.Name) and node.id in {"minimize", "_minimize"}:
            hits.append(node.id)
    return hits


def test_allocation_paths_exist() -> None:
    names = {p.name for p in ALLOCATION_PATHS}
    assert {
        "contact_force_allocator.py",
        "multi_engine_torque_allocator.py",
        "redistribution.py",
        "qp_core.py",
        "solvers.py",
    } <= names


@pytest.mark.parametrize("path", ALLOCATION_PATHS, ids=lambda p: p.name)
def test_allocator_module_has_no_slsqp_or_scipy_minimize(
    path: pathlib.Path,
) -> None:
    tree = _tree(path)
    assert _slsqp_literals(tree) == 0, f"{path.name} still names SLSQP"
    assert _scipy_minimize_uses(tree) == [], f"{path.name} still uses minimize"


def test_every_other_slsqp_use_is_allow_listed_with_a_reason() -> None:
    found: set[str] = set()
    for path in SRC.rglob("*.py"):
        rel = path.relative_to(REPO_ROOT).as_posix()
        if "/tests/" in rel or path.name.startswith("test_"):
            continue
        try:
            if "SLSQP" not in path.read_text(encoding="utf-8"):
                continue  # cheap prefilter: parsing all of src/ is slow
            tree = _tree(path)
        except (SyntaxError, UnicodeDecodeError):
            continue
        if _slsqp_literals(tree):
            found.add(rel)
    unlisted = sorted(found - set(ALLOWED_SLSQP))
    assert not unlisted, (
        "new SLSQP use outside the allow-list; route convex problems through "
        f"loop_nullspace.qp_core or add a justified entry: {unlisted}"
    )
    stale = sorted(set(ALLOWED_SLSQP) - found)
    assert not stale, f"allow-list entries without an SLSQP use: {stale}"
    assert all(len(reason) > 20 for reason in ALLOWED_SLSQP.values())
