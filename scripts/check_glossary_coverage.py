"""Check glossary coverage for physics parameter names.

Scans Python source files under ``src/shared/python/physics/`` for
``@dataclass``/``pydantic.BaseModel`` field names and reports which ones
are missing from the glossary.

Per issue #3165, this is an informational report and always exits 0.

Usage:
    python3 scripts/check_glossary_coverage.py
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import re
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
GLOSSARY_PATH = (
    REPO_ROOT / "src" / "shared" / "python" / "ai" / "data" / "glossary_core.json"
)
SRC_ROOT = REPO_ROOT / "src"
PHYSICS_ROOT = REPO_ROOT / "src" / "shared" / "python" / "physics"

# Patterns that suggest physics parameter names
_PHYSICS_PATTERN = re.compile(
    r"^(?:"
    r"velocity|speed|acceleration|force|torque|mass|inertia|energy|momentum|"
    r"angular_velocity|angular_acceleration|spin|lift|drag|gravity|pressure|"
    r"density|coefficient|friction|restitution|stiffness|damping|"
    r"kinetic_energy|potential_energy|launch_angle|backspin|sidespin|"
    r"ball_speed|club_speed|impact|smash_factor|carry|total_distance|"
    r"loft|lie|shaft_flex|swing_path|face_angle|attack_angle|"
    r"dynamic_loft|spin_axis|apex|flight_time|trajectory"
    r").*$"
)


def _extract_dataclass_fields(src_root: Path) -> set[str]:
    """Collect field names from dataclasses / pydantic BaseModels.

    Walks ``src_root`` looking for class bodies marked with the
    ``@dataclass`` decorator or inheriting from ``BaseModel``, and
    returns their annotated field names.
    """
    fields: set[str] = set()
    if not src_root.exists():
        return fields
    for py_file in src_root.rglob("*.py"):
        try:
            source = py_file.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(source, filename=str(py_file))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            is_dataclass = any(
                (isinstance(d, ast.Name) and d.id == "dataclass")
                or (
                    isinstance(d, ast.Call)
                    and isinstance(d.func, ast.Name)
                    and d.func.id == "dataclass"
                )
                for d in node.decorator_list
            )
            inherits_basemodel = any(
                isinstance(b, ast.Name) and b.id == "BaseModel" for b in node.bases
            )
            if not (is_dataclass or inherits_basemodel):
                continue
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(
                    stmt.target, ast.Name
                ):
                    fields.add(stmt.target.id)
    return fields


def _load_glossary_keys() -> set[str]:
    """Load all glossary keys.

    Returns:
        Set of glossary term keys.
    """
    if not GLOSSARY_PATH.exists():
        logger.error("Glossary not found at %s", GLOSSARY_PATH)
        return set()
    with open(GLOSSARY_PATH, encoding="utf-8") as f:
        data: list[dict] = json.load(f)
    return {entry["key"] for entry in data}


def _extract_parameter_names(src_root: Path) -> set[str]:
    """Walk src/ and extract snake_case names matching physics patterns.

    Uses AST parsing to find function argument names, variable assignments,
    and class attribute names from all Python files.

    Args:
        src_root: Root of source tree to scan.

    Returns:
        Set of candidate physics parameter names.
    """
    candidates: set[str] = set()
    for py_file in src_root.rglob("*.py"):
        try:
            source = py_file.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(source, filename=str(py_file))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.FunctionDef):
                names = [arg.arg for arg in node.args.args]
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names = [node.target.id]
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        names.append(t.id)
            for name in names:
                if _PHYSICS_PATTERN.match(name):
                    candidates.add(name)
    return candidates


def main() -> None:
    """Run glossary coverage check (informational, never fails).

    Per issue #3165 this walks ``src/shared/python/physics`` for dataclass
    and pydantic BaseModel field names, cross-references them against the
    glossary keys, and prints uncovered field names to stdout. Always
    exits 0.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--physics-only",
        action="store_true",
        default=True,
        help=(
            "Scan only src/shared/python/physics (default True). "
            "Legacy flag kept for compatibility."
        ),
    )
    _ = parser.parse_args()

    glossary_keys = _load_glossary_keys()
    if not glossary_keys:
        logger.warning("No glossary keys loaded.")

    field_names = _extract_dataclass_fields(PHYSICS_ROOT)
    if not field_names:
        logger.info("No dataclass/BaseModel fields found under %s", PHYSICS_ROOT)
        sys.exit(0)

    covered = {p for p in field_names if p in glossary_keys}
    missing = sorted(field_names - covered)

    total = len(field_names)
    coverage_pct = (len(covered) / total * 100) if total else 0.0
    logger.info(
        "Glossary coverage (physics/): %d/%d fields (%.1f%%)",
        len(covered),
        total,
        coverage_pct,
    )

    if missing:
        logger.info("Uncovered fields (%d):", len(missing))
        for name in missing:
            # Print to stdout so CI logs capture them.
            sys.stdout.write(f"  - {name}\n")

    sys.exit(0)


if __name__ == "__main__":
    main()
