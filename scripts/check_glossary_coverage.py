"""Check glossary coverage for physics parameter names.

Scans Python source files under src/ for physics parameter names
(snake_case identifiers that look like physics quantities) and checks
whether each one has a corresponding entry in the glossary.

Exit code 0 if coverage >= threshold, 1 otherwise.

Usage:
    python3 scripts/check_glossary_coverage.py [--threshold 0.3]
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
    """Run glossary coverage check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Minimum fraction of params that must be in glossary (default: 0.3)",
    )
    args = parser.parse_args()

    glossary_keys = _load_glossary_keys()
    if not glossary_keys:
        logger.error("No glossary keys loaded — aborting.")
        sys.exit(1)

    param_names = _extract_parameter_names(SRC_ROOT)
    if not param_names:
        logger.warning("No physics parameter names found in %s", SRC_ROOT)
        sys.exit(0)

    covered = {p for p in param_names if p in glossary_keys}
    missing = param_names - covered

    coverage = len(covered) / len(param_names)
    logger.info(
        "Glossary coverage: %d/%d param names (%.1f%%)",
        len(covered),
        len(param_names),
        coverage * 100,
    )

    if missing:
        logger.info("Parameters NOT in glossary (%d):", len(missing))
        for name in sorted(missing):
            logger.info("  - %s", name)

    if coverage < args.threshold:
        logger.error(
            "Coverage %.1f%% is below threshold %.1f%%",
            coverage * 100,
            args.threshold * 100,
        )
        sys.exit(1)

    logger.info("Glossary coverage check passed.")


if __name__ == "__main__":
    main()
