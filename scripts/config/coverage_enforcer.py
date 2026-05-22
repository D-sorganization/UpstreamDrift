#!/usr/bin/env python3
"""Coverage threshold enforcer for CI (issue #5910).

Reads ``pyproject.toml`` for the configured ``fail_under`` value, validates
it is consistent with the threshold expected by CI, and optionally parses a
``coverage.xml`` report to surface per-package coverage below configurable
per-module floors.

Usage::

    python3 scripts/config/coverage_enforcer.py [coverage.xml]

Exit codes:
    0  All thresholds met (or no coverage.xml provided — config-only check).
    1  One or more thresholds violated.
"""

from __future__ import annotations

import logging
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger(__name__)

# Minimum overall threshold that CI requires (matches --cov-fail-under in ci-standard.yml).
_CI_MIN_THRESHOLD: float = 50.0

# Per-package coverage floors (production-critical surfaces — issue #3939).
# Keys are dotted module path prefixes; values are minimum percentages.
_PER_PACKAGE_FLOORS: dict[str, float] = {
    "src.api": 85.0,
    "src.engines.common": 85.0,
    "src.shared.python.task_management": 80.0,
    "src.shared.python": 70.0,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_pyproject_threshold(repo_root: Path) -> float:
    """Return the ``fail_under`` value from ``pyproject.toml``.

    Supports both Python 3.11+ ``tomllib`` (stdlib) and the ``tomli``
    back-port used on 3.10.

    Args:
        repo_root: Path to the repository root directory.

    Returns:
        The configured ``fail_under`` value as a float.

    Raises:
        FileNotFoundError: If ``pyproject.toml`` does not exist.
        KeyError: If the ``[tool.coverage.report]`` section is missing.
    """
    toml_path = repo_root / "pyproject.toml"
    if not toml_path.is_file():
        raise FileNotFoundError(f"pyproject.toml not found at {toml_path}")

    if sys.version_info >= (3, 11):
        import tomllib  # noqa: PLC0415  (stdlib on 3.11+)

        with open(toml_path, "rb") as fh:
            data = tomllib.load(fh)
    else:
        try:
            import tomli  # noqa: PLC0415  (back-port for 3.10)

            with open(toml_path, "rb") as fh:
                data = tomli.load(fh)
        except ImportError as exc:
            raise ImportError(
                "tomli is required on Python < 3.11 to parse pyproject.toml; "
                "add it to dev dependencies."
            ) from exc

    try:
        threshold: float = float(data["tool"]["coverage"]["report"]["fail_under"])
    except KeyError as exc:
        raise KeyError(
            "Expected [tool.coverage.report].fail_under in pyproject.toml"
        ) from exc

    return threshold


def _parse_coverage_xml(xml_path: Path) -> dict[str, float]:
    """Parse a ``coverage.xml`` file and return per-package line-rate percentages.

    Args:
        xml_path: Path to the coverage.xml file produced by pytest-cov.

    Returns:
        Mapping of package name (dotted) to coverage percentage (0–100).
    """
    tree = ET.parse(xml_path)  # noqa: S314 — local file, not network data
    root = tree.getroot()

    results: dict[str, float] = {}
    for pkg in root.iter("package"):
        name: str | None = pkg.get("name")
        line_rate_str: str | None = pkg.get("line-rate")
        if name is None or line_rate_str is None:
            continue
        try:
            pct = float(line_rate_str) * 100.0
        except ValueError:
            log.warning("Cannot parse line-rate=%r for package %r", line_rate_str, name)
            continue
        # Normalise path separators to dotted module form.
        dotted = name.replace("/", ".").replace("\\", ".").strip(".")
        results[dotted] = pct

    return results


def _check_per_package(
    coverage: dict[str, float],
    floors: dict[str, float],
) -> list[str]:
    """Return a list of violation messages for packages below their floor.

    Args:
        coverage: Mapping of package dotted name to coverage percentage.
        floors: Mapping of package prefix to minimum coverage percentage.

    Returns:
        List of human-readable violation strings (empty if all pass).
    """
    violations: list[str] = []
    for prefix, floor in floors.items():
        matching = {
            name: pct
            for name, pct in coverage.items()
            if name == prefix or name.startswith(prefix + ".")
        }
        if not matching:
            log.debug("No packages matched floor prefix %r — skipping", prefix)
            continue
        for name, pct in matching.items():
            if pct < floor:
                violations.append(
                    f"  {name}: {pct:.1f}% < {floor:.0f}% (floor for {prefix})"
                )
    return violations


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """Run the coverage enforcer.

    Returns:
        Exit code (0 = pass, 1 = violations found).
    """
    repo_root = Path(__file__).resolve().parents[2]
    violations: list[str] = []

    # --- Step 1: Validate pyproject.toml threshold against CI expectation ---
    try:
        configured = _load_pyproject_threshold(repo_root)
    except (FileNotFoundError, KeyError, ImportError) as exc:
        log.error("Cannot read coverage threshold from pyproject.toml: %s", exc)
        return 1

    log.info("pyproject.toml [tool.coverage.report] fail_under = %.1f%%", configured)

    if configured < _CI_MIN_THRESHOLD:
        violations.append(
            f"pyproject.toml fail_under={configured:.1f}% is below the CI minimum "
            f"of {_CI_MIN_THRESHOLD:.1f}% (issue #5910)"
        )

    # --- Step 2: Optional per-package check from coverage.xml ---------------
    xml_args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if xml_args:
        xml_path = Path(xml_args[0])
        if not xml_path.is_file():
            log.error("coverage.xml not found at %s", xml_path)
            return 1

        log.info("Parsing coverage report: %s", xml_path)
        coverage = _parse_coverage_xml(xml_path)
        log.info("Found %d packages in coverage report", len(coverage))

        pkg_violations = _check_per_package(coverage, _PER_PACKAGE_FLOORS)
        violations.extend(pkg_violations)
    else:
        log.info(
            "No coverage.xml provided — skipping per-package threshold check. "
            "Pass a coverage.xml path to enable it."
        )

    # --- Report ---------------------------------------------------------------
    if violations:
        log.error("Coverage threshold violations found:")
        for v in violations:
            log.error(v)
        return 1

    log.info("All coverage thresholds satisfied.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
