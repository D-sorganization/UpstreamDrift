"""Entry-point invariants for launcher documentation and packaging (#8831, #8859).

Guards against the documented entry point drifting away from the canonical
``launch_upstream_drift.py`` / ``upstream-drift`` console script, and against
the quarantined dead launcher shells being resurrected.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]

CANONICAL_COMMAND = "python launch_upstream_drift.py"
CANONICAL_MODES = ("--classic", "--api-only", "--engine")

DELETED_LAUNCHER_MODULES = (
    "src/launchers/unified_launcher.py",
    "src/launchers/golf_suite_launcher.py",
    "src/launchers/model_registry.py",
)


def test_docs_readme_uses_canonical_launch_command() -> None:
    """docs/README.md must instruct the canonical launcher, with its modes."""
    text = (REPO_ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    assert CANONICAL_COMMAND in text, (
        f"docs/README.md must document the canonical entry point '{CANONICAL_COMMAND}'"
    )
    for mode in CANONICAL_MODES:
        assert mode in text, f"docs/README.md must mention launcher mode '{mode}'"


def test_docs_readme_does_not_present_deprecated_shim_as_primary() -> None:
    """The deprecated shim must not be documented as the command to run."""
    text = (REPO_ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    assert "python launch_golf_suite.py" not in text, (
        "docs/README.md still instructs the deprecated launch_golf_suite.py; "
        "the canonical command is python launch_upstream_drift.py (#8831)"
    )
    for match in re.finditer(r"^.*launch_golf_suite\.py.*$", text, flags=re.MULTILINE):
        line = match.group(0).lower()
        assert "primary" not in line, (
            "docs/README.md must not call launch_golf_suite.py 'primary': "
            f"{match.group(0)!r}"
        )


def test_root_readme_recommends_canonical_launcher() -> None:
    """Root README.md must recommend launch_upstream_drift.py."""
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert CANONICAL_COMMAND in text
    for mode in CANONICAL_MODES:
        assert mode in text


def test_console_script_points_at_canonical_launcher() -> None:
    """[project.scripts] upstream-drift must map to launch_upstream_drift:main."""
    with (REPO_ROOT / "pyproject.toml").open("rb") as fh:
        pyproject = tomllib.load(fh)
    scripts = pyproject["project"]["scripts"]
    assert scripts["upstream-drift"] == "launch_upstream_drift:main"


def test_quarantined_launcher_shells_stay_deleted() -> None:
    """Dead launcher shells removed in #8831/#8859 must not be re-added."""
    for rel_path in DELETED_LAUNCHER_MODULES:
        assert not (REPO_ROOT / rel_path).exists(), (
            f"{rel_path} was deleted (#8831/#8859) because it shadowed the "
            "canonical launcher/registry; do not re-add it"
        )
