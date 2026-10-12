"""Build isolated regular packages for frozen-source verifier fixtures."""

from pathlib import Path


def create_frozen_study_package(root: Path) -> Path:
    """Return a study package whose parents cannot merge with installed sources."""
    package = root / "src/tools/shot_pattern_analysis"
    package.mkdir(parents=True)
    for directory in (root / "src", root / "src/tools", package):
        (directory / "__init__.py").write_text("", encoding="utf-8")
    return package
