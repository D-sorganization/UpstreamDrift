"""Executable boundary for the OpenCap integration (ADR-0053, #11401)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ADR = ROOT / "docs" / "adr" / "0053-opencap-sidecar-licence-and-privacy-boundary.md"
SRC = ROOT / "src"
VOCABULARY = (
    SRC / "shared" / "python" / "motion_pipeline" / "sources" / "opencap_markers.py"
)

pytestmark = pytest.mark.gate

# Top-level packages that must stay out of the product's import graph.
_SIDECAR_ONLY_PACKAGES = ("opencap", "utilsAugmenter", "tensorflow")


def _text(path: Path) -> str:
    assert path.is_file(), f"missing governed document: {path.relative_to(ROOT)}"
    return path.read_text(encoding="utf-8")


def test_adr_records_licence_privacy_and_evidence_boundaries() -> None:
    text = _text(ADR).casefold()
    for phrase in (
        "apache-2.0",
        "subprocess or container",
        "never vendored",
        "non-commercial",
        "hrnet/mmpose",
        "opt-in",
        "recorded consent",
        "model-conditioned",
        "do not satisfy adr-0041 reconstruction",
        "units come from the model",
        "#11400",
    ):
        assert phrase in text, phrase


def _module_scope_imports(path: Path) -> set[str]:
    """Top-level package names imported at module scope (not in functions)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


def test_opencap_and_tensorflow_are_never_imported_at_module_scope() -> None:
    targets = set(_SIDECAR_ONLY_PACKAGES)
    offenders: list[str] = []
    for path in SRC.rglob("*.py"):
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if any(pkg in content for pkg in targets):
            for package in _module_scope_imports(path) & targets:
                offenders.append(f"{path.relative_to(ROOT)} imports {package}")
    assert not offenders, offenders


def test_opencap_marker_vocabulary_has_one_definition() -> None:
    assert VOCABULARY.is_file()
    defining = [
        path.relative_to(ROOT)
        for path in SRC.rglob("*.py")
        if "OPENCAP_AUGMENTED_MARKERS: tuple" in path.read_text(encoding="utf-8")
    ]
    assert defining == [VOCABULARY.relative_to(ROOT)], defining
