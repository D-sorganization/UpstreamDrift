"""NSX-45 help surface gate: loop null-space interpretation doc exists and is wired.

The guide must describe only behaviour that exists in source.  Every symbol it
cites is checked against the NSX package, every preset against the registry,
and every relative link against the filesystem.  Title case is enforced with
the same ``findings_for_text`` the repository's governance gate uses.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
HELP = ROOT / "docs/help/loop_nullspace.md"
PKG = ROOT / "src/shared/python/loop_nullspace"
REGISTRY = ROOT / "src/config/loop_nullspace_registry.json"
TITLE_CASE_SCRIPT = ROOT / "scripts/check_document_title_case.py"

REQUIRED_HEADINGS = (
    "## What You See",
    "## Invariant and Family-Dependent Quantities",
    "## Why the Hand Split Is Not Measured",
    "## Presets",
    "## Reading a Certificate",
    "## What Restriction Sensitivity Is Not",
    "## Non-Claims",
    "## Related",
)
CITED_SYMBOLS = (
    "AllocationDependence",
    "FAMILY_DEPENDENT",
    "INVARIANT",
    "net_club_wrench",
    "ValidationStatus",
    "ClosureKind",
    "WELD6",
    "SHAFT_FREE5",
    "BALL3",
    "LeafCertificate",
    "global-convex",
    "flat-invariant",
    "RESTRICTION_LABEL",
    "DisplayAction",
    "SignClass",
)


def _help_text() -> str:
    return HELP.read_text(encoding="utf-8")


def _package_source() -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in sorted(PKG.glob("*.py")))


def _title_case_module():
    spec = importlib.util.spec_from_file_location("title_case", TITLE_CASE_SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve the module by name
    spec.loader.exec_module(module)
    return module


def test_loop_nullspace_help_page_exists_with_title_case() -> None:
    text = _help_text()
    assert "title: Loop Null-Space Exploration" in text
    assert "## Non-Claims" in text
    assert "family-dependent" in text.lower() or "Family-dependent" in text


def test_help_page_has_every_required_interpretation_section() -> None:
    text = _help_text()
    for heading in REQUIRED_HEADINGS:
        assert heading in text, heading


def test_help_page_headings_are_title_case() -> None:
    findings = _title_case_module().findings_for_text(HELP, _help_text())
    assert findings == [], findings


def test_help_page_explains_the_hand_split_is_not_measured() -> None:
    text = _help_text()
    section = text.split("## Why the Hand Split Is Not Measured", 1)[1]
    section = section.split("\n## ", 1)[0]
    assert "not measured" in section.lower()
    assert "FAMILY_DEPENDENT" in section or "family-dependent" in section.lower()


def test_every_registry_preset_is_named() -> None:
    presets = json.loads(REGISTRY.read_text(encoding="utf-8"))["presets"]
    assert presets, "registry declares no presets"
    text = _help_text()
    for preset in presets:
        assert f"`{preset['id']}`" in text, preset["id"]


def test_cited_symbols_exist_in_the_nsx_package() -> None:
    text = _help_text()
    source = _package_source()
    for symbol in CITED_SYMBOLS:
        assert symbol in text, f"guide does not cite {symbol}"
        assert symbol in source, f"{symbol} is not defined in the NSX package"


def test_relative_links_resolve() -> None:
    text = _help_text()
    for target in re.findall(r"\]\((?!https?://|#)([^)#\s]+)", text):
        assert (HELP.parent / target).is_file(), target


def test_grip_wrench_help_links_the_guide() -> None:
    grip = (ROOT / "docs/help/grip_wrench_plots.md").read_text(encoding="utf-8")
    assert "loop_nullspace.md" in grip
