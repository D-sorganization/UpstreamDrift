#!/usr/bin/env python3
"""CI gate: enforce the layered URDF subsystem boundary from ADR 0007.

Per the layered architecture decision (issue #4521 / ADR 0007):

- ``model_generation`` is the low-level URDF / mesh / inertia toolkit.
- ``humanoid_character_builder`` is the anthropometric-domain layer that
  composes the toolkit.

This script enforces three rules statically (no execution required):

1. ``humanoid_character_builder/`` MUST NOT contain its own URDF XML
   writer. Files matching that pattern must delegate to
   ``model_generation/builders/urdf_writer.py``.
2. ``humanoid_character_builder/mesh/`` files MUST delegate to
   ``model_generation/inertia/`` for primitive inertia computation.
3. Source files in ``model_generation/`` MUST NOT import from
   ``humanoid_character_builder/`` (the toolkit layer cannot depend on
   the domain layer).

Exits 0 if all rules pass, 1 if any violation is detected.

Usage::

    python3 scripts/ci/check_urdf_layering.py

Output is suitable for GitHub Actions step summaries.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HCB = REPO_ROOT / "src" / "shared" / "python" / "humanoid_character_builder"
MG = REPO_ROOT / "src" / "shared" / "python" / "model_generation"

# Import patterns that signal a layering violation.
HCB_OWNED_URDF_WRITER_HINTS: tuple[str, ...] = (
    "<robot",
    "ET.SubElement",
    "ET.Element(",
    "DefusedET.parse",
)

# Files in humanoid_character_builder that *are allowed* to construct
# URDF XML directly (the existing emitter, until #4601 lands).
# Once #4601 ships these will be removed from this allowlist.
HCB_URDF_WRITER_ALLOWLIST: set[Path] = {
    HCB / "generators" / "urdf_xml_builder.py",
    HCB / "generators" / "_urdf_xml_writer.py",
    HCB / "generators" / "urdf_generator.py",
    # The canonical adapter explicitly composes model_generation's URDFWriter
    # — that's its whole purpose. The static check's heuristic flags it
    # because the file mentions <robot> in its docstring and calls .write(),
    # but it's the OPPOSITE of a violation: it's the bridge to the canonical
    # writer.
    HCB / "generators" / "_canonical_adapter.py",
}


def _iter_python_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*.py") if "__pycache__" not in p.parts]


# Files in model_generation that are explicitly *allowed* to import from
# humanoid_character_builder. These are intentional domain-aware shims
# that re-export the domain types under the toolkit's namespace for
# convenience. Per ADR 0007 §"Consequences": "Some integrations may need
# updating (any caller currently importing from
# humanoid_character_builder.mesh.primitive_inertia would now get a
# re-export from model_generation.inertia)."
MG_SHIM_ALLOWLIST: set[Path] = {
    MG / "humanoid" / "__init__.py",
    MG / "mesh" / "__init__.py",
}


def check_no_reverse_imports() -> list[str]:
    """Rule 3: model_generation must not import from humanoid_character_builder.

    Shim modules are allowlisted (see ``MG_SHIM_ALLOWLIST``).
    """
    violations: list[str] = []
    for path in _iter_python_files(MG):
        if path in MG_SHIM_ALLOWLIST:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if "humanoid_character_builder" in line and (
                "import " in line or "from " in line
            ):
                rel = path.relative_to(REPO_ROOT)
                violations.append(
                    f"{rel}:{lineno}: model_generation cannot import "
                    f"from humanoid_character_builder (toolkit layer cannot "
                    f"depend on domain layer). If this is an intentional "
                    f"shim, add it to MG_SHIM_ALLOWLIST in this script."
                )
    return violations


def check_inertia_delegation() -> list[str]:
    """Rule 2: hcb.mesh primitive inertia files must import from model_generation.inertia."""
    violations: list[str] = []
    target = HCB / "mesh" / "primitive_inertia.py"
    if not target.exists():
        return violations
    text = target.read_text(encoding="utf-8")
    if "from model_generation.inertia" not in text:
        rel = target.relative_to(REPO_ROOT)
        violations.append(
            f"{rel}: missing `from model_generation.inertia.primitives import ...`. "
            f"Per ADR 0007, primitive inertia must delegate to model_generation."
        )
    return violations


def check_urdf_writer_layer() -> list[str]:
    """Rule 1 (advisory until #4601 lands): non-allowlisted hcb files must not emit URDF XML directly."""
    violations: list[str] = []
    for path in _iter_python_files(HCB):
        if path in HCB_URDF_WRITER_ALLOWLIST:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        # Check whether the file constructs URDF root XML elements directly.
        # We're looking for any signs of low-level XML emission outside the
        # allowlist; the file should compose model_generation.builders.urdf_writer
        # if it needs to produce URDF XML.
        if "<robot" in text and "import " in text and ".write(" in text:
            rel = path.relative_to(REPO_ROOT)
            violations.append(
                f"{rel}: appears to construct URDF XML directly outside the "
                f"allowlist. Use model_generation.builders.urdf_writer.URDFWriter "
                f"(see ADR 0007 / #4601)."
            )
    return violations


def main() -> int:
    all_violations: list[str] = []
    all_violations += check_no_reverse_imports()
    all_violations += check_inertia_delegation()
    all_violations += check_urdf_writer_layer()

    if not all_violations:
        print("OK: URDF layering rules satisfied (ADR 0007).")
        return 0

    print("URDF layering violations detected:\n")
    for v in all_violations:
        print(f"  {v}")
    print("\nSee docs/adr/0007-canonical-urdf-subsystem.md for the rules.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
