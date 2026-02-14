#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = [
    ROOT / "docs" / "README.md",
    ROOT / "docs" / "assessments" / "README.md",
    ROOT / "docs" / "adr" / "README.md",
    ROOT / "docs" / "adr" / "ADR_TEMPLATE.md",
    ROOT / "docs" / "governance" / "DOCS_GOVERNANCE.md",
]


def _git_changed_files() -> list[str]:
    base_ref = "origin/main"
    cp = subprocess.run(
        ["git", "diff", "--name-only", f"{base_ref}...HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if cp.returncode != 0:
        return []
    return [line.strip() for line in cp.stdout.splitlines() if line.strip()]


def _fail(msg: str) -> int:
    sys.stderr.write(msg + "\n")
    return 1


def main() -> int:
    missing = [str(p.relative_to(ROOT)) for p in REQUIRED_FILES if not p.exists()]
    if missing:
        return _fail(
            "Missing required docs governance files:\n- " + "\n- ".join(missing)
        )

    changed = _git_changed_files()
    changed_set = set(changed)

    assessment_changes = [
        p
        for p in changed
        if p.startswith("docs/assessments/") and p != "docs/assessments/README.md"
    ]
    if assessment_changes and "docs/assessments/README.md" not in changed_set:
        return _fail(
            "docs/assessments changes detected without updating docs/assessments/README.md"
        )

    adr_changes = [
        p
        for p in changed
        if p.startswith("docs/adr/")
        and p not in {"docs/adr/README.md", "docs/adr/ADR_TEMPLATE.md"}
    ]
    if adr_changes and "docs/adr/README.md" not in changed_set:
        return _fail("ADR changes detected without updating docs/adr/README.md")

    sys.stdout.write("docs governance checks passed\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
