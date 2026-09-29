#!/usr/bin/env python3
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "technical_debt" / "TODO_FIXME_REGISTER.md"
PATTERN = re.compile(r"\b(TRACKED_TASK|TRACKED_DEFECT)\b")


def main() -> int:
    cp = subprocess.run(
        ["rg", "-n", "TRACKED_TASK|TRACKED_DEFECT", "src", "tests", "scripts"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    lines = [ln for ln in cp.stdout.splitlines() if ln.strip()]

    content = [
        "# TRACKED_TASK/TRACKED_DEFECT Debt Register",
        "",
        "This register is generated from inline TRACKED_TASK/TRACKED_DEFECT markers.",
        "Target SLA: convert each marker to a tracked issue within 14 days.",
        "",
        f"Total markers: {len(lines)}",
        "",
        "| Marker | Location | Suggested Action |",
        "|---|---|---|",
    ]

    for ln in lines:
        parts = ln.split(":", 2)
        if len(parts) < 3:
            continue
        file_path, line_no, text = parts
        marker = "TRACKED_TASK" if "TRACKED_TASK" in text else "TRACKED_DEFECT"
        action = "Create/Link GitHub issue and assign owner"
        content.append(f"| {marker} | `{file_path}:{line_no}` | {action} |")

    OUT.write_text("\n".join(content) + "\n", encoding="utf-8")
    sys.stdout.write(f"wrote {OUT.relative_to(ROOT)} with {len(lines)} markers\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
