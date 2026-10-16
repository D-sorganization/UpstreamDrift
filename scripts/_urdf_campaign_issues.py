#!/usr/bin/env python3
"""One-shot script: create the URDF Hardening Campaign issues.

Run from the UpstreamDrift repo root:
    python3 scripts/_urdf_campaign_issues.py

After running, prints a summary and writes issue numbers to
scripts/_urdf_campaign_issues.json.

This script is intentionally a one-off; safe to delete after use.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from _urdf_campaign_issues_parts import ISSUES

MILESTONE = "URDF Hardening Campaign"
COMMON_LABELS = ["URDF-hardening"]


def gh_issue_create(title: str, body: str, labels: list[str]) -> int:
    all_labels = sorted(set(COMMON_LABELS + labels))
    cmd = [
        "gh",
        "issue",
        "create",
        "--title",
        title,
        "--body",
        body,
        "--milestone",
        MILESTONE,
    ]
    for lbl in all_labels:
        cmd += ["--label", lbl]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True)
    # gh prints the URL on success; extract issue number
    url = out.stdout.strip().splitlines()[-1]
    num = int(url.rsplit("/", 1)[-1])
    print(f"  #{num}  {title}")
    return num


SKIP_TITLES_PREFIX = (
    "[Tracking] URDF",
    "[arch] Decide canonical",
    "[arch] Define unified BuildResult",
    "[arch] Document architectural",
    "[bug] CharacterBuildResult",
    "[bug] BuildResult missing",
    "[bug] GeneratedMeshResult",
    "[bug] SMPLXMeshGenerator",
    "[bug] mesh_generator module missing",
    "[bug] test_security_fixes",
    "[bug] test_simscape",
    "[bug] test_build_humanoid_models",
    "[quality] Remove duplicated",
    "[quality] Resolve ARCHITECTURE_DEBT",
    "[quality] Replace silent except",
    "[tests] Cross-engine smoke test:",
    "[tests] URDF schema validation",
    "[tests] Determinism:",
)


def main() -> None:
    remaining = [
        (t, lbls, b) for t, lbls, b in ISSUES if not t.startswith(SKIP_TITLES_PREFIX)
    ]
    print(
        f"Creating {len(remaining)} issues (skipping {len(ISSUES) - len(remaining)} already created) against milestone '{MILESTONE}'...\n"
    )
    created: list[dict[str, object]] = []
    for title, labels, body in remaining:
        try:
            num = gh_issue_create(title, body, labels)
            created.append({"number": num, "title": title, "labels": labels})
        except subprocess.CalledProcessError as e:
            print(f"  ERR creating {title!r}: {e.stderr}")
            raise

    out = Path("scripts/_urdf_campaign_issues.json")
    out.write_text(json.dumps(created, indent=2))
    print(f"\nWrote summary to {out}")


if __name__ == "__main__":
    main()
