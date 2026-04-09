#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if TYPE_CHECKING:
    from src.launchers.launcher_provider_compatibility import CompatibilityIssue


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a provider manifest against the shared launcher harness."
    )
    parser.add_argument("--manifest", required=True, help="Path to model_pack.yaml")
    parser.add_argument(
        "--provider-root",
        help="Optional provider root override; defaults to the manifest directory",
    )
    return parser


def _issue_to_dict(issue: CompatibilityIssue) -> dict[str, object]:
    return {
        "code": issue.code,
        "category": issue.category,
        "message": issue.message,
        "context": issue.context,
    }


def main(argv: list[str] | None = None) -> int:
    from src.launchers.launcher_provider_compatibility import (
        validate_provider_manifest,
    )

    parser = _build_parser()
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest)
    provider_root = Path(args.provider_root) if args.provider_root else None
    report = validate_provider_manifest(manifest_path, provider_root)

    payload = {
        "provider": report.provider,
        "pack_id": report.pack_id,
        "manifest_path": str(report.manifest_path),
        "is_compatible": report.is_compatible,
        "issues": [_issue_to_dict(issue) for issue in report.issues],
        "results": [
            {
                "model_id": result.model_id,
                "provider": result.provider,
                "canonical_id": result.canonical_id,
                "is_compatible": result.is_compatible,
                "issues": [_issue_to_dict(issue) for issue in result.issues],
            }
            for result in report.results
        ],
    }
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0 if report.is_compatible else 1


if __name__ == "__main__":
    raise SystemExit(main())
