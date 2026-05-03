#!/usr/bin/env python3
"""Validate pip-audit waiver metadata and emit ignore flags."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass(frozen=True)
class Waiver:
    """Structured representation of one pip-audit waiver."""

    id: str
    package: str
    reason: str
    expires_at: date


def load_waivers(path: Path) -> list[Waiver]:
    """Load waiver definitions from JSON.

    Preconditions:
        path points to a JSON object with a top-level ``waivers`` list.
    """
    if not path.exists():
        raise FileNotFoundError(f"pip-audit waiver file not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8") or "{}")
    if not isinstance(raw, dict):
        raise ValueError("waiver file must contain a JSON object")
    items = raw.get("waivers", [])
    if not isinstance(items, list):
        raise ValueError("waivers must be a list")

    waivers: list[Waiver] = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("each waiver entry must be a mapping")
        try:
            waiver = Waiver(
                id=str(item["id"]).strip(),
                package=str(item["package"]).strip(),
                reason=str(item["reason"]).strip(),
                expires_at=date.fromisoformat(str(item["expires_at"]).strip()),
            )
        except KeyError as exc:
            raise ValueError(f"missing waiver field: {exc.args[0]}") from exc

        if not waiver.id or not waiver.package or not waiver.reason:
            raise ValueError("waiver id, package, and reason must be non-empty")
        waivers.append(waiver)

    return waivers


def find_expired_waivers(
    waivers: list[Waiver], today: date | None = None
) -> list[Waiver]:
    """Return waivers whose expiry date has passed."""
    current = today or date.today()
    return [waiver for waiver in waivers if waiver.expires_at < current]


def build_ignore_flags(waivers: list[Waiver]) -> list[str]:
    """Build CLI flags for pip-audit."""
    flags: list[str] = []
    for waiver in waivers:
        flags.extend(["--ignore-vuln", waiver.id])
    return flags


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--waiver-file",
        type=Path,
        default=Path("scripts/config/pip_audit_waivers.json"),
        help="Path to the pip-audit waiver manifest.",
    )
    return parser.parse_args()


def main() -> int:
    """Validate waivers and print ignore flags for shell consumption."""
    args = parse_args()
    waivers = load_waivers(args.waiver_file)
    expired = find_expired_waivers(waivers)
    if expired:
        for waiver in expired:
            print(
                f"Expired pip-audit waiver: {waiver.id} for {waiver.package} "
                f"(expired {waiver.expires_at.isoformat()})"
            )
        return 1

    for flag in build_ignore_flags(waivers):
        print(flag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
