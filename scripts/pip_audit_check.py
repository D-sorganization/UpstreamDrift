#!/usr/bin/env python3
"""Run pip-audit with active (non-expired) waivers from a YAML file.

Implements the waiver mechanism for issue #3076: CVE ignore entries must
each carry a machine-readable expiry date, and expired waivers are NOT
passed to pip-audit (so CI fails until the waiver is renewed or the
dependency is bumped).

Usage::

    python3 scripts/pip_audit_check.py \\
        --requirements requirements-dev.lock \\
        --waivers .github/pip_audit_waivers.yaml

Additional args after ``--`` are forwarded to ``pip-audit``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import yaml  # PyYAML is a transitive dep of pip-audit
except ModuleNotFoundError as exc:  # pragma: no cover - defensive
    print(f"PyYAML is required: {exc}", file=sys.stderr)
    sys.exit(2)


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WAIVERS = REPO_ROOT / ".github" / "pip_audit_waivers.yaml"


def load_waivers(path: Path) -> list[dict[str, Any]]:
    """Load and schema-check the waivers file.

    Returns the list of waiver dicts. Raises SystemExit on schema error.
    """
    if not path.exists():
        return []
    raw = yaml.safe_load(path.read_text()) or {}
    waivers = raw.get("waivers", [])
    if not isinstance(waivers, list):
        raise SystemExit(
            f"{path}: top-level 'waivers' must be a list, got {type(waivers).__name__}"
        )
    required = {"cve", "package", "reason", "expires"}
    errors: list[str] = []
    for idx, entry in enumerate(waivers):
        if not isinstance(entry, dict):
            errors.append(f"waivers[{idx}] is not a mapping")
            continue
        missing = required - entry.keys()
        if missing:
            errors.append(
                f"waivers[{idx}] ({entry.get('cve', '?')}) missing fields: "
                f"{sorted(missing)}"
            )
            continue
        try:
            _parse_expiry(entry["expires"])
        except ValueError as exc:
            errors.append(f"waivers[{idx}] ({entry['cve']}) invalid expires: {exc}")
    if errors:
        raise SystemExit("; ".join(errors))
    return waivers


def _parse_expiry(value: Any) -> _dt.date:
    """Parse a YYYY-MM-DD string or date object into a date."""
    if isinstance(value, _dt.date) and not isinstance(value, _dt.datetime):
        return value
    if isinstance(value, _dt.datetime):
        return value.date()
    if isinstance(value, str):
        try:
            return _dt.date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"expected YYYY-MM-DD, got {value!r}") from exc
    raise ValueError(f"expected YYYY-MM-DD string, got {type(value).__name__}")


def active_ignores(
    waivers: list[dict[str, Any]],
    today: _dt.date | None = None,
) -> tuple[list[str], list[str]]:
    """Return (active_cve_ids, expired_cve_ids)."""
    today = today or _dt.date.today()
    active: list[str] = []
    expired: list[str] = []
    for entry in waivers:
        cve = str(entry["cve"])
        when = _parse_expiry(entry["expires"])
        if when < today:
            expired.append(f"{cve} (expired {when.isoformat()})")
        else:
            active.append(cve)
    return active, expired


def build_command(
    requirements: Path | None,
    active_cves: list[str],
    extra: list[str],
) -> list[str]:
    """Build the pip-audit command line."""
    cmd: list[str] = [sys.executable, "-m", "pip_audit"]
    if requirements is not None:
        cmd.extend(["-r", str(requirements)])
    for cve in active_cves:
        cmd.extend(["--ignore-vuln", cve])
    cmd.extend(extra)
    return cmd


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    argv = list(argv) if argv is not None else sys.argv[1:]
    # Split on '--' so args after it forward to pip-audit untouched.
    if "--" in argv:
        idx = argv.index("--")
        own_args, pass_through = argv[:idx], argv[idx + 1 :]
    else:
        own_args, pass_through = argv, []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirements", type=Path, default=None)
    parser.add_argument("--waivers", type=Path, default=DEFAULT_WAIVERS)
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print the active ignores and exit 0 without running pip-audit.",
    )
    args = parser.parse_args(own_args)
    waivers = load_waivers(args.waivers)
    active, expired = active_ignores(waivers)
    if expired:
        print(
            "pip_audit_check: expired waivers (will NOT be ignored): "
            + ", ".join(expired),
            file=sys.stderr,
        )
    if active:
        print(
            "pip_audit_check: active waivers: " + ", ".join(active),
            file=sys.stderr,
        )
    if args.print_only:
        return 0
    cmd = build_command(args.requirements, active, pass_through)
    print("pip_audit_check: running", " ".join(cmd), file=sys.stderr)
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
