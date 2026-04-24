#!/usr/bin/env python3
"""Rewrite GitHub Actions ``uses:`` references to commit-SHA pins.

Supply-chain hardening per issue #3066: every external action pinned by
tag/branch is replaced with a 40-character commit SHA, with a trailing
comment preserving the original tag for human legibility, e.g.::

    uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd  # v6

Local path references (``./.github/...``) and Docker image references
(``docker://...``) are left untouched. Already-SHA-pinned lines are
left untouched. Entries not present in the pin map are listed in
``.github/actions_not_pinned.txt`` as follow-up items.

Usage::

    python3 scripts/pin_actions_to_sha.py              # rewrite files
    python3 scripts/pin_actions_to_sha.py --check      # exit 1 if any
                                                       # external uses
                                                       # is not a 40-hex
                                                       # SHA

Pin map: ``.github/action_pins.json`` (``{"pins": {action@tag: sha}}``).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"
PIN_MAP_PATH = REPO_ROOT / ".github" / "action_pins.json"
NOT_PINNED_PATH = REPO_ROOT / ".github" / "actions_not_pinned.txt"

# Capture: leading whitespace, "uses:", spaces, the action reference
# (up to '#' or end-of-line). We intentionally do not use YAML parsing
# because the trailing tag-comment we emit is a YAML-safe bare comment
# and we want to preserve exact formatting/whitespace of the rest of
# each workflow file.
USES_RE = re.compile(
    r"^(?P<indent>\s*)(?:-\s*)?"
    r"(?P<key>uses:\s*)"
    r"(?P<ref>[^\s#]+)"
    r"(?P<trailing>\s*(?:#.*)?)$"
)
SHA40_RE = re.compile(r"^[0-9a-f]{40}$")


def load_pin_map(path: Path) -> dict[str, str]:
    """Load the ``action@tag -> sha`` pin map."""
    if not path.exists():
        raise SystemExit(f"pin map not found: {path}")
    data = json.loads(path.read_text())
    pins = data.get("pins")
    if not isinstance(pins, dict):
        raise SystemExit(f"pin map must contain a 'pins' object: {path}")
    for key, sha in pins.items():
        if not SHA40_RE.match(sha):
            raise SystemExit(f"pin map entry {key!r} has non-40-hex SHA: {sha!r}")
    return pins


def is_local_or_docker(ref: str) -> bool:
    """Return True for refs that must not be SHA-pinned."""
    return ref.startswith(("./", "docker://"))


def split_ref(ref: str) -> tuple[str, str] | None:
    """Split ``owner/repo[/path]@tag`` into (action, tag). None if no @."""
    if "@" not in ref:
        return None
    action, _, tag = ref.rpartition("@")
    return action, tag


def rewrite_line(
    line: str,
    pins: dict[str, str],
    unresolved: set[str],
) -> str:
    """Return the rewritten line, updating ``unresolved`` in-place."""
    match = USES_RE.match(line.rstrip("\n"))
    if not match:
        return line
    ref = match.group("ref")
    if is_local_or_docker(ref):
        return line
    split = split_ref(ref)
    if split is None:
        return line
    action, tag = split
    if SHA40_RE.match(tag):
        return line  # already SHA-pinned
    key = f"{action}@{tag}"
    sha = pins.get(key)
    if sha is None:
        unresolved.add(key)
        return line
    key_text = match.group("key")
    # Reconstruct prefix exactly from the original to preserve
    # list-item formatting and leading whitespace.
    prefix = line[: match.start("key")]
    new_ref = f"{action}@{sha}"
    newline = "\n" if line.endswith("\n") else ""
    return f"{prefix}{key_text}{new_ref}  # {tag}{newline}"


def iter_workflow_files() -> list[Path]:
    """All ``*.yml``/``*.yaml`` under .github/workflows/ recursively."""
    return sorted(
        p
        for p in WORKFLOWS_DIR.rglob("*")
        if p.is_file() and p.suffix in {".yml", ".yaml"}
    )


def rewrite_all(pins: dict[str, str]) -> tuple[int, set[str]]:
    """Rewrite every workflow file. Returns (changed_count, unresolved)."""
    changed = 0
    unresolved: set[str] = set()
    for path in iter_workflow_files():
        original = path.read_text()
        new_lines = [
            rewrite_line(line, pins, unresolved)
            for line in original.splitlines(keepends=True)
        ]
        new_text = "".join(new_lines)
        if new_text != original:
            path.write_text(new_text)
            changed += 1
    return changed, unresolved


def check_pinned(pins_exist: bool) -> int:
    """Verify every external ``uses:`` is SHA-pinned.

    ``pins_exist`` is only used to surface a clearer error if the pin
    map is missing but --check is still requested.
    """
    bad: list[tuple[Path, int, str]] = []
    for path in iter_workflow_files():
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            match = USES_RE.match(line)
            if not match:
                continue
            ref = match.group("ref")
            if is_local_or_docker(ref):
                continue
            split = split_ref(ref)
            if split is None:
                continue
            _, tag = split
            if not SHA40_RE.match(tag):
                bad.append((path, lineno, line.rstrip()))
    if bad:
        print("Unpinned external 'uses:' references found:", file=sys.stderr)
        for path, lineno, line in bad:
            rel = path.relative_to(REPO_ROOT)
            print(f"  {rel}:{lineno}: {line}", file=sys.stderr)
        if not pins_exist:
            print(
                "Note: .github/action_pins.json was not found.",
                file=sys.stderr,
            )
        return 1
    return 0


def write_unresolved(unresolved: set[str]) -> None:
    """Write the follow-up list (or remove it if empty)."""
    if not unresolved:
        if NOT_PINNED_PATH.exists():
            NOT_PINNED_PATH.unlink()
        return
    lines = [
        "# Actions that could not be SHA-pinned automatically.",
        "# Resolve with: git ls-remote https://github.com/<owner>/<repo> "
        "refs/tags/<tag>^{}",
        "# Then add the pin to .github/action_pins.json and rerun "
        "scripts/pin_actions_to_sha.py.",
        "",
    ]
    lines.extend(sorted(unresolved))
    NOT_PINNED_PATH.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 if any external 'uses:' is not a 40-hex SHA.",
    )
    args = parser.parse_args(argv)
    pins_exist = PIN_MAP_PATH.exists()
    if args.check:
        return check_pinned(pins_exist)
    pins = load_pin_map(PIN_MAP_PATH)
    changed, unresolved = rewrite_all(pins)
    write_unresolved(unresolved)
    print(
        f"pin_actions_to_sha: rewrote {changed} workflow file(s); "
        f"{len(unresolved)} unresolved"
    )
    if unresolved:
        print("Unresolved actions (see .github/actions_not_pinned.txt):")
        for item in sorted(unresolved):
            print(f"  {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
