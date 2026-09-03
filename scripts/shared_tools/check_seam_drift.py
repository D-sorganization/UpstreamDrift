"""CI gate for the UpstreamDrift <-> Tools shared-code seam (UD #9406).

Reads ``docs/shared_tools/seam_rulings.v1.json`` (one ruling per top-level
entry of the shared tree) and checks the working tree against it:

* ``tools-canonical`` + ``status: cleaned`` — no file of that package may
  still exist under ``src/shared/python`` except an approved import shim
  (a single ``__init__.py`` / module file containing ``SEAM_SHIM_MARKER``).
* ``split`` + ``status: cleaned`` — no file under the UD package may share a
  relative path with the vendored Tools package (UD-only modules may stay).
* ``ud-canonical`` — the package exists in the vendored Tools tree, so it must
  carry a ``tools_ledger_row`` reference (Tools #4915 divergence ledger). A
  ud-canonical package whose vendor copy is gone is reported as ``retired``.
* Every top-level entry of the vendored tree must have a ruling. A new Tools
  module without a ruling fails the gate (ratchet on coverage).
* ``pending-cleanup`` rulings are reported, never failed, unless ``--strict``.

Exit 0 when clean. Exit 1 on any violation. Exit 2 when the vendor tree is
missing (run ``git submodule update --init vendor/ud-tools``).

Program: D-sorganization/Repository_Management#1505 · RM #1507 (main-green).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

RULINGS = ("tools-canonical", "ud-canonical", "split", "deferred")
STATUSES = ("pending-cleanup", "cleaned", "n/a")
SEAM_SHIM_MARKER = "seam-shim: resolves to the pinned Tools package"

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RULINGS = Path("docs/shared_tools/seam_rulings.v1.json")
DEFAULT_UD_ROOT = Path("src/shared/python")
DEFAULT_TOOLS_ROOT = Path("vendor/ud-tools/src/shared/python")
_IGNORED = frozenset({"__pycache__", ".pytest_cache", ".mypy_cache"})
_IGNORED_TOP_LEVEL = frozenset({"__pycache__", "README_PACKAGE.md"})


@dataclass(frozen=True)
class Violation:
    package: str
    message: str

    def __str__(self) -> str:
        return f"{self.package}: {self.message}"


def load_rulings(path: Path) -> dict[str, dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rulings = data.get("rulings")
    if not isinstance(rulings, dict):
        raise ValueError(f"{path}: 'rulings' must be a mapping")
    for name, entry in rulings.items():
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: ruling {name!r} must be a mapping")
        if entry.get("ruling") not in RULINGS:
            raise ValueError(
                f"{path}: ruling {name!r} has invalid 'ruling' {entry.get('ruling')!r}"
            )
        if entry.get("status") not in STATUSES:
            raise ValueError(
                f"{path}: ruling {name!r} has invalid 'status' {entry.get('status')!r}"
            )
        if not str(entry.get("rationale", "")).strip():
            raise ValueError(f"{path}: ruling {name!r} needs a one-line 'rationale'")
    return rulings


def _files_under(root: Path) -> set[str]:
    if root.is_file():
        return {root.name}
    if not root.is_dir():
        return set()
    out: set[str] = set()
    for path in root.rglob("*"):
        if path.is_file() and not any(
            p in _IGNORED for p in path.relative_to(root).parts
        ):
            if path.suffix not in {".pyc", ".pyo"}:
                out.add(path.relative_to(root).as_posix())
    return out


def _is_shim(path: Path) -> bool:
    try:
        return SEAM_SHIM_MARKER in path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False


def _top_level_entries(root: Path) -> set[str]:
    if not root.is_dir():
        return set()
    return {
        entry.name
        for entry in root.iterdir()
        if entry.name not in _IGNORED_TOP_LEVEL and not entry.name.startswith(".")
    }


@dataclass(frozen=True)
class _PackageState:
    name: str
    entry: dict[str, object]
    ud_pkg: Path
    ud_files: set[str]
    tools_files: set[str]
    overlap: list[str]

    @property
    def status(self) -> str:
        return str(self.entry["status"])


def _check_tools_canonical(
    state: _PackageState, ud_root: Path, strict: bool
) -> tuple[list[Violation], list[str]]:
    name = state.name
    if not state.ud_files:
        if state.status == "pending-cleanup":
            return [], [f"{name}: no UD copy left; status can become 'cleaned'"]
        return [], []
    if state.status == "cleaned":
        pkg = state.ud_pkg
        leftovers = [
            f
            for f in sorted(state.ud_files)
            if not (
                (pkg.is_file() and _is_shim(pkg))
                or (f == "__init__.py" and _is_shim(pkg / f))
            )
        ]
        if leftovers:
            return [
                Violation(
                    name,
                    "ruled tools-canonical and marked cleaned but "
                    f"{len(leftovers)} file(s) remain under {ud_root}: "
                    + ", ".join(leftovers[:5]),
                )
            ], []
        return [], []
    if strict:
        return [
            Violation(name, f"pending-cleanup with {len(state.ud_files)} UD file(s)")
        ], []
    return [], [
        f"{name}: pending-cleanup, {len(state.ud_files)} UD file(s), "
        f"{len(state.overlap)} overlapping"
    ]


def _check_split(
    state: _PackageState, strict: bool
) -> tuple[list[Violation], list[str]]:
    name, overlap = state.name, state.overlap
    if state.status == "cleaned":
        if overlap and not (
            overlap == ["__init__.py"] and _is_shim(state.ud_pkg / "__init__.py")
        ):
            return [
                Violation(
                    name,
                    "ruled split and marked cleaned but "
                    f"{len(overlap)} path(s) still overlap the Tools copy: "
                    + ", ".join(overlap[:5]),
                )
            ], []
        return [], []
    if strict and overlap:
        return [Violation(name, f"pending-cleanup with {len(overlap)} overlap(s)")], []
    return [], [f"{name}: split pending, {len(overlap)} overlapping"]


def _check_ud_canonical(state: _PackageState) -> tuple[list[Violation], list[str]]:
    if state.tools_files and not str(state.entry.get("tools_ledger_row", "")).strip():
        return [
            Violation(
                state.name,
                "ruled ud-canonical, the pinned Tools tree still carries a "
                "copy, and no 'tools_ledger_row' points at the Tools "
                "divergence ledger (Tools #4915)",
            )
        ], []
    if not state.tools_files:
        return [], [f"{state.name}: ud-canonical, Tools copy retired"]
    return [], []


def _check_deferred(state: _PackageState) -> tuple[list[Violation], list[str]]:
    violations = []
    if not str(state.entry.get("deferred_reason", "")).strip():
        violations.append(
            Violation(state.name, "deferred rulings need 'deferred_reason'")
        )
    return violations, [f"{state.name}: deferred ({len(state.overlap)} overlapping)"]


def check(
    repo_root: Path,
    *,
    rulings_path: Path = DEFAULT_RULINGS,
    ud_root: Path = DEFAULT_UD_ROOT,
    tools_root: Path = DEFAULT_TOOLS_ROOT,
    strict: bool = False,
) -> tuple[list[Violation], list[str]]:
    """Return (violations, notes)."""
    rulings = load_rulings(repo_root / rulings_path)
    ud_abs = repo_root / ud_root
    tools_abs = repo_root / tools_root
    if not tools_abs.is_dir():
        raise FileNotFoundError(tools_abs)

    violations: list[Violation] = []
    notes: list[str] = []
    for name in sorted(_top_level_entries(tools_abs)):
        if name not in rulings:
            violations.append(
                Violation(name, "present in the pinned Tools tree but has no ruling")
            )

    for name, entry in sorted(rulings.items()):
        ud_files = _files_under(ud_abs / name)
        tools_files = _files_under(tools_abs / name)
        state = _PackageState(
            name,
            entry,
            ud_abs / name,
            ud_files,
            tools_files,
            sorted(ud_files & tools_files),
        )
        ruling = str(entry["ruling"])
        if ruling == "tools-canonical":
            found, noted = _check_tools_canonical(state, ud_root, strict)
        elif ruling == "split":
            found, noted = _check_split(state, strict)
        elif ruling == "ud-canonical":
            found, noted = _check_ud_canonical(state)
        else:
            found, noted = _check_deferred(state)
        violations.extend(found)
        notes.extend(noted)
    return violations, notes


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    parser.add_argument("--rulings", type=Path, default=DEFAULT_RULINGS)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat pending-cleanup rulings with UD files as failures.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    try:
        violations, notes = check(
            repo_root, rulings_path=args.rulings, strict=args.strict
        )
    except FileNotFoundError as exc:
        print(
            f"seam drift gate: vendor tree missing at {exc}; run "
            "`git submodule update --init vendor/ud-tools`"
        )
        return 2
    if not args.quiet:
        for note in notes:
            print(f"note: {note}")
    if violations:
        print(f"seam drift gate FAILED ({len(violations)} violation(s)):")
        for violation in violations:
            print(f"  - {violation}")
        return 1
    print(f"seam drift gate passed ({len(notes)} note(s)).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
