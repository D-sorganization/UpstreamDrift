#!/usr/bin/env python3
"""Git merge driver that unions SPEC.md change-log rows instead of conflicting.

Belt and braces for Repository_Management#1520. PR-keyed rows are the actual
fix — two pull requests can no longer choose the same key, so there is nothing
to renumber. This driver removes the *remaining* textual conflict: two rows
inserted at the same offset. Rows are independent facts about independent pull
requests, so a merge never has to choose between them; it keeps both.

Scope and honest limits
-----------------------
* Custom merge drivers are a **client-side** feature. They run during
  ``git merge``, ``git rebase`` and ``git cherry-pick`` in a clone that has the
  driver configured. They do **not** run on GitHub's servers, so a squash-merge
  performed by the GitHub UI or ``gh pr merge`` still uses the default driver.
  That is fine: with PR-keyed rows the default driver only conflicts when two
  rows land on the same line, and this driver exists so the local rebase that
  fixes that is a no-op.
* Git config is per-clone and cannot be committed, so ``.gitattributes`` alone
  is not enough. Run ``python scripts/install_spec_merge_driver.py`` once per
  clone (the workspace hook installer does it for you).

Invocation contract (``merge.spec-rows.driver``)::

    python scripts/spec_rows_merge_driver.py %O %A %B %P

``%O`` common ancestor, ``%A`` our version (**and the file the result must be
written to**), ``%B`` their version, ``%P`` the real pathname. Exit 0 for a
clean merge, non-zero to report a conflict.

Everything outside the change-log table is delegated verbatim to
``git merge-file``, so this driver never invents a resolution for real content.
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parent.parent


def _load_spec_changelog() -> ModuleType:
    """Import ``shared_scripts/spec_changelog.py`` by path.

    The propagated copy of this driver may live in a repository where
    ``shared_scripts`` is not an importable package, which is the same reason
    ``fleet_hooks.py`` loads ``development_log.py`` by path.
    """
    for candidate in (
        ROOT / "shared_scripts" / "spec_changelog.py",
        ROOT / "scripts" / "spec_changelog.py",
        Path(__file__).with_name("spec_changelog.py"),
    ):
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location(
                "fleet_spec_changelog", candidate
            )
            if spec is None or spec.loader is None:  # pragma: no cover - defensive
                continue
            module = importlib.util.module_from_spec(spec)
            # Register before exec: the module defines dataclasses, and
            # dataclasses.field resolution looks the defining module up in
            # sys.modules.
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module
    raise SystemExit(
        "spec_rows_merge_driver: cannot find spec_changelog.py; "
        "expected shared_scripts/spec_changelog.py"
    )


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def _merge_file(base: str, ours: str, theirs: str, label: str) -> tuple[str, int]:
    """Run ``git merge-file`` on three strings; return (result, exit status)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        paths = {
            "base": tmpdir / "base",
            "ours": tmpdir / "ours",
            "theirs": tmpdir / "theirs",
        }
        _write(paths["base"], base)
        _write(paths["ours"], ours)
        _write(paths["theirs"], theirs)
        result = subprocess.run(
            [
                "git",
                "merge-file",
                "-p",
                "-L",
                f"{label} (ours)",
                "-L",
                f"{label} (base)",
                "-L",
                f"{label} (theirs)",
                str(paths["ours"]),
                str(paths["base"]),
                str(paths["theirs"]),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    return result.stdout, result.returncode


def merge(
    base_text: str, ours_text: str, theirs_text: str, label: str
) -> tuple[str, int]:
    """Merge three SPEC.md versions, unioning change-log rows.

    Returns the merged text and an exit status (0 = clean).
    """
    sc = _load_spec_changelog()

    try:
        base_log = sc.parse_changelog(base_text)
        our_log = sc.parse_changelog(ours_text)
        their_log = sc.parse_changelog(theirs_text)
    except sc.SpecChangelogError:
        # No parsable table on some side: fall back entirely to git's own merge
        # rather than guessing.
        return _merge_file(base_text, ours_text, theirs_text, label)

    merged_rows = sc.union_rows(base_log.rows, our_log.rows, their_log.rows)

    # Neutralise the table on all three sides so `git merge-file` merges only
    # the surrounding prose, then splice the unioned rows back in.
    placeholder = [sc.Row(date="0000-00-00", key="#0", summary="ROWS-PLACEHOLDER")]
    stub_base = sc.replace_rows(base_text, base_log, placeholder)
    stub_ours = sc.replace_rows(ours_text, our_log, placeholder)
    stub_theirs = sc.replace_rows(theirs_text, their_log, placeholder)

    merged_stub, status = _merge_file(stub_base, stub_ours, stub_theirs, label)
    if not merged_stub:  # pragma: no cover - git merge-file always emits output
        return _merge_file(base_text, ours_text, theirs_text, label)

    try:
        stub_log = sc.parse_changelog(merged_stub)
    except sc.SpecChangelogError:  # pragma: no cover - defensive
        return _merge_file(base_text, ours_text, theirs_text, label)

    return sc.replace_rows(merged_stub, stub_log, merged_rows), status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", help="%O - common ancestor")
    parser.add_argument("ours", help="%A - our version; the result is written here")
    parser.add_argument("theirs", help="%B - their version")
    parser.add_argument("path", nargs="?", default="SPEC.md", help="%P - pathname")
    args = parser.parse_args(argv)

    ours_path = Path(args.ours)
    merged, status = merge(
        _read(Path(args.base)),
        _read(ours_path),
        _read(Path(args.theirs)),
        args.path,
    )
    _write(ours_path, merged)
    if status != 0:
        print(
            f"spec-rows: change-log rows merged cleanly, but {args.path} still "
            "conflicts outside the change-log table. Resolve the markers.",
            file=sys.stderr,
        )
    return 1 if status != 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
