#!/usr/bin/env python3
"""Per-PR change fragments (RM-5 / Repository_Management#1894).

Every pull request used to edit the same three files: the ``SPEC.md`` change
log, ``DEVELOPMENT_LOG.md`` and ``HANDOFF.md``. Every merge to ``main`` then
made every other open pull request conflict. A fragment removes the hotspot:
each pull request adds its *own* file, ``changes/<issue>-<slug>.md``, and a
post-merge step folds it into the shared files once the pull request number is
known.

Fragment format::

    ---
    issue: 1894
    summary: "One line for the SPEC.md change-log row"
    dl_state: "in_review"          # optional, development-log state
    next_step: "Merge the PR."     # required when dl_state is live
    title: "Entry title"           # optional; new DL entries only
    owner: "claude"                # optional; new DL entries only
    branch: "feat/x"               # optional; new DL entries only
    paths: "`shared_scripts/x.py`" # optional; new DL entries only
    ---

    Optional markdown body: the pull request's handoff notes.

Commands::

    changes_fragment.py new --issue N --summary "..." [--dl-state S --next-step T]
    changes_fragment.py validate [paths...]
    changes_fragment.py collate --pr N [--sha SHA] [--date YYYY-MM-DD] [paths...]

``collate`` appends one ``| date | #PR | summary |`` row to the SPEC.md change
log, creates or updates the ``DL-#<issue>`` development-log entry in place, and
deletes the fragment. It is idempotent: re-running it on the same fragment and
pull request leaves both files unchanged.

The module is portable (standard library only) and is copied fleet-wide next to
``fleet_hooks.py``, ``development_log.py`` and ``spec_changelog.py``, whose
parsers it reuses rather than reimplementing.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path

try:
    from shared_scripts import changes_fragment_schema as schema
except ImportError:  # pragma: no cover - non-package fleet copy
    import importlib.util

    _path = Path(__file__).with_name("changes_fragment_schema.py")
    _cached = sys.modules.get("_fleet_changes_fragment_schema")
    if _cached is None:
        _spec = importlib.util.spec_from_file_location(
            "_fleet_changes_fragment_schema", _path
        )
        if _spec is None or _spec.loader is None:
            raise
        _cached = importlib.util.module_from_spec(_spec)
        sys.modules[_spec.name] = _cached
        _spec.loader.exec_module(_cached)
    schema = _cached

try:
    from shared_scripts import changes_fragment_collate as _collate_mod
except ImportError:  # pragma: no cover - non-package fleet copy
    _collate_mod = schema.sibling("changes_fragment_collate")

log = logging.getLogger("changes_fragment")

# Public API (stable; imported by fleet_hooks.py and pre_pr.py).
CHANGES_DIR = schema.CHANGES_DIR
OPTIONAL_KEYS = schema.OPTIONAL_KEYS
Fragment = schema.Fragment
FragmentError = schema.FragmentError
is_fragment_path = schema.is_fragment_path
validate_fragment_file = schema.validate_fragment_file
load_fragment = schema.load_fragment
find_fragments = schema.find_fragments
slugify = schema.slugify
render_fragment = schema.render_fragment
apply_spec_row = _collate_mod.apply_spec_row
apply_devlog_entry = _collate_mod.apply_devlog_entry
collate = _collate_mod.collate

__all__ = [
    "Fragment",
    "FragmentError",
    "apply_devlog_entry",
    "apply_spec_row",
    "collate",
    "find_fragments",
    "is_fragment_path",
    "load_fragment",
    "main",
    "new_fragment",
    "render_fragment",
    "slugify",
    "validate_fragment_file",
]


def new_fragment(
    repo_root: Path,
    *,
    issue: int,
    summary: str,
    slug: str | None = None,
    handoff: str = "",
    **optional: str | None,
) -> Path:
    """Write ``changes/<issue>-<slug>.md`` and return its path.

    Preconditions: the rendered fragment validates; the file does not exist.
    Raises :class:`FragmentError` or :class:`FileExistsError` otherwise, in
    which case nothing is written.
    """
    unknown = set(optional) - set(OPTIONAL_KEYS)
    if unknown:
        raise FragmentError(f"unknown fragment fields: {sorted(unknown)}")
    stem = slugify(slug if slug is not None else summary)
    name = f"{issue}-{stem}.md" if stem else f"{issue}.md"
    text = render_fragment(issue, summary, handoff=handoff, **optional)
    _fragment, errors = schema.parse(text, name)
    if errors:
        raise FragmentError("; ".join(errors))
    path = repo_root / CHANGES_DIR / name
    if path.exists():
        raise FileExistsError(f"{path} already exists; edit it instead")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def _head_sha(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    sub = parser.add_subparsers(dest="command", required=True)

    new = sub.add_parser("new", help="write a new fragment")
    new.add_argument("--repo-root", type=Path, default=argparse.SUPPRESS)
    new.add_argument("--issue", type=int, required=True)
    new.add_argument("--summary", required=True)
    new.add_argument("--slug")
    new.add_argument("--dl-state", dest="dl_state")
    new.add_argument("--next-step", dest="next_step")
    for key in ("title", "owner", "branch", "paths"):
        new.add_argument(f"--{key}", dest=key)
    new.add_argument("--handoff", default="", help="markdown handoff notes")

    validate = sub.add_parser("validate", help="validate fragments")
    validate.add_argument("--repo-root", type=Path, default=argparse.SUPPRESS)
    validate.add_argument("paths", nargs="*", type=Path)

    coll = sub.add_parser("collate", help="fold fragments into the shared files")
    coll.add_argument("--repo-root", type=Path, default=argparse.SUPPRESS)
    coll.add_argument("--pr", type=int, required=True)
    coll.add_argument("--sha", help="merge commit (default: git rev-parse HEAD)")
    coll.add_argument("--date", type=date.fromisoformat, help="default: today UTC")
    coll.add_argument("paths", nargs="*", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _build_parser().parse_args(argv)
    root: Path = args.repo_root

    if args.command == "new":
        optional = {key: getattr(args, key) for key in OPTIONAL_KEYS}
        try:
            path = new_fragment(
                root,
                issue=args.issue,
                summary=args.summary,
                slug=args.slug,
                handoff=args.handoff,
                **optional,
            )
        except (FragmentError, FileExistsError) as exc:
            log.error("ERROR: %s", exc)
            return 1
        log.info("wrote %s", path)
        return 0

    paths: list[Path] = args.paths or find_fragments(root)
    if args.command == "validate":
        status = 0
        for path in paths:
            for finding in validate_fragment_file(path):
                log.error("ERROR: %s: %s", path, finding)
                status = 1
        if status == 0:
            log.info("%d fragment(s) valid.", len(paths))
        return status

    sha = args.sha or _head_sha(root)
    today = args.date or datetime.now(UTC).date()
    try:
        deleted = collate(root, paths, pr=args.pr, today=today, sha=sha)
    except (FragmentError, ValueError) as exc:
        log.error("ERROR: %s", exc)
        return 1
    log.info("collated %d fragment(s) for #%d", len(deleted), args.pr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
