#!/usr/bin/env python3
"""Check for broken relative links in Markdown files.

Run over the documentation tree by default, or over explicit roots::

    python -m src.tools.check_markdown_links
    python -m src.tools.check_markdown_links src docs/help

``main()`` exits non-zero when unresolvable links remain, so the tool can gate
CI (issue #8851 -- it previously exited 0 unconditionally and therefore could
never fail a build).

Coverage notes:

- ``docs/help/`` is scanned like any other documentation directory. It is named
  in :data:`DEFAULT_DOC_ROOTS` so its coverage is explicit rather than
  incidental.
- The default scope is ``docs/`` plus top-level Markdown files. Source-tree
  READMEs still carry broken links and are not fixed by this change; scan them
  by passing ``src`` explicitly.
- Generated, vendored, and virtual-environment trees are skipped, because their
  contents are not this repository's to fix.
- Link targets that are obviously documentation placeholders (``<name>``,
  ``{name}``) are not treated as paths.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from urllib.parse import unquote

from src.shared.python.contracts import require

logger = logging.getLogger(__name__)

LINK_PATTERN = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")

#: Placeholder syntax used inside documentation snippets, never a real path.
PLACEHOLDER_PATTERN = re.compile(r"[<>{}]")

#: Directory names never walked: generated output, vendored trees, and caches.
SKIPPED_DIR_NAMES = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "htmlcov",
        "node_modules",
        "site-packages",
        "vendor",
        "venv",
    }
)

#: Link prefixes whose targets legitimately may be absent from a checkout.
#: ``vendor/`` is a git submodule that is not initialised in every environment.
EXEMPT_LINK_PREFIXES: tuple[str, ...] = ("vendor/",)

#: Documentation roots scanned when no explicit paths are given. ``docs/help``
#: is listed separately so narrowing the scan cannot silently drop help-page
#: coverage, and so a reader can see that it is covered on purpose.
DEFAULT_DOC_ROOTS: tuple[str, ...] = ("docs", "docs/help")


def default_roots(base: Path | None = None) -> list[Path]:
    """Return the paths scanned when the caller names none.

    That is the documentation tree (including ``docs/help/``) plus every
    top-level Markdown file, discovered rather than hard-coded so a newly added
    root document is covered without editing this module. Source-tree READMEs
    are deliberately out of the default scope: pass them explicitly.
    """
    root = Path(".") if base is None else base
    roots = [root / item for item in DEFAULT_DOC_ROOTS]
    roots.extend(sorted(root.glob("*.md")))
    return [path for path in roots if path.exists()]


def extract_links_from_markdown(content: str) -> list[str]:
    """Extract raw link targets from Markdown content."""
    require(isinstance(content, str), "content must be a string")
    links: list[str] = []

    for match in LINK_PATTERN.finditer(content):
        link = match.group(2)
        # Ignore external or fragment-only links
        if link.startswith(("http://", "https://", "mailto:", "#")):
            continue

        # Ignore documentation placeholders such as "<pyproject URL>".
        if PLACEHOLDER_PATTERN.search(link):
            continue

        links.append(link)

    return links


def is_exempt_link(link: str) -> bool:
    """Return True when ``link`` may legitimately be absent from a checkout."""
    require(isinstance(link, str), "link must be a string")
    return link.startswith(EXEMPT_LINK_PREFIXES)


def resolve_and_verify_link(link: str, base_dir: Path) -> str | None:
    """Resolve a relative link against a base directory and verify existence."""
    if link is None:
        raise ValueError("link must be provided")
    require(isinstance(link, str), "link must be a string")
    require(isinstance(base_dir, Path), "base_dir must be a Path")

    # Strip anchor if present
    link_path = link.split("#", 1)[0] if "#" in link else link
    if not link_path:
        return None  # Only anchor remaining

    try:
        target = (base_dir / link_path).resolve()
        if target.exists():
            return None

        # Try unquoting
        decoded_link = unquote(link_path)
        target_decoded = (base_dir / decoded_link).resolve()
        if target_decoded.exists():
            return None

        return f"Broken link: {link} -> {target}"
    except (RuntimeError, ValueError, OSError) as e:
        return f"Invalid path configuration for link '{link}': {e}"


def check_markdown_file(md_file: Path) -> list[str]:
    """Return broken-link messages for a single Markdown file."""
    require(isinstance(md_file, Path), "md_file must be a Path")

    try:
        content = md_file.read_text(encoding="utf-8")
    except (PermissionError, OSError) as e:
        return [f"Could not read {md_file}: {e}"]

    errors: list[str] = []
    for link in extract_links_from_markdown(content):
        if is_exempt_link(link):
            continue
        error = resolve_and_verify_link(link, md_file.parent)
        if error:
            errors.append(f"{md_file}: {error}")
    return errors


def check_links(root_dir: Path) -> list[str]:
    """Validate internal links in all Markdown files under root_dir."""
    require(isinstance(root_dir, Path), "root_dir must be a Path")
    require(
        root_dir.exists(),
        f"Configuration error: root directory '{root_dir}' does not exist.",
    )

    if root_dir.is_file():
        return check_markdown_file(root_dir)

    errors: list[str] = []
    for md_file in sorted(root_dir.rglob("*.md")):
        if SKIPPED_DIR_NAMES.intersection(md_file.parts):
            continue
        errors.extend(check_markdown_file(md_file))

    return errors


def check_paths(paths: list[Path]) -> list[str]:
    """Validate links under every path, reporting each Markdown file once."""
    require(isinstance(paths, list), "paths must be a list")

    errors: list[str] = []
    seen: set[str] = set()
    for path in paths:
        if not path.exists():
            errors.append(f"Missing scan root: {path}")
            continue
        for error in check_links(path):
            if error not in seen:
                seen.add(error)
                errors.append(error)
    return errors


def main(argv: list[str] | None = None) -> None:
    """Scan the requested roots and exit non-zero when links are broken."""
    parser = argparse.ArgumentParser(description="Check relative Markdown links.")
    parser.add_argument(
        "paths",
        nargs="*",
        help=(
            "files or directories to scan (default: "
            f"{' '.join(DEFAULT_DOC_ROOTS)} plus top-level Markdown files)"
        ),
    )
    args = parser.parse_args(argv)

    roots = (
        [Path(item) for item in args.paths if Path(item).exists()]
        if args.paths
        else default_roots()
    )
    if not roots:
        # Nothing to scan is not a failure: an empty tree has no broken links.
        sys.exit(0)

    errors = check_paths(roots)
    if errors:
        for err in errors:
            logger.warning(err)
        logger.warning("%d broken Markdown link(s) found", len(errors))
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
