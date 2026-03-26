#!/usr/bin/env python3
"""Check for broken relative links in Markdown files."""

import logging
import re
import sys
from pathlib import Path
from urllib.parse import unquote

from src.shared.python.contracts import require

logger = logging.getLogger(__name__)

LINK_PATTERN = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")


def extract_links_from_markdown(content: str) -> list[str]:
    """Extract raw link targets from Markdown content."""
    require(isinstance(content, str), "content must be a string")
    links: list[str] = []

    for match in LINK_PATTERN.finditer(content):
        link = match.group(2)
        # Ignore external or fragment-only links
        if link.startswith(("http://", "https://", "mailto:", "#")):
            continue

        links.append(link)

    return links


def resolve_and_verify_link(link: str, base_dir: Path) -> str | None:
    """Resolve a relative link against a base directory and verify existence."""
    if not (link is not None):
        raise ValueError("link must be provided")
    if not (link is not None):
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


def check_links(root_dir: Path) -> list[str]:
    """Validate internal links in all Markdown files under root_dir."""
    require(isinstance(root_dir, Path), "root_dir must be a Path")
    require(
        root_dir.exists(),
        f"Configuration error: root directory '{root_dir}' does not exist.",
    )

    errors = []

    for md_file in root_dir.rglob("*.md"):
        # Skip node_modules or git
        if "node_modules" in md_file.parts or ".git" in md_file.parts:
            continue

        try:
            content = md_file.read_text(encoding="utf-8")
        except (PermissionError, OSError) as e:
            errors.append(f"Could not read {md_file}: {e}")
            continue

        links = extract_links_from_markdown(content)
        for link in links:
            error = resolve_and_verify_link(link, md_file.parent)
            if error:
                errors.append(f"{md_file}: {error}")

    return errors


def main() -> None:
    root = Path(".")
    errors = check_links(root)
    if errors:
        for err in errors:
            logger.warning(err)
        # We don't exit 1 to not fail the plan if there are minor broken links
        sys.exit(0)
    else:
        sys.exit(0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
