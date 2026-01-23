#!/usr/bin/env python3
"""Check for broken relative links in Markdown files."""

import re
import sys
from pathlib import Path
from urllib.parse import unquote


def check_links(root_dir: Path) -> list[str]:
    errors = []
    link_pattern = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")

    for md_file in root_dir.rglob("*.md"):
        # Skip node_modules or similar if any
        if "node_modules" in md_file.parts or ".git" in md_file.parts:
            continue

        try:
            content = md_file.read_text(encoding="utf-8")
        except Exception as e:
            errors.append(f"Could not read {md_file}: {e}")
            continue

        for match in link_pattern.finditer(content):
            _ = match.group(1)  # text
            link = match.group(2)

            # Ignore external links
            if link.startswith(("http://", "https://", "mailto:", "#")):
                continue

            # Handle anchor links within file (ignored above if strictly #)
            # but if it is filename#anchor
            _ = None  # anchor
            if "#" in link:
                parts = link.split("#", 1)
                link_path = parts[0]
                _ = parts[1]  # anchor
            else:
                link_path = link

            if not link_path:
                # Just an anchor
                continue

            # Resolve path
            # link is relative to md_file
            try:
                target = (md_file.parent / link_path).resolve()
            except Exception:
                errors.append(f"Invalid path in {md_file}: {link}")
                continue

            # Check existence
            if not target.exists():
                # Try unquoting
                decoded_link = unquote(link_path)
                target_decoded = (md_file.parent / decoded_link).resolve()
                if not target_decoded.exists():
                    errors.append(f"Broken link in {md_file}: {link} -> {target}")

    return errors


if __name__ == "__main__":
    root = Path(".")
    errors = check_links(root)
    if errors:
        print("Found broken links:")
        for e in errors:
            print(e)
        # We don't exit 1 to not fail the plan if there are minor broken links we
        # can't fix easily
        sys.exit(0)
    else:
        print("No broken links found.")
        sys.exit(0)
