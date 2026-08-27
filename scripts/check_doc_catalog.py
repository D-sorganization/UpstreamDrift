#!/usr/bin/env python3
"""Validate the canonical documentation catalog and rendered docs link."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"
INDEX_PATH = DOCS_DIR / "index.md"
README_PATH = ROOT / "README.md"
PYPROJECT_PATH = ROOT / "pyproject.toml"
VALID_STABILITY = {"stable", "draft", "archived"}
CATALOG_ROW = re.compile(
    r"^\|\s*`(?P<directory>[^`]+/)`\s*\|\s*(?P<owner>@[^|]+?)\s*\|"
    r"\s*(?P<stability>stable|draft|archived)\s*\|\s*(?P<description>[^|]+?)\s*\|$"
)
DOCUMENTATION_URL_ROW = re.compile(r'^Documentation\s*=\s*"(?P<url>[^"]+)"\s*$')


@dataclass(frozen=True)
class CatalogEntry:
    """Metadata describing a top-level documentation directory."""

    directory: str
    owner: str
    stability: str
    description: str

    def validate(self) -> list[str]:
        """Return contract violations for this catalog entry."""
        errors: list[str] = []
        if self.stability not in VALID_STABILITY:
            errors.append(f"{self.directory}: invalid stability {self.stability}")
        if not self.owner.startswith("@"):
            errors.append(f"{self.directory}: owner must start with @")
        if len(self.description.split()) < 4:
            errors.append(f"{self.directory}: description must be a sentence")
        return errors


def _top_level_docs_dirs() -> set[str]:
    if not DOCS_DIR.exists():
        return set()
    return {path.name + "/" for path in DOCS_DIR.iterdir() if path.is_dir()}


def _parse_catalog() -> dict[str, CatalogEntry]:
    entries: dict[str, CatalogEntry] = {}
    if not INDEX_PATH.exists():
        return entries
    for line in INDEX_PATH.read_text(encoding="utf-8").splitlines():
        match = CATALOG_ROW.match(line)
        if match is None:
            continue
        entry = CatalogEntry(
            directory=match.group("directory"),
            owner=match.group("owner").strip(),
            stability=match.group("stability").strip(),
            description=match.group("description").strip(),
        )
        entries[entry.directory] = entry
    return entries


def _documentation_url() -> str:
    if not PYPROJECT_PATH.exists():
        return ""
    in_project_urls = False
    for line in PYPROJECT_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_project_urls = stripped == "[project.urls]"
            continue
        if not in_project_urls:
            continue
        match = DOCUMENTATION_URL_ROW.match(stripped)
        if match is not None:
            return match.group("url")
    return ""


DOCS_HUB_LINK = re.compile(r"\[[Dd]ocumentation [Hh]ub\]\((?P<target>[^)\s]+)\)")
# The Documentation URL in pyproject may be an https link to the hub file on the
# forge; the README normally links the same file by repository-relative path.
# Both are the same destination, so accept either.
CANONICAL_HUB_PATH = "docs/README.md"


def _readme_links_rendered_docs(docs_url: str) -> bool:
    """Return True when the README links the same docs hub pyproject declares.

    Accepts the declared URL verbatim, or the repository-relative path to the
    same file when the declared URL is a forge link to it. The contract is that
    the README and the package metadata agree on where documentation lives -
    not that a particular spelling is used.
    """
    if not README_PATH.exists() or not docs_url:
        return False
    readme = README_PATH.read_text(encoding="utf-8")
    accepted = {docs_url}
    if docs_url.startswith("https://") and docs_url.endswith(CANONICAL_HUB_PATH):
        accepted.add(CANONICAL_HUB_PATH)
    return any(
        match.group("target") in accepted for match in DOCS_HUB_LINK.finditer(readme)
    )


def _catalog_errors() -> list[str]:
    expected = _top_level_docs_dirs()
    entries = _parse_catalog()
    errors = [
        f"docs/index.md missing `{name}` catalog entry"
        for name in expected - entries.keys()
    ]
    errors.extend(
        f"docs/index.md has stale `{name}` catalog entry"
        for name in entries.keys() - expected
    )
    for entry in entries.values():
        errors.extend(entry.validate())
    return sorted(errors)


def main() -> int:
    errors = _catalog_errors()
    docs_url = _documentation_url()
    if not docs_url.startswith("https://"):
        errors.append(
            "pyproject.toml [project.urls] Documentation must be an HTTPS URL"
        )
    if not _readme_links_rendered_docs(docs_url):
        errors.append("README.md Documentation Hub must link to the rendered docs URL")

    if errors:
        sys.stderr.write("doc catalog check failed:\n- " + "\n- ".join(errors) + "\n")
        return 1

    sys.stdout.write("doc catalog check passed\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
