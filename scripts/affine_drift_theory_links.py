"""AffineDrift theory URLs for UpstreamDrift user-facing help (Refs #15016)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = REPO_ROOT / "scripts" / "config" / "affine_drift_theory_links.v1.json"
HELP_ROOT = REPO_ROOT / "docs" / "help"
AFFINEDRIFT_HOST = "affinedrift.com"
BACKGROUND_HEADING = "## Background"


@dataclass(frozen=True)
class TheoryLink:
    """One AffineDrift article bound to a tile help page."""

    title: str
    url: str


def load_theory_registry(path: Path = REGISTRY_PATH) -> dict[str, TheoryLink]:
    """Return help basenames mapped to AffineDrift theory destinations."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    pages = payload.get("pages", {})
    if not isinstance(pages, dict):
        msg = "theory registry pages must be an object"
        raise ValueError(msg)
    registry: dict[str, TheoryLink] = {}
    for stem, entry in pages.items():
        if not isinstance(entry, dict):
            msg = f"theory entry for {stem!r} must be an object"
            raise ValueError(msg)
        title = entry.get("title")
        url = entry.get("url")
        if not isinstance(title, str) or not title.strip():
            msg = f"theory entry for {stem!r} needs a title"
            raise ValueError(msg)
        if not isinstance(url, str) or AFFINEDRIFT_HOST not in url:
            msg = f"theory entry for {stem!r} needs an affinedrift.com url"
            raise ValueError(msg)
        registry[stem] = TheoryLink(title=title.strip(), url=url.strip())
    return registry


def expected_background_markdown(link: TheoryLink) -> str:
    """Canonical Background section body for a help page."""
    return (
        f"{BACKGROUND_HEADING}\n\n"
        f"Theory and notation for this surface are on "
        f"[AffineDrift]({link.url}): *{link.title}*.\n"
    )


def help_markdown_paths(root: Path = HELP_ROOT) -> list[Path]:
    """Every committed tile help page under ``docs/help``."""
    return sorted(root.glob("*.md"))


def background_section_present(text: str, link: TheoryLink) -> bool:
    """Whether the page cites the expected AffineDrift theory destination."""
    if BACKGROUND_HEADING not in text:
        return False
    return link.url in text and AFFINEDRIFT_HOST in text


_BACKGROUND_PATTERN = re.compile(
    rf"^{re.escape(BACKGROUND_HEADING)}\n\n.*?\n",
    re.MULTILINE | re.DOTALL,
)


def upsert_background_section(text: str, link: TheoryLink) -> str:
    """Insert or replace the Background section on a help page."""
    block = expected_background_markdown(link)
    if _BACKGROUND_PATTERN.search(text):
        return _BACKGROUND_PATTERN.sub(block, text, count=1)
    if BACKGROUND_HEADING in text:
        return text
    for anchor in ("## Limitations", "## Related", "## See also"):
        index = text.find(anchor)
        if index != -1:
            return text[:index] + block + "\n" + text[index:]
    return text.rstrip() + "\n\n" + block
