"""Split Markdown, Quarto and LaTeX documents into heading-anchored chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass

import yaml

_FENCE_RE = re.compile(r"^\s{0,3}(```|~~~)")
_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_ATTR_RE = re.compile(r"\s*\{[^}]*\}\s*$")
_TEX_HEADING_RE = re.compile(
    r"^\s*\\(part|chapter|section|subsection|subsubsection|paragraph)\*?\s*\{([^{}]*)\}\s*(.*)$"
)
_TEX_LEVELS = {
    "part": 1,
    "chapter": 2,
    "section": 3,
    "subsection": 4,
    "subsubsection": 5,
    "paragraph": 6,
}
_TEX_SUFFIXES = frozenset({".tex", ".latex"})


@dataclass(frozen=True)
class Chunk:
    """A contiguous piece of one document under one heading path."""

    title: str
    anchor: str
    text: str
    status: str | None


def chunk_document(text: str, *, suffix: str, max_chars: int) -> list[Chunk]:
    """Chunk ``text`` by heading; long sections split at paragraph breaks.

    ``anchor`` is the slug path of the enclosing headings (``intro/timing``);
    ``status`` is the front-matter ``status:`` value, if any.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    status, body = _split_front_matter(text)
    if suffix.lower() in _TEX_SUFFIXES:
        sections = _tex_sections(body)
    else:
        sections = _markdown_sections(body)
    chunks: list[Chunk] = []
    for title, anchor, section_text in sections:
        for piece in _split_long(section_text.strip(), max_chars):
            chunks.append(Chunk(title=title, anchor=anchor, text=piece, status=status))
    return chunks


def slugify(title: str) -> str:
    """Lowercase, non-alphanumerics collapsed to ``-``."""
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug or "section"


def _split_front_matter(text: str) -> tuple[str | None, str]:
    if not text.startswith("---\n"):
        return None, text
    end = text.find("\n---", 4)
    if end < 0:
        return None, text
    after = text.find("\n", end + 4)
    body = "" if after < 0 else text[after + 1 :]
    try:
        meta = yaml.safe_load(text[4:end])
    except yaml.YAMLError:
        return None, body
    status = meta.get("status") if isinstance(meta, dict) else None
    return (str(status).strip().lower() if status else None), body


class _Outline:
    """Tracks the heading stack and accumulates (title, anchor, text) sections."""

    def __init__(self) -> None:
        self.sections: list[tuple[str, str, str]] = []
        self._stack: list[tuple[int, str]] = []
        self._title = ""
        self._lines: list[str] = []

    def heading(self, level: int, title: str) -> None:
        self._flush()
        while self._stack and self._stack[-1][0] >= level:
            self._stack.pop()
        self._stack.append((level, slugify(title)))
        self._title = title

    def line(self, line: str) -> None:
        self._lines.append(line)

    def finish(self) -> list[tuple[str, str, str]]:
        self._flush()
        return self.sections

    def _flush(self) -> None:
        text = "\n".join(self._lines).strip()
        if text:
            anchor = "/".join(slug for _, slug in self._stack)
            self.sections.append((self._title, anchor, text))
        self._lines = []


def _markdown_sections(body: str) -> list[tuple[str, str, str]]:
    outline = _Outline()
    fence: str | None = None
    for line in body.split("\n"):
        fence_match = _FENCE_RE.match(line)
        if fence_match:
            marker = fence_match.group(1)
            fence = None if fence == marker else (fence or marker)
            outline.line(line)
            continue
        heading = None if fence else _MD_HEADING_RE.match(line)
        if heading:
            title = _ATTR_RE.sub("", heading.group(2)).strip()
            outline.heading(len(heading.group(1)), title)
        else:
            outline.line(line)
    return outline.finish()


def _tex_sections(body: str) -> list[tuple[str, str, str]]:
    begin = body.find("\\begin{document}")
    if begin >= 0:
        body = body[begin + len("\\begin{document}") :]
    body = body.replace("\\end{document}", "")
    outline = _Outline()
    for line in body.split("\n"):
        heading = _TEX_HEADING_RE.match(line)
        if heading:
            outline.heading(_TEX_LEVELS[heading.group(1)], heading.group(2).strip())
            if heading.group(3).strip():
                outline.line(heading.group(3))
        elif not line.lstrip().startswith("%"):
            outline.line(line)
    return outline.finish()


def _split_long(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text] if text else []
    pieces: list[str] = []
    current = ""
    for paragraph in _paragraphs(text, max_chars):
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
        else:
            pieces.append(current)
            current = paragraph
    if current:
        pieces.append(current)
    return pieces


def _paragraphs(text: str, max_chars: int) -> list[str]:
    """Blank-line paragraphs, hard-wrapping any paragraph longer than ``max_chars``."""
    out: list[str] = []
    for paragraph in (p.strip() for p in re.split(r"\n\s*\n", text)):
        while len(paragraph) > max_chars:
            cut = paragraph.rfind(" ", 0, max_chars + 1)
            cut = cut if cut > 0 else max_chars
            out.append(paragraph[:cut].rstrip())
            paragraph = paragraph[cut:].lstrip()
        if paragraph:
            out.append(paragraph)
    return out
