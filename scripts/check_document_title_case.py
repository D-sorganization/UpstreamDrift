#!/usr/bin/env python3
"""Enforce title case in tracked Markdown, Quarto, LaTeX, Word, and PDF documents."""

from __future__ import annotations

import argparse
from contextlib import suppress
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile

MINOR = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "but",
    "by",
    "for",
    "if",
    "in",
    "nor",
    "of",
    "on",
    "or",
    "per",
    "so",
    "the",
    "to",
    "up",
    "via",
    "vs",
    "yet",
}
TERMS = {"cm", "kg", "km", "m", "mm", "ms", "nm", "rad", "s"}
PARTICLES = {"da", "de", "der", "di", "la", "le", "van", "von"}
WORD = re.compile(r"[^\W\d_][^\W_]*(?:['’][^\W_]+)?", re.UNICODE)
PROTECTED = re.compile(
    r"`[^`]+`|\$[^$]+\$|https?://\S+|@[\w:.-]+|\\[A-Za-z]+|\b[A-Z]\([^)]*\)|"
    r"\b[\w.-]+\.(?i:md|qmd|tex|docx|pdf|py|exe|app|appimage)\b"
)
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
YAML_TITLE = re.compile(
    r"^\s*(?P<kind>title|subtitle|fig-cap|fig-subcap)\s*:\s*(?P<quote>['\"]?)(?P<value>.*?)(?P=quote)\s*$"
)
LATEX_TITLE = re.compile(
    r"\\(?P<kind>title|subtitle|part|chapter|section|subsection|subsubsection|paragraph|subparagraph|caption)\*?(?:\[[^]]*\])?\{(?P<value>[^{}]*)\}"
)
ATTRIBUTES = re.compile(r"\s*\{[^{}]*}\s*$")
SUFFIXES = {".md", ".qmd", ".tex", ".docx", ".pdf"}
EXCLUDED = {
    ".git",
    ".quarto",
    "_site",
    "archive",
    "build",
    "dist",
    "legacy",
    "node_modules",
    ".venv",
    "venv",
}
DOCX_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
DOCX_VAL = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val"


@dataclass(frozen=True)
class Finding:
    path: Path
    line: int
    kind: str
    actual: str
    expected: str


def expected_title(value: str) -> str:
    spans = [(item.start(), item.end()) for item in PROTECTED.finditer(value)]
    words = [
        item
        for item in WORD.finditer(value)
        if not any(a <= item.start() < b for a, b in spans)
    ]
    pieces: list[str] = []
    cursor = previous_end = 0
    for index, item in enumerate(words):
        pieces.append(value[cursor : item.start()])
        word = item.group()
        lowered = word.lower()
        separator = value[previous_end : item.start()]
        boundary = index in {0, len(words) - 1} or bool(
            re.search(r"(?:[:!?—–]|-{2,}|[([{])\s*$", separator)
        )
        hyphens = "-‐‑"
        compound = (item.start() > 0 and value[item.start() - 1] in hyphens) != (
            item.end() < len(value) and value[item.end()] in hyphens
        )
        if word.isupper() or (word.islower() and lowered in TERMS | PARTICLES):
            replacement = word
        elif lowered in MINOR and not boundary and not compound:
            replacement = lowered
        elif any(char.isupper() for char in word[1:]) and not word.istitle():
            replacement = word
        else:
            replacement = word[:1].upper() + word[1:]
        pieces.append(replacement)
        cursor = previous_end = item.end()
    pieces.append(value[cursor:])
    return "".join(pieces)


def _finding(path: Path, line: int, kind: str, value: str) -> Finding | None:
    clean = value.strip()
    if not clean or clean in {"---", "—"} or "{{" in clean:
        return None
    expected = expected_title(clean)
    return None if expected == clean else Finding(path, line, kind, clean, expected)


def findings_for_text(path: Path, text: str) -> list[Finding]:
    findings: list[Finding] = []
    if path.suffix.lower() == ".tex":
        for number, line in enumerate(text.splitlines(), 1):
            for match in LATEX_TITLE.finditer(line.split("%", 1)[0]):
                finding = _finding(
                    path, number, match.group("kind"), match.group("value")
                )
                if finding:
                    findings.append(finding)
        return findings
    frontmatter = fence = False
    for number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if number == 1 and stripped == "---":
            frontmatter = True
            continue
        if frontmatter and stripped == "---":
            frontmatter = False
            continue
        if stripped.startswith(("```", "~~~")):
            fence = not fence
            continue
        yaml_match = YAML_TITLE.match(line) if frontmatter else None
        if yaml_match:
            finding = _finding(
                path, number, yaml_match.group("kind"), yaml_match.group("value")
            )
        elif not fence and (heading := HEADING.match(line)):
            finding = _finding(
                path, number, "heading", ATTRIBUTES.sub("", heading.group(2))
            )
        else:
            finding = None
        if finding:
            findings.append(finding)
    return findings


def findings_for_docx(path: Path, shown: Path) -> list[Finding]:
    try:
        with ZipFile(path) as archive:
            root = ElementTree.fromstring(archive.read("word/document.xml"))
    except (BadZipFile, KeyError, ElementTree.ParseError):
        return []
    findings = []
    for number, paragraph in enumerate(root.findall(".//w:p", DOCX_NS), 1):
        style = paragraph.find("./w:pPr/w:pStyle", DOCX_NS)
        style_name = "" if style is None else style.attrib.get(DOCX_VAL, "")
        normalized = re.sub(r"[\s_-]", "", style_name).lower()
        if not normalized.startswith(("title", "subtitle", "heading", "caption")):
            continue
        value = "".join(
            node.text or "" for node in paragraph.findall(".//w:t", DOCX_NS)
        )
        finding = _finding(shown, number, f"Word style {style_name}", value)
        if finding:
            findings.append(finding)
    return findings


def findings_for_pdf(path: Path, shown: Path) -> list[Finding]:
    try:
        from pypdf import PdfReader
        from pypdf.errors import PyPdfError
    except ImportError:
        return []
    try:
        reader = PdfReader(path)
    except (OSError, PyPdfError):
        return []
    findings = []
    title = getattr(reader.metadata, "title", None) if reader.metadata else None
    if title and (finding := _finding(shown, 0, "PDF metadata title", str(title))):
        findings.append(finding)

    def walk(items: list[object]) -> None:
        for item in items:
            if isinstance(item, list):
                walk(item)
            elif (value := getattr(item, "title", None)) and (
                finding := _finding(shown, 0, "PDF outline title", str(value))
            ):
                findings.append(finding)

    with suppress(AttributeError, IndexError, TypeError, PyPdfError):
        walk(reader.outline)
    return findings


def tracked_paths(root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"], cwd=root, capture_output=True, text=True, check=False
    )
    return [
        root / name
        for name in result.stdout.splitlines()
        if Path(name).suffix.lower() in SUFFIXES
        and not any(part in EXCLUDED for part in Path(name).parts)
        and (root / name).is_file()
    ]


def _added_lines_from_diff(diff: str) -> set[int]:
    """Return new-file line numbers from a zero-context unified diff."""
    lines: set[int] = set()
    pattern = r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@"
    for match in re.finditer(pattern, diff, re.MULTILINE):
        start = int(match.group(1))
        count = int(match.group(2) or "1")
        lines.update(range(start, start + count))
    return lines


def changed_lines_for_path(
    root: Path, path: Path, *, ref: str | None = None, staged: bool = False
) -> set[int]:
    """Return title lines newly introduced in the selected Git diff."""
    command = ["git", "diff", "--unified=0", "--diff-filter=ACMR"]
    if staged:
        command.append("--cached")
    elif ref:
        command.append(f"{ref}..HEAD")
    command.extend(["--", path.relative_to(root).as_posix()])
    result = subprocess.run(
        command, cwd=root, capture_output=True, text=True, check=False
    )
    return _added_lines_from_diff(result.stdout)


def findings_for_path(path: Path, root: Path) -> list[Finding]:
    shown = path.relative_to(root)
    if path.suffix.lower() == ".docx":
        return findings_for_docx(path, shown)
    if path.suffix.lower() == ".pdf":
        return findings_for_pdf(path, shown)
    return findings_for_text(shown, path.read_text(encoding="utf-8", errors="replace"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--changed-from", help="check document files changed from this Git ref"
    )
    parser.add_argument(
        "--staged", action="store_true", help="check only staged title lines"
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    paths = [item if item.is_absolute() else root / item for item in args.paths]
    if args.changed_from:
        result = subprocess.run(
            [
                "git",
                "diff",
                "--name-only",
                "--diff-filter=ACMR",
                f"{args.changed_from}..HEAD",
            ],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        paths = [
            root / name
            for name in result.stdout.splitlines()
            if Path(name).suffix.lower() in SUFFIXES
        ]
    elif not paths:
        paths = tracked_paths(root)
    findings = []
    for path in paths:
        if not path.is_file():
            continue
        current = findings_for_path(path, root)
        if args.changed_from or args.staged:
            changed = changed_lines_for_path(
                root, path, ref=args.changed_from, staged=args.staged
            )
            current = [
                finding
                for finding in current
                if finding.line == 0
                or path.suffix.lower() == ".docx"
                or finding.line in changed
            ]
        findings.extend(current)
    for finding in findings:
        location = f":{finding.line}" if finding.line else ""
        print(
            f"{finding.path.as_posix()}{location}: {finding.kind}: {finding.actual!r} -> {finding.expected!r}"
        )
    print(f"{len(paths)} document(s) checked; {len(findings)} violation(s).")
    return int(bool(findings))


if __name__ == "__main__":
    sys.exit(main())
