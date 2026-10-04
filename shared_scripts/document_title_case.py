#!/usr/bin/env python3
"""Enforce title capitalization in document sources and rendered artifacts.

The gate checks Markdown/Quarto headings and metadata, LaTeX structural titles,
Word title/subtitle/heading styles, PDF metadata and outline labels, and literal
Python chart titles. Ordinary prose is deliberately out of scope.
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from defusedxml import ElementTree

MINOR_WORDS = {
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
LOWERCASE_TERMS = {"cm", "kg", "km", "m", "mm", "ms", "nm", "rad", "s"}
LOWERCASE_PARTICLES = {"da", "de", "der", "di", "la", "le", "van", "von"}
WORD = re.compile(r"[^\W\d_][^\W_]*(?:['’][^\W_]+)?", re.UNICODE)
PROTECTED = re.compile(
    r"`[^`]+`|\$[^$]+\$|<[^>]+>|https?://\S+|\b[A-Z]\([^)]*\)|"
    r"@[\w:.-]+|\\[A-Za-z]+|\b[\w.-]+\.(?i:qmd|md|tex|py|html|css|yml|yaml|bib|pdf|docx)\b|"
    # Dotfile names and paths (``.gitattributes``, ``.git/hooks``) are literals.
    r"(?<![\w.])\.\w[\w./*-]*"
)
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
YAML_VALUE = re.compile(
    r"^(?P<prefix>\s*(?:title|subtitle|fig-cap|fig-subcap)\s*:\s*)"
    r"(?P<quote>['\"]?)(?P<value>.*?)(?P=quote)\s*$"
)
TRAILING_ATTRIBUTES = re.compile(r"\s*\{[^{}]*}\s*$")
LATEX_TITLE = re.compile(
    r"\\(?P<kind>title|subtitle|part|chapter|section|subsection|subsubsection|paragraph|subparagraph|caption)\*?"
    r"(?:\[[^]]*\])?\{(?P<value>[^{}]*)\}"
)
CHART_CALL = re.compile(
    r"\b(?:set_title|suptitle|(?:plt|pyplot)\.title)\s*\(\s*"
    r"(?:[fFrRuUbB]{0,2})?(?P<quote>['\"])(?P<value>.+?)(?P=quote)"
)
DOCX_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
DOCX_STYLE = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val"
DOCX_TITLE_STYLES = {"title", "subtitle", "heading", "caption"}
SUPPORTED_SUFFIXES = {".md", ".qmd", ".tex", ".docx", ".pdf", ".py"}
EXCLUDED_PARTS = {
    ".git",
    ".quarto",
    "_site",
    "build",
    "dist",
    "legacy",
    "node_modules",
    "site-packages",
    ".venv",
    "venv",
}


@dataclass(frozen=True)
class Finding:
    """One visible title that differs from the fleet convention."""

    path: Path
    line: int
    kind: str
    actual: str
    expected: str


def _protected_spans(value: str) -> list[tuple[int, int]]:
    return [(match.start(), match.end()) for match in PROTECTED.finditer(value)]


def _capitalize_word(word: str) -> str:
    if word.isupper() or (
        any(char.isupper() for char in word[1:]) and not word.istitle()
    ):
        return word
    return word[:1].upper() + word[1:]


def expected_title(value: str) -> str:
    """Return a title with significant words capitalized.

    Articles, coordinating conjunctions, and short prepositions remain lower
    case unless they begin/end a title, follow terminal punctuation or a colon,
    or form an edge of a hyphenated compound. Acronyms and technical literals
    are preserved.
    """
    value = re.sub(r"\bso(?=\(\d+\))", "SO", value)
    spans = _protected_spans(value)
    matches = [
        match
        for match in WORD.finditer(value)
        if not any(start <= match.start() < end for start, end in spans)
    ]
    if not matches:
        return value
    pieces: list[str] = []
    cursor = 0
    previous_end = 0
    for index, match in enumerate(matches):
        pieces.append(value[cursor : match.start()])
        word = match.group()
        separator = value[previous_end : match.start()]
        # A literal at the start or end of a title is that title's edge, so the
        # word next to it is not (Repository_Management#1832).
        boundary = (
            (index == 0 and not any(end <= match.start() for _, end in spans))
            or (
                index == len(matches) - 1
                and not any(start >= match.end() for start, _ in spans)
            )
            or bool(re.search(r"(?:[:!?—–]|-{2,}|[([{])\s*$", separator))
        )
        hyphens = "-‐‑"
        compound_edge = (match.start() > 0 and value[match.start() - 1] in hyphens) != (
            match.end() < len(value) and value[match.end()] in hyphens
        )
        lowered = word.lower()
        if word.isupper():
            replacement = word
        elif word.islower() and lowered in LOWERCASE_TERMS | LOWERCASE_PARTICLES:
            replacement = word
        elif lowered in MINOR_WORDS and not boundary and not compound_edge:
            replacement = lowered
        else:
            replacement = _capitalize_word(word)
        pieces.append(replacement)
        cursor = match.end()
        previous_end = match.end()
    pieces.append(value[cursor:])
    return "".join(pieces)


def _finding(path: Path, line: int, kind: str, value: str) -> Finding | None:
    clean = value.strip()
    if not clean or clean in {"---", "—"} or "{{" in clean:
        return None
    expected = expected_title(clean)
    return None if expected == clean else Finding(path, line, kind, clean, expected)


def _python_chart_findings(path: Path, text: str) -> list[Finding]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    findings: list[Finding] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        function = node.func
        valid = isinstance(function, ast.Attribute) and (
            function.attr in {"set_title", "suptitle"}
            or (
                function.attr == "title"
                and isinstance(function.value, ast.Name)
                and function.value.id in {"plt", "pyplot"}
            )
        )
        if not valid:
            continue
        for segment in ast.walk(node.args[0]):
            if isinstance(segment, ast.Constant) and isinstance(segment.value, str):
                finding = _finding(path, segment.lineno, "chart title", segment.value)
                if finding:
                    findings.append(finding)
    return findings


def findings_for_text(path: Path, text: str) -> list[Finding]:
    """Find titles in a Markdown, Quarto, LaTeX, or Python source."""
    if path.suffix.lower() == ".py":
        return _python_chart_findings(path, text)
    if path.suffix.lower() == ".tex":
        findings = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            content = line.split("%", 1)[0]
            for match in LATEX_TITLE.finditer(content):
                finding = _finding(
                    path, line_number, match.group("kind"), match.group("value")
                )
                if finding:
                    findings.append(finding)
        return findings

    findings: list[Finding] = []
    in_frontmatter = False
    in_fence = False
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if line_number == 1 and stripped == "---":
            in_frontmatter = True
            continue
        if in_frontmatter and stripped == "---":
            in_frontmatter = False
            continue
        if stripped.startswith(("```", "~~~")):
            in_fence = not in_fence
            continue
        match = YAML_VALUE.match(line) if in_frontmatter else None
        if match:
            kind = "subtitle" if "subtitle" in match.group("prefix") else "title"
            finding = _finding(path, line_number, kind, match.group("value"))
            if finding:
                findings.append(finding)
        elif not in_fence:
            heading = HEADING.match(line)
            if heading:
                value = TRAILING_ATTRIBUTES.sub("", heading.group(2)).strip()
                finding = _finding(path, line_number, "heading", value)
                if finding:
                    findings.append(finding)
        else:
            chart = CHART_CALL.search(line)
            if chart:
                finding = _finding(
                    path, line_number, "chart title", chart.group("value")
                )
                if finding:
                    findings.append(finding)
    return findings


def findings_for_docx(path: Path) -> list[Finding]:
    """Find DOCX paragraphs using title, subtitle, heading, or caption styles."""
    try:
        with ZipFile(path) as archive:
            root = ElementTree.fromstring(archive.read("word/document.xml"))
    except (BadZipFile, KeyError, ElementTree.ParseError):
        return []
    findings: list[Finding] = []
    for index, paragraph in enumerate(root.findall(".//w:p", DOCX_NS), start=1):
        style = paragraph.find("./w:pPr/w:pStyle", DOCX_NS)
        style_name = "" if style is None else style.attrib.get(DOCX_STYLE, "")
        normalized = re.sub(r"[\s_-]+", "", style_name).lower()
        if not any(normalized.startswith(prefix) for prefix in DOCX_TITLE_STYLES):
            continue
        value = "".join(
            node.text or "" for node in paragraph.findall(".//w:t", DOCX_NS)
        )
        finding = _finding(path, index, f"Word style {style_name}", value)
        if finding:
            findings.append(finding)
    return findings


def findings_for_pdf(path: Path) -> list[Finding]:
    """Find PDF document-title metadata and outline/bookmark labels when present."""
    try:
        from pypdf import PdfReader
    except ImportError:
        return []
    try:
        reader = PdfReader(path)
    except Exception:  # pypdf exposes several parser-specific exception classes
        return []
    findings: list[Finding] = []
    title = getattr(reader.metadata, "title", None) if reader.metadata else None
    if title:
        finding = _finding(path, 0, "PDF metadata title", str(title))
        if finding:
            findings.append(finding)

    def walk(items: Iterable[object]) -> None:
        for item in items:
            if isinstance(item, list):
                walk(item)
                continue
            value = getattr(item, "title", None)
            if value:
                finding = _finding(path, 0, "PDF outline title", str(value))
                if finding:
                    findings.append(finding)

    try:
        walk(reader.outline)
    except Exception:
        pass
    return findings


def findings_for_path(path: Path, display_path: Path | None = None) -> list[Finding]:
    shown = display_path or path
    suffix = path.suffix.lower()
    if suffix == ".docx":
        return [
            Finding(shown, f.line, f.kind, f.actual, f.expected)
            for f in findings_for_docx(path)
        ]
    if suffix == ".pdf":
        return [
            Finding(shown, f.line, f.kind, f.actual, f.expected)
            for f in findings_for_pdf(path)
        ]
    if suffix in SUPPORTED_SUFFIXES:
        text = path.read_text(encoding="utf-8", errors="replace")
        return findings_for_text(shown, text)
    return []


def tracked_document_paths(root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"], cwd=root, capture_output=True, text=True, check=False
    )
    names = result.stdout.splitlines() if result.returncode == 0 else []
    return sorted(
        root / name
        for name in names
        if Path(name).suffix.lower() in SUPPORTED_SUFFIXES
        and not any(part in EXCLUDED_PARTS for part in Path(name).parts)
        and (root / name).is_file()
    )


def apply_text_fixes(path: Path, findings: list[Finding]) -> None:
    if path.suffix.lower() not in {".md", ".qmd", ".tex", ".py"}:
        return
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines(keepends=True)
    for finding in findings:
        index = finding.line - 1
        lines[index] = lines[index].replace(finding.actual, finding.expected, 1)
    path.write_text("".join(lines), encoding="utf-8", newline="")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--fix", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    paths = [path if path.is_absolute() else root / path for path in args.paths]
    if not paths:
        paths = tracked_document_paths(root)
    findings: list[Finding] = []
    for path in paths:
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        shown = Path(path.relative_to(root)) if path.is_relative_to(root) else path
        current = findings_for_path(path, shown)
        findings.extend(current)
        if args.fix and current:
            apply_text_fixes(path, current)
    if args.fix:
        print(
            f"Corrected {len(findings)} title(s) across {len(paths)} document file(s)."
        )
        return 0
    for finding in findings:
        location = f":{finding.line}" if finding.line else ""
        print(
            f"{finding.path.as_posix()}{location}: {finding.kind}: "
            f"{finding.actual!r} -> {finding.expected!r}"
        )
    if findings:
        print(f"{len(findings)} title-capitalization violation(s).")
        return 1
    print(f"{len(paths)} document file(s) checked; titles follow the fleet convention.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
