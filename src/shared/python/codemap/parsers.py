"""Language parsers for the code-map indexer.

Provides lightweight AST-based symbol extraction using the standard library's
``ast`` module for Python (no tree-sitter dependency required for initial
rollout) and regex-based extractors for Rust, TypeScript, JavaScript, and
Markdown.

Each parser returns a list of :class:`~codemap.db.SymbolRow` objects.
"""

from __future__ import annotations

import ast
import hashlib
import logging
import re
from collections.abc import Callable
from pathlib import Path

from .db import SymbolRow

logger = logging.getLogger(__name__)

# ── Helpers ────────────────────────────────────────────────────────────────────


def _file_hash(path: Path) -> str:
    """Return a hex digest of the file contents (SHA-256, first 16 chars)."""
    try:
        content = path.read_bytes()
        return hashlib.sha256(content).hexdigest()[:16]
    except OSError:
        return ""


def _repo_rel(path: Path, repo_root: Path) -> str:
    """Return *path* relative to *repo_root* using POSIX separators."""
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return path.as_posix()


def _first_paragraph(docstring: str | None) -> str:
    """Extract the first non-empty paragraph from a docstring."""
    if not docstring:
        return ""
    lines = docstring.strip().splitlines()
    para: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped and para:
            break
        if stripped:
            para.append(stripped)
    return " ".join(para)


# ── Python parser ──────────────────────────────────────────────────────────────


def parse_python(
    path: Path, repo_root: Path, module_prefix: str = ""
) -> list[SymbolRow]:
    """Extract symbols from a Python source file via the ``ast`` module.

    Handles: module, class, function/method, async function, constants.
    """
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.debug("Cannot read %s: %s", path, exc)
        return []

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        logger.debug("SyntaxError in %s: %s", path, exc)
        return []

    file_hash = _file_hash(path)
    rel_path = _repo_rel(path, repo_root)

    # Build qualified module name from path
    stem = path.stem
    module_qn = f"{module_prefix}.{stem}" if module_prefix else stem

    rows: list[SymbolRow] = []

    # Module-level symbol
    module_doc = _first_paragraph(ast.get_docstring(tree))
    rows.append(
        SymbolRow(
            kind="module",
            qualified_name=module_qn,
            path=rel_path,
            line_start=1,
            line_end=len(source.splitlines()),
            signature=f"module {module_qn}",
            docstring=module_doc,
            blake3_hash=file_hash,
        )
    )

    def _visit(node: ast.AST, parent_qn: str) -> None:
        if isinstance(node, ast.ClassDef):
            qn = f"{parent_qn}.{node.name}"
            doc = _first_paragraph(ast.get_docstring(node))
            bases = ", ".join(ast.unparse(b) for b in node.bases) if node.bases else ""
            sig = f"class {node.name}({bases})" if bases else f"class {node.name}"
            rows.append(
                SymbolRow(
                    kind="class",
                    qualified_name=qn,
                    path=rel_path,
                    line_start=node.lineno,
                    line_end=node.end_lineno or node.lineno,
                    signature=sig,
                    docstring=doc,
                    blake3_hash=file_hash,
                )
            )
            for child in ast.iter_child_nodes(node):
                _visit(child, qn)

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            qn = f"{parent_qn}.{node.name}"
            doc = _first_paragraph(ast.get_docstring(node))
            is_method = parent_qn != module_qn
            kind = "method" if is_method else "function"
            try:
                sig = ast.unparse(node).splitlines()[0].rstrip(":")
            except Exception:
                sig = f"def {node.name}(...)"

            # Collect outbound call names
            calls: list[str] = []
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    if isinstance(child.func, ast.Name):
                        calls.append(child.func.id)
                    elif isinstance(child.func, ast.Attribute):
                        calls.append(child.func.attr)

            rows.append(
                SymbolRow(
                    kind=kind,
                    qualified_name=qn,
                    path=rel_path,
                    line_start=node.lineno,
                    line_end=node.end_lineno or node.lineno,
                    signature=sig,
                    docstring=doc,
                    calls_out=" ".join(sorted(set(calls))),
                    blake3_hash=file_hash,
                )
            )
            # Don't recurse into nested functions to keep index lean
        elif isinstance(node, ast.Assign):
            # Top-level constants (ALL_CAPS names only)
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    rows.append(
                        SymbolRow(
                            kind="constant",
                            qualified_name=f"{parent_qn}.{target.id}",
                            path=rel_path,
                            line_start=node.lineno,
                            line_end=node.end_lineno or node.lineno,
                            signature=f"{target.id} = ...",
                            blake3_hash=file_hash,
                        )
                    )

    for child in ast.iter_child_nodes(tree):
        _visit(child, module_qn)

    return rows


# ── Rust parser (regex-based) ──────────────────────────────────────────────────

_RUST_FN_RE = re.compile(
    r"^(?P<pub>pub(?:\s*\(crate\))?\s+)?(?P<async>async\s+)?fn\s+(?P<name>\w+)\s*(?P<sig>[^{]+)",
    re.MULTILINE,
)
_RUST_STRUCT_RE = re.compile(r"^pub\s+struct\s+(\w+)", re.MULTILINE)
_RUST_ENUM_RE = re.compile(r"^pub\s+enum\s+(\w+)", re.MULTILINE)
_RUST_TRAIT_RE = re.compile(r"^pub\s+trait\s+(\w+)", re.MULTILINE)


def parse_rust(
    path: Path, repo_root: Path, _module_prefix: str = ""
) -> list[SymbolRow]:
    """Extract public symbols from a Rust source file (regex-based)."""
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    file_hash = _file_hash(path)
    rel_path = _repo_rel(path, repo_root)
    lines = source.splitlines()
    rows: list[SymbolRow] = []

    def _line_no(match: re.Match) -> int:  # type: ignore[type-arg]
        return source[: match.start()].count("\n") + 1

    for m in _RUST_FN_RE.finditer(source):
        ln = _line_no(m)
        rows.append(
            SymbolRow(
                kind="function",
                qualified_name=f"{path.stem}::{m.group('name')}",
                path=rel_path,
                line_start=ln,
                line_end=ln,
                signature=f"fn {m.group('name')}{m.group('sig').strip()}",
                blake3_hash=file_hash,
            )
        )

    for pattern, kind in (
        (_RUST_STRUCT_RE, "class"),
        (_RUST_ENUM_RE, "class"),
        (_RUST_TRAIT_RE, "class"),
    ):
        for m in pattern.finditer(source):
            ln = _line_no(m)
            name = m.group(1)
            rows.append(
                SymbolRow(
                    kind=kind,
                    qualified_name=f"{path.stem}::{name}",
                    path=rel_path,
                    line_start=ln,
                    line_end=ln,
                    signature=m.group(0).strip(),
                    blake3_hash=file_hash,
                )
            )

    return rows


# ── TypeScript/JavaScript parser (regex-based) ─────────────────────────────────

_TS_FN_RE = re.compile(
    r"(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(",
    re.MULTILINE,
)
_TS_CLASS_RE = re.compile(r"(?:export\s+)?class\s+(\w+)", re.MULTILINE)
_TS_ARROW_RE = re.compile(
    r"(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\(",
    re.MULTILINE,
)


def parse_typescript(
    path: Path, repo_root: Path, _module_prefix: str = ""
) -> list[SymbolRow]:
    """Extract public symbols from a TypeScript/JavaScript file (regex-based)."""
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    file_hash = _file_hash(path)
    rel_path = _repo_rel(path, repo_root)
    rows: list[SymbolRow] = []

    def _line_no(match: re.Match) -> int:  # type: ignore[type-arg]
        return source[: match.start()].count("\n") + 1

    for m in _TS_FN_RE.finditer(source):
        rows.append(
            SymbolRow(
                kind="function",
                qualified_name=m.group(1),
                path=rel_path,
                line_start=_line_no(m),
                line_end=_line_no(m),
                signature=f"function {m.group(1)}(...)",
                blake3_hash=file_hash,
            )
        )
    for m in _TS_CLASS_RE.finditer(source):
        rows.append(
            SymbolRow(
                kind="class",
                qualified_name=m.group(1),
                path=rel_path,
                line_start=_line_no(m),
                line_end=_line_no(m),
                signature=f"class {m.group(1)}",
                blake3_hash=file_hash,
            )
        )
    for m in _TS_ARROW_RE.finditer(source):
        rows.append(
            SymbolRow(
                kind="function",
                qualified_name=m.group(1),
                path=rel_path,
                line_start=_line_no(m),
                line_end=_line_no(m),
                signature=f"const {m.group(1)} = (...) => ...",
                blake3_hash=file_hash,
            )
        )

    return rows


# ── Markdown parser ────────────────────────────────────────────────────────────

_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def parse_markdown(
    path: Path, repo_root: Path, _module_prefix: str = ""
) -> list[SymbolRow]:
    """Index headings from a Markdown file as 'section' symbols."""
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    file_hash = _file_hash(path)
    rel_path = _repo_rel(path, repo_root)
    rows: list[SymbolRow] = []

    for m in _MD_HEADING_RE.finditer(source):
        level = len(m.group(1))
        title = m.group(2).strip()
        ln = source[: m.start()].count("\n") + 1
        rows.append(
            SymbolRow(
                kind="section",
                qualified_name=f"{path.stem}::{title}",
                path=rel_path,
                line_start=ln,
                line_end=ln,
                signature=f"{'#' * level} {title}",
                blake3_hash=file_hash,
            )
        )

    return rows


# ── Dispatcher ─────────────────────────────────────────────────────────────────

ParserFn = Callable[[Path, Path, str], list[SymbolRow]]

PARSERS: dict[str, ParserFn] = {
    ".py": parse_python,
    ".rs": parse_rust,
    ".ts": parse_typescript,
    ".tsx": parse_typescript,
    ".js": parse_typescript,
    ".jsx": parse_typescript,
    ".md": parse_markdown,
    ".mdx": parse_markdown,
}


def parse_file(path: Path, repo_root: Path, module_prefix: str = "") -> list[SymbolRow]:
    """Dispatch to the appropriate parser based on file extension.

    Returns an empty list for unsupported extensions.
    """
    parser = PARSERS.get(path.suffix.lower())
    if parser is None:
        return []
    try:
        return parser(path, repo_root, module_prefix)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Parser error for %s: %s", path, exc)
        return []
