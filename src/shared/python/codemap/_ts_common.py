"""Shared tree-sitter helpers used by every per-language parser module.

Centralises the lazy-import dance, the parser cache, and small node helpers
so each language module stays small and focused on its grammar mapping.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ParsedSymbol:
    """Lightweight symbol record produced by parsers."""

    kind: str
    name: str
    qualified: str
    sig: str
    docstring: str
    start_line: int  # 1-indexed inclusive
    end_line: int  # 1-indexed inclusive
    calls_out: list[str] = field(default_factory=list)


@dataclass
class ParseResult:
    """All symbols + imports extracted from a single file."""

    language: str
    imports: list[str]
    symbols: list[ParsedSymbol]


_LOCK = threading.Lock()
_LANG_CACHE: dict[str, object] = {}
_PARSER_CACHE: dict[str, object] = {}
_MISSING_WARNED: set[str] = set()

LANG_MODULES: dict[str, tuple[str, str]] = {
    "python": ("tree_sitter_python", "language"),
    "javascript": ("tree_sitter_javascript", "language"),
    "typescript": ("tree_sitter_typescript", "language_typescript"),
    "tsx": ("tree_sitter_typescript", "language_tsx"),
    "rust": ("tree_sitter_rust", "language"),
    "markdown": ("tree_sitter_markdown", "language"),
}


def get_parser(lang_id: str) -> Any:
    """Return a cached tree-sitter Parser for ``lang_id`` or None."""
    with _LOCK:
        if lang_id in _PARSER_CACHE:
            return _PARSER_CACHE[lang_id]
        spec = LANG_MODULES.get(lang_id)
        if spec is None:
            return None
        module_name, attr = spec
        try:
            mod = __import__(module_name)
            from tree_sitter import Language, Parser
        except Exception as exc:  # pragma: no cover
            if module_name not in _MISSING_WARNED:
                _MISSING_WARNED.add(module_name)
                logger.warning(
                    "codemap: language '%s' unavailable (%s); skipping",
                    lang_id,
                    exc,
                )
            _PARSER_CACHE[lang_id] = None
            return None
        try:
            raw = getattr(mod, attr)()
            language = Language(raw)
            parser = Parser(language)
        except Exception as exc:  # pragma: no cover
            if module_name not in _MISSING_WARNED:
                _MISSING_WARNED.add(module_name)
                logger.warning(
                    "codemap: failed to initialise '%s' parser: %s",
                    lang_id,
                    exc,
                )
            _PARSER_CACHE[lang_id] = None
            return None
        _LANG_CACHE[lang_id] = language
        _PARSER_CACHE[lang_id] = parser
        return parser


def to_bytes(source: str | bytes) -> bytes:
    if isinstance(source, bytes):
        return source
    return source.encode("utf-8", errors="replace")


def text_of(node: Any, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def first_child(node: Any, type_name: str) -> Any:
    for c in node.children:
        if c.type == type_name:
            return c
    return None


def line_range(node: Any) -> tuple[int, int]:
    return node.start_point[0] + 1, node.end_point[0] + 1


__all__ = [
    "LANG_MODULES",
    "ParseResult",
    "ParsedSymbol",
    "first_child",
    "get_parser",
    "line_range",
    "text_of",
    "to_bytes",
]
