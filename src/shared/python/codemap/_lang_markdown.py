"""Markdown tree-sitter extractor — emits headings as symbols."""

from __future__ import annotations

from ._ts_common import (
    ParsedSymbol,
    ParseResult,
    first_child,
    get_parser,
    line_range,
    text_of,
    to_bytes,
)


def _walk(node, source: bytes, out: list[ParsedSymbol]) -> None:
    for c in node.children:
        if c.type in ("atx_heading", "setext_heading"):
            text_node = first_child(c, "inline") or first_child(c, "heading_content")
            if text_node is None:
                title = text_of(c, source).strip("# \n")
            else:
                title = text_of(text_node, source).strip()
            if not title:
                continue
            start, end = line_range(c)
            out.append(
                ParsedSymbol(
                    kind="heading",
                    name=title[:80],
                    qualified=title[:200],
                    sig=title[:200],
                    docstring="",
                    start_line=start,
                    end_line=end,
                )
            )
        _walk(c, source, out)


def extract(path: str, source: str | bytes) -> ParseResult:
    parser = get_parser("markdown")
    if parser is None:
        return ParseResult("markdown", [], [])
    src = to_bytes(source)
    tree = parser.parse(src)
    symbols: list[ParsedSymbol] = []
    _walk(tree.root_node, src, symbols)
    return ParseResult("markdown", [], symbols)


__all__ = ["extract"]
