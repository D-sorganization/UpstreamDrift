"""Python tree-sitter extractor."""

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


def _docstring(body_node, source: bytes) -> str:
    if body_node is None:
        return ""
    for c in body_node.children:
        if c.type == "expression_statement":
            inner = c.children[0] if c.children else None
            if inner is not None and inner.type == "string":
                raw = text_of(inner, source).strip()
                for q in ('"""', "'''", '"', "'"):
                    if raw.startswith(q) and raw.endswith(q):
                        raw = raw[len(q) : -len(q)]
                        break
                return raw.strip().splitlines()[0] if raw.strip() else ""
            return ""
        if c.type not in ("comment",):
            return ""
    return ""


def _signature(def_node, source: bytes) -> str:
    end = def_node.start_byte
    for c in def_node.children:
        if c.type == ":":
            end = c.end_byte
            break
    sig = source[def_node.start_byte : end].decode("utf-8", errors="replace")
    return sig.splitlines()[0].strip()


def _collect_calls(node, source: bytes, out: list[str]) -> None:
    if node.type == "call":
        func = first_child(node, "attribute") or first_child(node, "identifier")
        if func is None and node.children:
            func = node.children[0]
        if func is not None:
            out.append(text_of(func, source))
    for c in node.children:
        _collect_calls(c, source, out)


def _walk(node, source: bytes, prefix: str, out: list[ParsedSymbol]) -> None:
    for child in node.children:
        if child.type == "decorated_definition":
            inner = child.children[-1] if child.children else None
            if inner is not None:
                _walk_def(inner, source, prefix, out)
        elif child.type in ("function_definition", "class_definition"):
            _walk_def(child, source, prefix, out)
        elif child.type == "block":
            _walk(child, source, prefix, out)


def _walk_def(node, source: bytes, prefix: str, out: list[ParsedSymbol]) -> None:
    name_node = first_child(node, "identifier")
    if name_node is None:
        return
    name = text_of(name_node, source)
    qualified = f"{prefix}.{name}" if prefix else name
    body = first_child(node, "block")
    start, end = line_range(node)
    if node.type == "function_definition":
        calls: list[str] = []
        if body is not None:
            _collect_calls(body, source, calls)
        out.append(
            ParsedSymbol(
                kind="method" if "." in qualified else "function",
                name=name,
                qualified=qualified,
                sig=_signature(node, source),
                docstring=_docstring(body, source),
                start_line=start,
                end_line=end,
                calls_out=sorted(set(calls))[:64],
            )
        )
    elif node.type == "class_definition":
        out.append(
            ParsedSymbol(
                kind="class",
                name=name,
                qualified=qualified,
                sig=_signature(node, source),
                docstring=_docstring(body, source),
                start_line=start,
                end_line=end,
            )
        )
        if body is not None:
            _walk(body, source, qualified, out)


def _imports(root, source: bytes) -> list[str]:
    out: list[str] = []
    for c in root.children:
        if c.type == "import_statement":
            for sub in c.children:
                if sub.type == "dotted_name":
                    out.append(text_of(sub, source))
        elif c.type == "import_from_statement":
            mod = first_child(c, "dotted_name") or first_child(c, "relative_import")
            if mod is not None:
                out.append(text_of(mod, source))
    return out


def extract(path: str, source: str | bytes) -> ParseResult:
    parser = get_parser("python")
    if parser is None:
        return ParseResult("python", [], [])
    src = to_bytes(source)
    tree = parser.parse(src)
    symbols: list[ParsedSymbol] = []
    _walk(tree.root_node, src, "", symbols)
    return ParseResult("python", _imports(tree.root_node, src), symbols)


__all__ = ["extract"]
