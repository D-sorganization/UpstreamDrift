"""JavaScript / TypeScript / TSX tree-sitter extractors."""

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


def _walk(node, source: bytes, prefix: str, out: list[ParsedSymbol]) -> None:
    for child in node.children:
        t = child.type
        if t == "function_declaration":
            name_node = first_child(child, "identifier")
            if name_node is None:
                continue
            name = text_of(name_node, source)
            qualified = f"{prefix}.{name}" if prefix else name
            start, end = line_range(child)
            out.append(
                ParsedSymbol(
                    kind="function",
                    name=name,
                    qualified=qualified,
                    sig=text_of(child, source).splitlines()[0],
                    docstring="",
                    start_line=start,
                    end_line=end,
                )
            )
        elif t in ("class_declaration", "abstract_class_declaration"):
            name_node = first_child(child, "type_identifier") or first_child(
                child, "identifier"
            )
            if name_node is None:
                continue
            name = text_of(name_node, source)
            qualified = f"{prefix}.{name}" if prefix else name
            start, end = line_range(child)
            out.append(
                ParsedSymbol(
                    kind="class",
                    name=name,
                    qualified=qualified,
                    sig=f"class {name}",
                    docstring="",
                    start_line=start,
                    end_line=end,
                )
            )
            body = first_child(child, "class_body")
            if body is not None:
                for member in body.children:
                    if member.type == "method_definition":
                        mname_node = first_child(
                            member, "property_identifier"
                        ) or first_child(member, "identifier")
                        if mname_node is None:
                            continue
                        mname = text_of(mname_node, source)
                        mstart, mend = line_range(member)
                        out.append(
                            ParsedSymbol(
                                kind="method",
                                name=mname,
                                qualified=f"{qualified}.{mname}",
                                sig=text_of(member, source).splitlines()[0],
                                docstring="",
                                start_line=mstart,
                                end_line=mend,
                            )
                        )
        elif t in ("export_statement", "ambient_declaration"):
            _walk(child, source, prefix, out)
        elif t in ("lexical_declaration", "variable_declaration"):
            for sub in child.children:
                if sub.type == "variable_declarator":
                    name_node = first_child(sub, "identifier")
                    val = sub.children[-1] if sub.children else None
                    if name_node is None or val is None:
                        continue
                    if val.type in (
                        "arrow_function",
                        "function_expression",
                        "function",
                    ):
                        name = text_of(name_node, source)
                        start, end = line_range(child)
                        out.append(
                            ParsedSymbol(
                                kind="function",
                                name=name,
                                qualified=f"{prefix}.{name}" if prefix else name,
                                sig=text_of(child, source).splitlines()[0],
                                docstring="",
                                start_line=start,
                                end_line=end,
                            )
                        )


def _imports(root, source: bytes) -> list[str]:
    out: list[str] = []
    for c in root.children:
        if c.type == "import_statement":
            src_node = first_child(c, "string")
            if src_node is not None:
                out.append(text_of(src_node, source).strip("'\""))
    return out


def extract_javascript(path: str, source: str | bytes) -> ParseResult:
    parser = get_parser("javascript")
    if parser is None:
        return ParseResult("javascript", [], [])
    src = to_bytes(source)
    tree = parser.parse(src)
    symbols: list[ParsedSymbol] = []
    _walk(tree.root_node, src, "", symbols)
    return ParseResult("javascript", _imports(tree.root_node, src), symbols)


def extract_typescript(path: str, source: str | bytes) -> ParseResult:
    lang_id = "tsx" if str(path).endswith(".tsx") else "typescript"
    parser = get_parser(lang_id)
    if parser is None:
        return ParseResult(lang_id, [], [])
    src = to_bytes(source)
    tree = parser.parse(src)
    symbols: list[ParsedSymbol] = []
    _walk(tree.root_node, src, "", symbols)
    return ParseResult(lang_id, _imports(tree.root_node, src), symbols)


__all__ = ["extract_javascript", "extract_typescript"]
