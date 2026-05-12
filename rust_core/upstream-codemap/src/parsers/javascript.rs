//! JavaScript / TypeScript / TSX extractor — mirrors `_lang_js.py`.

use tree_sitter::{Node, Parser};

use super::{first_child_of, line_range, text_of, ParseResult, ParsedSymbol};

fn js_parser() -> Parser {
    let mut p = Parser::new();
    let lang: tree_sitter::Language = tree_sitter_javascript::LANGUAGE.into();
    p.set_language(&lang).expect("javascript grammar");
    p
}

fn ts_parser(tsx: bool) -> Parser {
    let mut p = Parser::new();
    let lang: tree_sitter::Language = if tsx {
        tree_sitter_typescript::LANGUAGE_TSX.into()
    } else {
        tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()
    };
    p.set_language(&lang).expect("typescript grammar");
    p
}

fn walk(node: Node, src: &[u8], prefix: &str, out: &mut Vec<ParsedSymbol>) {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "function_declaration" => {
                let Some(name_node) = first_child_of(&child, "identifier") else {
                    continue;
                };
                let name = text_of(&name_node, src).to_string();
                let qualified = if prefix.is_empty() {
                    name.clone()
                } else {
                    format!("{prefix}.{name}")
                };
                let (start, end) = line_range(&child);
                let sig = text_of(&child, src)
                    .lines()
                    .next()
                    .unwrap_or("")
                    .to_string();
                out.push(ParsedSymbol {
                    kind: "function".into(),
                    name,
                    qualified,
                    sig,
                    docstring: String::new(),
                    start_line: start,
                    end_line: end,
                    calls_out: Vec::new(),
                });
            }
            "class_declaration" | "abstract_class_declaration" => {
                let name_node = first_child_of(&child, "type_identifier")
                    .or_else(|| first_child_of(&child, "identifier"));
                let Some(name_node) = name_node else { continue };
                let name = text_of(&name_node, src).to_string();
                let qualified = if prefix.is_empty() {
                    name.clone()
                } else {
                    format!("{prefix}.{name}")
                };
                let (start, end) = line_range(&child);
                out.push(ParsedSymbol {
                    kind: "class".into(),
                    name: name.clone(),
                    qualified: qualified.clone(),
                    sig: format!("class {name}"),
                    docstring: String::new(),
                    start_line: start,
                    end_line: end,
                    calls_out: Vec::new(),
                });
                if let Some(body) = first_child_of(&child, "class_body") {
                    let mut cc = body.walk();
                    for member in body.children(&mut cc) {
                        if member.kind() == "method_definition" {
                            let mname_node = first_child_of(&member, "property_identifier")
                                .or_else(|| first_child_of(&member, "identifier"));
                            let Some(mname_node) = mname_node else {
                                continue;
                            };
                            let mname = text_of(&mname_node, src).to_string();
                            let (ms, me) = line_range(&member);
                            let sig = text_of(&member, src)
                                .lines()
                                .next()
                                .unwrap_or("")
                                .to_string();
                            out.push(ParsedSymbol {
                                kind: "method".into(),
                                name: mname.clone(),
                                qualified: format!("{qualified}.{mname}"),
                                sig,
                                docstring: String::new(),
                                start_line: ms,
                                end_line: me,
                                calls_out: Vec::new(),
                            });
                        }
                    }
                }
            }
            "export_statement" | "ambient_declaration" => {
                walk(child, src, prefix, out);
            }
            "lexical_declaration" | "variable_declaration" => {
                let mut cc = child.walk();
                for sub in child.children(&mut cc) {
                    if sub.kind() == "variable_declarator" {
                        let name_node = first_child_of(&sub, "identifier");
                        let mut sc = sub.walk();
                        let val = sub.children(&mut sc).last();
                        let (Some(name_node), Some(val)) = (name_node, val) else {
                            continue;
                        };
                        if matches!(
                            val.kind(),
                            "arrow_function" | "function_expression" | "function"
                        ) {
                            let name = text_of(&name_node, src).to_string();
                            let (start, end) = line_range(&child);
                            let sig = text_of(&child, src)
                                .lines()
                                .next()
                                .unwrap_or("")
                                .to_string();
                            out.push(ParsedSymbol {
                                kind: "function".into(),
                                name: name.clone(),
                                qualified: if prefix.is_empty() {
                                    name
                                } else {
                                    format!("{prefix}.{name}")
                                },
                                sig,
                                docstring: String::new(),
                                start_line: start,
                                end_line: end,
                                calls_out: Vec::new(),
                            });
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

fn imports(root: Node, src: &[u8]) -> Vec<String> {
    let mut out = Vec::new();
    let mut cursor = root.walk();
    for c in root.children(&mut cursor) {
        if c.kind() == "import_statement" {
            if let Some(src_node) = first_child_of(&c, "string") {
                let raw = text_of(&src_node, src);
                out.push(raw.trim_matches(|c| c == '"' || c == '\'').to_string());
            }
        }
    }
    out
}

pub fn extract_javascript(_path: &str, source: &[u8]) -> ParseResult {
    let mut p = js_parser();
    let Some(tree) = p.parse(source, None) else {
        return ParseResult {
            language: "javascript".into(),
            imports: vec![],
            symbols: vec![],
        };
    };
    let root = tree.root_node();
    let mut symbols = Vec::new();
    walk(root, source, "", &mut symbols);
    ParseResult {
        language: "javascript".into(),
        imports: imports(root, source),
        symbols,
    }
}

pub fn extract_typescript(path: &str, source: &[u8]) -> ParseResult {
    let tsx = path.ends_with(".tsx");
    let lang_id = if tsx { "tsx" } else { "typescript" };
    let mut p = ts_parser(tsx);
    let Some(tree) = p.parse(source, None) else {
        return ParseResult {
            language: lang_id.into(),
            imports: vec![],
            symbols: vec![],
        };
    };
    let root = tree.root_node();
    let mut symbols = Vec::new();
    walk(root, source, "", &mut symbols);
    ParseResult {
        language: lang_id.into(),
        imports: imports(root, source),
        symbols,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn js_basics() {
        let src = br#"
import { x } from "mod";
function top() {}
class Foo {
  method() {}
}
const arrow = () => 1;
"#;
        let pr = extract_javascript("t.js", src);
        let names: Vec<&str> = pr.symbols.iter().map(|s| s.qualified.as_str()).collect();
        assert!(names.contains(&"top"));
        assert!(names.contains(&"Foo"));
        assert!(names.contains(&"Foo.method"));
        assert!(names.contains(&"arrow"));
        assert!(pr.imports.iter().any(|s| s == "mod"));
    }
}
