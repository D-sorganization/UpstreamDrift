//! Rust extractor — mirrors `_lang_rust.py`.

use tree_sitter::{Node, Parser};

use super::{first_child_of, line_range, text_of, ParseResult, ParsedSymbol};

fn parser() -> Parser {
    let mut p = Parser::new();
    let lang: tree_sitter::Language = tree_sitter_rust::LANGUAGE.into();
    p.set_language(&lang).expect("rust grammar");
    p
}

fn walk(node: Node, src: &[u8], prefix: &str, out: &mut Vec<ParsedSymbol>) {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "function_item" => {
                let Some(name_node) = first_child_of(&child, "identifier") else {
                    continue;
                };
                let name = text_of(&name_node, src).to_string();
                let qualified = if prefix.is_empty() {
                    name.clone()
                } else {
                    format!("{prefix}::{name}")
                };
                let (start, end) = line_range(&child);
                let raw = text_of(&child, src);
                let sig = raw
                    .lines()
                    .next()
                    .unwrap_or("")
                    .trim_end_matches(" {")
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
            "struct_item" => {
                let Some(name_node) = first_child_of(&child, "type_identifier") else {
                    continue;
                };
                let name = text_of(&name_node, src).to_string();
                let qualified = if prefix.is_empty() {
                    name.clone()
                } else {
                    format!("{prefix}::{name}")
                };
                let (start, end) = line_range(&child);
                let sig = format!("struct {name}");
                out.push(ParsedSymbol {
                    kind: "struct".into(),
                    name,
                    qualified,
                    sig,
                    docstring: String::new(),
                    start_line: start,
                    end_line: end,
                    calls_out: Vec::new(),
                });
            }
            "impl_item" => {
                let type_node = first_child_of(&child, "type_identifier");
                let sub_prefix = if let Some(tn) = type_node {
                    let ty = text_of(&tn, src);
                    if prefix.is_empty() {
                        ty.to_string()
                    } else {
                        format!("{prefix}::{ty}")
                    }
                } else {
                    prefix.to_string()
                };
                if let Some(body) = first_child_of(&child, "declaration_list") {
                    walk(body, src, &sub_prefix, out);
                }
            }
            "mod_item" => {
                let Some(name_node) = first_child_of(&child, "identifier") else {
                    continue;
                };
                let name = text_of(&name_node, src);
                let sub_prefix = if prefix.is_empty() {
                    name.to_string()
                } else {
                    format!("{prefix}::{name}")
                };
                if let Some(body) = first_child_of(&child, "declaration_list") {
                    walk(body, src, &sub_prefix, out);
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
        if c.kind() == "use_declaration" {
            let raw = text_of(&c, src).trim().trim_end_matches(';');
            out.push(raw.to_string());
        }
    }
    out
}

pub fn extract(_path: &str, source: &[u8]) -> ParseResult {
    let mut p = parser();
    let Some(tree) = p.parse(source, None) else {
        return ParseResult {
            language: "rust".into(),
            imports: vec![],
            symbols: vec![],
        };
    };
    let root = tree.root_node();
    let mut symbols = Vec::new();
    walk(root, source, "", &mut symbols);
    ParseResult {
        language: "rust".into(),
        imports: imports(root, source),
        symbols,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_fn_struct_impl() {
        let src = br#"
use std::path::Path;

pub fn top() {}

pub struct Foo;

impl Foo {
    pub fn bar(&self) {}
}
"#;
        let pr = extract("t.rs", src);
        let names: Vec<&str> = pr.symbols.iter().map(|s| s.qualified.as_str()).collect();
        assert!(names.contains(&"top"));
        assert!(names.contains(&"Foo"));
        assert!(names.contains(&"Foo::bar"));
        assert!(pr.imports.iter().any(|s| s.contains("std::path::Path")));
    }
}
