//! Python extractor — mirrors `_lang_python.py`.

use tree_sitter::{Node, Parser};

use super::{first_child_of, line_range, text_of, ParseResult, ParsedSymbol};

fn parser() -> Parser {
    let mut p = Parser::new();
    let lang: tree_sitter::Language = tree_sitter_python::LANGUAGE.into();
    p.set_language(&lang).expect("python grammar");
    p
}

fn docstring(body: Option<Node>, src: &[u8]) -> String {
    let Some(body) = body else {
        return String::new();
    };
    let mut cursor = body.walk();
    for c in body.children(&mut cursor) {
        match c.kind() {
            "expression_statement" => {
                let mut cc = c.walk();
                let inner = c.children(&mut cc).next();
                if let Some(inner) = inner {
                    if inner.kind() == "string" {
                        let raw = text_of(&inner, src).trim();
                        for q in ["\"\"\"", "'''", "\"", "'"] {
                            if raw.starts_with(q) && raw.ends_with(q) && raw.len() >= 2 * q.len() {
                                let inner_s = &raw[q.len()..raw.len() - q.len()];
                                let first = inner_s.trim().lines().next().unwrap_or("");
                                return first.to_string();
                            }
                        }
                    }
                }
                return String::new();
            }
            "comment" => continue,
            _ => return String::new(),
        }
    }
    String::new()
}

fn signature(def_node: &Node, src: &[u8]) -> String {
    let mut end = def_node.start_byte();
    let mut cursor = def_node.walk();
    for c in def_node.children(&mut cursor) {
        if c.kind() == ":" {
            end = c.end_byte();
            break;
        }
    }
    let bytes = &src[def_node.start_byte()..end.max(def_node.start_byte())];
    let s = std::str::from_utf8(bytes).unwrap_or("");
    s.lines().next().unwrap_or("").trim().to_string()
}

fn collect_calls(node: Node, src: &[u8], out: &mut Vec<String>) {
    if node.kind() == "call" {
        let func = first_child_of(&node, "attribute")
            .or_else(|| first_child_of(&node, "identifier"))
            .or_else(|| node.child(0));
        if let Some(func) = func {
            out.push(text_of(&func, src).to_string());
        }
    }
    let mut cursor = node.walk();
    for c in node.children(&mut cursor) {
        collect_calls(c, src, out);
    }
}

fn walk(node: Node, src: &[u8], prefix: &str, out: &mut Vec<ParsedSymbol>) {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "decorated_definition" => {
                let mut cc = child.walk();
                if let Some(inner) = child.children(&mut cc).last() {
                    walk_def(inner, src, prefix, out);
                }
            }
            "function_definition" | "class_definition" => {
                walk_def(child, src, prefix, out);
            }
            "block" => walk(child, src, prefix, out),
            _ => {}
        }
    }
}

fn walk_def(node: Node, src: &[u8], prefix: &str, out: &mut Vec<ParsedSymbol>) {
    let Some(name_node) = first_child_of(&node, "identifier") else {
        return;
    };
    let name = text_of(&name_node, src).to_string();
    let qualified = if prefix.is_empty() {
        name.clone()
    } else {
        format!("{prefix}.{name}")
    };
    let body = first_child_of(&node, "block");
    let (start, end) = line_range(&node);
    match node.kind() {
        "function_definition" => {
            let mut calls: Vec<String> = Vec::new();
            if let Some(body) = body {
                collect_calls(body, src, &mut calls);
            }
            calls.sort();
            calls.dedup();
            calls.truncate(64);
            out.push(ParsedSymbol {
                kind: if qualified.contains('.') {
                    "method".into()
                } else {
                    "function".into()
                },
                name,
                qualified,
                sig: signature(&node, src),
                docstring: docstring(body, src),
                start_line: start,
                end_line: end,
                calls_out: calls,
            });
        }
        "class_definition" => {
            out.push(ParsedSymbol {
                kind: "class".into(),
                name,
                qualified: qualified.clone(),
                sig: signature(&node, src),
                docstring: docstring(body, src),
                start_line: start,
                end_line: end,
                calls_out: Vec::new(),
            });
            if let Some(body) = body {
                walk(body, src, &qualified, out);
            }
        }
        _ => {}
    }
}

fn imports(root: Node, src: &[u8]) -> Vec<String> {
    let mut out = Vec::new();
    let mut cursor = root.walk();
    for c in root.children(&mut cursor) {
        match c.kind() {
            "import_statement" => {
                let mut cc = c.walk();
                for sub in c.children(&mut cc) {
                    if sub.kind() == "dotted_name" {
                        out.push(text_of(&sub, src).to_string());
                    }
                }
            }
            "import_from_statement" => {
                let m = first_child_of(&c, "dotted_name")
                    .or_else(|| first_child_of(&c, "relative_import"));
                if let Some(m) = m {
                    out.push(text_of(&m, src).to_string());
                }
            }
            _ => {}
        }
    }
    out
}

pub fn extract(_path: &str, source: &[u8]) -> ParseResult {
    let mut p = parser();
    let Some(tree) = p.parse(source, None) else {
        return ParseResult {
            language: "python".into(),
            imports: vec![],
            symbols: vec![],
        };
    };
    let root = tree.root_node();
    let mut symbols = Vec::new();
    walk(root, source, "", &mut symbols);
    ParseResult {
        language: "python".into(),
        imports: imports(root, source),
        symbols,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_functions_and_classes() {
        let src = br#"
"""module docstring"""
import os
from foo import bar

def hello(name):
    """greet someone"""
    print(name)

class Greeter:
    """a greeter"""
    def hi(self):
        """say hi"""
        self.helper()
"#;
        let pr = extract("t.py", src);
        let names: Vec<&str> = pr.symbols.iter().map(|s| s.qualified.as_str()).collect();
        assert!(names.contains(&"hello"));
        assert!(names.contains(&"Greeter"));
        assert!(names.contains(&"Greeter.hi"));
        let hello = pr.symbols.iter().find(|s| s.name == "hello").unwrap();
        assert_eq!(hello.kind, "function");
        assert_eq!(hello.docstring, "greet someone");
        let hi = pr.symbols.iter().find(|s| s.name == "hi").unwrap();
        assert_eq!(hi.kind, "method");
        assert!(pr.imports.iter().any(|s| s == "os"));
        assert!(pr.imports.iter().any(|s| s == "foo"));
    }
}
