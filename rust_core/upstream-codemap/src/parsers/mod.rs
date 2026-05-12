//! Tree-sitter extractors for Python, JavaScript, TypeScript, Rust, Markdown.
//!
//! Mirrors `src/shared/python/codemap/_lang_*.py`. Each language emits the same
//! `(kind, name, qualified, sig, docstring, start_line, end_line, calls_out)`
//! tuples so that a Rust-produced index is interchangeable with the Python one.

use std::path::Path;

pub mod javascript;
pub mod markdown;
pub mod python;
pub mod rust;

/// A single symbol extracted from a source file. Lines are 1-indexed inclusive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedSymbol {
    pub kind: String,
    pub name: String,
    pub qualified: String,
    pub sig: String,
    pub docstring: String,
    pub start_line: u32,
    pub end_line: u32,
    pub calls_out: Vec<String>,
}

/// All symbols + imports extracted from a single file.
#[derive(Debug, Clone)]
pub struct ParseResult {
    pub language: String,
    pub imports: Vec<String>,
    pub symbols: Vec<ParsedSymbol>,
}

/// Map a file extension to the language id used by the indexer.
pub fn language_for(path: &Path) -> Option<&'static str> {
    let ext = path.extension()?.to_str()?.to_ascii_lowercase();
    Some(match ext.as_str() {
        "py" | "pyi" => "python",
        "js" | "mjs" | "cjs" | "jsx" => "javascript",
        "ts" => "typescript",
        "tsx" => "tsx",
        "rs" => "rust",
        "md" | "markdown" => "markdown",
        _ => return None,
    })
}

/// Parse `source` according to the extension of `path`. Returns `None` for
/// unsupported languages.
pub fn dispatch(path: &Path, source: &[u8]) -> Option<ParseResult> {
    let lang = language_for(path)?;
    let path_str = path.to_string_lossy();
    Some(match lang {
        "python" => python::extract(&path_str, source),
        "javascript" => javascript::extract_javascript(&path_str, source),
        "typescript" | "tsx" => javascript::extract_typescript(&path_str, source),
        "rust" => rust::extract(&path_str, source),
        "markdown" => markdown::extract(&path_str, source),
        _ => return None,
    })
}

// ---- shared helpers ----

pub(crate) fn line_range(node: &tree_sitter::Node) -> (u32, u32) {
    (
        (node.start_position().row as u32) + 1,
        (node.end_position().row as u32) + 1,
    )
}

pub(crate) fn text_of<'a>(node: &tree_sitter::Node, src: &'a [u8]) -> &'a str {
    let s = &src[node.start_byte()..node.end_byte()];
    std::str::from_utf8(s).unwrap_or("")
}

pub(crate) fn first_child_of<'a>(
    node: &'a tree_sitter::Node<'a>,
    kind: &str,
) -> Option<tree_sitter::Node<'a>> {
    let mut cursor = node.walk();
    let found = node.children(&mut cursor).find(|c| c.kind() == kind);
    found
}
