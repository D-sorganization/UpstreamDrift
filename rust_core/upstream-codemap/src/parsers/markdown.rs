//! Markdown extractor — mirrors `_lang_markdown.py`. Emits headings as symbols.

use tree_sitter::{Node, Parser};

use super::{first_child_of, line_range, text_of, ParseResult, ParsedSymbol};

fn parser() -> Parser {
    let mut p = Parser::new();
    let lang: tree_sitter::Language = tree_sitter_md::LANGUAGE.into();
    p.set_language(&lang).expect("markdown grammar");
    p
}

fn walk(node: Node, src: &[u8], out: &mut Vec<ParsedSymbol>) {
    let mut cursor = node.walk();
    for c in node.children(&mut cursor) {
        if c.kind() == "atx_heading" || c.kind() == "setext_heading" {
            let text_node =
                first_child_of(&c, "inline").or_else(|| first_child_of(&c, "heading_content"));
            let title = match text_node {
                Some(n) => text_of(&n, src).trim().to_string(),
                None => text_of(&c, src)
                    .trim_matches(|x: char| x == '#' || x.is_whitespace())
                    .to_string(),
            };
            if title.is_empty() {
                continue;
            }
            let (start, end) = line_range(&c);
            let short_name: String = title.chars().take(80).collect();
            let short_q: String = title.chars().take(200).collect();
            out.push(ParsedSymbol {
                kind: "heading".into(),
                name: short_name,
                qualified: short_q.clone(),
                sig: short_q,
                docstring: String::new(),
                start_line: start,
                end_line: end,
                calls_out: Vec::new(),
            });
        }
        walk(c, src, out);
    }
}

pub fn extract(_path: &str, source: &[u8]) -> ParseResult {
    let mut p = parser();
    let Some(tree) = p.parse(source, None) else {
        return ParseResult {
            language: "markdown".into(),
            imports: vec![],
            symbols: vec![],
        };
    };
    let root = tree.root_node();
    let mut symbols = Vec::new();
    walk(root, source, &mut symbols);
    ParseResult {
        language: "markdown".into(),
        imports: vec![],
        symbols,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn headings_become_symbols() {
        let src = b"# Top\n\nsome text\n\n## Sub\n";
        let pr = extract("t.md", src);
        let names: Vec<&str> = pr.symbols.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"Top"));
        assert!(names.contains(&"Sub"));
    }
}
