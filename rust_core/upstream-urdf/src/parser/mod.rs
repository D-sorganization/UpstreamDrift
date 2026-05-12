//! Parsers from XML source to the typed AST.
//!
//! `urdf` is the primary, fully-implemented parser. `mjcf` is staged but
//! is intentionally scoped to a follow-up issue.

pub mod mjcf;
pub mod urdf;
