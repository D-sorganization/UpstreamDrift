# CodeMap Indexer

The CodeMap indexer is a high-performance, tree-sitter based syntax indexer backed by SQLite FTS5.

The canonical implementation lives in [Tools](https://github.com/D-sorganization/Tools)
under `src/shared/python/codemap/`; this repo carries a byte-identical copy of
those modules at `src/shared/python/codemap/` plus a thin consumer-side
adapter at `src/shared/python/ai/tools/codemap_tools.py` that wires the
canonical six-function API into the in-app chat `ToolRegistry`. Bug fixes
to the indexer/api/parsers belong in Tools; downstream tests covering the
chat-tool wiring live here.

## Features

- **Fast Rebuilds**: Cold rebuilds use a `BLAKE3` hash cache to skip unmodified files.
- **In-App Integration**: Wired directly into `GasificationToolExecutor` and `UpstreamDrift` Chat backends.
- **Agent Integration**: Exposes an [MCP Server](agents.md) for external agents (Claude Code, Codex).
- **JSONL Export**: Allows exporting the full semantic index for batch ingestion.

## CLI Usage

```bash
# Rebuild the index
codemap rebuild

# Search for a symbol
codemap search "Solver"

# Find callers
codemap who-calls "calculate_entropy"

# Index summary
codemap summary
```

The canonical console-script entry points are `codemap` (CLI) and
`codemap-mcp` (MCP server). The equivalent module-form invocations are
`python -m codemap.cli ...` and `python -m codemap.mcp_server`.

## Optional Rust binary (`upstream-codemap`)

The Rust crate at `rust_core/upstream-codemap/` produces the same
`.codemap/index.db` schema and is a drop-in for the rebuild step. Build it
with `cargo build --release -p upstream-codemap`, then either invoke it
directly:

```bash
./target/release/upstream-codemap rebuild
./target/release/upstream-codemap search "Solver" -k 10
./target/release/upstream-codemap watch --debounce-ms 500
```

…or opt-in to it from the Python CLI:

```bash
codemap rebuild --rust   # delegates to upstream-codemap if on PATH, else
                         # falls back transparently to the Python indexer
```

On a clean checkout the cold rebuild walks the entire repo in ~30s vs the
Python implementation's minutes. Tracked under
[issue #5217](https://github.com/D-sorganization/UpstreamDrift/issues/5217).

See [Agent Integration](agents.md) for using this with AI tools.
