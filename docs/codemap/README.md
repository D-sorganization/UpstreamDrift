# CodeMap Indexer

The CodeMap indexer is a high-performance, tree-sitter based syntax indexer backed by SQLite FTS5.
It is shared across the D-sorganization fleet via symlinks from `Tools_Private`.

## Features
- **Fast Rebuilds**: Cold rebuilds use a `BLAKE3` hash cache to skip unmodified files.
- **In-App Integration**: Wired directly into `GasificationToolExecutor` and `UpstreamDrift` Chat backends.
- **Agent Integration**: Exposes an [MCP Server](agents.md) for external agents (Claude Code, Codex).
- **JSONL Export**: Allows exporting the full semantic index for batch ingestion.

## CLI Usage

```bash
# Rebuild the index
python -m shared.python.codemap.cli rebuild

# Search for a symbol
python -m shared.python.codemap.cli search "Solver"

# Find callers
python -m shared.python.codemap.cli who-calls "calculate_entropy"

# Export to JSONL gzip
python -m shared.python.codemap.cli export --jsonl
```

See [Agent Integration](agents.md) for using this with AI tools.
