# Agent Integration (MCP)

The CodeMap index is exposed to external coding agents via the Model Context Protocol (MCP).
This allows agents like Claude Code or Codex to query the SQLite FTS5 index directly, bypassing standard file-system searching, resulting in significantly faster workspace navigation.

## Usage with Claude Code (FastMCP)

To expose the MCP server to Claude, you can configure it via `~/.mcp.json` or through the Claude Desktop configuration:

```json
{
  "mcpServers": {
    "CodeMap": {
      "command": "python",
      "args": ["-m", "shared.python.codemap.mcp"],
      "env": {
        "PYTHONPATH": "src"
      }
    }
  }
}
```

## Usage with Codex / AI Tools

For AI platforms supporting JSONL ingestion, you can perform a full export of the codebase index:

```bash
python -m shared.python.codemap export --jsonl
```

This generates `.codemap/exports/code_map.jsonl.gz`, which includes all indexed functions, classes, and docstrings.

```toml
# Example ~/.codex/config.toml
[workspace]
index_path = ".codemap/exports/code_map.jsonl.gz"
```
