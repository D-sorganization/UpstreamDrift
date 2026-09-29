# Code-map integration: in-app chat + external agents

Both the in-app AI chat (the `ChatDockWidget` rendered inside Golf Suite,
Pose Studio, etc.) and external CLI agents (Claude Code, Codex) read from
**the same `.codemap/index.db`** at the repo root. This page documents how
the wiring works end-to-end and how to keep the index fresh.

## Architecture

```
┌─────────────────────────────┐         ┌─────────────────────────────┐
│  In-app chat (PyQt6 dock)   │         │  External agents (gh CLI)   │
│  ChatDockWidget (WebSocket) │         │  Claude Code / Codex        │
└──────────────┬──────────────┘         └──────────────┬──────────────┘
               │ WS                                    │ stdio MCP
               ▼                                       ▼
┌─────────────────────────────┐         ┌─────────────────────────────┐
│  FastAPI chat server        │         │  codemap-mcp server         │
│  ToolRegistry + ChatToolBridge        │  (from Tools, PR #2563)     │
└──────────────┬──────────────┘         └──────────────┬──────────────┘
               │ in-process call                       │ in-process call
               └──────────────────┬────────────────────┘
                                  ▼
                  ┌─────────────────────────────┐
                  │  codemap.api  (6 functions) │
                  │  search_code / get_symbol / │
                  │  who_calls / imports_of /   │
                  │  neighbors / repo_summary   │
                  └──────────────┬──────────────┘
                                 ▼
                       .codemap/index.db
                       (SQLite + FTS5)
```

- **In-app chat path:** The FastAPI server's `ToolRegistry` registers the six
  `codemap.api` functions as LLM-callable tools via
  `src/shared/python/ai/tools/codemap_tools.py`. When the model emits a
  tool-call like `search_code("ChatDockWidget")`, the response (formatted
  symbol lines: `[kind] qualified  path:start-end  signature`) is streamed
  back through the WebSocket and rendered inline in `ChatMessageBubble`.
- **External agent path:** Configured via `.mcp.json` at the repo root.
  Both Claude Code and Codex spawn the `codemap-mcp` stdio server, which
  imports the same `codemap.api` and reads the same `.codemap/index.db`.

## Keeping the index fresh

There are three update mechanisms; pick whichever fits your workflow.

### 1. Live watcher daemon (recommended for active development)

```bash
make codemap-watch &
```

This runs `codemap-watch` in the background, observing file changes via
`watchdog` and updating the index in real time. Use this when iterating
heavily — chat queries always see the latest code.

### 2. Post-commit hook (fallback when the watcher isn't running)

A ready-to-symlink hook lives at
`scripts/hooks/post-commit-codemap.sh`. Enable it once per clone:

```bash
ln -s ../../scripts/hooks/post-commit-codemap.sh .git/hooks/post-commit
chmod +x .git/hooks/post-commit
```

After each `git commit`, the hook runs `codemap rebuild --since HEAD~1`
in the background. The hook silently no-ops when `codemap` isn't on
`PATH` or when `.codemap/index.db` doesn't exist yet — first-time devs
should run `make codemap` once before relying on incremental updates.

### 3. Manual rebuild

```bash
make codemap            # one-shot full rebuild
codemap rebuild         # equivalent if the console script is installed
```

## MCP server configuration

The repo ships a `.mcp.json` at its root that **Claude Code reads
automatically** when launched from anywhere inside the repo tree. It
points at the installed `codemap-mcp` console script and sets
`CODEMAP_REPO_ROOT` to the repo's absolute path so the server queries
this checkout (not the user's CWD).

If you also use **Codex**, add the equivalent stanza to
`~/.codex/config.toml`:

```toml
[mcp_servers.codemap]
command = "codemap-mcp"
args = []

[mcp_servers.codemap.env]
CODEMAP_REPO_ROOT = "/absolute/path/to/UpstreamDrift"
```

(Codex resolves MCP servers from the user's global config rather than a
per-repo file, so this snippet lives outside the repo.)

## Graceful degradation

If `.codemap/index.db` is missing, the chat tools return a friendly
hint rather than crashing:

> Code-map index not yet built. Run `codemap rebuild` (or `make codemap`)
> from the repo root to create `.codemap/index.db`, then retry.

The MCP server returns an empty result set with the same hint. Neither
path will block the chat or the external agent from operating; they
simply can't answer code-search questions until the index exists.

## Related issues / design

- Tracker: [#5169 — chat code-map integration](https://github.com/D-sorganization/UpstreamDrift/issues/5169)
- Sub-issues: #5172 (consumer wiring), #5173 (incremental updates),
  #5174 (MCP config), #5175 (tests + perf budgets).
- Design: `chat_codemap_design.md` Part 2 §5 (Query API) & §6 (Agent
  integration).
- Upstream package: `D-sorganization/Tools` PR #2563.
