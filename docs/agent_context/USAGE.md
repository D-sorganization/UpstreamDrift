# Development Context Usage

## Start Here

This repository keeps component and integration context in Git. Open the
[component map](README.md), [offline browser](index.html), or retrieve focused
source evidence with the shared local CLI. Existing scientific and product
registries keep their authority; this catalog adds development boundaries.

## Setup and Retrieval

Use the repository's virtual environment and initialize the exact Tools pin:

```bash
git submodule update --init vendor/ud-tools
python3 -m pip install ./vendor/ud-tools/packages/agent-context
python3 -m agent_context --root . status
python3 -m agent_context --root . search "QUERY"
python3 -m agent_context --root . context COMPONENT --max-chars 16000
```

Replace QUERY and COMPONENT with the relevant concept and returned component ID.
Reinstall after changing the provider pin with `python3 -m pip install --force-reinstall ./vendor/ud-tools/packages/agent-context`. A runtime built from another Tools
revision is not verified merely because the submodule itself is clean.

The application Tools dependency in `requirements-tools.txt` and the Rust
`Cargo.toml` revision must match the same gitlink. The required context tests
reject an older release wheel or a different source revision. The small context
package remains independently installable; it does not require installing the
full application Tools package.

## Navigating Code Graphs

The component map includes a generated Mermaid graph of registered integrations.
Start with `python3 -m agent_context --root . search "QUERY"`, then request a
component with `context COMPONENT`. The JSON includes upstream providers and
downstream consumers, boundary contracts, public interfaces and relevant tests.
Follow those edges before changing an interface; read the cited source at both
ends. The offline browser supports the same links without an external service.

For symbol-level calls and imports, use the existing CodeMap in the configured
project environment: `codemap --repo . rebuild`, then `codemap --repo . search SYMBOL`
and `codemap --repo . who-calls QUALIFIED_SYMBOL`. Its Python API exposes
`imports_of(path, repo_root=...)` and `neighbors(symbol, hops=1, repo_root=...)`.
Install the optional CodeMap environment following the [Tools CodeMap guide](https://github.com/D-sorganization/Tools/blob/main/docs/codemap.md); the small agent-context package does not install it.
These lexical edges are best-effort candidates, especially for dynamic dispatch
and plugins. Verify the actual call sites. Stale queries fail; partial parser
coverage must be resolved or inspected directly in source. Do not commit the
local SQLite index or maintain a second hand-drawn code graph.

## Changing an Integration

Read the public interfaces and contract at both ends. Update the actual
implementation and executable tests. Revise the contract when public behavior,
units, frames, schema, ownership or failure handling changes. After reviewing
those exact inputs and running their relevant tests, record a specific rationale:

```bash
python3 -m agent_context --root . review RELATION --rationale "Describe the reviewed change and actual validation."
python3 -m agent_context --root . render
python3 -m agent_context --root . check
python3 -m agent_context --root . evaluate
```

Commit catalog, contract, review and generated-view changes together. Rendering
cannot automatically renew reviews. Update the existing development-log entry
and handoff, and use the fleet presence/mailbox for agent coordination. A review
record does not substitute for passing tests or scientific approval.

## Scope and Recovery

This is a curated integration catalog, not a claim to explain every file.
Registry membership is not runtime availability. Queries include the current
worktree, source digest, line citations, coverage and review state. If evidence
is stale or absent, read source directly and fix the missing context; do not
invent an empty implementation from an empty search result. Keep .codemap caches
local and disposable. Obsidian is an optional viewer of these same Markdown files.

For optional MCP setup and package details, see the
[Tools Guide](https://github.com/D-sorganization/Tools/blob/main/docs/agent-context.md).
For fleet policy and adoption, see the
[Fleet Guide](https://github.com/D-sorganization/Repository_Management/blob/main/docs/agent-context.md).

The [existing capability atlas](../architecture/CAPABILITY_ATLAS.md) remains the product-wide feature and workflow map. Regenerate it with `python3 -m scripts.generate_capability_atlas` after changing its source registries.

CI runs context checks and navigation tests before verifying that the provider
commit is published on Tools main. A paired
consumer PR may test a candidate pin, but remains blocked until that provider
is merged and the final published revision is pinned. Local source verification
checks checkout identity; it does not imply release approval.
