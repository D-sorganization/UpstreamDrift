---
issue: 15022
summary: "Close #15022 — conformance template registry test after #15073 landed FastMCP"
dl_state: in_review
next_step: "Merge for issue-closure evidence; implementation already on main via #15073 (808174d49)."
branch: cursor/close-agent-api-15022-4eef
---

Fixes #15022 — adds a unit registry ensuring the vendored conformance template `required_tools` matches `SimulationMcpServer.tool_specs()`. FastMCP migration, CI workflow, and conformance runner shipped in #15073.