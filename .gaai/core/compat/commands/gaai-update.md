---
description: Update the GAAI framework or switch AI tool adapter
---

# /gaai-update

Update the GAAI framework or switch AI tool adapter.

## What This Does

1. Updates `.gaai/core/` to the latest version from the framework repo
2. Preserves `.gaai/project/` (your backlog, memory, artefacts)
3. Optionally switches the AI tool adapter (Claude Code, Cursor, Windsurf)
4. Runs a health check to verify integrity

## When to Use

- After pulling a newer version of the GAAI framework repo
- To switch AI tool (e.g., from Cursor to Claude Code)
- To redeploy adapters or slash commands after a corrupt state

## Instructions for Claude Code

You are running a GAAI framework update.

**Step 1 — Find the installer**

Look for `.gaai/core/scripts/install.sh` in the current working directory. If it is not present, tell the user: "No `.gaai/core/scripts/install.sh` found. Ensure `.gaai/` is present in this project."

**Step 2 — Ask what to update**

Ask the user:
