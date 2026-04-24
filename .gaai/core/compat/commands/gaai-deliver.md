---
description: Implement the next ready backlog item via Delivery Agent
---

# /gaai-deliver

Implement the next ready backlog item via an **isolated** Delivery Agent.

## Context Isolation — Non-Negotiable

**ALWAYS spawn the Delivery Agent as an isolated sub-agent.**

Discovery and Delivery system prompts must NEVER coexist in the same context window. The current session may contain Discovery context, human conversation, or other work. The Delivery Agent must start with a clean context — only its own agent definition, the workflow, and the story.

## What This Does

Spawns an isolated sub-agent that runs the Delivery Loop:
