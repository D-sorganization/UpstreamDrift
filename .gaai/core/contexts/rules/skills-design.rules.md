---
type: rules
category: skills
id: RULES-SKILLS-DESIGN-002
tags:
  - skills
  - design
  - hardcoding
  - modularity
  - registry
created_at: 2026-02-21
updated_at: 2026-02-21
---

# GAAI Skills Design Rules — Modularity & Dynamic Resolution

This document defines **mandatory design constraints for skill authoring**.
It is a companion to `skills.rules.md` (execution isolation) and applies at authoring time, not execution time.

A skill that violates these rules compiles and runs — but silently drifts as the project evolves.
That is worse than a skill that fails loudly.

---

## Why This Matters

Skills are **framework-level artefacts**. They outlive any single project state.

When a skill hardcodes a resource that belongs to the project (a file path, a category name, a technology name, a provider name), it makes an assumption about the project's current shape. That assumption is true today. It becomes false the moment the project evolves — and the skill won't tell you.

**The failure mode is silent.** An agent executing a skill with stale hardcoded paths does not error — it loads nothing, skips context it should have, or routes knowledge to a category that no longer exists. The system continues. The output is subtly wrong.

**Structural constants are different.** State machines, output format schemas, naming conventions, severity levels — these define the framework's grammar. They are correct to hardcode because changing them is a deliberate governance decision, not a project evolution.

The line is: **hardcode the framework, discover the project.**

---

## R8 — Never Hardcode Resource Paths That Belong to the Project

A skill MUST NOT hardcode specific file paths to project-owned resources.

**Forbidden examples:**
