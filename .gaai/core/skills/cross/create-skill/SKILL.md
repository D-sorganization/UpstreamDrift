---
name: create-skill
description: Guide creation of a new GAAI skill following the agentskills.io spec and GAAI best practices. Activate when adding a new skill to the .gaai/core/skills/ catalog.
license: MIT
compatibility: Works with any filesystem-based AI coding agent
metadata:
  author: gaai-framework
  version: "1.0"
  category: cross
  track: cross-cutting
  id: SKILL-CRS-016
  updated_at: 2026-02-26
  status: stable
inputs:
  - skill_intent: description of what the skill should do
  - skill_category: discovery|delivery|cross
  - skill_track: discovery|delivery|cross-cutting
  - existing_skills: list of current skills to check for overlap
outputs:
  - .gaai/core/skills/{category}/{skill-name}/SKILL.md
  - updated .gaai/core/skills/README.skills.md entry
---

# Create Skill

## Purpose / When to Activate

Activate when:
name: { skill-name } # matches directory name exactly
description: { one sentence } # WHAT it does + WHEN to activate
license: MIT
compatibility: Works with any filesystem-based AI coding agent
metadata:
author: { author }
version: "1.0"
category: { discovery|delivery|cross }
track: { discovery|delivery|cross-cutting }
id: SKILL-{CAT}-{NNN}
updated_at: { YYYY-MM-DD }
status: stable|experimental # use experimental for new/unproven skills
inputs:

- { input_1 }: { description }
- { input_2 }: { description }
  outputs:
- { output_path_or_description }

---

```

**Description rule:** Must be one sentence, ≤ 1024 characters. Format:
```
