---
issue: 15141
summary: "Replace grip-hand vertex proximity with signed triangle penetration; trim rounded MCP shell ends and rebake appearance assets."
dl_state: "in_review"
next_step: "Independent exact-head review, protected CI, and human visual disposition of the grip-hand appearance."
branch: "cursor/grip-hand-render-mesh-fix-8498"
---

Analytic convex and concave fixtures validate containment and triangle distances.
Both hands fail the new depth guard before the shell correction. Tolerances
remain unchanged. Static 1200-pixel views retain the grip at three visual rolls;
physical contact and anatomical/human visual acceptance remain unqualified.
