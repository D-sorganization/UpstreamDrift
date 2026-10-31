---
issue: 14132
summary: "Install Spec Kit Antigravity skills for all speckit Gemini command TOMLs."
---

Install `agy` integration skills under `.agents/skills/` (core commands via
`specify integration install agy` plus git extension skills) and record hashes in
`.specify/integrations/agy.manifest.json`. Hygiene test guards TOML-to-skill parity.
