# GAAI — Claude Code Integration

> GAAI framework installed in `.gaai/`. Read `.gaai/core/GAAI.md` for full governance spec.

## You Are Operating Under GAAI Governance

### Rules (Always Active)
@.gaai/core/contexts/rules/base.rules.md
@.gaai/project/contexts/rules/project.rules.md

### Canonical Files
| Purpose | File |
|---|---|
| Active backlog | `.gaai/project/contexts/backlog/active.backlog.yaml` |
| Skills index | `.gaai/core/skills/README.skills.md` |

## Project: UpstreamDrift (GAAI Fleet)

**Constraints:**
- All work happens on `staging` branch; PRs target `staging`
- Never push directly to `main`

## Slash Commands
- `/gaai-deliver` — Run Delivery Loop for next ready backlog item
- `/gaai-status` — Show current backlog and memory state
