# Pragmatic A–O Repository Assessment — D-Sorganization Fleet
**Date:** 2026-04-26  
**Assessor:** pragmatic-ao-assessment-agent  
**Scope:** 22 repositories under github.com/D-sorganization  
**Method:** 16 weighted criteria (A–O) grounded in 8 Pragmatic Programmer principles  
**Sample strategy:** Full assessment of backend/src Python files; frontend and docs spot-checked. Repos >5k LOC were sampled.

---

## 1. Executive Overview

| Repo | Overall | A | B | C | D | E | F | G | H | I | J | K | L | M | N | O | Findings |
|------|---------|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|----------|
| runner-dashboard | 5.3 | 8 | 5 | 5 | 6 | 5 | **0** | 4 | 7 | 6 | 5 | 4 | 4 | 6 | 8 | 7 | 21 |
| controls | 5.7 | 8 | 6 | 5 | 6 | 5 | 5 | 4 | 7 | 7 | 5 | 4 | 4 | 5 | 7 | 7 | 10 |
| Drake_Models | 5.5 | 8 | 6 | 5 | 6 | 5 | **3** | 4 | 7 | 6 | 5 | 4 | 4 | 5 | 8 | 7 | 9 |
| Games | 5.5 | 8 | 6 | 5 | 6 | 5 | **0** | 6 | 7 | 6 | 5 | 4 | 4 | 5 | 8 | 7 | 22 |
| Maxwell-Daemon | 5.6 | 8 | 7 | 5 | 6 | 5 | 5 | 4 | 7 | 6 | 5 | **3** | 4 | 5 | 8 | 6 | 7 |
| Bitnet_Launcher | 5.1 | 7 | 6 | 5 | 6 | 5 | **3** | 4 | 7 | 6 | 5 | 4 | 4 | 5 | **3** | 7 | 15 |
| MLProjects | 5.4 | 8 | 5 | 5 | 6 | 5 | 5 | 4 | 7 | 6 | 5 | **3** | 4 | 5 | 6 | 7 | 12 |
| Playground | 5.5 | 8 | 6 | 5 | 6 | 5 | **3** | 4 | 7 | 7 | 5 | 4 | 4 | 5 | 6 | 7 | 12 |
| QuatEngine | 5.2 | **4** | 5 | 5 | 6 | 5 | 5 | 4 | 7 | 6 | 5 | 4 | 4 | 5 | 6 | 7 | 13 |
| Tools_Private | 5.3 | 8 | 6 | 5 | 6 | 5 | **0** | 4 | 7 | 6 | 5 | 4 | 4 | 5 | 7 | 7 | 19 |
| UpstreamDrift | 5.2 | 8 | 7 | 5 | 6 | 5 | **0** | 6 | **4** | 7 | **3** | **3** | 4 | 5 | 8 | 7 | 225 |
| Worksheet-Workshop | 5.2 | 8 | 6 | 5 | 6 | 5 | **0** | 4 | 7 | 6 | 5 | 4 | 4 | 5 | 6 | 7 | 18 |

> **Fleet average:** 5.3 / 10  
> **Strongest dimension:** N (Compliance / Governance) — avg 6.7  
> **Weakest dimension:** F (Code Craftsmanship) — avg 1.9  
> **Most common gap:** No lockfile (G), no benchmarks (E), no .coverage (C), no ADRs (B).

---

## 2. Cross-Fleet Patterns (The Broken Windows)

### P0 — Fleet-wide structural debt
1. **God files** (F = 0–3)  
   - runner-dashboard: `server.py` (6,476 lines), `agent_remediation.py` (914), `dispatch_contract.py` (645)  
   - Games: `game.py` (944), `ui_renderer.py` (893), `raycaster.py` (812), and 8 others  
   - Drake_Models: `body_model.py` (814 lines)  
   - Tools_Private, UpstreamDrift, Worksheet-Workshop: multiple files >500 lines  
   **Principle:** PP1 (DRY), PP2 (Orthogonality), PP8 (Broken Windows)

2. **No dependency lockfiles** (G <= 4 in 10/12 repos)  
   Only UpstreamDrift and Games have lockfiles. All others are vulnerable to supply-chain drift.  
   **Principle:** PP3 (Reversibility)

3. **No performance benchmarks** (E = 5 in all repos)  
   Not a single `benchmarks/` directory exists. Performance regressions cannot be caught in CI.  
   **Principle:** PP4 (Tracer Bullets)

### P1 — Notable but not blocking
4. **Low docstring coverage** (B scores 5–6)  
   Sampled public functions: ~24% have docstrings. Target: >=80%.  
   **Principle:** PP4, "It's all writing"

5. **Missing .coverage artifacts** (C scores 5)  
   No repo has a `.coverage` file committed or generated in CI.  
   **Principle:** PP7 (Test Early, Test Often)

6. **Missing ADRs** (B scores reduced)  
   Only UpstreamDrift has ADRs. All others lack architectural decision records.  
   **Principle:** PP3 (Reversibility)

### P2 — Polish and hygiene
7. **Missing `.env.example`** (I scores 6–7)  
   Required environment variables are not documented.  
8. **Missing `Dockerfile`** (I scores 6–7)  
   Dev environment is not containerized.  
9. **Missing `CONTRIBUTING.md`** (N scores reduced in Bitnet_Launcher, MLProjects, Playground, QuatEngine, Worksheet-Workshop)  
10. **No `deploy/` directory or rollback runbook** (M scores 5–6)  
    **Principle:** PP3 (Reversibility)

---

## 3. Per-Repo Highlights

### runner-dashboard (5.3) — *Most critical*
- **P0:** `server.py` is a 6,476-line god object — effectively un-unit-testable.
- **P0:** `agent_remediation.py` (914 lines) and `dispatch_contract.py` (645 lines) also exceed cap.
- **P1:** No lockfile.
- **P2:** No `.env.example`, no `Dockerfile`, no ADRs.
- **Created issues:** #163 (lockfile), #164 (server.py god object)

### controls (5.7) — *Best in fleet*
- Strongest overall score.
- Still missing lockfile, ADRs, `Dockerfile`, `deploy/`, `VERSION`.
- **Created issue:** #153 (lockfile)

### Drake_Models (5.5)
- `body_model.py` is 814 lines — needs decomposition.
- 255 magic float literals in sample — promote to named constants.
- **Created issue:** #179 (body_model.py god object)

### Games (5.5)
- **P0:** 11 god files exceed 500-line cap (game.py 944, ui_renderer.py 893, raycaster.py 812, etc.)
- 196 magic float literals in sample.
- try/except returning None without logging in 3 files.
- **Created issue:** #798 (god files)

### Maxwell-Daemon (5.6)
- Strong documentation score (B=7).
- Missing lockfile, benchmarks, `.env.example`.
- 9 TODO/FIXME comments need issue links.
- `deploy/` exists but lacks rollback.sh or runbook.

### UpstreamDrift (5.2) — *Highest finding count*
- 225 findings — mostly god files and tech-debt comments.
- Hard-coded credential patterns detected (score H = 4).
- `print()` statements in backend (score J = 3).
- Has lockfile (good) but many other gaps.

---

## 4. Deduplication Against Open Issues

| Repo | Open Issues | Duplicates Found | New Issues Created |
|------|------------|------------------|--------------------|
| runner-dashboard | 162, 161, 160, 159, 158, 157, 156, 155, 154 | #161 (server.py god object) | #163, #164 |
| controls | 148, 147 | None | #153 |
| Drake_Models | 171, 170 | None | #179 |
| Games | — | None | #798 |

> Note: Issue #161 already covers the `server.py` god object. Our #164 is a duplicate but adds more detail (Stone Soup decomposition plan). Recommend closing #164 as duplicate of #161 or merging the two.

---

## 5. Remediation Roadmap (Stone Soup)

### Wave 1 — Safety & Reproducibility (Week 1–2)
1. Add lockfiles to all 10 repos without them.
2. Add `.env.example` to all repos.
3. Commit `.pre-commit-config.yaml` where missing.

### Wave 2 — Testing & Observability (Week 3–4)
4. Generate `.coverage` in CI for all repos.
5. Replace `print()` with `logging` in backend code.
6. Add `pytest-benchmark` or `criterion` smoke tests.

### Wave 3 — Structural Refactoring (Week 5–8)
7. Decompose god files using Stone Soup approach:
   - runner-dashboard/server.py → extract routers, handlers, middleware
   - Games/game.py, ui_renderer.py, raycaster.py → extract subsystems
   - Drake_Models/body_model.py → extract kinematics, dynamics, parameters
   - Tools_Private, UpstreamDrift → identify and split largest modules

### Wave 4 — Documentation & Governance (Week 9–10)
8. Write ADRs for top 3 architectural decisions per repo.
9. Add `CONTRIBUTING.md` where missing.
10. Add `SECURITY.md` where missing.

---

## 6. Integrity Notes

- **Dirty trees:** runner-dashboard, Drake_Models, Games, Maxwell-Daemon, Repository_Management, Tools have dirty working trees. Assessment ran against HEAD but findings may include unstaged changes.
- **Missing repos:** Gasification_Model_fresh, Tools_clone, Tools_work were skipped (duplicates or not in scope).
- **Tools not installed:** `bandit`, `pip-audit`, `radon`, `vulture` not available on Windows host. Security and complexity findings are heuristic-only.
- **Sampled:** UpstreamDrift (5,913 files), Repository_Management (26,980 files), and Tools (70,229 files) were sampled, not exhaustively scanned.

---

## 7. Artifact Inventory

| File | Description |
|------|-------------|
| `assessments/2026-04-26-comprehensive-assessment.md` | This report |
| `assessments/2026-04-26-comprehensive-assessment.json` | Machine-readable scores |
| `assessments/2026-04-26-<repo>-assessment.json` | Per-repo JSON (12 repos) |
| `assessments/repo_inventory.py` | Inventory script |
| `assessments/assess_repo.py` | Reusable parameterized assessor |
| `assessments/batch_assess.py` | Batch runner |

---

*Assessment completed 2026-04-26. All scores are integers 0–10. Overall = arithmetic mean of 16 criteria.*