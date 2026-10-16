# F-UD-015

**Agent instruction files contradict repo policy (`black`, "implement NotImplementedError") and each other**

| Severity  | Category   | Confidence | Effort | Status   |
| --------- | ---------- | ---------- | ------ | -------- |
| 🟨 Medium | governance | high       | S      | proposed |

**Locations:** `.claude/skills/lint/SKILL.md:18`, `.agent/skills/lint/SKILL.md:18`, `.kiro/steering/ci-local-testing.md:1`, `scripts/check_formatter_guidance.py:57`, `CLAUDE.md:110`

## Finding

- Severity: Medium - Category: governance
- Confidence: high
- Location: .claude/skills/lint/SKILL.md:18-22; .agent/skills/lint/SKILL.md:18-22; .kiro/steering/ci-local-testing.md:1-30;
  scripts/check_formatter_guidance.py:57-60; CLAUDE.md:110-111
- Evidence: `black .`, `mypy . --ignore-missing-imports`, "Search for `NotImplementedError` that should be implemented …
  implement the functionality" (lint skill); `black --check .` under `inclusion: always` (Kiro steering). The formatter-guidance
  hook checks `Makefile`, docs and `.pre-commit-config.yaml` for the word `black` — not the files agents actually execute.
  `.agent/skills/*` and `.claude/skills/*` are the same five skills, differing by one blank line.
- Failure scenario: an overnight agent runs `black .`, reformats files ruff-format disagrees with, the `ruff format --check`
  gate fails, and a second agent "fixes" by re-running ruff; or an agent "implements" an abstract method's
  `NotImplementedError`. Eight agent dot-dirs (`.agent .ai .claude .gaai .gemini .jules .kiro .specify`, 261 files, 2.3 MB)
  plus `CLAUDE.md`/`AGENTS.md`/`AGENT_HANDOFF.md`/`GEMINI.md` are four to eight parallel sources of truth.
- Why a newer engineer misses this: each framework scaffolds its own directory and nobody diffs them; the governance check was
  pointed at human docs.
- Proposed change: one `agents/skills/` directory with symlinks (or generated copies verified by a test) per framework; extend
  `check_formatter_guidance.py` to scan `.*/**/*.md`; delete the `black` and "implement placeholders" steps.
- Verification: `grep -rl "black" .agent .claude .kiro .gemini .jules` → empty.
- Effort: S

## Draft Pull Request

**Title:** `chore(agents): remove black and placeholder steps from agent skills`

### Why This Matters (The Pitch)

Agent instruction files contradict repo policy. The lint skill tells agents to run black . and mypy ., the Kiro steering file puts black --check under inclusion: always, and the lint skill says to implement any NotImplementedError it finds. The formatter-guidance hook checks docs, the Makefile and pre-commit for the word black, but not the files agents actually execute. An overnight agent that runs black reformats code ruff-format disagrees with, the ruff format --check gate fails, and a second agent has to clean up; one that obeys the NotImplementedError step may 'implement' an abstract method. There are eight agent dot-directories (261 files, 2.3 MB), and .agent/skills and .claude/skills are the same five skills differing by a blank line. Collapsing them to one source and extending the hook to scan them removes the contradictions and a pile of duplicate files nobody should have to keep in sync.

### Proposed PR Body

## Problem

`.claude/skills/lint/SKILL.md:18-22`, `.agent/skills/lint/SKILL.md:18-22` and `.kiro/steering/ci-local-testing.md:1-30` instruct `black .`, `mypy . --ignore-missing-imports` and "implement" `NotImplementedError`; `scripts/check_formatter_guidance.py:57-60` scans only docs, Makefile and pre-commit, and `CLAUDE.md:110-111` says ruff is the formatter. `.agent/skills/*` and `.claude/skills/*` are the same five skills.

## Change

- [ ] Create one `agents/skills/` directory with symlinks (or generated, test-verified copies) per framework
- [ ] Extend `check_formatter_guidance.py` to scan `.*/**/*.md`
- [ ] Delete the `black` and "implement placeholders" steps

## Verification

`grep -rl "black" .agent .claude .kiro .gemini .jules` → empty.

## Risk and Rollback

Low: documentation/config only; agent frameworks that cannot follow symlinks need generated copies. Rollback is a revert.

## Handoff

- **Branch, commit, and pull request:** `<branch>` @ `<commit>` — PR #<n> (fill in when opened)
- **Validation:** the command(s) under _Verification_ above, run locally and green; plus the repo's fast gates
  (`ruff check`, `ruff format --check`, the relevant `-m unit` subset)
- **Blockers:** none known at drafting time — record any that appear
- **Next step:** open the PR against `main`, arm auto-merge with `scripts/automerge_guard.py`, end the session
- **Follow-ups:** none

Refs: adversarial review 2026-10-16, F-UD-015

---

_Generated 2026-10-16 from an adversarial review of `UpstreamDrift` @ `87d6d6ee`. This is a proposal, not a verdict — verify the cited lines before acting._
