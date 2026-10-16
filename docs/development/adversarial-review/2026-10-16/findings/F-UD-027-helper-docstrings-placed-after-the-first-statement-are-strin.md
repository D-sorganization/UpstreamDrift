# F-UD-027

**Helper "docstrings" placed after the first statement are string expressions, not docstrings**

| Severity | Category | Confidence | Effort | Status   |
| -------- | -------- | ---------- | ------ | -------- |
| ⬜ Low   | hygiene  | high       | S      | proposed |

**Locations:** `src/shared/python/motion_matching/cost.py:172`, `src/shared/python/motion_matching/cost.py:184`, `src/shared/python/motion_matching/cost.py:191`

## Finding

- Severity: Low - Category: hygiene
- Confidence: high
- Location: src/shared/python/motion_matching/cost.py:172-173; src/shared/python/motion_matching/cost.py:184-185;
  src/shared/python/motion_matching/cost.py:191-196
- Evidence:
  ```python
  def _position_term(sim_out, target):
      club: Any = getattr(target, "club", target)
      """``mean_n( ||r_butt_diff||^2 + ||r_ch_diff||^2 )``."""
  ```
- Failure scenario: `help()`/Sphinx show no docstring; ruff's B018 exempts string constants so this is silent. A sign that
  edits were applied mechanically (a `getattr` line inserted above the docstring by an agent).
- Why a newer engineer misses this: it still "looks" documented.
- Proposed change: move the strings to the first statement.
- Verification: `python -c "import ...cost as c; assert c._position_term.__doc__"`.
- Effort: S

## Draft Pull Request

**Title:** `style(motion-matching): move helper docstrings to the first statement`

### Why This Matters (The Pitch)

In cost.py, several helper docstrings sit after the first statement, so they are just string expressions. help() and Sphinx show nothing, and ruff's B018 exempts string constants, so nothing complains. It looks like a getattr line was inserted above the docstring mechanically. Moving the strings to the first statement is a size-S fix that restores the documentation the code already claims to have.

### Proposed PR Body

## Problem

`src/shared/python/motion_matching/cost.py:172-173`, `:184-185` and `:191-196` place the docstring after `club: Any = getattr(target, "club", target)`, so `_position_term.__doc__` is empty.

## Change

- [ ] Move the string literals to be the first statement of each helper

## Verification

`python -c "import ...cost as c; assert c._position_term.__doc__"`.

## Risk and Rollback

Negligible: comment-level change. Rollback is a revert.

## Handoff

None

Refs: adversarial review 2026-10-16, F-UD-027

---

_Generated 2026-10-16 from an adversarial review of `UpstreamDrift` @ `87d6d6ee`. This is a proposal, not a verdict — verify the cited lines before acting._
