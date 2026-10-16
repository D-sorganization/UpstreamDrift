# F-UD-022

**Tools child-copy shadowing: 631 duplicated files (~110k LOC), 271 drifted, 142 "frozen with nowhere to edit"**

| Severity  | Category | Confidence | Effort | Status   |
| --------- | -------- | ---------- | ------ | -------- |
| 🟨 Medium | hygiene  | high       | L      | proposed |

**Locations:** `tests/unit/repo_hygiene/tools_child_copy_missing_counterparts.txt:1`, `CLAUDE.md:248`, `pyproject.toml:572`, `src/shared/python/_seam_redirect.py:62`

## Finding

- Severity: Medium - Category: hygiene
- Confidence: high
- Location: tests/unit/repo_hygiene/tools_child_copy_missing_counterparts.txt:1-13; CLAUDE.md:248-255; pyproject.toml:572-578;
  src/shared/python/\_seam_redirect.py:62-82
- Evidence: comparing `src/shared/python` with Tools HEAD: 631 files have a counterpart (360 identical, 271 differ), 1,746 are
  UD-only; the header file says "a header with no counterpart means the file is frozen against editing in UpstreamDrift with
  nowhere to edit it instead" — 142 entries. `sidekick/` alone is 66k LOC, `ai/` 33k, `chat/` 11k.
- Failure scenario: a bug in `ai/tools/file_ops.py` (F-UD-007) is in a child copy: CLAUDE.md forbids fixing it here, Tools
  does not have it (it is on the missing-counterparts list), so the correct process is "wait". Agents resolve this by editing
  anyway or by adding to allow-lists; the 20-root `_seam_redirect` finder is the compat layer that keeps both spellings alive.
- Why a newer engineer misses this: vendoring + a submodule + an import redirect + an ownership manifest each solve one
  symptom; together they are a second package manager with no uninstall.
- Proposed change: finish #9406: delete the 360 identical copies now (they resolve from the pin), move the 142 orphan files to
  UD ownership (strip the header), and diff-review the 271 drifted files one package at a time.
- Verification: `test_tools_child_copy_headers_have_tools_counterparts` passes with an empty missing-counterparts file.
- Effort: L

## Draft Pull Request

**Title:** `refactor(vendor): finish #9406 by deleting identical Tools child copies`

### Why This Matters (The Pitch)

Comparing src/shared/python with Tools HEAD finds 631 files with a counterpart: 360 identical and 271 drifted. Another 142 sit on a list the repo's own header describes as 'frozen against editing in UpstreamDrift with nowhere to edit it instead'. CLAUDE.md forbids fixing child copies here, yet Tools does not have those files, so the correct process is to wait. Agents resolve this by editing anyway or adding to allow-lists, and a 20-root \_seam_redirect finder keeps both import spellings alive. Vendoring plus a submodule plus a redirect plus an ownership manifest has become a second package manager with no uninstall. Finishing #9406 means deleting the 360 identical copies that already resolve from the pin (roughly 60k LOC), transferring the 142 orphans to UD ownership, and diff-reviewing the 271 drifted ones one package at a time. Less to clone, lint and read, and a clear answer to who owns each file. Size L.

### Proposed PR Body

## Problem

`tests/unit/repo_hygiene/tools_child_copy_missing_counterparts.txt:1-13`, `CLAUDE.md:248-255` and `pyproject.toml:572-578` describe a Tools child-copy regime. Against Tools HEAD, 631 files have a counterpart (360 identical, 271 differ), 142 have none, and `src/shared/python/_seam_redirect.py:62-82` keeps a 20-root redirect alive.

## Change

- [ ] Delete the 360 identical copies (they resolve from the pin)
- [ ] Move the 142 orphan files to UD ownership and strip the header
- [ ] Diff-review the 271 drifted files one package at a time

## Verification

`test_tools_child_copy_headers_have_tools_counterparts` passes with an empty missing-counterparts file.

## Risk and Rollback

Medium to high: large deletions may expose import paths that only resolve through the copies; do it per package. Rollback is reverting the package-level commit.

## Handoff

- **Branch, commit, and pull request:** `<branch>` @ `<commit>` — PR #<n> (fill in when opened)
- **Validation:** the command(s) under _Verification_ above, run locally and green; plus the repo's fast gates
  (`ruff check`, `ruff format --check`, the relevant `-m unit` subset)
- **Blockers:** none known at drafting time — record any that appear
- **Next step:** open the PR against `main`, arm auto-merge with `scripts/automerge_guard.py`, end the session
- **Follow-ups:** Follow-ups: drifted-file review is per package (sidekick, ai, chat, ...); F-UD-005 narrows BLE001 to these same roots.

Refs: adversarial review 2026-10-16, F-UD-022

---

_Generated 2026-10-16 from an adversarial review of `UpstreamDrift` @ `87d6d6ee`. This is a proposal, not a verdict — verify the cited lines before acting._
