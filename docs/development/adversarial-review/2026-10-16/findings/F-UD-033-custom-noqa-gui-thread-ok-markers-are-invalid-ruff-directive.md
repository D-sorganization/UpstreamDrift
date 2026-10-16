# F-UD-033

**Custom `# noqa: gui-thread/ok` markers are invalid ruff directives (warnings on every lint run)**

| Severity | Category | Confidence | Effort | Status   |
| -------- | -------- | ---------- | ------ | -------- |
| ⬜ Low   | hygiene  | high       | S      | proposed |

**Locations:** `src/tools/character_builder/gui.py:22`, `src/tools/grip_wrench_plots/gui.py:63`, `src/tools/ground_reaction_plots/gui.py:54`, `src/tools/impact_parameters_panel/gui.py:147`, `src/tools/necromatcher/gui.py:3`

## Finding

- Severity: Low - Category: hygiene
- Confidence: high
- Location: src/tools/character_builder/gui.py:22; src/tools/grip_wrench_plots/gui.py:63; src/tools/ground_reaction_plots/gui.py:54;
  src/tools/impact_parameters_panel/gui.py:147; src/tools/necromatcher/gui.py:3
- Evidence: `warning: Invalid '# noqa' directive on src/tools/necromatcher/gui.py:3: expected a comma-separated list of codes`.
- Failure scenario: the GUI-thread ratchet reuses ruff's `noqa` namespace; ruff treats the line as a malformed blanket noqa,
  and the suppression ratchet counts "bare noqa" with a regex that may or may not include these. Noise trains people to ignore
  warnings.
- Why a newer engineer misses this: `# noqa:` looks like a free-form tag.
- Proposed change: use `# gui-thread: ok - reason` in `check_gui_thread_blocking_ratchet.py`.
- Verification: `ruff check .` emits zero warnings.
- Effort: S

## Draft Pull Request

**Title:** `chore(lint): replace invalid noqa gui-thread markers with own tag`

### Why This Matters (The Pitch)

Five GUI files use '# noqa: gui-thread/ok', which ruff rejects as a malformed directive, so every lint run emits warnings. The suppression ratchet may or may not count them as bare noqa. Warnings nobody can act on train people to ignore warnings. Switching to a dedicated '# gui-thread: ok - reason' tag read by check_gui_thread_blocking_ratchet.py makes ruff output clean.

### Proposed PR Body

## Problem

`src/tools/character_builder/gui.py:22`, `grip_wrench_plots/gui.py:63`, `ground_reaction_plots/gui.py:54`, `impact_parameters_panel/gui.py:147` and `necromatcher/gui.py:3` produce `warning: Invalid '# noqa' directive ... expected a comma-separated list of codes`.

## Change

- [ ] Use `# gui-thread: ok - reason` and update `check_gui_thread_blocking_ratchet.py` to read it

## Verification

`ruff check .` emits zero warnings.

## Risk and Rollback

Low: comment markers and one regex. Rollback is a revert.

## Handoff

None

Refs: adversarial review 2026-10-16, F-UD-033

---

_Generated 2026-10-16 from an adversarial review of `UpstreamDrift` @ `87d6d6ee`. This is a proposal, not a verdict — verify the cited lines before acting._
