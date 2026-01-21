---
title: "CRITICAL: Audit Trail Violation & Quality Bypasses in URDF Generator"
labels: ["jules:code-quality", "critical", "compliance"]
---

## Description
A code quality review of recent git history (Jan 20-21, 2026) identified critical violations of project standards.

### 1. Atomic Commit Violation (Audit Trail)
Commit `ad29ed9` ("fix: resolve priority issues from daily assessment (#589)") introduced **3444 files** and **534,559 insertions**.
- **Violation**: Breaks "Atomic Commits" rule in `AGENTS.md`.
- **Impact**: Destroys git bisect capability and makes code review impossible.
- **Mislabeling**: A 500k line change is not a "fix".

### 2. CI/CD Gaming & Hacks
The file `tools/urdf_generator/mujoco_viewer.py` contains:
- **Blanket Ignore**: `# mypy: ignore-errors` at the top of the file. This violates "No Blanket Exclusions".
- **Magic Number Hack**: `GRAVITY_M_S2 = 9.810`. The trailing zero appears to be a workaround to bypass static analysis checks for the magic number `9.81`.

## Required Actions
1.  **Refactor**: Remove `# mypy: ignore-errors` and fix type issues or use granular `# type: ignore[code]`.
2.  **Standardize**: Replace `GRAVITY_M_S2` with a standard import from `shared.python`.
3.  **Process**: Ensure strict enforcement of commit size limits in CI to prevent recurrence of `ad29ed9`-style dumps.
