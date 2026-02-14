# Canonical Quality Status

This document defines the single source of truth for current code-quality status.

## Canonical Current Set

- `docs/assessments/README.md` (framework, policy, pointers)
- `docs/assessments/ARCHITECTURE_QUALITY_ASSESSMENT_2026-02-12.md`
- `docs/assessments/QUALITATIVE_CODE_QUALITY_ASSESSMENT_2026-02-13.md`
- `docs/assessments/FOLDER_ORGANIZATION_ASSESSMENT.md`

## Transitional / Historical

All superseded assessment artifacts should move under `docs/assessments/archive/`.
Use `docs/assessments/archive/INDEX.md` to track historical report locations.

## Update Rule

Every PR that introduces or supersedes a quality assessment must:

1. Update this file.
2. Update `docs/assessments/README.md`.
3. Add archive index entries for superseded reports.
