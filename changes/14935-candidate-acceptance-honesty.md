---
issue: 14935
summary: "Honest candidate acceptance flags and fail-closed gates_imply_accepted guard (#14935)"
---

- Set `candidates/candidate_ms_080s_seeded_package.json` to `accepted: false` to match its 1/5 gates (no fabricated passes).
- Add `gates_imply_accepted` and unify `validate_candidate_package` consistency checks; remove the seeded-package waiver.
- Normalize host-absolute `pipeline_argv` paths in `tests/fixtures/club_face/provenance.json` and scan that fixture in `validate_committed_evidence`.