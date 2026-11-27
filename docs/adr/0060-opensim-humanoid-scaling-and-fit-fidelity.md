# ADR-0060: OpenSim Humanoid Scaling and Fit Fidelity — Panel Plan and TDD Slices

- Status: Proposed
- Date: 2026-11-06
- Decision Makers: Frontier panel (/panel), repository maintainers
- Related Issues/PRs: #13991, #13434, #13158, #13160, #13757, #13758, #13840, #11617

## Context

Issue #13991 records a mismatch between OpenSim skeleton fits and MuJoCo fits on the same capture: marker RMS can stay moderate while posture is anatomically wrong. Suspected drivers include skin-marker-based segment scaling, missing club or hand coupling in IK, truncated horizons, duplicate anthropometry paths, and Rajagopal topology limits. The issue asks for a **frontier-agent panel** transcript (out of repo), an ADR, a short research note, and an ordered list of **TDD slices** under fleet constraints (tests first, DbC, LoD, DRY, \<500 lines per PR).

This ADR is the **architecture-facing plan** slice; it does not claim the panel has met or that any OpenSim receipt is regenerated.

## Decision

Adopt the following **default technical direction** pending panel dissent recorded on #13991:

1. **Single anthropometry source.** Treat `motion_matching.anthropometry`, `hand_size`, `hip_centre_map`, and the scaled spec builder as the normative human dimensions fed to every engine. OpenSim scaling that duplicates this path is phased out in favour of generating OpenSim skeletons from the same scaled spec used by MuJoCo (reuse `musculoskeletal_spec_model.build_spec_musculoskeletal_model` and related graft pipeline where applicable — #11617).

2. **Scaling method (primary).** Prefer **functional joint-centre calibration** (SCoRE/SARA style) plus **de Leva / ANSUR II priors** (#13840) over raw skin-marker pair ratios at frame 0. Where OpenSim `ScaleTool` remains, it consumes measurements derived from the shared anthropometry document, not independent heuristics.

3. **Marker / IK registration.** Combine (a) per-marker offsets from the address hold with anatomical priors (OS-9 style), (b) weighted IK with joint-limit and coupling priors, and (c) explicit **club coupling** — represent the club as a constrained body using `grip_contact.GripInterface` weld or equivalent, so club markers constrain arm posture.

4. **Acceptance beyond RMS.** Gate fits on segment-length plausibility vs anthropometric tables, joint-angle screening (`tour_matching.marker_plausibility` family), and per-segment RMS at key events; marker RMS alone is insufficient (#13991 questions 5–6).

5. **Execution model.** Each slice below ships as its own PR: failing test first, then minimal implementation, then evidence path update. No tolerance widening without measured justification (fleet guard).

## Alternatives Considered

1. **Stock OpenSim `ScaleTool` only** — fast to wire, but repeats OS-3b skin-marker humerus inflation (#13434) and violates DRY with MuJoCo.
2. **Keep independent OpenSim scaling indefinitely** — preserves status quo, continues divergent golfers per capture.
3. **Heavy club coupling in IK before scaling is fixed** — risks masking root scaling defects; defer until segment lengths are plausible.

## Consequences

- Positive: One anthropometry story across engines; IK constraints align with capture physics; tests catch "wrong but low-RMS" poses.
- Negative: Touches graft, tour matching, and lift bundles; requires OpenSim host time for receipts; needs careful ordering to avoid churn.
- Follow-ups: Child issues in §Validation; fix `musculoskeletal_graft.pelvis_frame_from_spec` vs `HIP_MIRROR` (#11737) and grip rotation defect #13758 inside the relevant slices.

## Validation

- **ADR itself:** `docs/adr/README.md` index row; `python3 scripts/check_document_title_case.py` on changed docs if CI enforces title case (follow repo checklist).
- **Plan execution:** Each child issue merges with tests proving its acceptance line; OpenSim receipts land under `docs/development/` evidence trees with hashed inputs.

## Ordered TDD Slices (Child-Issue Candidates)

Each item is intentionally one PR-sized step; IDs are placeholders until filed from #13991.

1. **Humerus scale sanity gate** — fail when any segment scale factor from OpenSim preprocessing lies outside anthropometric corridors vs shared de Leva / ANSUR bands (#13434).
2. **Address-pose agreement** — OpenSim static pose vs MuJoCo spec pose within stated bounds on shared markers.
3. **Club coupling contract** — failing test for IK without club constraints; passing path wires club body / weld from `grip_contact`.
4. **Horizon parity** — state length / time base matches MuJoCo match window for the same capture (no silent truncation).
5. **Topology / graft blockers** — `HIP_MIRROR` graft crash (#11737) and wrist orientation (#13758) fixed with regression tests.
6. **Fit quality metrics** — implement screening metrics suite; keep marker RMS as a secondary gate.
7. **Receipt refresh** — regenerate OpenSim skeleton and muscle graft fits for capture-A (driver) and capture-B (7-iron) with provenance hashes.

## Panel Dissent and Transcript

Record each seat’s dissent and the adopted recommendation **on #13991**; this ADR is updated to **Accepted** only after that comment exists (no transcript in-repo — private video evidence stays out of git per #13991).
