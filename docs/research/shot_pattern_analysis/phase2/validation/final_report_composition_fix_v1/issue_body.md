## Overview

A read-only composition audit of the phase-two shot pattern final report (`docs/research/shot_pattern_analysis/phase2/validation/final_report_composition_audit_v1/REVIEW.md`) identified six definite findings and composition gaps between the input assembler, report builder, interpretation renderers, and independently qualified conclusions draft.

This issue tracks the consolidated resolution of all six findings with minimal, source-bound repairs following Design-by-Contract, Law of Demeter, and DRY principles without altering scientific sources, paired statistics, or figure producers.

## Definite Audit Findings and Acceptance Criteria

### 1. Resolved Zero-Lean Results in Main Selected Table (F1)
- **Finding**: `build_final_report.py:332` reads only primary cells, leaving four PW curve-2 selections unavailable despite qualified supplemental outcomes. The resolved 24-cell/72-pattern view is not integrated.
- **Acceptance Criteria**:
  - [ ] Preserve original primary table/accounting with original unavailable statuses intact.
  - [ ] Add an explicitly protocol-labeled resolved 24-cell/72-pattern view assembled through `final_report_inputs.py:assemble_qualified_interpretation` using verified supplemental cell records.
  - [ ] Guard against missing or duplicate requested identities and assert separate individual endpoint and common-three scoring counts.

### 2. Displayed Interaction Selection Differs From Qualified Draft (F2)
- **Finding**: `final_report_inputs.py:336–353` uses `next()` by input order, picking only 6 combined2 SD1 curve-strength contrasts, whereas the qualified draft specifies 12 contrasts (SD2 and both shaft_only and combined2 profiles).
- **Acceptance Criteria**:
  - [ ] Replace implicit `next()` ordering with exact reviewed ordered protocol/cell identities for the intended 12 draft contrasts.
  - [ ] Permutation of input comparison order leaves selection unchanged.
  - [ ] Duplicates or omissions fail closed or register as explicit unavailable rather than silently substituting another pair.
  - [ ] All 12 selected full-precision objects agree with final verified JSON and render via `render_effect_table`.

### 3. Substantive Conclusion and Mechanism Synthesis Integration (F3)
- **Finding**: The production path displays tables and caveats but never synthesizes conclusions from the selected-72 evidence or conclusions draft.
- **Acceptance Criteria**:
  - [ ] Add bounded prose after the direct interactions/fixed-target bounds section (`build_final_report.py:607` onward) using source-bound numbers from the selected inputs.
  - [ ] Explain opposite selected Driver versus iron/wedge lateral/downrange correlation, realized mean-carry shifts as SD increases, and local shaft/face/loft coupling linked to `DESIGN.md` and the editable LaTeX reference.
  - [ ] Avoid inferring causation from quadrants or marginal confidence intervals; zero-spanning intervals do not become no-effect claims; every numerical headline has an exact cell/pair/protocol and population.
  - [ ] Bind successful V5 receipt provenance once available rather than historical draft provenance.

### 4. Nominal Speed Bound Qualification and Disclosure (F4)
- **Finding**: `build_final_report.py:244` states 75–125% nominal speed search without protocol qualification, omitting the Driver control supplement's 75–130% exception and absolute face box.
- **Acceptance Criteria**:
  - [ ] Scope the 75–125% speed search to original protocols with exact protocol ID.
  - [ ] Disclose the Driver control supplement's 75–130% exception and reviewed absolute face bounds with explicit protocol IDs in `_bounds_control_lines`.
  - [ ] Avoid claiming adapted bounds were preregistered or imply physical impossibility outside them.

### 5. Figure-Population Prose Scope Disambiguation (F5)
- **Finding**: `build_final_report.py:620` claims "Density contours use all supported endpoints", which is overbroad since paired densities use the common-six intersection.
- **Acceptance Criteria**:
  - [ ] Differentiate individual-cell densities (which use all individually supported endpoints) from paired densities (which use the shared six-population supported-index intersection).
  - [ ] Keep point and path trajectory subsampling distinct from density contour support.

### 6. Correct Stale Draft Revision Reference (F6)
- **Finding**: `PHASE_TWO_CONCLUSIONS_DRAFT.md:52` refers to a completed v4 receipt, but v4 terminated without completion while line 45 correctly names V5 reconciliation and pending rendering.
- **Acceptance Criteria**:
  - [ ] Correct the stale draft revision reference in `PHASE_TWO_CONCLUSIONS_DRAFT.md` to reference pending/successful V5 receipt upon integration.
  - [ ] Preserve historical v3 numerical provenance without claiming a nonexistent v4 final receipt.
