# Assessment Prompt: Completist Audit

## Assessment Overview
You are the **Completist Auditor**. Your mission is to **identify, categorize, and prioritize** every incomplete implementation, missing feature, and technical debt marker in the codebase.

You will base your assessment on the data collected in `.jules/completist_data/` and the generated report `docs/assessments/completist/Completist_Report_*.md`.

---

## Instructions

1. **Review Collected Data**:
   - `todo_markers.txt`: Contains `TODO`, `FIXME`, `XXX` tags.
   - `not_implemented.txt`: Contains `NotImplementedError` and empty method bodies.
   - `stub_functions.txt`: Contains signatures with only `pass` or `...`.
   - `incomplete_docs.txt`: Contains placeholders in docstrings.

2. **Analyze the Generated Report**:
   - Read the latest `Completist_Report_DATE.md`.
   - Pay attention to the **Mermaid Visualization** to gauge the scale of technical debt.
   - Review the **Critical Incomplete** table.

3. **Generate Your Assessment**:
   - Create `docs/assessments/Assessment_Completist.md`.
   - **Do NOT just copy the script output.** providing meaningful *synthesis*.
   - Answer:
     - "Is this codebase production-ready?"
     - "Which features are purely aspirational (defined but not built)?"
     - "What is the 'Bus Factor' risk based on undocumented complexity?"

## Output Template

```markdown
# Assessment: Completist Audit

## Executive Summary
[Synthesize the state of completion. Are we 50% done? 90% done?]

## Visualization Analysis
[Comment on the Mermaid chart findings. Is there a backlog of TODOs growing?]

## Critical Gaps (Top 5)
1. **[Feature Name]**: [Description of gap]
   - Impact: [High/Med/Low]
   - Recommendation: [Action]

## Feature Implementation Status
| Module | Defined Features | Implemented | Gaps | Status |
|--------|------------------|-------------|------|--------|
| ...    | ...              | ...         | ...  | ...    |

## Technical Debt Roadmap
- **Short Term (Next Sprint)**: [Fix critical NotImplementedErrors]
- **Medium Term**: [Address High Priority TODOs]
- **Long Term**: [Refactor FIXMEs]

## Conclusion
[Final verdict on completeness]
```

## Grading Criteria (0-10)
- **10**: No `NotImplementedError`, no critical `TODO`s, full documentation.
- **8**: Minor `TODO`s, all features functional.
- **5**: Some features crash (NotImplemented), major gaps.
- **1**: Skeleton code only.
