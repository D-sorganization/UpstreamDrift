# Assessment Highlight: Executive Summary

## Assessment Overview

You are an **engineering director** synthesizing findings from all 15 assessments (A-O) into an **executive summary** with prioritized remediation roadmap.

---

## Summary Structure

### 1. Overall Score (0-100)

Calculate weighted average from all assessments:

| Category       | Assessments | Weight |
| -------------- | ----------- | ------ |
| Core Technical | A, B, C     | 25%    |
| User-Facing    | D, E, F     | 25%    |
| Reliability    | G, H, I     | 20%    |
| Sustainability | J, K, L     | 15%    |
| Communication  | M, N, O     | 15%    |

### 2. Top 10 Risks

Ranked by (Severity × Impact × Effort-to-fix):

| Rank | Risk | Assessment | Severity | Impact | Recommended Action |
| ---- | ---- | ---------- | -------- | ------ | ------------------ |
| 1    | ...  | X          | CRITICAL | High   | ...                |
| 2    | ...  | X          | MAJOR    | High   | ...                |

### 3. Remediation Roadmap

**Phase 1: Critical (48 hours)**

- BLOCKER and CRITICAL issues only

**Phase 2: Major (2 weeks)**

- MAJOR issues affecting user experience

**Phase 3: Full (6 weeks)**

- Complete quality alignment

### 4. Go/No-Go Criteria

**Ship when:**

- [ ] All BLOCKER issues resolved
- [ ] All CRITICAL issues have mitigation
- [ ] > 80% test coverage
- [ ] Installation success >90%
- [ ] First result time <30 min

**DO NOT ship if:**

- Any BLOCKER unresolved
- Installation fails >20%
- No tutorials available

### 5. Assessment Highlights

| Assessment      | Score | Key Finding | Priority     |
| --------------- | ----- | ----------- | ------------ |
| A: Architecture | X/10  | Summary     | High/Med/Low |
| B: Code Quality | X/10  | Summary     | High/Med/Low |
| ...             | ...   | ...         | ...          |

---

## Output Format

```markdown
# Comprehensive Assessment Summary

**Overall Score: X/100**
**Assessment Date: YYYY-MM-DD**
**Assessed By: [Agent/Human]**

## Executive Summary

[5-bullet executive summary]

## Score Breakdown

[Table of all 15 assessments with scores]

## Critical Risks

[Top 10 risks with mitigation]

## Remediation Roadmap

[Phased plan]

## Appendix: Individual Assessment Links

[Links to A-O results]
```

---

_This is the summary assessment. Run after completing all A-O assessments._
