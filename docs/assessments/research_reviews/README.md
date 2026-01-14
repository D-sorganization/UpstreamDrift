# Research Reviews

This directory contains formal research reviews required before implementing new metrics and analysis features.

## Purpose

Every metric specified in the project design guidelines (particularly Section U and the Kinematic Metrics Addendum) requires a formal research review before implementation. This ensures:

1. **Methodological correctness** - We implement published, validated methods
2. **Transparency** - Users know exactly what we compute and how
3. **Reproducibility** - Methods are traceable to primary sources
4. **Scientific integrity** - We acknowledge limitations and controversies

## Required Reviews

The following metrics require research reviews before implementation:

### Kinematic Sequence
- [ ] `kinematic_sequence.md` - 4-segment angular velocity sequence timing

### X-Factor Metrics
- [ ] `x_factor.md` - X-factor, X-factor stretch, separation angles

### Biomechanical Loads
- [ ] `spinal_loads.md` - Lumbar compression, shear, torsion computation
- [ ] `joint_loads.md` - Hip, shoulder, elbow, wrist, knee reaction forces

### Trajectory Optimization
- [ ] `swing_optimization.md` - Objective functions, constraints, Pareto methods

## Review Template

Each research review must follow this structure:

```markdown
# Research Review: [Metric Name]

**Status:** DRAFT | IN REVIEW | APPROVED
**Date:** YYYY-MM-DD
**Reviewer(s):** [Names]

## 1. Metric Definition

### 1.1 What is being measured
[Clear physical/biomechanical definition]

### 1.2 Units and conventions
[SI units, sign conventions, reference frames]

### 1.3 Practical significance
[Why this metric matters, without value judgment]

## 2. Literature Review

### 2.1 Seminal papers
[Key publications that established the metric]

### 2.2 Methodological variations
[Different ways researchers have defined/computed this metric]

### 2.3 Controversies and limitations
[Known issues, disagreements in the field]

## 3. Implementation Choice

### 3.1 Selected methodology
[Which specific approach we implement]

### 3.2 Rationale
[Why this approach was chosen]

### 3.3 Known limitations
[What our implementation cannot capture]

## 4. Validation Approach

### 4.1 Reference data
[What we validate against]

### 4.2 Acceptance criteria
[Numerical tolerances, qualitative checks]

### 4.3 Cross-engine validation
[How we ensure consistency across physics engines]

## 5. Uncertainty Quantification

### 5.1 Sources of uncertainty
[Measurement noise, model assumptions, parameter uncertainty]

### 5.2 Propagation method
[How uncertainty is computed]

### 5.3 Reporting format
[How uncertainty is presented to users]

## 6. References

[Complete bibliography in consistent format]

## Approval

- [ ] Technical accuracy verified
- [ ] Literature coverage adequate
- [ ] Implementation choice justified
- [ ] Validation plan feasible
- [ ] Approved for implementation

**Approved by:** _________________ **Date:** _________
```

## Approval Process

1. **Draft** - Initial research review created
2. **Technical Review** - Reviewed for accuracy by domain expert
3. **Implementation Review** - Reviewed for feasibility by developer
4. **Approval** - Sign-off by project lead

Reviews in APPROVED status may proceed to implementation.

## Updating Reviews

If implementation reveals issues with the original methodology:

1. Update the research review with findings
2. Document any deviations from original specification
3. Re-review if changes are substantial

## File Naming Convention

```
{metric_category}_{specific_metric}.md

Examples:
- kinematic_sequence_timing.md
- x_factor_stretch.md
- spinal_compression_loads.md
- hip_joint_reaction_forces.md
```
