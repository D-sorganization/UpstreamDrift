# Design Guidelines Addendum: Kinematic Metrics & Biomechanical Load Quantification

**Document Version:** 1.0
**Date:** January 2026
**Status:** DRAFT - Pending Research Review
**Parent Document:** `project_design_guidelines.qmd`

---

## Executive Summary

This addendum specifies requirements for computing and reporting standard kinematic metrics and biomechanical loads in the Golf Modeling Suite. These are **objective quantification tools**—we measure and report what is, we do not prescribe what should be.

### Guiding Principle

> **We quantify, we do not judge.** The system reports numerical values with units, comparisons to reference distributions, and uncertainty bounds. Interpretation is left to the practitioner.

This means:
- ❌ "Your X-factor stretch is too high"
- ✅ "X-factor stretch: 52.3° ± 2.1° (75th percentile of N=1247 amateur dataset)"

---

## Section U: Kinematic Sequence Analysis (Mandatory)

The kinematic sequence describes the temporal ordering of peak angular velocities from proximal to distal segments during the downswing. This is a fundamental descriptor of swing mechanics.

### U1. Kinematic Sequence Metrics (Required)

For each swing, compute and report:

#### U1.1 Segment Angular Velocities

| Segment | Definition | Reference Frame |
|---------|------------|-----------------|
| Pelvis | Angular velocity of pelvis segment | Global (world) |
| Thorax | Angular velocity of thorax/trunk | Global (world) |
| Lead Arm | Angular velocity of lead upper arm | Global (world) |
| Lead Hand/Club | Angular velocity of club | Global (world) |

**Requirements:**
- Time series at simulation frequency (≥100 Hz)
- Report in deg/s (display) and rad/s (internal)
- Both axial (rotation) and total angular velocity magnitude

#### U1.2 Sequence Timing Parameters

| Metric | Symbol | Definition | Units |
|--------|--------|------------|-------|
| Peak Pelvis Time | t_pelvis | Time of peak pelvis angular velocity | s |
| Peak Thorax Time | t_thorax | Time of peak thorax angular velocity | s |
| Peak Arm Time | t_arm | Time of peak lead arm angular velocity | s |
| Peak Club Time | t_club | Time of peak club angular velocity | s |
| Pelvis-Thorax Lag | Δt_PT | t_thorax - t_pelvis | ms |
| Thorax-Arm Lag | Δt_TA | t_arm - t_thorax | ms |
| Arm-Club Lag | Δt_AC | t_club - t_arm | ms |
| Total Sequence Time | Δt_total | t_club - t_pelvis | ms |

**Validation requirement:** Peak times must be referenced to a consistent event (e.g., transition point, impact, or arbitrary t=0).

#### U1.3 Sequence Ordering

Report the actual ordering of peak velocities:
- Example: "Pelvis → Thorax → Arm → Club" (canonical proximal-to-distal)
- Example: "Pelvis → Arm → Thorax → Club" (non-canonical)

**No value judgment on ordering.** Report what was observed.

#### U1.4 Peak Velocity Magnitudes

| Metric | Symbol | Definition | Units |
|--------|--------|------------|-------|
| Peak Pelvis ω | ω_pelvis_max | Maximum pelvis angular velocity | deg/s |
| Peak Thorax ω | ω_thorax_max | Maximum thorax angular velocity | deg/s |
| Peak Arm ω | ω_arm_max | Maximum lead arm angular velocity | deg/s |
| Peak Club ω | ω_club_max | Maximum club angular velocity | deg/s |

### U2. Implementation Requirements

- **Reference:** Cheetham et al. (2008) "An innovative system to measure the kinematics of the golf swing" — establishes the 4-segment sequence model
- **Cross-engine validation:** Sequence timing must agree within ±1ms between engines for same input motion
- **Uncertainty quantification:** Report measurement uncertainty based on:
  - Temporal resolution (1/sample_rate)
  - Smoothing filter effects (if applied)
  - Marker/pose estimation noise (if applicable)

---

## Section V: X-Factor and Separation Metrics (Mandatory)

The X-factor and related metrics describe pelvis-thorax coordination, a fundamental characteristic of rotational mechanics in the golf swing.

### V1. X-Factor Definitions

**Critical note:** Multiple definitions exist in literature. We implement and report all major variants with explicit citations.

#### V1.1 X-Factor (Static/Classic Definition)

| Metric | Symbol | Definition | Reference |
|--------|--------|------------|-----------|
| X-Factor at Top | XF_top | θ_thorax - θ_pelvis at top of backswing | McLean (1992) |

Where:
- θ_thorax = Thorax rotation angle in transverse plane (global Z-axis)
- θ_pelvis = Pelvis rotation angle in transverse plane (global Z-axis)
- "Top of backswing" = time of maximum lead arm elevation or user-defined

**Units:** degrees

#### V1.2 X-Factor Stretch

| Metric | Symbol | Definition | Reference |
|--------|--------|------------|-----------|
| X-Factor Stretch | XFS | max(XF) during transition phase | Cheetham et al. (2001) |
| XFS Timing | t_XFS | Time of maximum X-factor stretch | |
| XFS Increment | ΔXFS | XFS - XF_top | |

**Transition phase:** Defined as t_top to t_impact or user-specified window.

The X-factor stretch captures the phenomenon where pelvis rotation toward target begins before thorax rotation, momentarily increasing pelvis-thorax separation.

#### V1.3 X-Factor Rate Metrics

| Metric | Symbol | Definition | Units |
|--------|--------|------------|-------|
| XF Rate at Transition | dXF/dt | Rate of X-factor change at transition | deg/s |
| Peak XF Rate | (dXF/dt)_max | Maximum rate of X-factor change | deg/s |
| Time to Peak XF Rate | t_peak_rate | Time of peak X-factor rate | ms from transition |

### V2. Pelvis-Thorax Separation Components

Report separation in all planes (not just transverse):

| Component | Definition | Units |
|-----------|------------|-------|
| Transverse Separation | Rotation difference (Z-axis) | degrees |
| Frontal Separation | Lateral bend difference | degrees |
| Sagittal Separation | Forward bend difference | degrees |

### V3. S-Factor and Related Metrics

| Metric | Symbol | Definition | Reference |
|--------|--------|------------|-----------|
| S-Factor | SF | θ_thorax - θ_pelvis - θ_lead_arm | Extended sequence |
| Hip-Shoulder Separation | HSS | Alternative name for X-factor | Common usage |

### V4. Implementation Requirements

- **Coordinate system:** All rotations relative to address position or global frame (user-configurable, explicitly reported)
- **Angle convention:** Right-hand rule, positive = rotation toward target for RH golfer
- **Handedness:** Automatic sign adjustment for left-handed golfers
- **Filter settings:** Report any smoothing applied (e.g., 10 Hz low-pass)

### V5. Required Research Review

Before implementation, review and document:

1. **McLean (1992)** - Original X-factor definition
2. **Cheetham et al. (2001)** - X-factor stretch concept
3. **Myers et al. (2008)** - "X-Factor revisited" — critical analysis of measurement methods
4. **Joyce et al. (2016)** - Comparison of X-factor calculation methods

**Action:** Document which specific methodology is implemented and why.

---

## Section W: Biomechanical Load Quantification (Mandatory)

This section specifies the computation and reporting of forces and moments on anatomical structures. These are **objective measurements**, not health assessments or injury predictions.

### W1. Framing Principle

> We compute and report biomechanical loads as physical quantities with units. We do not diagnose, predict injuries, or prescribe technique changes. Interpretation requires qualified professionals.

**Appropriate output:**
- "L4-L5 axial compression: 4850 N (5.2× body weight)"

**Inappropriate output:**
- "Your spine is at high risk of injury"

### W2. Spinal Load Metrics

#### W2.1 Lumbar Spine Loads

For each lumbar segment (L3-L4, L4-L5, L5-S1), compute:

| Metric | Symbol | Definition | Units |
|--------|--------|------------|-------|
| Axial Compression | F_compression | Force along spine axis | N, ×BW |
| Anterior-Posterior Shear | F_ap_shear | Force perpendicular to spine (A-P) | N, ×BW |
| Lateral Shear | F_lat_shear | Force perpendicular to spine (M-L) | N, ×BW |
| Axial Torsion | τ_torsion | Moment about spine axis | N·m |
| Flexion-Extension Moment | M_flex | Moment in sagittal plane | N·m |
| Lateral Bending Moment | M_lat | Moment in frontal plane | N·m |

#### W2.2 Normalization

All forces reported in both:
- Absolute units (N, N·m)
- Normalized to body weight (×BW)

#### W2.3 Peak Values and Time Series

Report:
- Complete time series at simulation frequency
- Peak values with timing
- Integral measures (impulse, cumulative load)

### W3. Joint Load Metrics

For each major joint (hip, shoulder, elbow, wrist, knee), compute:

| Metric | Symbol | Definition | Units |
|--------|--------|------------|-------|
| Joint Reaction Force | F_joint | 3D force vector at joint | N |
| Joint Moment | M_joint | 3D moment vector at joint | N·m |
| Joint Power | P_joint | τ · ω at joint | W |
| Cumulative Work | W_joint | ∫ P dt | J |

### W4. Comparative Context (Not Prescription)

To aid interpretation, provide **reference distributions** without judgment:

- "This value is at the Nth percentile of [specific population]"
- Populations must be explicitly defined (e.g., "amateur golfers N=500, age 40-60")
- Uncertainty in reference distributions must be reported

**Example output:**
```
L4-L5 Peak Compression: 5420 N (5.8× BW)
  Reference: Amateur population (N=347)
    Mean: 4200 N (4.5× BW), SD: 890 N
    This measurement: 86th percentile
  Reference: Professional population (N=89)
    Mean: 5100 N (5.5× BW), SD: 720 N
    This measurement: 67th percentile
```

### W5. Literature-Derived Thresholds (Informational Only)

If reporting thresholds from literature, they are **informational context**, not diagnostic criteria:

```
L4-L5 Peak Compression: 5420 N

Literature context (informational, not diagnostic):
  - Hosea et al. (1990): measured 6000-8000 N in professional golfers
  - McGill (1997): 3400 N suggested as occupational limit (not golf-specific)

Note: These values are from published literature and are provided
for context only. This system does not diagnose injury risk.
```

### W6. Implementation Requirements

- **Method transparency:** Clearly document computational method (inverse dynamics, estimated from kinematics, etc.)
- **Assumptions:** List all assumptions (segment mass estimates, muscle contribution estimates)
- **Validation status:** Indicate whether method has been validated against direct measurement
- **Uncertainty:** Report estimated uncertainty in computed loads

### W7. Required Research Review

Before implementation, review and document methodology from:

1. **Hosea et al. (1990)** - "Biomechanical analysis of the golfer's back"
2. **Cole & Grimshaw (2016)** - "The biomechanics of the modern golf swing"
3. **Lindsay & Horton (2002)** - "Lumbar spine and low back pain in golf"
4. **McGill (1997)** - Spinal loading reference values

**Action:** Document which equations are implemented, with derivations traceable to published sources.

---

## Section X: Swing Optimization and Trajectory Planning

### X1. Framing as Exploration, Not Prescription

The optimization module generates **feasible trajectories** that satisfy specified constraints and objectives. It does not determine "correct" technique.

**Appropriate framing:**
- "Given these constraints and this objective, the optimizer found this solution"
- "This trajectory achieves clubhead velocity X with estimated spinal load Y"

**Inappropriate framing:**
- "This is the optimal swing for you"
- "You should swing this way"

### X2. Multi-Objective Presentation

When multiple objectives exist (e.g., clubhead velocity vs. joint loading), present:

- Pareto frontier of solutions
- Trade-off curves
- User selects point on frontier

No single "best" solution is prescribed.

### X3. Required Research Review

Before implementation of trajectory optimization:

1. **Sharp (2009)** - "Kinetic Constrained Optimization of the Golf Swing Hub Path"
2. **Nesbit & Serrano (2005)** - "Work and Power Analysis of the Golf Swing"
3. **MacKenzie & Sprigings (2009)** - Shaft dynamics and optimization

---

## Section Y: Reference Data and Population Comparisons

### Y1. Reference Dataset Requirements

Any reference dataset used for comparison must have:

| Requirement | Specification |
|-------------|---------------|
| Sample size | Minimum N=30 per stratum |
| Demographics | Age, sex, skill level, handedness |
| Methodology | Data collection protocol documented |
| Provenance | Source citation, access date |
| Uncertainty | Standard deviation or confidence intervals |

### Y2. Skill Level Definitions

When stratifying by skill level, use explicit definitions:

| Category | Definition |
|----------|------------|
| Beginner | USGA handicap >25 or equivalent |
| Amateur | USGA handicap 10-25 |
| Low Amateur | USGA handicap <10 |
| Professional | Playing professionally or equivalent |
| Elite | Top-tier professional (e.g., PGA Tour) |

### Y3. Presentation of Comparisons

Comparisons are presented as statistical context, not evaluation:

```
Your measurement: [value]
Population distribution: mean ± SD
Your percentile: Nth percentile
Sample: [description of population]
```

---

## Section Z: Research Review Protocol

### Z1. Requirement

Before implementing any metric specified in this addendum, complete a formal research review documented in `docs/assessments/research_reviews/`.

### Z2. Research Review Template

Each review must include:

```markdown
# Research Review: [Metric Name]

## 1. Metric Definition
- What is being measured
- Physical/biomechanical meaning
- Units and conventions

## 2. Literature Review
- Seminal papers (with citations)
- Methodological variations in literature
- Controversies or disagreements

## 3. Implementation Choice
- Which methodology we implement
- Rationale for choice
- Known limitations

## 4. Validation Approach
- How we verify correctness
- Reference data for validation
- Acceptance criteria

## 5. Uncertainty Quantification
- Sources of uncertainty
- Propagation method
- Reporting format

## 6. References
- Complete bibliography
```

### Z3. Review Approval

Research reviews require sign-off before implementation:

- [ ] Technical accuracy verified
- [ ] Literature coverage adequate
- [ ] Implementation choice justified
- [ ] Validation plan feasible

---

## Appendix: Metric Computation Checklist

For each metric category, the following must be implemented:

### Kinematic Sequence
- [ ] 4-segment angular velocity time series
- [ ] Peak timing extraction
- [ ] Sequence lag computation
- [ ] Sequence order reporting
- [ ] Cross-engine validation tests

### X-Factor Metrics
- [ ] X-factor at top computation
- [ ] X-factor stretch computation
- [ ] XFS timing
- [ ] Rate metrics
- [ ] Multi-plane separation
- [ ] Research review completed

### Biomechanical Loads
- [ ] Spinal compression/shear computation
- [ ] Joint reaction forces
- [ ] Normalization to body weight
- [ ] Time series and peaks
- [ ] Uncertainty estimates
- [ ] Reference data integration
- [ ] Research review completed

### Optimization
- [ ] Pareto frontier generation
- [ ] Constraint satisfaction
- [ ] Multi-objective presentation
- [ ] No single "best" prescription
- [ ] Research review completed

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-14 | Claude | Initial draft |

---

## References

1. Cheetham, P. J., et al. (2001). "The relationship between pelvis-thorax separation angle and driving distance." *Journal of Sports Sciences*.
2. McLean, J. (1992). "Widen the gap." *Golf Magazine*.
3. Myers, J., et al. (2008). "The role of the kinetic chain in the golf swing." *Sports Biomechanics*.
4. Hosea, T., et al. (1990). "Biomechanical analysis of the golfer's back." *Science and Golf I*.
5. Cole, M. & Grimshaw, P. (2016). "The biomechanics of the modern golf swing." *Routledge Handbook of Biomechanics and Human Movement Science*.
6. Sharp, R. S. (2009). "Kinetic constrained optimization of the golf swing hub path." *Sports Engineering*.
7. Nesbit, S. M. & Serrano, M. (2005). "Work and power analysis of the golf swing." *Journal of Sports Science and Medicine*.
