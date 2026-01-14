# Research Review: Spinal Load Quantification

**Status:** APPROVED
**Date:** 2026-01-14
**Reviewer(s):** Research compilation from peer-reviewed literature

---

## 1. Metric Definition

### 1.1 What is being measured

Spinal load quantification measures the mechanical forces and moments acting on lumbar vertebral segments during the golf swing. This includes:

- **Axial compression:** Force along the spine axis (superior-inferior)
- **Shear forces:** Forces perpendicular to the spine axis
  - Anterior-posterior (A-P) shear
  - Lateral (medial-lateral) shear
- **Moments/Torques:** Rotational loads about spinal axes
  - Flexion-extension moment
  - Lateral bending moment
  - Axial torsion (rotation moment)

### 1.2 Physical/biomechanical meaning

These loads represent the mechanical stress experienced by spinal structures (intervertebral discs, facet joints, ligaments, muscles). They result from:

1. **Body segment weight** above the measurement level
2. **Inertial forces** from acceleration of body segments
3. **Muscle forces** required to produce and control motion
4. **Ground reaction forces** transmitted through the kinetic chain

### 1.3 Units and conventions

| Parameter | Symbol | Units | Typical Normalization |
|-----------|--------|-------|----------------------|
| Compression force | F_c | Newtons (N) | × body weight (BW) |
| A-P shear force | F_ap | Newtons (N) | × body weight (BW) |
| Lateral shear force | F_lat | Newtons (N) | × body weight (BW) |
| Flexion-extension moment | M_fe | Newton-meters (N·m) | Absolute |
| Lateral bending moment | M_lb | Newton-meters (N·m) | Absolute |
| Axial torsion | M_ax | Newton-meters (N·m) | Absolute |

### 1.4 Anatomical reference

Standard measurement level: **L4-L5 or L5-S1** (most common sites of golf-related injury)

Coordinate system (ISB recommendations):
- **X-axis:** Anterior-posterior (+ = anterior)
- **Y-axis:** Superior-inferior (+ = superior)
- **Z-axis:** Medial-lateral (+ = right)

---

## 2. Literature Review

### 2.1 Seminal Papers

#### Hosea et al. (1990) - Foundational spinal loading study

**Citation:** Hosea, T. M., Gatt, C. J., Galli, K. M., Langrana, N. A., & Zawadsky, J. P. (1990). Biomechanical analysis of the golfer's back. In A. J. Cochran (Ed.), *Science and Golf: Proceedings of the First World Scientific Congress of Golf* (pp. 43-48). London: E & FN Spon.

**Key findings:**
- First study to quantify lumbar spine forces during golf swing
- Used inverse dynamics with EMG-assisted muscle force estimation
- Measured at L3-L4 segment

**Results:**

| Group | Compression (N) | A-P Shear (N) |
|-------|----------------|---------------|
| Professional (n=4) | 7584 | 329 |
| Amateur (n=4) | 6100 | 596 |

**Critical values:**
- Peak compression: ~8× body weight (professionals)
- For comparison: Running produces ~3× BW compression
- Peak forces occur just before or at ball impact

#### Cole & Grimshaw (2016) - Modern swing biomechanics review

**Citation:** Cole, M. H., & Grimshaw, P. N. (2016). The biomechanics of the modern golf swing: Implications for lower back injuries. *Sports Medicine*, 46(3), 339-351.

**Key findings:**
- Modern swing emphasizes restricted pelvis turn with increased thorax rotation
- This creates higher X-factor but also higher spinal loading
- Lateral slide of pelvis during downswing contributes to shear forces
- Reviewed relationship between swing mechanics and injury mechanisms

**Loading mechanisms identified:**
1. High axial compression from muscle co-contraction
2. Lateral shear from asymmetric trunk positioning
3. Torsion from rapid axial rotation
4. Combined loading (compression + rotation) most damaging

#### Lindsay & Horton (2002) - Comprehensive review

**Citation:** Lindsay, D., & Horton, J. (2002). Comparison of spine motion in elite golfers with and without low back pain. *Journal of Sports Sciences*, 20(8), 599-605.

**Related review:** Lindsay, D. M., & Vandervoort, A. A. (2014). Golf-related low back pain: A review of causative factors and prevention strategies. *Asian Journal of Sports Medicine*, 5(4), e24289.

**Key findings:**
- Back pain affects 25-58% of golfers
- Three loading types: compression, shear, torsion
- "Crunch factor" concept: combined lateral bend + rotation

### 2.2 Computational Methods in Literature

#### Inverse dynamics approach

**Citation:** Lim, Y. T., & Chow, J. W. (2004). Lumbar spinal loads and muscle activity during a golf swing. In M. Hubbard, R. D. Mehta, & J. M. Pallis (Eds.), *The Engineering of Sport 5* (Vol. 2, pp. 414-420). Sheffield: ISEA.

**Method:**
1. Capture kinematics (markers on trunk, pelvis)
2. Compute segment accelerations
3. Apply Newton-Euler equations from distal to proximal
4. Estimate muscle forces from EMG or optimization
5. Sum all force contributions at spinal level

**Muscle model:**
- 22 trunk muscles included
- EMG-to-force relationship calibrated per subject
- Muscle moment arms from anatomical data

#### The "Crunch Factor" parameter

**Citation:** Defined in multiple studies; synthesized in systematic reviews.

**Definition:**
```
Crunch Factor = Lateral Inclination × Axial Rotation Velocity
```

**Purpose:**
- Captures combined effect of lateral bending and rotation
- Both create stress in intervertebral disc
- Axial rotation speed accounts for viscoelastic effects
- High values associated with increased disc injury risk

### 2.3 Controversies and limitations

1. **Model complexity trade-offs:**
   - Simple models underestimate muscle contributions
   - Complex models require many assumptions about muscle force sharing

2. **Individual variation:**
   - Muscle strength, geometry vary substantially
   - Generic models may not represent specific individuals

3. **Validation challenges:**
   - Direct measurement of spinal forces is invasive
   - In vivo validation limited to instrumented vertebral implants

4. **Causal relationship:**
   - High loads correlate with, but don't necessarily cause, injury
   - Tissue tolerance varies by individual and exposure history

---

## 3. Implementation Choice

### 3.1 Selected methodology

We implement a **simplified inverse dynamics model** with the following characteristics:

1. **Segmentation:** Rigid body model with defined trunk and pelvis segments
2. **Force computation:** Newton-Euler inverse dynamics
3. **Muscle contribution:** Estimated from joint torques using moment arm assumptions
4. **Output level:** L4-L5 segment (most commonly reported)

### 3.2 Rationale

- Inverse dynamics is established gold standard in biomechanics
- Does not require EMG (enables use with motion capture alone)
- Comparable to majority of published literature
- Computational efficiency for real-time feedback applications

### 3.3 Known limitations

1. Muscle force distribution is estimated, not measured
2. Does not account for antagonist co-contraction (may underestimate compression)
3. Assumes rigid segments (no spinal flexibility)
4. Moment arm values are population averages

### 3.4 Computational procedure

```
Algorithm: Spinal Load Estimation (Inverse Dynamics)

Input:
  - Segment kinematics: position, velocity, acceleration of trunk and pelvis
  - Segment inertial properties: mass, COM, inertia tensor
  - Body weight (BW)
  - External forces (GRF if available)

Parameters:
  - HAT mass ratio: 0.678 (head-arms-trunk as fraction of BW)
  - L4-L5 position: 0.3 × trunk length from pelvis
  - Trunk muscle moment arm: 0.05 m (typical)

Procedure:

  1. Compute mass above L4-L5:
     m_above = BW × HAT_ratio × (1 - L4L5_position_ratio)

  2. Compute gravitational contribution:
     F_grav = m_above × g × [sin(trunk_angle); 0; cos(trunk_angle)]

  3. Compute inertial forces from segment accelerations:
     F_inertia = m_above × a_COM_above

  4. Compute joint reaction forces:
     F_joint = F_grav + F_inertia - F_distal

  5. Decompose into spinal axes:
     F_compression = F_joint · spine_axis
     F_ap_shear = F_joint · anterior_axis
     F_lat_shear = F_joint · lateral_axis

  6. Estimate muscle contribution to compression:
     τ_required = inverse_dynamics_torque
     F_muscle = |τ_required| / moment_arm
     F_compression_total = F_compression + F_muscle

  7. Compute moments at L4-L5:
     M = cross(r_application, F_joint) + τ_required

Output:
  - F_compression (N and ×BW)
  - F_ap_shear (N and ×BW)
  - F_lat_shear (N and ×BW)
  - M_flexion_extension (N·m)
  - M_lateral_bend (N·m)
  - M_axial_torsion (N·m)
  - Crunch_factor (deg × deg/s)
```

---

## 4. Validation Approach

### 4.1 Reference data

**Hosea et al. (1990) values:**

| Parameter | Professional | Amateur | Units |
|-----------|-------------|---------|-------|
| Peak Compression | 7584 | 6100 | N |
| Peak Compression | ~8.0 | ~7.0 | ×BW |
| Peak A-P Shear | 329 | 596 | N |
| Peak A-P Shear | ~0.4 | ~0.7 | ×BW |

**Literature ranges (various studies):**

| Parameter | Typical Range | Units |
|-----------|--------------|-------|
| Peak Compression | 4-8 | ×BW |
| Peak Lateral Shear | 0.3-1.5 | ×BW |
| Peak A-P Shear | 0.2-0.8 | ×BW |
| Peak Axial Torsion | 50-150 | N·m |

### 4.2 Acceptance criteria

| Criterion | Tolerance | Notes |
|-----------|-----------|-------|
| Compression magnitude | Within literature range | 4-8× BW |
| Peak timing | ±10 ms of impact | Consistent with literature |
| Force direction | Anatomically plausible | Compression > shear |
| Cross-engine consistency | ±5% | For identical input |

### 4.3 Validation tests

1. **Static equilibrium:** With zero motion, only gravity load should appear
2. **Conservation:** Impulse should be consistent with momentum change
3. **Literature comparison:** Values within published ranges
4. **Cross-engine:** Identical input produces consistent output

---

## 5. Uncertainty Quantification

### 5.1 Sources of uncertainty

| Source | Magnitude | Notes |
|--------|-----------|-------|
| Segment mass estimation | ±10-15% | Anthropometric tables |
| COM location | ±2 cm | Affects moment arms |
| Moment arm assumption | ±20-30% | Individual variation |
| Acceleration noise | ±5-10% | Differentiation amplifies noise |
| Muscle force estimation | ±30-50% | Largest source |

### 5.2 Propagation method

Given uncertainty in muscle contribution, report range:
```
F_compression = F_skeletal + F_muscle_estimate
F_compression_range = [F_skeletal, F_skeletal + 2×F_muscle_estimate]
```

### 5.3 Reporting format

**CRITICAL: These are physical measurements, not diagnoses.**

```
Spinal Load Analysis (L4-L5 Segment)
Method: Inverse dynamics with muscle force estimation

Peak Compression: 5420 N (5.8× BW)
  Timing: 12 ms before impact
  Range: 4200-6600 N (model uncertainty)

Peak Lateral Shear: 890 N (0.95× BW)
  Timing: 8 ms before impact
  Side: Right (trail side for RH golfer)

Peak A-P Shear: 420 N (0.45× BW)
  Direction: Anterior

Peak Axial Torsion: 87 N·m
  Direction: Left rotation

Crunch Factor (peak): 1840 deg × deg/s
  Timing: 15 ms before impact

Literature Context:
  - Hosea et al. (1990): 6100-7584 N in amateur/professional golfers
  - These values are within published ranges for golf swing

NOTE: These are biomechanical measurements, not clinical diagnoses.
Interpretation requires qualified professional assessment.
```

---

## 6. References

1. Hosea, T. M., Gatt, C. J., Galli, K. M., Langrana, N. A., & Zawadsky, J. P. (1990). Biomechanical analysis of the golfer's back. In A. J. Cochran (Ed.), *Science and Golf I* (pp. 43-48). London: E & FN Spon.

2. Cole, M. H., & Grimshaw, P. N. (2016). The biomechanics of the modern golf swing: Implications for lower back injuries. *Sports Medicine*, 46(3), 339-351.
   - https://pubmed.ncbi.nlm.nih.gov/26604102/

3. Lindsay, D. M., & Horton, J. F. (2002). Comparison of spine motion in elite golfers with and without low back pain. *Journal of Sports Sciences*, 20(8), 599-605.

4. Lindsay, D. M., & Vandervoort, A. A. (2014). Golf-related low back pain: A review of causative factors and prevention strategies. *Asian Journal of Sports Medicine*, 5(4), e24289.
   - https://pmc.ncbi.nlm.nih.gov/articles/PMC4335481/

5. Lim, Y. T., & Chow, J. W. (2004). Lumbar spinal loads and muscle activity during a golf swing. *The Engineering of Sport 5*, 2, 414-420.

6. Gluck, G. S., Bendo, J. A., & Spivak, J. M. (2008). The lumbar spine and low back pain in golf: A literature review of swing biomechanics and injury prevention. *The Spine Journal*, 8(5), 778-788.
   - https://pubmed.ncbi.nlm.nih.gov/17938007/

7. McHardy, A., Pollard, H., & Luo, K. (2006). Golf injuries: A review of the literature. *Sports Medicine*, 36(2), 171-187.

8. Sugaya, H., Tsuchiya, A., Moriya, H., Morgan, D. A., & Banks, S. A. (1999). Low back injury in elite and professional golfers: An epidemiologic and radiographic study. In M. R. Farrally & A. J. Cochran (Eds.), *Science and Golf III* (pp. 83-91). Champaign, IL: Human Kinetics.

9. McGill, S. M. (1997). The biomechanics of low back injury: Implications on current practice in industry and the clinic. *Journal of Biomechanics*, 30(5), 465-475.

10. Cejka, N., et al. (2024). Biomechanical parameters of the golf swing associated with lower back pain: A systematic review. *Journal of Sports Sciences*.
    - https://www.tandfonline.com/doi/full/10.1080/02640414.2024.2319443

---

## Approval

- [x] Technical accuracy verified
- [x] Literature coverage adequate
- [x] Implementation choice justified
- [x] Validation plan feasible
- [x] Approved for implementation

**Approved by:** Research Review Process **Date:** 2026-01-14
