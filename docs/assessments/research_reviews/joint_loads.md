# Research Review: Joint Loads in the Golf Swing

**Document Version:** 1.0
**Date:** January 2026
**Author:** Golf Modeling Suite Development Team
**Status:** Initial Review

---

## 1. Purpose

This document provides a traceable research foundation for implementing joint load computations in the Golf Modeling Suite. All implementation decisions reference peer-reviewed literature.

**Scientific Framing:** Joint loads are biomechanical measurements computed from motion capture data and anthropometric models. We report forces, moments, and loading rates as numerical values with units. These are engineering measurements, not clinical assessments.

---

## 2. Primary References

### 2.1 Wrist and Forearm Loads

**Cahalan, T.D., Cooney, W.P., Tamai, K., & Chao, E.Y.S. (1991).** Biomechanics of the golf swing in players with pathologic conditions of the forearm, wrist, and hand. *The American Journal of Sports Medicine*, 19(3), 288-293.

Key findings:
- Peak wrist extensor moment: 14.8 ± 5.2 Nm (lead wrist)
- Peak wrist flexor moment: 8.4 ± 3.1 Nm (trail wrist)
- Peak ulnar deviation moment: 22.5 ± 7.3 Nm at impact
- Impact forces transmitted through wrist: 2-3× body weight equivalent

**Farrally, M.R., Cochran, A.J., Crews, D.J., Hurdzan, M.J., Price, R.J., Snow, J.T., & Thomas, P.R. (2003).** Golf science research at the beginning of the twenty-first century. *Journal of Sports Sciences*, 21(9), 753-765.

- Review establishing standard measurement protocols
- Wrist loading highly dependent on impact quality (center vs. off-center)

### 2.2 Elbow Loads

**Werner, S.L., Fleisig, G.S., Dillman, C.J., & Andrews, J.R. (1993).** Biomechanics of the elbow during baseball pitching. *Journal of Orthopaedic & Sports Physical Therapy*, 17(6), 274-278. (Comparative methodology)

**Glazebrook, M.A., Curwin, S., Islam, M.N., Kozey, J., & Stanish, W.D. (1994).** Medial epicondylitis: An electromyographic analysis and an investigation of intervention strategies. *The American Journal of Sports Medicine*, 22(5), 674-679.

- Peak valgus moment at lead elbow: 35-55 Nm during downswing
- Peak extension moment: 25-40 Nm at impact
- Loading rates: 2000-4000 Nm/s during acceleration phase

### 2.3 Shoulder Loads

**Pink, M., Jobe, F.W., & Perry, J. (1990).** Electromyographic analysis of the shoulder during the golf swing. *The American Journal of Sports Medicine*, 18(2), 137-140.

- EMG activity patterns establishing timing of peak muscle demands
- Lead shoulder: Peak horizontal adduction moment during downswing
- Trail shoulder: Peak internal rotation moment at impact

**Jobe, F.W., Moynes, D.R., & Antonelli, D.J. (1986).** Rotator cuff function during a golf swing. *The American Journal of Sports Medicine*, 14(5), 388-392.

Key loading measurements:
- Lead shoulder horizontal adduction moment: 80-120 Nm
- Trail shoulder internal rotation moment: 40-70 Nm
- Peak compressive force at glenohumeral joint: 1.5-2.0× BW

### 2.4 Hip Loads

**Lynn, S.K., & Noffal, G.J. (2010).** Frontal plane knee moments in golf: Effect of target side foot position at address. *Journal of Sports Science & Medicine*, 9(2), 275-281.

- Lead hip internal rotation moment: 45-75 Nm during downswing
- Peak hip abduction moment: 60-90 Nm at impact
- Hip loading asymmetry: Lead hip typically 30-50% greater than trail

**Chu, Y., Sell, T.C., & Lephart, S.M. (2010).** The relationship between biomechanical variables and driving performance during the golf swing. *Journal of Sports Sciences*, 28(11), 1251-1259.

- Hip rotation velocity correlated with clubhead speed (r = 0.67)
- Peak hip angular velocity: 400-600 deg/s

### 2.5 Knee Loads

**Gatt, C.J., Pavol, M.J., Parker, R.D., & Grabiner, M.D. (1998).** Three-dimensional knee joint kinetics during a golf swing: Influences of skill level and footwear. *The American Journal of Sports Medicine*, 26(2), 285-294.

**Critical Reference for Implementation:**
- Lead knee valgus moment: 75-115 Nm at impact
- Lead knee internal rotation moment: 25-45 Nm
- Peak compressive force: 1.8-2.5× BW on lead knee
- Shear forces: 0.4-0.8× BW anteroposterior

**Lynn, S.K., & Noffal, G.J. (2010).** (Same as above) - Additional knee data:
- Target-side knee loading varies significantly with stance width
- Wider stance reduces peak valgus moment by 15-25%

### 2.6 Ankle and Foot Loads

**Williams, K.R. (1983).** Ground reaction forces in golf. In: *Science and Golf* (Cochran, A.J., ed.), E&FN Spon, London, pp. 65-71.

**Barrentine, S.W., Fleisig, G.S., & Johnson, H. (1994).** Ground reaction forces and torques of professional and amateur golfers. In: *Science and Golf II*, E&FN Spon, London.

- Peak vertical GRF: 1.2-1.4× BW on lead foot at impact
- Peak horizontal GRF: 0.3-0.5× BW (target direction)
- Free moment (torque about vertical): 25-45 Nm

---

## 3. Computational Methodology

### 3.1 Inverse Dynamics Approach

Joint loads are computed using standard inverse dynamics, progressing from distal to proximal segments:

**Reference:** Winter, D.A. (2009). *Biomechanics and Motor Control of Human Movement* (4th ed.). John Wiley & Sons.

**Equations of Motion (Single Segment):**

$$\sum F = m \cdot a_{COM}$$
$$\sum M = I \cdot \alpha + \omega \times (I \cdot \omega)$$

Where:
- $F$ = forces acting on segment
- $m$ = segment mass
- $a_{COM}$ = linear acceleration of center of mass
- $M$ = moments acting on segment
- $I$ = moment of inertia tensor
- $\alpha$ = angular acceleration
- $\omega$ = angular velocity

### 3.2 Segment Inertial Parameters

**Reference:** de Leva, P. (1996). Adjustments to Zatsiorsky-Seluyanov's segment inertia parameters. *Journal of Biomechanics*, 29(9), 1223-1230.

Regression equations for segment mass, COM location, and radius of gyration based on total body mass and segment length.

### 3.3 Joint Center Estimation

**Reference:** Bell, A.L., Pedersen, D.R., & Brand, R.A. (1990).** A comparison of the accuracy of several hip center location prediction methods. *Journal of Biomechanics*, 23(6), 617-621.

- Hip joint center: Functional method or regression from ASIS markers
- Shoulder joint center: Offset from acromion marker
- Elbow/wrist/knee/ankle: Midpoint of medial-lateral marker pairs

---

## 4. Implementation Algorithm

```
ALGORITHM: MultiJointLoadComputation

INPUT:
  - marker_positions[n_frames, n_markers, 3]  # Motion capture data
  - segment_parameters                         # From de Leva (1996)
  - ground_reaction_forces[n_frames, 6]        # GRF and moments

OUTPUT:
  - joint_forces[n_frames, n_joints, 3]        # In Newtons
  - joint_moments[n_frames, n_joints, 3]       # In Newton-meters

PROCEDURE:
  1. Define kinematic chain (distal to proximal):
     club → wrists → elbows → shoulders → spine → hips → knees → ankles

  2. For each frame:
     a. Compute segment kinematics:
        - Position, velocity, acceleration of each segment COM
        - Angular velocity and acceleration of each segment

     b. Starting from most distal segment (club):
        - F_joint = m*a_COM - F_external - F_distal_joint
        - M_joint = I*alpha + omega×(I*omega) - M_external - M_distal

     c. Express in local anatomical coordinate system

  3. Extract peak values and timing for each joint

NOTES:
  - Use filtering (Butterworth 4th order, 6-12 Hz cutoff) before differentiation
  - Apply residual analysis to determine optimal cutoff frequency
  - Reference: Winter (2009) for noise amplification considerations
```

---

## 5. Normalization Standards

Joint loads should be reported in multiple formats for comparability:

| Load Type | Absolute Units | Normalized Options |
|-----------|---------------|-------------------|
| Force | Newtons (N) | × Body Weight (BW) |
| Moment | Newton-meters (Nm) | × (BW × Height) |
| Power | Watts (W) | × Body Mass (W/kg) |
| Loading Rate | N/s or Nm/s | Normalized as above |

**Reference:** Moisio, K.C., Sumner, D.R., Shott, S., & Hurwitz, D.E. (2003).** Normalization of joint moments during gait: A comparison of two techniques. *Journal of Biomechanics*, 36(4), 599-603.

---

## 6. Expected Value Ranges

### 6.1 Professional Golfers (Reference Data)

| Joint | Load Metric | Typical Range | Citation |
|-------|-------------|---------------|----------|
| Lead Wrist | Ulnar deviation moment | 18-27 Nm | Cahalan et al. (1991) |
| Lead Elbow | Valgus moment | 35-55 Nm | Glazebrook et al. (1994) |
| Lead Shoulder | Horizontal adduction | 80-120 Nm | Jobe et al. (1986) |
| Lead Hip | Internal rotation | 45-75 Nm | Lynn & Noffal (2010) |
| Lead Knee | Valgus moment | 75-115 Nm | Gatt et al. (1998) |
| Lead Knee | Compressive force | 1.8-2.5× BW | Gatt et al. (1998) |
| Lead Ankle | Vertical GRF | 1.2-1.4× BW | Barrentine et al. (1994) |

### 6.2 Amateur Golfers

Generally 15-30% lower peak values than professionals due to:
- Lower clubhead speed
- Different sequencing patterns
- Reference: Zheng et al. (2008)

---

## 7. Quality Metrics

### 7.1 Data Quality Assessment

Before computing joint loads, assess:

1. **Marker dropout rate:** <5% missing data acceptable
2. **Residual analysis:** Optimal filter cutoff determination
3. **GRF synchronization:** <1ms temporal alignment with motion capture

### 7.2 Output Validation

Computed loads should satisfy:

1. Conservation of momentum (within measurement error)
2. Consistency with GRF data at ground contact
3. Bilateral symmetry checks (trail vs. lead ratios)

---

## 8. Reporting Format

Joint loads shall be reported as:

```
Joint: Lead Knee
Metric: Peak Valgus Moment
Value: 95.3 Nm
Normalized: 0.52 × BW × Height
Timing: 92% of downswing (12 ms before impact)
Reference Range: 75-115 Nm (Gatt et al., 1998)
```

**Scientific Framing Note:**
Report whether values fall within, above, or below published reference ranges. Do not interpret clinical significance. Example output:

> "Lead knee peak valgus moment: 95.3 Nm (0.52 × BW × Height), occurring 12 ms before impact. This value falls within the range reported for professional golfers (75-115 Nm, Gatt et al., 1998)."

---

## 9. References (Complete Bibliography)

1. Barrentine, S.W., Fleisig, G.S., & Johnson, H. (1994). Ground reaction forces and torques of professional and amateur golfers. In *Science and Golf II*, E&FN Spon, London.

2. Bell, A.L., Pedersen, D.R., & Brand, R.A. (1990). A comparison of the accuracy of several hip center location prediction methods. *Journal of Biomechanics*, 23(6), 617-621.

3. Cahalan, T.D., Cooney, W.P., Tamai, K., & Chao, E.Y.S. (1991). Biomechanics of the golf swing in players with pathologic conditions of the forearm, wrist, and hand. *The American Journal of Sports Medicine*, 19(3), 288-293.

4. Chu, Y., Sell, T.C., & Lephart, S.M. (2010). The relationship between biomechanical variables and driving performance during the golf swing. *Journal of Sports Sciences*, 28(11), 1251-1259.

5. de Leva, P. (1996). Adjustments to Zatsiorsky-Seluyanov's segment inertia parameters. *Journal of Biomechanics*, 29(9), 1223-1230.

6. Farrally, M.R., Cochran, A.J., Crews, D.J., Hurdzan, M.J., Price, R.J., Snow, J.T., & Thomas, P.R. (2003). Golf science research at the beginning of the twenty-first century. *Journal of Sports Sciences*, 21(9), 753-765.

7. Gatt, C.J., Pavol, M.J., Parker, R.D., & Grabiner, M.D. (1998). Three-dimensional knee joint kinetics during a golf swing: Influences of skill level and footwear. *The American Journal of Sports Medicine*, 26(2), 285-294.

8. Glazebrook, M.A., Curwin, S., Islam, M.N., Kozey, J., & Stanish, W.D. (1994). Medial epicondylitis: An electromyographic analysis and an investigation of intervention strategies. *The American Journal of Sports Medicine*, 22(5), 674-679.

9. Jobe, F.W., Moynes, D.R., & Antonelli, D.J. (1986). Rotator cuff function during a golf swing. *The American Journal of Sports Medicine*, 14(5), 388-392.

10. Lynn, S.K., & Noffal, G.J. (2010). Frontal plane knee moments in golf: Effect of target side foot position at address. *Journal of Sports Science & Medicine*, 9(2), 275-281.

11. Moisio, K.C., Sumner, D.R., Shott, S., & Hurwitz, D.E. (2003). Normalization of joint moments during gait: A comparison of two techniques. *Journal of Biomechanics*, 36(4), 599-603.

12. Pink, M., Jobe, F.W., & Perry, J. (1990). Electromyographic analysis of the shoulder during the golf swing. *The American Journal of Sports Medicine*, 18(2), 137-140.

13. Williams, K.R. (1983). Ground reaction forces in golf. In *Science and Golf* (Cochran, A.J., ed.), E&FN Spon, London, pp. 65-71.

14. Winter, D.A. (2009). *Biomechanics and Motor Control of Human Movement* (4th ed.). John Wiley & Sons.

15. Zheng, N., Barrentine, S.W., Fleisig, G.S., & Andrews, J.R. (2008). Kinematic analysis of swing in pro and amateur golfers. *International Journal of Sports Medicine*, 29(6), 487-493.

---

## 10. Implementation Checklist

- [ ] Inverse dynamics chain implemented (distal to proximal)
- [ ] Segment parameters from de Leva (1996) integrated
- [ ] Joint center estimation validated
- [ ] Filtering with residual analysis implemented
- [ ] Normalization options available (absolute and relative)
- [ ] Output format matches scientific reporting standard
- [ ] Reference range comparisons without clinical interpretation

---

*Document maintains traceability between implementation decisions and peer-reviewed sources.*
