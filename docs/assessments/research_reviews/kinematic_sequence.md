# Research Review: Kinematic Sequence

**Status:** APPROVED
**Date:** 2026-01-14
**Reviewer(s):** Research compilation from peer-reviewed literature

---

## 1. Metric Definition

### 1.1 What is being measured

The **kinematic sequence** describes the temporal ordering and relative timing of peak angular velocities across body segments during the golf swing downswing. It quantifies the proximal-to-distal energy transfer pattern from large muscle groups (pelvis, trunk) to smaller, faster-moving segments (arms, club).

### 1.2 Physical/biomechanical meaning

The kinematic sequence captures the "summation of speed" principle: each successive segment in the kinetic chain achieves a higher peak angular velocity than the preceding segment. The timing of these peaks—and the deceleration of proximal segments as distal segments accelerate—reflects the efficiency of energy transfer through the system.

### 1.3 Units and conventions

| Parameter | Symbol | Units | Sign Convention |
|-----------|--------|-------|-----------------|
| Angular velocity | ω | deg/s or rad/s | Positive = rotation toward target (RH golfer) |
| Peak time | t_peak | seconds (s) or milliseconds (ms) | Relative to transition (t=0) or impact |
| Inter-segment lag | Δt | milliseconds (ms) | t_distal - t_proximal |

### 1.4 Segment definitions

The standard 4-segment model (Cheetham et al., 2008):

| Segment | Anatomical Definition | Typical Marker Set |
|---------|----------------------|-------------------|
| Pelvis | Rotation of pelvic girdle | ASIS (L/R), PSIS (L/R) |
| Thorax/Trunk | Rotation of upper torso | Acromion (L/R), C7, T10 |
| Lead Arm | Rotation of lead upper arm | Shoulder, Elbow, Wrist |
| Club | Rotation of golf club | Grip, Shaft, Clubhead |

---

## 2. Literature Review

### 2.1 Seminal Papers

#### Cheetham et al. (2008) - Foundational kinematic sequence paper

**Citation:** Cheetham, P. J., Rose, G. A., Hinrichs, R. N., Neal, R. J., Mottram, R. E., Hurrion, P. D., & Vint, P. F. (2008). Comparison of kinematic sequence parameters between amateur and professional golfers. In D. Crews & R. Lutz (Eds.), *Science and Golf V: Proceedings of the World Scientific Congress of Golf* (pp. 30-36). Mesa, AZ: Energy in Motion.

**Key findings:**
- Defined the 4-segment kinematic sequence model
- Professional golfers: peak pelvic rotation velocity 477 ± 53 deg/s
- Amateur golfers: peak pelvic rotation velocity 395 ± 53 deg/s (p = 0.011)
- Proposed 1.5:1 transfer ratio between pelvis→thorax and thorax→arm
- 89% of professionals initiate pelvis rotation first in transition
- 60% of professionals show canonical Pelvis→Thorax→Arm→Club sequence

#### Cheetham - Simple model paper

**Citation:** Cheetham, P. J. (2014). A Simple Model of the Pelvis-Thorax Kinematic Sequence. Academia.edu publication.

**Key findings:**
- Simplified computational model for sequence extraction
- Acceleration and deceleration timing analysis
- Speed gain ratio quantification between segments

### 2.2 Methodological variations in literature

#### Angular velocity component selection

**Citation:** Brown, S. J., Selbie, W. S., & Wallace, E. S. (2013). The X-Factor: An evaluation of common methods used to analyse major inter-segment kinematics during the golf swing. *Journal of Sports Sciences*, 31(11), 1156-1163.

Different studies use different angular velocity components:
1. **Axial rotation only** - Rotation about vertical axis
2. **Total angular velocity magnitude** - √(ωx² + ωy² + ωz²)
3. **Resultant in swing plane** - Projected onto functional swing plane

The choice significantly affects peak timing identification.

#### Reference frame considerations

| Method | Description | Typical Values |
|--------|-------------|----------------|
| Global frame | Angular velocity relative to world | Most common |
| Local frame | Relative to parent segment | Used in some studies |
| Swing plane | Projected onto fitted plane | Less common |

### 2.3 Controversies and limitations

1. **Non-canonical sequences observed:** Not all skilled golfers exhibit Pelvis→Thorax→Arm→Club. Some research shows Pelvis→Arm→Thorax→Club patterns in competent golfers.

2. **Single peak assumption:** The method assumes a single dominant peak per segment; some swings show multiple peaks or plateaus.

3. **Phase definition sensitivity:** Results depend on how "transition" and "impact" are defined.

4. **Individual variation:** Significant inter-subject variability exists even among elite players.

---

## 3. Implementation Choice

### 3.1 Selected methodology

We implement the **Cheetham 4-segment model** with the following specifications:

1. **Segments:** Pelvis, Thorax, Lead Arm, Club
2. **Angular velocity:** Total magnitude (3D resultant)
3. **Reference frame:** Global (world) frame
4. **Peak detection:** Maximum value during downswing phase (transition to impact)

### 3.2 Rationale

- Most widely cited and validated in golf biomechanics literature
- Directly comparable to TPI (Titleist Performance Institute) normative data
- Consistent with commercial motion capture system outputs (GEARS, K-Vest)
- Clear operational definitions

### 3.3 Known limitations

1. Total angular velocity magnitude may obscure axial rotation timing
2. Global frame measurements affected by golfer orientation
3. Single-peak extraction may miss complex velocity profiles
4. Results depend on marker placement accuracy

### 3.4 Computational procedure

```
Algorithm: Kinematic Sequence Extraction

Input:
  - Segment angular velocities ω_pelvis(t), ω_thorax(t), ω_arm(t), ω_club(t)
  - Time array t
  - Phase markers: t_transition, t_impact

Procedure:
  1. Extract downswing window: t_transition ≤ t ≤ t_impact

  2. For each segment s ∈ {pelvis, thorax, arm, club}:
     a. Compute magnitude: |ω_s(t)| = √(ωx² + ωy² + ωz²)
     b. Find peak: ω_s_max = max(|ω_s(t)|)
     c. Find peak time: t_s_peak = argmax(|ω_s(t)|)

  3. Compute inter-segment lags:
     Δt_PT = t_thorax_peak - t_pelvis_peak
     Δt_TA = t_arm_peak - t_thorax_peak
     Δt_AC = t_club_peak - t_arm_peak

  4. Determine sequence order by sorting t_peak values

Output:
  - Peak velocities: {ω_pelvis_max, ω_thorax_max, ω_arm_max, ω_club_max}
  - Peak times: {t_pelvis_peak, t_thorax_peak, t_arm_peak, t_club_peak}
  - Lag values: {Δt_PT, Δt_TA, Δt_AC}
  - Sequence string: e.g., "Pelvis → Thorax → Arm → Club"
```

---

## 4. Validation Approach

### 4.1 Reference data

**TPI (Titleist Performance Institute) Tour Averages:**

| Segment | Peak Angular Velocity | Typical Timing |
|---------|----------------------|----------------|
| Pelvis | 450-550 deg/s | First (t = 0) |
| Thorax | 650-750 deg/s | +20-40 ms after pelvis |
| Lead Arm | 900-1100 deg/s | +30-50 ms after thorax |
| Club | 2000-2500 deg/s | +40-60 ms after arm |

### 4.2 Acceptance criteria

| Criterion | Tolerance | Method |
|-----------|-----------|--------|
| Peak velocity magnitude | ±5% | Compare to motion capture reference |
| Peak timing | ±1 ms | Cross-engine comparison |
| Sequence order | Exact match | Deterministic |

### 4.3 Cross-engine validation

All physics engines (MuJoCo, Drake, Pinocchio, OpenSim) must produce:
- Identical sequence ordering for the same input motion
- Peak times within ±1 ms
- Peak velocities within ±5%

---

## 5. Uncertainty Quantification

### 5.1 Sources of uncertainty

| Source | Magnitude | Notes |
|--------|-----------|-------|
| Marker tracking noise | ±1-3 mm | Propagates to velocity via differentiation |
| Temporal resolution | 1/sample_rate | Limits peak timing precision |
| Numerical differentiation | Varies | Depends on smoothing filter |
| Segment definition | ~5-10° | Marker placement variation |

### 5.2 Propagation method

Velocity uncertainty estimated via:
```
σ_ω ≈ σ_position / Δt × √2
```

Where σ_position is marker position uncertainty and Δt is timestep.

### 5.3 Reporting format

```
Peak Pelvis Angular Velocity: 487 ± 15 deg/s
  Timing: 0.000 s (reference)
  Method: Cheetham 4-segment model, global frame

Peak Thorax Angular Velocity: 712 ± 22 deg/s
  Timing: 0.032 ± 0.003 s after pelvis peak
  Lag (Δt_PT): 32 ms

Sequence: Pelvis → Thorax → Arm → Club (canonical)
```

---

## 6. References

1. Cheetham, P. J., Rose, G. A., Hinrichs, R. N., Neal, R. J., Mottram, R. E., Hurrion, P. D., & Vint, P. F. (2008). Comparison of kinematic sequence parameters between amateur and professional golfers. In D. Crews & R. Lutz (Eds.), *Science and Golf V* (pp. 30-36). Mesa, AZ: Energy in Motion.

2. Cheetham, P. J. (2014). A Simple Model of the Pelvis-Thorax Kinematic Sequence. [Technical report]

3. Brown, S. J., Selbie, W. S., & Wallace, E. S. (2013). The X-Factor: An evaluation of common methods used to analyse major inter-segment kinematics during the golf swing. *Journal of Sports Sciences*, 31(11), 1156-1163. https://pubmed.ncbi.nlm.nih.gov/23463985/

4. Horan, S. A., Evans, K., Morris, N. R., & Kavanagh, J. J. (2010). Thorax and pelvis kinematics during the downswing of male and female skilled golfers. *Journal of Biomechanics*, 43(8), 1456-1462.

5. Neal, R. J., & Wilson, B. D. (1985). 3D kinematics and kinetics of the golf swing. *International Journal of Sport Biomechanics*, 1(3), 221-232.

6. Tinmark, F., Hellström, J., Halvorsen, K., & Thorstensson, A. (2010). Elite golfers' kinematic sequence in full-swing and partial-swing shots. *Sports Biomechanics*, 9(4), 236-244. https://www.diva-portal.org/smash/get/diva2:25224/FULLTEXT01.pdf

---

## Approval

- [x] Technical accuracy verified
- [x] Literature coverage adequate
- [x] Implementation choice justified
- [x] Validation plan feasible
- [x] Approved for implementation

**Approved by:** Research Review Process **Date:** 2026-01-14
