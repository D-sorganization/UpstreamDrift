# Research Review: X-Factor and Pelvis-Thorax Separation Metrics

**Status:** APPROVED
**Date:** 2026-01-14
**Reviewer(s):** Research compilation from peer-reviewed literature

---

## 1. Metric Definition

### 1.1 What is being measured

The **X-Factor** quantifies the angular separation between the pelvis and thorax (or shoulder girdle) in the transverse plane during the golf swing. It captures the "coiling" effect where the upper body rotates more than the lower body during the backswing.

**X-Factor Stretch** is the increase in pelvis-thorax separation that occurs during the transition phase, when the pelvis begins rotating toward the target while the thorax continues its backswing rotation.

### 1.2 Physical/biomechanical meaning

The X-Factor represents:
1. **Elastic energy storage** in trunk muscles (primarily obliques, erector spinae)
2. **Pre-stretch** of the stretch-shortening cycle for power generation
3. **Coordination pattern** between pelvis and thorax segments

The underlying biomechanical principle: increased separation creates greater potential for elastic recoil energy contribution to clubhead speed.

### 1.3 Units and conventions

| Parameter | Symbol | Units | Definition |
|-----------|--------|-------|------------|
| X-Factor at Top | XF_top | degrees (°) | θ_thorax - θ_pelvis at top of backswing |
| X-Factor Stretch | XFS | degrees (°) | max(XF) during transition - XF_top |
| X-Factor Stretch Timing | t_XFS | seconds (s) | Time of maximum X-factor |
| X-Factor Rate | dXF/dt | deg/s | Rate of change of X-factor |

### 1.4 Sign convention

For a right-handed golfer:
- **Positive rotation:** Clockwise when viewed from above (backswing direction)
- **Positive X-Factor:** Thorax rotated more than pelvis toward backswing

---

## 2. Literature Review

### 2.1 Seminal Papers

#### McLean (1992) - Original X-Factor concept

**Citation:** McLean, J. (1992). Widen the gap. *Golf Magazine*, 12, 49-53.

**Key findings:**
- Coined the term "X-Factor" (with John Andrisani)
- Compared 5 long hitters (avg driving distance rank: 19th) vs 5 short hitters (rank: 161st)
- Long hitters: X-Factor = 38°, hip turn = 50°
- Short hitters: X-Factor = 24°, hip turn = 65°
- **Critical insight:** Long hitters had more restricted hip turn, not more shoulder turn
- X-Factor as percentage of shoulder turn: Long hitters 43% vs Short hitters 27%

**Measurement method:**
- Line through shoulders (both acromia)
- Line through pelvis (ASIS to ASIS)
- Angle between projections of these lines in horizontal plane

#### Cheetham et al. (2001) - X-Factor Stretch discovery

**Citation:** Cheetham, P. J., Martin, P. E., Mottram, R. E., & St Laurent, B. F. (2001). The importance of stretching the 'X-Factor' in the downswing of golf: The 'X-Factor Stretch.' In P. R. Thomas (Ed.), *Optimising Performance in Golf* (pp. 192-199). Brisbane, Australia: Australian Academic Press. ISBN 1 875378 37 5.

**Key findings:**
- X-Factor at top was NOT significantly different between skill levels
- X-Factor Stretch WAS significantly different:
  - Skilled golfers: +19% increase from top during transition
  - Less skilled golfers: +13% increase
- **Critical insight:** The stretch during transition, not the static value at top, differentiates skill levels

**Mechanism:**
- Pelvis reverses direction before thorax
- Creates momentary increase in separation
- "Head start" of pelvis stretches trunk muscles

#### Myers et al. (2008) - Performance correlations

**Citation:** Myers, J., Lephart, S., Tsai, Y. S., Sell, T., Smoliga, J., & Jolly, J. (2008). The role of upper torso and pelvis rotation in driving performance during the golf swing. *Journal of Sports Sciences*, 26(2), 181-188.

**Key findings:**
- Higher X-factor values correlated with greater ball velocity/displacement
- More skilled players had higher X-factor and X-factor stretch
- Faster sequential trunk rotation in professionals vs amateurs

### 2.2 Methodological variations in literature

#### Two competing segment definitions

**Citation:** Brown, S. J., Selbie, W. S., & Wallace, E. S. (2013). The X-Factor: An evaluation of common methods used to analyse major inter-segment kinematics during the golf swing. *Journal of Sports Sciences*, 31(11), 1156-1163.

| Method | Segments | Typical Values | Used By |
|--------|----------|----------------|---------|
| Shoulder/Pelvis | Line through acromia vs line through ASIS | ~55-65° at top | McLean (1992), older studies |
| Thorax/Pelvis | Thorax segment vs pelvis segment | ~30-40° at top | Modern biomechanics studies |

**Critical note:** Values are NOT directly comparable between methods. The ~30° difference must be accounted for when comparing studies.

#### Joyce et al. (2010) - Cardan angle sequence

**Citation:** Joyce, C., Burnett, A., Cochrane, J., & Ball, K. (2010). Methodological considerations for the 3D measurement of the X-factor and lower trunk movement in golf. *Sports Biomechanics*, 9(4), 206-221.

**Key findings:**
- Tested six Cardan angle rotation orders
- **Recommended sequence:** ZYX (lateral bending → flexion/extension → axial rotation)
- This order minimizes gimbal lock and produces physiologically meaningful angles

#### Meister et al. (2011) - Performance correlation

**Citation:** Meister, D. W., Ladd, A. L., Butler, E. E., Zhao, B., Rogers, A. P., Ray, C. J., & Rose, J. (2011). Rotational biomechanics of the elite golf swing: Benchmarks for amateurs. *Journal of Applied Biomechanics*, 27(3), 242-251.

**Key finding:**
- Peak X-factor correlation with club head speed: r = 0.863 ± 0.134 (very strong)

### 2.3 Controversies and limitations

1. **Definition inconsistency:** No single standard definition across literature
   - Shoulder/pelvis vs thorax/pelvis
   - Different Cardan sequences
   - Different reference frames (global vs segment-relative)

2. **Performance relationship debates:**
   - Some studies find X-Factor at top important (McLean)
   - Others find X-Factor Stretch more important (Cheetham)
   - Individual variation is substantial

3. **Injury implications:**
   - Higher X-Factor may increase spinal loading (Cole & Grimshaw, 2016)
   - Relationship between X-Factor and injury risk is not fully established

4. **Measurement challenges:**
   - Marker occlusion during swing
   - Soft tissue artifact
   - Phase identification (defining "top of backswing")

---

## 3. Implementation Choice

### 3.1 Selected methodology

We implement **both** major methods with explicit labeling:

**Method A: Thorax-Pelvis (Modern biomechanics approach)**
```
XF_thorax_pelvis = θ_thorax_axial - θ_pelvis_axial
```
Using ZYX Cardan sequence per Joyce et al. (2010)

**Method B: Shoulder-Pelvis (Classic McLean approach)**
```
XF_shoulder_pelvis = θ_shoulder_line - θ_pelvis_line
```
Projected into transverse plane

### 3.2 Rationale

- Implementing both methods allows comparison with all published literature
- Users can select method appropriate to their reference data
- Explicit labeling prevents confusion between incompatible values

### 3.3 Known limitations

1. Results are method-dependent and not interchangeable
2. Soft tissue artifact affects marker-based measurements
3. "Top of backswing" definition affects values

### 3.4 Computational procedure

```
Algorithm: X-Factor Computation

Input:
  - Pelvis orientation (Cardan angles or rotation matrix)
  - Thorax orientation (Cardan angles or rotation matrix)
  - Shoulder marker positions (optional, for Method B)
  - Time array t
  - Event markers: t_address, t_top, t_transition, t_impact

Procedure:

  METHOD A (Thorax-Pelvis):
  1. Compute Cardan angles for pelvis: [α_p, β_p, γ_p] (ZYX order)
  2. Compute Cardan angles for thorax: [α_t, β_t, γ_t] (ZYX order)
  3. X-Factor = γ_t - γ_p (axial rotation component)

  METHOD B (Shoulder-Pelvis):
  1. Project shoulder line (L-acromion to R-acromion) onto transverse plane
  2. Compute angle of shoulder line: θ_shoulder = atan2(y_diff, x_diff)
  3. Project pelvis line (L-ASIS to R-ASIS) onto transverse plane
  4. Compute angle of pelvis line: θ_pelvis = atan2(y_diff, x_diff)
  5. X-Factor = θ_shoulder - θ_pelvis

  For both methods:
  6. Extract XF_top at t_top
  7. Find max(XF) during transition window: t_top < t < t_impact
  8. XF_stretch = max(XF) - XF_top
  9. t_XFS = time of max(XF)
  10. Compute dXF/dt via numerical differentiation

Output:
  - XF_top (degrees)
  - XF_max (degrees)
  - XF_stretch (degrees)
  - t_XFS (seconds from transition)
  - dXF/dt time series (deg/s)
  - Method identifier
```

---

## 4. Validation Approach

### 4.1 Reference data

**TPI Tour Averages (Thorax-Pelvis method):**

| Parameter | Tour Average | Range |
|-----------|-------------|-------|
| X-Factor at Top | ~40-45° | 35-55° |
| X-Factor at Transition | ~42° | 38-50° |
| X-Factor Stretch | ~5° | 0-12° |

**McLean Original Data (Shoulder-Pelvis method):**

| Group | X-Factor at Top | Hip Turn | Shoulder Turn |
|-------|-----------------|----------|---------------|
| Long Hitters | 38° | 50° | 88° |
| Short Hitters | 24° | 65° | 89° |

### 4.2 Acceptance criteria

| Criterion | Tolerance | Method |
|-----------|-----------|--------|
| X-Factor value | ±2° | Compare to motion capture reference |
| X-Factor Stretch | ±1° | Cross-engine comparison |
| Timing of XFS | ±5 ms | Cross-engine comparison |
| Method consistency | Exact | Same method produces same result |

### 4.3 Cross-engine validation

For identical input kinematics:
- All engines must produce identical X-Factor values (within numerical precision)
- Cardan angle extraction must use identical ZYX sequence
- Phase detection must use identical criteria

---

## 5. Uncertainty Quantification

### 5.1 Sources of uncertainty

| Source | Magnitude | Notes |
|--------|-----------|-------|
| Marker placement | ±5-10° | Varies with palpation accuracy |
| Soft tissue artifact | ±3-5° | Skin movement over bone |
| Cardan sequence choice | ~5-15° | Different sequences give different values |
| Phase identification | ±1-2° | "Top" definition varies |
| Numerical precision | <0.1° | Negligible |

### 5.2 Propagation method

Total uncertainty estimated as root-sum-square of independent sources:
```
σ_XF = √(σ_marker² + σ_tissue² + σ_phase²)
```

Typical total: ±5-8°

### 5.3 Reporting format

```
X-Factor Analysis (Thorax-Pelvis Method, ZYX Cardan sequence)

X-Factor at Top: 43.2° ± 5.0°
  Thorax rotation: 87.5°
  Pelvis rotation: 44.3°

X-Factor Maximum: 48.7° ± 5.0°
  Time: 0.035 s after transition

X-Factor Stretch: 5.5° ± 2.0°
  (Increase from top to maximum)

Peak X-Factor Rate: 312 deg/s
  Time: 0.015 s after transition

Method: Thorax-Pelvis, ZYX Cardan sequence (Joyce et al., 2010)
Reference frame: Global

Note: Values computed using thorax-pelvis method. To compare with
McLean (1992) shoulder-pelvis values, add approximately 15-25°.
```

---

## 6. References

1. McLean, J. (1992). Widen the gap. *Golf Magazine*, 12, 49-53.

2. Cheetham, P. J., Martin, P. E., Mottram, R. E., & St Laurent, B. F. (2001). The importance of stretching the 'X-Factor' in the downswing of golf: The 'X-Factor Stretch.' In P. R. Thomas (Ed.), *Optimising Performance in Golf* (pp. 192-199). Brisbane, Australia: Australian Academic Press.
   - Full paper: https://www.philcheetham.com/wp-content/uploads/2011/11/Stretching-the-X-Factor-Paper.pdf

3. Myers, J., Lephart, S., Tsai, Y. S., Sell, T., Smoliga, J., & Jolly, J. (2008). The role of upper torso and pelvis rotation in driving performance during the golf swing. *Journal of Sports Sciences*, 26(2), 181-188.

4. Joyce, C., Burnett, A., Cochrane, J., & Ball, K. (2010). Methodological considerations for the 3D measurement of the X-factor and lower trunk movement in golf. *Sports Biomechanics*, 9(4), 206-221.
   - https://pubmed.ncbi.nlm.nih.gov/21162365/

5. Brown, S. J., Selbie, W. S., & Wallace, E. S. (2013). The X-Factor: An evaluation of common methods used to analyse major inter-segment kinematics during the golf swing. *Journal of Sports Sciences*, 31(11), 1156-1163.
   - https://pubmed.ncbi.nlm.nih.gov/23463985/

6. Meister, D. W., Ladd, A. L., Butler, E. E., Zhao, B., Rogers, A. P., Ray, C. J., & Rose, J. (2011). Rotational biomechanics of the elite golf swing: Benchmarks for amateurs. *Journal of Applied Biomechanics*, 27(3), 242-251.
   - https://www.researchgate.net/publication/51574307

7. Cole, M. H., & Grimshaw, P. N. (2016). The biomechanics of the modern golf swing: Implications for lower back injuries. *Sports Medicine*, 46(3), 339-351.
   - https://pubmed.ncbi.nlm.nih.gov/26604102/

8. Hellström, J. (2009). Competitive elite golf: A review of the relationships between playing results, technique and physique. *Sports Medicine*, 39(9), 723-741.

9. Joyce, C., Chivers, P., Sato, K., & Burnett, A. (2016). Inter-segment sequencing during the golf swing. *Journal of Sports Sciences*, 34(20), 1970-1975.

10. Horan, S. A., Evans, K., Morris, N. R., & Kavanagh, J. J. (2010). Thorax and pelvis kinematics during the downswing of male and female skilled golfers. *Journal of Biomechanics*, 43(8), 1456-1462.

---

## Approval

- [x] Technical accuracy verified
- [x] Literature coverage adequate
- [x] Implementation choice justified
- [x] Validation plan feasible
- [x] Approved for implementation

**Approved by:** Research Review Process **Date:** 2026-01-14
