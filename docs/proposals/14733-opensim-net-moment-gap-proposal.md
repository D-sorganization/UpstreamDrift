# Proposal: #13560 OpenSim Net-Moment Gap — Recorder Inconsistency Finding, Conditioning Math, and Options

**Proposal / write-up only.** Refs #13560, #14711.

This issue changes no code, no test, no tolerance and no xfail. The net-wrench gate stays in force exactly as it is, and #14711 stays unarmed. Nothing here takes effect unless Dieter approves a follow-up separately. Per the owner's ruling, per-hand values remain diagnostics only. All numbers come from local diagnostic runs (WSL `osim2`, OpenSim 4.6, Pinocchio reference, 0.04 s horizon, 21 samples at 2 ms). The scripts are not committed.

---

## Executive Summary

1. **Severe cancellation conditioning:** The net moment about the club center of mass (COM) is a small residual resulting from the cancellation of much larger per-hand terms. The gate tolerance is approximately 0.3% of the individual hand moments; consequently, a contact force-law relative error of only ~0.049% (0.035% worst-case) is sufficient to fail the gate (Section 1).
2. **Finding A (Recorder inconsistency):** In the reference engine, recorded `net_moment_com` matches the club's own motion-implied rate of change of angular momentum $d\mathbf{h}/dt$ to 0.0044–0.0046 N·m RMS across all six cases. In contrast, OpenSim's recorded `net_moment_com` misses its own integrated club motion by 0.0195–0.0368 N·m RMS (7.4%–11.7% of peak). The motion-implied moments between the two engines agree to 0.0024–0.0086 N·m (0.7%–2.2%). The gate gap is therefore primarily a recording or bookkeeping inconsistency inside OpenSim's contact wrench assembly, rather than fundamentally divergent contact physics (Section 2).
3. **Option B (Recorder fix):** A targeted recorder fix within OpenSim's contact wrench assembly could allow OpenSim to pass the UNCHANGED gate tolerances without altering the contact law or weakening verification criteria. Because this touches the gated path, any implementation requires an explicit supervisor ruling and Dieter's prior approval (Section 3).
4. **Diagnostic-only consistency metric:** A dynamic-consistency check measuring internal $d\mathbf{h}/dt$ agreement is formulated as an additional diagnostic metric only, never as a replacement or relaxation of the primary net-wrench gate (Section 4).

---

## 1. Conditioning Math

The primary net-wrench gate quantity is the net moment about the club COM:
$$\mathbf{M} = \sum_{h \in \{L, R\}} \mathbf{m}_h, \quad \text{where} \quad \mathbf{m}_h = \boldsymbol{\tau}_h + \mathbf{r}_h \times \mathbf{F}_h$$
implemented in `GripKineticsSeries.net_moment_com_nm`. Normalized error is evaluated as:
$$\text{error} = \frac{\text{RMS}(\mathbf{M}_{\text{cand}} - \mathbf{M}_{\text{ref}})}{\max \|\mathbf{M}_{\text{ref}}\|}$$
checked against thresholds of 2.0% RMS and 5.0% peak via `compute_quantity_error`.

### Measured Magnitudes (Reference Run, Peak Over 0.04 s Horizon)

| Case             | Peak $\|\mathbf{m}_L\| / \|\mathbf{m}_R\|$ (N·m) | Peak $\|\mathbf{M}\|$ (N·m) | Cancellation Ratio $(\|\mathbf{m}_L\| + \|\mathbf{m}_R\|) / \|\mathbf{M}\|$ | 2.0% RMS Tolerance (N·m) |
| ---------------- | ------------------------------------------------ | --------------------------- | --------------------------------------------------------------------------- | ------------------------ |
| overlap-swing    | 13.5 / 13.4                                      | 0.470                       | 57                                                                          | 0.0094                   |
| ten_finger-swing | 12.6 / 12.5                                      | 0.473                       | 53                                                                          | 0.0095                   |
| overlap-hold     | 15.9 / 15.9                                      | 0.179                       | 177                                                                         | 0.0036                   |
| ten_finger-hold  | 16.1 / 16.1                                      | 0.160                       | 201                                                                         | 0.0032                   |

### Error Sensitivity Derivation

If each hand's moment carries an independent relative error $\varepsilon$, the net moment error is:
$$\sigma_{\text{net}} \approx \varepsilon \sqrt{\|\mathbf{m}_L\|^2 + \|\mathbf{m}_R\|^2}$$
In the worst-case scenario with fully correlated errors:
$$\sigma_{\text{worst}} = \varepsilon (\|\mathbf{m}_L\| + \|\mathbf{m}_R\|)$$

For `overlap-swing`:
$$\sqrt{13.5^2 + 13.4^2} = 19.0\text{ N}\cdot\text{m}$$
To fail the 2.0% RMS tolerance ($0.0094\text{ N}\cdot\text{m}$):
$$\varepsilon \ge \frac{0.0094}{19.0} \approx \mathbf{0.049\%} \quad (0.035\% \text{ worst-case})$$
For hold cases with cancellation ratios $\approx 180\text{--}200$, the failure threshold drops to approximately **0.014%–0.016%**.

In the measured overlap-swing run:
$$\text{RMS}\|\mathbf{M}_{\text{osim}} - \mathbf{M}_{\text{ref}}\| = 0.0365\text{ N}\cdot\text{m} \implies \varepsilon \approx 0.19\%$$
The individual hand $\mathbf{r} \times \mathbf{F}$ terms differ by $\approx 3\text{ N}\cdot\text{m}$ RMS per hand (25% of peak), but these differences are almost perfectly opposite and cancel in the summation. The left/right force allocation of an internal squeeze on a rigid cylinder is statically indeterminate; this under-identifiability is the foundational reason the gate evaluates the total net wrench.

---

## 2. Finding A: OpenSim Recorded Net Moment Inconsistent With Integrated Motion

### Methodology: Angular Momentum Differential Verification

To test dynamic self-consistency for each engine, we compute the club angular momentum about its COM and differentiate it with respect to time:

1. **Club orientation recovery:**
   $$\mathbf{R}_{\text{club}} = \mathbf{R}_{\text{grip}L} \mathbf{M}_L^T$$
   where $\mathbf{R}_{\text{grip}L}$ is the recorded orientation of the left club-side grip frame and $\mathbf{M}_L = \text{interface.frame("L").matrix()}[:3, :3]$.
2. **Angular momentum calculation:**
   $$\mathbf{h} = \mathbf{R}_{\text{club}} \mathbf{I}_c \mathbf{R}_{\text{club}}^T \boldsymbol{\omega}$$
   using club inertia $\mathbf{I}_c$ from `ClubDynamics.from_spec` and recorded club angular velocity $\boldsymbol{\omega}$.
3. **Time derivative:**
   $$\mathbf{N} = \frac{d\mathbf{h}}{dt}$$
   evaluated via central difference across the 2 ms sampling grid.
4. **Self-consistency residual:**
   $$\text{Residual} = \text{RMS}(\mathbf{M}_{\text{recorded}} - \mathbf{N})$$
   evaluated over the interior trajectory samples.

### Verification of Input Parity (D56)

All physical and kinetic inputs are bit-equal or verified to match:

- Club mass ($0.313\text{ kg}$), COM, inertia matrix, and gravity are bit-equal between OpenSim and the Pinocchio reference.
- Initial club angular velocity $\boldsymbol{\omega}_0$ and linear velocity $\mathbf{v}_0$ match to $10^{-15}$.
- Temporal sampling grids are identical ($2\text{ ms}$ intervals over $0.04\text{ s}$).
- Prescribed hand pose and velocity series match to $10^{-9}\text{ m}$.

### Quantitative Results Across All Six Grip Cases

Residuals in $\text{N}\cdot\text{m}$ RMS and expressed as percentage of reference net-moment peak:

| Case             | Current Gate Error (Peak / RMS) | Gate Result | Reference Self-Residual (N·m RMS) | OpenSim Self-Residual (N·m RMS) | Cross-Engine Motion-Implied Diff (N·m RMS) |
| ---------------- | ------------------------------- | ----------- | --------------------------------- | ------------------------------- | ------------------------------------------ |
| overlap-hold     | 12.6% / 9.8%                    | FAIL        | 0.0044 (2.5%)                     | 0.0195 (10.9%)                  | 0.0039 (2.2%)                              |
| overlap-swing    | 6.6% / 7.8%                     | FAIL        | 0.0046 (1.0%)                     | 0.0349 (7.4%)                   | 0.0086 (1.8%)                              |
| interlock-hold   | 12.6% / 9.8%                    | FAIL        | 0.0044 (2.5%)                     | 0.0195 (10.9%)                  | 0.0039 (2.2%)                              |
| interlock-swing  | 6.6% / 7.8%                     | FAIL        | 0.0046 (1.0%)                     | 0.0349 (7.4%)                   | 0.0086 (1.8%)                              |
| ten_finger-hold  | 10.7% / 11.7%                   | FAIL        | 0.0046 (2.9%)                     | 0.0197 (12.3%)                  | 0.0042 (2.6%)                              |
| ten_finger-swing | 6.9% / 8.4%                     | FAIL        | 0.0046 (1.0%)                     | 0.0368 (7.8%)                   | 0.0024 (0.5%)                              |

### Analytical Conclusion From Finding A

- The reference engine's recorded net moment tracks its own club angular momentum derivative to $0.0044\text{--}0.0046\text{ N}\cdot\text{m}$ RMS (1.0% in swing cases).
- OpenSim's recorded net moment diverges from its own club angular momentum derivative by $0.0195\text{--}0.0368\text{ N}\cdot\text{m}$ RMS (7.4%–12.3% of peak).
- Crucially, the **motion-implied moments derived from the two engines' actual club trajectories agree within 0.0024–0.0086 N·m (0.5%–2.2%)**.
- Therefore, the physical trajectories integrated by OpenSim match the reference within or near tolerance; the gate failure is predominantly caused by an inconsistency in how OpenSim reports/records its contact wrenches.

### Ruled-Out Hypotheses (D54–D56)

Extensive prior diagnostic sweeps established that the discrepancy is not caused by:

- Winkler quadratic vs. linear curvature models (accounts for $< 4\%$ of the observed gap).
- Temporal shift or latency (cross-correlation confirms no phase offset).
- OpenSim hunt-crossley dissipation parameters ($0\text{--}20\text{ s/m}$).
- Grip contact mesh discretization density (finer meshes do not resolve the residual).
- Rigid body mass properties or initial condition discrepancies.

---

## 3. Option B: Recorder Fix, Gate and Tolerances Unchanged

Because the underlying motion trajectories agree, the candidate remedy lies in OpenSim's contact wrench recording pipeline rather than the constitutive contact equations:

- `ContactGripSimulator._assemble` and the force-reporter column mapping in `src/shared/python/engines/opensim/python/grip_contact_osim_sim.py`, which feed into `PairTraceRecorder`.
- **Diagnostic next step:** At each recorded time sample, compare the reporter's assembled body wrench directly against the generalized wrench implied by the integrated acceleration state via Simbody's `realizeAcceleration`.
- **Governance Gate:** Modifying `grip_contact_osim_sim.py` or `PairTraceRecorder` alters the execution path evaluated by the gate. A recorder-fix PR under #13560 strictly requires a supervisor ruling and Dieter's explicit written approval before initiation.

---

## 4. Dynamic-Consistency Check: Additional Diagnostic Only (Not a Replacement Gate)

A dynamic-consistency metric can be formalized as:
$$\text{Consistency} = \frac{\text{RMS}(\mathbf{M}_{\text{recorded}} - d\mathbf{h}/dt)}{\max \|\mathbf{M}_{\text{ref}}\|}$$

- This metric is described strictly as an additional diagnostic only, never as a replacement gate.
- It must **never replace, relax, or substitute for the existing net-wrench gate**.
- A discrete differentiation floor of 1.0%–2.9% exists due to numerical differencing over 2 ms samples.

---

## 5. Risks and Technical Limitations

1. **Estimator noise floor:** Numerical differentiation of angular momentum using central differences over finite 2 ms sample spacing introduces an estimator noise floor of 1%–3% of peak, preventing this check from independently certifying sub-2% tolerances without finer sampling.
2. **Blindness to coupled errors:** A self-consistency check verifies whether an engine's reported wrench matches its own acceleration; it does not catch errors where both motion and recorded wrench are erroneous in identical proportions.
3. **Scope of evidence:** Current evidence is derived from 0.04 s forward integration horizons under OpenSim 4.6 in the local WSL `osim2` environment. Behavior under continuous multi-second swing simulations must be separately verified.
4. **Gate integrity:** The existing net-moment gate remains unchanged and active.

---

## 6. Reproduction Record

- **Environment:** WSL `osim2` conda environment with OpenSim 4.6 and Pinocchio reference backends.
- **Horizon & Resolution:** 0.04 s forward integration, 21 equidistant evaluation samples at 2 ms intervals.
- **Reference Code:** `src/shared/python/engines/pinocchio/` and `src/shared/python/engines/opensim/python/grip_contact_osim_sim.py`.
