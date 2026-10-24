# Weakness Catalog — 2026-10-23

**Repository:** D-sorganization/UpstreamDrift  
**Critic seat:** Bravo (Fleet Critic)  
**Review window:** Recent physical grip-contact implementation and its
documented validation status (HRG-GS-M12, issue #13334).

---

## Weakness 1: Contact Parameters Lack Capture-Specific Calibration Evidence

### Summary of Concern

The fixture labels its contact law physical and uses a default squeeze of
1100 N per hand plus fixed normal stiffness, damping, friction, and patch
parameters. The report describes how the parameters are generated and shared
between implementations, but does not identify measured hand–grip data,
material tests, parameter fitting, or uncertainty bounds that justify these
values for the modeled capture. Agreement between generated code and its
Python source cannot validate the parameter values.

### Location

- **File/Document:** `src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/exploratory_gs3dx/docs/GRIP_CONTACT.md`
- **Section:** “Canonical Python-to-MATLAB Codegen Kernel” and “R2025b Measured Values”
- **Claim or Equation:** Parameters include `F_sq = 1100 N per hand`; generated values include `normal_stiffness_n_m = 171095.008...`, `damping_s_m = 0.3827`, `static_friction = 0.9`, and `dynamic_friction = 0.7`.
- **Source:** `src/shared/python/grip_contact/matlab_kernel.py:155–202`; generated values in `tools/generated/gs3dx_grip_contact_params.json`.

### Nature of the Issue

- [ ] Logical gap
- [x] Unstated assumption
- [ ] Overgeneralization
- [x] Empirical insufficiency
- [ ] Terminological ambiguity
- [ ] Literature conflict

### Why This Is a Problem

The normal load sets the available frictional wrench, and stiffness and damping
shape the transient contact response. Consequently these choices can drive the
club-wrench parity result. Without an identified calibration target, the
reported parity threshold could be met or missed because of selected parameters
rather than because the contact model represents the golfer. The stated
1100 N-per-hand value especially needs an operational definition: whether it is
an integrated squeeze load, a per-contact normal load, or a model tuning input.

Golf-specific studies measure lead- and trail-hand grip forces and their
variation during swings, demonstrating that these loads are observable and
phase-dependent; the fixture should connect its load convention and parameter
fit to such a measurement basis. See [Komi, Roberts, and Rothberg (2008)](https://journals.sagepub.com/doi/abs/10.1243/17543371jset9) and [Kobayashi et al. (2016)](https://www.jstage.jst.go.jp/article/kikaic1979/72/714/72_714_561/_article/-char/en).

### Evidence / References

- `GRIP_CONTACT.md` says only the constitutive golden test has been run; all
  Simscape build, parity, convergence, save/reload, and force-sign checks are
  pending.
- The implementation default is `DEFAULT_SQUEEZE_PER_HAND_N = 1100.0` in
  `matlab_kernel.py`; this is a configured input, not an observed subject value.
- [Measurement and Analysis of Grip Force During a Golf Shot](https://journals.sagepub.com/doi/abs/10.1243/17543371jset9) measures grip force during a golf shot.
- [Development of a Grip Force Sensor in Golf Swing](https://www.jstage.jst.go.jp/article/kikaic1979/72/714/72_714_561/_article/-char/en) describes a golf-swing grip sensor and hand-specific roles.

### Severity

- [ ] Low (clarification needed)
- [ ] Medium (argument tightening required)
- [x] High (core claim at risk)

### Suggested Remedies

- State the mechanical meaning and derivation of `squeeze_per_hand_n`, with
  units and how it maps to each modeled contact pair.
- Fit or bound the parameters against instrumented lead/trail hand forces or
  another capture-specific calibration dataset; report uncertainty and held-out
  residuals.
- Until calibration exists, describe the fixture as a parameterized contact
  prototype and avoid implying athlete-specific physical validation.

---

## Weakness 2: Prescribed-Hand Fixture Does Not Establish Reciprocal Hand Dynamics

### Summary of Concern

The fixture drives both hand poses and twists as prescribed trajectories and
connects the generated contact wrench to the club's external wrench block. The
kernel also computes hand-side wrenches, but the model description identifies
club wrench injection and does not describe those hand-side outputs being
applied to dynamic hand bodies. This makes the fixture useful for evaluating a
club under imposed hand motion, but it does not establish that contact loads
feed back into the hands or golfer model.

### Location

- **File/Document:** `src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/exploratory_gs3dx/docs/GRIP_CONTACT.md`
- **Section:** “Fixture Model Architecture”
- **Claim or Equation:** Prescribed `Hand_L`, `Hand_R` world poses and twists on
  Bushing Joint blocks; world-frame wrench applied to the club origin.
- **Source:** `src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/exploratory_gs3dx/tools/gs3dx_build_grip_contact.m:22–24, 97–100, 450–465`.

### Nature of the Issue

- [x] Logical gap
- [x] Unstated assumption
- [ ] Overgeneralization
- [x] Empirical insufficiency
- [ ] Terminological ambiguity
- [ ] Literature conflict

### Why This Is a Problem

The externally prescribed hand trajectories can supply or absorb whatever
reaction is needed to enforce those motions. Club motion under those inputs is
not evidence that the modeled hand–arm–torso chain would produce the same motion
under reciprocal contact loads. If results are later described as golfer–club
coupled dynamics, the missing reaction pathway becomes a load-bearing
assumption.

### Evidence / References

- `GRIP_CONTACT.md` explicitly separates the fixture from `GS3DX_Human` and
  states the hands are prescribed; it also reports pending Pinocchio net-wrench
  parity.
- The model builder wires `F_club_w` and `M_club_w` to `Club External Wrench`
  and logs `F_hand_L_w`, `M_hand_L_w`, `F_hand_R_w`, and `M_hand_R_w` as signals.
- [Three Dimensional Upper Limb Joint Kinetics of a Golf Swing with Measured Internal Grip Force](https://pmc.ncbi.nlm.nih.gov/articles/PMC7374515/) connects measured internal grip forces and torques with upper-limb kinetics.

### Severity

- [ ] Low (clarification needed)
- [x] Medium (argument tightening required)
- [ ] High (core claim at risk)

### Suggested Remedies

- Limit conclusions to club response under prescribed hand motion and avoid
  inferring coupled golfer dynamics from this fixture.
- If coupled dynamics are a target, apply equal-and-opposite contact wrenches
  to modeled hand bodies and validate the reaction pathway against the native
  model or measured upper-limb kinetics.
- Add a momentum/wrench-balance check over each contact pair and state how
  prescribed-motion actuators contribute reactions.

---

## Weakness 3: Golden-Vector Parity Is Implementation Verification, Not Physical Validation

### Summary of Concern

The 51 golden vectors are generated from the Python reference and checked
against generated MATLAB code. This tests code generation and numerical parity
for selected states. Since both implementations encode the same law and the
vectors are produced from that law, this does not test whether the law predicts
real hand–club contact, including its friction, compliance, contact geometry,
or loading history.

### Location

- **File/Document:** `src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/exploratory_gs3dx/docs/GRIP_CONTACT.md`
- **Section:** “Constitutive Golden Vectors”
- **Claim:** 51 cases “verified in MATLAB R2025b” to stated numerical tolerances.
- **Source:** `.../exploratory_gs3dx/tests/fixtures/gs3dx_grip_pair_golden.json` and `tests/test_gs3dx_grip_pair_wrench.m`.

### Nature of the Issue

- [x] Logical gap
- [ ] Unstated assumption
- [ ] Overgeneralization
- [x] Empirical insufficiency
- [x] Terminological ambiguity
- [ ] Literature conflict

### Why This Is a Problem

Small numerical error against a reference implementation can create a strong
impression of validated physical accuracy. Here it answers whether MATLAB
faithfully reproduces the Python equations, not whether either implementation
matches measured contact forces. Likewise, solver convergence or save/reload
repeatability would establish numerical stability and reproducibility, not
empirical validity. Those are separately identified as pending, which should
remain clear in any later summary of results.

### Evidence / References

- The report states the golden vectors were generated directly from the Python
  reference and that only this constitutive test has been measured in R2025b.
- The report's separate Pinocchio net-wrench parity, convergence, and force-sign
  rows remain pending; even those checks require an empirical target to support
  physical-validity claims.
- Experimental validation of Hunt–Crossley-type contact models is treated as a
  comparison against measured outcomes, not merely a second implementation:
  [Validation of Nonlinear Viscoelastic Contact Force Models for Low Speed Impact](https://www.ndl.iitkgp.ac.in/re_document/asme_dl/1234567_asme_journal_JAMCAV/11674).

### Severity

- [ ] Low (clarification needed)
- [x] Medium (argument tightening required)
- [ ] High (core claim at risk)

### Suggested Remedies

- Call this result code-generation or constitutive implementation parity, not
  physical validation.
- Keep numerical acceptance and physical acceptance as distinct gates in the
  report and downstream model cards.
- Add a blinded or held-out comparison to instrumented grip-wrench data across
  multiple swing phases, with force and moment errors, sensor uncertainty, and
  parameter provenance reported.
