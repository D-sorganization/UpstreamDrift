# Run73 Sampled Rotation-Chart Audit

The existing SerialRotationChart provider evaluated its rate-map condition number
at every saved run73 capture sample. Maximum values are2.680 for the hip,17.886
for the left shoulder and2.889 for the right shoulder. The left-shoulder maximum
occurs at0.8333333333 s. The0.6 s interior node is well away from the provider's
inverse-map rejection threshold in all three groups.

These sampled coordinate maps do not by themselves explain the7.45e7 full
marker-Jacobian condition measured in audit74. Quaternion coordinates cannot be
assumed to remove that complete trajectory/control conditioning. Between-sample
extrema, other joints, mass-matrix conditioning and full dynamics are not bounded
by this audit. No representation or physics change was made.

Source, state-file and specification hashes, exact sample times and all condition
values are retained in report.json. The original model bytes are read from the
run73 archive; the local state NPZ must match the archived bytes exactly. Both
raw and canonical physical fingerprints are retained, and native coordinate
inventory is checked against the saved states.

Reproduce from the worktree root:

```powershell
python -m docs.development.simscape_tour_matching.native_evidence.returned73_chart_audit.audit_charts
```
