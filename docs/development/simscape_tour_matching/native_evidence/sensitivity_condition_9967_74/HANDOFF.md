# Saved Jacobian Conditioning Audit 74

This offline audit consumes the exact run72 archive; it performs no additional
dynamics solve and changes no fit parameters. The 81-column marker Jacobian is
scaled by the runner's amplitude10 and includes its terminal residual weight10.
Valid capture masks are preserved. Effort-penalty rows and bounds are excluded.

The resulting23100-by-81 matrix has condition number **7.452e7**. Normalizing
each column to unit norm still gives **3.767e7**. Its numerical rank is81 under
the recorded floating-point SVD threshold; this is not a claim that all weak
directions are reliably identified. The strongest column norms include HipInputX,
SpineInputY and SpineInputX B4. Source/archive hashes, every singular value and
column norm are retained in report.json.

The result supports investigating correlated control sensitivities as well as
parameter scaling. Column scaling alone does not remove the measured conditioning.
It does not prove why a nonlinear optimization stalls, certify all derivatives,
or establish a full-swing match. Let bounded fit73 finish before selecting another
experiment. If progress remains small, compare actual versus predicted reduction
and qualify a full mixed-control directional derivative before increasing compute.
Then revisit short-window multiple shooting using integrated, closure-consistent
nodes and continuous global-sixth-order controls; avoid reintroducing the previous
tight static-pose node boxes. All final acceptance remains original-state replay.

Reproduce from the worktree root:

```powershell
python -m docs.development.simscape_tour_matching.native_evidence.sensitivity_condition_9967_74.audit_condition
```

NumPy2.4.1 was used locally. This diagnostic is independent of the active remote
fit and does not require a native engine runtime.
