# Independent Numerical Preflight

These deterministic checks assess numerical consistency and bounded nominal
feasibility. They do not establish empirical ball/club accuracy and do not sample
the final Monte Carlo population.

Run from the repository root, after freezing the scientific sources:

```bash
export QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
PYTHONPATH=. /tmp/shot-pattern-validation-venv-20261009/bin/python docs/research/shot_pattern_analysis/phase2/validation/independent_contact_audit.py > docs/research/shot_pattern_analysis/phase2/validation/independent_contact_audit.json
PYTHONPATH=. /tmp/shot-pattern-validation-venv-20261009/bin/python docs/research/shot_pattern_analysis/phase2/validation/independent_preflight.py --output docs/research/shot_pattern_analysis/phase2/validation/independent_preflight.json
PYTHONPATH=. /tmp/shot-pattern-validation-venv-20261009/bin/python docs/research/shot_pattern_analysis/phase2/validation/independent_head_sensitivity.py --output docs/research/shot_pattern_analysis/phase2/validation/independent_head_sensitivity.json
```

The named environment contains the native flight kernel and Numba. Another
runtime must reproduce and record those dependencies; do not assume the path
exists on another machine. Each receipt records scientific source hashes and
whether those sources remained unchanged throughout execution.

The landing grid covers three clubs, lean 0/5/10°, twist −4/0/+4°, and the nine
combinations of horizontal/vertical strike offsets −12.7/0/+12.7 mm (two component
standard deviations). Contact refinement uses 128/256/512 steps at flight dt
0.01 s; flight refinement uses dt 0.02/0.01/0.005 s at 512 contact steps. The
predeclared fine-change acceptance is 0.05 m Euclidean landing displacement.
The 128→512 contact change is also reported to judge the proposed primary setting.
These extremal checks do not bound every Gaussian tail.

Ninety nominal matching cases cover five face-to-path values, each club/lean, and
both within-lean and across-lean targets. Speed bounds remain ±25% of the preset;
face bounds remain ±3° around the nominal reference. Fine replay reports both
optimizer residual and numerical endpoint change. Unsupported domain states,
bounded unreachable targets, and numerical failures are distinct. No case is
resampled or silently omitted. Matching feasibility is conditional on the stated
bounded numerical search; an optimizer failure is not a proof of global physical
unreachability.

## Head-Assumption Sensitivity

The separate head script holds nominal delivery fixed and changes one assumption
at a time: CG depth ±20%, all principal inertias ±20%, driver bulge/roll radii
together ±20%, and the Newton-QP contact-law comparator. It covers center and
four cardinal 12.7 mm strike offsets at all three club/lean combinations. It
records raw spin/launch/endpoints, differences, unsupported states, and source
provenance. These perturbations are not empirical parameter ranges or confidence
intervals. No delivery is rematched after a parameter change.
