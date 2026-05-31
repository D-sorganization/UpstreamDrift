# JaxSim Version Pin Policy

JaxSim is optional in UpstreamDrift and is used for differentiable swing checks.
The dependency is pinned exactly because upstream 0.9.x APIs can change without
notice. Treat each bump as an engine upgrade, not a routine dependency refresh.

## Current Pin

- Package: `jaxsim`
- Pin: `jaxsim==0.9.0`
- Declared in: `pyproject.toml` under the `jaxsim` optional extra
- CI guard: `.github/workflows/jaxsim-upgrade-guard.yml`

## Upgrade Procedure

1. Change only the `jaxsim` optional extra in `pyproject.toml`.
2. Run the JaxSim upgrade guard locally or through `workflow_dispatch`.
3. Include the equivalence and gradient results in the PR body.
4. Update `docs/engines/jaxsim.md` if the native state, velocity, or units
   convention changed.

The upgrade is acceptable only when the guarded equivalence and gradient checks
pass with the new pin. If a new JaxSim release requires adapter changes, keep
those changes in the same PR as the pin bump so reviewers can evaluate the
runtime behavior and dependency change together.

## Required Checks

- Cross-engine equivalence: `tests/motion_matching/test_cross_engine_equivalence.py`
- Gradient behavior: `tests/unit/engines/pinocchio/test_fit_swing_gradient_math.py`

Those checks are the current repository proxies for the 2.1 equivalence and 3.1
gradient acceptance gates. When dedicated JaxSim tests land, add them to the
guard before widening or bumping the pin.
