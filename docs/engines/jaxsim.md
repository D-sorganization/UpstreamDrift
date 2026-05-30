# JaxSim Optional Engine Spike

Issue #6649 keeps JaxSim behind an explicit optional extra while the integration is still gated:

```bash
python -m pip install "upstream-drift[jaxsim]"
```

The pinned package is `jaxsim==0.9.0`. Its base wheel resolves with CPU JAX (`jax`/`jaxlib`) and does not require CUDA. GPU JAX is intentionally not selected by this extra because CUDA wheels are hardware, driver, and platform specific; future GPU enablement should use a separate extra or documented install command after CI coverage exists.

`jaxsim` is not included in `all-engines` yet. The current `all-engines` extra remains the established Drake/Pinocchio rollup while the JaxSim gate verifies coexistence on Linux CI. A Windows dry-run showed that the existing `pin` dependency can fall back to native builds when wheels are unavailable, so combining the new stack with Pinocchio must be validated on the Linux runner before it becomes part of any broad engine bundle.

The lightweight smoke coverage uses `tests/fixtures/jaxsim/single_link.sdf` and steps a minimal SDF model. URDF conversion is intentionally left to issue #6648 because `rod` delegates URDF processing to `gz`/`ign` sdformat tooling; machines without those executables can still install and smoke-test the base JaxSim stack against SDF input.
