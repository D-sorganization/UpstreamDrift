# Supported Engine Tiers

This document defines the support contract for engine-backed development,
testing, and releases in UpstreamDrift.

## Tier Summary

| Tier         | Engines           | Install Profile                        | Validation Path                                                                                          | Intended Use                                               |
| ------------ | ----------------- | -------------------------------------- | -------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| Supported    | MuJoCo            | `pip install -e ".[dev]"`              | Required PR CI in `.github/workflows/ci-standard.yml`                                                    | Default development, demos, and release readiness          |
| Extended     | Drake, Pinocchio  | `pip install -e ".[dev,all-engines]"`  | Nightly cross-engine validation in `.github/workflows/nightly-cross-engine.yml` plus targeted local runs | Cross-engine comparisons and advanced rigid-body workflows |
| Experimental | OpenSim, MyoSuite | `pip install -e ".[dev,biomechanics]"` | Best-effort local validation only                                                                        | Integration spikes and long-horizon biomechanics work      |

## What Each Tier Means

### Supported

- The workflow must remain usable from a clean checkout.
- Required PR checks exercise this path.
- Regressions here should block merges.

### Extended

- The engines are part of the documented product surface.
- We validate them through scheduled or targeted cross-engine workflows rather
  than every required PR check.
- Regressions should be triaged quickly, but the install footprint is treated as
  heavier than the default developer path.

### Experimental

- The code and docs remain in the repository, but the workflow is not currently
  a release gate.
- Missing capabilities or stub implementations are expected.
- Changes here should not silently change the guarantees of the supported tiers.

## Capability Detail

Use these documents together:

- `docs/engines/engine_capabilities.md` for feature-by-feature support
- `docs/engine_selection_guide.md` for engine-selection guidance
- `docs/troubleshooting/cross_engine_deviations.md` when nightly validation
  detects a drift between engines
