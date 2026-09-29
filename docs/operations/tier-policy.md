# Engine Tier Policy

UpstreamDrift uses explicit engine tiers to separate the default supported
runtime from optional and exploratory integrations. The canonical metadata is
stored in `src/engines/tiers.py` and each in-scope engine package declares its
own `_tier.py`.

| Tier         | Examples                | Stability bar                                            | Dependencies                                 | Vulnerability SLA     |
| ------------ | ----------------------- | -------------------------------------------------------- | -------------------------------------------- | --------------------- |
| core         | MuJoCo, FastAPI, shared | Must pass on every PR; semver-stable public API; no skip | Installed by default                         | High/Critical: 7 days |
| extended     | Drake, Pinocchio        | Must pass nightly; semver-stable within major versions   | Installed only through extras                | High: 30 days         |
| experimental | OpenSim, MyoSuite       | Best effort; may be skipped; API may break               | Installed only through extras; emits warning | Best effort           |
| archived     | None today              | Read-only; not built; not tested                         | Not installed                                | n/a                   |

## Tier Changes

Tier changes require a pull request that updates `SPEC.md`,
`docs/operations/tier-policy.md`, `src/engines/tiers.py`, package `_tier.py`
metadata, tests, and release notes when user-visible behavior changes.

Promotion from experimental to extended requires reviewer approval from the
engine owner and evidence that dependency installation, import behavior, and
focused tests are stable in CI or nightly validation. Promotion from extended
to core additionally requires PR-gated tests and a semver-stable public API.

Demotion from core is a breaking change and requires a major version bump.
Demotion from extended to experimental requires a release note and a migration
path for users who rely on the prior support contract.

## Re-Evaluation Cadence

Experimental engines should be reviewed before each minor release. Extended
engines should be reviewed at least quarterly against installation health,
nightly validation, and security triage history.

## Warning Contract

Constructing an experimental engine adapter emits
`src.engines.tiers.ExperimentalTierWarning`. Callers may suppress that warning
explicitly, but the default runtime should signal when users leave the core or
extended support surface.
