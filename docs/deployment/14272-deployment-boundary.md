# Deployment Target Decision for Issue #14272

## Decision

Use the canonical container build as the target for deploying the UpstreamDrift
service. Keep the Python wheel as a separately supported distribution for
Python consumers. This selects an artifact boundary; it does not authorize a
deployment or choose an orchestration platform. Do not add Helm or Kubernetes
configuration without evidence of a cluster requirement.

## Repository Evidence

- [ADR-0021](../adr/0021-container-strategy.md) identifies the root `Dockerfile`
  as the canonical production build. The modular Dockerfile remains experimental.
- [Docker quick reference](../../docker/README.md) identifies the root
  `Dockerfile` as the canonical release image and lists its smoke, size, and
  security checks. This is stronger operational evidence than the experimental
  modular profile path.
- [`pyproject.toml`](../../pyproject.toml) defines a Python 3.11–3.12 package
  built with Hatchling. The [tagged-release workflow](../../.github/workflows/release.yml)
  builds the wheel, runs wheel smoke tests, and publishes it to PyPI. This
  supports Python package consumers, but does not replace the repository's
  documented container boundary for the service.
- [`docs/deployment/interim_setup.md`](interim_setup.md) contains older
  development instructions for starting an API server. It is not the canonical
  production deployment contract and should not override ADR-0021.

## Boundary and Follow-Up

For service deployment, build and qualify the root `Dockerfile`. Treat the
published wheel as a Python library/application distribution with its own
release checks. Keep deployment configuration and execution outside this
decision; this change touches documentation only.

The repository evidence does not establish an automated production image
publication path. A future release-workflow change would need its own scope and
validation. No orchestration choice is justified by the current evidence.
