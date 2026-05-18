# Subsystem Status Governance

This document describes the subsystem status governance system used to ensure production-ready code maintains passing tests.

## Overview

The subsystem status system provides a declarative way to:

1. Define the maturity level of each subsystem in the project
2. Automatically enforce that production subsystems have passing tests
3. Track ownership and review status of subsystems

## Status Levels

| Status       | Description                            | CI Enforcement                           |
| ------------ | -------------------------------------- | ---------------------------------------- |
| `production` | Stable, production-ready code          | **Required** - CI fails if tests are red |
| `beta`       | Feature-complete, testing in progress  | Reported but not blocking                |
| `alpha`      | Experimental, under active development | Not enforced                             |
| `deprecated` | Being phased out                       | Not enforced                             |

## File Locations

| File                                   | Purpose                                        |
| -------------------------------------- | ---------------------------------------------- |
| `docs/status/SUBSYSTEM_STATUS.yaml`    | Declarative subsystem registry                 |
| `scripts/ci/check_subsystem_status.py` | CI script that validates production subsystems |
| `docs/governance/SUBSYSTEM_STATUS.md`  | This documentation                             |

## Subsystem Registry Format

The registry is a YAML file with the following structure:

```yaml
subsystems:
  - name: subsystem_name
    status: production|beta|alpha|deprecated
    description: Human-readable description
    test_paths:
      - tests/path/to/tests1/
      - tests/path/to/tests2/
    owner: team-name
    last_reviewed: YYYY-MM-DD
```

### Field Descriptions

- **name**: Unique identifier for the subsystem (lowercase with underscores)
- **status**: Maturity level (see table above)
- **description**: Brief description of what the subsystem does
- **test_paths**: List of test directory paths relative to project root
- **owner**: Team or individual responsible for the subsystem
- **last_reviewed**: Date when the subsystem status was last reviewed

## CI Integration

The subsystem status check runs automatically in CI as part of the test pipeline.

### Manual Execution

```bash
# Run all production subsystem checks
python scripts/ci/check_subsystem_status.py

# Verbose output
python scripts/ci/check_subsystem_status.py --verbose

# Check a specific subsystem
python scripts/ci/check_subsystem_status.py --subsystem drake_engine
```

### Exit Codes

| Code | Meaning                                                    |
| ---- | ---------------------------------------------------------- |
| 0    | All production subsystems have passing tests               |
| 1    | One or more production subsystems have failing tests       |
| 2    | Configuration error (missing registry, invalid YAML, etc.) |

## Adding a New Subsystem

1. Create your subsystem code in the appropriate directory
2. Write tests in `tests/` or `tests/unit/` directories
3. Add an entry to `docs/status/SUBSYSTEM_STATUS.yaml` with status `alpha`
4. As the subsystem matures, update status to `beta`, then `production`

## Updating Subsystem Status

When updating a subsystem's status:

1. Update the `status` field in `SUBSYSTEM_STATUS.yaml`
2. Update the `last_reviewed` date
3. For promotion to `production`, ensure all tests pass
4. Include the status change in your PR description

## PR Template

When submitting a PR that affects a subsystem, answer:

- [ ] Did you update the subsystem registry if adding/modifying tests?
- [ ] If promoting to production, have all tests been verified locally?

## Governance Process

### Status Review Cadence

- **Production subsystems**: Reviewed monthly
- **Beta subsystems**: Reviewed bi-weekly
- **Alpha subsystems**: Reviewed weekly during active development

### Status Demotion

A production subsystem may be demoted to beta if:

- Tests consistently fail in CI
- The subsystem is no longer actively maintained
- Critical bugs are discovered that require significant refactoring

### Status Promotion

To promote a subsystem to production:

1. Ensure all tests pass consistently
2. Update `SUBSYSTEM_STATUS.yaml` with `status: production`
3. Update `last_reviewed` date
4. Get approval from the subsystem owner
5. Include in PR with clear description of changes

## Examples

### Example: Production Subsystem

```yaml
- name: drake_engine
  status: production
  description: Drake physics engine integration
  test_paths:
    - tests/engines/drake/
    - tests/unit/engines/drake/
  owner: engine-team
  last_reviewed: 2026-05-01
```

### Example: Alpha Subsystem

```yaml
- name: new_feature
  status: alpha
  description: Experimental new feature under development
  test_paths:
    - tests/unit/new_feature/
  owner: developer-name
  last_reviewed: 2026-05-08
```

## Related Documents

- [CI/CD Pipeline Documentation](../architecture/CI_CD.md)
- [Testing Guidelines](../architecture/TESTING.md)
- [Code Ownership](./CODE_OWNERSHIP.md)
