# Performance Benchmarks

This directory stores pytest-benchmark results for tracking performance
over time.

## Running Benchmarks

```bash
pytest tests/ -m benchmark --benchmark-autosave
```

## Directory Structure

- `*.json` — pytest-benchmark output files (auto-generated)
- `README.md` — This file

## Baseline

Initial baseline established 2026-02-10. Benchmarks are tracked
in CI via the `ci-standard.yml` workflow.
