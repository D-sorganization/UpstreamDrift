# SPEC.md — Repository Specification Document

<!--
  TEMPLATE VERSION: 1.0.0
  LAST UPDATED: 2026-06-18

  This is the canonical specification template for all repositories in the
  D-sorganization fleet. Every repo MUST have a SPEC.md at its root.

  INSTRUCTIONS:
  1. Copy this template to the root of your repository as SPEC.md
  2. Fill in every section — leave nothing as "[TODO]"
  3. Keep this document updated with every PR that changes functionality
  4. CI will block merges if SPEC.md is stale (source changed but spec didn't)

  AUDIENCE: This document is designed for both human developers AND AI agents.
  Write clearly, use concrete examples, and avoid ambiguity.
-->

## SPEC Ownership and Update Cadence

- **Owner:** @diete (responsible for accepting SPEC.md edits)
- **Update triggers (mandatory):**
  - Any PR that adds, removes, or moves a top-level `src/` package or a public
    engine adapter must update §6 (Component Locations) and §7 (Feature Status).
  - Any PR that changes the version in `pyproject.toml` must update §1 (Identity).
  - Any PR that changes a CI gate threshold must update §X (Quality Gates).
- **Review cadence:** SPEC.md is reviewed for staleness on every release
  (per `docs/operations/release-runbook.md`, see #3842).

## 1. Identity

| Field                   | Value                                              |
| ----------------------- | -------------------------------------------------- |
| **Repository Name**     | `UpstreamDrift`                                    |
| **GitHub URL**          | `https://github.com/D-sorganization/UpstreamDrift` |
| **Owner**               | D-sorganization                                    |
| **Primary Language(s)** | Python 3.11+, Rust, TypeScript                     |
| **License**             | MIT                                                |
| **Current Version**     | 2.1.1                                              |

| **Spec Version** | 1.0.479 |
| **Last Spec Update** | 2026-07-27 |

- Bolt: optimized error norm calculation in Pinocchio diff IK loop by using `math.sqrt(np.vdot(err, err))` to bypass array allocation overhead.
