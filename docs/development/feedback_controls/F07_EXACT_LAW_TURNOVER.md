# Exact Native Muscle-Law Admission Turnover

## Scope and Dependency

Child #11867 of F07 #11791, branch `fix/feedback-exact-law-11867`, starts from
the exact Thelen readiness commit `c094cc1716dd1446886fc0c6f0d3b229683b1dfa`
in PR #11829. The queued parent branch is untouched. The guard's own commit
contains only exact concrete-class admission, receipt policy identity and its
tests/documentation. Per coordinated dependency handling, the PR initially
targets `feat/feedback-native-thelen-11826`, unarmed. After #11829 merges it
must be retargeted to main and checked before protected auto-merge is armed;
it must not merge into or modify the queued Thelen branch.

## Implementation and Evidence

Recursive muscle admission now requires the exact native concrete names
`Millard2012EquilibriumMuscle` or `Thelen2003Muscle` before system
initialization. Successful `safeDownCast` alone is insufficient for an unknown
derived force/state law. The supported laws retain native downcasts for their
own activation/fiber minima. Adapter version 1.3.0 records and hashes
`exact-supported-concrete-law/1.0.0` alongside the ordered actual muscle laws.
All existing ignore-mode, force, controller, constraint, input and complete-state
checks remain active.

TDD: after the proxy identity setup was corrected for SWIG's separately exposed
Component and Muscle methods, both unknown-law tests reached the initialization
trap before the implementation fix. Both positive replay cases also lacked the
new receipt policy. The exact allowlist and hashed policy make these pass.
The negative test changes only reported proxy class identity and preserves
successful actual native casts; it is not a compiled C++ plugin test.

Native Python 3.12/OpenSim 4.6: 69 replay and manual-governance tests passed,
serially with `-o addopts=''`. Ruff passes. Final central pre-PR and hook results
belong in the PR handoff. Canonical source remains chapter 13; no generated
artifact or manual release is claimed. The state laws are unchanged.

## Remaining Work

F07 remains open. Full model anatomy, rigid-tendon modes, donor path
force/length consistency (#11856), physiological input domains, grip/contact,
source/license/resource closure and full-capture full-state independent replay
remain required. No donor/runtime patch, capture download or plugin installation
is part of this change. The sparse Tools checkout borrows from the retained
native-admission object store; preserve that store until borrowers are independent.
