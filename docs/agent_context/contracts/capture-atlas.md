# Capture Workflow to Capability Atlas

## Responsibilities

`src/tools/capture_rig/workflow.py` owns capture step declarations and action
readiness. The atlas reads literal `Step` metadata with Python AST inspection;
it does not import Qt, run readiness checks or start camera hardware.

## Data Contract

Step keys, titles, purposes, requirements, instructions and actions project
into workflow nodes. Architecture edges distinguish explicit artifact exchange
from direct execution. Single-view analysis is 2-D; calibrated reconstruction
requires multiple views and adequate geometric observability.

## Lifecycle and Failures

Change workflow declarations at their source, then regenerate the capability
atlas. Missing discoverable steps fail generation. The reference map explains
the workflow; current runtime state and enabled actions come from the application.
An atlas edge must not be presented as an automatic processing connection when
the user must export and import an artifact.

## Evidence

- [Capture State Authority](../../../src/tools/capture_rig/workflow.py)
- [Static Workflow Extraction](../../../scripts/capability_atlas/model.py)
- [Atlas Tests](../../../tests/scripts/test_capability_atlas.py)
- [Capture Guide](../../motion_capture/capture_rig.md)

## Rationale

Reusing step declarations keeps user guidance synchronized without introducing
a second state machine. Review workflow semantics and graph edge types when
adding a new capture stage or export.
