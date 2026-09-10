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

Camera intrinsics belong to the camera and lens capture settings. Optical zoom,
focus, resolution or crop changes require matching calibration; camera placement
is evaluated separately. Keep these requirements in the shared step declarations
so both the application guide and the atlas present the same prerequisites.

## Lifecycle and Failures

Change workflow declarations at their source, then regenerate the capability
atlas. Missing discoverable steps fail generation. The reference map explains
the workflow; current runtime state and enabled actions come from the application.
An atlas edge must not be presented as an automatic processing connection when
the user must export and import an artifact. Equipment assignment preserves
a capture-owned snapshot and provenance; the current body models do not apply
club constraints. Preserve that limit when projecting the equipment graph.

Guided outcomes reuse `capture_goals` from the connections authority and the
pure `goal_planner` prerequisite resolver. `goal_catalog.validate_bindings`
checks node IDs, allowed navigation actions and workflow keys without launching
Qt or hardware. Single-view and multi-view routes can be incompatible; resolving
such a selection fails rather than silently weakening camera requirements.
Optional steps remain explicit. Readiness stays with the existing workflow and
artifact evidence; visiting a page alone never marks an operation complete.

Model analysis and comparison now share drawing and scene-bound metric
reference routes in the connections authority. Their edges retain distinctions
between virtual-camera pixels, calibrated metric projection and geometric
readouts. Trace imports require marker channels and explicit axes; an atlas
edge must not imply inferred FK from generalized coordinates alone. These
architecture routes do not add executable goal prerequisites or relax readiness.

## Evidence

- [Capture State Authority](../../../src/tools/capture_rig/workflow.py)
- [Static Workflow Extraction](../../../scripts/capability_atlas/model.py)
- [Atlas Tests](../../../tests/scripts/test_capability_atlas.py)
- [Capture Guide](../../motion_capture/capture_rig.md)

## Rationale

Reusing step declarations keeps user guidance synchronized without introducing
a second state machine. Review workflow semantics and graph edge types when
adding a new capture stage or export.
