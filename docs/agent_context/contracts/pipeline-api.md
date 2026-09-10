# Motion Pipeline to REST API

## Responsibilities

`MotionPipeline` owns adapter → preprocessing → scaling → inverse kinematics →
matching. `motion_pipeline.api.create_app` owns HTTP input validation, uploaded
file handling and response translation. `PipelineRequest.to_pipeline_config`
is the request adapter; extend this mapping when adding configuration fields.

## Data Contract

Use `motion_pipeline.contracts` for stage payloads and `MotionMatchingResult`.
Preserve source format, adapter options, ordered preprocessing, scaling,
backend names, model path and cost weights. Do not assume a common coordinate
frame or units across imported formats: inspect the selected source adapter
and canonical contract. Production MuJoCo matching requires a real model;
experimental backend declarations are not proof of supported execution.

## Lifecycle and Failures

The request creates pipeline configuration and runs the ordered stages.
`InvalidInputError` identifies caller contract violations; internal stage faults
must remain distinguishable from HTTP client errors. Strict hook failures use
`HookExecutionError`. Inspect the API response mapping when changing stage
failure classification; do not flatten failures into successful empty results.

## Evidence

- [Request and Response Mapping](../../../src/shared/python/motion_pipeline/api.py)
- [Stage Orchestration](../../../src/shared/python/motion_pipeline/orchestrator.py)
- [API Integration Tests](../../../tests/unit/motion_pipeline/orchestrator/test_api.py)
- [Runnable Pipeline Guide](../../motion_pipeline/README.md)

## Rationale

Keep scientific execution behind the same pipeline contract for Python and
HTTP callers. Update configuration mapping and API tests in the same PR as
public pipeline changes. Context freshness is separate from engine qualification.
