"""Deterministic canonical-core setup validation and wizard view models.

The setup wizard is intentionally a pure validation layer. It does not call
Sidekick, an LLM, or any autonomous agent service; hosts pass a candidate config
mapping in, and receive structured issues plus deterministic suggested fixes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

import numpy as np

__all__ = [
    "SetupValidationIssue",
    "SetupValidationReport",
    "SetupWizardSnapshot",
    "SetupWizardStep",
    "SetupWizardStepId",
    "SetupWizardViewModel",
    "validate_canonical_setup_config",
]

IssueSeverity = Literal["error", "warning"]
SetupWizardStepId = Literal["units_frames", "model", "calibration", "review"]

_EXPECTED_UNITS = {
    "length": "m",
    "mass": "kg",
    "time": "s",
    "angle": "rad",
    "force": "N",
    "torque": "N*m",
}
_TORQUE_UNIT_ALIASES = {"N*m", "N-m", "Nm", "N·m"}
_STEP_ORDER: tuple[SetupWizardStepId, ...] = (
    "units_frames",
    "model",
    "calibration",
    "review",
)


class _StepStatus(str, Enum):
    COMPLETE = "complete"
    BLOCKED = "blocked"
    READY = "ready"
    WAITING = "waiting"


@dataclass(frozen=True, slots=True)
class SetupValidationIssue:
    """One plain-language setup validation issue.

    Invariants:
        - ``code``, ``field_path``, ``message``, and ``suggested_fix`` are
          non-empty strings.
        - ``severity`` is either ``"error"`` or ``"warning"``.
    """

    code: str
    field_path: str
    message: str
    suggested_fix: str
    severity: IssueSeverity = "error"

    def __post_init__(self) -> None:
        for field_name in ("code", "field_path", "message", "suggested_fix"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if self.severity not in {"error", "warning"}:
            raise ValueError("severity must be 'error' or 'warning'")


@dataclass(frozen=True, slots=True)
class SetupValidationReport:
    """Validation result for a candidate canonical-core run config."""

    issues: tuple[SetupValidationIssue, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.issues, tuple):
            raise TypeError("issues must be a tuple")
        for issue in self.issues:
            if not isinstance(issue, SetupValidationIssue):
                raise TypeError(
                    "issues entries must be SetupValidationIssue "
                    f"(got {type(issue).__name__})"
                )

    @property
    def errors(self) -> tuple[SetupValidationIssue, ...]:
        """Return blocking validation errors."""

        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[SetupValidationIssue, ...]:
        """Return non-blocking validation warnings."""

        return tuple(issue for issue in self.issues if issue.severity == "warning")

    @property
    def is_valid(self) -> bool:
        """Return ``True`` when no blocking setup errors were found."""

        return not self.errors

    def issues_for_step(
        self, step_id: SetupWizardStepId
    ) -> tuple[SetupValidationIssue, ...]:
        """Return issues owned by a wizard step."""

        if step_id not in _STEP_ORDER:
            raise ValueError(f"unknown setup wizard step: {step_id!r}")
        prefixes = {
            "units_frames": ("convention", "units", "frame", "world_frame", "gravity"),
            "model": ("model",),
            "calibration": ("calibration",),
            "review": (),
        }[step_id]
        if not prefixes:
            return self.issues
        return tuple(
            issue
            for issue in self.issues
            if any(
                issue.field_path == prefix or issue.field_path.startswith(f"{prefix}.")
                for prefix in prefixes
            )
        )


@dataclass(frozen=True, slots=True)
class SetupWizardStep:
    """Render-ready state for one deterministic wizard step."""

    step_id: SetupWizardStepId
    title: str
    status: str
    issue_count: int
    can_advance: bool

    def __post_init__(self) -> None:
        if self.step_id not in _STEP_ORDER:
            raise ValueError(f"unknown setup wizard step: {self.step_id!r}")
        if not isinstance(self.title, str) or not self.title.strip():
            raise ValueError("title must be a non-empty string")
        if self.status not in {item.value for item in _StepStatus}:
            raise ValueError(f"unknown setup wizard status: {self.status!r}")
        if not isinstance(self.issue_count, int) or self.issue_count < 0:
            raise ValueError("issue_count must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class SetupWizardSnapshot:
    """Top-level read model for embedding the setup wizard."""

    current_step: SetupWizardStepId
    steps: tuple[SetupWizardStep, ...]
    report: SetupValidationReport

    def __post_init__(self) -> None:
        if self.current_step not in _STEP_ORDER:
            raise ValueError(f"unknown current_step: {self.current_step!r}")
        if not isinstance(self.steps, tuple):
            raise TypeError("steps must be a tuple")
        if len(self.steps) != len(_STEP_ORDER):
            raise ValueError("steps must include every setup wizard step")
        if not isinstance(self.report, SetupValidationReport):
            raise TypeError("report must be a SetupValidationReport")

    @property
    def current_index(self) -> int:
        """Return the zero-based index of ``current_step``."""

        return _STEP_ORDER.index(self.current_step)


def validate_canonical_setup_config(
    config: dict[str, Any],
) -> SetupValidationReport:
    """Validate a canonical-core run config before a simulation run.

    The MVP enforces the preconditions that are currently stable on
    ``origin/main``:

    - CC-1 units and frames: ``canonical-v2``, SI units, ``world_Zup``.
    - CC-3 model shape: a named model with non-empty joints and ``nq == nv + 1``.
    - CC-21 calibration gate: complete subject calibration is present.

    Raises:
        TypeError: If ``config`` is not a dictionary.
    """

    if not isinstance(config, dict):
        raise TypeError(f"config must be a dict (got {type(config).__name__})")
    issues: list[SetupValidationIssue] = []
    _validate_units_and_frames(config, issues)
    _validate_model(config, issues)
    _validate_calibration(config, issues)
    return SetupValidationReport(tuple(issues))


class SetupWizardViewModel:
    """Pure state machine for the guided setup wizard.

    Hosts own persistence and UI controls. This class only validates candidate
    configs and advances through deterministic steps when the current step has
    no blocking issues.
    """

    def __init__(self) -> None:
        self._current_step: SetupWizardStepId = _STEP_ORDER[0]
        self._report = SetupValidationReport()

    @property
    def current_step(self) -> SetupWizardStepId:
        """Return the currently selected wizard step id."""

        return self._current_step

    def validate(self, config: dict[str, Any]) -> SetupWizardSnapshot:
        """Validate ``config`` and return a render-ready snapshot."""

        self._report = validate_canonical_setup_config(config)
        return self.snapshot()

    def can_advance(self) -> bool:
        """Return whether the current step can advance."""

        current_errors = self._current_step_errors()
        return not current_errors

    def advance(self, config: dict[str, Any]) -> SetupWizardSnapshot:
        """Validate and move to the next step if the current step passes."""

        self.validate(config)
        if self.can_advance():
            index = _STEP_ORDER.index(self._current_step)
            if index < len(_STEP_ORDER) - 1:
                self._current_step = _STEP_ORDER[index + 1]
        return self.snapshot()

    def retreat(self) -> SetupWizardSnapshot:
        """Move back one step, if possible, and return a snapshot."""

        index = _STEP_ORDER.index(self._current_step)
        if index > 0:
            self._current_step = _STEP_ORDER[index - 1]
        return self.snapshot()

    def snapshot(self) -> SetupWizardSnapshot:
        """Return the current render-ready wizard state."""

        return SetupWizardSnapshot(
            current_step=self._current_step,
            steps=tuple(self._build_step(step_id) for step_id in _STEP_ORDER),
            report=self._report,
        )

    def _current_step_errors(self) -> tuple[SetupValidationIssue, ...]:
        issues = self._report.issues_for_step(self._current_step)
        return tuple(issue for issue in issues if issue.severity == "error")

    def _build_step(self, step_id: SetupWizardStepId) -> SetupWizardStep:
        issues = self._report.issues_for_step(step_id)
        errors = tuple(issue for issue in issues if issue.severity == "error")
        current_index = _STEP_ORDER.index(self._current_step)
        step_index = _STEP_ORDER.index(step_id)
        if errors:
            status = _StepStatus.BLOCKED.value
        elif step_index < current_index:
            status = _StepStatus.COMPLETE.value
        elif step_index == current_index:
            status = _StepStatus.READY.value
        else:
            status = _StepStatus.WAITING.value
        return SetupWizardStep(
            step_id=step_id,
            title=_step_title(step_id),
            status=status,
            issue_count=len(issues),
            can_advance=not errors,
        )


def _step_title(step_id: SetupWizardStepId) -> str:
    return {
        "units_frames": "Units and Frames",
        "model": "Canonical Model",
        "calibration": "Calibration",
        "review": "Review",
    }[step_id]


def _validate_units_and_frames(
    config: dict[str, Any], issues: list[SetupValidationIssue]
) -> None:
    convention = config.get("convention")
    if convention != "canonical-v2":
        issues.append(
            SetupValidationIssue(
                code="CC36_CONVENTION",
                field_path="convention",
                message="This setup must use the canonical-v2 state convention.",
                suggested_fix='Set "convention" to "canonical-v2".',
            )
        )

    units = config.get("units")
    if units == "SI":
        pass
    elif isinstance(units, dict):
        _validate_unit_mapping(units, issues)
    else:
        issues.append(
            SetupValidationIssue(
                code="CC36_UNITS",
                field_path="units",
                message="Units must be declared as SI before the run starts.",
                suggested_fix=(
                    'Set "units" to "SI", or provide length=m, mass=kg, '
                    "time=s, angle=rad, force=N, torque=N*m."
                ),
            )
        )

    frame = config.get("frame", config.get("world_frame"))
    if frame != "world_Zup":
        issues.append(
            SetupValidationIssue(
                code="CC36_WORLD_FRAME",
                field_path="frame",
                message="The canonical-core world frame is Z-up.",
                suggested_fix='Set "frame" or "world_frame" to "world_Zup".',
            )
        )

    gravity = config.get("gravity")
    if gravity is not None:
        _validate_gravity(gravity, issues)


def _validate_unit_mapping(
    units: dict[str, Any], issues: list[SetupValidationIssue]
) -> None:
    for name, expected in _EXPECTED_UNITS.items():
        actual = units.get(name)
        if name == "torque" and actual in _TORQUE_UNIT_ALIASES:
            continue
        if actual != expected:
            issues.append(
                SetupValidationIssue(
                    code="CC36_UNIT_MISMATCH",
                    field_path=f"units.{name}",
                    message=f"{name.title()} must use canonical SI unit {expected}.",
                    suggested_fix=f'Set "units.{name}" to "{expected}".',
                )
            )


def _validate_gravity(gravity: Any, issues: list[SetupValidationIssue]) -> None:
    arr = np.asarray(gravity, dtype=float)
    if arr.shape != (3,) or not np.all(np.isfinite(arr)):
        issues.append(
            SetupValidationIssue(
                code="CC36_GRAVITY_SHAPE",
                field_path="gravity",
                message="Gravity must be a finite 3-vector in m/s^2.",
                suggested_fix='Use "gravity": [0.0, 0.0, -9.80665].',
            )
        )
        return
    if not np.allclose(arr, np.array([0.0, 0.0, -9.80665]), atol=1e-6):
        issues.append(
            SetupValidationIssue(
                code="CC36_GRAVITY_FRAME",
                field_path="gravity",
                message="Gravity must point down in the canonical Z-up world frame.",
                suggested_fix='Use "gravity": [0.0, 0.0, -9.80665].',
            )
        )


def _validate_model(config: dict[str, Any], issues: list[SetupValidationIssue]) -> None:
    model = config.get("model")
    if not isinstance(model, dict):
        issues.append(
            SetupValidationIssue(
                code="CC36_MODEL_REQUIRED",
                field_path="model",
                message="A canonical model block is required before simulation.",
                suggested_fix=(
                    'Add "model" with canonical_id, joint_names, nq, and nv.'
                ),
            )
        )
        return

    model_id = model.get("canonical_id", model.get("id"))
    if not isinstance(model_id, str) or not model_id.strip():
        issues.append(
            SetupValidationIssue(
                code="CC36_MODEL_ID",
                field_path="model.canonical_id",
                message="The model must have a stable canonical identifier.",
                suggested_fix='Set "model.canonical_id" to a non-empty string.',
            )
        )

    joint_names = model.get("joint_names")
    if not _is_non_empty_str_sequence(joint_names):
        issues.append(
            SetupValidationIssue(
                code="CC36_MODEL_JOINTS",
                field_path="model.joint_names",
                message="The model must declare at least one named joint.",
                suggested_fix='Set "model.joint_names" to a non-empty list of strings.',
            )
        )

    nq = model.get("nq")
    nv = model.get("nv")
    nq_int = _as_positive_int(nq)
    nv_int = _as_positive_int(nv)
    if nq_int is None:
        issues.append(_dimension_issue("model.nq", "nq"))
    if nv_int is None:
        issues.append(_dimension_issue("model.nv", "nv"))
    if nq_int is not None and nv_int is not None and nq_int != nv_int + 1:
        issues.append(
            SetupValidationIssue(
                code="CC36_MODEL_DIMENSIONS",
                field_path="model.nq",
                message=(
                    "canonical-v2 floating-base models must satisfy nq == nv + 1 "
                    "because the quaternion has one redundant coordinate."
                ),
                suggested_fix="Check the model adapter dimensions and set nq to nv + 1.",
            )
        )


def _validate_calibration(
    config: dict[str, Any], issues: list[SetupValidationIssue]
) -> None:
    calibration = config.get("calibration")
    if not isinstance(calibration, dict):
        issues.append(
            SetupValidationIssue(
                code="CC36_CALIBRATION_REQUIRED",
                field_path="calibration",
                message="Subject calibration is required before this run can start.",
                suggested_fix=(
                    "Run or import the calibration step and attach a calibration "
                    "block with status=complete and anthropometrics_ref."
                ),
            )
        )
        return

    status = calibration.get("status")
    validated = calibration.get("validated")
    if status != "complete" and validated is not True:
        issues.append(
            SetupValidationIssue(
                code="CC36_CALIBRATION_INCOMPLETE",
                field_path="calibration.status",
                message="Calibration must be marked complete or validated.",
                suggested_fix='Set "calibration.status" to "complete" after validation.',
            )
        )

    has_subject_ref = any(
        calibration.get(name)
        for name in (
            "anthropometrics_ref",
            "subject_anthropometrics",
            "subject_id",
        )
    )
    if not has_subject_ref:
        issues.append(
            SetupValidationIssue(
                code="CC36_CALIBRATION_SUBJECT",
                field_path="calibration.anthropometrics_ref",
                message="Calibration must identify the calibrated subject.",
                suggested_fix=(
                    'Set "calibration.anthropometrics_ref" or "calibration.subject_id".'
                ),
            )
        )


def _dimension_issue(field_path: str, name: str) -> SetupValidationIssue:
    return SetupValidationIssue(
        code="CC36_MODEL_DIMENSION_REQUIRED",
        field_path=field_path,
        message=f"The model must declare positive integer {name}.",
        suggested_fix=f'Set "model.{name}" to the adapter-reported dimension.',
    )


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _as_positive_int(value: Any) -> int | None:
    if _is_positive_int(value):
        return value
    return None


def _is_non_empty_str_sequence(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or not value:
        return False
    return all(isinstance(item, str) and item.strip() for item in value)
