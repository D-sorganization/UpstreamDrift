"""Types and contracts for NM-12 model cards, reproduction commands and turnover.

Governing issue: #10627 (epic #10603).
Schema: neural-model-reproduction-card/1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class FlowStepStatus(str, Enum):
    """Execution status of an end-to-end user flow step."""

    PASSED = "PASSED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    BLOCKED = "BLOCKED"


class PromotionVerdict(str, Enum):
    """Production promotion decision based on measured qualification."""

    PROMOTED = "PROMOTED"
    RESEARCH_ONLY = "RESEARCH_ONLY"
    BLOCKED_PREREQUISITE = "BLOCKED_PREREQUISITE"
    REFERENCE_ONLY = "REFERENCE_ONLY"


@dataclass(frozen=True, slots=True)
class ModelReproductionCard:
    """Publication-ready model card with provenance, metrics, and reproduction commands."""

    model_id: str
    intended_task: str
    input_observations: tuple[str, ...]
    physical_assumptions: tuple[str, ...]
    dataset_split_provenance: Mapping[str, str]
    native_validation: Mapping[str, Any]
    performance_economics: Mapping[str, Any]
    limits_and_licensing: tuple[str, ...]
    commands: Mapping[str, str]
    verdict: PromotionVerdict
    schema_version: str = "neural-model-reproduction-card/1.0.0"

    def __post_init__(self) -> None:
        if not self.model_id or not isinstance(self.model_id, str):
            raise ValueError("model_id must be a non-empty string")
        if not self.intended_task or not isinstance(self.intended_task, str):
            raise ValueError("intended_task must be a non-empty string")
        if not self.input_observations:
            raise ValueError("input_observations must be a non-empty tuple")
        if not self.physical_assumptions:
            raise ValueError("physical_assumptions must be a non-empty tuple")
        required_commands = ("generate", "train", "evaluate", "infer", "replay")
        for cmd_key in required_commands:
            if cmd_key not in self.commands:
                raise ValueError(f"commands mapping missing required key: {cmd_key!r}")

    def to_dict(self) -> dict[str, Any]:
        """Losslessly serialize reproduction card to a JSON-compatible dict."""
        return {
            "model_id": self.model_id,
            "intended_task": self.intended_task,
            "input_observations": list(self.input_observations),
            "physical_assumptions": list(self.physical_assumptions),
            "dataset_split_provenance": dict(self.dataset_split_provenance),
            "native_validation": dict(self.native_validation),
            "performance_economics": dict(self.performance_economics),
            "limits_and_licensing": list(self.limits_and_licensing),
            "commands": dict(self.commands),
            "verdict": self.verdict.value,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ModelReproductionCard:
        """Construct card from mapping with schema validation."""
        schema = str(data.get("schema_version", ""))
        if schema != "neural-model-reproduction-card/1.0.0":
            raise ValueError(f"Unsupported schema_version: {schema!r}")
        return cls(
            model_id=str(data["model_id"]),
            intended_task=str(data["intended_task"]),
            input_observations=tuple(data["input_observations"]),
            physical_assumptions=tuple(data["physical_assumptions"]),
            dataset_split_provenance=dict(data.get("dataset_split_provenance", {})),
            native_validation=dict(data.get("native_validation", {})),
            performance_economics=dict(data.get("performance_economics", {})),
            limits_and_licensing=tuple(data.get("limits_and_licensing", ())),
            commands=dict(data["commands"]),
            verdict=PromotionVerdict(str(data["verdict"])),
            schema_version=schema,
        )


@dataclass(frozen=True, slots=True)
class FlowStepOutcome:
    """Outcome of a single step in the end-to-end verification flow."""

    step_name: str
    status: FlowStepStatus
    duration_s: float
    message: str
    details: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class EndToEndFlowReport:
    """Complete report of end-to-end user flow verification."""

    model_id: str
    overall_success: bool
    duration_s: float
    steps: tuple[FlowStepOutcome, ...]
    verdict: PromotionVerdict
    reproduction_command: str
