"""Audit coverage and scientific readiness of the momentum-transfer agenda."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
QUESTION_REGISTRY = ARTICLE / "data/momentum_transfer_question_registry.json"
EXPERIMENT_REGISTRY = ARTICLE / "data/momentum_transfer_experiment_registry.json"
OUTPUT = ARTICLE / "data/momentum_transfer_readiness_audit.json"

REQUIRED_POINT_IDS = {f"MTQ-{index:02d}" for index in range(1, 10)}
ALLOWED_ANSWER_STATES = {
    "answered_within_declared_model_tiers",
    "partly_answered",
    "unresolved",
    "not_supported_as_general_rule",
    "unresolved_until_typed",
}
ALLOWED_PLAN_STATES = {"executed", "partly_executed", "registered", "blocked"}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_readiness_audit(
    question_registry: dict[str, Any], experiment_registry: dict[str, Any]
) -> dict[str, Any]:
    """Build and validate a point-by-point readiness audit."""

    questions = {item["id"]: item for item in question_registry["questions"]}
    experiments = experiment_registry["experiments"]
    experiment_ids = {item["id"] for item in experiments}
    if len(experiment_ids) != len(experiments):
        raise ValueError("Duplicate experiment id")

    points = question_registry.get("critical_points", [])
    point_ids = {item["id"] for item in points}
    if point_ids != REQUIRED_POINT_IDS or len(points) != len(REQUIRED_POINT_IDS):
        raise ValueError(
            "Critical-point register must contain MTQ-01 through MTQ-09 exactly once"
        )

    rows: list[dict[str, Any]] = []
    for point in points:
        question_id = point["question_id"]
        if question_id not in questions:
            raise ValueError(f"Unknown question_id for {point['id']}: {question_id}")
        if point["answer_state"] not in ALLOWED_ANSWER_STATES:
            raise ValueError(f"Invalid answer_state for {point['id']}")
        if point["model_plan_state"] not in ALLOWED_PLAN_STATES:
            raise ValueError(f"Invalid model_plan_state for {point['id']}")
        if point["human_plan_state"] not in ALLOWED_PLAN_STATES:
            raise ValueError(f"Invalid human_plan_state for {point['id']}")

        linked = [
            experiment
            for experiment in experiments
            if question_id in experiment["questions"]
        ]
        linked_ids = {item["id"] for item in linked}
        declared = set(point["experiment_ids"])
        if not declared <= linked_ids:
            raise ValueError(
                f"{point['id']} cites an experiment that does not cover {question_id}"
            )
        if "MT-H01" not in linked_ids:
            raise ValueError(
                f"{point['id']} lacks the registered human falsification stage"
            )
        if not point["present_answer"] or not point["decisive_next_test"]:
            raise ValueError(f"{point['id']} lacks an answer or decisive next test")
        if not point["falsifier"]:
            raise ValueError(f"{point['id']} lacks a falsifier")
        if not point.get("evidence_artifacts"):
            raise ValueError(f"{point['id']} lacks inspectable evidence artifacts")

        rows.append(
            {
                "id": point["id"],
                "question_id": question_id,
                "topic": point["topic"],
                "answer_state": point["answer_state"],
                "model_plan_state": point["model_plan_state"],
                "human_plan_state": point["human_plan_state"],
                "experiment_ids": sorted(declared),
                "present_answer": point["present_answer"],
                "decisive_next_test": point["decisive_next_test"],
                "falsifier": point["falsifier"],
                "data_gate": point["data_gate"],
                "evidence_artifacts": point["evidence_artifacts"],
            }
        )

    question_coverage = {
        question_id: sorted(
            item["id"] for item in experiments if question_id in item["questions"]
        )
        for question_id in sorted(questions)
    }
    uncovered = [key for key, value in question_coverage.items() if not value]
    if uncovered:
        raise ValueError(f"Questions without experiments: {uncovered}")

    return {
        "schema_version": "momentum-transfer-readiness-audit/v1",
        "source_record": question_registry["source_record"],
        "program_issue": question_registry["program_issue"],
        "summary": {
            "critical_point_count": len(rows),
            "answered_or_partly_answered": sum(
                row["answer_state"]
                in {
                    "answered_within_declared_model_tiers",
                    "partly_answered",
                    "not_supported_as_general_rule",
                }
                for row in rows
            ),
            "unresolved_or_definition_gated": sum(
                row["answer_state"] in {"unresolved", "unresolved_until_typed"}
                for row in rows
            ),
            "unresolved_point_ids": [
                row["id"]
                for row in rows
                if row["answer_state"] in {"unresolved", "unresolved_until_typed"}
            ],
            "model_plan_registered_for_all": all(
                row["model_plan_state"] in ALLOWED_PLAN_STATES for row in rows
            ),
            "human_plan_registered_for_all": all(
                row["human_plan_state"] in ALLOWED_PLAN_STATES for row in rows
            ),
            "human_execution_blocked": True,
            "human_blocker": (
                "No qualifying governed participant dataset with synchronized "
                "bilateral six-axis grip wrenches is available."
            ),
        },
        "question_coverage": question_coverage,
        "critical_points": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("build", "validate"), nargs="?", default="build"
    )
    args = parser.parse_args()
    result = build_readiness_audit(_load(QUESTION_REGISTRY), _load(EXPERIMENT_REGISTRY))
    if args.command == "build":
        OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    elif not OUTPUT.exists() or _load(OUTPUT) != result:
        raise SystemExit(
            "Momentum-transfer readiness audit is stale; run the build command"
        )
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
