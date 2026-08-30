"""Governed provider-workflow contracts for companion consumers (#9190)."""

from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import jsonschema
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "docs/api/contracts/upstreamdrift-companion-v1.schema.json"
REGISTRY_PATH = REPO_ROOT / "scripts/config/companion_workflows.v1.json"
pytestmark = pytest.mark.unit


def _catalog_module():
    from scripts import companion_catalog

    return companion_catalog


def _workflow_module():
    from scripts import companion_workflows

    return companion_workflows


def _minimal_record() -> dict[str, Any]:
    return {
        "id": "fixture-success",
        "title": "Fixture Success",
        "kind": "workflow",
        "executor": "python-module",
        "argv": ["scripts.companion_workflow_tasks"],
        "cwd": ".",
        "environment": {
            "inherit": ["PATH"],
            "fixed": {"PYTHONHASHSEED": "0"},
            "python_paths": [],
        },
        "expected_exit_code": 0,
        "expected_artifacts": [
            {
                "path": "dist/companion-workflows/artifacts/fixture-success.json",
                "type": "json",
                "required": True,
                "verification": {
                    "method": "json-object",
                    "minimum_bytes": 3,
                },
            }
        ],
        "documentation_paths": ["docs/operations/companion-publication.md"],
        "program_ids": ["model_explorer"],
        "prerequisites": ["Python 3.11 or 3.12"],
        "support_tier": "supported",
        "availability": {"state": "available", "reason": None},
        "determinism": {
            "fixture_kind": "inline",
            "fixture_paths": [],
            "absolute_tolerance": 0.0,
            "relative_tolerance": 0.0,
        },
        "verification_method": "Exact exit code and declared-artifact contract",
        "scientific_limitations": [
            "This workflow exercises software behavior, not scientific qualification."
        ],
        "source_commit": "a" * 40,
    }


def _registry_payload(*records: dict[str, Any]) -> bytes:
    raw_records = []
    for record in records:
        raw = deepcopy(record)
        raw.pop("source_commit", None)
        raw_records.append(raw)
    return json.dumps(
        {
            "registry_id": "upstreamdrift-companion-workflows",
            "version": "1.0.0",
            "workflows": raw_records,
        }
    ).encode()


def test_catalog_exports_complete_governed_workflow_inventory() -> None:
    catalog = _catalog_module().build_catalog(REPO_ROOT, require_clean=False)
    workflows = catalog["workflows"]

    available = [
        row for row in workflows if row["availability"]["state"] == "available"
    ]
    successes = [row for row in available if row["kind"] == "workflow"]
    failures = [row for row in available if row["kind"] == "failure-fixture"]
    unavailable = [
        row for row in workflows if row["availability"]["state"] == "unavailable"
    ]

    assert len(successes) >= 10
    assert len(failures) == 4
    assert {row["failure_class"] for row in failures} == {
        "bad-input",
        "stale-version",
        "unavailable-engine",
        "unsupported-dependency",
    }
    assert unavailable
    assert [row["id"] for row in workflows] == sorted(row["id"] for row in workflows)
    assert all(row["source_commit"] == catalog["source"]["commit"] for row in workflows)
    assert all(row["scientific_limitations"] for row in workflows)
    assert all(row["documentation_paths"] for row in workflows)
    assert all(row["program_ids"] for row in workflows)


def test_registry_is_an_exact_hashed_catalog_input() -> None:
    catalog = _catalog_module().build_catalog(REPO_ROOT, require_clean=False)
    inputs = {row["path"]: row["sha256"] for row in catalog["source"]["inputs"]}

    assert REGISTRY_PATH.relative_to(REPO_ROOT).as_posix() in inputs
    assert len(inputs[REGISTRY_PATH.relative_to(REPO_ROOT).as_posix()]) == 64
    assert catalog["summary"]["workflow_records"] == len(catalog["workflows"])
    assert catalog["summary"]["executable_workflow_records"] >= 14


def test_extended_workflow_schema_is_strict_and_validates_catalog() -> None:
    catalog = _catalog_module().build_catalog(REPO_ROOT, require_clean=False)
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    jsonschema.Draft202012Validator(schema).validate(catalog)
    workflow_schema = schema["$defs"]["workflow"]
    assert workflow_schema["additionalProperties"] is False
    assert {
        "executor",
        "argv",
        "cwd",
        "environment",
        "expected_artifacts",
        "source_commit",
        "determinism",
        "scientific_limitations",
    } <= set(workflow_schema["required"])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (("executor", "shell"), "executor"),
        (
            ("argv", ["scripts.companion_workflow_tasks", "; rm -rf output"]),
            "metacharacter",
        ),
        (
            ("argv", ["scripts.companion_workflow_tasks", "../outside"]),
            "traversal",
        ),
        (
            ("argv", ["scripts.companion_workflow_tasks", "https://example.test/main"]),
            "mutable external reference",
        ),
        (("argv", ["json.tool"]), "approved workflow boundary"),
        (("cwd", "../outside"), "repo-relative"),
        (
            (
                "expected_artifacts",
                [
                    {
                        "path": "C:/outside.json",
                        "type": "json",
                        "required": True,
                        "verification": {
                            "method": "json-object",
                            "minimum_bytes": 1,
                        },
                    }
                ],
            ),
            "repo-relative",
        ),
        (("program_ids", ["dangling-program"]), "program"),
    ],
)
def test_registry_rejects_unsafe_or_dangling_records(
    mutation: tuple[str, Any], message: str
) -> None:
    module = _workflow_module()
    record = _minimal_record()
    key, value = mutation
    record[key] = value
    payload = _registry_payload(record)

    with pytest.raises(module.WorkflowContractError, match=message):
        module.parse_registry(
            payload,
            repo_root=REPO_ROOT,
            source_commit="a" * 40,
            program_ids={"model_explorer"},
        )


def test_registry_rejects_duplicate_ids_and_environment_overlap() -> None:
    module = _workflow_module()
    record = _minimal_record()
    duplicate = deepcopy(record)
    duplicate["environment"]["inherit"].append("PYTHONHASHSEED")
    payload = _registry_payload(record, duplicate)

    with pytest.raises(module.WorkflowContractError, match="environment|duplicate"):
        module.parse_registry(
            payload,
            repo_root=REPO_ROOT,
            source_commit="a" * 40,
            program_ids={"model_explorer"},
        )

    undeclared = _minimal_record()
    undeclared["environment"]["fixed"]["SECRET_NOT_DECLARED"] = "unsafe"
    with pytest.raises(module.WorkflowContractError, match="environment keys"):
        module.parse_registry(
            _registry_payload(undeclared),
            repo_root=REPO_ROOT,
            source_commit="a" * 40,
            program_ids={"model_explorer"},
        )


def test_executor_uses_shell_false_and_only_declared_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _workflow_module()
    record = _minimal_record()
    record["cwd"] = "work"
    record["expected_artifacts"][0]["path"] = "work/result.json"
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setenv("PATH", "declared-path")
    monkeypatch.setenv("SECRET_NOT_DECLARED", "must-not-leak")
    observed: dict[str, Any] = {}

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        observed["command"] = command
        observed.update(kwargs)
        (tmp_path / "work/result.json").write_text("{}\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "ok", "")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    evidence = module.execute_record(tmp_path, record)

    assert observed["shell"] is False
    assert observed["command"] == [
        sys.executable,
        "-m",
        "scripts.companion_workflow_tasks",
    ]
    assert observed["env"] == {"PATH": "declared-path", "PYTHONHASHSEED": "0"}
    assert evidence["workflow_id"] == "fixture-success"
    assert len(evidence["artifacts"][0]["sha256"]) == 64


def test_executor_rejects_missing_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _workflow_module()
    record = _minimal_record()
    record["expected_artifacts"][0]["path"] = "missing.json"
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, "", ""),
    )

    with pytest.raises(module.WorkflowExecutionError, match="missing"):
        module.execute_record(tmp_path, record)


def test_executor_rejects_unexpected_exit_and_undeclared_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _workflow_module()
    record = _minimal_record()
    record["expected_artifacts"][0]["path"] = "result.json"

    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 7, "", "bad"),
    )
    with pytest.raises(module.WorkflowExecutionError, match="exit code"):
        module.execute_record(tmp_path, record)

    def create_extra(
        command: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        (tmp_path / "result.json").write_text("{}\n", encoding="utf-8")
        (tmp_path / "extra.txt").write_text("undeclared\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(module.subprocess, "run", create_extra)
    with pytest.raises(module.WorkflowExecutionError, match="undeclared"):
        module.execute_record(tmp_path, record)


def test_executor_refuses_unavailable_and_stale_source_records(tmp_path: Path) -> None:
    module = _workflow_module()
    unavailable = _minimal_record()
    unavailable["availability"] = {
        "state": "unavailable",
        "reason": "Optional engine is absent",
    }
    unavailable["expected_exit_code"] = None
    unavailable["expected_artifacts"] = []
    with pytest.raises(module.WorkflowExecutionError, match="unavailable"):
        module.execute_record(tmp_path, unavailable)

    stale = _minimal_record()
    stale["source_commit"] = "0" * 40
    with pytest.raises(module.WorkflowExecutionError, match="source commit"):
        module.execute_record(tmp_path, stale, expected_source_commit="a" * 40)


def test_provider_workflows_execute_and_emit_complete_evidence(tmp_path: Path) -> None:
    module = _workflow_module()
    report_path = tmp_path / "execution-report.v1.json"

    report = module.execute_all(REPO_ROOT, report_path=report_path, require_clean=False)

    assert report_path.is_file()
    assert (
        report["source_commit"]
        == _catalog_module().build_catalog(REPO_ROOT, require_clean=False)["source"][
            "commit"
        ]
    )
    assert len(report["registry_sha256"]) == 64
    assert len(report["executed_workflow_ids"]) >= 13
    assert len(report["failure_fixture_ids"]) == 4
    assert report["unavailable_workflow_ids"]
    assert report["development_skipped_workflow_ids"] == ["companion-export"]
    assert all(row["artifacts"] for row in report["executions"])


def test_provider_ci_executes_and_aggregates_governed_workflows() -> None:
    workflow = (REPO_ROOT / ".github/workflows/ci-standard.yml").read_text(
        encoding="utf-8"
    )

    assert "companion-workflows:" in workflow
    assert "pydantic==2.12.5" in workflow
    assert "python3 -m scripts.companion_workflows" in workflow
    assert "--report dist/companion-workflows/execution-report.v1.json" in workflow
    assert "upstreamdrift-companion-workflows-${{ github.sha }}" in workflow
    quality_gate = workflow.split("  quality-gate:", maxsplit=1)[1]
    assert "companion-workflows," in quality_gate
    assert "needs.companion-workflows.result" in quality_gate
