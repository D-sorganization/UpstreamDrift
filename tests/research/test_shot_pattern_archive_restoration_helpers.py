"""Synthetic safety and parity tests for the report archive restoration helper."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from scripts.research.shot_pattern_analysis.archive_data import build_archives
from docs.research.shot_pattern_analysis.phase2.validation import restore_archive_common
from docs.research.shot_pattern_analysis.phase2.validation import (
    restore_final_control_archives,
    restore_final_primary_archives,
)

pytestmark = pytest.mark.unit


def test_verified_archive_restoration_checks_before_load_and_preserves_bytes(
    tmp_path, monkeypatch
):
    source = tmp_path / "dataset"
    (source / "nested").mkdir(parents=True)
    (source / "matrix_plan.json").write_bytes(b"plan bytes")
    (source / "nested" / "cell.json").write_bytes(b"cell bytes")
    bundle = tmp_path / "bundle"
    build_archives(source, bundle)
    manifest = json.loads((bundle / "archive_manifest.json").read_text())
    expected = {
        name: {key: entry[key] for key in ("sha256", "bytes")}
        for name, entry in manifest["files"].items()
    }
    events = []
    original_verify = restore_archive_common.verify_archives

    def record_verify(path):
        events.append("verify")
        return original_verify(path)

    def load_report(path):
        events.append("load")
        return {
            "plan": (path / "matrix_plan.json").read_bytes(),
            "cell": (path / "nested/cell.json").read_bytes(),
        }

    monkeypatch.setattr(restore_archive_common, "verify_archives", record_verify)
    verified = restore_archive_common.verify_archive_bundle(bundle)
    restore_archive_common.restore_archive_and_compare(
        verified,
        "dataset",
        expected,
        "synthetic-restore-",
        load_report,
        {"plan": b"plan bytes", "cell": b"cell bytes"},
    )

    assert verified.file_count == 2
    assert events == ["verify", "load"]
    assert not list(tmp_path.glob("synthetic-restore-*"))
    assert not (tmp_path.parent / "escaped.txt").exists()


def test_unsafe_archive_is_rejected_before_temporary_extraction(tmp_path):
    bundle = tmp_path / "unsafe_bundle"
    bundle.mkdir()
    archive_path = bundle / "unsafe.zip"
    payload = b"must not escape"
    with zipfile.ZipFile(archive_path, "w") as archive:
        info = zipfile.ZipInfo("../escaped.txt")
        info.create_system = 3
        info.external_attr = 0o100644 << 16
        archive.writestr(info, payload)
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    (bundle / "archive_manifest.json").write_text(
        json.dumps(
            {
                "schema": "shot-study-archives/1",
                "source_name": "unsafe",
                "archives": {
                    "unsafe.zip": {
                        "sha256": digest,
                        "bytes": archive_path.stat().st_size,
                    }
                },
                "files": {
                    "../escaped.txt": {
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "bytes": len(payload),
                        "archive": "unsafe.zip",
                    }
                },
            }
        )
    )
    expected = {
        "../escaped.txt": {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }
    }

    with pytest.raises(ValueError, match="unsafe archive member name"):
        restore_archive_common.verify_archive_bundle(bundle)

    assert not list(tmp_path.glob("must-not-extract-*"))
    assert not (tmp_path / "escaped.txt").exists()


def test_restoration_byte_mismatch_keeps_legacy_failure_text_and_cleans_temp(
    tmp_path,
):
    source = tmp_path / "dataset"
    source.mkdir()
    (source / "file.json").write_bytes(b"actual")
    bundle = tmp_path / "bundle"
    build_archives(source, bundle)
    verified = restore_archive_common.verify_archive_bundle(bundle)
    wrong_inventory = {"dataset/file.json": {"sha256": "wrong", "bytes": 5}}

    with pytest.raises(ValueError, match="^restored file bytes$"):
        restore_archive_common.restore_archive_and_compare(
            verified,
            "dataset",
            wrong_inventory,
            "failure-restore-",
            lambda path: None,
            None,
        )

    assert not list(tmp_path.glob("failure-restore-*"))


def test_report_binding_rejects_status_or_source_identity_drift():
    report = {
        "counts": {"completed": 1, "unavailable": 0, "not_run": 0},
        "planned_cells": 1,
        "stage": "final",
        "plan_sha256": "plan",
        "execution": {"source": "runtime"},
        "cell_manifest_sha256": {"cell-a": "manifest"},
        "cells": [{"cell": "cell-a", "status": "completed"}],
    }
    audit = {
        "plan_sha256": "plan",
        "execution": {"source": "runtime"},
        "cells": {"cell-a": {"manifest_sha256": "manifest", "status": "completed"}},
    }
    counts = {"completed": 1, "unavailable": 0, "not_run": 0}

    restore_archive_common.validate_report_binding(report, audit, counts, 1)
    altered = {**report, "plan_sha256": "different"}
    with pytest.raises(ValueError, match="audit plan binding"):
        restore_archive_common.validate_report_binding(altered, audit, counts, 1)
    altered_status = {**report, "cells": [{"cell": "cell-a", "status": "unavailable"}]}
    with pytest.raises(ValueError, match="audit status binding"):
        restore_archive_common.validate_report_binding(altered_status, audit, counts, 1)


def test_audit_completion_gate_binds_exact_cell_and_row_totals():
    audit = {
        "pass": True,
        "complete": True,
        "failures": [],
        "audited_cells": 2,
        "planned_cells": 2,
        "data_stage": "final",
        "cells": {"a": {"rows": 4}, "b": {"rows": 6}},
    }
    assert restore_archive_common.validate_complete_audit(audit, 2, 10) == 10
    with pytest.raises(ValueError, match="audit rows"):
        restore_archive_common.validate_complete_audit(audit, 2, 9)


@pytest.mark.parametrize(
    ("module", "dataset", "bundle_name", "audit_name", "cells", "rows", "counts"),
    [
        (
            restore_final_primary_archives,
            "final_v1",
            "final_primary_v1",
            "independent_primary_final_audit.json",
            324,
            8100000,
            {"completed": 270, "unavailable": 54, "not_run": 0},
        ),
        (
            restore_final_control_archives,
            "control_final_v1",
            "final_control_v1",
            "independent_control_final_audit.json",
            72,
            1560000,
            {"completed": 52, "unavailable": 20, "not_run": 0},
        ),
    ],
)
def test_restoration_run_receipt_contract_without_touching_saved_data(
    tmp_path, monkeypatch, module, dataset, bundle_name, audit_name, cells, rows, counts
):
    root = Path(__file__).resolve().parents[2]
    phase = root / "docs/research/shot_pattern_analysis/phase2"
    original = phase / dataset
    bundle = phase / "data_archives" / bundle_name
    audit_path = phase / "validation" / audit_name
    audit = {
        "pass": True,
        "complete": True,
        "failures": [],
        "audited_cells": cells,
        "planned_cells": cells,
        "data_stage": "final",
        "plan_sha256": "plan",
        "execution": {"source": "fixture"},
        "cells": {
            f"c{i}": {
                "rows": rows // cells + (1 if i < rows % cells else 0),
                "manifest_sha256": f"manifest-{i}",
                "status": "completed" if i < counts["completed"] else "unavailable",
            }
            for i in range(cells)
        },
    }
    manifest = {
        "source_name": dataset,
        "files": {"fixture.json": {"sha256": "fixture", "bytes": 1}},
    }
    report = {
        "plan_sha256": "plan",
        "execution": {"source": "fixture"},
        "counts": counts,
        "planned_cells": cells,
        "stage": "final",
        "cell_manifest_sha256": {f"c{i}": f"manifest-{i}" for i in range(cells)},
        "cells": [
            {
                "cell": f"c{i}",
                "status": "completed" if i < counts["completed"] else "unavailable",
            }
            for i in range(cells)
        ],
    }
    source_inventory = {"fixture.json": {"sha256": "fixture", "bytes": 1}}
    verified = restore_archive_common.VerifiedArchiveBundle(bundle, manifest, 0)
    monkeypatch.setattr(module.Path, "cwd", classmethod(lambda cls: root))
    original_loads = module.json.loads
    monkeypatch.setattr(module.json, "loads", lambda raw: audit)
    monkeypatch.setattr(module, "sha", lambda path: "fixed-hash")
    monkeypatch.setattr(module, "inventory", lambda path: source_inventory)
    monkeypatch.setattr(module, "verify_archive_bundle", lambda path: verified)
    monkeypatch.setattr(module, "load_matrix_report", lambda *a, **k: report)
    restoration_messages = []
    monkeypatch.setattr(
        module,
        "restore_archive_and_compare",
        lambda *args, **_kwargs: restoration_messages.append(args[-1]),
    )
    monkeypatch.setattr(module, "assert_restoration_unchanged", lambda *a, **k: None)
    output = tmp_path / "new-synthetic-receipt.json"

    result = module.run(output)
    monkeypatch.setattr(module.json, "loads", original_loads)

    assert result["status"] == "passed"
    assert result["scientific_audit"]["rows"] == rows
    assert result["matrix_counts"] == counts
    assert result["files_restored"] == 0
    # Archived control source line 95 used "restored bytes"; primary used the
    # longer message. Keep each failure contract through the shared helper.
    assert restoration_messages == [
        "restored bytes" if dataset == "control_final_v1" else "restored file bytes"
    ]
    assert result["source_sha256"]
    assert set(result) == {
        "schema",
        "status",
        "created_utc",
        "source_sha256",
        "scientific_audit",
        "bundle_files",
        "files_restored",
        "archive_manifest_sha256",
        "plan_sha256",
        "matrix_counts",
        "planned_cells",
        "fresh_directory",
        "temporary_directory_removed",
        "all_file_hashes_and_sizes_equal",
        "full_restored_report_equals_original",
        "audit_plan_execution_cells_statuses_bound",
        "sources_originals_bundles_audit_unchanged",
        "limitations",
    }
    assert json.loads(output.read_text()) == result
