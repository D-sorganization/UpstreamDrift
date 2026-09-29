"""Tests for tiered security SLA and SBOM metadata helpers."""

from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path

import pytest


def _load_security_module(name: str):
    script_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "security" / f"{name}.py"
    )
    spec = importlib.util.spec_from_file_location(name, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_sla_deadline_uses_severity_and_tier():
    module = _load_security_module("triage_osv_findings")

    deadline = module.compute_sla_deadline(
        tier="core",
        severity="critical",
        detected_on=date(2026, 5, 3),
    )

    assert deadline == date(2026, 5, 4)


def test_sla_deadline_rejects_unknown_tier():
    module = _load_security_module("triage_osv_findings")

    with pytest.raises(ValueError, match="unsupported tier: legacy"):
        module.compute_sla_deadline(
            tier="legacy",
            severity="high",
            detected_on=date(2026, 5, 3),
        )


def test_osv_finding_summary_classifies_package_tier_and_deadline():
    module = _load_security_module("triage_osv_findings")
    finding = {
        "id": "GHSA-demo",
        "package": {"name": "opensim", "ecosystem": "PyPI"},
        "severity": [
            {
                "type": "CVSS_V3",
                "score": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
            }
        ],
    }

    summary = module.summarize_finding(finding, detected_on=date(2026, 5, 3))

    assert summary.vulnerability_id == "GHSA-demo"
    assert summary.package == "opensim"
    assert summary.tier == "experimental"
    assert summary.severity == "critical"
    assert summary.sla_deadline == date(2026, 5, 10)


def test_sbom_tier_specs_define_required_artifacts():
    module = _load_security_module("sbom_tiers")

    artifacts = module.expected_artifact_names("2.1.0")

    assert "upstream-drift-2.1.0.cyclonedx.core.json" in artifacts
    assert "upstream-drift-2.1.0.spdx.extended.json" in artifacts
    assert "upstream-drift-2.1.0.cyclonedx.full.json" in artifacts
    assert module.install_spec_for_tier("core", "2.1.0") == "upstream-drift==2.1.0"
    assert (
        module.install_spec_for_tier("extended", "2.1.0")
        == "upstream-drift[all-engines]==2.1.0"
    )
