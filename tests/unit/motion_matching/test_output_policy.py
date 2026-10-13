"""Fail-closed output policy for golf outputs (#10382, MS-108).

Owner ruling 2026-10-11: advertise clubhead position/speed and sequencing with
uncertainty; withhold face angle, joint loads and impact/ball-flight as
"not validated" unless calibration and evidence are supplied.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any
from unittest.mock import patch

import pytest

from src.shared.python.motion_matching.club_observability import FaceObservability
from src.shared.python.motion_matching.export import export_report
from src.shared.python.motion_matching.output_policy import (
    ADVERTISED_OUTPUTS,
    WITHHELD_OUTPUTS,
    OutputPolicyError,
    evaluate_output_policy,
    render_output_policy_markdown,
)

pytestmark = [pytest.mark.unit]


def _receipt(outputs: Any = None) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "engine": "mujoco",
        "capture": "driver",
        "candidate_sha256": "a" * 64,
        "acceptance": {"verdict": "REJECTED"},
    }
    if outputs is not None:
        receipt["outputs"] = outputs
    return receipt


EVIDENCE = {"calibration": "cal-2026-09-01", "evidence": "sha256:" + "b" * 64}


class TestEvaluateOutputPolicy:
    def test_withheld_output_without_claim_is_not_validated(self) -> None:
        statuses = evaluate_output_policy(_receipt())
        for name in WITHHELD_OUTPUTS:
            assert statuses[name].status == "withheld"
            assert statuses[name].reason == "not validated"

    def test_advertised_outputs_are_listed_without_validation_claim(self) -> None:
        statuses = evaluate_output_policy(_receipt())
        for name in ADVERTISED_OUTPUTS:
            assert statuses[name].status == "advertised"

    def test_withheld_output_claimed_validated_without_evidence_is_rejected(
        self,
    ) -> None:
        receipt = _receipt({"face_angle": {"validated": True, "value": 2.5}})
        with pytest.raises(OutputPolicyError, match="face_angle.*calibration"):
            evaluate_output_policy(receipt)

    def test_withheld_output_with_calibration_and_evidence_is_validated(self) -> None:
        receipt = _receipt(
            {"joint_loads": {"validated": True, "value": 410.0, **EVIDENCE}}
        )
        statuses = evaluate_output_policy(receipt)
        assert statuses["joint_loads"].status == "validated"

    def test_unknown_output_name_is_rejected(self) -> None:
        with pytest.raises(OutputPolicyError, match="unknown output 'ball_spin'"):
            evaluate_output_policy(_receipt({"ball_spin": {"validated": False}}))

    def test_advertised_value_without_uncertainty_is_rejected(self) -> None:
        receipt = _receipt({"clubhead_speed": {"value": 42.0}})
        with pytest.raises(OutputPolicyError, match="clubhead_speed.*uncertainty"):
            evaluate_output_policy(receipt)

    def test_advertised_value_with_uncertainty_is_accepted(self) -> None:
        receipt = _receipt(
            {"clubhead_speed": {"value": 42.0, "uncertainty": 0.8, "unit": "m/s"}}
        )
        statuses = evaluate_output_policy(receipt)
        assert statuses["clubhead_speed"].status == "advertised"

    def test_non_finite_value_is_rejected(self) -> None:
        receipt = _receipt(
            {"clubhead_speed": {"value": float("nan"), "uncertainty": 1}}
        )
        with pytest.raises(OutputPolicyError, match="clubhead_speed.*finite"):
            evaluate_output_policy(receipt)

    def test_outputs_must_be_a_mapping(self) -> None:
        with pytest.raises(OutputPolicyError, match="outputs must be a mapping"):
            evaluate_output_policy(_receipt(["face_angle"]))


class TestFaceObservabilityPolicy:
    """Unobservable face angle stays unavailable, never validated (#10382)."""

    UNOBSERVABLE = FaceObservability(
        observable=False,
        reason="club head markers are collinear or coincident",
        observed_frames=0,
        total_frames=4,
        gaps=((0, 4),),
    )
    OBSERVABLE = FaceObservability(
        observable=True,
        reason="observable",
        observed_frames=4,
        total_frames=4,
        gaps=(),
    )

    def test_unobservable_face_angle_is_unavailable(self) -> None:
        statuses = evaluate_output_policy(
            _receipt(), face_observability=self.UNOBSERVABLE
        )
        assert statuses["face_angle"].status == "unavailable"
        assert "collinear" in statuses["face_angle"].reason

    def test_unobservable_face_angle_cannot_be_validated_by_evidence(self) -> None:
        receipt = _receipt({"face_angle": {"validated": True, **EVIDENCE}})
        with pytest.raises(OutputPolicyError, match="face_angle.*not observable"):
            evaluate_output_policy(receipt, face_observability=self.UNOBSERVABLE)

    def test_observable_face_angle_keeps_the_withheld_policy(self) -> None:
        statuses = evaluate_output_policy(
            _receipt(), face_observability=self.OBSERVABLE
        )
        assert statuses["face_angle"].status == "withheld"

    def test_unavailable_face_angle_renders_as_unavailable(self) -> None:
        statuses = evaluate_output_policy(
            _receipt(), face_observability=self.UNOBSERVABLE
        )
        table = render_output_policy_markdown(statuses)
        assert "| `face_angle` | **UNAVAILABLE** |" in table


class TestExportReportPolicy:
    def test_report_marks_withheld_outputs_not_validated_and_hides_values(
        self, tmp_path: Path
    ) -> None:
        out_md = tmp_path / "policy.md"
        receipt = _receipt({"face_angle": {"validated": False, "value": 7.77}})
        export_report(receipt, out_md)
        content = out_md.read_text(encoding="utf-8")
        assert "Output Validation Policy" in content
        assert "face_angle" in content and "NOT VALIDATED" in content
        assert "7.77" not in content

    def test_invalid_claim_writes_no_report(self, tmp_path: Path) -> None:
        out_md = tmp_path / "bad.md"
        receipt = _receipt({"impact": {"validated": True, "value": 0.1}})
        with pytest.raises(ValueError, match="impact"):
            export_report(receipt, out_md)
        assert not out_md.exists()


class TestExportReportCLIGuard:
    def test_cli_rejects_invalid_claim_with_exit_one(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import json

        from src.shared.python.motion_matching.__main__ import main

        receipt_file = tmp_path / "receipt.json"
        receipt_file.write_text(
            json.dumps(_receipt({"joint_loads": {"validated": True, "value": 1.0}})),
            encoding="utf-8",
        )
        out_md = tmp_path / "out.md"
        argv = [
            "python",
            "export-report",
            "--receipt",
            str(receipt_file),
            "--out",
            str(out_md),
        ]
        with patch.object(sys, "argv", argv):
            ret = main()
        assert ret == 1
        assert "joint_loads" in capsys.readouterr().err
        assert not out_md.exists()


class TestOutputPolicyPayload:
    """Structured policy payload shared by the report, API and viewers."""

    @staticmethod
    def _by_name(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
        return {row["name"]: row for row in payload["outputs"]}

    def test_every_output_has_status_label_and_reason(self) -> None:
        from src.shared.python.motion_matching.output_policy import (
            output_policy_payload,
        )

        payload = output_policy_payload(_receipt())
        rows = self._by_name(payload)
        assert set(rows) == set(ADVERTISED_OUTPUTS) | set(WITHHELD_OUTPUTS)
        assert rows["clubhead_speed"]["status"] == "advertised"
        assert rows["face_angle"]["status"] == "withheld"
        assert rows["face_angle"]["label"] == "NOT VALIDATED"
        assert rows["face_angle"]["reason"] == "not validated"
        assert payload["face_observability"] is None
        assert "value" not in rows["face_angle"]

    def test_face_audit_coverage_and_gaps_are_structured(self) -> None:
        from src.shared.python.motion_matching.output_policy import (
            output_policy_payload,
        )

        receipt = _receipt()
        receipt["ik"] = {
            "face_observability": {
                "observable": False,
                "reason": "club head markers are collinear or coincident",
                "observed_frames": 1,
                "total_frames": 4,
                "gaps": [[1, 4]],
            }
        }
        payload = output_policy_payload(receipt)
        assert self._by_name(payload)["face_angle"]["status"] == "unavailable"
        face = payload["face_observability"]
        assert face["observable"] is False
        assert face["coverage"] == pytest.approx(0.25)
        assert face["gaps"] == [[1, 4]]

    def test_invalid_receipt_fails_closed_to_unavailable(self) -> None:
        from src.shared.python.motion_matching.output_policy import (
            output_policy_payload,
        )

        payload = output_policy_payload(_receipt({"face_angle": {"validated": True}}))
        for row in payload["outputs"]:
            assert row["status"] == "unavailable"
            assert row["reason"].startswith("policy error:")
        assert payload["face_observability"] is None

    def test_markdown_and_payload_agree_on_status_labels(self) -> None:
        from src.shared.python.motion_matching.output_policy import (
            output_policy_payload,
        )

        payload = output_policy_payload(_receipt())
        markdown = render_output_policy_markdown(evaluate_output_policy(_receipt()))
        for row in payload["outputs"]:
            assert f"| `{row['name']}` | **{row['label']}** |" in markdown
