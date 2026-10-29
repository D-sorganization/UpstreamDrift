"""Cross-engine address knee flexion over the committed receipts (R4, #12684).

The check reads receipts only (no engine SDK), so it runs everywhere. An engine
without a usable receipt is reported ``unavailable`` with a reason, never as 0,
and a malformed receipt fails closed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from scripts import address_knee_engine_check as chk

pytestmark = [pytest.mark.unit]

PINNED = {
    "capture_sha256": "c" * 64,
    "candidate_sha256": "d" * 64,
    "base_spec_sha256": "b" * 64,
    "foot_progression_mode": "style",
}


def _receipt(
    engine: str, left: Any = 28.5, right: Any = 26.1, **over: Any
) -> dict[str, Any]:
    knees: dict[str, Any] = {}
    for side, value in (("left", left), ("right", right)):
        if value is not None:
            knees[side] = {"model_deg": value}
    doc: dict[str, Any] = {
        "backend": "mujoco",
        "address": {
            "knee_flexion_deg": {"enabled": True, "mode": "capture", "knees": knees}
        },
        **PINNED,
    }
    if engine != "mujoco":
        doc["engine"] = engine
    doc.update(over)
    return doc


def _write(root: Path, name: str, doc: Any) -> str:
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    text = doc if isinstance(doc, str) else json.dumps(doc)
    (folder / "receipt.json").write_text(text, encoding="utf-8")
    return name


def _case(root: Path, **receipts: Any) -> chk.Case:
    paths: dict[str, str | None] = dict.fromkeys(chk.ENGINES)
    for engine, doc in receipts.items():
        paths[engine] = _write(root, engine, doc)
    return chk.Case("unit", paths)


@pytest.mark.parametrize(
    "doc",
    [
        _receipt("mujoco", left=None),
        _receipt("mujoco", left=float("nan")),
        _receipt("mujoco", left="28.5"),
        _receipt("mujoco", left=True),
        _receipt("mujoco", left=181.0),
        {**_receipt("mujoco"), "address": {}},
        {**_receipt("mujoco"), "address": []},
        {**_receipt("mujoco"), "address": {"knee_flexion_deg": {"enabled": False}}},
        {**_receipt("mujoco"), "address": {"knee_flexion_deg": {"enabled": True}}},
        _receipt("drake"),  # labelled drake, read as mujoco
    ],
)
def test_malformed_receipt_fails_closed(doc: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        chk.read_model_knees(doc, "mujoco")


def test_agreement_within_tolerance(tmp_path: Path) -> None:
    case = _case(
        tmp_path, mujoco=_receipt("mujoco"), drake=_receipt("drake", 28.55, 26.1)
    )
    out = chk.evaluate_case(case, tmp_path)
    assert out["engines"]["drake"]["available"] is True
    assert out["engines"]["drake"]["delta_deg"]["left"] == pytest.approx(0.05)
    assert out["disagreements"] == []
    assert out["ok"] is True


def test_disagreement_over_tolerance_reports_per_engine_values(tmp_path: Path) -> None:
    case = _case(
        tmp_path, mujoco=_receipt("mujoco"), drake=_receipt("drake", 28.5, 26.4)
    )
    out = chk.evaluate_case(case, tmp_path)
    assert out["ok"] is False
    [bad] = out["disagreements"]
    assert bad["engine"] == "drake" and bad["side"] == "right"
    assert bad["reference_deg"] == 26.1 and bad["engine_deg"] == 26.4
    assert math.isclose(bad["delta_deg"], 0.3, abs_tol=1e-9)
    assert bad["tolerance_deg"] == chk.TOLERANCE_DEG == 0.1


def test_missing_engines_are_unavailable_with_reason_never_zero(tmp_path: Path) -> None:
    out = chk.evaluate_case(_case(tmp_path, mujoco=_receipt("mujoco")), tmp_path)
    for engine in ("drake", "pinocchio", "opensim", "myosuite"):
        entry = out["engines"][engine]
        assert entry["available"] is False
        assert entry["model_deg"] is None
        assert engine in entry["reason"]
    assert out["ok"] is True  # an unavailable engine is reported, not a pass or a fail


@pytest.mark.parametrize(
    ("engine", "doc", "expected"),
    [
        (
            "pinocchio",
            {"status": "blocked", "reason": "SDK unavailable here"},
            "SDK unavailable",
        ),
        (
            "opensim",
            {"backend": "opensim", "note": "dropped address"},
            "dropped address",
        ),
    ],
)
def test_receipt_without_address_reports_its_own_reason(
    tmp_path: Path, engine: str, doc: dict[str, Any], expected: str
) -> None:
    case = _case(
        tmp_path, mujoco=_receipt("mujoco"), **{engine: {"engine": engine, **doc}}
    )
    entry = chk.evaluate_case(case, tmp_path)["engines"][engine]
    assert entry["available"] is False
    assert expected in entry["reason"]


def test_reference_without_knee_block_fails_closed(tmp_path: Path) -> None:
    case = _case(tmp_path, mujoco={"backend": "mujoco", **PINNED})
    with pytest.raises(ValueError, match="mujoco"):
        chk.evaluate_case(case, tmp_path)


def test_engine_receipt_with_address_but_no_knee_block_fails_closed(
    tmp_path: Path,
) -> None:
    drake = {**_receipt("drake"), "address": {"calibrated": {}}}
    case = _case(tmp_path, mujoco=_receipt("mujoco"), drake=drake)
    with pytest.raises(ValueError, match="drake"):
        chk.evaluate_case(case, tmp_path)


@pytest.mark.parametrize("key", sorted(PINNED))
def test_incomparable_receipts_fail_closed(tmp_path: Path, key: str) -> None:
    other = _receipt("drake", **{key: "x"})
    with pytest.raises(ValueError, match=key):
        chk.evaluate_case(
            _case(tmp_path, mujoco=_receipt("mujoco"), drake=other), tmp_path
        )


def test_main_exit_codes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, doc in (
        ("ref", _receipt("mujoco")),
        ("good", _receipt("drake", 28.5, 26.1)),
        ("bad", _receipt("drake", 29.5, 26.1)),
    ):
        _write(tmp_path, name, doc)

    def run(**receipts: str) -> int:
        monkeypatch.setattr(chk, "CASES", (chk.Case("t", dict(receipts)),))
        return chk.main(["--evidence-dir", str(tmp_path)])

    assert run(mujoco="ref", drake="good") == 0
    assert run(mujoco="ref", drake="bad") == 1
    out = capsys.readouterr().out
    assert "DISAGREE" in out and "drake" in out
    assert run(mujoco="nope") == 2


@pytest.fixture(scope="module")
def committed() -> dict[str, Any]:
    return chk.build_report()


def test_committed_driver_drake_agrees_with_mujoco(committed: dict[str, Any]) -> None:
    driver = committed["cases"]["driver"]
    assert driver["ok"] is True
    drake = driver["engines"]["drake"]
    assert drake["available"] is True
    assert max(abs(v) for v in drake["delta_deg"].values()) <= chk.TOLERANCE_DEG


def test_committed_missing_engines_report_a_reason(committed: dict[str, Any]) -> None:
    driver = committed["cases"]["driver"]["engines"]
    assert "SDK unavailable" in driver["pinocchio"]["reason"]
    for engine in ("pinocchio", "opensim", "myosuite"):
        assert driver[engine]["available"] is False
        assert driver[engine]["model_deg"] is None
    iron = committed["cases"]["iron"]["engines"]
    assert not any(iron[e]["available"] for e in ("drake", "pinocchio", "opensim"))
    assert committed["ok"] is True
