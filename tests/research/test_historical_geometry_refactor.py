"""Synthetic geometry-audit parity against integrity-checked archived source."""

import hashlib
import io
import json
import sys
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from tests.research.test_historical_fixture_security import (
    fixture_key_for_source,
    read_fixture,
)

from docs.research.shot_pattern_analysis.phase2.validation import (
    independent_geometry_derivatives as current,
)

pytestmark = pytest.mark.unit


def _archived_module(source):
    path = Path(source.__file__)
    raw = read_fixture(fixture_key_for_source(path), archived=True)
    module = ModuleType("archived_" + path.stem)
    # Both receipts intentionally report the current active filename/hash;
    # the archived implementation itself is verified independently above.
    module.__file__ = str(path)
    # Trusted fixture AST: fixed path/digests verified before extraction; no external code.
    # nosemgrep: python.lang.security.audit.exec-detected.exec-detected
    exec(compile(raw, str(path) + "::<archived>", "exec"), module.__dict__)
    return module


def _pilot(root):
    for club in ("driver", "seven_iron", "pitching_wedge"):
        for lean in (0, 5, 10):
            for curve in (1, 2):
                name = f"{club}_lean{lean}_curve{curve}_rotation1_shaft_only"
                directory = root / name
                directory.mkdir(parents=True)
                data = {
                    "status": "completed",
                    "config": {"club": asdict(current.matched_preset(club))},
                    "nominals": {
                        pattern: {
                            "face_deg": index - 1.0,
                            "path_deg": index * 0.5,
                            "speed_mps": 30.0 + index,
                        }
                        for index, pattern in enumerate(("Straight", "Draw", "Fade"))
                    },
                }
                if lean == 10 and curve == 2:
                    data = {"status": "unavailable", "reason": "synthetic domain"}
                (directory / "summary.json").write_text(json.dumps(data))


def _run(module, pilot, output, mode, monkeypatch):
    calls = []

    class SyntheticPhysics:
        def simulate_delivery(self, face, path, config, *, shaft_rotation_deg):
            calls.append([face, path, asdict(config), shaft_rotation_deg])
            outcome = SimpleNamespace(
                carry_x_m=face
                + 0.5 * path
                + config.club_speed_mps
                + shaft_rotation_deg * (2 + config.dt_s),
                carry_y_m=config.shaft_lean_deg
                + shaft_rotation_deg * (3 - config.dt_s),
            )
            return SimpleNamespace(outcome=outcome)

    snapshots = iter(
        (
            {"source": "before"},
            {"source": "after" if mode == "source_change" else "before"},
        )
    )
    monkeypatch.setattr(module, "MatchedShotPhysics", SyntheticPhysics)
    monkeypatch.setattr(module, "matched_source_snapshot", lambda: next(snapshots))
    if mode == "geometry_error":
        original = module.club_delivery

        def changed(*args):
            result = original(*args)
            return SimpleNamespace(
                face_deg=result.face_deg + args[-1] * 0.1, loft_deg=result.loft_deg
            )

        monkeypatch.setattr(module, "club_delivery", changed)
    monkeypatch.setattr(
        sys, "argv", ["geometry", "--pilot", str(pilot), "--output", str(output)]
    )
    stream = io.StringIO()
    error = None
    with redirect_stdout(stream):
        try:
            module.main()
        except RuntimeError as exc:
            error = str(exc)
    result = json.loads(output.read_text())
    assert (
        result["script_sha256"]
        == hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    )
    return {
        "receipt_bytes": output.read_text(),
        "calls": calls,
        "stdout": stream.getvalue(),
        "error": error,
    }


@pytest.mark.parametrize("mode", ["success", "source_change", "geometry_error"])
def test_geometry_audit_preserves_exact_synthetic_receipt_and_failure_boundary(
    tmp_path, monkeypatch, mode
):
    pilot = tmp_path / "pilot"
    _pilot(pilot)
    old = _archived_module(current)
    with monkeypatch.context() as patch:
        before = _run(old, pilot, tmp_path / "before.json", mode, patch)
    with monkeypatch.context() as patch:
        after = _run(current, pilot, tmp_path / "after.json", mode, patch)
    assert before == after


def test_geometry_checks_cover_exact_ordered_grid():
    rows = current._geometry_checks()
    assert [
        (row["club"], row["lean_deg"], row["nominal_face_deg"]) for row in rows
    ] == [
        (club, lean, face)
        for club in ("driver", "seven_iron", "pitching_wedge")
        for lean in (0.0, 5.0, 10.0)
        for face in (-3.0, 0.0, 3.0)
    ]
    assert all(
        [item["epsilon_deg"] for item in row["finite_differences"]] == [1e-4, 1e-5]
        for row in rows
    )
