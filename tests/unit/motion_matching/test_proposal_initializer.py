"""Unit tests for ML-WS-8: ProposalInitializer protocol and warm-start hook (#13785)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.shared.python.motion_matching.proposal_initializer import (
    ProposalInitializer,
    StubProposalInitializer,
    validate_proposal,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(300)]


def test_classical_mode_is_bit_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The classical mode (default) gives bit-identical results with or without warm-start code."""
    pytest.importorskip("jax")
    pytest.importorskip("mujoco.mjx")

    from src.shared.python.motion_matching.pipeline.trajectory_optimiser import (
        run_trajectory_optimiser,
    )
    from tests.unit.engines.mujoco.mjx_toy_package import write_toy_package

    write_toy_package(tmp_path)
    monkeypatch.setattr(
        "src.shared.python.motion_matching.execution.mjx_export.export_mjx_package",
        lambda run: {},
    )

    summary1 = run_trajectory_optimiser(
        "mjx-knots", tmp_path, iterations=0, neural_mode="classical"
    )
    assert summary1 is not None

    receipt_path = tmp_path / "mjx_optimisation_receipt.json"
    receipt1 = json.loads(receipt_path.read_text(encoding="utf-8"))

    # Re-run identical settings
    summary2 = run_trajectory_optimiser(
        "mjx-knots", tmp_path, iterations=0, neural_mode="classical"
    )
    assert summary2 is not None
    receipt2 = json.loads(receipt_path.read_text(encoding="utf-8"))

    assert summary1["best_replay_marker_rms_m"] == summary2["best_replay_marker_rms_m"]
    assert receipt1["history"] == receipt2["history"]
    assert "proposal" not in receipt1.get("neural", {})


def test_preview_mode_starts_from_stub_proposal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """In preview mode, optimise_reference starts from the injected stub proposal."""
    pytest.importorskip("jax")
    pytest.importorskip("mujoco.mjx")

    from src.engines.physics_engines.mujoco.python.motion_matching.mjx_knot_optimiser import (
        load_mjx_package,
    )
    from src.shared.python.motion_matching.pipeline.trajectory_optimiser import (
        run_trajectory_optimiser,
    )
    from tests.unit.engines.mujoco.mjx_toy_package import write_toy_package

    write_toy_package(tmp_path)
    monkeypatch.setattr(
        "src.shared.python.motion_matching.execution.mjx_export.export_mjx_package",
        lambda run: {},
    )

    # 1. Classical baseline
    run_trajectory_optimiser(
        "mjx-knots", tmp_path, iterations=0, neural_mode="classical"
    )
    receipt_classical = json.loads(
        (tmp_path / "mjx_optimisation_receipt.json").read_text(encoding="utf-8")
    )
    h0_classical = receipt_classical["history"][0]["total_cost"]

    # 2. Preview mode with non-zero stub proposal
    # knots: 6, actuated: 2 (from toy package: 0.2s duration / 0.04s knot spacing + 1 = 6 knots)
    stub_delta = np.full((6, 2), 0.05, dtype=np.float32)
    stub = StubProposalInitializer(delta=stub_delta, source_id="test_stub")

    run_trajectory_optimiser(
        "mjx-knots",
        tmp_path,
        iterations=0,
        neural_mode="preview",
        proposal_initializer=stub,
    )
    receipt_preview = json.loads(
        (tmp_path / "mjx_optimisation_receipt.json").read_text(encoding="utf-8")
    )
    h0_preview = receipt_preview["history"][0]["total_cost"]

    # Non-zero initial delta must produce different initial cost
    assert h0_preview != pytest.approx(h0_classical, rel=1e-3)
    proposal_info = receipt_preview["neural"]["proposal"]
    assert proposal_info["accepted"] is True
    assert proposal_info["source"] == "test_stub"
    assert proposal_info["fallback_reason"] is None


def test_invalid_proposal_falls_back_and_is_logged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Invalid proposals (wrong shape, NaN, inf) fall back to classical start and log reason."""
    pytest.importorskip("jax")
    pytest.importorskip("mujoco.mjx")

    from src.shared.python.motion_matching.pipeline.trajectory_optimiser import (
        run_trajectory_optimiser,
    )
    from src.tools.motion_matching.neural_cli import NeuralRunRejectedError
    from tests.unit.engines.mujoco.mjx_toy_package import write_toy_package

    write_toy_package(tmp_path)
    monkeypatch.setattr(
        "src.shared.python.motion_matching.execution.mjx_export.export_mjx_package",
        lambda run: {},
    )

    # A) Wrong shape -> fall back to classical
    bad_shape_delta = np.zeros((3, 7), dtype=np.float32)
    bad_stub = StubProposalInitializer(delta=bad_shape_delta, source_id="bad_shape")
    run_trajectory_optimiser(
        "mjx-knots",
        tmp_path,
        iterations=0,
        neural_mode="preview",
        allow_fallback=True,
        proposal_initializer=bad_stub,
    )
    receipt_fb = json.loads(
        (tmp_path / "mjx_optimisation_receipt.json").read_text(encoding="utf-8")
    )
    prop = receipt_fb["neural"]["proposal"]
    assert prop["accepted"] is False
    assert prop["fallback_reason"] is not None
    assert "shape" in prop["fallback_reason"].lower()

    # B) Wrong shape with allow_fallback=False -> raises error
    with pytest.raises((NeuralRunRejectedError, ValueError)):
        run_trajectory_optimiser(
            "mjx-knots",
            tmp_path,
            iterations=0,
            neural_mode="preview",
            allow_fallback=False,
            proposal_initializer=bad_stub,
        )

    # C) NaN values -> fall back to classical
    nan_delta = np.full((6, 2), np.nan, dtype=np.float32)
    nan_stub = StubProposalInitializer(delta=nan_delta, source_id="nan_stub")
    run_trajectory_optimiser(
        "mjx-knots",
        tmp_path,
        iterations=0,
        neural_mode="preview",
        allow_fallback=True,
        proposal_initializer=nan_stub,
    )
    receipt_nan = json.loads(
        (tmp_path / "mjx_optimisation_receipt.json").read_text(encoding="utf-8")
    )
    prop_nan = receipt_nan["neural"]["proposal"]
    assert prop_nan["accepted"] is False
    assert (
        "finite" in prop_nan["fallback_reason"].lower()
        or "nan" in prop_nan["fallback_reason"].lower()
    )


def test_receipt_records_proposal_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The receipt records the sha256 checksum of the accepted proposal."""
    pytest.importorskip("jax")
    pytest.importorskip("mujoco.mjx")

    from src.shared.python.motion_matching.pipeline.trajectory_optimiser import (
        run_trajectory_optimiser,
    )
    from tests.unit.engines.mujoco.mjx_toy_package import write_toy_package

    write_toy_package(tmp_path)
    monkeypatch.setattr(
        "src.shared.python.motion_matching.execution.mjx_export.export_mjx_package",
        lambda run: {},
    )

    stub_delta = np.full((6, 2), 0.02, dtype=np.float32)
    expected_hash = hashlib.sha256(stub_delta.tobytes()).hexdigest()
    stub = StubProposalInitializer(delta=stub_delta, source_id="hash_test")

    run_trajectory_optimiser(
        "mjx-knots",
        tmp_path,
        iterations=0,
        neural_mode="preview",
        proposal_initializer=stub,
    )
    receipt = json.loads(
        (tmp_path / "mjx_optimisation_receipt.json").read_text(encoding="utf-8")
    )
    prop = receipt["neural"]["proposal"]
    assert prop["accepted"] is True
    assert prop["sha256"] == expected_hash
    assert prop["source"] == "hash_test"
