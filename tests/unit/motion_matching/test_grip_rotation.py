"""Setting a committed document's hand rotation to a fitted total (#12221)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching import grip_fit
from src.shared.python.motion_matching.execution.grip_rotation import (
    apply_grip_rotation,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
DOCUMENT = (
    REPO_ROOT
    / "docs"
    / "development"
    / "full_body_models"
    / "full_body_spec_anthro_driver.json"
)
BASE = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
HAND = {"L": "Clubface Vector", "R": "RHandStandoff"}


@pytest.fixture(scope="module")
def document() -> dict:
    return json.loads(DOCUMENT.read_text(encoding="utf-8"))


def _c2f(doc: dict, side: str) -> np.ndarray:
    joint = next(j for j in doc["joints"] if j["child"].endswith(HAND[side]))
    return np.asarray(joint["child_to_follower"], dtype=float)


def test_same_target_leaves_the_document_unchanged(document: dict) -> None:
    current = document["subject"]["grip_rotation_deg"]
    out, changed = apply_grip_rotation(document, {"L": current["L"], "R": current["R"]})
    assert changed is False
    assert out == document


def test_new_target_recomposes_the_wrist_rotation(document: dict) -> None:
    before = copy.deepcopy(document)
    target = (-80.0, 40.0, 12.5)
    out, changed = apply_grip_rotation(document, {"L": target}, note="; test note")
    assert document == before  # input is not mutated
    assert changed is True
    current = grip_fit.grip_rotation(before["subject"]["grip_rotation_deg"]["L"])
    expected = _c2f(before, "L")
    expected[:3, :3] = expected[:3, :3] @ current.T @ grip_fit.grip_rotation(target)
    assert np.allclose(_c2f(out, "L"), expected, atol=1e-12)
    assert out["subject"]["grip_rotation_deg"]["L"] == list(target)
    assert out["provenance"] == before["provenance"] + "; test note"


def test_other_side_is_untouched(document: dict) -> None:
    out, _ = apply_grip_rotation(document, {"L": (1.0, 2.0, 3.0)})
    assert np.array_equal(_c2f(out, "R"), _c2f(document, "R"))
    assert (
        out["subject"]["grip_rotation_deg"]["R"]
        == document["subject"]["grip_rotation_deg"]["R"]
    )


def test_second_application_is_a_no_op(document: dict) -> None:
    targets = {"L": (-80.0, 40.0, 12.5), "R": (-30.0, 50.0, 60.0)}
    once, changed = apply_grip_rotation(document, targets, note="; n")
    twice, changed_again = apply_grip_rotation(once, targets, note="; n")
    assert changed is True and changed_again is False
    assert twice == once


def test_fitted_angles_follow_the_recomposed_chain(document: dict) -> None:
    """Re-composing c2f moves the relative rotation by ``G(R).T @ G(T)``."""
    side, target = "R", (10.0, -20.0, 30.0)
    current = grip_fit.grip_rotation(document["subject"]["grip_rotation_deg"][side])
    out, _ = apply_grip_rotation(document, {side: target})
    rng = np.random.default_rng(0)
    # Hand-body rotations (frames, 3, 3); relative = hand @ c2f.
    hand = grip_fit.wrist_rotation(rng.uniform(-0.5, 0.5, size=(20, 3)), BASE)
    old = hand @ _c2f(document, side)[:3, :3]
    new = hand @ _c2f(out, side)[:3, :3]
    shifted = old @ current.T @ grip_fit.grip_rotation(target)
    assert np.allclose(new, shifted, atol=1e-12)
    assert np.allclose(
        grip_fit.wrist_angles(new, BASE), grip_fit.wrist_angles(shifted, BASE)
    )


@pytest.mark.parametrize(
    "targets",
    [
        {"X": (0.0, 0.0, 0.0)},
        {"L": (0.0, 0.0)},
        {"L": (0.0, float("nan"), 0.0)},
        {"L": (0.0, float("inf"), 0.0)},
    ],
)
def test_invalid_targets_raise(document: dict, targets: dict) -> None:
    with pytest.raises(ValueError):
        apply_grip_rotation(document, targets)


def test_document_without_a_grip_rotation_raises(document: dict) -> None:
    bare = copy.deepcopy(document)
    del bare["subject"]["grip_rotation_deg"]["L"]
    with pytest.raises(ValueError, match="grip_rotation_deg"):
        apply_grip_rotation(bare, {"L": (0.0, 0.0, 0.0)})
