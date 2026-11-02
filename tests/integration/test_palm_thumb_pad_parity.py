"""Palm-thumb pad parity across engines (HRG-H2, #13190 PR 2).

The pad never uses native collision geometry: every engine supplies only its
own forward kinematics of the two grip-hand frames, and the force comes from
the one shared law (:func:`palm_thumb_pad.palm_thumb_pad_force`). Parity is
therefore that law fed each engine's frames along a prescribed trail-wrist
oscillation, against the same law fed the spec-analytic frames, within the
shared :data:`parity.PEAK_TOLERANCE` / :data:`parity.RMS_TOLERANCE`.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.grip_contact.parity import (
    PEAK_TOLERANCE,
    RMS_TOLERANCE,
    compute_quantity_error,
)
from tests.helpers.hand_consistency import (
    GATING_ENGINES,
    preload_engines,
    spec_frame_poses,
)
from tests.helpers.palm_thumb_pad_parity import (
    PAD_QUANTITIES,
    engine_pad_series,
    pad_parity_doc,
    pad_series,
    spec_pad_series,
)

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module", autouse=True)
def _engines_loaded() -> None:
    preload_engines()


@pytest.fixture(scope="module")
def doc() -> dict:
    return pad_parity_doc()


@pytest.fixture(scope="module")
def reference(doc: dict) -> dict:
    return spec_pad_series(doc)


def test_reference_pad_stays_loaded_and_varies(reference: dict) -> None:
    normal = reference["normal_force"]
    assert (normal > 0.0).all()  # in contact at every sample
    assert normal.max() - normal.min() > 0.1 * normal.max()  # not a constant


@pytest.mark.parametrize("engine", GATING_ENGINES)
def test_palm_thumb_pad_parity(engine: str, doc: dict, reference: dict) -> None:
    series = engine_pad_series(engine, doc)
    for name in PAD_QUANTITIES:
        error = compute_quantity_error(series[name], reference[name], name=name)
        assert error.peak_error <= PEAK_TOLERANCE, (engine, name, error.to_dict())
        assert error.rms_error <= RMS_TOLERANCE, (engine, name, error.to_dict())


def test_parity_detects_a_shifted_hand_frame(doc: dict, reference: dict) -> None:
    def shifted(q: dict[str, float]) -> dict:
        poses = {
            k: np.array(v, dtype=float) for k, v in spec_frame_poses(doc, q).items()
        }
        poses["H2PAD_R"][:3, 3] += (5.0e-5, 0.0, 0.0)  # 0.05 mm on the trail hand
        return poses

    error = compute_quantity_error(
        pad_series(shifted, doc)["normal_force"], reference["normal_force"]
    )
    assert error.peak_error > PEAK_TOLERANCE  # the gate sees a 0.05 mm frame error
