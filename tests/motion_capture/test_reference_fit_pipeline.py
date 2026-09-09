"""The real solver produces reusable reference assets, with honest residuals."""

import json
from pathlib import Path

import numpy as np
import pytest

from src.motion_capture.reconstruct.model import ArticulatedModel, FitOptions
from src.motion_capture.reconstruct.model.registry import get_model
from src.motion_capture.reference.fit_pipeline import fit_reference, save_reference_fit
from src.motion_capture.reference.fitting import MarkerProfile
from src.motion_capture.reference.importers import MotionDraft
from src.motion_capture.reference.model import ReferenceSource
from src.motion_capture.reference.storage import ReferenceLibrary

pytestmark = pytest.mark.unit


def pendulum_input() -> tuple[MotionDraft, MarkerProfile]:
    model = ArticulatedModel(get_model("double_pendulum").spec)
    q = np.zeros((5, model.n_dof))
    q[:, 1] = 1.5
    q[:, -1] = np.linspace(0.1, 0.2, 5)
    points = model.landmarks(q)
    source = ReferenceSource(path="synthetic.c3d", sha256="b" * 64, format="c3d")
    draft = MotionDraft(
        source, ("pivot", "hands"), tuple(np.arange(5) / 30), points, "m", True
    )
    profile = MarkerProfile(
        name="synthetic",
        joints={
            "left_shoulder": ("pivot",),
            "right_shoulder": ("pivot",),
            "left_wrist": ("hands",),
            "right_wrist": ("hands",),
        },
    )
    return draft, profile


def test_real_fit_roundtrip_and_forward_kinematics(tmp_path: Path) -> None:
    draft, profile = pendulum_input()
    result = fit_reference(
        draft, profile, "double_pendulum", options=FitOptions(max_iterations=30)
    )
    assert result.report["all_observed_rms_m"] < 0.005
    assert result.asset.edges == ((0, 1),)
    assert result.asset.time_s == draft.time_s
    points = np.asarray(result.asset.points_m)
    np.testing.assert_allclose(
        np.linalg.norm(points[:, 1] - points[:, 0], axis=1), 0.62
    )
    output = save_reference_fit(result, tmp_path / "fit")
    payload = json.loads((output / "manifest.json").read_text())
    assert payload["profile"]["name"] == "synthetic"
    assert payload["model_spec"]["name"] == get_model("double_pendulum").spec.name
    assert payload["source"]["sha256"] == "b" * 64
    loaded = ReferenceLibrary(output / "references").load(result.asset.id)
    assert loaded == result.asset
    assert np.load(output / "q.npy").shape == result.fit.q.shape
    assert np.load(output / "observed_m.npy").shape == result.fit.landmarks_m.shape


def test_deterministic_identity_and_fit() -> None:
    draft, profile = pendulum_input()
    first = fit_reference(
        draft, profile, "double_pendulum", options=FitOptions(max_iterations=3)
    )
    second = fit_reference(
        draft, profile, "double_pendulum", options=FitOptions(max_iterations=3)
    )
    assert first.asset.id == second.asset.id
    np.testing.assert_array_equal(first.fit.q, second.fit.q)


def test_unknown_model_and_unobserved_model_rejected() -> None:
    draft, _ = pendulum_input()
    profile = MarkerProfile(name="no-match", joints={"left_ankle": ("pivot",)})
    with pytest.raises(ValueError, match="observ"):
        fit_reference(draft, profile, "double_pendulum")


def test_nonuniform_clock_rejected() -> None:
    draft, profile = pendulum_input()
    from dataclasses import replace

    irregular = replace(draft, time_s=(0, 0.1, 0.2, 0.4, 0.5))
    with pytest.raises(ValueError, match="uniform"):
        fit_reference(irregular, profile, "double_pendulum")


def test_direct_registered_model_supports_external_adapters() -> None:
    draft, profile = pendulum_input()
    result = fit_reference(draft, profile, get_model("double_pendulum"))
    assert result.asset.model_identity == get_model("double_pendulum").spec.name


def test_identity_covers_actual_samples_and_not_source_location() -> None:
    from dataclasses import replace

    draft, profile = pendulum_input()
    first = fit_reference(draft, profile, "double_pendulum")
    moved = replace(
        draft, source=draft.source.model_copy(update={"path": "other/capture.c3d"})
    )
    second = fit_reference(moved, profile, "double_pendulum")
    assert first.asset.id == second.asset.id
    assert first.asset.title != second.asset.title
    assert "capture" in second.asset.title
    assert "30 Hz" in second.asset.title
    changed = replace(draft, points=draft.points + 0.01)
    third = fit_reference(changed, profile, "double_pendulum")
    assert first.asset.id != third.asset.id


@pytest.mark.parametrize("invalid_length", [-0.1, float("nan"), float("inf")])
def test_invalid_fitted_dimensions_cannot_be_published(
    monkeypatch: pytest.MonkeyPatch, invalid_length: float
) -> None:
    from dataclasses import replace

    draft, profile = pendulum_input()
    valid = fit_reference(draft, profile, "double_pendulum")
    lengths = dict(valid.fit.lengths_m)
    lengths[next(iter(lengths))] = invalid_length
    invalid = replace(valid.fit, lengths_m=lengths)
    monkeypatch.setattr(
        "src.motion_capture.reference.fit_pipeline.fit_trajectory",
        lambda *args, **kwargs: invalid,
    )
    with pytest.raises(ValueError, match="dimensions"):
        fit_reference(draft, profile, "double_pendulum")
