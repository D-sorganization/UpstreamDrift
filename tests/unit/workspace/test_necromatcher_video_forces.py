"""Opt-in force/torque layer on the Necromatcher fitted-model video export (#11313)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

# Reuse the real two-frame capture fixture; pytest's default import mode puts this
# directory on sys.path, so the sibling module is importable by name.
from test_necromatcher_video import video_case  # noqa: F401

pytestmark = pytest.mark.unit

QUALIFICATION = "research fit - unqualified camera and dynamics; not measured forces"


def _factory():
    from src.shared.python.workspace.necromatcher_video_worker import (
        mujoco_force_sampler,
    )

    return mujoco_force_sampler


def _layer(**overrides):
    from src.shared.python.workspace.necromatcher_video_forces import ForceLayer

    return ForceLayer(**{"enabled": True, **overrides})


def _with_derivatives(library, source: Path, payload: dict) -> None:
    """Attach a preserved cubic-Hermite spline so v and a exist for the fit."""
    original = payload["evidence"]["original_fit"]
    free = payload["coordinate_order"][3:5]
    original.update(
        free_coordinates=free,
        coordinate_order=payload["coordinate_order"],
        knot_times=[0.0, 1 / 30],
        spline_coefficients=[0.0, 0.0, 0.1, 0.1, 0.5, 0.5, 0.5, 0.5],
    )
    source.write_text(json.dumps(payload))
    library.add_fit("force-fit", "practice", source)


def _decode(path: Path) -> list[np.ndarray]:
    import cv2

    reader = cv2.VideoCapture(str(path))
    try:
        frames = []
        while True:
            ok, image = reader.read()
            if not ok:
                return frames
            frames.append(image)
    finally:
        reader.release()


def test_force_layer_draws_glyphs_and_records_receipts(video_case, tmp_path):  # noqa: F811
    from src.shared.python.workspace.necromatcher_video import export_fit_video

    library, _, payload = video_case
    _with_derivatives(library, tmp_path / "fit.json", payload)
    off = export_fit_video(library, "force-fit", tmp_path / "off")
    on = export_fit_video(
        library,
        "force-fit",
        tmp_path / "on",
        force_layer=_layer(),
        force_sampler_factory=_factory(),
    )
    assert on["force_layer"]["available"] is True
    assert on["force_layer"]["settings"]["kinds"] == ["joint_reaction"]
    assert [row["force_glyphs"]["drawn"] > 0 for row in on["frames"]] == [True, True]
    plain = _decode(tmp_path / "off" / "overlay.mp4")
    layered = _decode(tmp_path / "on" / "overlay.mp4")
    assert all(np.any(a != b) for a, b in zip(plain, layered, strict=True))
    assert "force_layer" not in off


def test_layer_off_is_byte_identical_to_the_existing_export(video_case, tmp_path):  # noqa: F811
    from src.shared.python.workspace.necromatcher_video import export_fit_video

    library, _, _ = video_case
    export_fit_video(library, "video-fit", tmp_path / "a")
    export_fit_video(
        library,
        "video-fit",
        tmp_path / "b",
        force_layer=_layer(enabled=False),
    )
    assert (tmp_path / "a" / "overlay.mp4").read_bytes() == (
        tmp_path / "b" / "overlay.mp4"
    ).read_bytes()
    manifest = json.loads((tmp_path / "b" / "manifest.json").read_text())
    assert "force_layer" not in manifest


def test_fit_without_derivatives_exports_kinematic_only(video_case, tmp_path):  # noqa: F811
    from src.shared.python.workspace.necromatcher_video import export_fit_video

    library, _, _ = video_case
    manifest = export_fit_video(
        library,
        "video-fit",
        tmp_path / "kinematic",
        force_layer=_layer(),
        force_sampler_factory=_factory(),
    )
    assert manifest["force_layer"]["available"] is False
    assert "derivatives" in manifest["force_layer"]["reason"]
    assert all("force_glyphs" not in row for row in manifest["frames"])
    assert len(_decode(tmp_path / "kinematic" / "overlay.mp4")) == 2


def test_legend_carries_the_qualification_text(video_case, tmp_path, monkeypatch):  # noqa: F811
    from src.shared.python.force_overlay.renderers import opencv_glyphs
    from src.shared.python.workspace import necromatcher_video_forces as forces
    from src.shared.python.workspace.necromatcher_video import export_fit_video

    library, _, payload = video_case
    _with_derivatives(library, tmp_path / "fit.json", payload)
    seen = []
    real = opencv_glyphs.draw_glyphs_on_frame

    def spy(*args, **kwargs):
        seen.append(kwargs.get("qualification"))
        return real(*args, **kwargs)

    monkeypatch.setattr(forces, "draw_glyphs_on_frame", spy)
    export_fit_video(
        library,
        "force-fit",
        tmp_path / "legend",
        force_layer=_layer(),
        force_sampler_factory=_factory(),
    )
    assert seen == [QUALIFICATION] * 2
    assert forces.QUALIFICATION == QUALIFICATION


def test_segment_shading_is_optional_and_recorded(video_case, tmp_path):  # noqa: F811
    from src.shared.python.workspace.necromatcher_video import export_fit_video

    library, _, payload = video_case
    _with_derivatives(library, tmp_path / "fit.json", payload)
    manifest = export_fit_video(
        library,
        "force-fit",
        tmp_path / "shaded",
        force_layer=_layer(segment_shading=True),
        force_sampler_factory=_factory(),
    )
    assert manifest["force_layer"]["settings"]["segment_shading"] is True
    assert all("segment_shading" in row for row in manifest["frames"])


def test_enabled_layer_without_a_provider_factory_fails_loudly(video_case, tmp_path):  # noqa: F811
    from src.shared.python.workspace.necromatcher_video import export_fit_video

    with pytest.raises(ValueError, match="force sampler"):
        export_fit_video(
            video_case[0], "video-fit", tmp_path / "nope", force_layer=_layer()
        )
    assert not (tmp_path / "nope").exists()


@pytest.mark.parametrize(
    "bad", [{"kinds": ("nonsense",)}, {"scale": 0.0}, {"scale": float("nan")}]
)
def test_layer_settings_reject_invalid_values(bad):
    with pytest.raises(ValueError):
        _layer(**bad)
