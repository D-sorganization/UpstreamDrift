"""Full-mesh MuJoCo diagnostic render contract (Epic #15808, WP-0b)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.force_overlay import ForceTorqueFrame
from src.tools.native_viewer_export.core import (
    ExportSettings,
    OverlayFeed,
    SwingInput,
    default_glyph_style,
)
from src.tools.native_viewer_export.diagnostic_contract import (
    AdmittedInterval,
    ContractViolation,
    DiagnosticRenderRequest,
    ReplayBadge,
    banner_text,
    diagnostic_settings,
    export_diagnostic,
    frame_validity,
)
from src.tools.native_viewer_export.overlay_feed import STANDARD_GRAVITY_M_S2

from tests.unit.tools.native_viewer_export.test_core import (
    _FakeBackend,
    _swing,
    _Writer,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

INTERVAL = AdmittedInterval(0.02, 0.06)


def _request(**overrides) -> DiagnosticRenderRequest:
    """Build a valid diagnostic request, applying ``overrides``."""
    base = {
        "badge": ReplayBadge.ALLOCATION_ONLY,
        "interval": INTERVAL,
        "body_mass_kg": 80.0,
    }
    return DiagnosticRenderRequest(**{**base, **overrides})


class _MujocoBackend(_FakeBackend):
    engine = "mujoco"


def _feed(style=None) -> OverlayFeed:
    """Overlay feed with one empty wrench frame."""
    frame = ForceTorqueFrame(time_s=0.0, engine="mujoco", wrenches=())
    return OverlayFeed(lambda _i: frame, style=style or default_glyph_style())


# --- request contract ---------------------------------------------------


@pytest.mark.parametrize(
    "geometry", ["stick_skeleton", "skeleton", "stick_figure", "wireframe", "lines"]
)
def test_stick_skeleton_geometry_is_forbidden(geometry: str) -> None:
    """Stick and wireframe geometry cannot satisfy the full-mesh contract."""
    with pytest.raises(ContractViolation, match="full_mesh"):
        _request(geometry=geometry)


def test_renderer_must_be_native_viewer_export() -> None:
    """Diagnostics refuse any renderer other than native viewer export."""
    with pytest.raises(ContractViolation, match="native_viewer_export"):
        _request(renderer="matplotlib_stick")


def test_engine_must_be_mujoco() -> None:
    """Diagnostics refuse a non-MuJoCo engine."""
    with pytest.raises(ContractViolation, match="mujoco"):
        _request(engine="drake")


def test_badge_must_be_replay_badge_not_free_text() -> None:
    """The badge must be a ReplayBadge, not a free-text label."""
    with pytest.raises(ContractViolation, match="badge"):
        _request(badge="forward-replayed")


@pytest.mark.parametrize("mass", [0.0, -1.0, math.nan, math.inf])
def test_body_mass_must_be_finite_positive(mass: float) -> None:
    """Body mass must be finite and positive."""
    with pytest.raises(ContractViolation, match="body_mass_kg"):
        _request(body_mass_kg=mass)


def test_interval_is_required() -> None:
    """A diagnostic request without an admitted interval is rejected."""
    with pytest.raises(ContractViolation, match="interval"):
        _request(interval=None)


@pytest.mark.parametrize("bounds", [(0.1, 0.1), (0.2, 0.1), (math.nan, 1.0)])
def test_interval_bounds_validated(bounds: tuple[float, float]) -> None:
    """Empty or non-finite intervals are rejected."""
    with pytest.raises(ContractViolation, match="interval"):
        AdmittedInterval(*bounds)


# --- validity, banner, settings ----------------------------------------


def test_frame_validity_is_inclusive_of_interval_ends() -> None:
    """Frame times on the interval endpoints count as admitted."""
    times = np.array([0.0, 0.02, 0.04, 0.06, 0.08])
    assert frame_validity(times, INTERVAL) == (False, True, True, True, False)


def test_banner_marks_badge_and_interval_inside() -> None:
    """An in-interval banner names the badge and the admitted window."""
    text = banner_text(_request(badge=ReplayBadge.FORWARD_REPLAYED), 0.03)
    assert "FORWARD-REPLAYED" in text
    assert "0.020" in text and "0.060" in text
    assert "OUTSIDE" not in text


def test_banner_flags_frames_outside_admitted_interval() -> None:
    """Frames outside the interval are marked as carrying no claims."""
    text = banner_text(_request(), 0.08)
    assert "OUTSIDE ADMITTED INTERVAL" in text
    assert "ALLOCATION-ONLY" in text


def test_diagnostic_settings_force_overlays_and_hud() -> None:
    """Diagnostic export settings keep overlays, force, torque, and HUD on."""
    s = diagnostic_settings(_request(), views=("face_on",), multiview=False)
    assert isinstance(s, ExportSettings)
    assert s.overlays and s.force_vectors and s.torque_vectors and s.hud
    assert s.views == ("face_on",)


@pytest.mark.parametrize(
    "override", ["overlays", "force_vectors", "torque_vectors", "hud"]
)
def test_diagnostic_settings_reject_disabling_evidence(override: str) -> None:
    """Evidence-carrying settings cannot be switched off."""
    with pytest.raises(ContractViolation, match=override):
        diagnostic_settings(_request(), **{override: False})


# --- export routing ------------------------------------------------------


def _export(tmp_path: Path, request=None, backend=None, overlay="default"):
    """Export one face-on diagnostic clip into ``tmp_path``."""
    writers: dict = {}
    feed = _feed() if overlay == "default" else overlay
    result = export_diagnostic(
        backend or _MujocoBackend(),
        _swing(),
        request or _request(),
        tmp_path,
        overlay=feed,
        writer_factory=lambda p, fps: _Writer(p, fps, writers),
        views=("face_on",),
        multiview=False,
        speeds=(1.0,),
        fps=60,
        width=64,
        height=48,
    )
    return result, writers


def test_export_rejects_non_mujoco_backend(tmp_path: Path) -> None:
    """Export refuses a backend whose engine is not MuJoCo."""

    class _Other(_FakeBackend):
        engine = "drake"

    with pytest.raises(ContractViolation, match="mujoco"):
        _export(tmp_path, backend=_Other())


def test_export_requires_overlay_feed(tmp_path: Path) -> None:
    """Export refuses to render a diagnostic without an overlay feed."""
    with pytest.raises(ContractViolation, match="overlay"):
        _export(tmp_path, overlay=None)


def test_export_burns_banner_and_flags_frames_outside_interval(
    tmp_path: Path,
) -> None:
    """Exported frames differ from the blank plate and record validity."""
    result, writers = _export(tmp_path)
    frames = writers["driver_mujoco_face_on_1x.mp4"].frames
    plain = np.full((48, 64, 3), 40, np.uint8)
    assert len(frames) == len(result.frame_valid["face_on_1x"])
    assert not np.array_equal(frames[0], plain)  # banner drawn outside interval
    assert result.frame_valid["face_on_1x"][0] is False
    assert True in result.frame_valid["face_on_1x"]


def test_export_manifest_records_truth(tmp_path: Path) -> None:
    """The manifest records renderer, badge, interval, and force scale."""
    result, _ = _export(tmp_path, _request(badge=ReplayBadge.FORWARD_REPLAYED))
    data = json.loads(result.manifest_path.read_text())
    assert data["renderer"] == "native_viewer_export"
    assert data["geometry"] == "full_mesh"
    assert data["badge"] == "forward-replayed"
    assert data["admitted_interval_s"] == [0.02, 0.06]
    assert data["force_reference_n"] == pytest.approx(80.0 * STANDARD_GRAVITY_M_S2)
    assert data["force_scale_mode"] == "body_weight"
    assert data["frame_valid"]["face_on_1x"] == list(result.frame_valid["face_on_1x"])


def test_export_applies_shared_body_weight_scale_to_overlay(tmp_path: Path) -> None:
    """The overlay uses the shared body-weight force scale."""
    seen: list = []

    class _Spy(_MujocoBackend):
        def render(self, swing, settings, indices, overlay, ball=None):
            """Capture the overlay style passed into the backend."""
            seen.append(overlay.style)
            yield from super().render(swing, settings, indices, overlay)

    _export(tmp_path, backend=_Spy(), overlay=_feed())
    assert seen and seen[0].scale_mode == "body_weight"
    assert seen[0].reference_force_n == pytest.approx(80.0 * STANDARD_GRAVITY_M_S2)
