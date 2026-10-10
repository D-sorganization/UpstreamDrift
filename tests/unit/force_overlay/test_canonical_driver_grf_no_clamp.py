"""Canonical driver swing is never clamped by the native-export glyph style (GCV-4).

The calibrated driver run ``anthro_driver_seeds`` commits its per-frame dynamics
record beside its receipt.  ``default_glyph_style`` scales force arrows at one
body weight per ``BODY_WEIGHT_ARROW_M`` with a ``max_length_m`` ceiling; these
tests prove the committed series never reaches that ceiling, both from the
recorded ground-reaction series and from the glyphs the native export builds.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.force_overlay import OverlayWrench, WrenchKind
from src.shared.python.force_overlay.contracts import ForceTorqueFrame
from src.shared.python.force_overlay.glyphs import GlyphSet, build_glyphs
from src.tools.native_viewer_export.core import (
    STANDARD_GRAVITY_M_S2,
    default_glyph_style,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

EVIDENCE = (
    Path(__file__).resolve().parents[3]
    / "docs/development/full_body_models/evidence/ground_support/anthro_driver_seeds"
)
RECORD = EVIDENCE / "dynamics_record.npz"
RECEIPT = EVIDENCE / "receipt.json"
SPEC = EVIDENCE / "full_body_spec_hipcal_scaled.json"


def count_clamped_force_arrows(glyph_sets: list[GlyphSet]) -> int:
    """Number of force arrows flagged ``clamped`` across ``glyph_sets``.

    Raises ``TypeError`` for a non-list and ``ValueError`` for an empty list
    (an empty series would pass vacuously).  Postcondition: result >= 0.
    """
    if not isinstance(glyph_sets, list):
        raise TypeError("glyph_sets must be a list of GlyphSet")
    if not glyph_sets:
        raise ValueError("glyph_sets must not be empty")
    return sum(1 for gs in glyph_sets for a in gs.arrows if a.clamped)


def _ceiling_bw(mass_kg: float) -> float:
    """Arrow-length ceiling of the default style expressed in body weights."""
    style = default_glyph_style(body_mass_kg=mass_kg)
    return style.max_length_m / style.reference_length_m


def _load_record() -> dict[str, np.ndarray]:
    assert RECORD.is_file(), f"canonical driver record missing: {RECORD}"
    with np.load(RECORD, allow_pickle=False) as data:
        rec = {k: data[k] for k in data.files}
    n = rec["time_s"].shape[0]
    for key in ("q", "v", "tau", "normal_force_n", "weight_fraction", "cop_m"):
        assert rec[key].shape[0] == n, f"{key} has {rec[key].shape[0]} rows, not {n}"
    return rec


def test_synthetic_over_ceiling_frame_is_reported_clamped() -> None:
    mass = 80.0
    style = default_glyph_style(body_mass_kg=mass)
    bw_n = mass * STANDARD_GRAVITY_M_S2

    def frame(bw: float) -> GlyphSet:
        wrench = OverlayWrench(
            WrenchKind.CONTACT,
            "contact:grf_net",
            "ground",
            (0.0, 0.0, 0.0),
            force_n=(0.0, 0.0, bw * bw_n),
            source="synthetic",
        )
        frame_ = ForceTorqueFrame(
            time_s=0.0, engine="synthetic", wrenches=(wrench,), metadata={}
        )
        return build_glyphs(frame_, style)

    assert count_clamped_force_arrows([frame(2.0), frame(5.9)]) == 0
    assert count_clamped_force_arrows([frame(2.0), frame(7.0), frame(1.0)]) == 1
    with pytest.raises(ValueError):
        count_clamped_force_arrows([])


def test_recorded_grf_peak_is_below_the_clamp_ceiling() -> None:
    rec = _load_record()
    receipt = json.loads(RECEIPT.read_text())
    wf = rec["weight_fraction"]
    normal = rec["normal_force_n"]
    assert np.isfinite(wf).all() and np.isfinite(normal).all()

    # One body weight in newtons, recovered from the record itself.
    loaded = wf > 0.5
    mass_kg = float(np.median(normal[loaded] / wf[loaded])) / STANDARD_GRAVITY_M_S2
    assert 40.0 < mass_kg < 150.0, f"implausible body mass {mass_kg:.1f} kg"

    peak_bw = float(wf.max())
    ceiling = _ceiling_bw(mass_kg)
    assert peak_bw <= ceiling, (
        f"driver GRF peak {peak_bw:.3f} BW exceeds the {ceiling:.2f} BW ceiling"
    )
    assert ceiling - peak_bw > 1.0, (
        f"margin {ceiling - peak_bw:.3f} BW (peak {peak_bw:.3f}, ceiling {ceiling:.2f})"
    )
    # The record is the run behind the committed receipt.
    receipt_peak = receipt["dynamics"]["weight_fraction"]["max"]
    assert peak_bw == pytest.approx(receipt_peak, rel=1e-9), (
        f"record peak {peak_bw} != receipt peak {receipt_peak}"
    )


def test_native_export_glyphs_are_never_clamped_over_the_driver_swing() -> None:
    pytest.importorskip("mujoco")
    from src.engines.physics_engines.mujoco.python.overlay_source import (
        MujocoOverlaySource,
    )
    from src.shared.python.force_overlay.bundle_provider import BundleOverlayProvider
    from src.shared.python.motion_matching.same_input import InputBundle

    rec = _load_record()
    spec_bytes = SPEC.read_bytes()
    names = tuple(json.loads(spec_bytes)["coordinate_order"])
    q, v = rec["q"], rec["v"]
    assert q.shape[1] == len(names)
    dt_s = float(np.median(np.diff(rec["time_s"])))
    bundle = InputBundle(
        spec_bytes=spec_bytes,
        coordinate_order=names,
        dt_s=dt_s,
        q0=q[0],
        v0=v[0],
        efforts=rec["tau"][:-1],
        reference_q=q,
        reference_v=v,
        reference_engine="mujoco",
    )
    source = MujocoOverlaySource(spec_bytes)
    provider = BundleOverlayProvider(bundle, source, source, engine="mujoco", q=q, v=v)
    style = default_glyph_style(body_mass_kg=float(source.total_mass_kg))

    glyph_sets = [build_glyphs(provider.frame_at(i), style) for i in range(len(q))]
    n_clamped = count_clamped_force_arrows(glyph_sets)
    bw_n = float(source.total_mass_kg) * STANDARD_GRAVITY_M_S2
    peak_bw = max(
        (a.magnitude / bw_n for gs in glyph_sets for a in gs.arrows), default=0.0
    )
    assert n_clamped == 0, (
        f"{n_clamped} clamped force arrows over {len(glyph_sets)} frames "
        f"(max arrow {peak_bw:.3f} BW, ceiling {_ceiling_bw(source.total_mass_kg):.2f} BW)"
    )
    net = [
        a.magnitude / bw_n
        for gs in glyph_sets
        for a in gs.arrows
        if a.label == "contact:grf_net"
    ]
    assert net, "no net GRF arrow was built; the check would be vacuous"
