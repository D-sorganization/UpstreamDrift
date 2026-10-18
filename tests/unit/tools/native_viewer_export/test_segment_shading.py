"""Tension/compression shading and vector toggles of the native export (#13146)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.body_part_viz.axial_loads import AxialLoadFrame
from src.shared.python.body_part_viz.force_colors import ForceColorScale
from src.shared.python.force_overlay.bundle_provider import (
    AxialLoadSource,
    BundleOverlayProvider,
)
from src.shared.python.force_overlay.contracts import (
    ForceTorqueFrame,
    OverlayWrench,
    WrenchKind,
)
from src.shared.python.motion_matching.same_input import InputBundle
from src.tools.native_viewer_export import cli
from src.tools.native_viewer_export.backends._subprocess import stage_job
from src.tools.native_viewer_export.backends.registry import make_backend
from src.tools.native_viewer_export.compositor import HudInfo
from src.tools.native_viewer_export.core import (
    ExportSettings,
    OverlayFeed,
    SwingInput,
)
from src.tools.native_viewer_export.shading import (
    frames_from_json,
    frames_to_json,
    shading_legend,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

ROOT = Path(__file__).resolve().parents[4]
SPEC = ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
BASE = ["--bundle", "b.npz", "--out", "o", "--swing", "s"]


def _settings(*extra: str) -> ExportSettings:
    return cli.build_settings(cli.build_parser().parse_args([*BASE, *extra]))


def _wrench_frame(time_s: float = 0.0, axial: AxialLoadFrame | None = None):
    return ForceTorqueFrame(
        time_s=time_s,
        engine="mujoco",
        wrenches=(
            OverlayWrench(
                WrenchKind.CONTACT,
                "contact:foot",
                "foot",
                (0, 0, 0),
                (0, 0, 400.0),
                source="t",
            ),
            OverlayWrench(
                WrenchKind.JOINT_ACTUATOR,
                "actuator:hip",
                "hip",
                (0, 0, 1.0),
                torque_nm=(0, 0, 90.0),
                source="t",
            ),
        ),
        axial_loads=axial,
    )


# --- CLI -----------------------------------------------------------------


def test_defaults_preserve_current_output() -> None:
    s = _settings()
    assert s.shading is None and not s.shading_enabled
    assert s.overlays and s.force_vectors and s.torque_vectors


def test_shading_options_build_one_scale() -> None:
    s = _settings(
        "--segment-shading",
        "--shading-opacity",
        "0.25",
        "--shading-mode",
        "replace",
        "--tension-color",
        "#00FF00",
        "--compression-color",
        "#ffaa00",
        "--shading-limit-n",
        "800",
        "--compression-limit-n",
        "1200",
    )
    assert s.shading == ForceColorScale(
        enabled=True,
        tension_limit_n=800.0,
        compression_limit_n=1200.0,
        tension_color="#00ff00",
        compression_color="#ffaa00",
        opacity=0.25,
        blend_mode="replace",
    )
    assert s.shading_enabled


@pytest.mark.parametrize(
    "bad",
    [
        ["--tension-color", "blue"],
        ["--compression-color", "#12345"],
        ["--shading-opacity", "1.5"],
        ["--shading-opacity", "-0.1"],
        ["--shading-limit-n", "0"],
    ],
)
def test_invalid_shading_options_exit_with_error(
    bad: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main([*BASE, "--segment-shading", *bad]) == 2
    assert "error:" in capsys.readouterr().err


def test_invalid_mode_is_an_argparse_error() -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([*BASE, "--shading-mode", "glow"])


def test_no_overlay_disables_everything() -> None:
    s = _settings("--no-overlay", "--segment-shading")
    assert not s.overlays and s.shading is None and not s.shading_enabled


def test_vector_toggles_parse() -> None:
    s = _settings("--no-force-vectors", "--torque-vectors")
    assert (s.force_vectors, s.torque_vectors) == (False, True)


def test_help_documents_the_options() -> None:
    text = cli.build_parser().format_help()
    for flag in (
        "--segment-shading",
        "--shading-opacity",
        "--shading-mode",
        "--tension-color",
        "--compression-color",
        "--shading-limit-n",
        "--force-vectors",
        "--torque-vectors",
    ):
        assert flag in text


def test_settings_reject_wrong_types() -> None:
    with pytest.raises(TypeError):
        ExportSettings(shading="red")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ExportSettings(force_vectors=1)  # type: ignore[arg-type]


# --- glyph toggles and legend --------------------------------------------


@pytest.mark.parametrize(
    ("force", "torque", "arrows", "arcs"),
    [(True, True, True, True), (False, True, False, True), (True, False, True, False)],
)
def test_feed_restricts_glyph_kinds(
    force: bool, torque: bool, arrows: bool, arcs: bool
) -> None:
    feed = OverlayFeed(lambda _i: _wrench_frame(), show_force=force, show_torque=torque)
    glyphs = feed.glyphs_at(0)
    assert bool(glyphs.arrows) is arrows
    assert bool(glyphs.torque_arcs) is arcs


def test_restrict_frame_is_shared_with_the_mujoco_overlay_module() -> None:
    from src.shared.python.force_overlay.conversions import restrict_frame

    pytest.importorskip("mujoco")
    from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf import (
        force_glyph_overlay,
    )

    assert force_glyph_overlay.restrict_frame is restrict_frame


def test_legend_names_colours_and_saturation() -> None:
    scale = ForceColorScale(enabled=True, tension_limit_n=900.0, opacity=0.5)
    line = shading_legend(scale)
    assert "Tension #0000ff" in line and "Compression #ff0000" in line
    assert "900 N" in line
    hud = HudInfo(0.0, "MuJoCo", "Driver", "", shading_legend=line)
    assert hud.lines()[-1] == line
    assert len(HudInfo(0.0, "MuJoCo", "Driver", "").lines()) == 1
    with pytest.raises(TypeError):
        shading_legend("red")  # type: ignore[arg-type]


# --- provider and worker payload -----------------------------------------


class _Kin:
    total_mass_kg = 80.0

    def joint_frames(self, coordinates):
        return {}

    def center_of_mass_m(self, coordinates):
        return (0.0, 0.0, 1.0)

    def evaluate_contact_samples(self, coordinates, rates):
        return {}


class _Axial:
    def __init__(self) -> None:
        self.calls: list[tuple[dict, dict, dict, float]] = []

    def axial_load_frame(
        self, coordinates, rates, accelerations, time_s, external_wrenches=None
    ):
        self.calls.append((coordinates, rates, accelerations, time_s))
        self.wrenches = external_wrenches
        return AxialLoadFrame(time_s, {"seg": 1.0}, "fake")


def _tiny_bundle(steps: int = 4, dt: float = 0.01) -> InputBundle:
    t = np.arange(steps + 1) * dt
    q = np.stack([t, 0 * t], axis=1)
    v = np.stack([3.0 * t, 0 * t], axis=1)  # constant acceleration 3
    return InputBundle(
        spec_bytes=json.dumps({"coordinate_order": ["A", "B"]}).encode(),
        coordinate_order=("A", "B"),
        dt_s=dt,
        q0=q[0],
        v0=v[0],
        efforts=np.zeros((steps, 2)),
        reference_q=q,
        reference_v=v,
        reference_engine="mujoco",
    )


def test_provider_attaches_axial_loads_from_its_own_kinematics() -> None:
    bundle, axial = _tiny_bundle(), _Axial()
    kin = _Kin()
    assert isinstance(axial, AxialLoadSource)
    provider = BundleOverlayProvider(bundle, kin, kin, engine="mujoco")
    assert provider.frame_at(1).axial_loads is None
    assert provider.with_axial_loads(axial) is provider
    frame = provider.frame_at(2)
    q, v, a, t = axial.calls[-1]
    assert q["A"] == pytest.approx(0.02) and v["A"] == pytest.approx(0.06)
    assert a["A"] == pytest.approx(3.0) and a["B"] == 0.0
    assert t == pytest.approx(0.02) and frame.axial_loads.time_s == t
    with pytest.raises(TypeError):
        provider.with_axial_loads(object())  # type: ignore[arg-type]


def test_provider_passes_its_contact_wrenches_to_the_axial_source() -> None:
    """One source of truth: the drawn ground wrenches also support the legs."""
    from types import SimpleNamespace

    sample = SimpleNamespace(
        normal_force_n=(0.0, 0.0, 400.0),
        friction_force_n=(0.0, 0.0, 0.0),
        contact_point_m=(0.1, 0.0, 0.0),
    )

    class _Ground(_Kin):
        def evaluate_contact_samples(self, coordinates, rates):
            return {"sphere_foot": sample}

    kin, axial = _Ground(), _Axial()
    provider = BundleOverlayProvider(
        _tiny_bundle(), kin, kin, engine="mujoco"
    ).with_axial_loads(axial)
    provider.frame_at(1)
    (wrench,) = axial.wrenches
    assert wrench.force_n == (0.0, 0.0, 400.0) and wrench.point_m == (0.1, 0.0, 0.0)
    assert wrench.body == "foot"


def test_axial_frames_round_trip_json(tmp_path: Path) -> None:
    frames = [AxialLoadFrame(0.5, {"a": 3.0, "b": None}, "src"), None]
    frames_to_json(frames, tmp_path / "f.json")
    back = frames_from_json(tmp_path / "f.json")
    assert back[1] is None
    assert back[0].time_s == 0.5 and dict(back[0].values_n) == {"a": 3.0, "b": None}
    with pytest.raises(TypeError):
        frames_to_json([{"a": 1}], tmp_path / "x.json")  # type: ignore[list-item]


def _swing(spec_bytes: bytes | None = None, steps: int = 2) -> SwingInput:
    if spec_bytes is None:
        bundle = _tiny_bundle(steps)
    else:
        names = tuple(json.loads(spec_bytes)["coordinate_order"])
        q = np.zeros((steps + 1, len(names)))
        bundle = InputBundle(
            spec_bytes=spec_bytes,
            coordinate_order=names,
            dt_s=0.001,
            q0=q[0],
            v0=q[0],
            efforts=np.zeros((steps, len(names))),
            reference_q=q,
            reference_v=q.copy(),
            reference_engine="mujoco",
        )
    return SwingInput(bundle, bundle.reference_q.copy(), "t", "Driver", "mujoco")


def _axial_feed(value: float | None) -> OverlayFeed:
    def frame_at(i: int) -> ForceTorqueFrame:
        loads = (
            None
            if value is None
            else AxialLoadFrame(0.0, {"GolfSwing3D_Kinetic/Head": value}, "t")
        )
        return ForceTorqueFrame(time_s=0.0, engine="mujoco", axial_loads=loads)

    return OverlayFeed(frame_at)


def test_scale_and_loads_reach_the_worker_payload(tmp_path: Path) -> None:
    scale = ForceColorScale(
        enabled=True, opacity=0.4, blend_mode="replace", tension_color="#00ff00"
    )
    settings = ExportSettings(shading=scale, views=("face_on",), multiview=False)
    (tmp_path / "w").mkdir()
    job = stage_job(_swing(), settings, [0, 1], _axial_feed(250.0), tmp_path / "w")
    assert ForceColorScale.from_dict(job.shading) == scale
    assert job.load_shading() == scale
    frames = job.load_axial_frames()
    assert [f.values_n["GolfSwing3D_Kinetic/Head"] for f in frames] == [250.0, 250.0]
    off = ExportSettings(views=("face_on",), multiview=False)
    (tmp_path / "x").mkdir()
    plain = stage_job(_swing(), off, [0], _axial_feed(250.0), tmp_path / "x")
    assert plain.shading is None and plain.load_axial_frames() is None


# --- end-to-end render (needs a GL context) ---------------------------------

MINI_MJCF = """<mujoco><worldbody>
<light pos="0 0 3" dir="0 0 -1"/>
<body name="seg"><geom type="capsule" size=".1 .3" rgba=".7 .7 .7 1"/></body>
</worldbody></mujoco>"""


def _mini_render(scale: ForceColorScale | None, load: float | None) -> np.ndarray:
    """Worker order on a one-body model: update_scene, shading, render."""
    mujoco = pytest.importorskip("mujoco")
    from src.shared.python.body_part_viz.mujoco_force_colors import (
        apply_mujoco_scene_colors,
    )

    model = mujoco.MjModel.from_xml_string(MINI_MJCF)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    cam = mujoco.MjvCamera()
    cam.distance, cam.lookat[:] = 2.0, 0.0
    try:
        renderer = mujoco.Renderer(model, 64, 64)
    except (RuntimeError, ImportError, AttributeError, mujoco.FatalError) as exc:
        pytest.skip(f"no usable GL context: {exc}")
    try:
        renderer.update_scene(data, camera=cam)
        if scale is not None:
            frame = None if load is None else AxialLoadFrame(0.0, {"seg": load}, "t")
            apply_mujoco_scene_colors(model, renderer.scene, frame, scale)
        return renderer.render().copy()
    finally:
        renderer.close()


@pytest.mark.requires_gl
def test_loaded_segment_pixels_differ_and_zero_opacity_is_identical() -> None:
    off = _mini_render(None, None)
    for mode in ("tint", "replace"):
        full = ForceColorScale(enabled=True, blend_mode=mode, opacity=0.8)
        tension, compression = (_mini_render(full, f) for f in (1000.0, -1000.0))
        assert np.any(tension != off) and np.any(compression != off)
        assert np.any(tension != compression)
        zero = ForceColorScale(enabled=True, blend_mode=mode, opacity=0.0)
        assert np.array_equal(_mini_render(zero, 1000.0), off)
    assert np.array_equal(_mini_render(full, None), off)
    half = ForceColorScale(enabled=True, blend_mode="tint", opacity=0.4)
    weak = np.abs(_mini_render(half, 1000.0).astype(int) - off.astype(int)).sum()
    strong = np.abs(tension.astype(int) - off.astype(int)).sum()
    assert 0 < weak < strong


@pytest.mark.requires_gl
@pytest.mark.slow
@pytest.mark.timeout(300)
def test_worker_subprocess_applies_the_staged_shading() -> None:
    """Full path: staged payload, worker subprocess, appearance model (slow)."""
    pytest.importorskip("mujoco")
    swing = _swing(SPEC.read_bytes(), steps=1)
    backend = make_backend("mujoco")

    def render(scale: ForceColorScale | None) -> np.ndarray:
        settings = ExportSettings(
            views=("face_on",), width=320, height=240, multiview=False, shading=scale
        )
        try:
            out = next(iter(backend.render(swing, settings, [0], _axial_feed(1000.0))))
        except RuntimeError as exc:
            pytest.skip(f"no usable GL context: {exc}")
        return out["face_on"]

    off = render(None)
    shaded = render(ForceColorScale(enabled=True, blend_mode="replace", opacity=0.8))
    assert np.any(shaded != off)
